"""
FastAPI server for the PDF extraction POC.

Endpoints:
  GET  /                         — UI
  GET  /api/pdfs                 — list available PDFs
  POST /api/upload               — upload a new PDF (multipart/form-data)
  POST /api/extract              — run extraction (v1/v2/v3/v4)
  GET  /api/results              — counts matrix per PDF × version
  GET  /pdfs/<filename>          — static: source PDFs
  GET  /images/<stem>/<v>/...    — static: per-version output (images, etc.)

Run:
  uvicorn src.server:app --host 0.0.0.0 --port 5000 --reload
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import queue
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import extract as extractor_v1
import extract_v2 as extractor_v2
import warmup as _warmup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = _HERE.parent
_INPUT_DOCS = _ROOT / "input" / "docs"
_OUTPUT = _ROOT / "output"
_TEMPLATES = _ROOT / "templates"

_INPUT_DOCS.mkdir(parents=True, exist_ok=True)
_OUTPUT.mkdir(parents=True, exist_ok=True)

VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6")

app = FastAPI(title="PDF Extraction POC")


@app.on_event("startup")
def _prewarm_heavy_models() -> None:
    """Kick off v3/v4 model downloads in background threads at server start.

    The first call to v3 (unstructured.io) or v4 (docling) otherwise downloads
    several hundred MB of layout/OCR/IBM models on demand, hanging the request
    for 30s–several minutes. Doing the downloads in the background here means
    the first user-facing extraction is roughly as fast as subsequent ones.

    Warmups are best-effort and never block server startup or raise.
    """
    _warmup.warmup_all_in_background()

# Serve per-version output (images live under /images/<stem>/<version>/images/)
app.mount("/images", StaticFiles(directory=str(_OUTPUT)), name="images")
# Serve source PDFs
app.mount("/pdfs", StaticFiles(directory=str(_INPUT_DOCS)), name="pdfs")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    pdf_name: str
    version: str = "v2"
    force: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def _safe_filename(name: str) -> str:
    """Strip path components and reject anything that isn't a tame filename."""
    base = Path(name).name  # drops directories
    base = _SAFE_NAME.sub("_", base).strip()
    if not base:
        raise HTTPException(status_code=400, detail="Empty filename after sanitization")
    if not base.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Filename must end in .pdf")
    return base


def _unique_path(target: Path) -> Path:
    """Auto-suffix `_1`, `_2`… if a file with this name already exists."""
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        candidate = target.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _get_extractor(version: str):
    if version == "v1":
        return extractor_v1
    if version == "v2":
        return extractor_v2
    if version == "v3":
        # Lazy import — heavy ML deps
        try:
            return importlib.import_module("extract_v3")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"v3 (unstructured.io) is not available in this environment: {e}",
            )
    if version == "v4":
        try:
            return importlib.import_module("extract_v4")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"v4 (docling) is not available in this environment: {e}",
            )
    if version == "v5":
        try:
            return importlib.import_module("extract_v5")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"v5 (tagged structure tree) is not available: {e}",
            )
    if version == "v6":
        try:
            return importlib.import_module("extract_v6")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"v6 (Camelot lattice) is not available in this environment: {e}",
            )
    raise HTTPException(status_code=400, detail=f"Unknown version: {version}")


def _attach_image_urls(images: list[dict], stem: str, version: str) -> None:
    for img in images:
        img["url"] = f"/images/{stem}/{version}/images/{img['filename']}"


def _load_cached_result(output_dir: Path) -> dict | None:
    """Return a result dict loaded from persisted JSON, or None if incomplete.

    A cache is considered valid only when all three files (chunks/tables/images)
    exist and parse as JSON lists.
    """
    result: dict[str, list] = {}
    mtimes: list[float] = []
    for key in ("chunks", "tables", "images"):
        jf = output_dir / f"{key}.json"
        if not jf.exists():
            return None
        try:
            with jf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        result[key] = data
        try:
            mtimes.append(jf.stat().st_mtime)
        except OSError:
            pass
    if mtimes:
        result["__cached_at__"] = max(mtimes)
    return result


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = _TEMPLATES / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/warmup")
def warmup_status():
    """Return per-version model warmup status.

    Used by the UI to show a "Models still downloading…" banner on the v3/v4
    tabs while the background downloads kicked off at server startup are still
    in flight. Each entry has a `status` of pending/in_progress/ready/failed.
    """
    status = _warmup.get_status()
    out: dict[str, dict] = {}
    for v, info in status.items():
        entry = {"status": info.get("status", "pending"), "detail": info.get("detail")}
        started = info.get("started_at")
        finished = info.get("finished_at")
        now = time.time()
        if isinstance(started, (int, float)):
            end = finished if isinstance(finished, (int, float)) else now
            entry["elapsed"] = round(max(0.0, end - started), 1)
        out[v] = entry
    return {"versions": out}


@app.post("/api/warmup/retry")
def warmup_retry():
    """Re-run warmup in background for any version currently in "failed" state.

    Versions that are pending / in_progress / ready are left alone so a retry
    click can't clobber an in-flight or already-successful warmup. Returns the
    list of versions that were actually restarted.
    """
    import threading

    fns = {"v3": _warmup.warmup_v3, "v4": _warmup.warmup_v4}
    status = _warmup.get_status()
    restarted: list[str] = []
    now = time.time()
    for v, fn in fns.items():
        if status.get(v, {}).get("status") == "failed":
            # Flip to in_progress immediately so a fast poll doesn't still see
            # "failed" before the worker thread re-enters warmup_vX().
            _warmup._set_status(v, status="in_progress", started_at=now, finished_at=None, detail=None)
            threading.Thread(target=fn, name=f"warmup-{v}-retry", daemon=True).start()
            restarted.append(v)
    return {"restarted": restarted}


@app.get("/api/pdfs")
def list_pdfs():
    if not _INPUT_DOCS.exists():
        return {"pdfs": []}
    pdfs = sorted(p.name for p in _INPUT_DOCS.iterdir() if p.suffix.lower() == ".pdf")
    return {"pdfs": pdfs}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF upload, save it into input/docs/ with a safe filename."""
    if file.content_type and "pdf" not in file.content_type.lower() and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    safe_name = _safe_filename(file.filename or "uploaded.pdf")
    target = _unique_path(_INPUT_DOCS / safe_name)

    # Read in chunks so we don't blow up memory on large PDFs.
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    # Light sanity check — should start with %PDF
    with target.open("rb") as fh:
        header = fh.read(5)
    if header[:4] != b"%PDF":
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

    return {"pdf_name": target.name}


def _validate_extract_request(req: ExtractRequest) -> tuple[Path, str, Path]:
    """Shared validation for /api/extract and /api/extract/stream."""
    pdf_name = Path(req.pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_name}")
    if req.version not in VERSIONS:
        raise HTTPException(status_code=400, detail=f"version must be one of {VERSIONS}")
    req.pdf_name = pdf_name
    stem = pdf_path.stem
    output_dir = _OUTPUT / stem / req.version
    output_dir.mkdir(parents=True, exist_ok=True)
    return pdf_path, stem, output_dir


def _build_result_payload(req: ExtractRequest, result: dict) -> dict:
    return {
        "pdf": req.pdf_name,
        "version": req.version,
        "stats": {
            "chunks": len(result["chunks"]),
            "tables": len(result["tables"]),
            "images": len(result["images"]),
        },
        "chunks": result["chunks"],
        "tables": result["tables"],
        "images": result["images"],
        "pdf_url": f"/pdfs/{req.pdf_name}",
    }


# Per-version timeout (seconds) for streaming extraction. v3/v4 can be slow on
# the first run because models are downloaded.
_EXTRACT_TIMEOUTS = {"v1": 120, "v2": 180, "v3": 600, "v4": 1200, "v5": 300, "v6": 180}


@app.post("/api/extract")
def run_extraction(req: ExtractRequest):
    pdf_path, stem, output_dir = _validate_extract_request(req)

    if not req.force:
        cached = _load_cached_result(output_dir)
        if cached is not None:
            _attach_image_urls(cached["images"], stem, req.version)
            payload = _build_result_payload(req, cached)
            payload["cached"] = True
            cached_at = cached.get("__cached_at__")
            if cached_at is not None:
                payload["cached_at"] = _iso_utc(cached_at)
            return payload

    extractor = _get_extractor(req.version)
    try:
        result = extractor.extract_pdf(str(pdf_path), str(output_dir))
    except HTTPException:
        raise
    except ModuleNotFoundError as e:
        # v3/v4 lazy-import their heavy ML deps inside extract_pdf, so a
        # missing module surfaces here rather than in _get_extractor.
        raise HTTPException(
            status_code=503,
            detail=f"{req.version} requires an optional dependency that isn't installed: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    _attach_image_urls(result.get("images", []), stem, req.version)
    return _build_result_payload(req, result)


class _CancelledExtraction(Exception):
    """Raised inside an extractor when the client cancels the request."""


@app.post("/api/extract/stream")
async def run_extraction_stream(req: ExtractRequest, request: Request):
    """
    Stream extraction progress as newline-delimited JSON.

    Each line is a JSON object with an "event" field:
      - {"event": "progress", "phase": str, "message": str, "page": int?, "total": int?, "elapsed": float}
      - {"event": "heartbeat", "elapsed": float}                 — emitted every ~2s when the worker is busy
      - {"event": "done", "result": {...full extraction payload...}}
      - {"event": "error", "error": str}
    """
    pdf_path, stem, output_dir = _validate_extract_request(req)

    if not req.force:
        cached = _load_cached_result(output_dir)
        if cached is not None:
            _attach_image_urls(cached["images"], stem, req.version)
            payload = _build_result_payload(req, cached)
            payload["cached"] = True
            cached_at_ts = cached.get("__cached_at__")
            cached_at_iso = _iso_utc(cached_at_ts) if cached_at_ts is not None else None
            if cached_at_iso:
                payload["cached_at"] = cached_at_iso

            def cached_gen():
                yield json.dumps({
                    "event": "progress",
                    "phase": "cache",
                    "message": f"Loaded cached {req.version} results.",
                    "elapsed": 0.0,
                }) + "\n"
                done_evt = {
                    "event": "done",
                    "elapsed": 0.0,
                    "cached": True,
                    "result": payload,
                }
                if cached_at_iso:
                    done_evt["cached_at"] = cached_at_iso
                yield json.dumps(done_evt) + "\n"

            return StreamingResponse(cached_gen(), media_type="application/x-ndjson")

    extractor = _get_extractor(req.version)
    timeout = _EXTRACT_TIMEOUTS.get(req.version, 600)

    q: "queue.Queue[tuple[str, object]]" = queue.Queue()
    cancel_event = threading.Event()

    def cancel_check() -> bool:
        return cancel_event.is_set()

    def progress_cb(**evt):
        if cancel_event.is_set():
            raise _CancelledExtraction()
        q.put(("progress", evt))

    # Only pass cancel_check to extractors whose extract_pdf accepts it
    # (v3/v4). v1/v2 don't support it, so we omit the kwarg there.
    extractor_kwargs: dict = {"progress_cb": progress_cb}
    try:
        if "cancel_check" in inspect.signature(extractor.extract_pdf).parameters:
            extractor_kwargs["cancel_check"] = cancel_check
    except (TypeError, ValueError):
        pass

    def worker():
        try:
            result = extractor.extract_pdf(str(pdf_path), str(output_dir), **extractor_kwargs)
            if cancel_event.is_set():
                q.put(("cancelled", None))
                return
            _attach_image_urls(result.get("images", []), stem, req.version)
            q.put(("done", result))
        except _CancelledExtraction:
            q.put(("cancelled", None))
        except Exception as e:
            if cancel_event.is_set():
                q.put(("cancelled", None))
            else:
                q.put(("error", f"{type(e).__name__}: {e}"))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started = time.time()

    async def gen():
        # Initial event so the client sees something immediately.
        yield json.dumps({
            "event": "progress",
            "phase": "starting",
            "message": f"Starting {req.version} extraction…",
            "elapsed": 0.0,
        }) + "\n"

        try:
            last_heartbeat = time.time()
            while True:
                # If the client went away, signal cancellation and stop streaming.
                if await request.is_disconnected():
                    cancel_event.set()
                    return

                try:
                    kind, data = await asyncio.to_thread(q.get, True, 0.5)
                except queue.Empty:
                    elapsed = time.time() - started
                    if not t.is_alive():
                        yield json.dumps({
                            "event": "error",
                            "error": "Extractor exited unexpectedly without producing a result.",
                        }) + "\n"
                        return
                    if elapsed > timeout:
                        cancel_event.set()
                        yield json.dumps({
                            "event": "error",
                            "error": f"Extraction timed out after {int(elapsed)}s "
                                     f"(limit {timeout}s for {req.version}). "
                                     f"The background job may still be running on the server.",
                        }) + "\n"
                        return
                    if time.time() - last_heartbeat >= 2.0:
                        last_heartbeat = time.time()
                        yield json.dumps({"event": "heartbeat", "elapsed": round(elapsed, 1)}) + "\n"
                    continue

                elapsed = round(time.time() - started, 1)
                if kind == "progress":
                    payload = {"event": "progress", "elapsed": elapsed}
                    if isinstance(data, dict):
                        payload.update(data)
                    yield json.dumps(payload, default=str) + "\n"
                elif kind == "done":
                    yield json.dumps({
                        "event": "done",
                        "elapsed": elapsed,
                        "result": _build_result_payload(req, data),  # type: ignore[arg-type]
                    }) + "\n"
                    return
                elif kind == "cancelled":
                    yield json.dumps({
                        "event": "cancelled",
                        "elapsed": elapsed,
                        "message": "Extraction cancelled.",
                    }) + "\n"
                    return
                elif kind == "error":
                    yield json.dumps({"event": "error", "elapsed": elapsed, "error": str(data)}) + "\n"
                    return
        finally:
            # Client disconnected mid-stream, or we're otherwise tearing down —
            # tell the worker to bail at its next checkpoint so the next
            # extraction can start immediately.
            cancel_event.set()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/extraction")
def get_extraction(pdf_name: str, version: str):
    """
    Return persisted extraction data for a single PDF × version.

    Reads chunks.json/tables.json/images.json from output/<stem>/<version>/.
    Returns 404 if the version hasn't been run for that PDF.
    """
    pdf_name = Path(pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_name}")
    if version not in VERSIONS:
        raise HTTPException(status_code=400, detail=f"version must be one of {VERSIONS}")

    stem = pdf_path.stem
    vdir = _OUTPUT / stem / version
    if not vdir.exists():
        raise HTTPException(status_code=404, detail=f"No persisted output for {pdf_name} / {version}")

    out: dict[str, list] = {"chunks": [], "tables": [], "images": []}
    found_any = False
    for key in ("chunks", "tables", "images"):
        jf = vdir / f"{key}.json"
        if jf.exists():
            try:
                with jf.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    out[key] = data
                    found_any = True
            except Exception:
                pass

    if not found_any:
        raise HTTPException(status_code=404, detail=f"No persisted output for {pdf_name} / {version}")

    _attach_image_urls(out["images"], stem, version)

    return {
        "pdf": pdf_name,
        "version": version,
        "stats": {
            "chunks": len(out["chunks"]),
            "tables": len(out["tables"]),
            "images": len(out["images"]),
        },
        "chunks": out["chunks"],
        "tables": out["tables"],
        "images": out["images"],
        "pdf_url": f"/pdfs/{pdf_name}",
    }


@app.get("/api/results")
def results_matrix():
    """
    Return a matrix of counts (chunks/tables/images) for each PDF × version.

    Reads persisted JSON in output/<stem>/<version>/. No re-extraction.
    """
    pdfs = sorted(p.name for p in _INPUT_DOCS.iterdir() if p.suffix.lower() == ".pdf") if _INPUT_DOCS.exists() else []

    matrix: dict[str, dict[str, dict[str, int] | None]] = {}
    for pdf_name in pdfs:
        stem = Path(pdf_name).stem
        per_version: dict[str, dict[str, int] | None] = {}
        for v in VERSIONS:
            vdir = _OUTPUT / stem / v
            if not vdir.exists():
                per_version[v] = None
                continue
            counts = {"chunks": 0, "tables": 0, "images": 0}
            ok = False
            for key in ("chunks", "tables", "images"):
                jf = vdir / f"{key}.json"
                if jf.exists():
                    try:
                        with jf.open("r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        counts[key] = len(data) if isinstance(data, list) else 0
                        ok = True
                    except Exception:
                        counts[key] = 0
            per_version[v] = counts if ok else None
        matrix[pdf_name] = per_version

    return {"versions": list(VERSIONS), "pdfs": pdfs, "matrix": matrix}
