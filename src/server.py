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
import extract_vision
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


# ---------------------------------------------------------------------------
# Claude Vision extractor (separate page at /vision)
# ---------------------------------------------------------------------------
class VisionExtractRequest(BaseModel):
    pdf_name: str
    model: str = "haiku"
    force: bool = False
    resume: bool = False


def _vision_dir(stem: str) -> Path:
    d = _OUTPUT / stem / "vision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collect_vision_history() -> dict[str, list[dict]]:
    """Scan cached vision runs and group their token usage by model id.

    Returns a dict mapping model id to a list of
    `{"pages": int, "input_tokens": int, "output_tokens": int}` summaries
    drawn from every `output/<stem>/vision/extraction.json` on disk.
    Used to calibrate the pre-run cost estimate against real history.
    """
    history: dict[str, list[dict]] = {}
    if not _OUTPUT.exists():
        return history
    for nf in _OUTPUT.glob("*/vision/extraction.json"):
        try:
            with nf.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        model = data.get("model")
        usage = data.get("usage") or {}
        pages = data.get("pages") or 0
        if not model or not pages:
            continue
        history.setdefault(model, []).append({
            "pages": pages,
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
        })
    return history


def _load_vision_cache(stem: str) -> dict | None:
    d = _OUTPUT / stem / "vision"
    nf = d / "extraction.json"
    if not nf.exists():
        return None
    try:
        with nf.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        try:
            data["__cached_at__"] = nf.stat().st_mtime
        except OSError:
            pass
        return data
    except Exception:
        return None


def _write_vision_outputs(stem: str, result, standard: dict) -> dict:
    d = _vision_dir(stem)
    native = result.to_dict()
    with (d / "extraction.json").open("w", encoding="utf-8") as f:
        json.dump(native, f, indent=2, ensure_ascii=False)
    with (d / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(standard["chunks"], f, indent=2, ensure_ascii=False)
    with (d / "tables.json").open("w", encoding="utf-8") as f:
        json.dump(standard["tables"], f, indent=2, ensure_ascii=False)
    with (d / "images.json").open("w", encoding="utf-8") as f:
        json.dump(standard["images"], f, indent=2, ensure_ascii=False)
    return native


def _delete_vision_cache(stem: str) -> bool:
    """Remove cached vision extraction artefacts. Returns True if anything was removed."""
    d = _OUTPUT / stem / "vision"
    if not d.exists():
        return False
    removed = False
    for name in ("extraction.json", "chunks.json", "tables.json", "images.json"):
        p = d / name
        if p.exists():
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
    return removed


def _build_vision_payload(pdf_name: str, data: dict) -> dict:
    payload = {
        "pdf": pdf_name,
        "pages": data.get("pages", 0),
        "model": data.get("model", ""),
        "usage": data.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        "estimated_cost_usd": data.get("estimated_cost_usd", 0),
        "text_by_page": data.get("text_by_page", []),
        "tables": data.get("tables", []),
        "figures": data.get("figures", []),
        "pdf_url": f"/pdfs/{pdf_name}",
        "partial": bool(data.get("partial", False)),
        "chunks_done": int(data.get("chunks_done", 0) or 0),
        "chunks_total": int(data.get("chunks_total", 0) or 0),
        "pages_done": int(data.get("pages_done", 0) or 0),
    }
    cached_at = data.get("__cached_at__")
    if cached_at is not None:
        payload["cached_at"] = _iso_utc(cached_at)
    return payload


def _validate_vision_request(req: VisionExtractRequest) -> tuple[Path, str, str]:
    pdf_name = Path(req.pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_name}")
    if req.model not in extract_vision.MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {list(extract_vision.MODELS)}",
        )
    req.pdf_name = pdf_name
    return pdf_path, pdf_path.stem, extract_vision.MODELS[req.model]


@app.get("/vision", response_class=HTMLResponse)
def vision_page():
    html_path = _TEMPLATES / "vision.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="vision.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/vision/estimate")
def get_vision_estimate(pdf_name: str):
    """Return page count and approximate USD cost ranges for both models."""
    pdf_name = Path(pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_name}")
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(pdf_path)) as doc:
            pages = doc.page_count
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {exc}")
    return {
        "pdf": pdf_name,
        "pages": pages,
        "estimates": extract_vision.estimate_cost(
            pages, history=_collect_vision_history()
        ),
    }


@app.get("/api/vision/extraction")
def get_vision_extraction(pdf_name: str):
    pdf_name = Path(pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    cached = _load_vision_cache(pdf_path.stem)
    if cached is None:
        raise HTTPException(status_code=404, detail=f"No cached vision extraction for {pdf_name}")
    return _build_vision_payload(pdf_name, cached)


@app.delete("/api/vision/extraction")
def delete_vision_extraction(pdf_name: str):
    pdf_name = Path(pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF name")
    removed = _delete_vision_cache(pdf_path.stem)
    return {"deleted": removed, "pdf": pdf_name}


@app.post("/api/vision/extract/stream")
async def vision_extract_stream(req: VisionExtractRequest, request: Request):
    pdf_path, stem, model_id = _validate_vision_request(req)

    if not extract_vision.get_api_key():
        raise HTTPException(
            status_code=503,
            detail="OpenRouter API key not configured. Set OPENROUTER_API_KEY (or OPENAI_API_KEY).",
        )

    resume_from: dict | None = None
    if req.resume:
        cached = _load_vision_cache(stem)
        if (
            cached is None
            or cached.get("model") != model_id
            or not cached.get("partial")
            or int(cached.get("chunks_done") or 0) <= 0
        ):
            raise HTTPException(
                status_code=409,
                detail="No partial vision result available to resume for this PDF/model.",
            )
        resume_from = cached
    elif not req.force:
        cached = _load_vision_cache(stem)
        if (
            cached is not None
            and cached.get("model") == model_id
            and not cached.get("partial")
        ):
            payload = _build_vision_payload(req.pdf_name, cached)

            def cached_gen():
                yield json.dumps({
                    "event": "progress", "phase": "cache",
                    "message": "Loaded cached vision result.", "elapsed": 0.0,
                }) + "\n"
                done_evt = {
                    "event": "done", "elapsed": 0.0,
                    "cached": True, "result": payload,
                }
                if "cached_at" in payload:
                    done_evt["cached_at"] = payload["cached_at"]
                yield json.dumps(done_evt) + "\n"

            return StreamingResponse(cached_gen(), media_type="application/x-ndjson")

    q: "queue.Queue[tuple[str, object]]" = queue.Queue()
    cancel_event = threading.Event()

    def cancel_check() -> bool:
        return cancel_event.is_set()

    def progress_cb(**evt):
        if cancel_event.is_set():
            raise extract_vision.CancelledExtraction("Extraction cancelled.")
        q.put(("progress", evt))

    def _persist_partial(partial_result) -> dict | None:
        if partial_result is None or getattr(partial_result, "chunks_done", 0) <= 0:
            return None
        try:
            # If everything actually finished, persist as a complete cache so
            # subsequent loads don't force a re-run.
            chunks_done = getattr(partial_result, "chunks_done", 0)
            chunks_total = getattr(partial_result, "chunks_total", 0)
            partial_result.partial = bool(
                chunks_total <= 0 or chunks_done < chunks_total
            )
            standard = extract_vision.to_standard(partial_result)
            return _write_vision_outputs(stem, partial_result, standard)
        except Exception:
            return None

    def worker():
        try:
            result = extract_vision.extract_pdf(
                str(pdf_path), model=model_id,
                progress_cb=progress_cb, cancel_check=cancel_check,
                resume_from=resume_from,
            )
            if cancel_event.is_set():
                # Cancel arrived after the run finished — keep whatever we have.
                # _persist_partial decides whether it's truly partial or complete.
                q.put(("cancelled", _persist_partial(result)))
                return
            standard = extract_vision.to_standard(result)
            native = _write_vision_outputs(stem, result, standard)
            q.put(("done", native))
        except extract_vision.CancelledExtraction as exc:
            q.put(("cancelled", _persist_partial(getattr(exc, "result", None))))
        except Exception as e:
            if cancel_event.is_set():
                q.put(("cancelled", None))
            else:
                q.put(("error", f"{type(e).__name__}: {e}"))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started = time.time()
    timeout = 1800

    async def gen():
        yield json.dumps({
            "event": "progress", "phase": "starting",
            "message": "Starting Claude vision extraction…", "elapsed": 0.0,
        }) + "\n"

        try:
            last_hb = time.time()
            while True:
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
                            "error": "Vision worker exited without producing a result.",
                        }) + "\n"
                        return
                    if elapsed > timeout:
                        cancel_event.set()
                        yield json.dumps({
                            "event": "error",
                            "error": f"Vision extraction timed out after {int(elapsed)}s.",
                        }) + "\n"
                        return
                    if time.time() - last_hb >= 2.0:
                        last_hb = time.time()
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
                        "event": "done", "elapsed": elapsed,
                        "result": _build_vision_payload(req.pdf_name, data),  # type: ignore[arg-type]
                    }) + "\n"
                    return
                elif kind == "cancelled":
                    cancelled_evt = {
                        "event": "cancelled", "elapsed": elapsed,
                        "message": "Extraction cancelled.",
                    }
                    if isinstance(data, dict):
                        partial_payload = _build_vision_payload(req.pdf_name, data)
                        cancelled_evt["partial"] = True
                        cancelled_evt["result"] = partial_payload
                        cancelled_evt["chunks_done"] = partial_payload.get("chunks_done", 0)
                        cancelled_evt["chunks_total"] = partial_payload.get("chunks_total", 0)
                    yield json.dumps(cancelled_evt, default=str) + "\n"
                    return
                elif kind == "error":
                    yield json.dumps({"event": "error", "elapsed": elapsed, "error": str(data)}) + "\n"
                    return
        finally:
            # Tell the worker to bail at its next checkpoint if we're tearing down.
            cancel_event.set()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


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
