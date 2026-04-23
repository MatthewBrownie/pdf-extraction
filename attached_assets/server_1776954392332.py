"""
FastAPI server for the Vision PDF Extractor POC.

Endpoints:
  GET  /                       — UI
  GET  /api/pdfs               — list uploaded PDFs
  POST /api/upload             — upload a PDF
  POST /api/extract/stream     — SSE extraction stream
  GET  /api/extraction         — retrieve cached result
  GET  /pdfs/<filename>        — serve source PDFs
  GET  /output/<path>          — serve output files

Run:
  cd C:\\Users\\mbrow\\Documents\\pocs\\vision
  uvicorn src.server:app --reload --port 3001
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).parent
_ROOT = _HERE.parent

load_dotenv(_ROOT / ".env")

from .extractor import MODEL_BULK, MODEL_PRECISE, ExtractionResult, extract_pdf
from .transform import to_standard

_INPUT_DOCS = _ROOT / "input" / "docs"
_OUTPUT = _ROOT / "output"
_TEMPLATES = _ROOT / "templates"

_INPUT_DOCS.mkdir(parents=True, exist_ok=True)
_OUTPUT.mkdir(parents=True, exist_ok=True)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")

MODELS = {
    "haiku": MODEL_BULK,
    "sonnet": MODEL_PRECISE,
}

app = FastAPI(title="Vision PDF Extractor")

app.mount("/pdfs", StaticFiles(directory=str(_INPUT_DOCS)), name="pdfs")
app.mount("/output", StaticFiles(directory=str(_OUTPUT)), name="output")


# ── helpers ─────────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = _SAFE_NAME.sub("_", base).strip()
    if not base:
        raise HTTPException(400, "Empty filename after sanitisation")
    if not base.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")
    return base


def _unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        candidate = target.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _output_dir(stem: str) -> Path:
    d = _OUTPUT / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_cache(stem: str) -> dict | None:
    d = _OUTPUT / stem
    native_f = d / "extraction.json"
    if not native_f.exists():
        return None
    try:
        with native_f.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _write_outputs(stem: str, result: ExtractionResult, standard: dict) -> dict:
    d = _output_dir(stem)
    native = result.to_dict()

    with (d / "extraction.json").open("w", encoding="utf-8") as f:
        json.dump(native, f, indent=2, ensure_ascii=False)
    with (d / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(standard["chunks"], f, indent=2, ensure_ascii=False)
    with (d / "tables.json").open("w", encoding="utf-8") as f:
        json.dump(standard["tables"], f, indent=2, ensure_ascii=False)
    with (d / "images.json").open("w", encoding="utf-8") as f:
        json.dump(standard["images"], f, indent=2, ensure_ascii=False)

    return {**native, "standard": standard}


def _build_payload(pdf_name: str, data: dict) -> dict:
    return {
        "pdf": pdf_name,
        "pages": data.get("pages", 0),
        "model": data.get("model", ""),
        "usage": data.get("usage", {}),
        "estimated_cost_usd": data.get("estimated_cost_usd", 0),
        "text_by_page": data.get("text_by_page", []),
        "tables": data.get("tables", []),
        "figures": data.get("figures", []),
        "standard": data.get("standard", {"chunks": [], "tables": [], "images": []}),
        "pdf_url": f"/pdfs/{pdf_name}",
    }


# ── routes ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    p = _TEMPLATES / "index.html"
    if not p.exists():
        raise HTTPException(404, "index.html not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))


@app.get("/api/pdfs")
def list_pdfs():
    if not _INPUT_DOCS.exists():
        return {"pdfs": []}
    pdfs = sorted(p.name for p in _INPUT_DOCS.iterdir() if p.suffix.lower() == ".pdf")
    return {"pdfs": pdfs}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    safe_name = _safe_filename(file.filename or "uploaded.pdf")
    target = _unique_path(_INPUT_DOCS / safe_name)
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    with target.open("rb") as fh:
        header = fh.read(5)
    if header[:4] != b"%PDF":
        target.unlink(missing_ok=True)
        raise HTTPException(400, "Not a valid PDF")
    return {"pdf_name": target.name}


class ExtractRequest(BaseModel):
    pdf_name: str
    model: str = "haiku"
    force: bool = False


@app.post("/api/extract/stream")
async def extract_stream(req: ExtractRequest, request: Request):
    pdf_name = Path(req.pdf_name).name
    pdf_path = (_INPUT_DOCS / pdf_name).resolve()
    try:
        pdf_path.relative_to(_INPUT_DOCS.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid PDF name")
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found: {pdf_name}")

    model_id = MODELS.get(req.model, MODEL_BULK)
    stem = pdf_path.stem

    if not req.force:
        cached = _load_cache(stem)
        if cached is not None:
            # Check the cached model matches
            if cached.get("model") == model_id:
                payload = _build_payload(pdf_name, cached)

                def _cached_gen():
                    yield json.dumps({
                        "event": "progress", "phase": "cache",
                        "message": "Loaded cached result.", "elapsed": 0.0,
                    }) + "\n"
                    yield json.dumps({
                        "event": "done", "elapsed": 0.0,
                        "cached": True, "result": payload,
                    }) + "\n"

                return StreamingResponse(_cached_gen(), media_type="application/x-ndjson")

    q: "queue.Queue[tuple[str, object]]" = queue.Queue()

    def _progress(**kw):
        q.put(("progress", kw))

    def _worker():
        try:
            result = extract_pdf(str(pdf_path), model=model_id, progress_cb=_progress)
            standard = to_standard(result)
            combined = _write_outputs(stem, result, standard)
            q.put(("done", combined))
        except Exception as e:
            q.put(("error", f"{type(e).__name__}: {e}"))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    started = time.time()

    async def gen():
        yield json.dumps({
            "event": "progress", "phase": "starting",
            "message": "Starting extraction…", "elapsed": 0.0,
        }) + "\n"

        try:
            last_hb = time.time()
            while True:
                if await request.is_disconnected():
                    return

                try:
                    kind, data = await asyncio.to_thread(q.get, True, 0.5)
                except queue.Empty:
                    elapsed = time.time() - started
                    if not t.is_alive():
                        yield json.dumps({
                            "event": "error",
                            "error": "Worker exited without producing a result.",
                        }) + "\n"
                        return
                    if elapsed > 1800:
                        yield json.dumps({
                            "event": "error",
                            "error": f"Timed out after {int(elapsed)}s.",
                        }) + "\n"
                        return
                    if time.time() - last_hb >= 2.0:
                        last_hb = time.time()
                        yield json.dumps({
                            "event": "heartbeat",
                            "elapsed": round(elapsed, 1),
                        }) + "\n"
                    continue

                elapsed = round(time.time() - started, 1)
                if kind == "progress":
                    yield json.dumps({"event": "progress", "elapsed": elapsed, **data}, default=str) + "\n"
                elif kind == "done":
                    yield json.dumps({
                        "event": "done", "elapsed": elapsed,
                        "result": _build_payload(pdf_name, data),
                    }) + "\n"
                    return
                elif kind == "error":
                    yield json.dumps({"event": "error", "elapsed": elapsed, "error": str(data)}) + "\n"
                    return
        finally:
            pass

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/extraction")
def get_extraction(pdf_name: str):
    pdf_name = Path(pdf_name).name
    stem = Path(pdf_name).stem
    cached = _load_cache(stem)
    if cached is None:
        raise HTTPException(404, f"No cached extraction for {pdf_name}")
    return _build_payload(pdf_name, cached)
