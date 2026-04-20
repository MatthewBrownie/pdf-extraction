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

import importlib
import json
import re
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import extract as extractor_v1
import extract_v2 as extractor_v2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = _HERE.parent
_INPUT_DOCS = _ROOT / "input" / "docs"
_OUTPUT = _ROOT / "output"
_TEMPLATES = _ROOT / "templates"

_INPUT_DOCS.mkdir(parents=True, exist_ok=True)
_OUTPUT.mkdir(parents=True, exist_ok=True)

VERSIONS = ("v1", "v2", "v3", "v4")

app = FastAPI(title="PDF Extraction POC")

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
    raise HTTPException(status_code=400, detail=f"Unknown version: {version}")


def _attach_image_urls(images: list[dict], stem: str, version: str) -> None:
    for img in images:
        img["url"] = f"/images/{stem}/{version}/images/{img['filename']}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = _TEMPLATES / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


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


@app.post("/api/extract")
def run_extraction(req: ExtractRequest):
    # Strip any path components to block traversal
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

    extractor = _get_extractor(req.version)
    try:
        result = extractor.extract_pdf(str(pdf_path), str(output_dir))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    _attach_image_urls(result.get("images", []), stem, req.version)

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
