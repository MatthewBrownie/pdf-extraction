"""
FastAPI server for the PDF extraction POC.

Serves a browser UI at / and exposes endpoints to list PDFs, run extraction,
and serve the resulting images.

Run:
    uvicorn server:app --reload --port 3000

    Or from the repo root:
    uvicorn src.server:app --reload --port 3000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow importing sibling modules whether run from src/ or from the repo root.
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

app = FastAPI(title="PDF Extraction POC")

# Serve extracted images as static files under /images/<stem>/<filename>
app.mount("/images", StaticFiles(directory=str(_OUTPUT)), name="images")

# Serve source PDFs so the browser can render them in an iframe
app.mount("/pdfs", StaticFiles(directory=str(_INPUT_DOCS)), name="pdfs")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    pdf_name: str
    version: str = "v2"  # "v1" or "v2"


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
    """Return the list of PDF filenames available in input/docs/."""
    if not _INPUT_DOCS.exists():
        return {"pdfs": []}
    pdfs = sorted(p.name for p in _INPUT_DOCS.iterdir() if p.suffix.lower() == ".pdf")
    return {"pdfs": pdfs}


@app.post("/api/extract")
def run_extraction(req: ExtractRequest):
    """
    Run extraction on the requested PDF.

    Results are saved under output/<pdf_stem>/ and returned as JSON so the
    browser can render them immediately without a second request.
    """
    pdf_path = _INPUT_DOCS / req.pdf_name
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_name}")

    if req.version not in ("v1", "v2"):
        raise HTTPException(status_code=400, detail="version must be 'v1' or 'v2'")

    stem = pdf_path.stem
    output_dir = _OUTPUT / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = extractor_v1 if req.version == "v1" else extractor_v2
    result = extractor.extract_pdf(str(pdf_path), str(output_dir))

    # Make image paths URL-friendly — strip the absolute output dir prefix and
    # return a path the browser can fetch via /images/<stem>/images/<filename>.
    for img in result.get("images", []):
        img["url"] = f"/images/{stem}/images/{img['filename']}"

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
    }
