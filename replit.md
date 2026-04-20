# PDF Extraction POC

## Overview
A browser-based tool for extracting structured content (text chunks, tables, and images) from digital PDFs. Run four different extractors side-by-side, visualize bounding boxes directly on the PDF, and compare how each version performs across a whole document set.

## Tech Stack
- **Language:** Python 3.12
- **Backend:** FastAPI + Uvicorn (port 5000)
- **PDF libs:** PyMuPDF (fitz), pdfplumber, unstructured.io, docling
- **Frontend:** Vanilla HTML/CSS/JS + PDF.js (canvas rendering, SVG overlay for bboxes)

## Project Structure
```
pdf-extraction/
├── src/
│   ├── extract.py        # v1 — PyMuPDF + pdfplumber baseline
│   ├── extract_v2.py     # v2 — improved tables (full-grid / h-rules / whitespace)
│   ├── extract_v3.py     # v3 — unstructured.io partitioner (hi_res w/ fast fallback)
│   ├── extract_v4.py     # v4 — docling DocumentConverter
│   └── server.py         # FastAPI server
├── templates/
│   └── index.html        # Single-page frontend (PDF.js viewer + panels + results matrix)
├── input/
│   ├── docs/             # Source PDFs (upload target)
│   └── glossary/         # Reference data
├── output/
│   └── <pdf-stem>/
│       └── <version>/    # Per-version results
│           ├── chunks.json
│           ├── tables.json
│           ├── images.json
│           └── images/*.png
└── requirements.txt
```

## Running the App
```
uvicorn src.server:app --host 0.0.0.0 --port 5000 --reload
```
The workflow `Start application` runs this on port 5000.

## Output Format
All four extractors return the same shape so the frontend renders them identically:
```jsonc
{
  "chunks":  [{ "page": 1, "paragraph": 1, "text": "...", "font_size": 11.0,
                "bbox": [x0,y0,x1,y1], "page_size": [w,h], "category": "..." }],
  "tables":  [{ "page": 1, "table": [[...]], "bbox": [...], "page_size": [w,h],
                "table_type": "full_grid|h_rules|whitespace|unstructured|docling" }],
  "images":  [{ "page": 1, "filename": "...png", "url": "/images/...",
                "bbox": [...], "page_size": [w,h] }]
}
```
`bbox` is in PDF points with **top-left origin** (PyMuPDF convention). `page_size` is the page's width/height in PDF points, used by the frontend to scale bboxes onto the rendered PDF.js canvas.

## API Endpoints
- `GET  /`               — Serves the UI
- `GET  /api/pdfs`       — Lists PDFs in `input/docs/`
- `POST /api/upload`     — Multipart upload. Sanitizes the filename, auto-suffixes duplicates (`name_1.pdf`), and rejects non-PDFs via `%PDF` magic-byte check
- `POST /api/extract`    — Runs extraction. Body: `{ "pdf_name": "...", "version": "v1|v2|v3|v4" }`. v3/v4 return HTTP 503 if their heavy deps aren't installed.
- `GET  /api/results`    — Matrix of counts (chunks/tables/images) per PDF × version, read from persisted JSON
- `GET  /pdfs/<name>`    — Static: source PDFs
- `GET  /images/<stem>/<version>/images/<file>` — Static: extracted image files

## Extraction Versions
- **v1 — baseline:** PyMuPDF blocks for text; pdfplumber default table finder (full-grid only).
- **v2 — improved:** Classifies tables as `full_grid`, `h_rules`, or `whitespace`; dual-pass detection with IoU-based dedup; higher-quality chunk grouping.
- **v3 — unstructured.io:** `partition_pdf` with `hi_res` strategy (falls back to `fast`). Provides element categories via `chunk.category`. Bboxes come from `metadata.coordinates` (rescaled from detection-resolution px to PDF points).
- **v4 — docling:** `DocumentConverter` with document-level text/tables/pictures. Bboxes from docling `ProvenanceItem`s, converted from bottom-left (when applicable) to top-left origin.

> **First-run note for v3/v4:** both libraries may download ML models on first invocation (layout/OCR for unstructured; IBM docling models for docling). The first extraction per PDF can take 30s–several minutes; subsequent runs are much faster.

## Frontend Features
- **Two top-level screens**
  - **Workspace** — the active extraction tools
  - **Results** — a matrix of chunk/table/image counts for every PDF × version combination that's been run
- **PDF viewer** (PDF.js): page navigation, zoom, canvas rendering
- **Bounding-box overlay** (SVG on top of the canvas)
  - Click a chunk, table, or image → page jumps and the matching box is outlined and filled
  - "Show all boxes" toggle draws every box on the current page, color-coded (chunks green, tables yellow, images blue)
- **Upload** — drop a PDF into `input/docs/` straight from the sidebar
- **Chunks panel** — paginated (50/page), click-to-highlight
- **Tables panel** — each table card shows its `table_type` badge; click the card to highlight its bbox
- **Images panel** — grid of cropped images; click opens a lightbox and highlights the bbox

## Deployment
Configured as an autoscale deployment using `gunicorn -k uvicorn.workers.UvicornWorker src.server:app`. Port 5000 is the sole application port.

## System Dependencies (Nix)
- `xorg.libxcb` — required by unstructured/onnx imports
- `tesseract` — required by v3 (unstructured.io) hi_res OCR
- `poppler` — provides `pdftoppm` used during PDF rasterization

## Recent Changes
- 2026-04-20: Installed v3/v4 ML stacks (`unstructured[pdf]`, `docling`) plus system deps (`xorg.libxcb`, `tesseract`, `poppler`); both extractors now run end-to-end and produce chunks/tables.
- 2026-04-20: Added v3 (unstructured.io) and v4 (docling) extractors.
- 2026-04-20: Added `/api/upload` and in-browser PDF upload UI.
- 2026-04-20: Added bounding-box overlays via PDF.js + SVG; added "Show all boxes" toggle.
- 2026-04-20: Added results-matrix dashboard at `/api/results` and in the UI.
- 2026-04-20: Switched output layout to `output/<stem>/<version>/` so all four versions persist side-by-side.
