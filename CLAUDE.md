# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A browser-based POC that extracts structured content (text chunks, tables, embedded images) from digital PDFs across four extractor versions (v1–v4), with a side-by-side comparison UI and SVG bounding-box overlays on the PDF viewer.

No test suite, no linting config — intentionally a POC.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start server locally (http://localhost:3000)
cd src
python -m uvicorn server:app --reload --port 3000

# Production (as deployed on Replit, port 5000)
uvicorn src.server:app --host 0.0.0.0 --port 5000 --reload

# Pre-warm v3/v4 ML models manually
python -m src.warmup --only v3
python -m src.warmup --only v4
```

**v3/v4 require system packages** (`tesseract`, `poppler`, `libxcb`). On non-Replit environments install these via your package manager before running `pip install unstructured[pdf] docling`.

## Architecture

### Extractor versions

| Version | Core library | Table strategy |
|---|---|---|
| v1 | PyMuPDF + pdfplumber | pdfplumber defaults (full-grid only) |
| v2 | PyMuPDF + pdfplumber | Dual-pass detection + `_classify_table()` (full_grid / h_rules / whitespace) with IoU dedup |
| v3 | `unstructured.partition_pdf` | ML layout detection (YOLOX), hi_res strategy, Tesseract OCR |
| v4 | `docling.DocumentConverter` | IBM layout/table/OCR models |

v3 and v4 use **lazy imports** — their heavy dependencies are not imported at server startup. `src/warmup.py` downloads models in background daemon threads at startup so the UI is immediately usable. Poll `/api/warmup` for download status.

### Data flow

1. `POST /api/extract` (or `/api/extract/stream` for SSE progress) receives `{pdf_name, version, force?}`
2. Server checks cache: if `output/<pdf-stem>/<version>/{chunks,tables,images}.json` all exist → return cached (skip unless `force=true`)
3. Calls the relevant `extract_pdf(pdf_path, output_dir, progress_cb, cancel_check)` function
4. Extractor writes results to `output/<pdf-stem>/<version>/` and returns a dict
5. Server attaches image URLs (`/images/<stem>/<version>/images/<file>`) and returns full payload

All bboxes are `[x0, y0, x1, y1]` in PDF points, **top-left origin** (PyMuPDF convention). v3 rescales from pixel-space; v4 flips from bottom-left origin when `coord_origin.name == "BOTTOMLEFT"`.

### Output schema

```
output/<pdf-stem>/<version>/
├── chunks.json    [{page, paragraph, text, font_size, bbox, page_size, category?}]
├── tables.json    [{page, table_type, table: [[str]], bbox, page_size}]
├── images.json    [{page, filename, bbox, page_size}]
└── images/        PNG files of extracted images
```

`table_type` values: `full_grid`, `h_rules`, `whitespace` (v1/v2); `unstructured` (v3); `docling` (v4).

### Key modules

**`src/server.py`** — FastAPI app. Notable endpoints beyond extract:
- `POST /api/upload` — validates magic bytes (`%PDF`), auto-suffixes duplicates
- `GET /api/results` — scans all `output/` dirs, returns counts matrix (PDF × version)
- `GET /api/extraction` — retrieves persisted results for PDF × version (used by Compare screen)
- `GET/POST /api/warmup` + `/api/warmup/retry` — v3/v4 model download status

**`src/warmup.py`** — Runs `warmup_v3()` and `warmup_v4()` as daemon threads at FastAPI startup. Status dict is thread-safe and read by `/api/warmup`.

**`templates/index.html`** — Single-file ~1000-line vanilla JS app. Three screens:
- **Workspace** — PDF.js viewer with SVG bbox overlay, extraction controls, chunked/table/image tabs
- **Results** — 4×N metrics matrix with CSV export
- **Compare** — Side-by-side two-version diff with per-page delta counts and toggleable per-version overlays

### Extraction timeouts (server.py)

v1=120s, v2=180s, v3=600s, v4=1200s — v3/v4 account for first-run model downloads.

### Known limitations

- Whitespace-delimited tables (no border lines) are detected but skipped by v1/v2 — needs v3/v4 or vision LLM
- Word fusion in tight-spacing PDFs (e.g. S1000D) produces artifacts like `"Documenttitle"`
- Scanned/image-based PDFs not supported — no OCR routing in v1/v2
