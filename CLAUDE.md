# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A proof-of-concept FastAPI app that extracts structured content (text chunks, tables, embedded images) from digital PDFs and renders results in a browser UI with four tabs: PDF viewer, text chunks, tables, images.

No test suite, no linting config, no CI — this is intentionally a POC.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (browse to http://localhost:3000)
cd src
python -m uvicorn server:app --reload --port 3000

# Run extraction from the CLI (v1)
cd src
python extract.py /path/to/input.pdf /path/to/output_dir

# Run extraction from the CLI (v2)
cd src
python extract_v2.py /path/to/input.pdf /path/to/output_dir
```

## Architecture

### Data flow

1. User picks a PDF from the sidebar dropdown and clicks **Extract**
2. Browser POSTs to `/api/extract` with `{ pdf_name, version }`
3. `server.py` calls either `extract.py` or `extract_v2.py`
4. Extractor writes results to `output/<pdf_stem>/`:
   - `chunks.json` — text paragraphs with page number, font size, text
   - `tables.json` — table grid data + `table_type` field (v2 only)
   - `images.json` — embedded image metadata
   - `images/` — extracted PNG files
5. Server returns the same JSON to the browser
6. `templates/index.html` renders each tab from that response

### Key modules

**`src/server.py`** — FastAPI app. Mounts `/pdfs/` → `input/docs/` and `/images/` → `output/`. The `ExtractRequest` Pydantic model validates `pdf_name` and `version` ("v1" or "v2").

**`src/extract.py`** (v1 baseline)
- **Text:** PyMuPDF (`fitz`) block/line/span parsing with two-column y-bucketing (20pt threshold), paragraph merging (gap > 1.5× font size), 3-char minimum filter
- **Tables:** pdfplumber defaults — works well on full-grid ruled tables, misses h-rules and whitespace tables
- **Images:** PyMuPDF raster image extraction, saved as PNG

**`src/extract_v2.py`** (v2 — improved tables)
- Same text and image logic as v1
- Dual-pass table detection: `find_tables()` with line-based strategy, then text-based strategy; IoU > 0.5 deduplication
- `_classify_table()` inspects PDF drawing objects to categorise each table as `full_grid`, `h_rules`, or `whitespace`; whitespace tables are skipped
- Per-type extraction settings in `_TABLE_SETTINGS` dict

**`templates/index.html`** — 937-line single-page vanilla JS/HTML dark-theme UI. No build step — served directly by FastAPI's `Jinja2Templates`.

### Known limitations

- Scanned PDFs not supported (no OCR)
- Word fusion artifacts in tight S1000D-style PDFs
- Large PDFs can take significant extraction time (no streaming progress)
- v2 still produces false positives on decorative borders and D20 roll-table layouts

### Input/output locations

- Source PDFs: `input/docs/`
- Extraction results: `output/<pdf_stem>/` (written at runtime, gitignored)
- Glossary reference (unused by extraction): `input/glossary/`
