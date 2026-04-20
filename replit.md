# PDF Extraction POC

## Overview
A browser-based tool for extracting structured content (text chunks, tables, and images) from digital PDFs. Provides a UI to run extractions and inspect the results.

## Tech Stack
- **Language:** Python 3.12
- **Backend:** FastAPI + Uvicorn (port 5000)
- **PDF Libraries:** PyMuPDF (fitz), pdfplumber
- **Frontend:** Vanilla HTML/JS (served by FastAPI)

## Project Structure
```
pdf-extraction/
├── src/
│   ├── extract.py       # v1 extraction (PyMuPDF + pdfplumber, full-grid tables)
│   ├── extract_v2.py    # v2 extraction (improved table detection/classification)
│   └── server.py        # FastAPI server
├── templates/
│   └── index.html       # Single-page frontend
├── input/
│   ├── docs/            # Source PDFs for processing
│   └── glossary/        # Reference data
├── output/              # Extracted results (JSON + images)
└── requirements.txt
```

## Running the App
```
uvicorn src.server:app --host 0.0.0.0 --port 5000 --reload
```

## API Endpoints
- `GET /` — Serves the UI
- `GET /api/pdfs` — Lists available PDFs in `input/docs/`
- `POST /api/extract` — Triggers extraction for a PDF (`pdf_name`, `version`: "v1" or "v2")
- `GET /images/<stem>/...` — Static: extracted images
- `GET /pdfs/<filename>` — Static: source PDF files

## Extraction Versions
- **v1:** Basic extraction using PyMuPDF + pdfplumber with default settings
- **v2:** Enhanced table detection (classifies tables as full-grid, h-rules, or whitespace; dual-pass detection with IoU deduplication)
