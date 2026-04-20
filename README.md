# PDF Extraction POC

A local proof-of-concept for extracting structured content (text chunks, tables, embedded images) from digital PDFs, with a browser UI to inspect results.

**Stack:** PyMuPDF · pdfplumber · FastAPI · vanilla HTML/JS

---

## Quick start

```bash
pip install -r requirements.txt
cd src
python -m uvicorn server:app --reload --port 3000
```

Open `http://localhost:3000`.

---

## Usage

1. Pick a PDF from the dropdown — it loads immediately in the **PDF** tab
2. Choose extractor version (v1 or v2)
3. Click **Extract**
4. Browse results across four tabs:

| Tab | Contents |
|---|---|
| **PDF** | Native browser PDF viewer — scroll the source document |
| **Chunks** | Text paragraphs with page number and font size, paginated |
| **Tables** | Extracted tables rendered as HTML, badged by type |
| **Images** | Embedded raster images, click to enlarge |

---

## Project structure

```
pdf-extraction/
├── src/
│   ├── extract.py       # v1 — PyMuPDF + pdfplumber, default settings
│   ├── extract_v2.py    # v2 — table classification + dual-pass detection
│   └── server.py        # FastAPI server (port 3000)
├── templates/
│   └── index.html       # Single-page UI
├── input/
│   ├── docs/            # Source PDFs
│   └── glossary/        # AI terminology CSV (reference)
├── output/              # Per-PDF extraction results written here at runtime
├── research/            # Background reading
│   ├── overview.md          # Tool comparison + recommended stack
│   ├── table-extraction.md  # Table fundamentals, failure modes, benchmarks
│   ├── language-comparison.md  # Python vs Node/Java/Go/C#
│   ├── poc-review.md        # Session review of v2 work
│   └── table_issues.md      # Ground-truth accuracy analysis + improvement proposals
└── requirements.txt
```

---

## Extractor versions

### v1 — baseline
- PyMuPDF for text and images
- pdfplumber with default settings for tables
- Works well on full-grid (ruled) tables; misses h-rules and whitespace tables

### v2 — improved
- Same text/image extraction
- Classifies each table region before extracting: `full_grid` / `h_rules` / `whitespace`
- Dual detection pass (line-based + text-based) with IoU deduplication
- Routes to the appropriate pdfplumber strategy per table type
- Drops whitespace tables (no borders) — accuracy too low without an ML tool

**Ground truth (manual count):**

| Document | Actual tables | Table types |
|---|---|---|
| PMC-S1000DBIKE | ~100+ | S1000D boilerplate procedural tables (references, conditions, equipment, spares, consumables — one set per data module) + ~10 substantive content tables (parts catalog, maintenance schedule, tire pressure, shifter/brake correlations) |
| creative_story_document | 5 | Narrative tables embedded in story: route log, artifact archive, house register, decision matrix, ledger of endings |
| Options for Queue Based Enrichment | 4 | Pros/Cons comparison tables, one per architecture option |
| Latamar Notes | 3 | D&D random-roll tables (d20 \| Occupants/Treasure) |
| Wraidorian | 0 | Pure prose — no tables |
| Threads of Time | 0 | Pure prose — no tables |
| Halcyon | 0 | Pure prose — no tables |
| test | 0 | Three sentences of terminology — no tables |

**Corpus results (v2):**

| Document | Tables found | Notes |
|---|---|---|
| S1000DBIKE | 61 | ToC, parts tables, procedure tables |
| Creative Story | 6 | Rectangle-bordered tables |
| Options Queue | 5 | Comparison tables |
| Latamar Notes | 11 | 3 minor false positives from page borders |
| Wraidorian (RPG) | 6 | Stat and reference tables |
| Threads of Time | 2 | H-rules tables |
| Halcyon | 0 | Correct — pure prose |

---

## Known limitations

- **Whitespace tables** (no border lines) are not extracted — needs Docling or a vision LLM
- **Word fusion** in S1000D PDFs: tight character spacing produces `"Documenttitle"` style artefacts
- **Scanned PDFs** not supported — no OCR routing
- **Large PDFs** (e.g. S1000D spec at 51 MB) will take significant time; the spinner stays up during extraction

---

## Next steps (if taking further)

1. Integrate Docling as a fallback for whitespace-classified table regions (~93–96% accuracy)
2. Add OCR routing: detect empty text layer → Tesseract/Docling
3. Try pdfplumber `layout` extraction mode to fix word fusion
4. Add a page-filter input so large documents don't need full extraction
