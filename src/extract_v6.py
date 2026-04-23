"""
PDF extraction POC v6 — Camelot lattice mode for tables.

Uses Camelot's lattice flavour (OpenCV morphological line detection) for table
extraction and PyMuPDF for text chunks and images. The key thing Camelot adds
that v1-v5 don't: a per-table `parsing_report.accuracy` score, stored in the
output as `camelot_accuracy`. Use this as a selection signal when building a
"best extractor per table" router.

Only lattice mode is used. Stream is omitted — it silently corrupts multi-table
pages. The newer network/hybrid flavours are not yet battle-tested.

Returns the same shape as v1-v5: { "chunks": [...], "tables": [...], "images": [...] }

Requires: camelot-py[cv] (Camelot + OpenCV) and Ghostscript installed on the
system. If Ghostscript is missing, table extraction will raise.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import fitz  # PyMuPDF


# ── page sizes ──────────────────────────────────────────────────────────────────

def _page_sizes(pdf_path: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out[i + 1] = (page.rect.width, page.rect.height)
    return out


# ── text extraction (PyMuPDF, same logic as v1) ─────────────────────────────────

def _build_chunk(page_num, para_num, lines, page_size) -> dict | None:
    line_texts, all_sizes, xs0, ys0, xs1, ys1 = [], [], [], [], [], []
    for line in lines:
        line_texts.append("".join(s["text"] for s in line["spans"]))
        for span in line["spans"]:
            if span.get("size"):
                all_sizes.append(span["size"])
        bx0, by0, bx1, by1 = line["bbox"]
        xs0.append(bx0); ys0.append(by0); xs1.append(bx1); ys1.append(by1)
    text = " ".join(line_texts).strip()
    if len(text) < 3:
        return None
    font_size = statistics.median(all_sizes) if all_sizes else 0.0
    bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
    return {
        "page": page_num,
        "paragraph": para_num,
        "text": text,
        "font_size": round(font_size, 2),
        "bbox": [round(v, 2) for v in bbox],
        "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
    }


def _extract_text(pdf_path: str, total_pages: int, progress_cb) -> list[dict]:
    chunks: list[dict] = []
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        if progress_cb:
            try:
                progress_cb(phase="text", message=f"Page {page_num}/{total_pages}: text",
                            page=page_num, total=total_pages)
            except Exception:
                pass
        page_size = (page.rect.width, page.rect.height)
        blocks = sorted(
            [b for b in page.get_text("dict")["blocks"] if b["type"] == 0],
            key=lambda b: (round(b["bbox"][1] / 20) * 20, b["bbox"][0]),
        )
        para_num = 1
        for block in blocks:
            lines = block["lines"]
            current: list[dict] = []
            for i, line in enumerate(lines):
                current.append(line)
                if i < len(lines) - 1:
                    sizes = [s["size"] for s in line["spans"] if s.get("size")]
                    fs = max(sizes) if sizes else 12.0
                    if lines[i + 1]["bbox"][1] - line["bbox"][3] <= 1.5 * fs:
                        continue
                chunk = _build_chunk(page_num, para_num, current, page_size)
                if chunk:
                    chunks.append(chunk)
                    para_num += 1
                current = []
    doc.close()
    return chunks


# ── image extraction (PyMuPDF, same as v1) ─────────────────────────────────────

def _extract_images(pdf_path: str, images_dir: str, total_pages: int, progress_cb) -> list[dict]:
    images: list[dict] = []
    doc = fitz.open(pdf_path)
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        if progress_cb:
            try:
                progress_cb(phase="images", message=f"Page {page_num}/{total_pages}: images",
                            page=page_num, total=total_pages)
            except Exception:
                pass
        page_size = (page.rect.width, page.rect.height)
        for img_index, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            filename = f"page_{page_num}_img_{img_index + 1}.png"
            with open(os.path.join(images_dir, filename), "wb") as f:
                f.write(base_image["image"])
            bbox = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
            except Exception:
                pass
            images.append({
                "page": page_num,
                "filename": filename,
                "bbox": bbox,
                "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            })
    doc.close()
    return images


# ── Camelot table extraction ────────────────────────────────────────────────────

def _camelot_bbox_to_fitz(table, page_height: float) -> list[float] | None:
    """Convert Camelot cell bboxes (PDF bottom-left origin) to fitz top-left."""
    try:
        xs1, ys1, xs2, ys2 = [], [], [], []
        for row in table.cells:
            for cell in row:
                xs1.append(cell.x1)
                ys1.append(cell.y1)
                xs2.append(cell.x2)
                ys2.append(cell.y2)
        if not xs1:
            return None
        x0, y_bottom = min(xs1), min(ys1)
        x1, y_top = max(xs2), max(ys2)
        return [round(x0, 2), round(page_height - y_top, 2),
                round(x1, 2), round(page_height - y_bottom, 2)]
    except Exception:
        return None


def _extract_tables_camelot(
    pdf_path: str,
    page_sizes: dict[int, tuple[float, float]],
    progress_cb,
) -> list[dict]:
    import camelot

    if progress_cb:
        try:
            progress_cb(phase="tables", message="Running Camelot lattice detection on all pages…")
        except Exception:
            pass

    camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")

    tables: list[dict] = []
    for t in camelot_tables:
        page_no = t.parsing_report.get("page", 1)
        page_size = page_sizes.get(page_no, (612.0, 792.0))
        bbox = _camelot_bbox_to_fitz(t, page_size[1])
        rows = t.df.fillna("").astype(str).values.tolist()
        tables.append({
            "page": page_no,
            "table_type": "camelot_lattice",
            "table": rows,
            "bbox": bbox,
            "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            "camelot_accuracy": round(t.parsing_report.get("accuracy", 0.0), 2),
            "camelot_whitespace": round(t.parsing_report.get("whitespace", 0.0), 2),
        })

    return tables


# ── main entry point ────────────────────────────────────────────────────────────

class CancelledExtraction(Exception):
    pass


def extract_pdf(pdf_path: str, output_dir: str, progress_cb=None, cancel_check=None) -> dict:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _check_cancel():
        if cancel_check is not None and cancel_check():
            raise CancelledExtraction()

    def _emit(**kw):
        _check_cancel()
        if progress_cb is not None:
            try:
                progress_cb(**kw)
            except CancelledExtraction:
                raise
            except Exception:
                pass
        _check_cancel()

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    page_sizes = _page_sizes(pdf_path)
    total_pages = len(page_sizes)

    _emit(phase="text", message="Extracting text…", total=total_pages)
    chunks = _extract_text(pdf_path, total_pages, progress_cb)
    _check_cancel()

    _emit(phase="images", message="Extracting images…", total=total_pages)
    images = _extract_images(pdf_path, images_dir, total_pages, progress_cb)
    _check_cancel()

    _emit(phase="tables", message="Running Camelot lattice detection…")
    tables = _extract_tables_camelot(pdf_path, page_sizes, progress_cb)
    _check_cancel()

    result = {"chunks": chunks, "tables": tables, "images": images}

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print("Extraction complete (v6 / Camelot lattice).")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Tables : {len(tables)}")
    print(f"  Images : {len(images)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_v6.py <pdf_path> <output_dir>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
