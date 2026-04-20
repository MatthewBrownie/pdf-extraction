"""
PDF extraction POC v4 — backed by docling.

Returns the same shape as v1/v2/v3:
  { "chunks": [...], "tables": [...], "images": [...] }

Each chunk/table/image carries page, bbox (PyMuPDF top-left coords), and
page_size for overlay rendering on the frontend.
"""

from __future__ import annotations

import json
import os
import sys

import fitz  # PyMuPDF — used for page sizes and image cropping


def _page_sizes(pdf_path: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out[i + 1] = (page.rect.width, page.rect.height)
    return out


def _docling_bbox_to_fitz(
    bbox_obj,
    page_size: tuple[float, float],
) -> list[float] | None:
    """
    Convert a docling `BoundingBox` (with `coord_origin` BOTTOMLEFT or
    TOPLEFT) to a fitz top-left bbox in PDF points.
    """
    if bbox_obj is None:
        return None
    try:
        l = float(bbox_obj.l)
        t = float(bbox_obj.t)
        r = float(bbox_obj.r)
        b = float(bbox_obj.b)
    except Exception:
        return None

    pdf_w, pdf_h = page_size
    origin = getattr(bbox_obj, "coord_origin", None)
    origin_name = getattr(origin, "name", None) or str(origin or "")
    if "BOTTOM" in origin_name.upper():
        # bottom-left origin: top means distance from bottom; flip to top-left
        y0 = pdf_h - max(t, b)
        y1 = pdf_h - min(t, b)
    else:
        # already top-left origin
        y0 = min(t, b)
        y1 = max(t, b)
    x0 = min(l, r)
    x1 = max(l, r)
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _first_provenance(item) -> tuple[int | None, list | None]:
    """Return (page_no, fitz_bbox) from the first ProvenanceItem on a docling item."""
    prov_list = getattr(item, "prov", None) or []
    if not prov_list:
        return None, None
    prov = prov_list[0]
    page_no = getattr(prov, "page_no", None)
    bbox_obj = getattr(prov, "bbox", None)
    return page_no, bbox_obj


def _table_to_rows(table_item) -> list[list[str]]:
    """Convert a docling TableItem into list[list[str]]."""
    data = getattr(table_item, "data", None)
    if data is None:
        return []
    grid = getattr(data, "grid", None)
    if grid:
        rows: list[list[str]] = []
        for row in grid:
            cells = []
            for cell in row:
                txt = getattr(cell, "text", "") or ""
                cells.append(txt.strip())
            rows.append(cells)
        return rows
    # Fallback: try export_to_dataframe-style attributes
    table_cells = getattr(data, "table_cells", None) or []
    if not table_cells:
        return []
    max_r = max((getattr(c, "end_row_offset_idx", 0) for c in table_cells), default=0)
    max_c = max((getattr(c, "end_col_offset_idx", 0) for c in table_cells), default=0)
    rows = [["" for _ in range(max_c)] for _ in range(max_r)]
    for c in table_cells:
        r0 = getattr(c, "start_row_offset_idx", 0)
        c0 = getattr(c, "start_col_offset_idx", 0)
        if 0 <= r0 < max_r and 0 <= c0 < max_c:
            rows[r0][c0] = (getattr(c, "text", "") or "").strip()
    return rows


class CancelledExtraction(Exception):
    """Raised inside the extractor when the caller requests cancellation."""


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

    # Lazy import — docling loads ML models the first time it runs
    _emit(phase="loading_models", message="Loading docling (first run downloads layout + table models)…")
    from docling.document_converter import DocumentConverter

    page_sizes = _page_sizes(pdf_path)
    total_pages = len(page_sizes)

    _emit(phase="converting", message=f"Running docling on {total_pages}-page PDF — this can take a while", total=total_pages)
    converter = DocumentConverter()
    conv = converter.convert(pdf_path)
    doc = conv.document
    _emit(phase="post_processing", message="Collecting text, tables, and pictures", total=total_pages)

    chunks: list[dict] = []
    tables: list[dict] = []
    images: list[dict] = []

    para_counters: dict[int, int] = {}
    image_counters: dict[int, int] = {}

    fitz_doc = fitz.open(pdf_path)

    # --- Text items ---
    for i, text_item in enumerate(getattr(doc, "texts", []) or []):
        if i % 50 == 0:
            _check_cancel()
        page_no, bbox_obj = _first_provenance(text_item)
        if page_no is None:
            continue
        page_size = page_sizes.get(page_no, (612.0, 792.0))
        bbox = _docling_bbox_to_fitz(bbox_obj, page_size)
        text = (getattr(text_item, "text", "") or "").strip()
        if len(text) < 3:
            continue
        label = getattr(text_item, "label", "")
        label_str = getattr(label, "value", None) or str(label or "")
        para_counters[page_no] = para_counters.get(page_no, 0) + 1
        chunks.append({
            "page": page_no,
            "paragraph": para_counters[page_no],
            "text": text,
            "font_size": 0.0,
            "bbox": bbox,
            "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            "category": label_str,
        })

    # --- Tables ---
    for table_item in getattr(doc, "tables", []) or []:
        page_no, bbox_obj = _first_provenance(table_item)
        if page_no is None:
            continue
        page_size = page_sizes.get(page_no, (612.0, 792.0))
        bbox = _docling_bbox_to_fitz(bbox_obj, page_size)
        rows = _table_to_rows(table_item)
        tables.append({
            "page": page_no,
            "table": rows,
            "bbox": bbox,
            "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            "table_type": "docling",
        })

    # --- Pictures ---
    for pic_item in getattr(doc, "pictures", []) or []:
        page_no, bbox_obj = _first_provenance(pic_item)
        if page_no is None:
            continue
        page_size = page_sizes.get(page_no, (612.0, 792.0))
        bbox = _docling_bbox_to_fitz(bbox_obj, page_size)
        image_counters[page_no] = image_counters.get(page_no, 0) + 1
        img_num = image_counters[page_no]
        filename = f"page_{page_no}_img_{img_num}.png"
        img_path = os.path.join(images_dir, filename)
        try:
            page = fitz_doc[page_no - 1]
            if bbox is not None:
                clip = fitz.Rect(*bbox)
                pix = page.get_pixmap(clip=clip, dpi=150)
            else:
                pix = page.get_pixmap(dpi=100)
            pix.save(img_path)
        except Exception:
            pass
        images.append({
            "page": page_no,
            "filename": filename,
            "bbox": bbox,
            "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
        })

    fitz_doc.close()

    result = {"chunks": chunks, "tables": tables, "images": images}

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print("Extraction complete (v4 / docling).")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Tables : {len(tables)}")
    print(f"  Images : {len(images)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_v4.py <pdf_path> <output_dir>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
