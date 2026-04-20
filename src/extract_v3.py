"""
PDF extraction POC v3 — backed by unstructured.io's PDF partitioner.

Returns the same shape as v1/v2:
  { "chunks": [...], "tables": [...], "images": [...] }

Each chunk/table/image carries page, bbox (PyMuPDF top-left coords), and
page_size for overlay rendering on the frontend.
"""

from __future__ import annotations

import json
import os
import sys

import fitz  # PyMuPDF — used for page sizes & rendering crops for image elements


def _page_sizes(pdf_path: str) -> dict[int, tuple[float, float]]:
    """Return {1-indexed page number: (width, height)} in PDF points."""
    out: dict[int, tuple[float, float]] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out[i + 1] = (page.rect.width, page.rect.height)
    return out


def _coords_to_bbox(
    coords,
    page_size: tuple[float, float],
) -> list[float] | None:
    """
    Convert an unstructured `metadata.coordinates` object to a fitz top-left
    bbox in PDF points.

    unstructured returns coordinates in pixels at the layout-detection
    resolution. The `system` carries `width` / `height` so we can rescale
    to the actual PDF page size.
    """
    if coords is None:
        return None
    points = getattr(coords, "points", None) or coords.get("points") if isinstance(coords, dict) else getattr(coords, "points", None)
    if not points:
        return None
    system = getattr(coords, "system", None)
    if system is None and isinstance(coords, dict):
        system = coords.get("system")

    sys_w = getattr(system, "width", None) if system else None
    sys_h = getattr(system, "height", None) if system else None
    if isinstance(system, dict):
        sys_w = system.get("width")
        sys_h = system.get("height")

    pdf_w, pdf_h = page_size
    sx = pdf_w / sys_w if sys_w and sys_w > 0 else 1.0
    sy = pdf_h / sys_h if sys_h and sys_h > 0 else 1.0

    xs = [p[0] * sx for p in points]
    ys = [p[1] * sy for p in points]
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def _table_html_to_rows(html: str) -> list[list[str]]:
    """Best-effort HTML <table> → list[list[str]] using stdlib HTMLParser."""
    from html.parser import HTMLParser

    class _T(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self.cur_row: list[str] | None = None
            self.cur_cell: list[str] | None = None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.cur_row = []
            elif tag in ("td", "th"):
                self.cur_cell = []

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.cur_cell is not None and self.cur_row is not None:
                self.cur_row.append("".join(self.cur_cell).strip())
                self.cur_cell = None
            elif tag == "tr" and self.cur_row is not None:
                self.rows.append(self.cur_row)
                self.cur_row = None

        def handle_data(self, data):
            if self.cur_cell is not None:
                self.cur_cell.append(data)

    parser = _T()
    parser.feed(html or "")
    return parser.rows


def extract_pdf(pdf_path: str, output_dir: str) -> dict:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Lazy import — unstructured pulls heavy deps at import time
    from unstructured.partition.pdf import partition_pdf

    page_sizes = _page_sizes(pdf_path)

    # `hi_res` strategy is needed to detect tables and images with bboxes.
    # If a model isn't available we fall back to `fast`, which still gives
    # us text + coordinates but no tables/images.
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False,  # we'll crop ourselves with PyMuPDF
        )
    except Exception:
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
        )

    chunks: list[dict] = []
    tables: list[dict] = []
    images: list[dict] = []

    para_counters: dict[int, int] = {}
    image_counters: dict[int, int] = {}

    fitz_doc = fitz.open(pdf_path)

    for el in elements:
        category = type(el).__name__
        meta = getattr(el, "metadata", None)
        page_num = getattr(meta, "page_number", None) if meta else None
        if not page_num:
            page_num = 1
        page_size = page_sizes.get(page_num, (612.0, 792.0))
        coords = getattr(meta, "coordinates", None) if meta else None
        bbox = _coords_to_bbox(coords, page_size)

        page_size_field = [round(page_size[0], 2), round(page_size[1], 2)]

        if category in ("Table",):
            text = getattr(el, "text", "") or ""
            html = ""
            if meta is not None:
                html = getattr(meta, "text_as_html", "") or ""
            rows = _table_html_to_rows(html) if html else []
            if not rows and text:
                # fallback: split text into one-cell rows so something renders
                rows = [[ln] for ln in text.splitlines() if ln.strip()]
            tables.append({
                "page": page_num,
                "table": rows,
                "bbox": bbox,
                "page_size": page_size_field,
                "table_type": "unstructured",
            })
        elif category in ("Image", "Figure", "FigureCaption", "PageBreak") and category in ("Image", "Figure"):
            image_counters[page_num] = image_counters.get(page_num, 0) + 1
            img_num = image_counters[page_num]
            filename = f"page_{page_num}_img_{img_num}.png"
            img_path = os.path.join(images_dir, filename)
            # Crop the bbox region from the source page and save as PNG
            try:
                if bbox is not None:
                    page = fitz_doc[page_num - 1]
                    clip = fitz.Rect(*bbox)
                    pix = page.get_pixmap(clip=clip, dpi=150)
                    pix.save(img_path)
                else:
                    # No bbox — render the whole page at low DPI
                    page = fitz_doc[page_num - 1]
                    pix = page.get_pixmap(dpi=100)
                    pix.save(img_path)
            except Exception:
                # If render fails, still record the element
                pass
            images.append({
                "page": page_num,
                "filename": filename,
                "bbox": bbox,
                "page_size": page_size_field,
            })
        else:
            text = (getattr(el, "text", "") or "").strip()
            if len(text) < 3:
                continue
            para_counters[page_num] = para_counters.get(page_num, 0) + 1
            chunks.append({
                "page": page_num,
                "paragraph": para_counters[page_num],
                "text": text,
                "font_size": 0.0,
                "bbox": bbox,
                "page_size": page_size_field,
                "category": category,
            })

    fitz_doc.close()

    result = {"chunks": chunks, "tables": tables, "images": images}

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print("Extraction complete (v3 / unstructured.io).")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Tables : {len(tables)}")
    print(f"  Images : {len(images)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_v3.py <pdf_path> <output_dir>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
