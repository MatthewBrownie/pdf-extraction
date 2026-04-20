"""
PDF extraction POC v1 — text chunks, tables, and images from a digital PDF.

Stack:
  - PyMuPDF (pymupdf) for text and image extraction
  - pdfplumber for table extraction (default settings)

Usage:
    from extract import extract_pdf
    result = extract_pdf("path/to/file.pdf", "path/to/output_dir")
"""

import json
import os
import statistics
import sys

import fitz  # PyMuPDF
import pdfplumber


def _build_chunk(
    page_num: int,
    para_num: int,
    lines: list[dict],
    page_size: tuple[float, float],
) -> dict | None:
    """Assemble a chunk dict with text, font_size, bbox, and page_size."""
    line_texts: list[str] = []
    all_span_sizes: list[float] = []
    xs0, ys0, xs1, ys1 = [], [], [], []

    for line in lines:
        span_texts = [span["text"] for span in line["spans"]]
        line_texts.append("".join(span_texts))

        for span in line["spans"]:
            size = span.get("size")
            if size:
                all_span_sizes.append(size)

        bx0, by0, bx1, by1 = line["bbox"]
        xs0.append(bx0); ys0.append(by0); xs1.append(bx1); ys1.append(by1)

    text = " ".join(line_texts).strip()
    if len(text) < 3:
        return None

    font_size = statistics.median(all_span_sizes) if all_span_sizes else 0.0
    bbox = [min(xs0), min(ys0), max(xs1), max(ys1)] if xs0 else [0, 0, 0, 0]

    return {
        "page": page_num,
        "paragraph": para_num,
        "text": text,
        "font_size": round(font_size, 2),
        "bbox": [round(v, 2) for v in bbox],
        "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
    }


def _plumber_bbox_to_fitz(bbox, page_height):
    """Convert pdfplumber (x0, top, x1, bottom) to fitz top-left (x0,y0,x1,y1)."""
    x0, top, x1, bottom = bbox
    return [x0, page_height - bottom, x1, page_height - top]


def extract_pdf(pdf_path: str, output_dir: str, progress_cb=None) -> dict:
    """Extract text chunks, tables, and images from a digital PDF."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def _emit(**kw):
        if progress_cb is not None:
            try:
                progress_cb(**kw)
            except Exception:
                pass

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    chunks = []
    tables = []
    images = []

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    _emit(phase="text_and_images", message="Scanning text and images", page=0, total=total_pages)

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        _emit(phase="text_and_images", message=f"Page {page_num}/{total_pages}: text + images", page=page_num, total=total_pages)
        page_size = (page.rect.width, page.rect.height)

        page_dict = page.get_text("dict")
        text_blocks = [b for b in page_dict["blocks"] if b["type"] == 0]

        def block_sort_key(block):
            x0, y0, _x1, _y1 = block["bbox"]
            y_bucket = round(y0 / 20) * 20
            return (y_bucket, x0)

        text_blocks.sort(key=block_sort_key)

        para_num = 1
        for block in text_blocks:
            lines = block["lines"]
            if not lines:
                continue

            current_para_lines: list[dict] = []
            for i, line in enumerate(lines):
                current_para_lines.append(line)
                is_last_line = i == len(lines) - 1
                if not is_last_line:
                    next_line = lines[i + 1]
                    all_sizes = [span["size"] for span in line["spans"] if span.get("size")]
                    font_size = max(all_sizes) if all_sizes else 12.0
                    gap = next_line["bbox"][1] - line["bbox"][3]
                    if gap <= 1.5 * font_size:
                        continue

                chunk = _build_chunk(page_num, para_num, current_para_lines, page_size)
                if chunk is not None:
                    chunks.append(chunk)
                    para_num += 1
                current_para_lines = []

        # Images
        image_list = page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            xref = img_ref[0]
            img_num = img_index + 1
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            filename = f"page_{page_num}_img_{img_num}.png"
            img_path = os.path.join(images_dir, filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            # bbox via page.get_image_bbox (may be empty if image not rendered)
            bbox = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    bbox = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
            except Exception:
                bbox = None
            images.append({
                "page": page_num,
                "filename": filename,
                "bbox": bbox,
                "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            })

    doc.close()

    # Tables (with bbox)
    _emit(phase="tables", message="Detecting tables", page=0, total=total_pages)
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            _emit(phase="tables", message=f"Page {page_num}/{total_pages}: tables", page=page_num, total=total_pages)
            page_height = page.height
            page_size = [round(page.width, 2), round(page_height, 2)]
            found = page.find_tables()
            for tbl in found:
                rows = tbl.extract()
                if not rows:
                    continue
                bbox = _plumber_bbox_to_fitz(tbl.bbox, page_height)
                tables.append({
                    "page": page_num,
                    "table": rows,
                    "bbox": [round(v, 2) for v in bbox],
                    "page_size": page_size,
                })

    result = {"chunks": chunks, "tables": tables, "images": images}

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print(f"Extraction complete.")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Tables : {len(tables)}")
    print(f"  Images : {len(images)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract.py <pdf_path> <output_dir>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
