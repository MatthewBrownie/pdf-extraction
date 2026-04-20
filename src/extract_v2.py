"""
PDF extraction POC v2 — text chunks, tables, and images from a digital PDF.

Stack:
  - PyMuPDF (pymupdf) for text, image extraction, and drawing inspection
  - pdfplumber for table extraction with per-table strategy routing

Table types extracted:
  - full_grid: vertical + horizontal ruling lines
  - h_rules:   horizontal lines only (no verticals)

Whitespace-delimited tables (no lines) are not extracted — accuracy is too
low without an ML-based tool (e.g. Docling).

Usage:
    from extract_v2 import extract_pdf
    result = extract_pdf("path/to/file.pdf", "path/to/output_dir")
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import fitz  # PyMuPDF
import pdfplumber

# pdfplumber extraction settings per table type
_TABLE_SETTINGS: dict[str, dict] = {
    'full_grid': {
        'vertical_strategy': 'lines',
        'horizontal_strategy': 'lines',
    },
    'h_rules': {
        'vertical_strategy': 'text',
        'horizontal_strategy': 'lines',
    },
}

# Second find_tables() pass to catch h_rules tables that line-based detection
# misses (horizontal lines only — no verticals for pdfplumber to anchor on).
_TEXT_FIND_SETTINGS: dict[str, object] = {
    'vertical_strategy': 'text',
    'horizontal_strategy': 'text',
    'min_words_vertical': 2,
    'min_words_horizontal': 1,
}


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union for two (x0, y0, x1, y1) bboxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _dedup_tables(table_objs: list, threshold: float = 0.5) -> list:
    """
    Remove duplicate table objects whose bboxes overlap above `threshold` IoU.
    First occurrence wins; subsequent overlapping entries are dropped.
    """
    kept: list = []
    for candidate in table_objs:
        overlaps_existing = any(
            _bbox_iou(candidate.bbox, existing.bbox) > threshold
            for existing in kept
        )
        if not overlaps_existing:
            kept.append(candidate)
    return kept


def _is_vertical_segment(path: dict) -> bool:
    """Return True if a drawing path segment is a near-vertical line."""
    for item in path.get('items', []):
        if item[0] != 'l':
            continue
        # item: ('l', p1, p2) where p1/p2 are fitz.Point
        _op, p1, p2 = item
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        if dx < 2 and dy > 10:
            return True
    return False


def _is_horizontal_segment(path: dict) -> bool:
    """Return True if a drawing path segment is a near-horizontal line."""
    for item in path.get('items', []):
        if item[0] != 'l':
            continue
        _op, p1, p2 = item
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        if dy < 2 and dx > 10:
            return True
    return False


def _bbox_contains_path(fitz_bbox: tuple[float, float, float, float], path: dict) -> bool:
    """
    Return True if any point of the path falls within fitz_bbox.

    fitz_bbox is (x0, y0, x1, y1) in PyMuPDF coordinates (top-left origin).
    """
    bx0, by0, bx1, by1 = fitz_bbox
    for item in path.get('items', []):
        if item[0] != 'l':
            continue
        _op, p1, p2 = item
        # Either endpoint inside the bbox is enough
        for pt in (p1, p2):
            if bx0 <= pt.x <= bx1 and by0 <= pt.y <= by1:
                return True
    return False


def _plumber_bbox_to_fitz(
    bbox: tuple[float, float, float, float],
    page_height: float,
) -> tuple[float, float, float, float]:
    """
    Convert a pdfplumber bbox to PyMuPDF coordinates.

    pdfplumber uses PDF user space: origin at bottom-left, y increases upward.
    PyMuPDF uses: origin at top-left, y increases downward.

    bbox: (x0, top, x1, bottom) in pdfplumber space
    """
    x0, top, x1, bottom = bbox
    fitz_y0 = page_height - bottom
    fitz_y1 = page_height - top
    return (x0, fitz_y0, x1, fitz_y1)


def _rect_overlaps_bbox(
    rect: tuple[float, float, float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    """Return True if two (x0, y0, x1, y1) rectangles overlap."""
    rx0, ry0, rx1, ry1 = rect
    bx0, by0, bx1, by1 = bbox
    return rx0 < bx1 and rx1 > bx0 and ry0 < by1 and ry1 > by0


def _classify_table(
    plumber_bbox: tuple[float, float, float, float],
    page_height: float,
    drawings: list[dict],
) -> str:
    """
    Classify a table region as 'full_grid', 'h_rules', or 'whitespace' based
    on which line types are present within its bounding box.

    PDFs draw table borders either as line segments ('l' items) or as filled
    rectangles ('re' items). Both are checked. 'whitespace' means no ruling
    lines detected — caller should discard these.
    """
    fitz_bbox = _plumber_bbox_to_fitz(plumber_bbox, page_height)

    bx0, by0, bx1, by1 = fitz_bbox
    table_area = (bx1 - bx0) * (by1 - by0)

    for path in drawings:
        items = path.get('items', [])
        for item in items:
            if item[0] == 're':
                r = item[1]
                cell_area = (r.x1 - r.x0) * (r.y1 - r.y0)
                # Skip rectangles larger than the table itself — these are page
                # background fills, not cell borders.
                if cell_area >= table_area:
                    continue
                if _rect_overlaps_bbox((r.x0, r.y0, r.x1, r.y1), fitz_bbox):
                    return 'full_grid'

    has_vertical = any(
        _is_vertical_segment(p) and _bbox_contains_path(fitz_bbox, p)
        for p in drawings
    )
    if has_vertical:
        return 'full_grid'

    has_horizontal = any(
        _is_horizontal_segment(p) and _bbox_contains_path(fitz_bbox, p)
        for p in drawings
    )
    if has_horizontal:
        return 'h_rules'

    return 'whitespace'


def _is_valid_table(rows: list[list] | None) -> bool:
    """Reject single-row, single-column, and empty extractions — common false positives."""
    if not rows or len(rows) < 2:
        return False
    max_cols = max((len(r) for r in rows if r), default=0)
    return max_cols >= 2


def _extract_tables_from_page(
    plumber_page: pdfplumber.page.Page,
    fitz_page: fitz.Page,
) -> list[dict]:
    """
    Find all tables on a page, classify each by line geometry, and extract
    using the appropriate pdfplumber strategy.
    """
    page_num = plumber_page.page_number  # pdfplumber is 1-indexed
    page_height = plumber_page.height
    drawings = fitz_page.get_drawings()

    line_tables = plumber_page.find_tables()
    text_tables = plumber_page.find_tables(_TEXT_FIND_SETTINGS)
    merged = _dedup_tables(list(line_tables) + list(text_tables))

    results = []
    for table_obj in merged:
        table_type = _classify_table(table_obj.bbox, page_height, drawings)

        if table_type == 'whitespace':
            continue

        if table_type == 'full_grid':
            rows = table_obj.extract(x_tolerance=5)
        else:
            # h_rules: crop and re-extract with text vertical strategy.
            # Table.extract() doesn't accept settings; extract_tables() does.
            # Clamp bbox to page bounds — text-strategy find_tables() can return
            # regions that slightly exceed the page edge.
            settings = {**_TABLE_SETTINGS['h_rules'], 'text_x_tolerance': 5}
            px0, py0, px1, py1 = plumber_page.bbox
            tx0, ty0, tx1, ty1 = table_obj.bbox
            safe_bbox = (max(tx0, px0), max(ty0, py0), min(tx1, px1), min(ty1, py1))
            cropped = plumber_page.crop(safe_bbox)
            extracted = cropped.extract_tables(settings)
            rows = extracted[0] if extracted else table_obj.extract(x_tolerance=5)

        if not _is_valid_table(rows):
            continue

        results.append({
            'page': page_num,
            'table_type': table_type,
            'table': rows,
        })

    return results


def _build_chunk(page_num: int, para_num: int, lines: list[dict]) -> dict | None:
    line_texts: list[str] = []
    all_span_sizes: list[float] = []

    for line in lines:
        span_texts = [span['text'] for span in line['spans']]
        line_texts.append(''.join(span_texts))
        for span in line['spans']:
            size = span.get('size')
            if size:
                all_span_sizes.append(size)

    text = ' '.join(line_texts).strip()
    if len(text) < 3:
        return None

    font_size = statistics.median(all_span_sizes) if all_span_sizes else 0.0
    return {
        'page': page_num,
        'paragraph': para_num,
        'text': text,
        'font_size': round(font_size, 2),
    }


def extract_pdf(pdf_path: str, output_dir: str) -> dict:
    """
    Extract text chunks, tables, and images from a digital PDF.

    Args:
        pdf_path:   Path to the source PDF file.
        output_dir: Root directory for any saved output (images go in a subdir).

    Returns:
        {
            "chunks": [{"page": int, "paragraph": int, "text": str, "font_size": float}, ...],
            "tables": [{"page": int, "table_type": str, "table": list[list]}, ...],
            "images": [{"page": int, "filename": str}, ...],
        }
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    chunks: list[dict] = []
    tables: list[dict] = []
    images: list[dict] = []

    fitz_doc = fitz.open(pdf_path)

    for page_index, fitz_page in enumerate(fitz_doc):
        page_num = page_index + 1

        # --- Text chunks ---
        page_dict = fitz_page.get_text('dict')
        text_blocks = [b for b in page_dict['blocks'] if b['type'] == 0]

        def block_sort_key(block: dict) -> tuple[float, float]:
            x0, y0, _x1, _y1 = block['bbox']
            y_bucket = round(y0 / 20) * 20
            return (y_bucket, x0)

        text_blocks.sort(key=block_sort_key)

        para_num = 1
        for block in text_blocks:
            lines = block['lines']
            if not lines:
                continue

            current_para_lines: list[dict] = []

            for i, line in enumerate(lines):
                current_para_lines.append(line)
                is_last_line = i == len(lines) - 1
                if not is_last_line:
                    next_line = lines[i + 1]
                    all_sizes = [span['size'] for span in line['spans'] if span.get('size')]
                    font_size = max(all_sizes) if all_sizes else 12.0
                    gap = next_line['bbox'][1] - line['bbox'][3]
                    if gap <= 1.5 * font_size:
                        continue

                chunk = _build_chunk(page_num, para_num, current_para_lines)
                if chunk is not None:
                    chunks.append(chunk)
                    para_num += 1
                current_para_lines = []

        # --- Images ---
        image_list = fitz_page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            xref = img_ref[0]
            img_num = img_index + 1
            base_image = fitz_doc.extract_image(xref)
            image_bytes = base_image['image']
            filename = f'page_{page_num}_img_{img_num}.png'
            img_path = os.path.join(images_dir, filename)
            with open(img_path, 'wb') as f:
                f.write(image_bytes)
            images.append({'page': page_num, 'filename': filename})

    # --- Tables via pdfplumber + PyMuPDF drawing inspection ---
    # Both files are opened independently — PyMuPDF is kept open above for
    # image extraction; pdfplumber needs its own handle for table detection.
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, plumber_page in enumerate(pdf.pages):
            fitz_page = fitz_doc[page_index]
            page_tables = _extract_tables_from_page(plumber_page, fitz_page)
            tables.extend(page_tables)

    fitz_doc.close()

    result = {'chunks': chunks, 'tables': tables, 'images': images}

    with open(os.path.join(output_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'tables.json'), 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, 'images.json'), 'w', encoding='utf-8') as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print('Extraction complete.')
    print(f'  Chunks : {len(chunks)}')
    print(f'  Tables : {len(tables)}')
    print(f'  Images : {len(images)}')

    return result


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python extract_v2.py <pdf_path> <output_dir>')
        sys.exit(1)

    pdf_path_arg = sys.argv[1]
    output_dir_arg = sys.argv[2]

    data = extract_pdf(pdf_path_arg, output_dir_arg)

    if data['chunks']:
        first = data['chunks'][0]
        preview = first['text'][:120].replace('\n', '  ')
        print(f"\nFirst chunk  (p{first['page']}, para {first['paragraph']}): {preview!r}")

    if data['tables']:
        first_t = data['tables'][0]
        row_count = len(first_t['table'])
        col_count = len(first_t['table'][0]) if first_t['table'] else 0
        print(
            f"First table  (p{first_t['page']}, type={first_t['table_type']}): "
            f'{row_count} rows x {col_count} cols'
        )

    if data['images']:
        print(f"First image  : {data['images'][0]['filename']}")
