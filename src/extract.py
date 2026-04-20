"""
PDF extraction POC — text chunks, tables, and images from a digital PDF.

Stack:
  - PyMuPDF (pymupdf) for text and image extraction
  - pdfplumber for table extraction

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


def _build_chunk(page_num: int, para_num: int, lines: list[dict]) -> dict | None:
    """
    Assemble a chunk dict from a list of PyMuPDF line dicts.

    Steps:
      - Join spans within a line (no separator — spans are pre-segmented).
      - Join lines within the paragraph with a single space.
      - Compute median span size across all spans for font_size.
      - Return None if the assembled text is shorter than 3 characters.
    """
    line_texts: list[str] = []
    all_span_sizes: list[float] = []

    for line in lines:
        span_texts = [span["text"] for span in line["spans"]]
        line_texts.append("".join(span_texts))

        for span in line["spans"]:
            size = span.get("size")
            if size:
                all_span_sizes.append(size)

    text = " ".join(line_texts).strip()

    # ASSUMPTION: chunks shorter than 3 characters are noise (e.g. stray page
    # numbers, single punctuation marks). Increase this threshold if short
    # headers or labels need to be dropped too.
    if len(text) < 3:
        return None

    font_size = statistics.median(all_span_sizes) if all_span_sizes else 0.0

    return {
        "page": page_num,
        "paragraph": para_num,
        "text": text,
        "font_size": round(font_size, 2),
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
            "tables": [{"page": int, "table": list[list]}, ...],
            "images": [{"page": int, "path": str}, ...],
        }
    """
    # Basic guard — not full error handling (POC)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    chunks = []
    tables = []
    images = []

    # --- Text + Images via PyMuPDF ---
    # ASSUMPTION: pages are 1-indexed in output (fitz uses 0-indexed internally)
    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        page_num = page_index + 1  # 1-indexed

        # --- Text chunks ---
        # Use dict mode to get structured block/line/span data with layout info.
        page_dict = page.get_text("dict")

        # Step 1: keep only text blocks (type 0); type 1 is image blocks.
        text_blocks = [b for b in page_dict["blocks"] if b["type"] == 0]

        # Step 2: sort blocks by reading order.
        # Primary sort: y0 (top of block). Two-column handling: if two blocks
        # have y0 values within 20pt of each other, sort by x0 (left col first).
        # ASSUMPTION: 20pt threshold is enough to treat blocks as on the same
        # visual row. Single-column PDFs are unaffected — y0 values will differ
        # by more than 20pt between successive blocks.
        def block_sort_key(block: dict) -> tuple[float, float]:
            x0, y0, _x1, _y1 = block["bbox"]
            # Round y0 to the nearest 20pt bucket so nearby rows sort together,
            # then x0 breaks the tie (left column before right column).
            # ASSUMPTION: 20pt bucket — see above.
            y_bucket = round(y0 / 20) * 20
            return (y_bucket, x0)

        text_blocks.sort(key=block_sort_key)

        # Steps 3–5: within each block, split lines into paragraphs by gap.
        para_num = 1
        for block in text_blocks:
            lines = block["lines"]
            if not lines:
                continue

            # Each paragraph accumulates lines until a large gap signals a break.
            current_para_lines: list[dict] = []

            for i, line in enumerate(lines):
                current_para_lines.append(line)

                # Determine whether to break after this line.
                is_last_line = i == len(lines) - 1
                if not is_last_line:
                    next_line = lines[i + 1]

                    # Dominant font size for the current line.
                    all_sizes = [
                        span["size"]
                        for span in line["spans"]
                        if span.get("size")
                    ]
                    font_size = max(all_sizes) if all_sizes else 12.0

                    # Gap between bottom of current line and top of next line.
                    gap = next_line["bbox"][1] - line["bbox"][3]

                    # ASSUMPTION: a gap greater than 1.5× the dominant font size
                    # indicates a paragraph break (e.g. blank-line spacing).
                    # Tighter gaps (leading, inline spacing) stay as one paragraph.
                    if gap <= 1.5 * font_size:
                        continue  # same paragraph — keep accumulating

                # Flush the current paragraph (either gap break or end of block).
                chunk = _build_chunk(page_num, para_num, current_para_lines)
                if chunk is not None:
                    chunks.append(chunk)
                    para_num += 1
                current_para_lines = []

        # --- Images ---
        # ASSUMPTION: we extract all raster images embedded in the PDF page.
        # Vector graphics (drawn with PDF operators) are not extracted — they
        # would require page rendering to a bitmap, which is out of scope here.
        image_list = page.get_images(full=True)
        for img_index, img_ref in enumerate(image_list):
            xref = img_ref[0]  # xref is the internal object reference
            img_num = img_index + 1  # 1-indexed
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # ASSUMPTION: always save as .png regardless of source format.
            # PyMuPDF can tell us the real extension via base_image["ext"],
            # but PNG is safe and lossless for a POC.
            filename = f"page_{page_num}_img_{img_num}.png"
            img_path = os.path.join(images_dir, filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images.append({"page": page_num, "filename": filename})

    doc.close()

    # --- Tables via pdfplumber ---
    # ASSUMPTION: pdfplumber's default table detection settings are used.
    # These work well for PDFs with visible ruling lines. Tables with
    # whitespace-only delimiters may not be detected at all.
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1  # 1-indexed
            # ASSUMPTION: extract_tables() returns each table as a list of rows,
            # each row as a list of cell strings (or None for empty cells).
            for raw_table in page.extract_tables():
                tables.append({"page": page_num, "table": raw_table})

    result = {"chunks": chunks, "tables": tables, "images": images}

    # Write JSON output to disk
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
    # TODO(prod): replace with argparse or a CLI framework
    if len(sys.argv) != 3:
        print("Usage: python extract.py <pdf_path> <output_dir>")
        sys.exit(1)

    pdf_path_arg = sys.argv[1]
    output_dir_arg = sys.argv[2]

    data = extract_pdf(pdf_path_arg, output_dir_arg)

    if data["chunks"]:
        first = data["chunks"][0]
        preview = first["text"][:120].replace("\n", " ")
        print(f"\nFirst chunk  (p{first['page']}, para {first['paragraph']}): {preview!r}")

    if data["tables"]:
        first_t = data["tables"][0]
        print(f"First table  (p{first_t['page']}): {len(first_t['table'])} rows x "
              f"{len(first_t['table'][0]) if first_t['table'] else 0} cols")

    if data["images"]:
        print(f"First image  : {data['images'][0]['filename']}")
