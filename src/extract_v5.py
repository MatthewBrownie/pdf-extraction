"""
PDF extraction POC v5 — logical structure tree for tagged PDFs.

Uses pdfminer.six to (1) detect whether the PDF carries a well-formed logical
structure tree, (2) build a per-MCID text map by running the page interpreter
with a lightweight custom device, and (3) walk the StructTreeRoot to produce
semantic chunks (H1-H6, P, …) and table rows (Table/TR/TH/TD).

Returns the same shape as v1-v4: { "chunks": [...], "tables": [...], "images": [...] }

When a PDF is not tagged (or the tags are absent), the extractor returns empty
chunks and tables — that is intentional POC data, not a silent fallback.

Images are always extracted via PyMuPDF.
"""

from __future__ import annotations

import json
import os
import sys

import fitz  # PyMuPDF


# ── role constants ──────────────────────────────────────────────────────────────

_HEADING_ROLES = frozenset({"H", "H1", "H2", "H3", "H4", "H5", "H6"})
_TEXT_ROLES = frozenset({
    "P", "Caption", "Quote", "Note", "Reference", "BibEntry", "Code",
    "Lbl", "LBody",
})
_TABLE_SECTION_ROLES = frozenset({"THead", "TBody", "TFoot"})


# ── page sizes ──────────────────────────────────────────────────────────────────

def _page_sizes(pdf_path: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            out[i + 1] = (page.rect.width, page.rect.height)
    return out


# ── tagged PDF detection ────────────────────────────────────────────────────────

def _is_tagged(pdf_path: str) -> bool:
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdftypes import resolve1
    with open(pdf_path, "rb") as fh:
        doc = PDFDocument(PDFParser(fh))
        mark_info = resolve1(doc.catalog.get("MarkInfo"))
        marked = bool(mark_info and mark_info.get("Marked"))
        return marked and "StructTreeRoot" in doc.catalog


# ── MCID → text map ─────────────────────────────────────────────────────────────

def _build_mcid_map(pdf_path: str) -> dict[tuple[int, int], str]:
    """
    Run the pdfminer page interpreter over every page and collect text
    per (page_index, MCID).  Uses font.decode() + font.to_unichr() so
    CID-mapped fonts are handled correctly.
    """
    from pdfminer.pdfdevice import PDFDevice
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter

    class _MCIDCollector(PDFDevice):
        def __init__(self, rsrcmgr):
            super().__init__(rsrcmgr)
            self.page_idx: int = 0
            self._stack: list[int | None] = []
            self.mcid_texts: dict[tuple[int, int], list[str]] = {}

        def begin_tag(self, tag, props=None):
            mcid = None
            if isinstance(props, dict):
                mcid = props.get("MCID") if props.get("MCID") is not None else props.get(b"MCID")
            self._stack.append(mcid if isinstance(mcid, int) else None)

        def end_tag(self):
            if self._stack:
                self._stack.pop()

        def render_string(self, textstate, seq, ncs, graphicstate):
            mcid = self._stack[-1] if self._stack else None
            if mcid is None:
                return
            font = textstate.font
            chars: list[str] = []
            for obj in seq:
                if isinstance(obj, bytes):
                    try:
                        for cid in font.decode(obj):
                            try:
                                chars.append(font.to_unichr(cid))
                            except Exception:
                                chars.append("\ufffd")
                    except Exception:
                        try:
                            chars.append(obj.decode("latin-1", "replace"))
                        except Exception:
                            pass
            text = "".join(chars)
            if text:
                self.mcid_texts.setdefault((self.page_idx, mcid), []).append(text)

    rsrcmgr = PDFResourceManager()
    device = _MCIDCollector(rsrcmgr)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with open(pdf_path, "rb") as fh:
        parser = PDFParser(fh)
        doc = PDFDocument(parser)
        for page_idx, page in enumerate(PDFPage.create_pages(doc)):
            device.page_idx = page_idx
            interpreter.process_page(page)

    return {k: "".join(v).strip() for k, v in device.mcid_texts.items()}


# ── structure tree helpers ──────────────────────────────────────────────────────

def _role_name(v) -> str:
    from pdfminer.psparser import PSLiteral
    if isinstance(v, PSLiteral):
        return v.name if hasattr(v, "name") else str(v)
    if isinstance(v, bytes):
        return v.decode("latin-1", "replace")
    return str(v or "")


def _apply_role_map(role: str, role_map: dict) -> str:
    seen: set[str] = set()
    current = role
    while current in role_map and current not in seen:
        seen.add(current)
        current = _role_name(role_map[current])
    return current


def _elem_page_no(node: dict, page_xref_map: dict, resolve1) -> int | None:
    """Return 1-indexed page number for a StructElem, or None."""
    from pdfminer.pdftypes import PDFObjRef
    pg = node.get("Pg")
    if isinstance(pg, PDFObjRef):
        idx = page_xref_map.get(pg.objid)
        if idx is not None:
            return idx + 1
    # Recurse into first kid to find a page reference
    kids = node.get("K")
    if kids is None:
        return None
    kids_r = resolve1(kids)
    if not isinstance(kids_r, list):
        kids_r = [kids_r]
    for kid in kids_r:
        kid_r = resolve1(kid)
        if isinstance(kid_r, dict):
            result = _elem_page_no(kid_r, page_xref_map, resolve1)
            if result is not None:
                return result
    return None


def _collect_mcids(node: dict, page_xref_map: dict, resolve1) -> list[tuple[int, int]]:
    """Recursively collect all (page_idx, mcid) pairs from a StructElem."""
    from pdfminer.pdftypes import PDFObjRef

    result: list[tuple[int, int]] = []
    kids = node.get("K")
    if kids is None:
        return result
    kids = resolve1(kids)
    if not isinstance(kids, list):
        kids = [kids]

    pg_ref = node.get("Pg")
    node_page_idx = page_xref_map.get(pg_ref.objid, -1) if isinstance(pg_ref, PDFObjRef) else -1

    for kid in kids:
        kid_r = resolve1(kid)
        if isinstance(kid_r, int):
            result.append((node_page_idx, kid_r))
        elif isinstance(kid_r, dict):
            type_name = _role_name(kid_r.get("Type", ""))
            if type_name == "MCR":
                mcid = kid_r.get("MCID")
                if isinstance(mcid, int):
                    pg = kid_r.get("Pg") or pg_ref
                    page_idx = page_xref_map.get(pg.objid, -1) if isinstance(pg, PDFObjRef) else node_page_idx
                    result.append((page_idx, mcid))
            elif type_name != "OBJR":
                result.extend(_collect_mcids(kid_r, page_xref_map, resolve1))
    return result


def _elem_text(node: dict, page_xref_map: dict, mcid_map: dict, resolve1) -> str:
    actual = node.get("ActualText")
    if actual:
        if isinstance(actual, bytes):
            return actual.decode("utf-16-be", "replace").strip()
        return str(actual).strip()
    pairs = _collect_mcids(node, page_xref_map, resolve1)
    return "".join(mcid_map.get(k, "") for k in pairs).strip()


# ── table extraction ────────────────────────────────────────────────────────────

def _collect_row_cells(tr_node, role_map, page_xref_map, mcid_map, resolve1) -> list[str]:
    kids = tr_node.get("K")
    if not kids:
        return []
    kids = resolve1(kids)
    if not isinstance(kids, list):
        kids = [kids]
    cells: list[str] = []
    for kid in kids:
        kid_r = resolve1(kid)
        if not isinstance(kid_r, dict):
            continue
        role = _apply_role_map(_role_name(kid_r.get("S", "")), role_map)
        if role in ("TH", "TD"):
            cells.append(_elem_text(kid_r, page_xref_map, mcid_map, resolve1))
    return cells


def _collect_table_rows(table_node, role_map, page_xref_map, mcid_map, resolve1) -> list[list[str]]:
    rows: list[list[str]] = []
    kids = table_node.get("K")
    if not kids:
        return rows
    kids = resolve1(kids)
    if not isinstance(kids, list):
        kids = [kids]
    for kid in kids:
        kid_r = resolve1(kid)
        if not isinstance(kid_r, dict):
            continue
        role = _apply_role_map(_role_name(kid_r.get("S", "")), role_map)
        if role in _TABLE_SECTION_ROLES:
            rows.extend(_collect_table_rows(kid_r, role_map, page_xref_map, mcid_map, resolve1))
        elif role == "TR":
            cells = _collect_row_cells(kid_r, role_map, page_xref_map, mcid_map, resolve1)
            if cells:
                rows.append(cells)
    return rows


# ── tree walk ───────────────────────────────────────────────────────────────────

def _walk_node(node, role_map, page_xref_map, mcid_map, resolve1,
               page_sizes, chunks, tables, para_counter) -> None:
    node = resolve1(node)
    if not isinstance(node, dict):
        return

    role = _apply_role_map(_role_name(node.get("S", "")), role_map)

    if role == "Table":
        rows = _collect_table_rows(node, role_map, page_xref_map, mcid_map, resolve1)
        if rows:
            page_no = _elem_page_no(node, page_xref_map, resolve1) or 1
            page_size = page_sizes.get(page_no, (612.0, 792.0))
            tables.append({
                "page": page_no,
                "table_type": "tagged",
                "table": rows,
                "bbox": None,
                "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
            })
        return  # don't recurse into table internals

    if role in _HEADING_ROLES or role in _TEXT_ROLES:
        text = _elem_text(node, page_xref_map, mcid_map, resolve1)
        if text and len(text) >= 3:
            page_no = _elem_page_no(node, page_xref_map, resolve1) or 1
            page_size = page_sizes.get(page_no, (612.0, 792.0))
            para_counter[page_no] = para_counter.get(page_no, 0) + 1
            chunks.append({
                "page": page_no,
                "paragraph": para_counter[page_no],
                "text": text,
                "font_size": 0.0,
                "bbox": None,
                "page_size": [round(page_size[0], 2), round(page_size[1], 2)],
                "category": role,
            })
        return

    # Container roles (Document, Sect, Div, L, LI, Figure, …) — recurse
    kids = node.get("K")
    if kids is None:
        return
    kids = resolve1(kids)
    if not isinstance(kids, list):
        kids = [kids]
    for kid in kids:
        _walk_node(kid, role_map, page_xref_map, mcid_map, resolve1,
                   page_sizes, chunks, tables, para_counter)


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

    # ── tagged detection ──────────────────────────────────────────────────────
    _emit(phase="detection", message="Checking for tagged PDF structure…")
    if not _is_tagged(pdf_path):
        _emit(phase="detection",
              message="PDF is not tagged — structure tree unavailable. Returning empty chunks/tables.")
        chunks: list[dict] = []
        tables: list[dict] = []
    else:
        # ── MCID → text (full page interpreter pass) ──────────────────────────
        _emit(phase="mcid_map", message=f"Building MCID text map ({total_pages} pages)…",
              total=total_pages)
        mcid_map = _build_mcid_map(pdf_path)
        _check_cancel()

        # ── walk structure tree ───────────────────────────────────────────────
        _emit(phase="tree_walk", message="Walking structure tree…")
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdftypes import resolve1

        chunks = []
        tables = []

        with open(pdf_path, "rb") as fh:
            doc = PDFDocument(PDFParser(fh))

            page_xref_map: dict[int, int] = {}
            for idx, page in enumerate(PDFPage.create_pages(doc)):
                page_xref_map[page.pageid] = idx

            root = resolve1(doc.catalog["StructTreeRoot"])
            role_map = resolve1(root.get("RoleMap", {})) or {}

            kids = root.get("K")
            if kids is not None:
                kids = resolve1(kids)
                if not isinstance(kids, list):
                    kids = [kids]
                para_counter: dict[int, int] = {}
                for kid in kids:
                    _walk_node(kid, role_map, page_xref_map, mcid_map, resolve1,
                               page_sizes, chunks, tables, para_counter)

    _check_cancel()

    # ── images ────────────────────────────────────────────────────────────────
    _emit(phase="images", message="Extracting images…", total=total_pages)
    images = _extract_images(pdf_path, images_dir, total_pages, progress_cb)

    result = {"chunks": chunks, "tables": tables, "images": images}

    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    print("Extraction complete (v5 / tagged structure tree).")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Tables : {len(tables)}")
    print(f"  Images : {len(images)}")

    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_v5.py <pdf_path> <output_dir>")
        sys.exit(1)
    extract_pdf(sys.argv[1], sys.argv[2])
