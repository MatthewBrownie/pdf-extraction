"""
Convert ExtractionResult (Claude native shape) to the standard POC shape:
  { chunks: [...], tables: [...], images: [...] }

This lets results be compared against the pdf-extraction POC versions.
Figures become chunks with category="Figure". No actual image files are
produced since Claude returns descriptions only.
"""

from __future__ import annotations

from .extractor import ExtractionResult


_HEADING_PREFIXES = [
    ("#### ", "H4"),
    ("### ", "H3"),
    ("## ", "H2"),
    ("# ", "H1"),
]


def _paragraph_category(text: str) -> str:
    for prefix, role in _HEADING_PREFIXES:
        if text.startswith(prefix):
            return role
    return "P"


def to_standard(
    result: ExtractionResult,
    page_sizes: dict[int, tuple[float, float]] | None = None,
) -> dict:
    def _ps(page_no: int) -> list[float]:
        if page_sizes and page_no in page_sizes:
            w, h = page_sizes[page_no]
            return [round(w, 2), round(h, 2)]
        return [612.0, 792.0]

    chunks: list[dict] = []

    for page_entry in result.text_by_page:
        page_no = page_entry.get("page", 1)
        markdown = (page_entry.get("markdown") or "").strip()
        paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            if len(para) < 3:
                continue
            chunks.append({
                "page": page_no,
                "paragraph": i + 1,
                "text": para,
                "font_size": 0.0,
                "bbox": None,
                "page_size": _ps(page_no),
                "category": _paragraph_category(para),
            })

    for fig in result.figures:
        page_no = fig.get("page", 1)
        desc = (fig.get("description") or "").strip()
        kind = (fig.get("kind") or "figure").strip()
        if desc:
            chunks.append({
                "page": page_no,
                "paragraph": 0,
                "text": f"[{kind.title()}] {desc}",
                "font_size": 0.0,
                "bbox": None,
                "page_size": _ps(page_no),
                "category": "Figure",
            })

    tables: list[dict] = []
    for t in result.tables:
        page_no = t.get("page_start", 1)
        headers = t.get("headers") or []
        rows = t.get("rows") or []
        all_rows = (
            [headers] if headers else []
        ) + [
            [str(c) if c is not None else "" for c in row]
            for row in rows
        ]
        tables.append({
            "page": page_no,
            "table_type": "claude_vision",
            "table": all_rows,
            "bbox": None,
            "page_size": _ps(page_no),
            "caption": t.get("caption"),
            "uncertain_cells": t.get("uncertain_cells") or [],
            "source_snippet": t.get("source_snippet") or "",
        })

    return {"chunks": chunks, "tables": tables, "images": []}
