"""
Claude Vision PDF extraction via OpenRouter.

Adapted from the attached prototype (`extractor.py` + `transform.py`) into the
project's flat `src/` layout — no relative imports, mirrors how `extract_v2`
etc. are loaded by `server.py`.

Stack:
  - openai SDK pointed at https://openrouter.ai/api/v1
  - PyMuPDF for PDF chunking and page counting
  - Forced tool-call structured output

Public surface:
  - MODEL_BULK / MODEL_PRECISE        (model IDs)
  - MODELS                            ({"haiku": ..., "sonnet": ...})
  - ExtractionResult dataclass
  - extract_pdf(pdf_path, model=..., progress_cb=...) -> ExtractionResult
  - to_standard(result) -> {"chunks": [...], "tables": [...], "images": []}
  - get_api_key() -> str | None
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF — chunking and page counting only

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except Exception:
    pass


MODEL_BULK = "anthropic/claude-haiku-4.5"
MODEL_PRECISE = "anthropic/claude-sonnet-4.6"

MODELS = {
    "haiku": MODEL_BULK,
    "sonnet": MODEL_PRECISE,
}

MAX_PAGES_PER_CHUNK = {
    MODEL_BULK:    2,
    MODEL_PRECISE: 8,
}
MAX_BYTES_PER_CHUNK = 28 * 1024 * 1024

_COST = {
    MODEL_BULK:    (1.0 / 1_000_000, 5.0 / 1_000_000),
    MODEL_PRECISE: (3.0 / 1_000_000, 15.0 / 1_000_000),
}

EST_INPUT_TOKENS_PER_PAGE = (1800, 2600)
EST_OUTPUT_TOKENS_PER_PAGE = (400, 1200)
EST_PROMPT_OVERHEAD_PER_CHUNK = 250


def _calibrated_per_page_range(
    samples: list[dict],
    static_lo: int,
    static_hi: int,
    field: str,
) -> tuple[float, float]:
    """Compute (low, high) per-page tokens from historical samples.

    Each sample is a dict with at least `pages` and the requested `field`
    (e.g. `input_tokens`). Samples with non-positive pages or usage are
    skipped. With one usable sample, returns ±15% around its per-page rate.
    With two or more, returns (min, max) per-page rates across samples.
    Falls back to (static_lo, static_hi) when nothing usable is provided.
    """
    rates: list[float] = []
    for s in samples:
        try:
            pg = int(s.get("pages") or 0)
            tok = int(s.get(field) or 0)
        except (TypeError, ValueError):
            continue
        if pg > 0 and tok > 0:
            rates.append(tok / pg)
    if not rates:
        return float(static_lo), float(static_hi)
    if len(rates) == 1:
        r = rates[0]
        return r * 0.85, r * 1.15
    return min(rates), max(rates)


SIMILAR_PAGES_LO_RATIO = 0.5
SIMILAR_PAGES_HI_RATIO = 2.0


def _filter_samples_by_pages(
    samples: list[dict], target_pages: int
) -> list[dict]:
    """Restrict samples to runs whose page count is within 0.5x–2x target.

    Returns the close-match subset when non-empty, otherwise the original
    list so callers transparently fall back to the broader pool.
    """
    if target_pages <= 0 or not samples:
        return samples
    lo = target_pages * SIMILAR_PAGES_LO_RATIO
    hi = target_pages * SIMILAR_PAGES_HI_RATIO
    close = [s for s in samples if lo <= (s.get("pages") or 0) <= hi]
    return close if close else samples


def estimate_cost(pages: int, history: dict | None = None) -> dict:
    """Approximate USD cost range for extracting a PDF of `pages` pages.

    Returns a dict keyed by short model name (`haiku`, `sonnet`) with
    `low`, `high`, `chunks`, `max_pages_per_chunk`, `model`, `source`
    (`"calibrated"` or `"heuristic"`), and `samples` (count of past runs
    used) fields. Estimates remain approximate.

    When `history` is provided, it should be a dict mapping model id
    (e.g. `"anthropic/claude-haiku-4.5"`) to a list of past-run summaries
    `{"pages": int, "input_tokens": int, "output_tokens": int}`. Per-page
    token rates are derived per model from those samples and used in place
    of the static heuristic. To keep the displayed range tracking the
    document at hand, calibration prefers historical runs whose page
    count is within 0.5x–2x of `pages`; when no close matches exist it
    falls back to all usable samples for that model, and finally to the
    static heuristic when there are none.
    """
    pages = max(0, int(pages))
    history = history or {}
    out: dict = {}
    for short, model in MODELS.items():
        max_pp = MAX_PAGES_PER_CHUNK.get(model, 2)
        chunks = (pages + max_pp - 1) // max_pp if pages else 0
        all_samples = [
            s for s in (history.get(model) or [])
            if (s.get("pages") or 0) > 0
            and (s.get("input_tokens") or 0) > 0
            and (s.get("output_tokens") or 0) > 0
        ]
        samples = _filter_samples_by_pages(all_samples, pages)
        in_per_lo, in_per_hi = _calibrated_per_page_range(
            samples, EST_INPUT_TOKENS_PER_PAGE[0], EST_INPUT_TOKENS_PER_PAGE[1],
            "input_tokens",
        )
        out_per_lo, out_per_hi = _calibrated_per_page_range(
            samples, EST_OUTPUT_TOKENS_PER_PAGE[0], EST_OUTPUT_TOKENS_PER_PAGE[1],
            "output_tokens",
        )
        calibrated = bool(samples)
        # Historical usage already includes the per-chunk prompt overhead;
        # only add it when falling back to the static heuristic.
        overhead = 0 if calibrated else chunks * EST_PROMPT_OVERHEAD_PER_CHUNK
        in_lo = pages * in_per_lo + overhead
        in_hi = pages * in_per_hi + overhead
        out_lo = pages * out_per_lo
        out_hi = pages * out_per_hi
        in_rate, out_rate = _COST[model]
        low = in_lo * in_rate + out_lo * out_rate
        high = in_hi * in_rate + out_hi * out_rate
        out[short] = {
            "model": model,
            "max_pages_per_chunk": max_pp,
            "chunks": chunks,
            "low": round(low, 4),
            "high": round(high, 4),
            "source": "calibrated" if calibrated else "heuristic",
            "samples": len(samples),
        }
    return out

SYSTEM_PROMPT = (
    "You extract structured content from PDFs for a downstream pipeline. "
    "Be faithful to the source. Never invent values. "
    "Prefer the text layer over visual interpretation. "
    "For tables: preserve exact numeric formatting (keep thousand separators, "
    "currency symbols, units). For uncertainty: emit null and record in uncertain_cells. "
    "Do NOT guess."
)

_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_extraction",
        "description": "Return the complete structured extraction for this PDF or PDF chunk.",
        "parameters": {
            "type": "object",
            "required": ["text_by_page", "tables", "figures"],
            "properties": {
                "text_by_page": {
                    "type": "array",
                    "description": "One entry per page.",
                    "items": {
                        "type": "object",
                        "required": ["page", "markdown"],
                        "properties": {
                            "page": {"type": "integer"},
                            "markdown": {"type": "string"},
                        },
                    },
                },
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["page_start", "headers", "rows"],
                        "properties": {
                            "page_start": {"type": "integer"},
                            "page_end": {"type": "integer"},
                            "caption": {"type": ["string", "null"]},
                            "headers": {"type": "array", "items": {"type": "string"}},
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": ["string", "null"]},
                                },
                            },
                            "uncertain_cells": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "row": {"type": "integer"},
                                        "col": {"type": "integer"},
                                        "reason": {"type": "string"},
                                    },
                                },
                            },
                            "source_snippet": {"type": "string"},
                        },
                    },
                },
                "figures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["page", "description"],
                        "properties": {
                            "page": {"type": "integer"},
                            "description": {"type": "string"},
                            "kind": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


@dataclass
class ExtractionResult:
    text_by_page: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    pages: int = 0
    model: str = ""
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    partial: bool = False
    chunks_done: int = 0
    chunks_total: int = 0
    pages_done: int = 0

    def estimated_cost_usd(self) -> float:
        in_cost, out_cost = _COST.get(self.model, (0.0, 0.0))
        return (
            self.usage["input_tokens"] * in_cost
            + self.usage["output_tokens"] * out_cost
        )

    def to_dict(self) -> dict:
        return {
            "text_by_page": self.text_by_page,
            "tables": self.tables,
            "figures": self.figures,
            "pages": self.pages,
            "model": self.model,
            "usage": self.usage,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "partial": self.partial,
            "chunks_done": self.chunks_done,
            "chunks_total": self.chunks_total,
            "pages_done": self.pages_done,
        }


def get_api_key() -> str | None:
    """Return the OpenRouter API key.

    Standardises on `OPENROUTER_API_KEY` but falls back to `OPENAI_API_KEY`
    so existing .env setups keep working.
    """
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _chunk_pdf(path: Path, max_pages: int) -> list[tuple[bytes, int, int]]:
    src = fitz.open(str(path))
    chunks: list[tuple[bytes, int, int]] = []
    start = 0
    while start < len(src):
        end = min(start + max_pages, len(src))
        sub = fitz.open()
        sub.insert_pdf(src, from_page=start, to_page=end - 1)
        data = sub.tobytes()
        if len(data) > MAX_BYTES_PER_CHUNK and end - start > 1:
            end = start + max(1, (end - start) // 2)
            sub = fitz.open()
            sub.insert_pdf(src, from_page=start, to_page=end - 1)
            data = sub.tobytes()
        chunks.append((data, start + 1, end - start))
        start = end
    return chunks


def _call_claude(
    pdf_bytes: bytes,
    filename: str,
    page_start: int,
    page_end: int,
    model: str,
    client,
) -> dict:
    b64 = base64.standard_b64encode(pdf_bytes).decode()

    user_text = (
        f"This document contains pages {page_start} to {page_end} of '{filename}'. "
        f"Number all pages in your output starting from page {page_start}. "
        "Extract all content: full reading-order Markdown per page, every table "
        "(merging continuations), and every figure/chart. "
        "If a cell value is unclear, emit null and add it to uncertain_cells."
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=16000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_extraction"}},
    )

    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError("Claude returned no tool call — cannot parse extraction.")

    raw_args = message.tool_calls[0].function.arguments
    try:
        data = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        out_tokens = response.usage.completion_tokens if response.usage else "?"
        raise RuntimeError(
            f"Claude response truncated at char {exc.pos} "
            f"({out_tokens} output tokens). "
            f"Try a smaller chunk size or switch to Sonnet."
        ) from exc

    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return {"data": data, "usage": usage}


class CancelledExtraction(RuntimeError):
    """Raised when a vision extraction is cancelled mid-run.

    When raised from within :func:`extract_pdf`, the partial
    :class:`ExtractionResult` collected so far is attached as ``result``.
    """

    def __init__(self, message: str = "Extraction cancelled.", result: "ExtractionResult | None" = None):
        super().__init__(message)
        self.result = result


def _coerce_resume(
    resume_from: "ExtractionResult | dict | None", model: str
) -> "ExtractionResult | None":
    """Build an ExtractionResult seeded from prior partial output.

    Accepts either an existing ExtractionResult or its persisted dict form.
    Returns None if `resume_from` is empty / has no completed chunks. Raises
    ValueError if the prior run used a different model (chunk boundaries
    depend on `MAX_PAGES_PER_CHUNK[model]` so we can't safely splice).
    """
    if resume_from is None:
        return None
    if isinstance(resume_from, ExtractionResult):
        prior = resume_from
    elif isinstance(resume_from, dict):
        usage = resume_from.get("usage") or {}
        prior = ExtractionResult(
            text_by_page=list(resume_from.get("text_by_page") or []),
            tables=list(resume_from.get("tables") or []),
            figures=list(resume_from.get("figures") or []),
            pages=int(resume_from.get("pages") or 0),
            model=str(resume_from.get("model") or ""),
            usage={
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            },
            partial=bool(resume_from.get("partial", False)),
            chunks_done=int(resume_from.get("chunks_done") or 0),
            chunks_total=int(resume_from.get("chunks_total") or 0),
            pages_done=int(resume_from.get("pages_done") or 0),
        )
    else:
        raise TypeError("resume_from must be ExtractionResult or dict")
    if prior.chunks_done <= 0:
        return None
    if prior.model and prior.model != model:
        raise ValueError(
            f"Cannot resume: prior run used model {prior.model!r}, "
            f"current request is {model!r}."
        )
    prior.model = model
    return prior


def extract_pdf(
    pdf_path: str | Path,
    *,
    model: str = MODEL_BULK,
    progress_cb=None,
    cancel_check=None,
    resume_from: "ExtractionResult | dict | None" = None,
) -> ExtractionResult:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key configured. "
            "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your environment."
        )

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://replit.com",
            "X-Title": "Vision PDF Extractor",
        },
    )

    def _emit(**kw):
        if progress_cb:
            try:
                progress_cb(**kw)
            except CancelledExtraction:
                raise
            except Exception:
                pass

    def _check_cancel():
        if cancel_check is not None:
            try:
                if cancel_check():
                    raise CancelledExtraction("Extraction cancelled.")
            except CancelledExtraction:
                raise
            except Exception:
                pass

    max_pages = MAX_PAGES_PER_CHUNK.get(model, 2)
    prior = _coerce_resume(resume_from, model)
    _emit(phase="chunking", message=f"Splitting PDF into chunks ({max_pages} pages each)…")
    _check_cancel()
    chunks = _chunk_pdf(path, max_pages)

    if prior is not None:
        result = prior
        result.chunks_total = len(chunks)
        # If the prior run reported more chunks done than we now see (e.g.
        # the PDF on disk changed), refuse to splice — safer to start fresh.
        if result.chunks_done > len(chunks):
            raise ValueError(
                f"Cannot resume: prior run completed {result.chunks_done} "
                f"chunks but PDF now produces only {len(chunks)}."
            )
        if result.chunks_done == len(chunks):
            # Everything was already done — nothing to do but mark complete.
            result.partial = False
            result.pages = sum(n for _, _, n in chunks)
            return result
        skip = result.chunks_done
        _emit(
            phase="resuming",
            message=f"Resuming from chunk {skip + 1}/{len(chunks)} "
                    f"(reusing {skip} chunk(s) from cache)…",
            page=skip,
            total=len(chunks),
        )
    else:
        result = ExtractionResult(model=model)
        result.chunks_total = len(chunks)
        skip = 0

    try:
        for i, (chunk_bytes, page_start, num_pages) in enumerate(chunks):
            if i < skip:
                continue
            _check_cancel()
            page_end = page_start + num_pages - 1
            _emit(
                phase="extracting",
                message=f"Extracting pages {page_start}–{page_end} (chunk {i + 1}/{len(chunks)})…",
                page=i + 1,
                total=len(chunks),
            )
            chunk_result = _call_claude(
                chunk_bytes, path.name, page_start, page_end, model, client
            )
            data = chunk_result["data"]
            result.text_by_page.extend(data.get("text_by_page", []))
            result.tables.extend(data.get("tables", []))
            result.figures.extend(data.get("figures", []))
            result.usage["input_tokens"] += chunk_result["usage"]["input_tokens"]
            result.usage["output_tokens"] += chunk_result["usage"]["output_tokens"]
            result.chunks_done = i + 1
            result.pages_done += num_pages
    except CancelledExtraction as exc:
        result.partial = True
        result.pages = result.pages_done
        exc.result = result
        raise

    result.pages = sum(n for _, _, n in chunks)
    result.partial = False
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Transform: native Claude shape -> standard {chunks, tables, images}
# ──────────────────────────────────────────────────────────────────────────────

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
