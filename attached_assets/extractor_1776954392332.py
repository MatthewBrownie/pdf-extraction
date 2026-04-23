"""
PDF extraction using Claude via OpenRouter.

API format: OpenAI-compatible (openrouter.ai/api/v1).
PDF input: file content block with base64 data URI.
Structured output: OpenAI-style function calling (forced tool use).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import fitz  # PyMuPDF — chunking and page counting only


MODEL_BULK = "anthropic/claude-haiku-4.5"
MODEL_PRECISE = "anthropic/claude-sonnet-4.6"

# Haiku output is capped at ~8192 tokens by OpenRouter; 2 pages ≈ 4k tokens, safe margin
# Sonnet supports 16k+ output tokens; 8 pages ≈ 12k tokens for dense technical content
MAX_PAGES_PER_CHUNK = {
    MODEL_BULK:    2,
    MODEL_PRECISE: 8,
}
MAX_BYTES_PER_CHUNK = 28 * 1024 * 1024  # 28 MB, safely below 32 MB limit

# Approximate per-token cost in USD (input / output) for cost estimation
_COST = {
    MODEL_BULK:    (1.0 / 1_000_000, 5.0 / 1_000_000),
    MODEL_PRECISE: (3.0 / 1_000_000, 15.0 / 1_000_000),
}

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
                            "page": {
                                "type": "integer",
                                "description": "1-indexed page number in the full document.",
                            },
                            "markdown": {
                                "type": "string",
                                "description": "Full page content in reading-order Markdown.",
                            },
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
                            "headers": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
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
                            "source_snippet": {
                                "type": "string",
                                "description": "Short verbatim text from the table for provenance validation.",
                            },
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
                            "kind": {
                                "type": "string",
                                "description": "chart, diagram, photo, illustration, screenshot, etc.",
                            },
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
        }


def _chunk_pdf(path: Path, max_pages: int) -> list[tuple[bytes, int, int]]:
    """
    Split PDF into chunks that fit within page and byte limits.
    Returns list of (pdf_bytes, start_page_1indexed, num_pages).
    """
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
        # Output was truncated mid-JSON — model hit its token limit for this chunk.
        out_tokens = response.usage.completion_tokens if response.usage else "?"
        raise RuntimeError(
            f"Claude response truncated at char {exc.pos} "
            f"({out_tokens} output tokens). "
            f"Try a smaller chunk size (MAX_PAGES_PER_CHUNK) or switch to Sonnet."
        ) from exc

    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return {"data": data, "usage": usage}


def extract_pdf(
    pdf_path: str | Path,
    *,
    model: str = MODEL_BULK,
    progress_cb=None,
) -> ExtractionResult:
    path = Path(pdf_path)

    from openai import OpenAI
    api_key = os.environ["OPENAI_API_KEY"]
    print(f"[extractor] key loaded: {api_key[:8]}... ({len(api_key)} chars)", flush=True)
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:3001",
            "X-Title": "Vision PDF Extractor",
        },
    )

    def _emit(**kw):
        if progress_cb:
            try:
                progress_cb(**kw)
            except Exception:
                pass

    max_pages = MAX_PAGES_PER_CHUNK.get(model, 2)
    _emit(phase="chunking", message=f"Splitting PDF into chunks ({max_pages} pages each)…")
    chunks = _chunk_pdf(path, max_pages)

    result = ExtractionResult(model=model)

    for i, (chunk_bytes, page_start, num_pages) in enumerate(chunks):
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

    result.pages = sum(n for _, _, n in chunks)
    return result
