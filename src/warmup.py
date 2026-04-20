"""
Pre-download / pre-load the heavy ML models used by extract_v3 (unstructured.io)
and extract_v4 (docling) so the first user-facing extraction doesn't pay a
30s–several-minute model-download cost.

Can be used two ways:

  1. As a standalone script (e.g. in a deploy hook):
        python -m src.warmup
        python -m src.warmup --only v3
        python -m src.warmup --only v4

  2. From the FastAPI server's startup hook (see server.py); each warmup runs
     in a background thread so the server starts serving immediately while
     models are fetched in the background.

All warmup steps are best-effort: missing optional dependencies and download
errors are logged but never raised, so the server keeps running normally.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

log = logging.getLogger("warmup")


def warmup_v3() -> bool:
    """Pre-load unstructured.io models (YOLOX layout detector, etc.).

    Returns True on success, False if the optional deps aren't installed or a
    download failed. Never raises.
    """
    t0 = time.time()
    try:
        log.info("v3 warmup: importing unstructured.partition.pdf …")
        from unstructured.partition.pdf import partition_pdf  # noqa: F401
    except Exception as e:
        log.warning("v3 warmup skipped — unstructured not importable: %s", e)
        return False

    # The hi_res strategy uses a YOLOX layout model that's downloaded on first
    # use. Pull it explicitly so the first real extraction is instant.
    yolox_ok = False
    try:
        from unstructured_inference.models.base import get_model

        log.info("v3 warmup: fetching YOLOX layout model …")
        get_model("yolox")
        yolox_ok = True
    except Exception as e:
        log.warning("v3 warmup: could not pre-fetch yolox model: %s", e)

    # Tesseract OCR data is a system package; nothing to download here, but we
    # touch pytesseract so its lazy import work is amortized.
    try:
        import pytesseract  # noqa: F401
    except Exception:
        pass

    log.info("v3 warmup %s in %.1fs", "complete" if yolox_ok else "partial", time.time() - t0)
    return yolox_ok


def warmup_v4() -> bool:
    """Pre-download docling's IBM layout/table models and OCR weights.

    Returns True on success, False if docling isn't installed. Never raises.
    """
    t0 = time.time()
    try:
        log.info("v4 warmup: importing docling …")
        from docling.document_converter import DocumentConverter
    except Exception as e:
        log.warning("v4 warmup skipped — docling not importable: %s", e)
        return False

    # Preferred path: docling exposes an explicit model downloader.
    downloaded = False
    try:
        from docling.utils.model_downloader import download_models

        log.info("v4 warmup: downloading docling models …")
        download_models()
        downloaded = True
    except Exception as e:
        log.warning("v4 warmup: download_models() unavailable or failed: %s", e)

    # Fallback (or follow-up): instantiating the converter forces lazy loaders
    # for layout / table-structure / OCR pipelines to populate caches.
    if not downloaded:
        try:
            log.info("v4 warmup: instantiating DocumentConverter to trigger lazy downloads …")
            DocumentConverter()
        except Exception as e:
            log.warning("v4 warmup: DocumentConverter() failed: %s", e)
            return False

    log.info("v4 warmup done in %.1fs", time.time() - t0)
    return True


def warmup_all_in_background() -> list[threading.Thread]:
    """Run v3 and v4 warmups in daemon background threads. Returns the threads."""
    threads: list[threading.Thread] = []
    for name, fn in (("warmup-v3", warmup_v3), ("warmup-v4", warmup_v4)):
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        threads.append(t)
    return threads


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download v3/v4 extraction models.")
    parser.add_argument(
        "--only",
        choices=("v3", "v4"),
        default=None,
        help="Only warm up one extractor (default: both).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ok = True
    if args.only in (None, "v3"):
        ok = warmup_v3() and ok
    if args.only in (None, "v4"):
        ok = warmup_v4() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
