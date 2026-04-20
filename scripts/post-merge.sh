#!/bin/bash
set -e

# Lightweight deps — always required for the server to start.
pip install --no-cache-dir \
    pymupdf pdfplumber fastapi 'uvicorn[standard]' pydantic python-multipart

# Heavy ML deps for v3 (unstructured.io) + v4 (docling). These pull in
# torch + CUDA wheels (~2GB) and can exceed the workspace disk quota.
# The server lazy-imports them and returns HTTP 503 if missing, so a
# failure here is non-fatal — we log and continue so the merge succeeds.
if python -c "import unstructured, docling" 2>/dev/null; then
    echo "v3/v4 ML deps already present, skipping."
else
    echo "Attempting to install v3/v4 ML deps (best effort)…"
    if pip install --no-cache-dir 'unstructured[pdf]' docling; then
        echo "v3/v4 ML deps installed."
    else
        echo "WARN: could not install unstructured / docling (likely disk quota)."
        echo "      v1/v2 still work; v3/v4 will return HTTP 503 in the UI."
        echo "      Re-run 'pip install \"unstructured[pdf]\" docling' manually if needed."
    fi
fi
