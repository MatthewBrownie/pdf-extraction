#!/bin/bash
set -e

# Install Python deps from requirements.txt. Heavy ML packages
# (unstructured[pdf], docling) may already be present from a prior run;
# pip is idempotent and will skip what's already installed.
pip install -r requirements.txt
