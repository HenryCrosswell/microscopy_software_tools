# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python toolkit for automated analysis of microscopy/neuroimaging data, including image preprocessing, cell segmentation, image registration/stitching, and interactive visualization via napari.

## Development Commands

```bash
# Install dependencies (uses uv)
uv sync --dev

# Lint and format
ruff check .
ruff format .

# Run all tests
pytest

# Run a single test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov

# Build documentation
mkdocs serve   # local preview
mkdocs build   # static build
```

> **Note:** `pyproject.toml` requires Python >=3.12, but `.github/workflows/python-app.yml` currently pins Python 3.10 — CI will fail until reconciled.

## Architecture

```
src/microscopy_tools/
├── preprocessing.py        # Image loading (NIfTI + standard formats) and normalization
├── stitch_and_register.py  # Multi-image alignment via SimpleITK and OpenCV Stitcher
└── main.py                 # napari GUI entry point — loads BIDS-formatted brain images
                            # and displays original, blurred, and normalized layers
```

**Data flow:** `main.py` calls `preprocessing.py` to load and normalize NIfTI images, then `stitch_and_register.py` to align multiple images, and finally renders everything in a napari viewer.

**Key dependencies:** nibabel (NIfTI I/O), SimpleITK (image registration), OpenCV (stitching), napari (interactive visualization), numpy/matplotlib (core numerics).

**Test data:** `tests/test_data/` is gitignored (image files excluded). Generate test fixtures locally before running tests.