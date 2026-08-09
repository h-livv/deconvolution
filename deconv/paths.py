"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path

# deconv/ → repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = REPO_ROOT / "images"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"
