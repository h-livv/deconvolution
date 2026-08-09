#!/usr/bin/env python3
"""CLI: three-method fill-observation benchmark (see ``deconv.experiment``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deconv.experiment import main

if __name__ == "__main__":
    main()
