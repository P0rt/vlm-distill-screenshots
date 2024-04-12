#!/usr/bin/env python3
"""Thin CLI wrapper over vlm_distill.benchmark (see SPEC §4).

Run via the synced environment, e.g. `uv run python scripts/benchmark.py` or the
installed console script (see README). The package must be installed (`uv sync`).
"""

from __future__ import annotations

from vlm_distill.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
