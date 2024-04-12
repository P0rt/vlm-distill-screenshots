#!/usr/bin/env python3
"""Thin CLI wrapper over vlm_distill.eval (see SPEC §4).

Run via the synced environment, e.g. `uv run python scripts/eval.py` or the
installed console script (see README). The package must be installed (`uv sync`).
"""

from __future__ import annotations

from vlm_distill.eval import main

if __name__ == "__main__":
    raise SystemExit(main())
