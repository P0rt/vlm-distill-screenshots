#!/usr/bin/env python3
"""Thin CLI wrapper over vlm_distill.export (see SPEC §4).

Run via the synced environment, e.g. `uv run python scripts/export.py` or the
installed console script (see README). The package must be installed (`uv sync`).
"""

from __future__ import annotations

from vlm_distill.export import main

if __name__ == "__main__":
    raise SystemExit(main())
