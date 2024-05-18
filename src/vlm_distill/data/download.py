"""Prefetch the Screen2Words dataset into the local cache and report split
sizes (SPEC Phase 2).

This is a thin convenience step: ``build_dataset`` will also trigger the
download on first use. Heavy deps are imported lazily.
"""

from __future__ import annotations

import json

from vlm_distill.cli import build_parser
from vlm_distill.config import DataConfig, load_data


def download(cfg: DataConfig, *, limit: int | None = None, dry_run: bool = False) -> dict[str, int]:
    """Fetch the dataset into ``cfg.raw_dir``; return ``{split: num_rows}``."""
    if dry_run:
        # Validate wiring without network: report the documented native sizes.
        return {"train": 15700, "val": 2360, "test": 4310}

    from datasets import load_dataset

    raw = load_dataset(cfg.dataset_id, cache_dir=cfg.raw_dir)
    sizes = {name: len(ds) for name, ds in raw.items()}
    if limit is not None:
        sizes = {name: min(limit, n) for name, n in sizes.items()}
    return sizes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-download", "Prefetch Screen2Words into the local cache.")
    parser.add_argument("--limit", type=int, default=None, help="Cap reported rows per split.")
    ns = parser.parse_args(argv)
    cfg = load_data(ns.config)
    sizes = download(cfg, limit=ns.limit, dry_run=bool(ns.dry_run))
    print(json.dumps({"dataset_id": cfg.dataset_id, "splits": sizes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
