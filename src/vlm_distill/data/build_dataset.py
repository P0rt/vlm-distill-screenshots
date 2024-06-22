"""Normalize raw Screen2Words data into a unified {image, prompt, target}
dataset and persist it with train/val/test splits (SPEC Phase 2).

Heavy deps (``datasets``, ``PIL``) are imported lazily inside functions so the
module stays importable in the lint/type CI job without the data stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_distill.cli import build_parser
from vlm_distill.config import DataConfig, config_hash, load_data
from vlm_distill.data.normalize import (
    build_record,
    compute_stats,
    render_stats_markdown,
)
from vlm_distill.data.splits import assign_split

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

# Original columns we carry through (the screenshot) vs. drop (heavy unused fields).
_RAW_TEXT_COLUMNS = ["screenId", "captions", "category"]
_TEXT_COLUMNS = ["id", "prompt", "target", "references", "category", "source", "split"]

README_BEGIN = "<!-- DATASET_STATS:BEGIN -->"
README_END = "<!-- DATASET_STATS:END -->"


def _normalize_split(ds: Dataset, cfg: DataConfig, native_split: str) -> Dataset:
    """Map a raw split to the unified schema, keeping the image column."""

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        rec = build_record(example, prompt=cfg.prompt_template, max_references=cfg.max_references)
        if cfg.use_native_splits:
            rec["split"] = native_split
        else:
            rec["split"] = assign_split(rec["id"], cfg.split)
        return rec

    keep = [c for c in _RAW_TEXT_COLUMNS if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in {"image", *keep}]
    if drop:
        ds = ds.remove_columns(drop)
    return ds.map(_map, remove_columns=keep, desc=f"normalize[{native_split}]")


def _build_synthetic() -> DatasetDict:
    """Tiny in-memory dataset for --dry-run graph validation (no network)."""
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    from datasets import Image as HfImage
    from PIL import Image as PILImage

    def _img(color: tuple[int, int, int]) -> PILImage.Image:
        return PILImage.new("RGB", (64, 96), color)

    rows = {
        "screenId": [1001, 1002, 1003, 1004],
        "captions": [
            ["a login screen with email and password fields", "sign in page"],
            ["settings menu with toggles", "app settings", "preferences"],
            ["a list of news articles", "news feed"],
            ["map view with search bar", "navigation map"],
        ],
        "category": ["Communication", "Tools", "News", "Maps"],
        "image": [_img(c) for c in [(20, 30, 40), (60, 60, 60), (200, 200, 200), (10, 90, 30)]],
    }
    features = Features(
        {
            "screenId": Value("int64"),
            "captions": Sequence(Value("string")),
            "category": Value("string"),
            "image": HfImage(),
        }
    )
    full = Dataset.from_dict(rows, features=features)
    return DatasetDict(
        {"train": full.select([0, 1]), "val": full.select([2]), "test": full.select([3])}
    )


def _load_raw(cfg: DataConfig, *, limit: int | None, dry_run: bool) -> DatasetDict:
    if dry_run:
        return _build_synthetic()
    from datasets import DatasetDict, load_dataset

    # No `split=` -> returns a DatasetDict with all native splits (train/val/test).
    raw = load_dataset(cfg.dataset_id, cache_dir=cfg.raw_dir)
    if limit is not None:
        raw = DatasetDict({name: ds.select(range(min(limit, len(ds)))) for name, ds in raw.items()})
    return raw


def _write_stats_and_readme(stats: dict[str, Any], cfg_hash: str, *, dry_run: bool) -> None:
    from vlm_distill.config import REPO_ROOT

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {"config_hash": cfg_hash, "dry_run": dry_run, **stats}
    (results_dir / "dataset_stats.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not dry_run:
        _update_readme(stats)


def _update_readme(stats: dict[str, Any]) -> None:
    from vlm_distill.config import REPO_ROOT

    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        return
    block = f"{README_BEGIN}\n{render_stats_markdown(stats)}\n{README_END}"
    pre = text.split(README_BEGIN)[0]
    post = text.split(README_END)[1]
    readme.write_text(pre + block + post, encoding="utf-8")


def build_dataset(
    cfg: DataConfig, *, limit: int | None = None, dry_run: bool = False
) -> dict[str, Any]:
    """Build + persist the unified dataset; returns the computed stats."""
    from datasets import DatasetDict

    raw = _load_raw(cfg, limit=limit, dry_run=dry_run)
    unified = DatasetDict({name: _normalize_split(ds, cfg, name) for name, ds in raw.items()})

    out_dir = Path(cfg.processed_dir + ("_dryrun" if dry_run else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    unified.save_to_disk(str(out_dir))

    # Stats over text columns only (avoid decoding images).
    records: list[dict[str, Any]] = []
    for ds in unified.values():
        cols = [c for c in _TEXT_COLUMNS if c in ds.column_names]
        records.extend(ds.select_columns(cols).to_list())
    stats = compute_stats(records)

    cfg_hash = config_hash(cfg)
    _write_stats_and_readme(stats, cfg_hash, dry_run=dry_run)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-build-dataset", "Normalize + split into a unified dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per split.")
    ns = parser.parse_args(argv)
    cfg = load_data(ns.config)
    stats = build_dataset(cfg, limit=ns.limit, dry_run=bool(ns.dry_run))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
