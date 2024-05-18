"""Deterministic train/val/test assignment.

Used when a corpus has no native splits (e.g. combined / teacher-synthesized
data in Phase 7). The assignment is a pure function of ``(example_id, seed)``,
so it is stable across runs and independent of iteration order — no global
shuffle, no leakage between splits.

Pure stdlib: importable and testable in CI without any heavy deps.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from vlm_distill.config import SplitConfig

SPLITS: tuple[str, str, str] = ("train", "val", "test")


def _unit_interval(example_id: str, seed: int) -> float:
    """Map an id to a stable, uniformly distributed value in ``[0, 1)``."""
    digest = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(example_id: str, cfg: SplitConfig) -> str:
    """Assign one example to ``train`` / ``val`` / ``test`` deterministically."""
    total = cfg.train + cfg.val + cfg.test
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    # Normalize so ratios need not sum to exactly 1.0.
    train_t = cfg.train / total
    val_t = train_t + cfg.val / total
    bucket = _unit_interval(example_id, cfg.seed)
    if bucket < train_t:
        return "train"
    if bucket < val_t:
        return "val"
    return "test"


def assign_splits(example_ids: Iterable[str], cfg: SplitConfig) -> dict[str, str]:
    """Assign a collection of ids, returning ``{id: split}``."""
    return {str(eid): assign_split(str(eid), cfg) for eid in example_ids}


def split_counts(example_ids: Iterable[str], cfg: SplitConfig) -> dict[str, int]:
    """Count examples per split (handy for stats / sanity checks)."""
    counts: Counter[str] = Counter(assign_split(str(eid), cfg) for eid in example_ids)
    return {split: counts.get(split, 0) for split in SPLITS}
