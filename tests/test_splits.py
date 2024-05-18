"""Tests for the deterministic hash-based split (SPEC Phase 2 acceptance)."""

from __future__ import annotations

from vlm_distill.config import SplitConfig
from vlm_distill.data.splits import SPLITS, assign_split, assign_splits, split_counts


def test_assignment_is_deterministic() -> None:
    cfg = SplitConfig()
    ids = [str(i) for i in range(1000)]
    first = assign_splits(ids, cfg)
    second = assign_splits(ids, cfg)
    assert first == second


def test_assignment_is_order_independent() -> None:
    cfg = SplitConfig()
    ids = [str(i) for i in range(500)]
    forward = assign_splits(ids, cfg)
    backward = assign_splits(list(reversed(ids)), cfg)
    assert forward == backward


def test_every_assignment_is_a_valid_split() -> None:
    cfg = SplitConfig()
    assigned = assign_splits([str(i) for i in range(200)], cfg)
    assert set(assigned.values()) <= set(SPLITS)


def test_ratios_are_approximately_honored() -> None:
    cfg = SplitConfig(train=0.8, val=0.1, test=0.1, seed=7)
    counts = split_counts([str(i) for i in range(10_000)], cfg)
    total = sum(counts.values())
    assert total == 10_000
    assert abs(counts["train"] / total - 0.8) < 0.03
    assert abs(counts["val"] / total - 0.1) < 0.03
    assert abs(counts["test"] / total - 0.1) < 0.03


def test_seed_changes_assignment() -> None:
    ids = [str(i) for i in range(2000)]
    a = assign_splits(ids, SplitConfig(seed=1))
    b = assign_splits(ids, SplitConfig(seed=2))
    differing = sum(1 for k in ids if a[k] != b[k])
    assert differing > 0


def test_unnormalized_ratios_are_handled() -> None:
    # Ratios need not sum to 1.0; they are normalized internally.
    cfg = SplitConfig(train=8, val=1, test=1, seed=0)
    assigned = assign_split("some-id", cfg)
    assert assigned in SPLITS
