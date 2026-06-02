"""Tests for the training orchestration (dry-run path, no heavy deps)."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_distill.config import DataConfig, DistillConfig, StudentConfig, TeacherConfig
from vlm_distill.train import _iters_for, run_training


def test_iters_from_epochs() -> None:
    cfg = DistillConfig(batch_size=4, num_epochs=3, max_steps=None)
    assert _iters_for(cfg, dataset_len=40) == (40 // 4) * 3


def test_iters_from_max_steps_overrides() -> None:
    cfg = DistillConfig(max_steps=7)
    assert _iters_for(cfg, dataset_len=1000) == 7


def test_iters_floor_on_tiny_dataset() -> None:
    cfg = DistillConfig(batch_size=8, num_epochs=2, max_steps=None)
    assert _iters_for(cfg, dataset_len=3) == 2  # max(1, 0) * 2


def test_run_training_dry_run(tmp_path: Path) -> None:
    summary = run_training(
        StudentConfig(),
        DistillConfig(batch_size=2, num_epochs=1, max_steps=None),
        DataConfig(),
        TeacherConfig(),
        limit=4,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["backend"] == "dry-run"
    assert summary["num_examples"] == 4
    assert summary["iters"] == (4 // 2) * 1

    from vlm_distill.config import REPO_ROOT

    manifest = REPO_ROOT / "results" / "checkpoints" / "dryrun" / "manifest.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text())["dry_run"] is True
