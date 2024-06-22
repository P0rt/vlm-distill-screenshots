"""Tests for teacher labeling: dry-run backend, resume/idempotency (Phase 3).

All run without heavy deps (DryRunTeacher + synthetic, image-free examples).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.config import DataConfig, TeacherConfig
from vlm_distill.data.teacher_label import (
    clean_and_validate_target,
    load_done_ids,
    output_dir_for,
    run_labeling,
)
from vlm_distill.models.teacher import DryRunTeacher, TeacherModel, make_teacher


def test_dry_run_teacher_produces_text() -> None:
    teacher = DryRunTeacher()
    out = teacher.describe(None, "describe this")
    assert isinstance(out, str)
    assert out.strip() != ""


def test_make_teacher_dry_run_is_protocol() -> None:
    teacher = make_teacher(TeacherConfig(), dry_run=True)
    assert isinstance(teacher, TeacherModel)
    assert isinstance(teacher, DryRunTeacher)


def test_make_teacher_unknown_backend_raises() -> None:
    cfg = TeacherConfig.model_validate({"backend": "mlx"})
    cfg = cfg.model_copy(update={"backend": "bogus"})  # bypass Literal at runtime
    with pytest.raises(ValueError, match="Unknown teacher backend"):
        make_teacher(cfg, dry_run=False)


def test_run_labeling_dry_run_writes_records(tmp_path: Path) -> None:
    summary = run_labeling(
        DataConfig(), TeacherConfig(), output_dir=tmp_path, limit=5, dry_run=True
    )
    assert summary["num_new_this_run"] == 5
    assert summary["num_labeled_total"] == 5
    assert summary["dry_run"] is True

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    rec = json.loads(lines[0])
    assert rec["teacher_target"].strip() != ""
    assert rec["model"] == "dry-run"
    assert (tmp_path / "manifest.json").exists()


def test_run_labeling_is_idempotent_and_resumable(tmp_path: Path) -> None:
    run_labeling(DataConfig(), TeacherConfig(), output_dir=tmp_path, limit=5, dry_run=True)
    # Second run over the same target set should label nothing new.
    second = run_labeling(DataConfig(), TeacherConfig(), output_dir=tmp_path, limit=5, dry_run=True)
    assert second["num_new_this_run"] == 0
    assert second["num_skipped"] == 5
    assert second["num_labeled_total"] == 5

    # Extending the limit only labels the delta.
    third = run_labeling(DataConfig(), TeacherConfig(), output_dir=tmp_path, limit=8, dry_run=True)
    assert third["num_new_this_run"] == 3
    assert third["num_labeled_total"] == 8


def test_load_done_ids(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    assert load_done_ids(path) == set()
    path.write_text('{"id": "a"}\n{"id": "b"}\ngarbage\n{"no_id": 1}\n', encoding="utf-8")
    assert load_done_ids(path) == {"a", "b"}


def test_output_dir_for() -> None:
    assert output_dir_for("abc123", dry_run=True).name == "dryrun"
    assert output_dir_for("abc123", dry_run=False).name == "abc123"


def test_clean_and_validate_target() -> None:
    cleaned, ok = clean_and_validate_target("  A login   screen\nwith fields. \t")
    assert cleaned == "A login screen with fields."
    assert ok is True

    empty, ok_empty = clean_and_validate_target("   \n  ")
    assert empty == ""
    assert ok_empty is False

    _short, ok_short = clean_and_validate_target("nope")
    assert ok_short is False  # below min_words
