"""Tests for distillation-set helpers (no heavy deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.data.distill_dataset import (
    read_labels,
    synthetic_distill_records,
    to_messages,
)


def test_to_messages_shape() -> None:
    msgs = to_messages("describe this", "a login screen")
    assert msgs == [
        {"role": "user", "content": "describe this"},
        {"role": "assistant", "content": "a login screen"},
    ]


def test_synthetic_distill_records() -> None:
    recs = synthetic_distill_records(6)
    assert len(recs) == 6
    assert all(set(r) == {"id", "question", "answer", "image"} for r in recs)
    assert all(r["image"] is None and r["answer"] for r in recs)


def test_read_labels_filters_invalid(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "1", "teacher_target": "good one here", "valid": True},
                {"id": "2", "teacher_target": "x", "valid": False},
                {"id": "3", "teacher_target": "another good", "valid": True},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels = read_labels(path)
    assert labels == {"1": "good one here", "3": "another good"}
    assert read_labels(path, valid_only=False).keys() == {"1", "2", "3"}


def test_read_labels_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_labels(tmp_path / "nope.jsonl")
