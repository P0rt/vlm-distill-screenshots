"""Tests for record normalization and dataset statistics (no heavy deps)."""

from __future__ import annotations

from vlm_distill.data.normalize import (
    build_record,
    compute_stats,
    render_stats_markdown,
)

PROMPT = "Describe this UI screenshot."


def test_build_record_basic() -> None:
    raw = {
        "screenId": 42,
        "captions": ["  a login screen  ", "sign in page", ""],
        "category": "Communication",
    }
    rec = build_record(raw, prompt=PROMPT, max_references=5)
    assert rec["id"] == "42"
    assert rec["prompt"] == PROMPT
    assert rec["target"] == "a login screen"  # first non-empty, stripped
    assert rec["references"] == ["a login screen", "sign in page"]  # empty dropped
    assert rec["category"] == "Communication"
    assert rec["source"] == "screen2words"


def test_build_record_respects_max_references() -> None:
    raw = {"screenId": 1, "captions": [f"c{i}" for i in range(10)], "category": "X"}
    rec = build_record(raw, prompt=PROMPT, max_references=3)
    assert rec["references"] == ["c0", "c1", "c2"]


def test_build_record_handles_missing_captions() -> None:
    rec = build_record({"screenId": 7}, prompt=PROMPT)
    assert rec["target"] == ""
    assert rec["references"] == []


def test_build_record_ignores_non_string_captions() -> None:
    raw = {"screenId": 9, "captions": [None, 123, "valid"], "category": "X"}
    rec = build_record(raw, prompt=PROMPT)
    assert rec["references"] == ["valid"]


def test_compute_stats() -> None:
    records = [
        {"target": "a b c", "references": ["a b c", "x"], "category": "News", "split": "train"},
        {"target": "one two", "references": ["one two"], "category": "Maps", "split": "train"},
        {"target": "single", "references": ["single"], "category": "News", "split": "test"},
    ]
    stats = compute_stats(records)
    assert stats["num_examples"] == 3
    assert stats["splits"] == {"test": 1, "train": 2}
    assert stats["num_categories"] == 2
    assert stats["target_words"]["max"] == 3
    assert stats["target_words"]["min"] == 1


def test_render_stats_markdown_is_nonempty() -> None:
    records = [{"target": "a b", "references": ["a b"], "category": "X", "split": "train"}]
    md = render_stats_markdown(compute_stats(records))
    assert "Examples:" in md
    assert "Splits:" in md
