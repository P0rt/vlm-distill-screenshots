"""Tests for the eval orchestration (dry-run path, no heavy deps)."""

from __future__ import annotations

from vlm_distill.config import DataConfig, StudentConfig, TeacherConfig
from vlm_distill.eval import evaluate, render_table


def test_evaluate_dry_run() -> None:
    report = evaluate(
        StudentConfig(),
        DataConfig(),
        TeacherConfig(),
        models=["student", "baseline"],
        limit=4,
        dry_run=True,
        write_outputs=False,
    )
    assert report["num_examples"] == 4
    assert set(report["models"]) == {"student", "baseline"}
    # dry-run predicts the reference itself -> perfect ROUGE-L.
    assert report["models"]["student"]["rougeL"] == 1.0
    assert report["models"]["baseline"]["bleu"] >= 0.0


def test_render_table() -> None:
    report = {
        "num_examples": 2,
        "models": {
            "student": {"rougeL": 0.42, "bleu": 0.31},
            "baseline": {"rougeL": 0.20, "bleu": 0.10},
        },
    }
    table = render_table(report)
    assert "ROUGE-L" in table and "BLEU" in table
    assert "student" in table and "baseline" in table
    assert "0.4200" in table


def test_render_table_with_judge() -> None:
    report = {
        "num_examples": 1,
        "models": {"student": {"rougeL": 0.5, "bleu": 0.4, "judge_mean": 4.2}},
    }
    table = render_table(report)
    assert "judge" in table.lower()
    assert "4.2" in table
