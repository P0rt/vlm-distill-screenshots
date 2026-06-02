"""Tests for benchmark stats + dry-run (no models)."""

from __future__ import annotations

from vlm_distill.benchmark import (
    benchmark,
    latency_stats,
    render_table,
    throughput_img_s,
)
from vlm_distill.config import DataConfig, StudentConfig, TeacherConfig


def test_latency_stats() -> None:
    s = latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
    assert s["min"] == 100.0
    assert s["max"] == 500.0
    assert s["mean"] == 300.0
    assert s["p50"] == 300.0
    assert s["p95"] >= s["p50"]


def test_latency_stats_empty() -> None:
    s = latency_stats([])
    assert s["p50"] == 0.0 and s["mean"] == 0.0


def test_throughput() -> None:
    # mean 250 ms -> 4 img/s
    assert throughput_img_s([250.0, 250.0]) == 4.0
    assert throughput_img_s([]) == 0.0


def test_benchmark_dry_run_computes_speedup() -> None:
    report = benchmark(
        TeacherConfig(),
        StudentConfig(),
        DataConfig(),
        models=["teacher", "student"],
        iters=5,
        dry_run=True,
    )
    assert set(report["models"]) == {"teacher", "student"}
    # synthetic: teacher ~900ms, student ~250ms -> speedup > 1
    assert report["speedup_p50"] > 1.0
    assert report["param_ratio"] > 1.0
    assert report["models"]["teacher"]["throughput_img_s"] > 0


def test_render_table() -> None:
    report = benchmark(
        TeacherConfig(),
        StudentConfig(),
        DataConfig(),
        models=["teacher", "student"],
        iters=3,
        dry_run=True,
    )
    table = render_table(report)
    assert "params (B)" in table
    assert "faster" in table
