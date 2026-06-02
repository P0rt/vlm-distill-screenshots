#!/usr/bin/env python3
"""Generate the trade-off / ablation figures for the article and README.

Reads results/{eval,benchmark,ablations}.json when present (those are gitignored
run outputs) and falls back to the measured constants below, so the figures
regenerate deterministically on a fresh clone. Writes PNGs to assets/.

    uv sync --extra viz
    uv run python scripts/make_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

# Measured fallbacks (Apple M4 Pro, MLX/MPS, 4-bit, PoC scale).
FALLBACK_QUALITY = {"teacher": 0.1636, "student": 0.1776, "baseline": 0.1529}
FALLBACK_EFFICIENCY = {
    "teacher": {"throughput_img_s": 0.63, "peak_mem_gb": 5.8, "params_b": 8.29},
    "student": {"throughput_img_s": 1.52, "peak_mem_gb": 2.4, "params_b": 2.21},
}
FALLBACK_ABLATION = [(0, 0.1516), (40, 0.1701), (80, 0.1716)]

COLORS = {"teacher": "#c0392b", "student": "#2980b9", "baseline": "#7f8c8d"}


def _load(name: str) -> dict | list | None:
    path = RESULTS / name
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _quality() -> dict[str, float]:
    data = _load("eval.json")
    if isinstance(data, dict) and "models" in data:
        out = {}
        for k in ("teacher", "student", "baseline"):
            if k in data["models"]:
                out[k] = float(data["models"][k]["rougeL"])
        if {"student", "baseline"} <= out.keys():
            return {**FALLBACK_QUALITY, **out}
    return FALLBACK_QUALITY


def _efficiency() -> dict[str, dict[str, float]]:
    data = _load("benchmark.json")
    if isinstance(data, dict) and "models" in data:
        eff = {}
        for k in ("teacher", "student"):
            if k in data["models"]:
                eff[k] = data["models"][k]
        if eff:
            return {**FALLBACK_EFFICIENCY, **eff}
    return FALLBACK_EFFICIENCY


def _ablation() -> list[tuple[int, float]]:
    data = _load("ablations.json")
    if isinstance(data, list):
        pts = [(int(r["steps"]), float(r["rougeL"])) for r in data if r.get("lora_r") in (8, None)]
        if pts:
            return sorted(pts)
    return FALLBACK_ABLATION


def _scatter(metric: str, xlabel: str, fname: str, title: str) -> None:
    quality = _quality()
    eff = _efficiency()
    # baseline shares the student's runtime cost (same 2B architecture).
    points = {
        "teacher": (eff["teacher"][metric], quality["teacher"]),
        "student (distilled)": (eff["student"][metric], quality["student"]),
        "baseline (untrained)": (eff["student"][metric], quality["baseline"]),
    }
    color = {
        "teacher": COLORS["teacher"],
        "student (distilled)": COLORS["student"],
        "baseline (untrained)": COLORS["baseline"],
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label, (x, y) in points.items():
        ax.scatter([x], [y], s=170, color=color[label], zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ROUGE-L vs human refs (test)")
    ax.set_title(title, fontsize=12)
    ax.margins(x=0.22, y=0.18)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / fname, dpi=150)
    plt.close(fig)
    print("wrote", ASSETS / fname)


def _ablation_plot() -> None:
    pts = _ablation()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(xs, ys, "-o", color=COLORS["student"], linewidth=2, markersize=8)
    for x, y in pts:
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax.set_xlabel("training steps (LoRA r=8)")
    ax.set_ylabel("ROUGE-L (test)")
    ax.set_title("Ablation: more distillation steps -> higher ROUGE-L")
    ax.margins(x=0.12, y=0.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "ablation_steps.png", dpi=150)
    plt.close(fig)
    print("wrote", ASSETS / "ablation_steps.png")


def main() -> int:
    _scatter(
        "throughput_img_s",
        "throughput (images / s)  -->  faster",
        "quality_vs_speed.png",
        "Quality vs speed (2B student is ~2.4x faster than the 7B teacher)",
    )
    _scatter(
        "peak_mem_gb",
        "peak memory (GB)  -->  lighter",
        "quality_vs_memory.png",
        "Quality vs memory (2B student runs in ~2.4x less memory)",
    )
    _ablation_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
