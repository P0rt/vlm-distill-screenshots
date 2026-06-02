"""Inference benchmark + trade-off (SPEC Phase 6).

Measures per-image latency (p50/p95), throughput (img/s), and peak process
memory for the teacher vs the student on the same hardware, alongside the
parameter-count ratio. Writes ``results/benchmark.json`` and refreshes the
README table with an explicit quality-vs-speed conclusion.

The latency-stats helpers are pure and unit-tested; ``--dry-run`` synthesizes
timings so the wiring runs in CI without any model.
"""

from __future__ import annotations

import json
import platform
import resource
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vlm_distill.cli import build_parser
from vlm_distill.config import (
    REPO_ROOT,
    DataConfig,
    StudentConfig,
    TeacherConfig,
    load_data,
    load_student,
    load_teacher,
)

if TYPE_CHECKING:
    from PIL.Image import Image

RESULTS = REPO_ROOT / "results"
README_BEGIN = "<!-- BENCH_TABLE:BEGIN -->"
README_END = "<!-- BENCH_TABLE:END -->"

# Approximate parameter counts (billions) for the trade-off table.
_PARAMS_B = {"Qwen2-VL-7B": 8.29, "Qwen2-VL-2B": 2.21}

PROMPT = (
    "Describe this UI screenshot in one sentence, then list the key interface "
    "elements as a comma-separated list."
)


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """p50 / p95 / mean / min / max for a list of per-image latencies (ms)."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(latencies_ms)
    return {
        "p50": round(_percentile(s, 50), 1),
        "p95": round(_percentile(s, 95), 1),
        "mean": round(sum(s) / len(s), 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
    }


def throughput_img_s(latencies_ms: Sequence[float]) -> float:
    """Throughput in images/second from the mean latency."""
    if not latencies_ms:
        return 0.0
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    return round(1000.0 / mean_ms, 3) if mean_ms > 0 else 0.0


def _params_b(model_id: str) -> float | None:
    for key, val in _PARAMS_B.items():
        if key in model_id:
            return val
    return None


def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    scale = 1 if platform.system() == "Darwin" else 1024
    return round(rss * scale / 1e9, 2)


def _mlx_reset_peak() -> None:
    """Reset the MLX allocator peak so we can measure one model in isolation."""
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass


def _mlx_peak_gb() -> float | None:
    try:
        import mlx.core as mx

        return round(float(mx.get_peak_memory()) / 1e9, 2)
    except Exception:
        return None


def _time_model(model: Any, image: Image | None, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        model.describe(image, PROMPT)
    latencies: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        model.describe(image, PROMPT)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def _make_image() -> Image:
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (360, 640), (245, 245, 245))
    for y in range(96):
        for x in range(360):
            img.putpixel((x, y), (40, 110, 200))
    return img


def benchmark(
    teacher_cfg: TeacherConfig,
    student_cfg: StudentConfig,
    data_cfg: DataConfig,
    *,
    models: list[str],
    adapter_path: str | None = None,
    iters: int = 10,
    warmup: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Benchmark the requested models; returns the report."""
    report: dict[str, Any] = {
        "hardware": platform.platform(),
        "iters": iters,
        "models": {},
    }
    image = None if dry_run else _make_image()

    for name in models:
        if dry_run:
            base = 900.0 if name == "teacher" else 250.0
            latencies = [base + 10 * i for i in range(iters)]
            model_id = teacher_cfg.mlx_model_id if name == "teacher" else student_cfg.mlx_model_id
            peak_gb = 5.5 if name == "teacher" else 1.8
        else:
            from vlm_distill.models.student import make_student
            from vlm_distill.models.teacher import make_teacher

            _mlx_reset_peak()  # isolate this model's peak allocation
            if name == "teacher":
                model = make_teacher(teacher_cfg, dry_run=False)
                model_id = (
                    teacher_cfg.mlx_model_id
                    if teacher_cfg.backend == "mlx"
                    else teacher_cfg.model_id
                )
            else:
                model = make_student(student_cfg, adapter_path=adapter_path)
                model_id = (
                    student_cfg.mlx_model_id
                    if student_cfg.backend == "mlx"
                    else student_cfg.model_id
                )
            latencies = _time_model(model, image, warmup=warmup, iters=iters)
            # MLX allocator peak is per-model after the reset; fall back to RSS.
            peak_gb = _mlx_peak_gb() or _peak_rss_gb()
            del model

        report["models"][name] = {
            "model_id": model_id,
            "params_b": _params_b(model_id),
            "latency_ms": latency_stats(latencies),
            "throughput_img_s": throughput_img_s(latencies),
            "peak_mem_gb": peak_gb,
        }

    _add_tradeoff(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not dry_run:
        _update_readme(report)
    return report


def _add_tradeoff(report: dict[str, Any]) -> None:
    m = report["models"]
    if "teacher" in m and "student" in m:
        t, s = m["teacher"], m["student"]
        if s["latency_ms"]["p50"] > 0:
            report["speedup_p50"] = round(t["latency_ms"]["p50"] / s["latency_ms"]["p50"], 2)
        if t.get("params_b") and s.get("params_b"):
            report["param_ratio"] = round(t["params_b"] / s["params_b"], 2)


def render_table(report: dict[str, Any]) -> str:
    rows = [
        "| model | params (B) | latency p50 (ms) | p95 (ms) | throughput (img/s) | peak mem (GB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, mm in report["models"].items():
        lat = mm["latency_ms"]
        params = mm.get("params_b")
        rows.append(
            f"| {name} | {params if params is not None else '?'} | {lat['p50']:.0f} | "
            f"{lat['p95']:.0f} | {mm['throughput_img_s']:.2f} | {mm['peak_mem_gb']:.1f} |"
        )
    extra = []
    if "speedup_p50" in report:
        extra.append(f"**{report['speedup_p50']}x faster** (p50)")
    if "param_ratio" in report:
        extra.append(f"**{report['param_ratio']}x fewer params**")
    if extra:
        rows.append("")
        rows.append("Student vs teacher: " + ", ".join(extra) + ".")
    return "\n".join(rows)


def _update_readme(report: dict[str, Any]) -> None:
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        return
    block = f"{README_BEGIN}\n{render_table(report)}\n{README_END}"
    readme.write_text(
        text.split(README_BEGIN)[0] + block + text.split(README_END)[1], encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-benchmark", "Benchmark teacher vs student inference.")
    parser.add_argument("--student-config", type=str, default=None)
    parser.add_argument("--teacher-config", type=str, default=None)
    parser.add_argument("--models", type=str, default="teacher,student")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    ns = parser.parse_args(argv)
    report = benchmark(
        load_teacher(ns.teacher_config),
        load_student(ns.student_config),
        load_data(ns.config),
        models=[m.strip() for m in ns.models.split(",") if m.strip()],
        adapter_path=ns.adapter,
        iters=ns.iters,
        warmup=ns.warmup,
        dry_run=bool(ns.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
