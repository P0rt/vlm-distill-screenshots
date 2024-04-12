"""Inference benchmark: latency (p50/p95), throughput (img/s), peak VRAM for
teacher vs student on the same hardware (SPEC Phase 6).

Writes ``results/benchmark.json``.
"""

from __future__ import annotations

from vlm_distill.cli import PhaseNotImplementedError, build_parser
from vlm_distill.config import StudentConfig, TeacherConfig, load_student, load_teacher


def benchmark(
    teacher_cfg: TeacherConfig, student_cfg: StudentConfig, *, dry_run: bool = False
) -> None:
    """Measure latency/throughput/VRAM for both models. Phase 6."""
    raise PhaseNotImplementedError("benchmark", phase=6)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-benchmark", "Benchmark teacher vs student inference.")
    parser.add_argument(
        "--student-config", type=str, default=None, help="Path to student.yaml override."
    )
    ns = parser.parse_args(argv)
    teacher_cfg = load_teacher(ns.config)
    student_cfg = load_student(ns.student_config)
    benchmark(teacher_cfg, student_cfg, dry_run=bool(ns.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
