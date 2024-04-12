"""Distillation training loop (accelerate + LoRA + W&B), SPEC Phase 4.

Supports ``--dry-run`` for a 1-step CPU smoke test so CI stays cheap.
"""

from __future__ import annotations

from vlm_distill.cli import PhaseNotImplementedError, build_parser
from vlm_distill.config import (
    DistillConfig,
    StudentConfig,
    load_distill,
    load_student,
)


def train(student_cfg: StudentConfig, distill_cfg: DistillConfig, *, dry_run: bool = False) -> None:
    """Run distillation training. Implemented in Phase 4."""
    raise PhaseNotImplementedError("train", phase=4)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-train", "Distill the teacher into the student.")
    parser.add_argument(
        "--student-config", type=str, default=None, help="Path to student.yaml override."
    )
    ns = parser.parse_args(argv)
    student_cfg = load_student(ns.student_config)
    distill_cfg = load_distill(ns.config)
    train(student_cfg, distill_cfg, dry_run=bool(ns.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
