"""Generate teacher targets (and optionally cache top-k logits) for the train
split, versioned by config hash and resumable (SPEC Phase 3)."""

from __future__ import annotations

from vlm_distill.cli import PhaseNotImplementedError, build_parser
from vlm_distill.config import DataConfig, TeacherConfig, load_data, load_teacher


def teacher_label(
    data_cfg: DataConfig, teacher_cfg: TeacherConfig, *, dry_run: bool = False
) -> None:
    """Label the train split with the teacher; idempotent + resumable. Phase 3."""
    raise PhaseNotImplementedError("data.teacher_label", phase=3)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-teacher-label", "Label screenshots with the teacher VLM.")
    parser.add_argument(
        "--teacher-config", type=str, default=None, help="Path to teacher.yaml override."
    )
    ns = parser.parse_args(argv)
    data_cfg = load_data(ns.config)
    teacher_cfg = load_teacher(ns.teacher_config)
    teacher_label(data_cfg, teacher_cfg, dry_run=bool(ns.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
