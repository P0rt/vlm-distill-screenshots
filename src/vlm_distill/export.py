"""ONNX export + output sanity-check against the torch model (SPEC Phase 8)."""

from __future__ import annotations

from vlm_distill.cli import PhaseNotImplementedError, build_parser, parse_common
from vlm_distill.config import StudentConfig, load_student


def export_onnx(student_cfg: StudentConfig, *, dry_run: bool = False) -> None:
    """Export the trained student to ONNX and verify outputs match. Phase 8."""
    raise PhaseNotImplementedError("export", phase=8)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-export", "Export the student to ONNX with a sanity check.")
    args = parse_common(parser, argv)
    student_cfg = load_student(args.config)
    export_onnx(student_cfg, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
