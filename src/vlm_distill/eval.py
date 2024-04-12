"""Task-quality evaluation: CIDEr/ROUGE-L/BLEU + optional LLM-as-judge (Phase 5).

Writes a reproducible report to ``results/eval.json``.
"""

from __future__ import annotations

from vlm_distill.cli import PhaseNotImplementedError, build_parser
from vlm_distill.config import DataConfig, StudentConfig, load_data, load_student


def evaluate(student_cfg: StudentConfig, data_cfg: DataConfig, *, dry_run: bool = False) -> None:
    """Evaluate student vs teacher vs baseline on the test split. Phase 5."""
    raise PhaseNotImplementedError("eval", phase=5)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-eval", "Evaluate task quality on the test split.")
    parser.add_argument(
        "--student-config", type=str, default=None, help="Path to student.yaml override."
    )
    ns = parser.parse_args(argv)
    data_cfg = load_data(ns.config)
    student_cfg = load_student(ns.student_config)
    evaluate(student_cfg, data_cfg, dry_run=bool(ns.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
