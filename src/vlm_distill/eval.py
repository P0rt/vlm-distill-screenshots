"""Task-quality evaluation on the test split (SPEC Phase 5).

Compares the distilled **student**, an **untrained baseline** (same base model,
no adapter), and optionally the **teacher** against the human reference captions,
using ROUGE-L / BLEU (and an optional teacher LLM-as-judge score). Writes a
reproducible report to ``results/eval.json`` and refreshes the README table.

``--dry-run`` scores synthetic predictions with no models (runs in CI).
"""

from __future__ import annotations

import json
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
from vlm_distill.data.normalize import _clean_captions
from vlm_distill.metrics import compute_caption_metrics

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.Image import Image

RESULTS = REPO_ROOT / "results"
README_BEGIN = "<!-- EVAL_TABLE:BEGIN -->"
README_END = "<!-- EVAL_TABLE:END -->"


def _iter_test(
    cfg: DataConfig, *, limit: int | None, dry_run: bool
) -> Iterator[tuple[str, Image | None, list[str]]]:
    if dry_run:
        dry_refs = ["a login screen with email and password fields and a button"]
        for i in range(limit or 4):
            yield f"dry-{i}", None, dry_refs
        return
    from datasets import load_dataset

    stream = load_dataset(cfg.dataset_id, split="test", streaming=True, cache_dir=cfg.raw_dir)
    for i, example in enumerate(stream):
        if limit is not None and i >= limit:
            break
        refs = _clean_captions(example.get("captions"), cfg.max_references)
        yield str(example.get("screenId", i)), example["image"], refs


def _judge_scores(teacher_cfg: TeacherConfig, items: list[tuple[Image | None, str]]) -> list[int]:
    """Teacher rates each (image, prediction) 1-5. Best-effort integer parse."""
    import re

    from vlm_distill.models.teacher import make_teacher

    judge = make_teacher(teacher_cfg, dry_run=False)
    prompt = (
        "Rate how well this description matches the screenshot, from 1 (poor) to "
        "5 (excellent). Reply with only the number.\n\nDescription: {desc}"
    )
    scores: list[int] = []
    for image, desc in items:
        if image is None:
            scores.append(0)
            continue
        raw = judge.describe(image, prompt.format(desc=desc))
        m = re.search(r"[1-5]", raw)
        scores.append(int(m.group()) if m else 0)
    return scores


def _predict(model: Any, examples: list[tuple[str, Image | None, list[str]]]) -> list[str]:
    prompt = (
        "Describe this UI screenshot in one sentence, then list the key interface "
        "elements as a comma-separated list."
    )
    return [model.describe(image, prompt) for _, image, _ in examples]


def evaluate(
    student_cfg: StudentConfig,
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    models: list[str],
    adapter_path: str | None = None,
    limit: int | None = None,
    judge: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate the requested models on the test split; returns the report."""
    examples = list(_iter_test(data_cfg, limit=limit, dry_run=dry_run))
    references = [refs for _, _, refs in examples]

    report: dict[str, Any] = {"num_examples": len(examples), "models": {}}

    for name in models:
        if dry_run:
            preds = [refs[0] if refs else "" for _, _, refs in examples]
        else:
            from vlm_distill.models.student import make_student
            from vlm_distill.models.teacher import make_teacher

            if name == "teacher":
                model = make_teacher(teacher_cfg, dry_run=False)
            elif name == "baseline":
                model = make_student(student_cfg, adapter_path=None)
            elif name == "student":
                model = make_student(student_cfg, adapter_path=adapter_path)
            else:
                raise ValueError(f"Unknown model variant: {name!r}")
            preds = _predict(model, examples)

        metrics: dict[str, Any] = compute_caption_metrics(preds, references)
        if judge and not dry_run:
            scores = _judge_scores(
                teacher_cfg, [(img, p) for (_, img, _), p in zip(examples, preds, strict=True)]
            )
            valid = [s for s in scores if s > 0]
            metrics["judge_mean"] = round(sum(valid) / len(valid), 3) if valid else None
        report["models"][name] = metrics

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not dry_run:
        _update_readme(report)
    return report


def render_table(report: dict[str, Any]) -> str:
    has_judge = any("judge_mean" in m for m in report["models"].values())
    header = "| model | ROUGE-L | BLEU |" + (" judge (1-5) |" if has_judge else "")
    sep = "| --- | --- | --- |" + (" --- |" if has_judge else "")
    rows = [header, sep]
    for name, m in report["models"].items():
        row = f"| {name} | {m.get('rougeL', 0):.4f} | {m.get('bleu', 0):.4f} |"
        if has_judge:
            jm = m.get("judge_mean")
            row += f" {jm if jm is not None else '—'} |"
        rows.append(row)
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
    parser = build_parser("vlm-eval", "Evaluate task quality on the test split.")
    parser.add_argument("--student-config", type=str, default=None)
    parser.add_argument("--teacher-config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap test examples.")
    parser.add_argument(
        "--models",
        type=str,
        default="student,baseline",
        help="Comma list: student,baseline,teacher.",
    )
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter dir for student.")
    parser.add_argument("--judge", action="store_true", help="Add teacher LLM-as-judge score.")
    ns = parser.parse_args(argv)
    report = evaluate(
        load_student(ns.student_config),
        load_data(ns.config),
        load_teacher(ns.teacher_config),
        models=[m.strip() for m in ns.models.split(",") if m.strip()],
        adapter_path=ns.adapter,
        limit=ns.limit,
        judge=bool(ns.judge),
        dry_run=bool(ns.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
