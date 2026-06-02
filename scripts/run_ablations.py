#!/usr/bin/env python3
"""Phase 7 ablation driver: train a small grid of distillation runs and eval each
on the test split, writing results/ablations.json and the README table.

Varies the axes our sequence-level SFT actually exposes — training **steps** and
**LoRA rank** — at fixed lr/batch/data. (The alpha/temperature/feature-loss axes
belong to the logit-KD variant in ``models.losses.response_kd_loss``, which the
SFT path does not use; that is the next ablation axis once teacher logits are
cached.)

Heavy (torch); run on a GPU or Apple MPS:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/run_ablations.py
"""

from __future__ import annotations

import json

from vlm_distill.config import (
    REPO_ROOT,
    DataConfig,
    DistillConfig,
    LoraConfig,
    StudentConfig,
    load_teacher,
)
from vlm_distill.eval import evaluate
from vlm_distill.train import run_training

README_BEGIN = "<!-- ABLATIONS:BEGIN -->"
README_END = "<!-- ABLATIONS:END -->"

TRAIN_LIMIT = 80
EVAL_LIMIT = 16

# (label, lora_rank, max_steps). steps=0 -> untrained baseline (no adapter).
GRID: list[tuple[str, int, int]] = [
    ("baseline (0 steps)", 8, 0),
    ("r8, 40 steps", 8, 40),
    ("r8, 80 steps", 8, 80),
    ("r16, 40 steps", 16, 40),
]


def _student(rank: int) -> StudentConfig:
    return StudentConfig(
        backend="hf",
        lora=LoraConfig(r=rank, alpha=2 * rank, target_modules=["q_proj", "v_proj"]),
        min_pixels=200704,
        max_pixels=200704,
    )


def _distill(max_steps: int) -> DistillConfig:
    return DistillConfig(
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        steps_per_report=20,
        max_seq_length=1024,
    )


def render_table(rows: list[dict[str, object]]) -> str:
    out = ["| run | LoRA r | steps | ROUGE-L | BLEU |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| {r['label']} | {r['lora_r']} | {r['steps']} | {r['rougeL']:.4f} | {r['bleu']:.4f} |"
        )
    return "\n".join(out)


def _update_readme(rows: list[dict[str, object]]) -> None:
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        return
    block = f"{README_BEGIN}\n{render_table(rows)}\n{README_END}"
    readme.write_text(
        text.split(README_BEGIN)[0] + block + text.split(README_END)[1], encoding="utf-8"
    )


def main() -> int:
    data_cfg = DataConfig()
    teacher_cfg = load_teacher(None)
    rows: list[dict[str, object]] = []

    for label, rank, steps in GRID:
        student_cfg = _student(rank)
        if steps == 0:
            adapter = None
            model_name = "baseline"
        else:
            summary = run_training(
                student_cfg, _distill(steps), data_cfg, teacher_cfg, limit=TRAIN_LIMIT
            )
            adapter = summary["adapter_dir"]
            model_name = "student"
        report = evaluate(
            student_cfg,
            data_cfg,
            teacher_cfg,
            models=[model_name],
            adapter_path=adapter,
            limit=EVAL_LIMIT,
            write_outputs=False,
        )
        m = report["models"][model_name]
        rows.append(
            {
                "label": label,
                "lora_r": rank,
                "steps": steps,
                "rougeL": m["rougeL"],
                "bleu": m["bleu"],
            }
        )
        print(f"[ablation] {label}: ROUGE-L {m['rougeL']:.4f}  BLEU {m['bleu']:.4f}", flush=True)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_readme(rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
