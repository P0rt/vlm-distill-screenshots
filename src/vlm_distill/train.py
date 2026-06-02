"""Distillation training (SPEC Phase 4): response-based / sequence-level KD —
the student learns to reproduce the teacher's text targets.

Backends:
  - ``hf`` : LoRA SFT via transformers + peft. The portable / CUDA path (SPEC §2);
    runs on a 24GB GPU, or CPU/MPS for a smoke check.
  - ``mlx``: LoRA SFT via mlx-vlm's trainer (Apple Silicon). NOTE: blocked on
    mlx-vlm 0.6.0 (``Primitive::vjp not implemented for CustomKernel`` during
    backprop); usable once that upstream gap is fixed.

``--dry-run`` validates the data -> messages wiring with no model / no heavy
deps (runs in CI). A real run needs teacher labels first (``vlm-teacher-label``)
and the matching extra (``ml`` for hf, ``mlx`` for mlx).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vlm_distill.cli import build_parser
from vlm_distill.config import (
    REPO_ROOT,
    DataConfig,
    DistillConfig,
    StudentConfig,
    TeacherConfig,
    config_hash,
    load_data,
    load_distill,
    load_student,
    load_teacher,
)
from vlm_distill.data.distill_dataset import (
    build_distill_dataset,
    synthetic_distill_records,
    to_messages,
)

CHECKPOINTS_ROOT = REPO_ROOT / "results" / "checkpoints"


def _iters_for(distill_cfg: DistillConfig, dataset_len: int) -> int:
    """Resolve the number of training iterations from the config."""
    if distill_cfg.max_steps is not None:
        return distill_cfg.max_steps
    steps_per_epoch = max(1, dataset_len // distill_cfg.batch_size)
    return steps_per_epoch * distill_cfg.num_epochs


def _dry_run(distill_cfg: DistillConfig, *, output_dir: Path, limit: int | None) -> dict[str, Any]:
    """No model, no heavy deps: exercise the record -> messages graph."""
    records = synthetic_distill_records(limit or 4)
    messages = [to_messages(r["question"], r["answer"]) for r in records]
    assert all(len(m) == 2 and m[0]["role"] == "user" for m in messages)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "backend": "dry-run",
        "num_examples": len(records),
        "iters": _iters_for(distill_cfg, len(records)),
        "batch_size": distill_cfg.batch_size,
        "dry_run": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _train_mlx(
    student_cfg: StudentConfig,
    distill_cfg: DistillConfig,
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    """LoRA SFT via mlx-vlm's trainer."""
    import mlx.optimizers as optim
    from datasets import load_from_disk
    from mlx_vlm import load
    from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
    from mlx_vlm.trainer import VisionDataset
    from mlx_vlm.trainer.sft_trainer import TrainingArgs, train

    data_dir, n = build_distill_dataset(data_cfg, teacher_cfg, limit=limit)
    dataset = load_from_disk(str(data_dir))

    model, processor = load(student_cfg.mlx_model_id, processor_config={"trust_remote_code": True})
    # Cap the visual token budget so large RICO screenshots don't blow past
    # max_seq_length (otherwise get_rope_index hits a shape mismatch).
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.min_pixels = student_cfg.min_pixels
        image_processor.max_pixels = student_cfg.max_pixels
        if getattr(image_processor, "size", None):
            image_processor.size = {
                "shortest_edge": student_cfg.min_pixels,
                "longest_edge": student_cfg.max_pixels,
            }
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    config = model.config.__dict__

    dataset = transform_dataset_to_messages(dataset, model_type)
    train_dataset = VisionDataset(dataset, config, processor)

    setup_args = SimpleNamespace(
        full_finetune=False,
        train_vision=False,
        lora_rank=student_cfg.lora.r,
        lora_alpha=student_cfg.lora.alpha,
        lora_dropout=student_cfg.lora.dropout,
    )
    model = setup_model_for_training(model, setup_args)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = output_dir / "adapters.safetensors"
    iters = _iters_for(distill_cfg, n)
    training_args = TrainingArgs(
        batch_size=distill_cfg.batch_size,
        iters=iters,
        steps_per_report=distill_cfg.steps_per_report,
        steps_per_eval=10**9,  # no val set in the MVP
        steps_per_save=max(1, iters),
        max_seq_length=distill_cfg.max_seq_length,
        adapter_file=str(adapter_file),
        learning_rate=distill_cfg.learning_rate,
        grad_clip=distill_cfg.grad_clip,
        gradient_accumulation_steps=distill_cfg.gradient_accumulation_steps,
    )
    optimizer = optim.Adam(learning_rate=distill_cfg.learning_rate)
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
        train_on_completions=True,
    )
    return {
        "backend": "mlx",
        "model": student_cfg.mlx_model_id,
        "num_examples": n,
        "iters": iters,
        "batch_size": distill_cfg.batch_size,
        "learning_rate": distill_cfg.learning_rate,
        "adapter_file": str(adapter_file),
        "dry_run": False,
    }


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _train_hf(
    student_cfg: StudentConfig,
    distill_cfg: DistillConfig,
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any]:
    """LoRA SFT via transformers + peft (the CUDA path, SPEC §2).

    Sequence-level KD: micro-batch of 1 with gradient accumulation (multimodal
    batches have variable image-grid sizes, so we accumulate rather than pad).
    Real runs target a 24GB GPU; CPU/MPS works for a smoke check.
    """
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    data_dir, n = build_distill_dataset(data_cfg, teacher_cfg, limit=limit)
    dataset = load_from_disk(str(data_dir))

    device = _select_device()
    dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    processor = AutoProcessor.from_pretrained(
        student_cfg.model_id,
        min_pixels=student_cfg.min_pixels,
        max_pixels=student_cfg.max_pixels,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(student_cfg.model_id, torch_dtype=dtype)
    lora = PeftLoraConfig(
        r=student_cfg.lora.r,
        lora_alpha=student_cfg.lora.alpha,
        lora_dropout=student_cfg.lora.dropout,
        target_modules=student_cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.train()

    def _example_to_inputs(example: dict[str, Any]) -> dict[str, Any]:
        question, answer, image = example["question"], example["answer"], example["image"]
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": answer},
        ]
        full = processor.apply_chat_template(messages, tokenize=False)
        prompt = processor.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[full], images=[image], return_tensors="pt")
        prompt_ids = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"]
        labels = inputs["input_ids"].clone()
        labels[:, : prompt_ids.shape[1]] = -100  # mask the prompt; train on the answer
        inputs["labels"] = labels
        return {k: v.to(device) for k, v in inputs.items()}

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=distill_cfg.learning_rate
    )
    accum = max(1, distill_cfg.batch_size)
    iters = _iters_for(distill_cfg, n)

    def _examples() -> Any:
        while True:
            yield from dataset

    stream = _examples()
    losses: list[float] = []
    for step in range(iters):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accum):
            inputs = _example_to_inputs(next(stream))
            loss = model(**inputs).loss / accum
            loss.backward()
            step_loss += float(loss.item())
        if distill_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), distill_cfg.grad_clip)
        optimizer.step()
        losses.append(step_loss)
        if (step + 1) % distill_cfg.steps_per_report == 0:
            print(f"iter {step + 1}/{iters}  train loss {step_loss:.4f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    return {
        "backend": "hf",
        "model": student_cfg.model_id,
        "device": device,
        "num_examples": n,
        "iters": iters,
        "effective_batch": accum,
        "learning_rate": distill_cfg.learning_rate,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "adapter_dir": str(output_dir),
        "dry_run": False,
    }


def run_training(
    student_cfg: StudentConfig,
    distill_cfg: DistillConfig,
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Distill the teacher into the student. Returns a run summary."""
    run_hash = config_hash(student_cfg, distill_cfg, data_cfg)
    output_dir = CHECKPOINTS_ROOT / ("dryrun" if dry_run else run_hash)

    if dry_run:
        return _dry_run(distill_cfg, output_dir=output_dir, limit=limit)
    if student_cfg.backend == "mlx":
        return _train_mlx(
            student_cfg, distill_cfg, data_cfg, teacher_cfg, output_dir=output_dir, limit=limit
        )
    if student_cfg.backend == "hf":
        return _train_hf(
            student_cfg, distill_cfg, data_cfg, teacher_cfg, output_dir=output_dir, limit=limit
        )
    raise ValueError(f"Unknown student backend: {student_cfg.backend!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-train", "Distill the teacher into the student.")
    parser.add_argument("--student-config", type=str, default=None)
    parser.add_argument("--teacher-config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap training examples.")
    ns = parser.parse_args(argv)
    student_cfg = load_student(ns.student_config)
    distill_cfg = load_distill(ns.config)
    data_cfg = load_data(None)
    teacher_cfg = load_teacher(ns.teacher_config)
    summary = run_training(
        student_cfg, distill_cfg, data_cfg, teacher_cfg, limit=ns.limit, dry_run=bool(ns.dry_run)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
