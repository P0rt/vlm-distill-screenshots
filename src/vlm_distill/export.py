"""Export the distilled student (SPEC Phase 8).

Merges the LoRA adapter into the base weights and saves a **standalone** student
(model + processor) you can load anywhere with plain ``transformers`` — no peft
needed at inference. Runs a sanity generation to confirm the merged model works.

Full-VLM **ONNX** export is finicky and is kept as a documented stretch goal
(SPEC §9); torch / MLX inference stays the canonical path. ``--dry-run`` writes a
manifest with no model (runs in CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vlm_distill.cli import build_parser
from vlm_distill.config import REPO_ROOT, StudentConfig, load_student

EXPORTS = REPO_ROOT / "results" / "exports"


def export_merged(
    student_cfg: StudentConfig,
    *,
    adapter_path: str | None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge the LoRA adapter into the base model and save a standalone student."""
    out_dir = output_dir or (EXPORTS / "merged-student")
    if dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"dry_run": True, "output_dir": str(out_dir), "merged": False}
        (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    if not adapter_path:
        raise ValueError("--adapter is required to export a trained student.")

    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        student_cfg.model_id, min_pixels=student_cfg.min_pixels, max_pixels=student_cfg.max_pixels
    )
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        student_cfg.model_id, torch_dtype=torch.float16
    )
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # Sanity check: reload nothing fancy, just generate once from the merged model.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    merged = merged.to(device).eval()
    image = Image.new("RGB", (360, 640), (240, 240, 240))
    content = [{"type": "image"}, {"type": "text", "text": "Describe this UI."}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = merged.generate(**inputs, max_new_tokens=24, do_sample=False)
    sample = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0].strip()

    summary = {
        "dry_run": False,
        "merged": True,
        "base_model": student_cfg.model_id,
        "adapter": adapter_path,
        "output_dir": str(out_dir),
        "sanity_generation": sample,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-export", "Export a standalone merged student (LoRA -> base).")
    parser.add_argument("--student-config", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None, help="Trained LoRA adapter dir.")
    parser.add_argument("--output", type=str, default=None, help="Output dir for the merged model.")
    ns = parser.parse_args(argv)
    summary = export_merged(
        load_student(ns.student_config),
        adapter_path=ns.adapter,
        output_dir=Path(ns.output) if ns.output else None,
        dry_run=bool(ns.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
