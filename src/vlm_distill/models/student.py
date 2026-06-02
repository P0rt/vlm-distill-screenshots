"""Student VLM wrapper: Qwen2-VL-2B + a trained LoRA adapter (SPEC Phase 4/5).

Mirrors ``teacher.py``: a ``StudentModel`` protocol with an MLX backend (loads
the base checkpoint plus an optional trained adapter) and a dependency-free
dry-run stand-in. Used for inference / eval; training lives in ``train.py``.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vlm_distill.config import StudentConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.Image import Image


@runtime_checkable
class StudentModel(Protocol):
    def describe(self, image: Image | None, prompt: str) -> str: ...


class DryRunStudent:
    """Deterministic, dependency-free stand-in for --dry-run / CI."""

    def describe(self, image: Image | None, prompt: str) -> str:
        size = "unknown-size" if image is None else f"{image.width}x{image.height}"
        return f"[dry-run student] UI screenshot ({size}) with a header and buttons."


@contextmanager
def _as_temp_png(image: Image) -> Iterator[str]:
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(name)
        yield name
    finally:
        Path(name).unlink(missing_ok=True)


class MlxStudent:
    """Qwen2-VL-2B via mlx-vlm, optionally with a trained LoRA adapter."""

    def __init__(self, cfg: StudentConfig, adapter_path: str | None = None) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.cfg = cfg
        self.model, self.processor = load(cfg.mlx_model_id, adapter_path=adapter_path)
        self.model_config = load_config(cfg.mlx_model_id)

    def describe(self, image: Image | None, prompt: str, *, max_tokens: int = 128) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if image is None:
            raise ValueError("MlxStudent requires an image.")
        formatted = apply_chat_template(self.processor, self.model_config, prompt, num_images=1)
        with _as_temp_png(image) as path:
            result = generate(
                self.model,
                self.processor,
                formatted,
                image=[path],
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()


class HfStudent:
    """Qwen2-VL-2B via transformers, with an optional trained LoRA adapter.

    Matches the ``hf`` training backend (runs on CUDA, or Apple MPS / CPU).
    """

    def __init__(self, cfg: StudentConfig, adapter_path: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.cfg = cfg
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if self.device in {"cuda", "mps"} else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=dtype)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device)
        self.model.eval()

    def describe(self, image: Image | None, prompt: str, *, max_tokens: int = 128) -> str:
        import torch

        if image is None:
            raise ValueError("HfStudent requires an image.")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        decoded: str = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded.strip()


def make_student(
    cfg: StudentConfig, *, adapter_path: str | None = None, dry_run: bool = False
) -> StudentModel:
    """Factory: dry-run stand-in, or the mlx / hf backend (with optional adapter)."""
    if dry_run:
        return DryRunStudent()
    if cfg.backend == "mlx":
        return MlxStudent(cfg, adapter_path=adapter_path)
    if cfg.backend == "hf":
        return HfStudent(cfg, adapter_path=adapter_path)
    raise ValueError(f"Unknown student backend: {cfg.backend!r}")
