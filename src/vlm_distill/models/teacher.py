"""Teacher VLM wrapper: load Qwen2-VL-7B and describe screenshots (SPEC Phase 3).

Two backends behind one ``TeacherModel`` protocol:
  - ``mlx``  -> mlx-vlm + a 4-bit MLX checkpoint (Apple Silicon).
  - ``hf``   -> transformers + bitsandbytes 4-bit (CUDA).
Plus a dependency-free ``DryRunTeacher`` for CI / graph validation.

Heavy deps (mlx_vlm / torch / transformers / PIL) are imported lazily inside
the concrete backends so this module imports without them.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vlm_distill.config import TeacherConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.Image import Image


@runtime_checkable
class TeacherModel(Protocol):
    """Anything that can turn (image, prompt) into a text description."""

    def describe(self, image: Image | None, prompt: str) -> str: ...


class DryRunTeacher:
    """Deterministic, dependency-free stand-in used by --dry-run and CI."""

    def describe(self, image: Image | None, prompt: str) -> str:
        size = "unknown-size" if image is None else f"{image.width}x{image.height}"
        return (
            f"[dry-run] A mobile UI screenshot ({size}). "
            "Key elements: top app bar, title text, primary button, list items."
        )


@contextmanager
def _as_temp_png(image: Image) -> Iterator[str]:
    """Write a PIL image to a temporary PNG and yield its path."""
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(name)
        yield name
    finally:
        Path(name).unlink(missing_ok=True)


class MlxTeacher:
    """Qwen2-VL via mlx-vlm (Apple Silicon, 4-bit)."""

    def __init__(self, cfg: TeacherConfig) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.cfg = cfg
        self.model, self.processor = load(cfg.mlx_model_id)
        self.model_config = load_config(cfg.mlx_model_id)

    def describe(self, image: Image | None, prompt: str) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if image is None:
            raise ValueError("MlxTeacher requires an image.")
        formatted = apply_chat_template(self.processor, self.model_config, prompt, num_images=1)
        gen = self.cfg.generation
        kwargs: dict[str, object] = {
            "max_tokens": gen.max_new_tokens,
            "temperature": gen.temperature if gen.do_sample else 0.0,
            "verbose": False,
        }
        if gen.do_sample:
            kwargs["top_p"] = gen.top_p
        with _as_temp_png(image) as path:
            result = generate(self.model, self.processor, formatted, image=[path], **kwargs)
        text = result.text if hasattr(result, "text") else str(result)
        return text.strip()


class HfTeacher:
    """Qwen2-VL via transformers + bitsandbytes 4-bit (CUDA)."""

    def __init__(self, cfg: TeacherConfig) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )

        self.cfg = cfg
        quant = None
        if cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, cfg.dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            device_map=cfg.device_map,
            torch_dtype=getattr(torch, cfg.dtype),
            quantization_config=quant,
        )
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
        )

    def describe(self, image: Image | None, prompt: str) -> str:
        if image is None:
            raise ValueError("HfTeacher requires an image.")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )
        gen = self.cfg.generation
        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
        )
        trimmed = out[:, inputs["input_ids"].shape[1] :]
        decoded: str = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.strip()


def make_teacher(cfg: TeacherConfig, *, dry_run: bool = False) -> TeacherModel:
    """Factory: pick a backend (or the dry-run stand-in)."""
    if dry_run:
        return DryRunTeacher()
    if cfg.backend == "mlx":
        return MlxTeacher(cfg)
    if cfg.backend == "hf":
        return HfTeacher(cfg)
    raise ValueError(f"Unknown teacher backend: {cfg.backend!r}")
