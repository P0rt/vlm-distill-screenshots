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


def make_student(
    cfg: StudentConfig, *, adapter_path: str | None = None, dry_run: bool = False
) -> StudentModel:
    """Factory: dry-run stand-in or the MLX backend (with optional adapter)."""
    if dry_run:
        return DryRunStudent()
    if cfg.backend == "mlx":
        return MlxStudent(cfg, adapter_path=adapter_path)
    if cfg.backend == "hf":
        raise NotImplementedError("hf student backend lands with the CUDA path (SPEC).")
    raise ValueError(f"Unknown student backend: {cfg.backend!r}")
