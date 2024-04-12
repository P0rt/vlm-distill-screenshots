"""Student VLM wrapper: Qwen2-VL-2B + LoRA, plus optional projection adapters
for the feature-alignment ablation (Phase 4 / Phase 7).

Heavy deps are imported lazily inside functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_distill.cli import PhaseNotImplementedError
from vlm_distill.config import StudentConfig

if TYPE_CHECKING:
    from PIL.Image import Image


class Student:
    """Loaded student model + processor wrapper."""

    def __init__(self, cfg: StudentConfig) -> None:
        self.cfg = cfg

    @classmethod
    def load(cls, cfg: StudentConfig) -> Student:
        """Load the student and attach LoRA adapters. Implemented in Phase 4."""
        raise PhaseNotImplementedError("models.student.load", phase=4)

    def describe(self, image: Image, prompt: str) -> str:
        """Generate a screenshot description for one image. Phase 4."""
        raise PhaseNotImplementedError("models.student.describe", phase=4)

    def forward_with_features(self, batch: Any) -> Any:
        """Forward pass returning logits + selected vision features. Phase 4/7."""
        raise PhaseNotImplementedError("models.student.forward_with_features", phase=4)
