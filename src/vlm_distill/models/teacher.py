"""Teacher VLM wrapper: load Qwen2-VL-7B in 4-bit + an inference helper (Phase 3).

Heavy deps (torch/transformers) are imported lazily inside functions so the
package stays importable in the lint/type CI job without the ML stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_distill.cli import PhaseNotImplementedError
from vlm_distill.config import TeacherConfig

if TYPE_CHECKING:
    from PIL.Image import Image


class Teacher:
    """Loaded teacher model + processor wrapper."""

    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg

    @classmethod
    def load(cls, cfg: TeacherConfig) -> Teacher:
        """Load the teacher (4-bit) and its processor. Implemented in Phase 3."""
        raise PhaseNotImplementedError("models.teacher.load", phase=3)

    def describe(self, image: Image, prompt: str) -> str:
        """Generate a screenshot description for one image. Phase 3."""
        raise PhaseNotImplementedError("models.teacher.describe", phase=3)

    def logits(self, image: Image, prompt: str, target: str) -> Any:
        """Return per-token (top-k) logits over the target for KD caching. Phase 3."""
        raise PhaseNotImplementedError("models.teacher.logits", phase=3)
