"""Distillation losses (SPEC §3). Implemented in Phase 4 (response KD) and
Phase 7 (feature alignment).

``torch`` is only imported under ``TYPE_CHECKING`` so this module is importable
without the ML stack; ``torch.Tensor`` resolves to ``Any`` for mypy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vlm_distill.cli import PhaseNotImplementedError

if TYPE_CHECKING:
    import torch


def response_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Response-based KD: ``alpha*CE(hard) + (1-alpha)*KL(student/T||teacher/T)*T^2``.

    Implemented in Phase 4.
    """
    raise PhaseNotImplementedError("losses.response_kd_loss", phase=4)


def feature_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    kind: str = "cosine",
) -> torch.Tensor:
    """Cosine/MSE alignment between projected vision features. Phase 7."""
    raise PhaseNotImplementedError("losses.feature_alignment_loss", phase=7)
