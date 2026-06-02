"""Distillation losses (SPEC §3).

``response_kd_loss`` is the canonical logit-level KD objective used by the
``hf`` training path and the Phase 7 logit-distillation ablation:

    L = alpha * CE(student, hard_labels)
        + (1 - alpha) * T^2 * KL(softmax(teacher/T) || log_softmax(student/T))

The MLX MVP path (Phase 4) distills at the *sequence* level (the student is
trained to reproduce the teacher's text), which is the hard-label term alone;
this function adds the soft-target term for when cached teacher logits are
available.

``torch`` is imported lazily so the module imports without the ML stack;
``torch.Tensor`` resolves to ``Any`` for mypy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vlm_distill.cli import PhaseNotImplementedError

if TYPE_CHECKING:
    import torch

IGNORE_INDEX = -100


def response_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Response-based KD (soft + hard targets). Shapes: logits ``(B, T, V)``,
    labels ``(B, T)`` with ``IGNORE_INDEX`` for non-target positions."""
    import torch.nn.functional as F

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    vocab = student_logits.size(-1)
    # Hard-label cross-entropy (ignores masked positions).
    hard = F.cross_entropy(
        student_logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )

    # Soft-target KL on the unmasked positions only.
    mask = labels.view(-1) != IGNORE_INDEX
    s = student_logits.view(-1, vocab)[mask]
    t = teacher_logits.view(-1, vocab)[mask]
    if s.numel() == 0:
        return alpha * hard
    student_logp = F.log_softmax(s / temperature, dim=-1)
    teacher_p = F.softmax(t / temperature, dim=-1)
    soft = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature**2)

    return alpha * hard + (1.0 - alpha) * soft


def feature_alignment_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    kind: str = "cosine",
) -> torch.Tensor:
    """Cosine/MSE alignment between projected vision features. Phase 7."""
    raise PhaseNotImplementedError("losses.feature_alignment_loss", phase=7)
