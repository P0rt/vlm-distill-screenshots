"""Tests for the KD loss. Requires torch; skipped where it isn't installed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vlm_distill.models.losses import IGNORE_INDEX, response_kd_loss  # noqa: E402


def _toy() -> tuple[object, object, object]:
    torch.manual_seed(0)
    student = torch.randn(2, 5, 16, requires_grad=True)
    teacher = torch.randn(2, 5, 16)
    labels = torch.randint(0, 16, (2, 5))
    labels[0, 0] = IGNORE_INDEX  # exercise the mask
    return student, teacher, labels


def test_response_kd_loss_is_finite_scalar() -> None:
    student, teacher, labels = _toy()
    loss = response_kd_loss(student, teacher, labels, alpha=0.5, temperature=2.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_response_kd_loss_matches_teacher_lowers_soft_term() -> None:
    # When student == teacher, the soft KL term is ~0, so total <= alpha*CE + eps.
    student, _teacher, labels = _toy()
    matched = response_kd_loss(student, student.detach(), labels, alpha=0.0, temperature=2.0)
    assert matched.item() < 1e-3


def test_response_kd_loss_validates_params() -> None:
    student, teacher, labels = _toy()
    with pytest.raises(ValueError):
        response_kd_loss(student, teacher, labels, alpha=1.5, temperature=2.0)
    with pytest.raises(ValueError):
        response_kd_loss(student, teacher, labels, alpha=0.5, temperature=0.0)
