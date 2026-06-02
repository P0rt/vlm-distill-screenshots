"""Smoke tests for the CLI scaffolding.

Every entrypoint must parse ``--help`` with no heavy ML deps imported.
"""

from __future__ import annotations

import importlib

import pytest

from vlm_distill.cli import PhaseNotImplementedError

# Every entrypoint must parse --help with no heavy deps. All phases (2-8) are
# implemented now; --help still works without the ML stack.
ALL_ENTRYPOINTS = [
    "vlm_distill.data.download",
    "vlm_distill.data.build_dataset",
    "vlm_distill.data.teacher_label",
    "vlm_distill.train",
    "vlm_distill.eval",
    "vlm_distill.benchmark",
    "vlm_distill.export",
]


@pytest.mark.parametrize("module_name", ALL_ENTRYPOINTS)
def test_help_exits_zero(module_name: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0


def test_phase_not_implemented_carries_metadata() -> None:
    # The helper is still used for documented stretch paths.
    err = PhaseNotImplementedError("some.step", phase=7)
    assert err.step == "some.step"
    assert err.phase == 7
    assert "Phase 7" in str(err)
