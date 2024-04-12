"""Smoke tests for the CLI scaffolding.

Phase 1 only guarantees the wiring: every entrypoint parses ``--help`` and,
when invoked, raises a clear ``PhaseNotImplementedError`` pointing at the phase
that will implement it. No heavy ML deps are imported here.
"""

from __future__ import annotations

import importlib

import pytest

from vlm_distill.cli import PhaseNotImplementedError

ENTRYPOINTS = [
    "vlm_distill.data.download",
    "vlm_distill.data.build_dataset",
    "vlm_distill.data.teacher_label",
    "vlm_distill.train",
    "vlm_distill.eval",
    "vlm_distill.benchmark",
    "vlm_distill.export",
]


@pytest.mark.parametrize("module_name", ENTRYPOINTS)
def test_help_exits_zero(module_name: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("module_name", ENTRYPOINTS)
def test_main_raises_phase_not_implemented(module_name: str) -> None:
    module = importlib.import_module(module_name)
    with pytest.raises(PhaseNotImplementedError):
        module.main([])


def test_phase_not_implemented_carries_metadata() -> None:
    err = PhaseNotImplementedError("some.step", phase=4)
    assert err.step == "some.step"
    assert err.phase == 4
    assert "Phase 4" in str(err)
