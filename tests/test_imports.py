"""Ensure every src module imports without the heavy ML stack installed."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "vlm_distill",
    "vlm_distill.cli",
    "vlm_distill.config",
    "vlm_distill.data",
    "vlm_distill.data.download",
    "vlm_distill.data.build_dataset",
    "vlm_distill.data.teacher_label",
    "vlm_distill.data.splits",
    "vlm_distill.data.normalize",
    "vlm_distill.models",
    "vlm_distill.models.teacher",
    "vlm_distill.models.student",
    "vlm_distill.models.losses",
    "vlm_distill.train",
    "vlm_distill.eval",
    "vlm_distill.benchmark",
    "vlm_distill.export",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_version_exposed() -> None:
    import vlm_distill

    assert isinstance(vlm_distill.__version__, str)
