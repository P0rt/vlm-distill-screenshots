"""Tests for export dry-run (no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_distill.config import StudentConfig
from vlm_distill.export import export_merged


def test_export_dry_run(tmp_path: Path) -> None:
    summary = export_merged(StudentConfig(), adapter_path=None, output_dir=tmp_path, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["merged"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dry_run"] is True


def test_export_requires_adapter_when_real() -> None:
    with pytest.raises(ValueError, match="adapter is required"):
        export_merged(StudentConfig(), adapter_path=None, dry_run=False)
