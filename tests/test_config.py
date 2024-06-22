"""Tests for config loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vlm_distill.config import (
    DataConfig,
    DistillConfig,
    StudentConfig,
    TeacherConfig,
    config_hash,
    load_data,
    load_distill,
    load_student,
    load_teacher,
)


def test_default_configs_load_from_disk() -> None:
    teacher = load_teacher()
    student = load_student()
    distill = load_distill()
    data = load_data()

    assert teacher.model_id == "Qwen/Qwen2-VL-7B-Instruct"
    assert teacher.backend == "mlx"
    assert teacher.mlx_model_id == "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    assert teacher.load_in_4bit is True
    assert student.model_id == "Qwen/Qwen2-VL-2B-Instruct"
    assert student.lora.enabled is True
    assert distill.use_response_kd is True
    assert 0.0 <= distill.alpha <= 1.0
    assert data.dataset_id == "rootsautomation/RICO-Screen2Words"
    assert data.use_native_splits is True
    assert data.max_references == 5
    assert data.split.train + data.split.val + data.split.test == pytest.approx(1.0)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TeacherConfig.model_validate({"model_id": "x", "bogus_key": 1})


def test_alpha_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        DistillConfig.model_validate({"alpha": 1.5})


def test_config_hash_is_deterministic_and_sensitive() -> None:
    a = config_hash(TeacherConfig(), DistillConfig())
    b = config_hash(TeacherConfig(), DistillConfig())
    c = config_hash(TeacherConfig(), DistillConfig(alpha=0.9))

    assert a == b
    assert a != c
    assert len(a) == 12


def test_student_fallback_defaults() -> None:
    student = StudentConfig()
    assert student.fallback.enabled is False
    assert "siglip" in student.fallback.vision_model_id.lower()


def test_data_prompt_template_present() -> None:
    data = DataConfig()
    assert data.prompt_template.strip() != ""
