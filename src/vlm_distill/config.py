"""Typed configuration models and YAML loading.

All hyperparameters and paths live in ``configs/*.yaml`` (never hardcoded in
code, per the project conventions). This module gives each config file a typed
``pydantic`` schema and a small loader that validates on read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Repository root, resolved relative to this file: src/vlm_distill/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class _StrictModel(BaseModel):
    """Base model: forbid unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid")


class GenerationConfig(_StrictModel):
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


class TeacherConfig(_StrictModel):
    """Teacher VLM (default: Qwen2-VL-7B-Instruct, loaded in 4-bit).

    Two inference backends:
      - ``mlx``: Apple Silicon, via mlx-vlm + a 4-bit MLX checkpoint
        (``mlx_model_id``). Runs on this project's dev machine (M-series, 24GB).
      - ``hf``: CUDA, via transformers + bitsandbytes 4-bit (``model_id``).
        For the rented-GPU labeling run.
    """

    backend: Literal["mlx", "hf"] = "mlx"
    # HF checkpoint (used by the `hf` backend and as the canonical model id).
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    # MLX 4-bit checkpoint (used by the `mlx` backend).
    mlx_model_id: str = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    # Alternatives are documented but not wired up in code yet (see SPEC §2).
    alternatives: list[str] = Field(
        default_factory=lambda: ["llava-hf/llava-v1.6-mistral-7b-hf", "OpenGVLab/InternVL2-8B"]
    )
    load_in_4bit: bool = True
    dtype: str = "bfloat16"
    device_map: str = "auto"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class LoraConfig(_StrictModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class StudentFallbackConfig(_StrictModel):
    """Backup architecture: frozen SigLIP vision + small Qwen2.5 decoder."""

    enabled: bool = False
    vision_model_id: str = "google/siglip-so400m-patch14-384"
    decoder_model_id: str = "Qwen/Qwen2.5-0.5B"


class StudentConfig(_StrictModel):
    """Student VLM (default: Qwen2-VL-2B-Instruct, fine-tuned with LoRA)."""

    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    dtype: str = "bfloat16"
    lora: LoraConfig = Field(default_factory=LoraConfig)
    fallback: StudentFallbackConfig = Field(default_factory=StudentFallbackConfig)


class FeatureLossConfig(_StrictModel):
    enabled: bool = False
    kind: str = "cosine"  # "cosine" | "mse"
    weight: float = 1.0
    teacher_layers: list[int] = Field(default_factory=list)
    student_layers: list[int] = Field(default_factory=list)


class DistillConfig(_StrictModel):
    """Distillation hyperparameters. Each signal is gated by a flag for ablations."""

    # Response-based KD (the MVP signal): L = a*CE(hard) + (1-a)*KL*T^2.
    use_response_kd: bool = True
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=2.0, gt=0.0)
    cache_teacher_logits: bool = True
    top_k_logits: int = 64

    # Feature-based alignment (ablation).
    feature_loss: FeatureLossConfig = Field(default_factory=FeatureLossConfig)

    # Self-distillation / teacher-generated synthetic data (ablation).
    use_synthetic_data: bool = False

    # Optimization.
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    seed: int = 42
    max_steps: int | None = None


class SplitConfig(_StrictModel):
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05
    seed: int = 42


class DataConfig(_StrictModel):
    """Datasets, paths, prompt template, and split configuration."""

    # Screen2Words captions bundled with their RICO screenshots (CC-BY-4.0).
    # Native splits: train 15700 / val 2360 / test 4310.
    dataset_id: str = "rootsautomation/RICO-Screen2Words"
    # RICO source for teacher-labeled augmentation (Phase 7); unused in Phase 2.
    rico_id: str = "rootsautomation/RICO-Screen2Words"
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    # Single source of truth for the task prompt (SPEC §2).
    prompt_template: str = (
        "Describe this UI screenshot in one sentence, then list the key interface "
        "elements as a comma-separated list."
    )
    max_target_length: int = 256
    # Number of human reference captions to keep per screen (for eval metrics).
    max_references: int = 5
    # Use the dataset's native train/val/test splits. When False, fall back to a
    # deterministic hash-based split (used for combined / synthetic corpora).
    use_native_splits: bool = True
    split: SplitConfig = Field(default_factory=SplitConfig)


_T = TypeVar("_T", bound=BaseModel)


def load_config(model: type[_T], path: str | Path) -> _T:
    """Load a YAML file and validate it against ``model``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(raw).__name__}")
    return model.model_validate(raw)


def load_teacher(path: str | Path | None = None) -> TeacherConfig:
    return load_config(TeacherConfig, path or CONFIGS_DIR / "teacher.yaml")


def load_student(path: str | Path | None = None) -> StudentConfig:
    return load_config(StudentConfig, path or CONFIGS_DIR / "student.yaml")


def load_distill(path: str | Path | None = None) -> DistillConfig:
    return load_config(DistillConfig, path or CONFIGS_DIR / "distill.yaml")


def load_data(path: str | Path | None = None) -> DataConfig:
    return load_config(DataConfig, path or CONFIGS_DIR / "data.yaml")


def config_hash(*configs: BaseModel) -> str:
    """Stable short hash of one or more configs.

    Used to make heavy steps (teacher labeling, training) idempotent and to
    version their outputs (SPEC §8).
    """
    payload = json.dumps(
        [c.model_dump(mode="json") for c in configs], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
