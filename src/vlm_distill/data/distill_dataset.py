"""Build the distillation training set: join teacher targets with their RICO
screenshots into ``{image, question, answer}`` records (SPEC Phase 4).

This is the input the student trains on (sequence-level / response-based KD:
the student learns to reproduce the teacher's text). The ``question`` is the
task prompt; the ``answer`` is the teacher target.

``datasets`` / ``PIL`` are imported lazily; the pure helpers (``to_messages``,
``read_labels``) need no heavy deps and run in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_distill.config import REPO_ROOT, DataConfig, TeacherConfig, config_hash
from vlm_distill.data.teacher_label import output_dir_for

if TYPE_CHECKING:
    from collections.abc import Iterator

DISTILL_ROOT = REPO_ROOT / "data" / "distill"


def to_messages(question: str, answer: str) -> list[dict[str, str]]:
    """Single-turn chat messages, matching mlx-vlm's SFT expectation."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def read_labels(jsonl_path: Path, *, valid_only: bool = True) -> dict[str, str]:
    """Read teacher labels into ``{id: teacher_target}``."""
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"No teacher labels at {jsonl_path}. Run `vlm-teacher-label` first."
        )
    labels: dict[str, str] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if valid_only and not rec.get("valid", True):
            continue
        labels[str(rec["id"])] = str(rec["teacher_target"])
    return labels


def labels_path_for(data_cfg: DataConfig, teacher_cfg: TeacherConfig) -> Path:
    """Locate the teacher-label JSONL for the given configs."""
    cfg_hash = config_hash(data_cfg, teacher_cfg)
    return output_dir_for(cfg_hash, dry_run=False) / "train.jsonl"


def synthetic_distill_records(n: int = 4) -> list[dict[str, Any]]:
    """Image-free synthetic records for --dry-run / CI graph validation."""
    samples = [
        ("a login screen", "A login screen with email and password fields and a sign-in button."),
        ("a settings page", "A settings page with several toggle switches and a save button."),
        ("a news feed", "A news feed listing article cards with thumbnails and headlines."),
        ("a map view", "A map view with a search bar, zoom controls, and a location pin."),
    ]
    out: list[dict[str, Any]] = []
    for i in range(n):
        q, a = samples[i % len(samples)]
        out.append({"id": f"dry-{i}", "question": q, "answer": a, "image": None})
    return out


def _iter_join(
    data_cfg: DataConfig, labels: dict[str, str], *, limit: int | None
) -> Iterator[dict[str, Any]]:
    """Stream the raw train split, yielding records for labeled screens only."""
    from datasets import load_dataset

    stream = load_dataset(
        data_cfg.dataset_id, split="train", streaming=True, cache_dir=data_cfg.raw_dir
    )
    n = 0
    for example in stream:
        if limit is not None and n >= limit:
            break
        screen_id = str(example.get("screenId", ""))
        answer = labels.get(screen_id)
        if answer is None:
            continue
        n += 1
        yield {
            "id": screen_id,
            "image": example["image"],
            "question": data_cfg.prompt_template,
            "answer": answer,
        }


def build_distill_dataset(
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    labels_path: Path | None = None,
    limit: int | None = None,
) -> tuple[Path, int]:
    """Materialize the {image, question, answer} train set; returns (dir, count)."""
    from datasets import Dataset
    from datasets import Image as HfImage

    jsonl = labels_path or labels_path_for(data_cfg, teacher_cfg)
    labels = read_labels(jsonl)

    rows = list(_iter_join(data_cfg, labels, limit=limit))
    if not rows:
        raise ValueError(f"No labeled screens joined from {jsonl} (labels={len(labels)}).")

    ds = Dataset.from_dict(
        {
            "image": [r["image"] for r in rows],
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
        }
    ).cast_column("image", HfImage())

    out_dir = DISTILL_ROOT / config_hash(data_cfg, teacher_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    return out_dir, len(rows)
