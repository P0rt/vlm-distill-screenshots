"""Generate teacher targets for the train split, versioned by config hash and
resumable / idempotent (SPEC Phase 3).

Output layout (under ``results/``, gitignored):

    results/teacher_labels/<config_hash>/
        train.jsonl     # one JSON record per screen: {id, prompt, teacher_target, ...}
        manifest.json   # config hash, backend, model, counts, timing

Re-running resumes: already-labeled ids are skipped, so an interrupted run
continues where it left off and a completed run is a no-op.

``datasets`` is imported lazily; ``--dry-run`` needs no heavy deps and runs in CI.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vlm_distill.cli import build_parser
from vlm_distill.config import (
    REPO_ROOT,
    DataConfig,
    TeacherConfig,
    config_hash,
    load_data,
    load_teacher,
)
from vlm_distill.models.teacher import make_teacher

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL.Image import Image

LABELS_ROOT = REPO_ROOT / "results" / "teacher_labels"


def output_dir_for(cfg_hash: str, *, dry_run: bool) -> Path:
    """Versioned output directory for a labeling run."""
    return LABELS_ROOT / ("dryrun" if dry_run else cfg_hash)


def load_done_ids(jsonl_path: Path) -> set[str]:
    """Read ids already labeled in ``train.jsonl`` (for resume)."""
    if not jsonl_path.exists():
        return set()
    done: set[str] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(str(json.loads(line)["id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


_WHITESPACE_RE = re.compile(r"\s+")


def clean_and_validate_target(text: str, *, min_words: int = 3) -> tuple[str, bool]:
    """Light post-validation of a teacher target (SPEC §9 mitigation).

    Collapses whitespace/newlines into single spaces and flags degenerate
    outputs (empty / too short) so downstream training can filter them.
    Returns ``(cleaned_text, is_valid)``.
    """
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    is_valid = len(cleaned.split()) >= min_words
    return cleaned, is_valid


def _synthetic_examples(prompt: str, n: int) -> Iterator[tuple[str, Image | None, str]]:
    for i in range(n):
        yield f"dry-{i}", None, prompt


def _iter_examples(
    cfg: DataConfig, *, limit: int | None, dry_run: bool
) -> Iterator[tuple[str, Image | None, str]]:
    """Yield ``(id, image, prompt)`` for the train split."""
    if dry_run:
        yield from _synthetic_examples(cfg.prompt_template, limit or 4)
        return
    from datasets import load_dataset

    # Stream so we never hold the whole split (or all images) in memory.
    stream = load_dataset(cfg.dataset_id, split="train", streaming=True, cache_dir=cfg.raw_dir)
    for i, example in enumerate(stream):
        if limit is not None and i >= limit:
            break
        screen_id = str(example.get("screenId", i))
        yield screen_id, example["image"], cfg.prompt_template


def _model_label(teacher_cfg: TeacherConfig, *, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    return teacher_cfg.mlx_model_id if teacher_cfg.backend == "mlx" else teacher_cfg.model_id


def run_labeling(
    data_cfg: DataConfig,
    teacher_cfg: TeacherConfig,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Label the train split with the teacher. Idempotent + resumable."""
    cfg_hash = config_hash(data_cfg, teacher_cfg)
    out_dir = output_dir or output_dir_for(cfg_hash, dry_run=dry_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"

    teacher = make_teacher(teacher_cfg, dry_run=dry_run)
    done = load_done_ids(jsonl_path)
    model_label = _model_label(teacher_cfg, dry_run=dry_run)

    n_new, n_skipped, n_invalid = 0, 0, 0
    start = time.perf_counter()
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for screen_id, image, prompt in _iter_examples(data_cfg, limit=limit, dry_run=dry_run):
            if screen_id in done:
                n_skipped += 1
                continue
            raw = teacher.describe(image, prompt)
            target, is_valid = clean_and_validate_target(raw)
            if not is_valid:
                n_invalid += 1
            record = {
                "id": screen_id,
                "prompt": prompt,
                "teacher_target": target,
                "valid": is_valid,
                "words": len(target.split()),
                "model": model_label,
                "source": "screen2words",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(screen_id)
            n_new += 1
    elapsed = time.perf_counter() - start

    summary = {
        "config_hash": cfg_hash,
        "backend": "dry-run" if dry_run else teacher_cfg.backend,
        "model": model_label,
        "prompt_template": data_cfg.prompt_template,
        "limit": limit,
        "dry_run": dry_run,
        "num_labeled_total": len(done),
        "num_new_this_run": n_new,
        "num_skipped": n_skipped,
        "num_invalid_this_run": n_invalid,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_example": round(elapsed / n_new, 3) if n_new else None,
        "output": str(jsonl_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def teacher_label(
    data_cfg: DataConfig, teacher_cfg: TeacherConfig, *, dry_run: bool = False
) -> dict[str, Any]:
    """Convenience wrapper used by tests / external callers."""
    return run_labeling(data_cfg, teacher_cfg, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("vlm-teacher-label", "Label screenshots with the teacher VLM.")
    parser.add_argument(
        "--teacher-config", type=str, default=None, help="Path to teacher.yaml override."
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap examples to label.")
    ns = parser.parse_args(argv)
    data_cfg = load_data(ns.config)
    teacher_cfg = load_teacher(ns.teacher_config)
    summary = run_labeling(data_cfg, teacher_cfg, limit=ns.limit, dry_run=bool(ns.dry_run))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
