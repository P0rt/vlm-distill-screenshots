"""Normalize raw Screen2Words examples into the unified record schema and
compute dataset statistics.

The unified record (text fields only; the image is carried alongside by
``build_dataset`` when writing to disk):

    {
        "id":         str,         # RICO screenId
        "prompt":     str,         # task prompt (from configs/data.yaml)
        "target":     str,         # chosen human caption (teacher overwrites in Phase 3)
        "references": list[str],   # all human captions, for eval metrics
        "category":   str,         # app category
        "source":     str,         # provenance tag
    }

Pure stdlib (uses ``statistics`` rather than numpy) so it is importable and
testable in CI without heavy deps.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

SOURCE_SCREEN2WORDS = "screen2words"


def _clean_captions(raw_captions: Any, max_references: int) -> list[str]:
    """Coerce the raw ``captions`` field into a clean, bounded list of strings."""
    if not isinstance(raw_captions, Sequence) or isinstance(raw_captions, str | bytes):
        return []
    out: list[str] = []
    for cap in raw_captions:
        if isinstance(cap, str):
            stripped = cap.strip()
            if stripped:
                out.append(stripped)
    return out[:max_references]


def build_record(
    raw: Mapping[str, Any],
    *,
    prompt: str,
    max_references: int = 5,
    source: str = SOURCE_SCREEN2WORDS,
) -> dict[str, Any]:
    """Build one unified text record from a raw Screen2Words example.

    Does not touch the image column — ``build_dataset`` carries it through.
    """
    references = _clean_captions(raw.get("captions"), max_references)
    target = references[0] if references else ""
    screen_id = raw.get("screenId", raw.get("id", ""))
    category = raw.get("category", "")
    return {
        "id": str(screen_id),
        "prompt": prompt,
        "target": target,
        "references": references,
        "category": str(category) if category is not None else "",
        "source": source,
    }


def _length_stats(values: Sequence[int]) -> dict[str, float]:
    """Min / mean / median / p10 / p90 / max for a sequence of lengths."""
    if not values:
        return {"min": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "max": 0}
    ordered = sorted(values)

    def _percentile(p: float) -> float:
        # Nearest-rank percentile (deterministic, no interpolation surprises).
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        return float(ordered[idx])

    return {
        "min": float(ordered[0]),
        "mean": round(statistics.fmean(ordered), 2),
        "median": float(statistics.median(ordered)),
        "p10": _percentile(10),
        "p90": _percentile(90),
        "max": float(ordered[-1]),
    }


def compute_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate stats over unified records: counts, splits, length distributions."""
    target_word_lens = [len(str(r.get("target", "")).split()) for r in records]
    target_char_lens = [len(str(r.get("target", ""))) for r in records]
    n_refs = [len(r.get("references", []) or []) for r in records]
    split_counts = Counter(str(r["split"]) for r in records if "split" in r)
    category_counts = Counter(str(r.get("category", "")) for r in records)

    return {
        "num_examples": len(records),
        "splits": dict(sorted(split_counts.items())),
        "num_categories": len([c for c in category_counts if c]),
        "target_words": _length_stats(target_word_lens),
        "target_chars": _length_stats(target_char_lens),
        "references_per_example": _length_stats(n_refs),
    }


def render_stats_markdown(stats: Mapping[str, Any]) -> str:
    """Render a compact Markdown block from ``compute_stats`` output."""
    lines: list[str] = []
    lines.append(f"- **Examples:** {stats['num_examples']:,}")
    if stats.get("splits"):
        split_str = ", ".join(f"{k} {v:,}" for k, v in stats["splits"].items())
        lines.append(f"- **Splits:** {split_str}")
    lines.append(f"- **App categories:** {stats.get('num_categories', 0)}")
    tw = stats["target_words"]
    lines.append(
        f"- **Target length (words):** min {tw['min']:.0f} / "
        f"median {tw['median']:.0f} / mean {tw['mean']:.1f} / "
        f"p90 {tw['p90']:.0f} / max {tw['max']:.0f}"
    )
    rp = stats["references_per_example"]
    lines.append(f"- **References per example:** median {rp['median']:.0f}, max {rp['max']:.0f}")
    return "\n".join(lines)
