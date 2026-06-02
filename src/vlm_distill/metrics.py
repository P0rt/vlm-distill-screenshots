"""Captioning metrics for screenshot descriptions (SPEC Phase 5).

Pure stdlib implementations of multi-reference **ROUGE-L** and **BLEU** so the
metrics are deterministic, dependency-free, and unit-tested in CI. CIDEr is left
as an optional follow-up (it needs corpus-level document frequencies).

Tokenization: lowercase, drop punctuation, split on whitespace.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.sub(" ", text.lower()).split()


# --------------------------------------------------------------------------- #
# ROUGE-L (LCS-based F-measure, max over references)
# --------------------------------------------------------------------------- #
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, references: Sequence[str], *, beta: float = 1.2) -> float:
    """Sentence-level ROUGE-L F-measure, taking the best matching reference."""
    hyp = tokenize(prediction)
    best = 0.0
    for ref_text in references:
        ref = tokenize(ref_text)
        lcs = _lcs(hyp, ref)
        if lcs == 0 or not hyp or not ref:
            continue
        recall = lcs / len(ref)
        precision = lcs / len(hyp)
        denom = recall + beta**2 * precision
        if denom > 0:
            best = max(best, (1 + beta**2) * recall * precision / denom)
    return best


# --------------------------------------------------------------------------- #
# BLEU (corpus-level, up to 4-grams, with brevity penalty + epsilon smoothing)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Sequence[str], references: Sequence[Sequence[str]], *, max_n: int = 4
) -> float:
    """Corpus BLEU-``max_n`` against multiple references per prediction."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must align 1:1")
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0
    for pred, refs in zip(predictions, references, strict=True):
        hyp = tokenize(pred)
        ref_tokens = [tokenize(r) for r in refs]
        hyp_len += len(hyp)
        # brevity penalty uses the reference length closest to the hypothesis
        ref_len += min((len(r) for r in ref_tokens), key=lambda rl: (abs(rl - len(hyp)), rl))
        for n in range(1, max_n + 1):
            hyp_ng = _ngrams(hyp, n)
            totals[n - 1] += max(0, len(hyp) - n + 1)
            if not hyp_ng:
                continue
            max_ref = Counter[tuple[str, ...]]()
            for r in ref_tokens:
                rn = _ngrams(r, n)
                for g, c in rn.items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            clipped[n - 1] += sum(min(c, max_ref[g]) for g, c in hyp_ng.items())
    if hyp_len == 0:
        return 0.0
    # Geometric mean over the n-gram orders that are actually possible (skip
    # orders where the corpus is too short to contain any n-gram). A genuine
    # zero-match order is epsilon-smoothed rather than dropped.
    log_p = 0.0
    orders = 0
    for c, t in zip(clipped, totals, strict=True):
        if t == 0:
            continue
        log_p += math.log((c if c > 0 else 1e-9) / t)
        orders += 1
    if orders == 0:
        return 0.0
    geo_mean = math.exp(log_p / orders)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * geo_mean


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def compute_caption_metrics(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> dict[str, float]:
    """ROUGE-L (mean) and corpus BLEU-4 over a list of (prediction, refs)."""
    if not predictions:
        return {"rougeL": 0.0, "bleu": 0.0, "n": 0}
    rouge_scores = [rouge_l(p, r) for p, r in zip(predictions, references, strict=True)]
    return {
        "rougeL": round(sum(rouge_scores) / len(rouge_scores), 4),
        "bleu": round(corpus_bleu(predictions, references), 4),
        "n": len(predictions),
    }
