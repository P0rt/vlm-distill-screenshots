"""Tests for the pure captioning metrics (ROUGE-L, BLEU)."""

from __future__ import annotations

from vlm_distill.metrics import compute_caption_metrics, corpus_bleu, rouge_l, tokenize


def test_tokenize_strips_punctuation_and_lowercases() -> None:
    assert tokenize("A Login, screen!") == ["a", "login", "screen"]


def test_rouge_l_identical_is_one() -> None:
    assert rouge_l("a login screen", ["a login screen"]) == 1.0


def test_rouge_l_disjoint_is_zero() -> None:
    assert rouge_l("apple banana", ["xyz qrs"]) == 0.0


def test_rouge_l_takes_best_reference() -> None:
    score = rouge_l("a login screen", ["totally different text", "a login screen"])
    assert score == 1.0


def test_rouge_l_partial_between_zero_and_one() -> None:
    score = rouge_l("a login screen here", ["a login screen"])
    assert 0.0 < score < 1.0


def test_corpus_bleu_identical_is_high() -> None:
    preds = ["a login screen with a button"]
    refs = [["a login screen with a button"]]
    assert corpus_bleu(preds, refs) > 0.99


def test_corpus_bleu_disjoint_is_low() -> None:
    score = corpus_bleu(["alpha beta gamma delta"], [["one two three four"]])
    assert score < 0.05


def test_corpus_bleu_brevity_penalty() -> None:
    # Short hypothesis vs long reference -> brevity penalty drags it down.
    short = corpus_bleu(["a login"], [["a login screen with email and password fields"]])
    full = corpus_bleu(
        ["a login screen with email and password fields"],
        [["a login screen with email and password fields"]],
    )
    assert short < full


def test_compute_caption_metrics_shape() -> None:
    out = compute_caption_metrics(["a login screen"], [["a login screen"]])
    assert out["n"] == 1
    assert out["rougeL"] == 1.0
    assert out["bleu"] > 0.99


def test_compute_caption_metrics_empty() -> None:
    out = compute_caption_metrics([], [])
    assert out == {"rougeL": 0.0, "bleu": 0.0, "n": 0}
