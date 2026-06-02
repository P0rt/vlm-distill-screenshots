---
title: "Distilling a 7B VLM into a 2B screenshot reader — on a laptop"
published: false
description: "How much quality do you lose for a 2.4× speedup? A hands-on knowledge-distillation project: Qwen2-VL-7B → 2B for UI-screenshot understanding, trained and benchmarked end-to-end on an M4 Pro."
tags: machinelearning, python, llm, distillation
canonical_url: https://github.com/P0rt/vlm-distill-screenshots
---

> Repo: <https://github.com/P0rt/vlm-distill-screenshots> · Model: <https://huggingface.co/p00rt/qwen2-vl-2b-screenshots-distill>
>
> **TL;DR.** I distilled a Qwen2‑VL‑7B "teacher" into a Qwen2‑VL‑2B student for one narrow job — describing UI screenshots — and measured the trade‑off. The 2B student runs **~2.4× faster** in **~2.4× less memory** with **3.75× fewer parameters**, and on the test split it beats the untrained 2B baseline (ROUGE‑L 0.18 vs 0.15). Everything below was trained and benchmarked on a single MacBook Pro (M4 Pro, 24 GB). This is an honest proof‑of‑concept, not a leaderboard run — and I'll flag every place that matters.

## 1. Why distill a VLM for a *narrow* domain?

The obvious objection: "just use a small VLM as‑is." Fair — Qwen2‑VL‑2B is already good. But "general small VLM" and "small VLM that's reliably good at *your* task" are different things. Distillation lets a big model's behavior on a **specific distribution** (here: mobile UI screenshots) become the small model's default, without you hand‑labelling anything.

Distilling a *vision‑language* model is also less‑travelled than the classic "distill BERT" story, and it drags in honest inference engineering: 4‑bit teachers, LoRA, quantized runtimes, memory budgets. That engineering is half the point.

The task I picked: **screenshot understanding** — given a UI screenshot, produce a one‑sentence summary plus a list of the key interface elements. Perception only; no agent, no clicking.

## 2. Setup: task, data, metrics

**Data — [Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words)** (`rootsautomation/RICO-Screen2Words`, CC‑BY‑4.0): 22,417 Android UI screenshots from RICO with human summaries (5 captions each), native train/val/test = 15,743 / 2,364 / 4,310, across 28 app categories. The human captions are short — median **7 words** — which matters later.

CC‑BY‑4.0 is a deliberate choice: it's publishable, unlike raw RICO's research‑only terms. I check licenses *before* pushing weights, not after.

**Metrics.** ROUGE‑L and BLEU against the human references, plus an optional teacher‑as‑judge score. (CIDEr is the classic captioning metric but needs corpus‑level document frequencies; I left it as a follow‑up and kept ROUGE‑L/BLEU, which are pure, deterministic, and unit‑tested.)

**The whole pipeline:**

```
download → build_dataset → teacher_label → train → eval → benchmark
```

Each stage is a typed CLI step; configs live in `configs/*.yaml`; every heavy step is resumable and versioned by a config hash.

## 3. Method

Three distillation signals are on the menu (each behind a flag, for clean ablations):

1. **Response‑based KD** — the teacher generates answers, the student learns to reproduce them. The full objective combines soft and hard targets:

   `L = α·CE(hard) + (1−α)·T²·KL(softmax(teacher/T) ‖ log_softmax(student/T))`

2. **Feature‑based** — align the student's vision features with the teacher's.
3. **Self‑distillation** — let the teacher label extra screenshots to grow the data.

The MVP I actually trained is the **hard‑target half of (1): sequence‑level distillation** — the student is fine‑tuned (LoRA) to reproduce the teacher's *text*. No teacher logits needed at train time. The soft‑KL term is implemented (`response_kd_loss`) and unit‑tested, but wiring it into training needs cached logits — that's the next step, and it's exactly the α/T ablation axis I *couldn't* run yet (more in §5).

**Teacher labeling.** Qwen2‑VL‑7B‑Instruct in 4‑bit, via MLX on Apple Silicon, labelled 200 train screenshots at **~10.2 s/screenshot** (≈34 min; ≈2.7 h projected for the full 15.7k). Zero outputs were flagged degenerate; mean target length 33.6 words. A light post‑validation normalizes whitespace and flags empty/too‑short outputs — cheap insurance against format drift. Example target:

> *"The UI screenshot shows a fitness app displaying an exercise called 'Lunges,' with a progress indicator showing 30% complete. Key interface elements include a progress bar, a figure performing the exercise, and the text 'Lunges.'"*

## 4. Experiments & the trade‑off

**Quality** (test split, ROUGE‑L / BLEU vs human refs):

| model | ROUGE‑L | BLEU |
| --- | --- | --- |
| **distilled student (2B + LoRA)** | **0.178** | 0.019 |
| untrained baseline (2B) | 0.153 | 0.018 |

The distilled student beats the untrained 2B by **+16% relative** ROUGE‑L after a short proof‑of‑concept run.

**Efficiency** (teacher 7B vs student 2B; M4 Pro, MLX, 4‑bit, 128 tokens):

| model | params (B) | latency p50 (ms) | throughput (img/s) | peak mem (GB) |
| --- | --- | --- | --- | --- |
| teacher (Qwen2‑VL‑7B) | 8.29 | 1538 | 0.63 | 5.8 |
| student (Qwen2‑VL‑2B) | 2.21 | 651 | 1.52 | 2.4 |

**The trade‑off in one line:** the 2B student is **~2.4× faster, in ~2.4× less memory, with 3.75× fewer parameters** — while staying ahead of the untrained baseline on quality. That's the nerve of the whole project: a real efficiency win at a modest, *measured* quality cost.

> ⚠️ Honesty box: "peak memory" on Apple Silicon is unified‑memory allocation, not CUDA VRAM. The quality numbers come from a **small** PoC (short training, 16–80 examples, N=16 test). Treat them as trends, not a benchmark result. The point of this project is the *method and the measurement harness* — both reproducible from the repo.

## 5. Ablations: what actually moved quality

| run | LoRA r | steps | ROUGE‑L | BLEU |
| --- | --- | --- | --- | --- |
| baseline | 8 | 0 | 0.152 | 0.017 |
| — | 8 | 40 | 0.170 | 0.018 |
| — | 8 | 80 | 0.172 | 0.020 |
| — | 16 | 40 | 0.171 | 0.019 |

1. **"Train at all" is the dominant lever** — baseline → distilled is the biggest single jump (+12% rel.).
2. **More steps help marginally**, and the gain is clearer on BLEU (exact phrasing sharpens) than on ROUGE‑L.
3. **LoRA rank is ~neutral** here — r8 ≈ r16. At this data scale, adapter *capacity* isn't the bottleneck.

The α / temperature / feature‑alignment ablations from §3 belong to the **logit‑level** KD variant, which needs cached teacher logits I haven't produced yet. I'd rather ship three honest comparisons than fabricate the others.

## 6. Inference engineering (the half that bites)

- **4‑bit teacher via MLX.** `bitsandbytes` is CUDA‑only, so on Apple Silicon the teacher runs through `mlx-vlm` with a 4‑bit checkpoint. Two backends (`mlx` / `hf`) sit behind one interface.
- **The screenshots are too tall.** Qwen2‑VL expands a big RICO screenshot into thousands of vision tokens; with a 1k context that blows up `get_rope_index` with a broadcast‑shape error. Fix: cap the visual‑token budget (`max_pixels`) so a screenshot stays well under the context window.
- **MLX training is blocked upstream.** LoRA SFT through `mlx-vlm` 0.6.0 dies in the backward pass with `Primitive::vjp not implemented for CustomKernel`. I confirmed the usual fast ops (SDPA, RMSNorm, RoPE) *do* have gradients, so it's a specific kernel — and both libraries are already on their latest release, so there's no version to bump to. So training runs on the **`hf` (transformers + peft) path on Apple MPS** instead (`PYTORCH_ENABLE_MPS_FALLBACK=1`), where it trains fine: loss **0.80 → 0.39**, and the reloaded adapter generates in the trained format. MLX stays the *inference* backend, where it's great.
- **A papercut worth naming:** recent `transformers` pulls in a Qwen2‑VL *video* processor that needs `torchvision` — easy to miss until the first run.
- **ONNX export** of a full VLM is famously finicky; per the plan I keep torch/MLX inference as the canonical path and treat ONNX as a stretch (the repo ships a merge‑and‑save export instead — a standalone student you can load anywhere).

## 7. Conclusions, limitations, future work

**Conclusion.** For a narrow perception task, sequence‑level distillation into a 2B VLM buys a clean ~2.4× speed/memory win over the 7B teacher while clearly beating the untrained baseline — and you can do the whole loop on a laptop.

**Honest limitations.** Small scale (short training, tiny eval N); narrow domain (RICO Android UI); BLEU is low for everyone because the student writes richer text than the terse human references (ROUGE‑L's LCS is the fairer lens here); and the headline efficiency numbers are MLX/4‑bit on Apple Silicon, not a server GPU.

**Future work.** Cache teacher logits and turn on the **soft‑KL** term (and finally run the α/T ablation); add **feature‑alignment**; grow data with **teacher‑labelled RICO**; a full‑scale training run on a 24 GB GPU; **grounding** (bounding boxes) and an **agent wrapper** as the natural next domain step.

## 8. Links

- Code: <https://github.com/P0rt/vlm-distill-screenshots>
- Model + card: <https://huggingface.co/p00rt/qwen2-vl-2b-screenshots-distill>
- Dataset: <https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words>

*Everything here is reproducible from the repo README — clone, `uv sync`, and follow the pipeline.*
