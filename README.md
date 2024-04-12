# vlm-distill-screenshots

Knowledge distillation of a **Qwen2-VL** teacher into a compact student for
**screenshot understanding** (short UI description + key-element listing).

The project measures not just task quality but **latency / throughput / memory** —
the central question is *how much quality we trade for how much speedup*. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for working
conventions.

> **Status:** Phase 1 (skeleton & infra) complete. Phases 2–8 are scaffolded —
> every entrypoint exists and raises a clear "not implemented yet" pointing at
> the phase that will land it.

## Models

| Role    | Default model              | Notes                                   |
| ------- | -------------------------- | --------------------------------------- |
| Teacher | `Qwen/Qwen2-VL-7B-Instruct` | loaded in 4-bit (≈24GB GPU)            |
| Student | `Qwen/Qwen2-VL-2B-Instruct` | LoRA fine-tune (~3.5× fewer params)     |

Backup student architecture (frozen SigLIP vision + Qwen2.5-0.5B decoder) is
configurable in `configs/student.yaml`.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```bash
# Dev environment (matches CI: lint + type + test, no heavy ML deps)
uv sync
uv run pytest
uv run ruff check . && uv run mypy src

# Add the heavy ML stack (torch/transformers/...). Needs a GPU box for real runs.
uv sync --extra ml
```

Copy `.env.example` to `.env` for W&B / Hugging Face tokens.

## Pipeline

```text
download → build_dataset → teacher_label → train → eval → benchmark → export
```

Each step has a thin CLI wrapper in `scripts/` and a console entrypoint:

```bash
uv run vlm-download          # Phase 2  — Screen2Words + RICO
uv run vlm-build-dataset     # Phase 2  — unified {image, prompt, target} + split
uv run vlm-teacher-label     # Phase 3  — teacher targets (+ optional top-k logits)
uv run vlm-train             # Phase 4  — distillation (LoRA + accelerate), --dry-run for CPU
uv run vlm-eval              # Phase 5  — CIDEr / ROUGE-L / BLEU + LLM-as-judge
uv run vlm-benchmark         # Phase 6  — latency / throughput / peak VRAM
uv run vlm-export            # Phase 8  — ONNX export + sanity check
```

## Configuration

All hyperparameters and paths live in `configs/*.yaml`, validated by typed
pydantic models in `src/vlm_distill/config.py`:

- `teacher.yaml` — teacher model, 4-bit, generation, visual-token budget
- `student.yaml` — student model, LoRA, fallback architecture
- `distill.yaml` — α, temperature, loss flags (response / feature / synthetic)
- `data.yaml` — datasets, prompt template, deterministic split

## Repo layout

Source under `src/vlm_distill/`, thin CLIs under
`scripts/`, tests under `tests/`, run outputs under `results/` (gitignored).

## License

Apache-2.0. Dataset and teacher-weight licenses are tracked in the model card
before any weights are published.
