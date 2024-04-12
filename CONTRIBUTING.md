# Development conventions

Short operational checklist for working in this repo.

## Workflow

- **One PR = one phase**. Open the PR with a short plan; close it
  with a checklist against that phase's acceptance criteria. Do not start the
  next phase until the current one's acceptance criteria are met.

## Code rules

- Every new module in `src/`: full type hints, passes `mypy --strict`.
- All hyperparameters and paths come from `configs/*.yaml`. **No hardcoded
  values** in code — add a field to the relevant pydantic model in
  `src/vlm_distill/config.py` instead.
- Heavy steps (teacher labeling, training) must be **resumable** and
  **idempotent by config hash** (`config_hash` in `config.py`).
- Anything with randomness fixes a seed; splits are deterministic.
- Heavy ML deps (`torch`, `transformers`, …) live in optional extras and are
  imported **lazily inside functions** (or under `TYPE_CHECKING`) so the package
  imports — and CI runs — without them.
- Any GPU step gets a `--dry-run` / CPU path so CI stays cheap.

## Artifacts & secrets

- Never commit weights or datasets. Commit paths + download instructions only.
  Large outputs go to `results/` (gitignored) or the HF Hub.
- Secrets (W&B / HF tokens) via env vars; document in `.env.example`, never in
  the repo.

## Before every commit

```bash
uv run ruff check . && uv run ruff format && uv run mypy src && uv run pytest
```

## Local setup

```bash
uv sync                  # core + dev tooling (what CI runs)
uv sync --extra data     # + data stack (datasets/pillow/pyarrow)
uv sync --extra mlx      # + MLX backend (Apple Silicon)
uv sync --extra ml       # + heavy CUDA stack (needs a GPU box)
```
