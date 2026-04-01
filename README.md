# auto-tuner

Turn one high-level tuning goal into a usable fine-tuning workflow.

`auto-tuner` takes an initial user prompt, uses OpenRouter to derive the generation and grading prompts, builds a synthetic dataset, scores and refines it, and then runs a managed training flow with a local UI, CLI, and API.

## Selling points

- **Prompt-in, pipeline-out** — start from a single meta-prompt instead of hand-authoring generation and grading prompts.
- **OpenRouter-backed prompt synthesis** — generate the input/data-generation prompt and grading prompt automatically from the user’s goal.
- **Production-shaped workflow** — CLI, FastAPI API, and SolidJS web UI all operate on the same run/artifact model.
- **Deterministic local development** — fake backend and fallback prompt generation keep tests stable without external dependencies.
- **Managed run artifacts** — every run records prompts, generated data, grades, refined data, training spec, training result, and report output.
- **Upgrade path to real training** — guarded Unsloth backend supports live execution only where the environment is compatible.
- **Operational tooling included** — list, inspect, export, download, and delete runs from the CLI or API.

## How it works

1. User provides a generic initial meta-prompt.
2. OpenRouter derives:
   - a generation prompt that includes concrete task/code examples
   - a grading prompt
3. The pipeline generates examples, grades them, refines them, and stores artifacts.
4. The training backend runs in fake mode by default or guarded Unsloth mode when enabled.
5. The UI and API expose run history, prompts, reports, and exports.

## Example meta-prompt

```text
Improve attribute access style and maintainability by encouraging direct, explicit, readable patterns over dynamic access patterns.
```

The initial goal should stay high-level and quality-oriented. Concrete code snippets should be generated later as part of the derived input/data-generation prompt, not hard-coded into the initial prompt.

## Before / after auto tuning example

### Before

```python
def read_value(obj):
    return getattr(obj, 'value')
```

### After

```python
def read_value(obj):
    return obj.value
```

## Small model examples

- `Qwen/Qwen2.5-0.5B-Instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

## Stack

- Python backend: FastAPI + Typer
- Frontend: SolidJS + Tailwind CSS + Vite
- Frontend package manager: pnpm
- Prompt generation: OpenRouter via `OPENROUTER_API_KEY`
- Training backend: fake backend by default, guarded optional `unsloth` backend
- Tests: pytest

## Quick start

### Backend

```bash
uv sync --extra dev
uv run auto-tuner run --config examples/sample_experiment.toml
uv run uvicorn auto_tuner.web.app:app --reload
```

### Frontend

```bash
pnpm --dir frontend install
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

The FastAPI app serves `frontend/dist/index.html` after the frontend build completes.

## Tests

```bash
uv run pytest
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

## Prompt generation behavior

With `OPENROUTER_API_KEY` set, the app asks OpenRouter to derive:
- the input/data-generation prompt
- the grading prompt

Without `OPENROUTER_API_KEY`, the app falls back to deterministic local prompt generation so tests remain stable.

## Run artifacts

Each run records:
- `prompts.json`
- `generated.jsonl`
- `graded.jsonl`
- `refined.jsonl`
- `training_spec.json`
- `training_result.json`
- `report.json`

## Unsloth backend

Install optional dependencies:

```bash
uv sync --extra dev --extra unsloth
AUTO_TUNER_BACKEND=unsloth uv run auto-tuner run --config examples/sample_experiment.toml
```

The Unsloth backend attempts live fine-tuning only when the environment is compatible. On unsupported environments such as local macOS, it returns a guarded unsupported result instead of trying to train.

## Run management

CLI:

```bash
uv run auto-tuner list-runs
uv run auto-tuner export-run .artifacts/runs/<run-id>
uv run auto-tuner delete-run .artifacts/runs/<run-id>
```

API:
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `DELETE /api/runs/{run_id}`
- `GET /api/runs/{run_id}/download/{name}`
- `GET /api/runs/{run_id}/export`
