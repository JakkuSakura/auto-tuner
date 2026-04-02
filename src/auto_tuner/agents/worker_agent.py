from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from auto_tuner.backends.base import TrainingBackend
from auto_tuner.backends.fake import FakeTrainingBackend
from auto_tuner.backends.mlx_tune import MlxTuneTrainingBackend
from auto_tuner.backends.unsloth_sdk import UnslothTrainingBackend
from auto_tuner.models.training import TrainingJob, TrainingSpec


class WorkerAgent(Protocol):
    requested_backend: str
    resolved_backend: str

    def generate_naive_solution(
        self,
        *,
        task_markdown: str,
        clean_solution: str,
        example_id: int,
        theme_hint: str,
        max_new_tokens: int = 512,
    ) -> str: ...

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob: ...


@dataclass(frozen=True)
class TrainingWorkerAgent:
    requested_backend: str
    resolved_backend: str
    model_name: str
    _backend: TrainingBackend

    @staticmethod
    def from_requested_backend(requested_backend: str, *, model_name: str) -> TrainingWorkerAgent:
        resolved_backend = _resolve_backend_name(requested_backend)
        backend = _select_backend(resolved_backend)
        return TrainingWorkerAgent(
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
            model_name=model_name,
            _backend=backend,
        )

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        self._backend.validate()
        return self._backend.train(dataset_path, spec)

    def generate_naive_solution(
        self,
        *,
        task_markdown: str,
        clean_solution: str,
        example_id: int,
        theme_hint: str,
        max_new_tokens: int = 512,
    ) -> str:
        backend_name = self._backend.name
        if backend_name == "fake":
            return _fake_naive_solution(
                task_markdown=task_markdown,
                clean_solution=clean_solution,
                example_id=example_id,
                theme_hint=theme_hint,
            )
        if backend_name == "unsloth":
            return _unsloth_naive_solution(
                model_name=self.model_name,
                task_markdown=task_markdown,
                clean_solution=clean_solution,
                max_new_tokens=max_new_tokens,
            )
        if backend_name == "mlx_tune":
            return _mlx_tune_naive_solution(
                model_name=self.model_name,
                task_markdown=task_markdown,
                clean_solution=clean_solution,
                max_new_tokens=max_new_tokens,
            )
        raise RuntimeError(f"Unsupported backend for naive generation: {backend_name}")

    @property
    def backend_name(self) -> str:
        return self._backend.name


def _resolve_backend_name(name: str) -> str:
    if name != "auto":
        return name

    # "Smart" auto mode:
    # - Prefer the native backend on the platform.
    # - If it's unavailable (optional deps not installed), try the other real backend.
    # - If neither real backend is available, fail loudly (or set backend="fake" explicitly).
    prefer_mlx = platform.system() == "Darwin"
    candidates = ["mlx_tune", "unsloth"] if prefer_mlx else ["unsloth", "mlx_tune"]
    for candidate in candidates:
        if candidate == "mlx_tune" and _is_backend_available(MlxTuneTrainingBackend()):
            return "mlx_tune"
        if candidate == "unsloth" and _is_backend_available(UnslothTrainingBackend()):
            return "unsloth"

    raise RuntimeError(
        "No real training backend is available for backend='auto'. "
        "Install one extra (uv sync --extra unsloth or uv sync --extra mlx_tune) "
        "or set training.backend='fake' explicitly."
    )


def _is_backend_available(backend: TrainingBackend) -> bool:
    try:
        backend.validate()
        return True
    except Exception:
        return False


def _select_backend(name: str) -> TrainingBackend:
    if name == "mlx_tune":
        return MlxTuneTrainingBackend()
    if name == "unsloth":
        return UnslothTrainingBackend()
    return FakeTrainingBackend()


def _fake_naive_solution(
    *,
    task_markdown: str,
    clean_solution: str,
    example_id: int,
    theme_hint: str,
) -> str:
    _ = task_markdown
    _ = clean_solution
    _ = example_id
    _ = theme_hint
    return "\n".join(
        [
            "from __future__ import annotations",
            "",
            "def read_value(obj):",
            "    # naive: dynamic lookup (anti-pattern)",
            "    return getattr(obj, \"value\")",
            "",
        ]
    )


def _unsloth_naive_solution(
    *,
    model_name: str,
    task_markdown: str,
    clean_solution: str,
    max_new_tokens: int,
) -> str:
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Unsloth naive generation requires the optional 'unsloth' dependency group. "
            "Install with: uv sync --extra unsloth"
        ) from exc

    try:  # pragma: no cover - live path only
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        prompt = _build_naive_generation_prompt(task_markdown, clean_solution)
        encoded = tokenizer([prompt], return_tensors="pt")
        try:
            encoded = encoded.to("cuda")
        except Exception:
            pass
        output = model.generate(**encoded, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return _extract_python_file(text) or text.strip() + "\n"
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unsloth naive generation failed: {exc}") from exc


def _mlx_tune_naive_solution(
    *,
    model_name: str,
    task_markdown: str,
    clean_solution: str,
    max_new_tokens: int,
) -> str:
    try:
        from mlx_tune import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MLX-Tune naive generation requires the optional 'mlx_tune' dependency group. "
            "Install with: uv sync --extra mlx_tune"
        ) from exc

    try:  # pragma: no cover - live path only
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        prompt = _build_naive_generation_prompt(task_markdown, clean_solution)

        # mlx-tune reuses an Unsloth-like API, but inference details can vary across forks.
        # We try a minimal `generate` path first; otherwise fail loudly.
        try:
            output = model.generate(prompt, max_new_tokens=max_new_tokens)
            if isinstance(output, str):
                return _extract_python_file(output) or output.strip() + "\n"
        except Exception:
            pass

        raise RuntimeError(
            "mlx_tune FastLanguageModel inference API is not compatible with this worker. "
            "Please update the worker inference implementation for your mlx_tune fork."
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MLX-Tune naive generation failed: {exc}") from exc


_PYTHON_FENCE = re.compile(r"```python\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_python_file(text: str) -> str | None:
    match = _PYTHON_FENCE.search(text)
    if not match:
        return None
    body = match["body"].strip()
    if not body:
        return None
    return body + "\n"


def _build_naive_generation_prompt(task_markdown: str, clean_solution: str) -> str:
    return "\n".join(
        [
            "You are generating a naive baseline solution to a Python refactoring task.",
            "The naive baseline should intentionally use dynamic/reflection-style access patterns.",
            "",
            "Task:",
            task_markdown.strip(),
            "",
            "Reference clean solution (do NOT copy it verbatim):",
            "```python",
            clean_solution.rstrip(),
            "```",
            "",
            "Return a single fenced code block with the full naive_solution.py file:",
            "```python",
            "...",
            "```",
            "",
        ]
    )
