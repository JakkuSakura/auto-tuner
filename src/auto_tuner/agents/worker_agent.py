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

_UNSLOTH_MODEL_CACHE: dict[str, tuple[object, object]] = {}
_MLX_TUNE_MODEL_CACHE: dict[str, tuple[object, object]] = {}


class WorkerAgent(Protocol):
    requested_backend: str
    resolved_backend: str

    def generate_refined_solution(
        self,
        *,
        workspace_dir: Path,
        task_path: Path,
        base_model_name: str,
        adapter_dir: Path,
        max_new_tokens: int = 512,
    ) -> str: ...

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob: ...


@dataclass(frozen=True)
class TrainingWorkerAgent:
    requested_backend: str
    resolved_backend: str
    _backend: TrainingBackend

    @staticmethod
    def from_requested_backend(
        requested_backend: str, *, method: str
    ) -> TrainingWorkerAgent:
        resolved_backend = _resolve_backend_name(requested_backend, method=method)
        backend = _select_backend(resolved_backend)
        return TrainingWorkerAgent(
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
            _backend=backend,
        )

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        self._backend.validate()
        return self._backend.train(dataset_path, spec)

    def generate_refined_solution(
        self,
        *,
        workspace_dir: Path,
        task_path: Path,
        base_model_name: str,
        adapter_dir: Path,
        max_new_tokens: int = 512,
    ) -> str:
        task_markdown = task_path.read_text()
        backend_name = self._backend.name
        if backend_name == "fake":
            raise RuntimeError("Refined solution generation is not available for backend='fake'.")
        if backend_name == "unsloth":
            return _unsloth_refined_solution(
                workspace_dir=workspace_dir,
                base_model_name=base_model_name,
                adapter_dir=adapter_dir,
                task_markdown=task_markdown,
                max_new_tokens=max_new_tokens,
            )
        if backend_name == "mlx_tune":
            return _mlx_tune_refined_solution(
                workspace_dir=workspace_dir,
                base_model_name=base_model_name,
                adapter_dir=adapter_dir,
                task_markdown=task_markdown,
                max_new_tokens=max_new_tokens,
            )
        raise RuntimeError(f"Unsupported backend for refined generation: {backend_name}")

    @property
    def backend_name(self) -> str:
        return self._backend.name


def _resolve_backend_name(name: str, *, method: str) -> str:
    if name != "auto":
        return name

    if method == "grpo":
        if _is_backend_available(UnslothTrainingBackend()):
            return "unsloth"
        raise RuntimeError(
            "No compatible GRPO backend is available for backend='auto'. "
            "GRPO currently requires backend='unsloth' with the 'unsloth' extra installed "
            "(uv sync --extra unsloth), and a Linux/Windows GPU environment."
        )

    # "Smart" auto mode for SFT:
    # - Prefer the native backend on the platform.
    # - If it's unavailable (optional deps not installed), try the other real backend.
    # - If neither real backend is available, fail loudly (or set backend='fake' explicitly).
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
    if name == "fake":
        return FakeTrainingBackend()
    raise RuntimeError(f"Unsupported training backend: {name}")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text)


def _require_python_file(text: str, *, raw_output_path: Path) -> str:
    extracted = _extract_python_file(text)
    if extracted is None:
        raise RuntimeError(
            "Model output did not include a usable Python file payload. "
            f"See raw output at: {raw_output_path}"
        )
    if extracted.strip() in {"...", "# ..."}:
        raise RuntimeError(
            "Model returned a placeholder ('...') instead of a real file. "
            f"See raw output at: {raw_output_path}"
        )
    return extracted


_PYTHON_FENCE_STRICT = re.compile(
    r"```(?:python|py)\s*\n(?P<body>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
_PYTHON_FENCE_UNTERMINATED = re.compile(
    r"```(?:python|py)\s*\n(?P<body>.*)\Z",
    re.DOTALL | re.IGNORECASE,
)


def _extract_python_file(text: str) -> str | None:
    strict = _PYTHON_FENCE_STRICT.search(text)
    if strict:
        body = strict["body"].strip()
        if body:
            return body + "\n"

    unterminated = _PYTHON_FENCE_UNTERMINATED.search(text)
    if unterminated:
        body = unterminated["body"].strip()
        if body:
            return body + "\n"

    if "```" not in text:
        candidate = text.strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        looks_like_python = any(
            token in lowered
            for token in (
                "import ",
                "from ",
                "def ",
                "class ",
                "if __name__",
                "dataclass",
            )
        )
        if looks_like_python:
            return candidate + "\n"

    return None


def _build_refined_generation_prompt(task_markdown: str) -> str:
    return "\n".join(
        [
            "You are generating a refined high-quality solution to a Python refactoring task.",
            "Follow the task requirements.",
            "",
            "Task:",
            task_markdown.strip(),
            "",
            "Return exactly 1 fenced code block with the full refined_solution.py file.",
            "The code block must start with ```python and end with ``` (closing fence required).",
            "Do not use placeholders like '...'.",
            "",
        ]
    )


def _unsloth_refined_solution(
    *,
    workspace_dir: Path,
    base_model_name: str,
    adapter_dir: Path,
    task_markdown: str,
    max_new_tokens: int,
) -> str:
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Unsloth refined generation requires the optional 'unsloth' dependency group. "
            "Install with: uv sync --extra unsloth"
        ) from exc

    prompt = _build_refined_generation_prompt(task_markdown)
    _write_text(workspace_dir / "refined_generation_prompt.md", prompt)

    try:  # pragma: no cover - live path only
        cache_key = str(adapter_dir)
        cached = _UNSLOTH_MODEL_CACHE.get(cache_key)
        if cached is None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            try:
                model.load_adapter(str(adapter_dir))
            except Exception:
                pass
            _UNSLOTH_MODEL_CACHE[cache_key] = (model, tokenizer)
        else:
            model, tokenizer = cached
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass

        encoded = tokenizer([prompt], return_tensors="pt")
        try:
            encoded = encoded.to("cuda")
        except Exception:
            pass
        output = model.generate(**encoded, max_new_tokens=max_new_tokens)
        raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_path = workspace_dir / "refined_generation_output.md"
        _write_text(raw_path, raw_output)
        return _require_python_file(raw_output, raw_output_path=raw_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Unsloth refined generation failed: {exc}") from exc


def _mlx_tune_refined_solution(
    *,
    workspace_dir: Path,
    base_model_name: str,
    adapter_dir: Path,
    task_markdown: str,
    max_new_tokens: int,
) -> str:
    try:
        from mlx_tune import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MLX-Tune refined generation requires the optional 'mlx_tune' dependency group. "
            "Install with: uv sync --extra mlx_tune"
        ) from exc

    prompt = _build_refined_generation_prompt(task_markdown)
    _write_text(workspace_dir / "refined_generation_prompt.md", prompt)
    try:  # pragma: no cover - live path only
        cache_key = f"{base_model_name}::{adapter_dir}"
        cached = _MLX_TUNE_MODEL_CACHE.get(cache_key)
        if cached is None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model.load_adapter(str(adapter_dir))
            _MLX_TUNE_MODEL_CACHE[cache_key] = (model, tokenizer)
        else:
            model, tokenizer = cached
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            pass
        output = model.generate(prompt=prompt, max_tokens=max_new_tokens)
        text = output if isinstance(output, str) else str(output)
        raw_path = workspace_dir / "refined_generation_output.md"
        _write_text(raw_path, text)
        return _require_python_file(text, raw_output_path=raw_path)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MLX-Tune refined generation failed: {exc}") from exc
