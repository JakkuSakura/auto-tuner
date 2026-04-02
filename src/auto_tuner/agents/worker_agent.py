from __future__ import annotations

import platform
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

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob: ...


@dataclass(frozen=True)
class TrainingWorkerAgent:
    requested_backend: str
    resolved_backend: str
    _backend: TrainingBackend

    @staticmethod
    def from_requested_backend(requested_backend: str) -> TrainingWorkerAgent:
        resolved_backend = _resolve_backend_name(requested_backend)
        backend = _select_backend(resolved_backend)
        return TrainingWorkerAgent(
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
            _backend=backend,
        )

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        self._backend.validate()
        return self._backend.train(dataset_path, spec)

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

