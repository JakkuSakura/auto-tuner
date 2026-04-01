from __future__ import annotations

from pathlib import Path
from typing import Protocol

from auto_tuner.models.training import TrainingJob, TrainingSpec


class TrainingBackend(Protocol):
    name: str

    def validate(self) -> None: ...

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob: ...
