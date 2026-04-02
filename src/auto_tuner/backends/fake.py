from __future__ import annotations

import re
from pathlib import Path

from auto_tuner.models.training import TrainingJob, TrainingSpec


def _parse_parameters_b(model_name: str) -> float | None:
    match = re.search(r"(?P<size>\d+(?:\.\d+)?)\s*[Bb]\b", model_name)
    if match is None:
        return None
    try:
        return float(match.group("size"))
    except ValueError:
        return None


class FakeTrainingBackend:
    name = "fake"

    def validate(self) -> None:
        return None

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        parameters_b = _parse_parameters_b(spec.model_name)
        return TrainingJob(
            job_id=f"fake-{dataset_path.stem}",
            status="completed",
            backend=self.name,
            mode="simulated",
            summary=(
                f"Simulated fine-tune for {spec.model_name} using {dataset_path.name} "
                f"with LoRA rank {spec.lora_rank}."
            ),
            artifacts={"dataset_path": str(dataset_path), "output_dir": spec.output_dir},
            metrics={
                "train_loss": 0.01,
                "epochs": spec.num_train_epochs,
                "learning_rate": spec.learning_rate,
                "base_model_parameters_b": parameters_b if parameters_b is not None else "unknown",
            },
        )
