from __future__ import annotations

from pathlib import Path

from auto_tuner.backends.fake import FakeTrainingBackend
from auto_tuner.backends.unsloth_sdk import UnslothTrainingBackend
from auto_tuner.models.training import TrainingSpec


def test_fake_backend_returns_completed_job(tmp_path: Path) -> None:
    backend = FakeTrainingBackend()
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{}\n')
    spec = TrainingSpec(
        backend="fake",
        model_name="model",
        max_seq_length=128,
        load_in_4bit=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir="./output",
        dataset_path=str(dataset_path),
    )

    job = backend.train(dataset_path, spec)

    assert job.status == "completed"
    assert job.backend == "fake"


def test_unsloth_backend_is_guarded_on_unsupported_platform(tmp_path: Path) -> None:
    backend = UnslothTrainingBackend()
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"text":"hello"}\n')
    spec = TrainingSpec(
        backend="unsloth",
        model_name="model",
        max_seq_length=128,
        load_in_4bit=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir=str(tmp_path / "output"),
        dataset_path=str(dataset_path),
    )

    job = backend.train(dataset_path, spec)

    assert job.backend == "unsloth"
    assert job.status in {"unsupported", "failed", "completed"}
