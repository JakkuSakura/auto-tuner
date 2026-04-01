from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingSpec(BaseModel):
    backend: str
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    num_train_epochs: int
    per_device_train_batch_size: int
    output_dir: str
    dataset_path: str
    lora_rank: int = 16
    learning_rate: float = 2e-4


class TrainingJob(BaseModel):
    job_id: str
    status: str
    backend: str
    mode: str = "dry-run"
    summary: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
