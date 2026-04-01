from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    artifacts_dir: Path = Path(".artifacts")
    frontend_dist: Path = Path("frontend/dist")


class OpenRouterConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    prompt_model: str = "openai/gpt-4o-mini"
    grading_model: str = "openai/gpt-4o-mini"
    http_referer: str = "http://localhost"
    app_name: str = "auto-tuner"


class GenerationConfig(BaseModel):
    sample_count: int = 3
    meta_prompt: str = (
        "Improve attribute access style and maintainability by encouraging direct, explicit, readable patterns over dynamic access patterns."
    )

    @field_validator("sample_count")
    @classmethod
    def validate_sample_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sample_count must be >= 1")
        return value


class GradingConfig(BaseModel):
    max_retries: int = 2
    required_forbidden_patterns: list[str] = Field(
        default_factory=lambda: ["getattr(", "hasattr(", ".__dict__", "vars("]
    )


class TrainingConfig(BaseModel):
    backend: str = "fake"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    output_dir: str = "./output"
    lora_rank: int = 16
    learning_rate: float = 2e-4

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        supported = {"fake", "unsloth"}
        if value not in supported:
            raise ValueError(f"backend must be one of {sorted(supported)}")
        return value


class DemoConfig(BaseModel):
    enabled: bool = True
    example_models: list[str] = Field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]
    )


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)


def load_settings(config_path: str | Path | None = None) -> Settings:
    candidate = Path(
        config_path or os.getenv("AUTO_TUNER_CONFIG", "examples/sample_experiment.toml")
    )
    data: dict[str, object] = {}
    if candidate.exists():
        data = tomllib.loads(candidate.read_text())

    settings = Settings.model_validate(data)

    if artifacts_dir := os.getenv("AUTO_TUNER_ARTIFACTS_DIR"):
        settings.app.artifacts_dir = Path(artifacts_dir)
    if frontend_dist := os.getenv("AUTO_TUNER_FRONTEND_DIST"):
        settings.app.frontend_dist = Path(frontend_dist)
    if openrouter_key := os.getenv("OPENROUTER_API_KEY"):
        settings.openrouter.api_key = openrouter_key

    return settings
