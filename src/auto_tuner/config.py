from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "YAML config support requires PyYAML. Install with: uv add pyyaml"
        ) from exc
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML config root must be a mapping")
    return loaded


class AppConfig(BaseModel):
    artifacts_dir: Path = Path(".artifacts")
    frontend_dist: Path = Path("frontend/dist")


class OpenRouterConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    prompt_model: str = "z-ai/glm-5"
    grading_model: str = "z-ai/glm-5"
    http_referer: str = "http://localhost"
    app_name: str = "auto-tuner"


class GenerationConfig(BaseModel):
    sample_count: int = 3
    meta_prompt: str = (
        "Improve Python code quality by encouraging direct, explicit, readable access patterns. "
        "Avoid reflection/introspection-driven access and other dynamic lookup patterns. "
        "Prefer straightforward attribute and mapping access that is easy to review and type-check."
    )

    @field_validator("sample_count")
    @classmethod
    def validate_sample_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sample_count must be >= 1")
        return value


class GradingConfig(BaseModel):
    max_retries: int = 2


class TrainingConfig(BaseModel):
    backend: str = "auto"
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
        supported = {"auto", "fake", "unsloth", "mlx_tune"}
        if value not in supported:
            raise ValueError(f"backend must be one of {sorted(supported)}")
        return value


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_settings(config_path: str | Path | None = None) -> Settings:
    candidate = Path(
        config_path or os.getenv("AUTO_TUNER_CONFIG", "examples/sample_experiment.yaml")
    )
    data: dict[str, object] = {}
    if candidate.exists():
        if candidate.suffix in {".yaml", ".yml"}:
            data = _load_yaml(candidate)
        else:
            data = tomllib.loads(candidate.read_text())

    settings = Settings.model_validate(data)

    if artifacts_dir := os.getenv("AUTO_TUNER_ARTIFACTS_DIR"):
        settings.app.artifacts_dir = Path(artifacts_dir)
    if frontend_dist := os.getenv("AUTO_TUNER_FRONTEND_DIST"):
        settings.app.frontend_dist = Path(frontend_dist)
    if openrouter_key := os.getenv("OPENROUTER_API_KEY"):
        settings.openrouter.api_key = openrouter_key

    return settings
