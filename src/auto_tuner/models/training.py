from __future__ import annotations

from pydantic import BaseModel, Field


class GrpoJudgeSpec(BaseModel):
    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "z-ai/glm-5"
    http_referer: str = "http://localhost"
    app_name: str = "auto-tuner"
    system_prompt: str = (
        "You are a strict coding-quality judge. "
        "Score the student's answer only by the provided rubric."
    )
    user_prompt_template: str = (
        "Task:\n{prompt}\n\nStudent answer:\n{completion}\n\n"
        "Give a score in [0, 1]. Output only the number."
    )
    timeout_seconds: float = 60.0


class GrpoRewardRules(BaseModel):
    require_python_fence: bool = False
    forbidden_substrings: list[str] = Field(default_factory=list)
    forbidden_substring_penalty: float = 0.25


class GrpoSpec(BaseModel):
    num_generations: int = 8
    max_prompt_length: int = 256
    max_completion_length: int = 512
    use_vllm: bool = False
    judge: GrpoJudgeSpec = Field(default_factory=GrpoJudgeSpec)
    rules: GrpoRewardRules = Field(default_factory=GrpoRewardRules)


class TrainingSpec(BaseModel):
    backend: str
    method: str = "sft"
    model_name: str
    max_seq_length: int
    load_in_4bit: bool
    num_train_epochs: int
    per_device_train_batch_size: int
    output_dir: str
    dataset_path: str
    lora_rank: int = 16
    learning_rate: float = 2e-4
    grpo: GrpoSpec | None = None


class TrainingJob(BaseModel):
    job_id: str
    status: str
    backend: str
    mode: str = "dry-run"
    summary: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
