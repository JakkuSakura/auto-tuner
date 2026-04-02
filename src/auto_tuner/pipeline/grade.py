from __future__ import annotations

from auto_tuner.config import GradingConfig
from auto_tuner.agents.supervisor_agent import SupervisorAgent
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample, GradeResult


def grade_example(
    example: DatasetExample,
    config: GradingConfig,
    prompts: PromptBundle,
    supervisor: SupervisorAgent,
) -> GradeResult:
    return supervisor.grade_example(
        meta_prompt=prompts.meta_prompt,
        grading_prompt=prompts.grading_prompt,
        task=example.task,
        naive_solution=example.naive_solution,
    )


def grade_examples(
    examples: list[DatasetExample],
    config: GradingConfig,
    prompts: PromptBundle,
    supervisor: SupervisorAgent,
) -> list[GradeResult]:
    results: list[GradeResult] = []
    for example in examples:
        last_exc: Exception | None = None
        for _attempt in range(0, config.max_retries + 1):
            try:
                results.append(grade_example(example, config, prompts, supervisor))
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(f"Grading failed after retries: {last_exc}") from last_exc
    return results
