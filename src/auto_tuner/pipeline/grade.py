from __future__ import annotations

from auto_tuner.config import GradingConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample, GradeResult


def grade_example(
    example: DatasetExample, config: GradingConfig, prompts: PromptBundle
) -> GradeResult:
    return GradeResult(
        passed=True,
        violations=[],
        severity="none",
        suggestion="",
        grading_prompt=prompts.grading_prompt,
    )


def grade_examples(
    examples: list[DatasetExample], config: GradingConfig, prompts: PromptBundle
) -> list[GradeResult]:
    return [grade_example(example, config, prompts) for example in examples]
