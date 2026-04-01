from __future__ import annotations

from auto_tuner.config import GradingConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample, GradeResult


def grade_example(example: DatasetExample, config: GradingConfig, prompts: PromptBundle) -> GradeResult:
    violations = [
        pattern for pattern in config.required_forbidden_patterns if pattern in example.clean_solution
    ]
    return GradeResult(
        passed=not violations,
        violations=violations,
        severity="major" if violations else "none",
        suggestion="Use direct attribute access or explicit dictionaries." if violations else "",
        grading_prompt=prompts.grading_prompt,
    )


def grade_examples(
    examples: list[DatasetExample], config: GradingConfig, prompts: PromptBundle
) -> list[GradeResult]:
    return [grade_example(example, config, prompts) for example in examples]
