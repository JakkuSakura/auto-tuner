from __future__ import annotations

from auto_tuner.config import GradingConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample
from auto_tuner.pipeline.grade import grade_example


def test_grade_example_flags_dynamic_patterns() -> None:
    example = DatasetExample(
        task="task",
        naive_solution="code",
        clean_solution="return getattr(obj, 'value')",
    )
    prompts = PromptBundle(
        auto_tune_prompt="goal",
        generation_prompt="generation",
        grading_prompt="grading rubric",
        source="fallback",
    )

    result = grade_example(example, GradingConfig(), prompts)

    assert result.passed is False
    assert "getattr(" in result.violations
    assert result.grading_prompt == "grading rubric"
