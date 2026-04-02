from __future__ import annotations

from auto_tuner.config import GradingConfig
from auto_tuner.agents.supervisor_agent import OpenRouterSupervisorAgent
from auto_tuner.config import OpenRouterConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample
from auto_tuner.pipeline.grade import grade_example


def test_grade_example_uses_supervisor(monkeypatch) -> None:
    from tests.support.openrouter_stub import install_openrouter_stub

    install_openrouter_stub(monkeypatch)
    supervisor = OpenRouterSupervisorAgent(OpenRouterConfig(api_key="test"))
    example = DatasetExample(
        task="task",
        naive_solution="code",
    )
    prompts = PromptBundle(
        meta_prompt="goal",
        generation_prompt="generation",
        grading_prompt="grading rubric",
        source="openrouter",
    )

    result = grade_example(example, GradingConfig(), prompts, supervisor)

    assert result.passed is True
    assert result.violations == []
    assert result.grading_prompt == "grading rubric"
