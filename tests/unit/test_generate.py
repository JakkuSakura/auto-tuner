from __future__ import annotations

from pathlib import Path

from auto_tuner.config import GenerationConfig, OpenRouterConfig
from auto_tuner.agents.supervisor_agent import OpenRouterSupervisorAgent
from auto_tuner.agents.worker_agent import TrainingWorkerAgent
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.pipeline.generate import generate_examples
from tests.support.openrouter_stub import install_openrouter_stub


def test_generate_examples_materializes_workspace_files(tmp_path: Path, monkeypatch) -> None:
    install_openrouter_stub(monkeypatch)
    supervisor = OpenRouterSupervisorAgent(OpenRouterConfig(api_key="test"))
    worker = TrainingWorkerAgent.from_requested_backend("fake", model_name="Qwen/Qwen2.5-0.5B-Instruct")
    prompts = PromptBundle(
        meta_prompt="goal",
        generation_prompt="generated prompt",
        grading_prompt="grading prompt",
        source="openrouter",
    )
    generated = generate_examples(
        config=GenerationConfig(sample_count=2),
        prompts=prompts,
        run_root=tmp_path,
        workspaces_root=tmp_path / "workspaces",
        supervisor=supervisor,
        worker=worker,
    )
    examples = generated.examples

    assert len(examples) == 2
    assert examples[0].generation_prompt == "generated prompt"
    assert examples[0].task
    assert (tmp_path / "workspaces" / "example_0001" / "task.md").exists()
    assert (tmp_path / "workspaces" / "example_0001" / "naive_solution.py").exists()
