from __future__ import annotations

from pathlib import Path

from auto_tuner.config import GenerationConfig, OpenRouterConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.pipeline.generate import generate_examples


def test_generate_examples_materializes_workspace_files(tmp_path: Path) -> None:
    prompts = PromptBundle(
        meta_prompt="goal",
        generation_prompt="generated prompt",
        grading_prompt="grading prompt",
        source="fallback",
    )
    generated = generate_examples(
        config=GenerationConfig(sample_count=2),
        prompts=prompts,
        run_root=tmp_path,
        workspaces_root=tmp_path / "workspaces",
        openrouter=OpenRouterConfig(api_key=""),
    )
    examples = generated.examples

    assert len(examples) == 2
    assert examples[0].generation_prompt == "generated prompt"
    assert examples[0].task.startswith("generated prompt")
    assert (tmp_path / "workspaces" / "example_0001" / "task.md").exists()
    assert (tmp_path / "workspaces" / "example_0001" / "naive_solution.py").exists()
    assert (tmp_path / "workspaces" / "example_0001" / "clean_solution.py").exists()
    assert (tmp_path / "workspaces" / "example_0001" / "agent_request.md").exists()
    assert (tmp_path / "workspaces" / "example_0001" / "agent_response.md").exists()
