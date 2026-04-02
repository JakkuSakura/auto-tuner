from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from auto_tuner.agents.workspace_agent import WorkspaceAgent
from auto_tuner.config import GenerationConfig, OpenRouterConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample


@dataclass(frozen=True)
class GeneratedExamples:
    examples: list[DatasetExample]
    workspace_records: list[dict[str, str]]


def generate_examples(
    *,
    config: GenerationConfig,
    prompts: PromptBundle,
    run_root: Path,
    workspaces_root: Path,
    openrouter: OpenRouterConfig,
) -> GeneratedExamples:
    agent = WorkspaceAgent(openrouter)
    examples: list[DatasetExample] = []
    workspace_records: list[dict[str, str]] = []

    themes = [
        "config",
        "serialization",
        "cli",
        "routing",
        "metrics",
        "io",
        "validation",
        "models",
    ]

    for example_id in range(1, config.sample_count + 1):
        workspace_dir = workspaces_root / f"example_{example_id:04d}"
        generated = agent.generate_example(
            workspace_dir=workspace_dir,
            meta_prompt=prompts.meta_prompt,
            generation_prompt=prompts.generation_prompt,
            example_id=example_id,
            theme_hint=themes[(example_id - 1) % len(themes)],
        )

        examples.append(
            DatasetExample(
                task=generated.task,
                naive_solution=generated.naive_solution,
                clean_solution=generated.clean_solution,
                generation_prompt=generated.generation_prompt,
            )
        )

        def rel(path: Path) -> str:
            return str(path.relative_to(run_root))

        workspace_records.append(
            {
                "example_id": str(example_id),
                "workspace_dir": rel(workspace_dir),
                "task_path": rel(generated.task_path),
                "naive_solution_path": rel(generated.naive_solution_path),
                "clean_solution_path": rel(generated.clean_solution_path),
                "agent_request_path": rel(generated.agent_request_path),
                "agent_response_path": rel(generated.agent_response_path),
            }
        )

    return GeneratedExamples(examples=examples, workspace_records=workspace_records)

