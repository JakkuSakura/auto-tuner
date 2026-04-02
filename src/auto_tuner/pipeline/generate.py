from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from auto_tuner.agents.supervisor_agent import SupervisorAgent
from auto_tuner.agents.worker_agent import WorkerAgent
from auto_tuner.config import GenerationConfig
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
    supervisor: SupervisorAgent,
    worker: WorkerAgent,
) -> GeneratedExamples:
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
        generated_clean = supervisor.generate_clean_example(
            workspace_dir=workspace_dir,
            meta_prompt=prompts.meta_prompt,
            generation_prompt=prompts.generation_prompt,
            example_id=example_id,
            theme_hint=themes[(example_id - 1) % len(themes)],
        )

        worker_request_path = workspace_dir / "worker_request.md"
        worker_response_path = workspace_dir / "worker_response.md"
        naive_solution_path = workspace_dir / "naive_solution.py"
        worker_request = "\n".join(
            [
                "# auto-tuner naive baseline request",
                "",
                f"- example_id: `{example_id}`",
                f"- theme_hint: `{themes[(example_id - 1) % len(themes)]}`",
                f"- worker_backend: `{worker.resolved_backend}`",
                "",
                "## Task",
                "",
                generated_clean.task.strip(),
                "",
                "## Clean solution",
                "",
                "```python",
                generated_clean.clean_solution.rstrip(),
                "```",
                "",
            ]
        )
        worker_request_path.write_text(worker_request)
        naive_solution = worker.generate_naive_solution(
            task_markdown=generated_clean.task,
            clean_solution=generated_clean.clean_solution,
            example_id=example_id,
            theme_hint=themes[(example_id - 1) % len(themes)],
        )
        worker_response_path.write_text(
            "\n".join(
                [
                    "```python",
                    naive_solution.rstrip(),
                    "```",
                    "",
                ]
            )
        )
        naive_solution_path.write_text(naive_solution)

        examples.append(
            DatasetExample(
                task=generated_clean.task,
                naive_solution=naive_solution,
                clean_solution=generated_clean.clean_solution,
                generation_prompt=generated_clean.generation_prompt,
            )
        )

        def rel(path: Path) -> str:
            return str(path.relative_to(run_root))

        workspace_records.append(
            {
                "example_id": str(example_id),
                "workspace_dir": rel(workspace_dir),
                "task_path": rel(generated_clean.task_path),
                "clean_solution_path": rel(generated_clean.clean_solution_path),
                "agent_request_path": rel(generated_clean.agent_request_path),
                "agent_response_path": rel(generated_clean.agent_response_path),
                "worker_request_path": rel(worker_request_path),
                "worker_response_path": rel(worker_response_path),
                "naive_solution_path": rel(naive_solution_path),
            }
        )

    return GeneratedExamples(examples=examples, workspace_records=workspace_records)
