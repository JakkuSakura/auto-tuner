from __future__ import annotations

from pathlib import Path

from rich.progress import Progress

from auto_tuner.agents.supervisor_agent import SupervisorAgent
from auto_tuner.config import GradingConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample, GradeResult


def grade_example(
    example: DatasetExample,
    config: GradingConfig,
    prompts: PromptBundle,
    supervisor: SupervisorAgent,
    workspace_dir: Path | None = None,
) -> GradeResult:
    result = supervisor.grade_example(
        workspace_dir=workspace_dir,
        meta_prompt=prompts.meta_prompt,
        grading_prompt=prompts.grading_prompt,
        task=example.task,
        naive_solution=example.naive_solution,
    )
    return result.model_copy(update={"passed": bool(result.score >= config.pass_score)})


def grade_examples(
    examples: list[DatasetExample],
    config: GradingConfig,
    prompts: PromptBundle,
    supervisor: SupervisorAgent,
    workspace_dirs: list[Path] | None = None,
    progress: Progress | None = None,
    progress_task_id: int | None = None,
) -> list[GradeResult]:
    results: list[GradeResult] = []
    for idx, example in enumerate(examples):
        workspace_dir = None
        if workspace_dirs is not None:
            workspace_dir = workspace_dirs[idx]
        last_exc: Exception | None = None
        for _attempt in range(0, config.max_retries + 1):
            try:
                results.append(
                    grade_example(
                        example,
                        config,
                        prompts,
                        supervisor,
                        workspace_dir=workspace_dir,
                    )
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise RuntimeError(f"Grading failed after retries: {last_exc}") from last_exc
        if progress is not None and progress_task_id is not None:
            progress.advance(progress_task_id)
    return results
