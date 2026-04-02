from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import GradeResult
from auto_tuner.models.run import RunPaths
from auto_tuner.models.training import TrainingJob, TrainingSpec


@dataclass(frozen=True)
class ArtifactRecord:
    label: str
    path: Path


def render_run_header(
    console: Console,
    run_paths: RunPaths,
    requested_backend: str,
    resolved_backend: str,
) -> None:
    run_id = run_paths.root.name
    console.rule(Text(f"auto-tuner run {run_id}", style="bold"))
    console.print(f"[bold]Requested backend:[/bold] {requested_backend}")
    console.print(f"[bold]Resolved backend:[/bold] {resolved_backend}")
    console.print(f"[bold]Artifacts:[/bold] {run_paths.root}")
    console.print(f"[bold]Workspaces:[/bold] {run_paths.workspaces_root}")


def render_prompts(console: Console, prompts: PromptBundle, prompt_source: str) -> None:
    console.rule("Prompts")
    console.print(Panel(prompts.meta_prompt, title="Meta Prompt", border_style="cyan"))
    console.print(Panel(prompts.generation_prompt, title="Generation Prompt", border_style="green"))
    console.print(Panel(prompts.grading_prompt, title="Grading Prompt", border_style="magenta"))
    console.print(f"[bold]Prompt source:[/bold] {prompt_source}")


def render_examples(console: Console, examples: list[dict[str, str]], run_root: Path) -> None:
    console.rule("Example Workspaces")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Workspace")
    table.add_column("Files")

    for example in examples:
        example_id = example.get("example_id", "?")
        workspace_dir = run_root / example["workspace_dir"]
        file_keys = [
            ("task_path", "task"),
            ("naive_solution_path", "naive"),
            ("clean_solution_path", "clean"),
            ("grade_path", "grade"),
            ("refined_solution_path", "refined"),
        ]
        parts: list[str] = []
        for key, label in file_keys:
            value = example.get(key)
            if not value:
                continue
            path = run_root / value
            try:
                size = path.stat().st_size
                lines = len(path.read_text().splitlines())
                parts.append(f"{label}={path.name} ({size}B/{lines}L)")
            except FileNotFoundError:
                parts.append(f"{label}={Path(value).name} (missing)")
        table.add_row(str(example_id), str(workspace_dir), ", ".join(parts))
    console.print(table)


def render_grades(console: Console, grades: list[GradeResult]) -> None:
    console.rule("Grades")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Passed")
    table.add_column("Severity")
    table.add_column("Violations")
    for index, grade in enumerate(grades, start=1):
        table.add_row(
            str(index),
            "yes" if grade.passed else "no",
            grade.severity,
            ", ".join(grade.violations) if grade.violations else "-",
        )
    console.print(table)


def render_training_spec(console: Console, spec: TrainingSpec) -> None:
    console.rule("Training")
    table = Table(show_header=False)
    table.add_row("Backend", spec.backend)
    table.add_row("Method", spec.method)
    table.add_row("Model", spec.model_name)
    table.add_row("Epochs", str(spec.num_train_epochs))
    table.add_row("Batch size", str(spec.per_device_train_batch_size))
    table.add_row("Learning rate", str(spec.learning_rate))
    table.add_row("LoRA rank", str(spec.lora_rank))
    table.add_row("Dataset", spec.dataset_path)
    table.add_row("Output", spec.output_dir)
    console.print(table)


def render_training_result(console: Console, job: TrainingJob) -> None:
    table = Table(show_header=False)
    table.add_row("Status", job.status)
    table.add_row("Mode", job.mode)
    table.add_row("Summary", job.summary or "-")
    if job.metrics:
        table.add_row("Metrics", ", ".join(f"{k}={v}" for k, v in job.metrics.items()))
    if job.warnings:
        table.add_row("Warnings", "\n".join(job.warnings))
    console.print(table)


def render_artifacts(console: Console, artifacts: list[ArtifactRecord]) -> None:
    console.rule("Artifacts")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("Path")
    table.add_column("Bytes", justify="right")
    for artifact in artifacts:
        size = artifact.path.stat().st_size if artifact.path.exists() else 0
        table.add_row(artifact.label, str(artifact.path), str(size))
    console.print(table)
