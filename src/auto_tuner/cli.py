from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from auto_tuner.config import load_settings
from auto_tuner.pipeline.orchestrator import run_pipeline
from auto_tuner.storage.artifacts import ArtifactStore
from auto_tuner.storage.runs import RunRepository

app = typer.Typer(no_args_is_help=True)


def _read_config_text(config_path: str | None) -> str:
    candidate = Path(config_path or "examples/sample_experiment.toml")
    return candidate.read_text() if candidate.exists() else ""


@app.command()
def run(config: str = typer.Option(None, help="Path to experiment config.")) -> None:
    settings = load_settings(config)
    console = Console()
    pipeline_run = run_pipeline(settings, _read_config_text(config), console=console)
    typer.echo(f"run_id={pipeline_run.run_id}")
    typer.echo(f"status={pipeline_run.status}")
    typer.echo(f"artifacts={pipeline_run.paths.root}")


@app.command()
def status(run_dir: str = typer.Argument(..., help="Path to a run directory.")) -> None:
    pipeline_run = RunRepository().load(Path(run_dir))
    typer.echo(f"run_id={pipeline_run.run_id}")
    typer.echo(f"status={pipeline_run.status}")


@app.command()
def report(run_dir: str = typer.Argument(..., help="Path to a run directory.")) -> None:
    report_path = Path(run_dir) / "report.json"
    typer.echo(report_path.read_text())


@app.command("list-runs")
def list_runs(config: str = typer.Option(None, help="Path to experiment config.")) -> None:
    settings = load_settings(config)
    runs = RunRepository(settings.app.artifacts_dir).list_runs()
    for run in runs:
        typer.echo(f"{run.run_id}\t{run.status}\t{run.paths.root}")


@app.command("delete-run")
def delete_run(run_dir: str = typer.Argument(..., help="Path to a run directory.")) -> None:
    target = Path(run_dir)
    RunRepository().delete(target)
    typer.echo(f"deleted={target}")


@app.command("export-run")
def export_run(run_dir: str = typer.Argument(..., help="Path to a run directory.")) -> None:
    archive = ArtifactStore.export_run(Path(run_dir))
    typer.echo(f"archive={archive}")
