from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from auto_tuner.cli import app


runner = CliRunner()


def test_cli_run_creates_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TUNER_ARTIFACTS_DIR", str(tmp_path / ".artifacts"))

    result = runner.invoke(app, ["run", "--config", "examples/sample_experiment.toml"])

    assert result.exit_code == 0
    artifacts_line = next(line for line in result.stdout.splitlines() if line.startswith("artifacts="))
    run_dir = Path(artifacts_line.split("=", 1)[1])
    assert (run_dir / "generated.jsonl").exists()
    assert (run_dir / "refined.jsonl").exists()
    assert (run_dir / "training_result.json").exists()
    assert (run_dir / "prompts.json").exists()
    assert "demo" in (run_dir / "report.json").read_text()


def test_cli_list_runs_shows_created_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TUNER_ARTIFACTS_DIR", str(tmp_path / ".artifacts"))

    run_result = runner.invoke(app, ["run", "--config", "examples/sample_experiment.toml"])
    assert run_result.exit_code == 0

    list_result = runner.invoke(app, ["list-runs", "--config", "examples/sample_experiment.toml"])
    assert list_result.exit_code == 0
    assert "completed" in list_result.stdout


def test_cli_export_and_delete_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TUNER_ARTIFACTS_DIR", str(tmp_path / ".artifacts"))

    run_result = runner.invoke(app, ["run", "--config", "examples/sample_experiment.toml"])
    artifacts_line = next(line for line in run_result.stdout.splitlines() if line.startswith("artifacts="))
    run_dir = Path(artifacts_line.split("=", 1)[1])

    export_result = runner.invoke(app, ["export-run", str(run_dir)])
    assert export_result.exit_code == 0
    archive_line = next(line for line in export_result.stdout.splitlines() if line.startswith("archive="))
    archive_path = Path(archive_line.split("=", 1)[1])
    assert archive_path.exists()

    delete_result = runner.invoke(app, ["delete-run", str(run_dir)])
    assert delete_result.exit_code == 0
    assert not run_dir.exists()
