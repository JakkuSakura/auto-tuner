from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from auto_tuner.config import load_settings
from auto_tuner.pipeline.orchestrator import run_pipeline
from auto_tuner.storage.artifacts import ArtifactStore
from auto_tuner.storage.runs import RunRepository

app = FastAPI(title="auto-tuner")


def _frontend_index() -> Path:
    settings = load_settings()
    return settings.app.frontend_dist / "index.html"


def _run_dir(run_id: str) -> Path:
    settings = load_settings()
    run_dir = settings.app.artifacts_dir / "runs" / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs")
def create_run(config_path: str | None = None) -> dict[str, str]:
    if config_path and not Path(config_path).exists():
        raise HTTPException(status_code=400, detail="Config file not found")
    settings = load_settings(config_path)
    config_text = (
        Path(config_path).read_text()
        if config_path and Path(config_path).exists()
        else ""
    )
    try:
        pipeline_run = run_pipeline(settings, config_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": pipeline_run.run_id, "status": pipeline_run.status}


@app.get("/api/runs")
def list_runs() -> list[dict[str, object]]:
    settings = load_settings()
    runs = RunRepository(settings.app.artifacts_dir).list_runs()
    return [run.model_dump(mode="json") for run in runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    run_dir = _run_dir(run_id)
    run = RunRepository().load(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    training = json.loads((run_dir / "training_result.json").read_text())
    prompts = json.loads((run_dir / "prompts.json").read_text())
    return {
        "run": run.model_dump(mode="json"),
        "report": report,
        "training": training,
        "prompts": prompts,
    }


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, str]:
    settings = load_settings()
    run_dir = _run_dir(run_id)
    RunRepository(settings.app.artifacts_dir).delete(run_dir)
    return {"status": "deleted", "run_id": run_id}


@app.get("/api/runs/{run_id}/download/{name}")
def download_run_file(run_id: str, name: str) -> FileResponse:
    run_dir = _run_dir(run_id)
    try:
        file_path = ArtifactStore.resolve_run_file(run_dir, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(file_path)


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str) -> FileResponse:
    run_dir = _run_dir(run_id)
    archive_path = ArtifactStore.export_run(run_dir)
    return FileResponse(archive_path)


@app.get("/api/frontend-config")
def frontend_config() -> dict[str, object]:
    settings = load_settings()
    return {
        "defaultConfigPath": "examples/sample_experiment.yaml",
        "backend": settings.training.backend,
        "defaultPrompt": settings.generation.meta_prompt,
    }


@app.get("/", response_class=HTMLResponse, response_model=None)
def index() -> Response:
    index_file = _frontend_index()
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<html><body><h1>auto-tuner frontend not built</h1></body></html>")
