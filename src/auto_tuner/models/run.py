from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RunPaths(BaseModel):
    root: Path
    generated_path: Path
    graded_path: Path
    refined_path: Path
    training_spec_path: Path
    training_result_path: Path
    report_path: Path
    config_snapshot_path: Path
    workspaces_root: Path
    workspaces_index_path: Path


class PipelineRun(BaseModel):
    run_id: str
    status: str
    paths: RunPaths
