from __future__ import annotations

import json
import shutil
from pathlib import Path

from auto_tuner.models.run import PipelineRun


class RunRepository:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    def save(self, run: PipelineRun) -> None:
        meta_path = run.paths.root / "run.json"
        meta_path.write_text(json.dumps(run.model_dump(mode="json"), indent=2))

    def load(self, run_root: Path) -> PipelineRun:
        return PipelineRun.model_validate_json((run_root / "run.json").read_text())

    def list_runs(self) -> list[PipelineRun]:
        if self.root is None:
            return []
        runs_root = self.root / "runs"
        if not runs_root.exists():
            return []
        return [self.load(path) for path in sorted(runs_root.iterdir(), reverse=True) if path.is_dir()]

    def delete(self, run_root: Path) -> None:
        shutil.rmtree(run_root)
