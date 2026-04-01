from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from auto_tuner.models.dataset import DatasetExample, DatasetRecord, GradeResult
from auto_tuner.models.run import RunPaths
from auto_tuner.models.training import TrainingJob, TrainingSpec


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create_run_paths(self) -> RunPaths:
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        run_root = self.root / "runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        return RunPaths(
            root=run_root,
            generated_path=run_root / "generated.jsonl",
            graded_path=run_root / "graded.jsonl",
            refined_path=run_root / "refined.jsonl",
            training_spec_path=run_root / "training_spec.json",
            training_result_path=run_root / "training_result.json",
            report_path=run_root / "report.json",
            config_snapshot_path=run_root / "config.snapshot.toml",
        )

    def write_examples(self, path: Path, examples: list[DatasetExample]) -> None:
        self._write_jsonl(path, [example.model_dump() for example in examples])

    def write_grade_results(self, path: Path, grades: list[GradeResult]) -> None:
        self._write_jsonl(path, [grade.model_dump() for grade in grades])

    def write_records(self, path: Path, records: list[DatasetRecord]) -> None:
        rows = []
        for record in records:
            conversation = record.as_conversation()
            rows.append({**conversation, "text": record.response})
        self._write_jsonl(path, rows)

    def write_training_spec(self, path: Path, spec: TrainingSpec) -> None:
        path.write_text(json.dumps(spec.model_dump(), indent=2))

    def write_training_result(self, path: Path, job: TrainingJob) -> None:
        path.write_text(json.dumps(job.model_dump(), indent=2))

    def write_report(self, path: Path, report: dict[str, object]) -> None:
        path.write_text(json.dumps(report, indent=2))

    def write_config_snapshot(self, path: Path, config_text: str) -> None:
        path.write_text(config_text)

    @staticmethod
    def write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
        path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def export_run(run_root: Path) -> Path:
        archive_path = run_root.with_suffix('.zip')
        with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as zip_file:
            for file_path in run_root.rglob('*'):
                if file_path.is_file():
                    zip_file.write(file_path, file_path.relative_to(run_root))
        return archive_path

    @staticmethod
    def resolve_run_file(run_root: Path, name: str) -> Path:
        candidate = (run_root / name).resolve()
        if run_root.resolve() not in candidate.parents and candidate != run_root.resolve():
            raise ValueError("Invalid run file path")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(name)
        return candidate

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
