from __future__ import annotations

import platform
from pathlib import Path

from auto_tuner.backends.fake import FakeTrainingBackend
from auto_tuner.backends.mlx_tune import MlxTuneTrainingBackend
from auto_tuner.backends.unsloth_sdk import UnslothTrainingBackend
from auto_tuner.config import Settings
from auto_tuner.llm.openrouter import build_prompt_provider
from auto_tuner.models.dataset import DatasetRecord
from auto_tuner.models.run import PipelineRun
from auto_tuner.models.training import TrainingJob, TrainingSpec
from auto_tuner.pipeline.display import (
    ArtifactRecord,
    render_artifacts,
    render_examples,
    render_grades,
    render_prompts,
    render_run_header,
    render_training_result,
    render_training_spec,
)
from auto_tuner.pipeline.generate import generate_examples
from auto_tuner.pipeline.grade import grade_examples
from auto_tuner.pipeline.refine import refine_examples
from auto_tuner.storage.artifacts import ArtifactStore
from auto_tuner.storage.runs import RunRepository


def _resolve_backend_name(name: str) -> str:
    if name != "auto":
        return name
    if platform.system() == "Darwin":
        return "mlx_tune"
    return "unsloth"


def _select_backend(name: str):
    resolved = _resolve_backend_name(name)
    if resolved == "mlx_tune":
        return MlxTuneTrainingBackend()
    if resolved == "unsloth":
        return UnslothTrainingBackend()
    return FakeTrainingBackend()


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _write_example_workspace(
    *,
    run_root: Path,
    workspaces_root: Path,
    example_id: int,
    task: str,
    generation_prompt: str,
    naive_solution: str,
    clean_solution: str,
) -> dict[str, str]:
    workspace_dir = workspaces_root / f"example_{example_id:04d}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    task_path = workspace_dir / "task.md"
    generation_prompt_path = workspace_dir / "generation_prompt.txt"
    naive_solution_path = workspace_dir / "naive_solution.py"
    clean_solution_path = workspace_dir / "clean_solution.py"

    task_path.write_text(task)
    generation_prompt_path.write_text(generation_prompt)
    naive_solution_path.write_text(naive_solution)
    clean_solution_path.write_text(clean_solution)

    return {
        "example_id": str(example_id),
        "workspace_dir": _relative_path(run_root, workspace_dir),
        "task_path": _relative_path(run_root, task_path),
        "generation_prompt_path": _relative_path(run_root, generation_prompt_path),
        "naive_solution_path": _relative_path(run_root, naive_solution_path),
        "clean_solution_path": _relative_path(run_root, clean_solution_path),
    }


def _write_grade_workspace(
    *, run_root: Path, workspace_dir: Path, grade: dict[str, object]
) -> dict[str, str]:
    grade_path = workspace_dir / "grade.json"
    ArtifactStore.write_json(grade_path, grade)
    return {"grade_path": _relative_path(run_root, grade_path)}


def _write_refined_workspace(
    *, run_root: Path, workspace_dir: Path, refined_solution: str
) -> dict[str, str]:
    refined_solution_path = workspace_dir / "refined_solution.py"
    refined_solution_path.write_text(refined_solution)
    return {"refined_solution_path": _relative_path(run_root, refined_solution_path)}


def _build_demo(settings: Settings, prompts: dict[str, str]) -> dict[str, object]:
    before = "def read_value(obj):\n    return getattr(obj, 'value')\n"
    after = "def read_value(obj):\n    return obj.value\n"
    return {
        "meta_prompt": prompts["meta_prompt"],
        "input_prompt": prompts["generation_prompt"],
        "grading_prompt": prompts["grading_prompt"],
        "prompt_source": prompts["prompt_source"],
        "before_auto_tuning": before,
        "after_auto_tuning": after,
        "recommended_small_models": settings.demo.example_models,
        "notes": [
            "Qwen/Qwen2.5-0.5B-Instruct is a good first small instruction-tuning target.",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0 is included as a second compact example.",
            "Use the fake backend for deterministic local development and tests.",
            "The Unsloth/MLX-Tune backends are guarded and may report unsupported.",
        ],
    }


def _failed_job(
    backend_name: str,
    dataset_path: Path,
    output_dir: str,
    exc: Exception,
) -> TrainingJob:
    return TrainingJob(
        job_id=f"{backend_name}-{dataset_path.stem}",
        status="failed",
        backend=backend_name,
        mode="guarded",
        summary=f"Training failed before start: {exc}",
        artifacts={"dataset_path": str(dataset_path), "output_dir": output_dir},
        warnings=[str(exc)],
    )


def run_pipeline(settings: Settings, config_text: str, console=None) -> PipelineRun:
    store = ArtifactStore(settings.app.artifacts_dir)
    run_paths = store.create_run_paths()
    artifacts: list[ArtifactRecord] = []
    requested_backend = settings.training.backend
    resolved_backend = _resolve_backend_name(requested_backend)

    if console is not None:
        render_run_header(console, run_paths, requested_backend, resolved_backend)
        with console.status("Writing config snapshot..."):
            store.write_config_snapshot(run_paths.config_snapshot_path, config_text)
    else:
        store.write_config_snapshot(run_paths.config_snapshot_path, config_text)
    artifacts.append(ArtifactRecord("config snapshot", run_paths.config_snapshot_path))

    prompt_provider = build_prompt_provider(settings.openrouter)
    if console is not None:
        with console.status("Building prompts..."):
            prompt_bundle = prompt_provider.build_prompts(settings.generation.meta_prompt)
    else:
        prompt_bundle = prompt_provider.build_prompts(settings.generation.meta_prompt)
    prompts_payload = {
        "meta_prompt": prompt_bundle.meta_prompt,
        "generation_prompt": prompt_bundle.generation_prompt,
        "grading_prompt": prompt_bundle.grading_prompt,
        "prompt_source": prompt_bundle.source,
    }
    prompts_path = run_paths.root / "prompts.json"
    ArtifactStore.write_json(prompts_path, prompts_payload)
    artifacts.append(ArtifactRecord("prompts", prompts_path))
    if console is not None:
        render_prompts(console, prompt_bundle, prompt_bundle.source)

    if console is not None:
        with console.status("Generating examples..."):
            generated = generate_examples(settings.generation, prompt_bundle)
    else:
        generated = generate_examples(settings.generation, prompt_bundle)
    workspace_records: list[dict[str, str]] = []
    for index, example in enumerate(generated, start=1):
        workspace_records.append(
            _write_example_workspace(
                run_root=run_paths.root,
                workspaces_root=run_paths.workspaces_root,
                example_id=index,
                task=example.task,
                generation_prompt=example.generation_prompt,
                naive_solution=example.naive_solution,
                clean_solution=example.clean_solution,
            )
        )
    workspace_index: dict[str, object] = {"version": 1, "examples": workspace_records}
    store.write_workspace_index(run_paths.workspaces_index_path, workspace_index)
    artifacts.append(ArtifactRecord("workspaces index", run_paths.workspaces_index_path))

    store.write_jsonl(run_paths.generated_path, workspace_records)
    artifacts.append(ArtifactRecord("generated examples", run_paths.generated_path))

    if console is not None:
        with console.status("Grading examples..."):
            grades = grade_examples(generated, settings.grading, prompt_bundle)
    else:
        grades = grade_examples(generated, settings.grading, prompt_bundle)
    grade_rows: list[dict[str, object]] = []
    for example_id, grade in enumerate(grades, start=1):
        workspace_dir = run_paths.workspaces_root / f"example_{example_id:04d}"
        grade_payload = grade.model_dump()
        workspace_records[example_id - 1].update(
            _write_grade_workspace(
                run_root=run_paths.root,
                workspace_dir=workspace_dir,
                grade=grade_payload,
            )
        )
        grade_rows.append(
            {
                "example_id": example_id,
                "workspace_dir": _relative_path(run_paths.root, workspace_dir),
                "grade_path": workspace_records[example_id - 1]["grade_path"],
                "passed": grade.passed,
                "severity": grade.severity,
                "violations": grade.violations,
            }
        )
    store.write_workspace_index(run_paths.workspaces_index_path, workspace_index)
    store.write_jsonl(run_paths.graded_path, grade_rows)
    artifacts.append(ArtifactRecord("grades", run_paths.graded_path))
    if console is not None:
        render_grades(console, grades)

    if console is not None:
        with console.status("Refining examples..."):
            refined_examples = refine_examples(generated, grades)
    else:
        refined_examples = refine_examples(generated, grades)
    for example_id, example in enumerate(refined_examples, start=1):
        workspace_dir = run_paths.workspaces_root / f"example_{example_id:04d}"
        workspace_records[example_id - 1].update(
            _write_refined_workspace(
                run_root=run_paths.root,
                workspace_dir=workspace_dir,
                refined_solution=example.clean_solution,
            )
        )
    store.write_workspace_index(run_paths.workspaces_index_path, workspace_index)
    records = [
        DatasetRecord(prompt=example.task, response=example.clean_solution)
        for example in refined_examples
    ]
    store.write_records(run_paths.refined_path, records)
    artifacts.append(ArtifactRecord("refined dataset", run_paths.refined_path))
    if console is not None:
        render_examples(console, workspace_records, run_paths.root)

    spec = TrainingSpec(
        backend=resolved_backend,
        model_name=settings.training.model_name,
        max_seq_length=settings.training.max_seq_length,
        load_in_4bit=settings.training.load_in_4bit,
        num_train_epochs=settings.training.num_train_epochs,
        per_device_train_batch_size=settings.training.per_device_train_batch_size,
        output_dir=settings.training.output_dir,
        dataset_path=str(run_paths.refined_path),
        lora_rank=settings.training.lora_rank,
        learning_rate=settings.training.learning_rate,
    )
    store.write_training_spec(run_paths.training_spec_path, spec)
    artifacts.append(ArtifactRecord("training spec", run_paths.training_spec_path))
    if console is not None:
        render_training_spec(console, spec)

    backend = _select_backend(resolved_backend)
    dataset_path = Path(spec.dataset_path)
    try:
        backend.validate()
        if console is not None:
            with console.status("Training..."):
                job = backend.train(dataset_path, spec)
        else:
            job = backend.train(dataset_path, spec)
    except Exception as exc:
        backend_name = resolved_backend
        if hasattr(backend, "name"):
            backend_name = backend.name
        job = _failed_job(backend_name, dataset_path, spec.output_dir, exc)
    store.write_training_result(run_paths.training_result_path, job)
    artifacts.append(ArtifactRecord("training result", run_paths.training_result_path))
    if console is not None:
        render_training_result(console, job)

    demo = _build_demo(settings, prompts_payload) if settings.demo.enabled else {}
    report = {
        "run_id": run_paths.root.name,
        "generated_examples": len(generated),
        "passed_examples": sum(1 for grade in grades if grade.passed),
        "backend": job.backend,
        "training_status": job.status,
        "training_mode": job.mode,
        "summary": job.summary,
        "warnings": job.warnings,
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "workspaces_root": str(run_paths.workspaces_root),
        "workspaces_index": str(run_paths.workspaces_index_path),
        "artifacts": {record.label: str(record.path) for record in artifacts},
        "demo": demo,
    }
    store.write_report(run_paths.report_path, report)
    artifacts.append(ArtifactRecord("report", run_paths.report_path))

    pipeline_run = PipelineRun(run_id=run_paths.root.name, status=job.status, paths=run_paths)
    RunRepository().save(pipeline_run)
    artifacts.append(ArtifactRecord("run metadata", run_paths.root / "run.json"))

    if console is not None:
        render_artifacts(console, artifacts)

    return pipeline_run
