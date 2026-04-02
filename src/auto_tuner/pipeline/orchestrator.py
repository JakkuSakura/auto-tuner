from __future__ import annotations

from pathlib import Path

from auto_tuner.agents.supervisor_agent import OpenRouterSupervisorAgent
from auto_tuner.agents.worker_agent import TrainingWorkerAgent
from auto_tuner.config import Settings
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
from auto_tuner.telemetry import GpuMonitor, collect_system_info


def _relative_path(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _write_grade_workspace(
    *, run_root: Path, workspace_dir: Path, grade: dict[str, object]
) -> dict[str, str]:
    grade_path = workspace_dir / "grade.json"
    ArtifactStore.write_json(grade_path, grade)
    return {"grade_path": _relative_path(run_root, grade_path)}


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
    if not settings.openrouter.api_key:
        raise RuntimeError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY or configure openrouter.api_key."
        )
    store = ArtifactStore(settings.app.artifacts_dir)
    run_paths = store.create_run_paths()
    artifacts: list[ArtifactRecord] = []
    requested_backend = settings.training.backend
    worker = TrainingWorkerAgent.from_requested_backend(requested_backend)
    resolved_backend = worker.resolved_backend

    if console is not None:
        render_run_header(console, run_paths, requested_backend, resolved_backend)
        with console.status("Writing config snapshot..."):
            store.write_config_snapshot(run_paths.config_snapshot_path, config_text)
    else:
        store.write_config_snapshot(run_paths.config_snapshot_path, config_text)
    artifacts.append(ArtifactRecord("config snapshot", run_paths.config_snapshot_path))

    system_info = collect_system_info()
    system_info_path = run_paths.root / "system_info.json"
    ArtifactStore.write_json(system_info_path, system_info)
    artifacts.append(ArtifactRecord("system info", system_info_path))

    supervisor = OpenRouterSupervisorAgent(settings.openrouter)
    if console is not None:
        with console.status("Building prompts..."):
            prompt_bundle = supervisor.build_prompts(settings.generation.meta_prompt)
    else:
        prompt_bundle = supervisor.build_prompts(settings.generation.meta_prompt)
    prompts_payload = {
        "meta_prompt": prompt_bundle.meta_prompt,
        "generation_prompt": prompt_bundle.generation_prompt,
        "grading_prompt": prompt_bundle.grading_prompt,
        "prompt_source": prompt_bundle.source,
    }
    prompts_path = run_paths.root / "prompts.json"
    ArtifactStore.write_json(prompts_path, prompts_payload)
    artifacts.append(ArtifactRecord("prompts", prompts_path))

    prompts_md_dir = run_paths.root / "prompts"
    prompts_md_dir.mkdir(parents=True, exist_ok=True)
    meta_prompt_path = prompts_md_dir / "meta_prompt.md"
    generation_prompt_path = prompts_md_dir / "generation_prompt.md"
    grading_prompt_path = prompts_md_dir / "grading_prompt.md"
    meta_prompt_path.write_text(prompt_bundle.meta_prompt)
    generation_prompt_path.write_text(prompt_bundle.generation_prompt)
    grading_prompt_path.write_text(prompt_bundle.grading_prompt)
    artifacts.append(ArtifactRecord("meta prompt (md)", meta_prompt_path))
    artifacts.append(ArtifactRecord("generation prompt (md)", generation_prompt_path))
    artifacts.append(ArtifactRecord("grading prompt (md)", grading_prompt_path))
    if console is not None:
        render_prompts(console, prompt_bundle, prompt_bundle.source)

    if console is not None:
        with console.status("Generating examples..."):
            generated_payload = generate_examples(
                config=settings.generation,
                prompts=prompt_bundle,
                run_root=run_paths.root,
                workspaces_root=run_paths.workspaces_root,
                supervisor=supervisor,
            )
    else:
        generated_payload = generate_examples(
            config=settings.generation,
            prompts=prompt_bundle,
            run_root=run_paths.root,
            workspaces_root=run_paths.workspaces_root,
            supervisor=supervisor,
        )
    generated = generated_payload.examples
    workspace_records = generated_payload.workspace_records
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
    records = [
        DatasetRecord(prompt=example.task, response=example.clean_solution)
        for example in refined_examples
    ]
    store.write_records(run_paths.refined_path, records)
    artifacts.append(ArtifactRecord("refined dataset", run_paths.refined_path))

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

    dataset_path = Path(spec.dataset_path)
    gpu_stats_path = run_paths.root / "gpu_stats.jsonl"
    monitor = GpuMonitor(gpu_stats_path)
    monitor.start()
    try:
        if console is not None:
            with console.status("Training..."):
                job = worker.train(dataset_path, spec)
        else:
            job = worker.train(dataset_path, spec)
    except Exception as exc:
        backend_name = resolved_backend
        try:
            backend_name = worker.backend_name
        except AttributeError:
            pass
        job = _failed_job(backend_name, dataset_path, spec.output_dir, exc)
    finally:
        monitor.stop()
    store.write_training_result(run_paths.training_result_path, job)
    artifacts.append(ArtifactRecord("training result", run_paths.training_result_path))
    artifacts.append(ArtifactRecord("gpu stats", gpu_stats_path))
    if console is not None:
        render_training_result(console, job)

    for record in workspace_records:
        workspace_dir = run_paths.root / record["workspace_dir"]
        task_path = run_paths.root / record["task_path"]
        clean_solution_path = run_paths.root / record["clean_solution_path"]
        try:
            refined_solution_path = supervisor.write_refined_solution_after_training(
                workspace_dir=workspace_dir,
                task_path=task_path,
                clean_solution_path=clean_solution_path,
                meta_prompt=prompt_bundle.meta_prompt,
                training_status=job.status,
                backend=job.backend,
                output_dir=spec.output_dir,
            )
        except Exception as exc:
            error_path = workspace_dir / "refinement_error.txt"
            error_path.write_text(f"{exc}\n")
            record["refinement_error_path"] = _relative_path(run_paths.root, error_path)
            continue

        if refined_solution_path is not None:
            record["refined_solution_path"] = _relative_path(run_paths.root, refined_solution_path)
            refinement_md = workspace_dir / "refinement.md"
            if refinement_md.exists():
                record["refinement_path"] = _relative_path(run_paths.root, refinement_md)
            refinement_request = workspace_dir / "refinement_request.md"
            if refinement_request.exists():
                record["refinement_request_path"] = _relative_path(
                    run_paths.root, refinement_request
                )
            refinement_response = workspace_dir / "refinement_response.md"
            if refinement_response.exists():
                record["refinement_response_path"] = _relative_path(
                    run_paths.root, refinement_response
                )
    store.write_workspace_index(run_paths.workspaces_index_path, workspace_index)
    if console is not None:
        render_examples(console, workspace_records, run_paths.root)

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
        "supervisor": {"type": "openrouter", "model": settings.openrouter.prompt_model},
        "workspaces_root": str(run_paths.workspaces_root),
        "workspaces_index": str(run_paths.workspaces_index_path),
        "artifacts": {record.label: str(record.path) for record in artifacts},
        "system": system_info,
        "model": {
            "name": spec.model_name,
            "load_in_4bit": spec.load_in_4bit,
            "max_seq_length": spec.max_seq_length,
            "output_dir": spec.output_dir,
            "dataset_path": spec.dataset_path,
            "metrics": job.metrics,
        },
    }
    store.write_report(run_paths.report_path, report)
    artifacts.append(ArtifactRecord("report", run_paths.report_path))

    pipeline_run = PipelineRun(run_id=run_paths.root.name, status=job.status, paths=run_paths)
    RunRepository().save(pipeline_run)
    artifacts.append(ArtifactRecord("run metadata", run_paths.root / "run.json"))

    if console is not None:
        render_artifacts(console, artifacts)

    return pipeline_run
