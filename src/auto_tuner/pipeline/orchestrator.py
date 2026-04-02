from __future__ import annotations

from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

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


def _round_dir(run_root: Path, pass_index: int) -> Path:
    return run_root / "passes" / f"pass_{pass_index:02d}"


def _workspace_pass_dir(workspaces_root: Path, example_id: int, pass_index: int) -> Path:
    return workspaces_root / f"example_{example_id:04d}" / f"pass_{pass_index:02d}"


def _write_target_solution(pass_workspace_dir: Path, solution: str) -> Path:
    path = pass_workspace_dir / "target_solution.py"
    path.write_text(solution)
    return path


def run_pipeline(settings: Settings, config_text: str, console=None) -> PipelineRun:
    if not settings.openrouter.api_key:
        raise RuntimeError(
            "OpenRouter API key is required. "
            "Set OPENROUTER_API_KEY or configure openrouter.api_key."
        )
    store = ArtifactStore(settings.app.artifacts_dir)
    run_paths = store.create_run_paths()
    artifacts: list[ArtifactRecord] = []
    requested_backend = settings.training.backend
    worker = TrainingWorkerAgent.from_requested_backend(
        requested_backend, method=settings.training.method
    )
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

    events_path = run_paths.root / "events.jsonl"
    events_path.write_text("")
    events: list[dict[str, object]] = []

    def record_event(payload: dict[str, object]) -> None:
        events.append(payload)
        store.append_jsonl(events_path, payload)

    supervisor = OpenRouterSupervisorAgent(settings.openrouter)
    prompt_bundle = None
    generated = []
    workspace_records: list[dict[str, str]] = []
    grades = []
    spec: TrainingSpec | None = None
    job: TrainingJob | None = None
    failure: Exception | None = None
    pass_summaries: list[dict[str, object]] = []

    try:
        record_event(
            {
                "stage": "started",
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
            }
        )
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
        record_event({"stage": "prompts_built", "prompt_source": prompt_bundle.source})

        if console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                generate_task_id = progress.add_task(
                    "Generating examples", total=settings.generation.sample_count
                )
                generated_payload = generate_examples(
                    config=settings.generation,
                    prompts=prompt_bundle,
                    run_root=run_paths.root,
                    workspaces_root=run_paths.workspaces_root,
                    supervisor=supervisor,
                    progress=progress,
                    progress_task_id=generate_task_id,
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

        record_event({"stage": "generated", "count": len(generated)})
        store.write_jsonl(run_paths.generated_path, workspace_records)
        artifacts.append(ArtifactRecord("generated examples", run_paths.generated_path))

        if console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                grade_task_id = progress.add_task("Grading examples", total=len(generated))
                workspace_dirs = [
                    run_paths.workspaces_root / f"example_{example_id:04d}"
                    for example_id in range(1, len(generated) + 1)
                ]
                grades = grade_examples(
                    generated,
                    settings.grading,
                    prompt_bundle,
                    supervisor,
                    workspace_dirs=workspace_dirs,
                    progress=progress,
                    progress_task_id=grade_task_id,
                )
        else:
            workspace_dirs = [
                run_paths.workspaces_root / f"example_{example_id:04d}"
                for example_id in range(1, len(generated) + 1)
            ]
            grades = grade_examples(
                generated, settings.grading, prompt_bundle, supervisor, workspace_dirs=workspace_dirs
            )

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
        record_event(
            {
                "stage": "graded",
                "count": len(grades),
                "passed": sum(1 for grade in grades if grade.passed),
            }
        )

        training_method = settings.training.method
        if training_method not in {"sft", "grpo"}:
            raise RuntimeError(f"Unsupported training.method: {training_method}")

        if training_method == "grpo":
            raise RuntimeError(
                "Strict multi-pass mode is not implemented for training.method='grpo' yet."
            )

        targets: list[str] = []
        for example_id, grade in enumerate(grades, start=1):
            if grade.passed:
                targets.append(generated[example_id - 1].naive_solution)
                continue
            pass_workspace_dir = _workspace_pass_dir(
                run_paths.workspaces_root, example_id, 1
            )
            pass_workspace_dir.mkdir(parents=True, exist_ok=True)
            target = supervisor.generate_target_solution(
                workspace_dir=pass_workspace_dir,
                meta_prompt=prompt_bundle.meta_prompt,
                task=generated[example_id - 1].task,
                candidate_solution=generated[example_id - 1].naive_solution,
                grade=grade,
            )
            targets.append(target)

        configured_output_dir = Path(settings.training.output_dir)
        base_output_dir = (
            configured_output_dir
            if configured_output_dir.is_absolute()
            else run_paths.root / configured_output_dir
        )

        gpu_stats_path = run_paths.root / "gpu_stats.jsonl"
        monitor = GpuMonitor(gpu_stats_path)
        monitor.start()
        try:
            for pass_index in range(1, settings.grading.max_passes + 1):
                record_event({"stage": "pass_started", "pass": pass_index})

                pass_root = _round_dir(run_paths.root, pass_index)
                pass_root.mkdir(parents=True, exist_ok=True)

                for example_id, record in enumerate(workspace_records, start=1):
                    pass_workspace_dir = _workspace_pass_dir(
                        run_paths.workspaces_root, example_id, pass_index
                    )
                    pass_workspace_dir.mkdir(parents=True, exist_ok=True)
                    target_solution_path = _write_target_solution(
                        pass_workspace_dir, targets[example_id - 1]
                    )
                    if "passes" not in record:
                        record["passes"] = []
                    record["passes"].append(
                        {
                            "pass": pass_index,
                            "workspace_dir": _relative_path(run_paths.root, pass_workspace_dir),
                            "target_solution_path": _relative_path(
                                run_paths.root, target_solution_path
                            ),
                        }
                    )

                records = [
                    DatasetRecord(prompt=example.task, response=targets[idx])
                    for idx, example in enumerate(generated)
                ]
                pass_dataset_path = pass_root / "training_dataset.jsonl"
                store.write_records(pass_dataset_path, records)
                store.write_records(run_paths.training_dataset_path, records)
                if pass_index == 1:
                    artifacts.append(
                        ArtifactRecord(
                            "training dataset (sft)", run_paths.training_dataset_path
                        )
                    )
                record_event({"stage": "dataset_built", "pass": pass_index, "method": "sft"})

                pass_output_dir = base_output_dir / f"pass_{pass_index:02d}"
                spec = TrainingSpec(
                    backend=resolved_backend,
                    method=training_method,
                    model_name=settings.training.model_name,
                    max_seq_length=settings.training.max_seq_length,
                    load_in_4bit=settings.training.load_in_4bit,
                    num_train_epochs=settings.training.num_train_epochs,
                    per_device_train_batch_size=settings.training.per_device_train_batch_size,
                    output_dir=str(pass_output_dir),
                    dataset_path=str(pass_dataset_path),
                    lora_rank=settings.training.lora_rank,
                    learning_rate=settings.training.learning_rate,
                    grpo=None,
                )
                store.write_training_spec(pass_root / "training_spec.json", spec)
                store.write_training_spec(run_paths.training_spec_path, spec)
                if pass_index == 1:
                    artifacts.append(ArtifactRecord("training spec", run_paths.training_spec_path))
                if console is not None:
                    console.rule(f"Pass {pass_index}")
                    render_training_spec(console, spec)

                dataset_path = Path(spec.dataset_path)
                try:
                    if console is not None:
                        with console.status(f"Training (pass {pass_index})..."):
                            job = worker.train(dataset_path, spec)
                    else:
                        job = worker.train(dataset_path, spec)
                except Exception as exc:
                    job = _failed_job(resolved_backend, dataset_path, spec.output_dir, exc)
                store.write_training_result(pass_root / "training_result.json", job)
                store.write_training_result(run_paths.training_result_path, job)
                if pass_index == 1:
                    artifacts.append(
                        ArtifactRecord("training result", run_paths.training_result_path)
                    )
                    artifacts.append(ArtifactRecord("gpu stats", gpu_stats_path))
                if console is not None:
                    render_training_result(console, job)

                record_event(
                    {
                        "stage": "trained",
                        "pass": pass_index,
                        "status": job.status,
                        "backend": job.backend,
                        "mode": job.mode,
                    }
                )

                if job.status != "completed":
                    raise RuntimeError(
                        f"Training did not complete in pass {pass_index}: {job.summary}"
                    )
                if worker.backend_name == "fake":
                    break

                adapter_dir = Path(spec.output_dir)
                refined_grades: list[dict[str, object]] = []
                passed_count = 0
                updated_targets: dict[int, str] = {}

                if console is not None:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        TextColumn("{task.completed}/{task.total}"),
                        console=console,
                    ) as progress:
                        refined_task_id = progress.add_task(
                            f"Refining+grading (pass {pass_index})", total=len(workspace_records)
                        )
                        for example_id, record in enumerate(workspace_records, start=1):
                            task_rel = record.get("task_path")
                            if not task_rel:
                                progress.advance(refined_task_id)
                                continue
                            pass_workspace_dir = _workspace_pass_dir(
                                run_paths.workspaces_root, example_id, pass_index
                            )
                            pass_workspace_dir.mkdir(parents=True, exist_ok=True)

                            refined_solution_path = pass_workspace_dir / "refined_solution.py"
                            refined_solution = worker.generate_refined_solution(
                                workspace_dir=pass_workspace_dir,
                                task_path=run_paths.root / task_rel,
                                base_model_name=spec.model_name,
                                adapter_dir=adapter_dir,
                            )
                            refined_solution_path.write_text(refined_solution)

                            refined_grade = supervisor.grade_example(
                                workspace_dir=pass_workspace_dir,
                                meta_prompt=prompt_bundle.meta_prompt,
                                grading_prompt=prompt_bundle.grading_prompt,
                                task=generated[example_id - 1].task,
                                naive_solution=refined_solution,
                            )
                            refined_grade_payload = refined_grade.model_dump()
                            ArtifactStore.write_json(
                                pass_workspace_dir / "refined_grade.json", refined_grade_payload
                            )
                            passes = record.get("passes")
                            if isinstance(passes, list):
                                for entry in reversed(passes):
                                    if (
                                        isinstance(entry, dict)
                                        and entry.get("pass") == pass_index
                                        and entry.get("refined_solution_path") is None
                                    ):
                                        entry["refined_solution_path"] = _relative_path(
                                            run_paths.root, refined_solution_path
                                        )
                                        entry["refined_grade_path"] = _relative_path(
                                            run_paths.root,
                                            pass_workspace_dir / "refined_grade.json",
                                        )
                                        entry["passed"] = refined_grade.passed
                                        entry["score"] = refined_grade.score
                                        entry["severity"] = refined_grade.severity
                                        entry["violations"] = refined_grade.violations
                                        break

                            if refined_grade.passed:
                                passed_count += 1
                            else:
                                if pass_index < settings.grading.max_passes:
                                    next_pass_dir = _workspace_pass_dir(
                                        run_paths.workspaces_root, example_id, pass_index + 1
                                    )
                                    next_pass_dir.mkdir(parents=True, exist_ok=True)
                                    updated = supervisor.generate_target_solution(
                                        workspace_dir=next_pass_dir,
                                        meta_prompt=prompt_bundle.meta_prompt,
                                        task=generated[example_id - 1].task,
                                        candidate_solution=refined_solution,
                                        grade=refined_grade,
                                    )
                                    updated_targets[example_id] = updated

                            refined_grades.append(
                                {
                                    "example_id": example_id,
                                    "pass": pass_index,
                                    "workspace_dir": _relative_path(
                                        run_paths.root, pass_workspace_dir
                                    ),
                                    "passed": refined_grade.passed,
                                    "score": refined_grade.score,
                                    "severity": refined_grade.severity,
                                    "violations": refined_grade.violations,
                                }
                            )
                            progress.advance(refined_task_id)
                else:
                    for example_id, record in enumerate(workspace_records, start=1):
                        task_rel = record.get("task_path")
                        if not task_rel:
                            continue
                        pass_workspace_dir = _workspace_pass_dir(
                            run_paths.workspaces_root, example_id, pass_index
                        )
                        pass_workspace_dir.mkdir(parents=True, exist_ok=True)

                        refined_solution_path = pass_workspace_dir / "refined_solution.py"
                        refined_solution = worker.generate_refined_solution(
                            workspace_dir=pass_workspace_dir,
                            task_path=run_paths.root / task_rel,
                            base_model_name=spec.model_name,
                            adapter_dir=adapter_dir,
                        )
                        refined_solution_path.write_text(refined_solution)

                        refined_grade = supervisor.grade_example(
                            workspace_dir=pass_workspace_dir,
                            meta_prompt=prompt_bundle.meta_prompt,
                            grading_prompt=prompt_bundle.grading_prompt,
                            task=generated[example_id - 1].task,
                            naive_solution=refined_solution,
                        )
                        refined_grade_payload = refined_grade.model_dump()
                        ArtifactStore.write_json(
                            pass_workspace_dir / "refined_grade.json", refined_grade_payload
                        )
                        passes = record.get("passes")
                        if isinstance(passes, list):
                            for entry in reversed(passes):
                                if (
                                    isinstance(entry, dict)
                                    and entry.get("pass") == pass_index
                                    and entry.get("refined_solution_path") is None
                                ):
                                    entry["refined_solution_path"] = _relative_path(
                                        run_paths.root, refined_solution_path
                                    )
                                    entry["refined_grade_path"] = _relative_path(
                                        run_paths.root,
                                        pass_workspace_dir / "refined_grade.json",
                                    )
                                    entry["passed"] = refined_grade.passed
                                    entry["score"] = refined_grade.score
                                    entry["severity"] = refined_grade.severity
                                    entry["violations"] = refined_grade.violations
                                    break

                        if refined_grade.passed:
                            passed_count += 1
                        else:
                            if pass_index < settings.grading.max_passes:
                                next_pass_dir = _workspace_pass_dir(
                                    run_paths.workspaces_root, example_id, pass_index + 1
                                )
                                next_pass_dir.mkdir(parents=True, exist_ok=True)
                                updated = supervisor.generate_target_solution(
                                    workspace_dir=next_pass_dir,
                                    meta_prompt=prompt_bundle.meta_prompt,
                                    task=generated[example_id - 1].task,
                                    candidate_solution=refined_solution,
                                    grade=refined_grade,
                                )
                                updated_targets[example_id] = updated

                        refined_grades.append(
                            {
                                "example_id": example_id,
                                "pass": pass_index,
                                "workspace_dir": _relative_path(run_paths.root, pass_workspace_dir),
                                "passed": refined_grade.passed,
                                "score": refined_grade.score,
                                "severity": refined_grade.severity,
                                "violations": refined_grade.violations,
                            }
                        )

                for example_id, updated in updated_targets.items():
                    targets[example_id - 1] = updated

                refined_grades_path = pass_root / "refined_grades.jsonl"
                store.write_jsonl(refined_grades_path, refined_grades)
                pass_summaries.append(
                    {
                        "pass": pass_index,
                        "adapter_dir": str(adapter_dir),
                        "dataset_path": str(pass_dataset_path),
                        "passed": passed_count,
                        "total": len(workspace_records),
                    }
                )

                record_event(
                    {
                        "stage": "refined_graded",
                        "pass": pass_index,
                        "passed": passed_count,
                        "total": len(workspace_records),
                        "grades_path": _relative_path(run_paths.root, refined_grades_path),
                    }
                )

                if passed_count == len(workspace_records):
                    for example_id, record in enumerate(workspace_records, start=1):
                        pass_workspace_dir = _workspace_pass_dir(
                            run_paths.workspaces_root, example_id, pass_index
                        )
                        final_refined_path = (
                            run_paths.workspaces_root
                            / f"example_{example_id:04d}"
                            / "refined_solution.py"
                        )
                        final_refined_path.write_text(
                            (pass_workspace_dir / "refined_solution.py").read_text()
                        )
                        record["refined_solution_path"] = _relative_path(
                            run_paths.root, final_refined_path
                        )
                    record_event({"stage": "refined", "pass": pass_index, "count": passed_count})
                    break

                if pass_index == settings.grading.max_passes:
                    raise RuntimeError(
                        f"Refined solutions still failed after {settings.grading.max_passes} passes: "
                        f"{len(workspace_records) - passed_count}/{len(workspace_records)}"
                    )
        finally:
            monitor.stop()
    except Exception as exc:
        failure = exc
        job = _failed_job(resolved_backend, run_paths.training_dataset_path, settings.training.output_dir, exc)
        record_event({"stage": "failed", "error": str(exc)})

    artifacts.append(ArtifactRecord("events", events_path))

    if workspace_records:
        workspace_index = {"version": 1, "examples": workspace_records}
        store.write_workspace_index(run_paths.workspaces_index_path, workspace_index)
        if console is not None:
            render_examples(console, workspace_records, run_paths.root)

    if job is None:
        job = _failed_job(
            resolved_backend,
            run_paths.training_dataset_path,
            settings.training.output_dir,
            RuntimeError("Pipeline aborted before training."),
        )

    report = {
        "run_id": run_paths.root.name,
        "generated_examples": len(generated),
        "passed_examples": sum(1 for grade in grades if grade.passed),
        "passes": pass_summaries,
        "backend": job.backend,
        "training_status": job.status,
        "training_mode": job.mode,
        "summary": job.summary,
        "warnings": job.warnings,
        "error": str(failure) if failure is not None else "",
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "supervisor": {"type": "openrouter", "model": settings.openrouter.prompt_model},
        "workspaces_root": str(run_paths.workspaces_root),
        "workspaces_index": str(run_paths.workspaces_index_path),
        "artifacts": {record.label: str(record.path) for record in artifacts},
        "system": system_info,
        "model": {
            "name": (spec.model_name if spec is not None else settings.training.model_name),
            "load_in_4bit": (
                spec.load_in_4bit if spec is not None else settings.training.load_in_4bit
            ),
            "max_seq_length": (
                spec.max_seq_length if spec is not None else settings.training.max_seq_length
            ),
            "output_dir": (spec.output_dir if spec is not None else ""),
            "dataset_path": (spec.dataset_path if spec is not None else ""),
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
