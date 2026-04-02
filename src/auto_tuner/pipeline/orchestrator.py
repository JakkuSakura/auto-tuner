from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from auto_tuner.backends.fake import FakeTrainingBackend
from auto_tuner.backends.mlx_tune import MlxTuneTrainingBackend
from auto_tuner.backends.unsloth_sdk import UnslothTrainingBackend
from auto_tuner.config import Settings
from auto_tuner.llm.openrouter import build_prompt_provider
from auto_tuner.models.dataset import DatasetRecord
from auto_tuner.models.run import PipelineRun
from auto_tuner.models.training import TrainingJob, TrainingSpec
from auto_tuner.pipeline.generate import generate_examples
from auto_tuner.pipeline.grade import grade_examples
from auto_tuner.pipeline.refine import refine_examples
from auto_tuner.storage.artifacts import ArtifactStore
from auto_tuner.storage.runs import RunRepository


@dataclass(frozen=True)
class ArtifactRecord:
    label: str
    path: Path


def _select_backend(name: str):
    if name == "mlx_tune":
        return MlxTuneTrainingBackend()
    if name == "unsloth":
        return UnslothTrainingBackend()
    return FakeTrainingBackend()


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


def _failed_job(backend_name: str, dataset_path: Path, output_dir: str, exc: Exception) -> TrainingJob:
    return TrainingJob(
        job_id=f"{backend_name}-{dataset_path.stem}",
        status="failed",
        backend=backend_name,
        mode="guarded",
        summary=f"Training failed before start: {exc}",
        artifacts={"dataset_path": str(dataset_path), "output_dir": output_dir},
        warnings=[str(exc)],
    )


def run_pipeline(settings: Settings, config_text: str) -> PipelineRun:
    store = ArtifactStore(settings.app.artifacts_dir)
    run_paths = store.create_run_paths()
    artifacts: list[ArtifactRecord] = []

    store.write_config_snapshot(run_paths.config_snapshot_path, config_text)
    artifacts.append(ArtifactRecord("config snapshot", run_paths.config_snapshot_path))

    prompt_provider = build_prompt_provider(settings.openrouter)
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

    generated = generate_examples(settings.generation, prompt_bundle)
    store.write_examples(run_paths.generated_path, generated)
    artifacts.append(ArtifactRecord("generated examples", run_paths.generated_path))

    grades = grade_examples(generated, settings.grading, prompt_bundle)
    store.write_grade_results(run_paths.graded_path, grades)
    artifacts.append(ArtifactRecord("grades", run_paths.graded_path))

    refined_examples = refine_examples(generated, grades)
    records = [
        DatasetRecord(prompt=example.task, response=example.clean_solution)
        for example in refined_examples
    ]
    store.write_records(run_paths.refined_path, records)
    artifacts.append(ArtifactRecord("refined dataset", run_paths.refined_path))

    spec = TrainingSpec(
        backend=settings.training.backend,
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

    backend = _select_backend(settings.training.backend)
    dataset_path = Path(spec.dataset_path)
    try:
        backend.validate()
        job = backend.train(dataset_path, spec)
    except RuntimeError as exc:
        backend_name = settings.training.backend
        try:
            backend_name = backend.name
        except AttributeError:
            pass
        job = _failed_job(backend_name, dataset_path, spec.output_dir, exc)
    store.write_training_result(run_paths.training_result_path, job)
    artifacts.append(ArtifactRecord("training result", run_paths.training_result_path))

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
        "artifacts": {record.label: str(record.path) for record in artifacts},
        "demo": demo,
    }
    store.write_report(run_paths.report_path, report)
    artifacts.append(ArtifactRecord("report", run_paths.report_path))

    pipeline_run = PipelineRun(run_id=run_paths.root.name, status=job.status, paths=run_paths)
    RunRepository().save(pipeline_run)
    artifacts.append(ArtifactRecord("run metadata", run_paths.root / "run.json"))

    return pipeline_run
