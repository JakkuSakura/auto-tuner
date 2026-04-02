from __future__ import annotations

import platform
from pathlib import Path
import subprocess
import sys

from auto_tuner.models.training import TrainingJob, TrainingSpec


class MlxTuneTrainingBackend:
    name = "mlx_tune"

    def validate(self) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError(
                "Live MLX-Tune fine-tuning is only available on macOS (Apple Silicon)."
            )
        try:
            import mlx_tune  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MLX-Tune backend requires the optional 'mlx_tune' dependency group."
            ) from exc

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        if spec.method != "sft":
            raise RuntimeError(
                f"mlx_tune backend only supports training.method='sft' (got {spec.method!r})."
            )
        if platform.system() != "Darwin":
            summary = "Live MLX-Tune fine-tuning is only available on macOS (Apple Silicon)."
            return TrainingJob(
                job_id=f"mlx-tune-{dataset_path.stem}",
                status="unsupported",
                backend=self.name,
                mode="guarded",
                summary=summary,
                artifacts={"dataset_path": str(dataset_path), "output_dir": spec.output_dir},
                warnings=[summary],
            )

        # Run MLX-Tune training in a subprocess. The underlying MLX/Metal stack can
        # hard-abort the process (e.g. GPU command buffer failures), which cannot be
        # reliably caught as a Python exception.
        output_dir = Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "mlx_tune_stdout.log"
        stderr_path = output_dir / "mlx_tune_stderr.log"
        job_path = output_dir / "mlx_tune_job.json"

        argv = [
            sys.executable,
            "-m",
            "auto_tuner.backends.mlx_tune_runner",
            "--dataset",
            str(dataset_path),
            "--job",
            str(job_path),
            "--model",
            spec.model_name,
            "--max-seq-length",
            str(spec.max_seq_length),
            "--load-in-4bit",
            "true" if spec.load_in_4bit else "false",
            "--num-train-epochs",
            str(spec.num_train_epochs),
            "--batch-size",
            str(spec.per_device_train_batch_size),
            "--learning-rate",
            str(spec.learning_rate),
            "--output-dir",
            str(output_dir),
            "--lora-rank",
            str(spec.lora_rank),
        ]

        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(argv, stdout=out, stderr=err, text=True)

        if job_path.exists():
            try:
                return TrainingJob.model_validate_json(job_path.read_text())
            except Exception as exc:
                return TrainingJob(
                    job_id=f"mlx-tune-{dataset_path.stem}",
                    status="failed",
                    backend=self.name,
                    mode="live",
                    summary=f"MLX-Tune produced an unreadable job file: {exc}",
                    artifacts={
                        "dataset_path": str(dataset_path),
                        "output_dir": str(output_dir),
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "job": str(job_path),
                    },
                    warnings=[str(exc)],
                )

        summary = (
            f"MLX-Tune subprocess failed with exit_code={proc.returncode}. "
            "See mlx_tune_stderr.log for details."
        )
        return TrainingJob(
            job_id=f"mlx-tune-{dataset_path.stem}",
            status="failed",
            backend=self.name,
            mode="live",
            summary=summary,
            artifacts={
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            warnings=[summary],
        )
