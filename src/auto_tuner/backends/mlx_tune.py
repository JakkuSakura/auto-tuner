from __future__ import annotations

import platform
from pathlib import Path

from auto_tuner.models.training import TrainingJob, TrainingSpec


class MlxTuneTrainingBackend:
    name = "mlx_tune"

    def validate(self) -> None:
        if platform.system() != "Darwin":
            return None
        try:
            import mlx_tune  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MLX-Tune backend requires the optional 'mlx_tune' dependency group."
            ) from exc

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
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

        try:  # pragma: no cover - requires live compatible environment
            from datasets import load_dataset
            from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=spec.model_name,
                max_seq_length=spec.max_seq_length,
                load_in_4bit=spec.load_in_4bit,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=spec.lora_rank,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=spec.lora_rank,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
            trainer_args = SFTConfig(
                output_dir=spec.output_dir,
                num_train_epochs=spec.num_train_epochs,
                per_device_train_batch_size=spec.per_device_train_batch_size,
                learning_rate=spec.learning_rate,
                dataset_text_field="text",
                max_seq_length=spec.max_seq_length,
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=trainer_args,
                max_seq_length=spec.max_seq_length,
            )
            result = trainer.train()

            # Save adapters (Unsloth-compatible API).
            model.save_pretrained(spec.output_dir)
            tokenizer.save_pretrained(spec.output_dir)

            loss = getattr(result, "training_loss", None)
            return TrainingJob(
                job_id=f"mlx-tune-{dataset_path.stem}",
                status="completed",
                backend=self.name,
                mode="live",
                summary="Completed live MLX-Tune fine-tuning run.",
                artifacts={
                    "dataset_path": str(dataset_path),
                    "output_dir": spec.output_dir,
                    "adapter_dir": spec.output_dir,
                },
                metrics={
                    "training_loss": loss if loss is not None else "unknown",
                    "num_train_epochs": spec.num_train_epochs,
                },
            )
        except Exception as exc:  # pragma: no cover - live path only
            return TrainingJob(
                job_id=f"mlx-tune-{dataset_path.stem}",
                status="failed",
                backend=self.name,
                mode="live",
                summary=f"MLX-Tune run failed: {exc}",
                artifacts={"dataset_path": str(dataset_path), "output_dir": spec.output_dir},
                warnings=[str(exc)],
            )
