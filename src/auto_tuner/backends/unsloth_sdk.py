from __future__ import annotations

import inspect
import platform
import sys
from pathlib import Path

from auto_tuner.models.training import TrainingJob, TrainingSpec


class UnslothTrainingBackend:
    name = "unsloth"

    def validate(self) -> None:
        unsupported = self._unsupported_reason()
        if unsupported:
            raise RuntimeError(unsupported)
        if missing := self._dependency_missing_reason():
            raise RuntimeError(missing)

    @staticmethod
    def _dependency_missing_reason() -> str | None:
        unsupported = UnslothTrainingBackend._unsupported_reason()
        if unsupported:
            return unsupported
        try:
            import datasets  # noqa: F401
            import trl  # noqa: F401
            import unsloth  # noqa: F401
        except ImportError:
            return (
                "Unsloth backend requires the optional 'unsloth' dependency group. "
                "Install with: uv sync --extra unsloth"
            )
        except Exception as exc:
            return f"Unsloth dependency is present but failed to initialize: {exc}"
        return None

    def train(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        if missing := self._dependency_missing_reason():
            return TrainingJob(
                job_id=f"unsloth-{dataset_path.stem}",
                status="unsupported",
                backend=self.name,
                mode="guarded",
                summary=missing,
                artifacts={"dataset_path": str(dataset_path), "output_dir": spec.output_dir},
                warnings=[missing],
            )

        try:  # pragma: no cover - requires live compatible environment
            from datasets import load_dataset
            from trl import SFTConfig, SFTTrainer
            from unsloth import FastLanguageModel

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
                use_rslora=False,
                loftq_config=None,
            )
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
            trainer_args = SFTConfig(
                output_dir=spec.output_dir,
                num_train_epochs=spec.num_train_epochs,
                per_device_train_batch_size=spec.per_device_train_batch_size,
                learning_rate=spec.learning_rate,
                report_to="none",
                dataset_text_field="text",
                max_length=spec.max_seq_length,
            )
            trainer_kwargs = {
                "model": model,
                "args": trainer_args,
                "train_dataset": dataset,
            }
            signature = inspect.signature(SFTTrainer)
            if "processing_class" in signature.parameters:
                trainer_kwargs["processing_class"] = tokenizer
            else:
                trainer_kwargs["tokenizer"] = tokenizer
            trainer = SFTTrainer(**trainer_kwargs)
            result = trainer.train()
            model.save_pretrained(spec.output_dir)
            tokenizer.save_pretrained(spec.output_dir)
            loss = getattr(result, "training_loss", None)
            return TrainingJob(
                job_id=f"unsloth-{dataset_path.stem}",
                status="completed",
                backend=self.name,
                mode="live",
                summary="Completed live Unsloth fine-tuning run.",
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
                job_id=f"unsloth-{dataset_path.stem}",
                status="failed",
                backend=self.name,
                mode="live",
                summary=f"Unsloth run failed: {exc}",
                artifacts={"dataset_path": str(dataset_path), "output_dir": spec.output_dir},
                warnings=[str(exc)],
            )

    @staticmethod
    def _unsupported_reason() -> str | None:
        if platform.system() == "Darwin":
            return (
                "Live Unsloth fine-tuning is guarded on macOS; "
                "use a Linux or Windows GPU environment."
            )
        if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
            return "Live Unsloth fine-tuning requires Python 3.11-3.13."
        return None
