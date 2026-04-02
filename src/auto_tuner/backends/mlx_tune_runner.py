from __future__ import annotations

import argparse
from pathlib import Path

from auto_tuner.models.training import TrainingJob


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="auto-tuner-mlx-tune-runner")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-seq-length", required=True, type=int)
    parser.add_argument("--load-in-4bit", required=True)
    parser.add_argument("--num-train-epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lora-rank", required=True, type=int)
    return parser.parse_args()


def main() -> int:  # pragma: no cover - live only
    args = _parse_args()
    dataset_path = Path(args.dataset)
    job_path = Path(args.job)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_in_4bit = str(args.load_in_4bit).lower() in {"1", "true", "yes", "y"}

    try:
        from datasets import load_dataset
        from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        base_model_name_or_path: str | None = None
        try:
            base_model_name_or_path = model.config.name_or_path
        except Exception:
            base_model_name_or_path = None

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        param_count: int | None = None
        try:
            param_count = sum(p.numel() for p in model.parameters())
        except Exception:
            param_count = None

        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        trainer_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=trainer_args,
            max_seq_length=args.max_seq_length,
        )
        result = trainer.train()

        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        loss: float | None = None
        try:
            loss = float(result.training_loss)
        except Exception:
            loss = None

        job = TrainingJob(
            job_id=f"mlx-tune-{dataset_path.stem}",
            status="completed",
            backend="mlx_tune",
            mode="live",
            summary="Completed live MLX-Tune fine-tuning run.",
            artifacts={
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "adapter_dir": str(output_dir),
                "base_model_name_or_path": base_model_name_or_path or "",
            },
            metrics={
                "training_loss": loss if loss is not None else "unknown",
                "num_train_epochs": args.num_train_epochs,
                "base_model_parameters": param_count if param_count is not None else "unknown",
                "base_model_parameters_b": (
                    (param_count / 1_000_000_000) if param_count is not None else "unknown"
                ),
            },
        )
        job_path.write_text(job.model_dump_json(indent=2))
        return 0
    except Exception as exc:
        job = TrainingJob(
            job_id=f"mlx-tune-{dataset_path.stem}",
            status="failed",
            backend="mlx_tune",
            mode="live",
            summary=f"MLX-Tune run failed: {exc}",
            artifacts={"dataset_path": str(dataset_path), "output_dir": str(output_dir)},
            warnings=[str(exc)],
        )
        job_path.write_text(job.model_dump_json(indent=2))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

