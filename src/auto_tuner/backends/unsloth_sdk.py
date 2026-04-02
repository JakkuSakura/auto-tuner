from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import httpx

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
            if spec.method == "grpo":
                return self._train_grpo(dataset_path, spec)
            if spec.method == "sft":
                return self._train_sft(dataset_path, spec)
            raise RuntimeError(f"Unsupported training.method: {spec.method}")
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

    def _train_sft(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec.model_name,
            max_seq_length=spec.max_seq_length,
            load_in_4bit=spec.load_in_4bit,
        )
        base_model_name_or_path: str | None = None
        try:
            base_model_name_or_path = model.config.name_or_path
        except Exception:
            base_model_name_or_path = None

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
        param_count: int | None = None
        try:
            param_count = sum(p.numel() for p in model.parameters())
        except Exception:
            param_count = None

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
        loss: float | None = None
        try:
            loss = result.training_loss
        except Exception:
            loss = None
        return TrainingJob(
            job_id=f"unsloth-{dataset_path.stem}",
            status="completed",
            backend=self.name,
            mode="live",
            summary="Completed live Unsloth SFT fine-tuning run.",
            artifacts={
                "dataset_path": str(dataset_path),
                "output_dir": spec.output_dir,
                "adapter_dir": spec.output_dir,
                "base_model_name_or_path": base_model_name_or_path or "",
            },
            metrics={
                "training_loss": loss if loss is not None else "unknown",
                "num_train_epochs": spec.num_train_epochs,
                "base_model_parameters": param_count if param_count is not None else "unknown",
                "base_model_parameters_b": (
                    (param_count / 1_000_000_000) if param_count is not None else "unknown"
                ),
            },
        )

    def _train_grpo(self, dataset_path: Path, spec: TrainingSpec) -> TrainingJob:
        if spec.grpo is None:
            raise RuntimeError("training.method='grpo' requires training.grpo configuration.")

        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel

        try:
            from unsloth import PatchGRPOTrainer  # type: ignore[import-not-found]
        except Exception:
            PatchGRPOTrainer = None

        if spec.grpo.use_vllm:
            try:
                import vllm  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "training.grpo.use_vllm=true requires the optional 'vllm' dependency."
                ) from exc

        if PatchGRPOTrainer is not None:
            PatchGRPOTrainer()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec.model_name,
            max_seq_length=spec.max_seq_length,
            load_in_4bit=spec.load_in_4bit,
        )
        base_model_name_or_path: str | None = None
        try:
            base_model_name_or_path = model.config.name_or_path
        except Exception:
            base_model_name_or_path = None

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

        judge = spec.grpo.judge
        rules = spec.grpo.rules
        judge_calls_path = dataset_path.parent / "judge_calls.jsonl"
        judge_cache: dict[str, float] = {}

        def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

        def _rule_score(completion: str) -> float:
            score = 1.0
            if rules.require_python_fence and (
                "```python" not in completion or "```" not in completion
            ):
                return 0.0
            lowered = completion.lower()
            for needle in rules.forbidden_substrings:
                if needle.lower() in lowered:
                    score = max(0.0, score - rules.forbidden_substring_penalty)
            return score

        def _parse_score(text: str) -> float:
            match = re.search(r"(?P<num>0(?:\\.\\d+)?|1(?:\\.0+)?)", text)
            value = float(match.group("num")) if match else 0.0
            return max(0.0, min(1.0, value))

        def _judge_score(prompt: str, completion: str) -> float:
            if judge.provider != "openrouter":
                raise RuntimeError(f"Unsupported judge provider: {judge.provider}")
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for GRPO judge scoring.")

            request_payload = {
                "model": judge.model,
                "messages": [
                    {"role": "system", "content": judge.system_prompt},
                    {
                        "role": "user",
                        "content": judge.user_prompt_template.format(
                            prompt=prompt, completion=completion
                        ),
                    },
                ],
            }

            started = time.perf_counter()
            timeout_seconds = float(judge.timeout_seconds)
            timeout = httpx.Timeout(
                connect=min(10.0, timeout_seconds),
                read=timeout_seconds,
                write=min(10.0, timeout_seconds),
                pool=min(10.0, timeout_seconds),
            )

            last_error: Exception | None = None
            content: str | None = None
            for attempt in range(1, 4):
                try:
                    with httpx.Client(
                        base_url=judge.base_url,
                        timeout=timeout,
                        limits=httpx.Limits(
                            max_keepalive_connections=5, max_connections=10
                        ),
                    ) as client:
                        response = client.post(
                            "/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "HTTP-Referer": judge.http_referer,
                                "X-Title": judge.app_name,
                            },
                            json=request_payload,
                        )
                        response.raise_for_status()
                        content = str(response.json()["choices"][0]["message"]["content"])
                        break
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                    httpx.HTTPStatusError,
                ) as exc:
                    last_error = exc
                    if isinstance(exc, httpx.HTTPStatusError):
                        status = exc.response.status_code
                        if 400 <= status < 500 and status != 429:
                            raise
                time.sleep(min(2.0, 0.3 * attempt))

            if content is None:
                assert last_error is not None
                raise last_error
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            score = _parse_score(content)
            _append_jsonl(
                judge_calls_path,
                {
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                    "model": judge.model,
                    "score": score,
                    "latency_ms": elapsed_ms,
                },
            )
            return score

        def llm_judge_reward_func(prompts, completions, **kwargs):  # type: ignore[no-untyped-def]
            _ = kwargs
            flat_prompts: list[str] = []
            flat_completions: list[str] = []

            if completions and isinstance(completions[0], list):
                for idx, group in enumerate(completions):
                    prompt = str(prompts[idx])
                    for completion in group:
                        flat_prompts.append(prompt)
                        flat_completions.append(str(completion))
            else:
                flat_prompts = [str(p) for p in prompts]
                flat_completions = [str(c) for c in completions]

            rewards: list[float] = []
            for prompt, completion in zip(flat_prompts, flat_completions, strict=False):
                cache_key = hashlib.sha256(
                    (prompt + "\n---\n" + completion).encode("utf-8")
                ).hexdigest()
                cached = judge_cache.get(cache_key)
                if cached is not None:
                    rewards.append(cached)
                    continue
                scored = _judge_score(prompt, completion)
                judge_cache[cache_key] = scored
                rewards.append(scored)
            return rewards

        def rule_reward_func(prompts, completions, **kwargs):  # type: ignore[no-untyped-def]
            _ = prompts
            _ = kwargs
            if completions and isinstance(completions[0], list):
                flattened: list[str] = []
                for group in completions:
                    flattened.extend([str(x) for x in group])
                return [_rule_score(c) for c in flattened]
            return [_rule_score(str(c)) for c in completions]

        config_kwargs = {
            "output_dir": spec.output_dir,
            "learning_rate": spec.learning_rate,
            "per_device_train_batch_size": spec.per_device_train_batch_size,
            "num_train_epochs": spec.num_train_epochs,
            "use_vllm": spec.grpo.use_vllm,
            "num_generations": spec.grpo.num_generations,
            "max_prompt_length": spec.grpo.max_prompt_length,
            "max_completion_length": spec.grpo.max_completion_length,
            "report_to": "none",
        }
        config_sig = inspect.signature(GRPOConfig)
        filtered = {k: v for k, v in config_kwargs.items() if k in config_sig.parameters}
        trainer_args = GRPOConfig(**filtered)

        trainer_kwargs: dict[str, object] = {
            "model": model,
            "args": trainer_args,
            "train_dataset": dataset,
        }
        trainer_sig = inspect.signature(GRPOTrainer)
        if "reward_funcs" in trainer_sig.parameters:
            trainer_kwargs["reward_funcs"] = [llm_judge_reward_func, rule_reward_func]
        elif "reward_function" in trainer_sig.parameters:
            trainer_kwargs["reward_function"] = llm_judge_reward_func
        else:
            raise RuntimeError(
                "Unsupported trl.GRPOTrainer signature (missing reward function parameter)."
            )

        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = GRPOTrainer(**trainer_kwargs)
        result = trainer.train()
        model.save_pretrained(spec.output_dir)
        tokenizer.save_pretrained(spec.output_dir)

        metrics: dict[str, float | int | str] = {
            "num_train_epochs": spec.num_train_epochs,
            "num_generations": spec.grpo.num_generations,
        }
        try:
            metrics["train_loss"] = float(result.training_loss)
        except Exception:
            pass

        return TrainingJob(
            job_id=f"unsloth-grpo-{dataset_path.stem}",
            status="completed",
            backend=self.name,
            mode="live",
            summary="Completed live Unsloth GRPO run (online scoring).",
            artifacts={
                "dataset_path": str(dataset_path),
                "output_dir": spec.output_dir,
                "adapter_dir": spec.output_dir,
                "base_model_name_or_path": base_model_name_or_path or "",
                "judge_calls": str(judge_calls_path),
            },
            metrics=metrics,
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
