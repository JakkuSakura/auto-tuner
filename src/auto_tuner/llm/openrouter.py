from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import httpx

from auto_tuner.config import OpenRouterConfig


@dataclass
class PromptBundle:
    meta_prompt: str
    generation_prompt: str
    grading_prompt: str
    source: str


class OpenRouterPromptProvider:
    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config

    def build_prompts(self, meta_prompt: str) -> PromptBundle:
        if not self.config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        try:
            generation_prompt = openrouter_chat_completion(
                self.config,
                model=self.config.prompt_model,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "Output only the final answer. Do not output reasoning.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "You design dataset-generation prompts for an auto fine-tuning pipeline. "
                            "Given the user's initial meta-prompt goal, "
                            "return one concise generation prompt only. "
                            "The generation prompt will be used to generate synthetic refactoring tasks.\n\n"
                            "Output rules (strict):\n"
                            "- Output only the generation prompt text.\n"
                            "- No analysis, no explanations, no Markdown headers.\n\n"
                            "Design requirements (important):\n"
                            "- The generated tasks must ask for a single Python file implementation/refactor.\n"
                            "- The task should include a small 'starting point' snippet that intentionally violates the goal.\n"
                            "- The task should specify acceptance criteria for the refactor.\n"
                            "- The task must NOT include the final solution.\n"
                            "- The task must not mention the user's codebase.\n\n"
                            f"User goal: {meta_prompt}"
                        ),
                    }
                ],
            )
            grading_prompt = openrouter_chat_completion(
                self.config,
                model=self.config.grading_model,
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "Output only the final answer. Do not output reasoning.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "You design strict grading prompts for an auto fine-tuning pipeline. "
                            "Given the user's initial meta-prompt goal, "
                            "return one concise grading rubric prompt only."
                            "\n\n"
                            "Output rules (strict):\n"
                            "- Output only the grading rubric prompt text.\n"
                            "- No analysis, no explanations, no Markdown headers.\n\n"
                            f"User goal: {meta_prompt}"
                        ),
                    }
                ],
            )

            return PromptBundle(
                meta_prompt=meta_prompt,
                generation_prompt=generation_prompt.strip(),
                grading_prompt=grading_prompt.strip(),
                source="openrouter",
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc


class OpenRouterEmptyContentError(RuntimeError):
    pass


def _timeout(config: OpenRouterConfig) -> httpx.Timeout:
    timeout_seconds = float(config.timeout_seconds)
    # Important: httpx's float timeout is per-operation, not "total request time".
    # Keep read time reasonably tight to avoid hanging forever on a slow stream.
    return httpx.Timeout(
        connect=min(10.0, timeout_seconds),
        read=timeout_seconds,
        write=min(10.0, timeout_seconds),
        pool=min(10.0, timeout_seconds),
    )


def openrouter_chat_completion(
    config: OpenRouterConfig,
    *,
    model: str,
    messages: list[dict[str, object]],
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    if not config.api_key:
        raise RuntimeError(
            "OpenRouter API key is required. "
            "Set OPENROUTER_API_KEY or configure openrouter.api_key."
        )

    attempts = int(config.max_attempts)
    if attempts < 1:
        attempts = 1

    last_error: Exception | None = None
    request_max_tokens = int(max_tokens)
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(
                base_url=config.base_url,
                timeout=_timeout(config),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ) as client:
                response = client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "HTTP-Referer": config.http_referer,
                        "X-Title": config.app_name,
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": int(request_max_tokens),
                        "temperature": float(temperature),
                    },
                )
                response.raise_for_status()
                payload = response.json()

            preview = ""
            try:
                import json as _json

                preview = _json.dumps(payload)[:2000]
            except Exception:
                preview = str(payload)[:2000]

            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise OpenRouterEmptyContentError(
                    "OpenRouter returned a response without choices. "
                    f"payload_preview={preview}"
                )

            choice = choices[0]
            if isinstance(choice, dict) and choice.get("error"):
                raise OpenRouterEmptyContentError(
                    "OpenRouter returned an error choice payload. "
                    f"payload_preview={preview}"
                )

            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                raise OpenRouterEmptyContentError(
                    "OpenRouter returned a choice without a message object. "
                    f"payload_preview={preview}"
                )

            content = message.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                content = "".join(parts)

            if isinstance(content, str) and content.strip():
                return content

            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            reasoning = message.get("reasoning")
            if (
                content is None
                and isinstance(reasoning, str)
                and reasoning.strip()
                and finish_reason == "length"
            ):
                raise OpenRouterEmptyContentError(
                    "OpenRouter produced reasoning-only output and hit max_tokens before returning "
                    "final content. Increase max_tokens (or reduce reasoning) and retry. "
                    f"payload_preview={preview}"
                )

            raise OpenRouterEmptyContentError(
                "OpenRouter returned an empty/non-string message content. "
                f"payload_preview={preview}"
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            last_error = exc
        except OpenRouterEmptyContentError as exc:
            last_error = exc
            # Some reasoning-style providers return `content=null` with a large `reasoning` field,
            # and only emit the final answer in `content` after the reasoning is complete.
            # If the response hit `finish_reason=length`, retry once with a higher token limit.
            text = str(exc)
            if (
                "reasoning-only output" in text
                and request_max_tokens < 8192
                and attempt < attempts
            ):
                request_max_tokens = min(8192, max(2048, request_max_tokens * 4))
                continue
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            if 400 <= status < 500 and status != 429:
                raise

        if attempt < attempts:
            base_delay = min(8.0, 0.8 * (2 ** (attempt - 1)))
            time.sleep(base_delay + random.random() * 0.2)

    assert last_error is not None
    raise last_error


def build_prompt_provider(config: OpenRouterConfig) -> OpenRouterPromptProvider:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and not config.api_key:
        config = config.model_copy(update={"api_key": env_key})
    return OpenRouterPromptProvider(config)
