from __future__ import annotations

import os
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
            return self._fallback(meta_prompt)

        try:
            with httpx.Client(base_url=self.config.base_url, timeout=30.0) as client:
                generation_prompt = self._complete(
                    client,
                    self.config.prompt_model,
                    (
                        "You design dataset-generation prompts for an auto fine-tuning pipeline. "
                        "Given the user's initial meta-prompt goal, return one concise generation prompt only. "
                        "The generation prompt should create concrete code snippets and tasks from the high-level goal.\n\n"
                        f"User goal: {meta_prompt}"
                    ),
                )
                grading_prompt = self._complete(
                    client,
                    self.config.grading_model,
                    (
                        "You design strict grading prompts for an auto fine-tuning pipeline. "
                        "Given the user's initial meta-prompt goal, return one concise grading rubric prompt only.\n\n"
                        f"User goal: {meta_prompt}"
                    ),
                )

            return PromptBundle(
                meta_prompt=meta_prompt,
                generation_prompt=generation_prompt.strip(),
                grading_prompt=grading_prompt.strip(),
                source="openrouter",
            )
        except httpx.HTTPError:
            return self._fallback(meta_prompt)

    def _complete(self, client: httpx.Client, model: str, prompt: str) -> str:
        response = client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "HTTP-Referer": self.config.http_referer,
                "X-Title": self.config.app_name,
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

    @staticmethod
    def _fallback(meta_prompt: str) -> PromptBundle:
        generation_prompt = (
            "Generate Python refactoring tasks based on this meta-prompt goal: "
            f"{meta_prompt}. Include a naive solution that uses dynamic attribute access "
            "and a clean solution that uses direct attributes or explicit mappings."
        )
        grading_prompt = (
            "Grade whether the assistant output fully satisfies this meta-prompt goal: "
            f"{meta_prompt}. Fail if the answer still uses getattr(), hasattr(), __dict__, "
            "vars(), or unnecessary dynamic dispatch."
        )
        return PromptBundle(
            meta_prompt=meta_prompt,
            generation_prompt=generation_prompt,
            grading_prompt=grading_prompt,
            source="fallback",
        )


def build_prompt_provider(config: OpenRouterConfig) -> OpenRouterPromptProvider:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and not config.api_key:
        config = config.model_copy(update={"api_key": env_key})
    return OpenRouterPromptProvider(config)
