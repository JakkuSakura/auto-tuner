from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from auto_tuner.config import OpenRouterConfig
from auto_tuner.llm.openrouter import PromptBundle, build_prompt_provider
from auto_tuner.models.dataset import GradeResult


@dataclass(frozen=True)
class GeneratedTaskExample:
    task: str
    generation_prompt: str
    task_path: Path


class SupervisorAgent(Protocol):
    def build_prompts(self, meta_prompt: str) -> PromptBundle: ...

    def generate_task_example(
        self,
        *,
        workspace_dir: Path,
        meta_prompt: str,
        generation_prompt: str,
        example_id: int,
        theme_hint: str,
    ) -> GeneratedTaskExample: ...

    def grade_example(
        self,
        *,
        meta_prompt: str,
        grading_prompt: str,
        task: str,
        naive_solution: str,
    ) -> GradeResult: ...


class OpenRouterSupervisorAgent:
    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._prompt_provider = build_prompt_provider(config)

    def build_prompts(self, meta_prompt: str) -> PromptBundle:
        return self._prompt_provider.build_prompts(meta_prompt)

    def generate_task_example(
        self,
        *,
        workspace_dir: Path,
        meta_prompt: str,
        generation_prompt: str,
        example_id: int,
        theme_hint: str,
    ) -> GeneratedTaskExample:
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        workspace_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_generation_request(
            meta_prompt=meta_prompt,
            generation_prompt=generation_prompt,
            example_id=example_id,
            theme_hint=theme_hint,
        )

        try:
            response = _openrouter_complete_markdown(self._config, prompt)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed during generation: {exc}") from exc

        task = _extract_task_markdown(response)

        task_path = workspace_dir / "task.md"
        task_path.write_text(task)

        return GeneratedTaskExample(
            task=task,
            generation_prompt=generation_prompt,
            task_path=task_path,
        )

    def grade_example(
        self,
        *,
        meta_prompt: str,
        grading_prompt: str,
        task: str,
        naive_solution: str,
    ) -> GradeResult:
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        prompt = _build_grading_request(
            meta_prompt=meta_prompt,
            grading_prompt=grading_prompt,
            task=task,
            naive_solution=naive_solution,
        )
        try:
            response = _openrouter_complete_markdown(self._config, prompt)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed during grading: {exc}") from exc

        grade_payload = _extract_json_from_markdown(response)
        refined_solution = _extract_python_from_markdown(response)
        if refined_solution is None:
            raise ValueError("Grading response did not contain a fenced ```python code block.")

        return GradeResult(
            passed=bool(grade_payload.get("passed", False)),
            violations=list(grade_payload.get("violations", [])),
            severity=str(grade_payload.get("severity", "none")),
            suggestion=str(grade_payload.get("suggestion", "")),
            grading_prompt=grading_prompt,
            refined_solution=refined_solution,
        )


def _build_generation_request(
    *,
    meta_prompt: str,
    generation_prompt: str,
    example_id: int,
    theme_hint: str,
) -> str:
    return "\n".join(
        [
            "# auto-tuner example generation request",
            "",
            f"- example_id: `{example_id}`",
            f"- theme_hint: `{theme_hint}`",
            "",
            "## Meta prompt goal",
            "",
            meta_prompt.strip(),
            "",
            "## Generation prompt",
            "",
            generation_prompt.strip(),
            "",
            "## Output format (strict)",
            "",
            "Return Markdown with exactly 1 section and exactly 1 fenced code block:",
            "",
            "### task.md",
            "```markdown",
            "...",
            "```",
            "",
            "Constraints:",
            "- The task must be meaningfully different from other examples.",
            "- Keep code small but realistic (multiple functions/classes if needed).",
            "",
        ]
    )

def _openrouter_complete_markdown(config: OpenRouterConfig, prompt: str) -> str:
    with httpx.Client(base_url=config.base_url, timeout=60.0) as client:
        response = client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": config.http_referer,
                "X-Title": config.app_name,
            },
            json={
                "model": config.prompt_model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]

_TASK_BLOCK = re.compile(r"###\s+task\.md\s*\n```markdown\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_task_markdown(markdown_text: str) -> str:
    match = _TASK_BLOCK.search(markdown_text)
    if not match:
        raise ValueError("Generation response did not contain a ### task.md fenced ```markdown block.")
    body = match["body"].strip()
    if not body:
        raise ValueError("Generation response task.md block was empty.")
    return body + "\n"


def _build_grading_request(
    *,
    meta_prompt: str,
    grading_prompt: str,
    task: str,
    naive_solution: str,
) -> str:
    return "\n".join(
        [
            "# auto-tuner grading request",
            "",
            "## Meta prompt goal",
            "",
            meta_prompt.strip(),
            "",
            "## Grading rubric",
            "",
            grading_prompt.strip(),
            "",
            "## Task",
            "",
            task.strip(),
            "",
            "## Naive solution",
            "",
            "```python",
            naive_solution.rstrip(),
            "```",
            "",
            "## Output format (strict)",
            "",
            "Return Markdown with exactly 2 fenced code blocks:",
            "",
            "```json",
            '{ "passed": true, "severity": "none", "violations": [], "suggestion": "" }',
            "```",
            "",
            "```python",
            "# full refined_solution.py file",
            "```",
            "",
            "Constraints for refined_solution.py:",
            "- Must satisfy the meta prompt goal.",
            "- Must preserve task intent and external behavior.",
            "- Must avoid reflection/introspection and dynamic dispatch tricks.",
            "",
        ]
    )


_JSON_FENCE = re.compile(r"```json\s*\n(?P<body>.*?)\n```", re.DOTALL)
_PYTHON_FENCE = re.compile(r"```python\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_json_from_markdown(markdown_text: str) -> dict[str, object]:
    match = _JSON_FENCE.search(markdown_text)
    if not match:
        raise ValueError("Grading response did not contain a fenced ```json code block.")
    body = match["body"].strip()
    if not body:
        raise ValueError("Grading response json block was empty.")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Grading response json block must be an object.")
    return payload


def _extract_python_from_markdown(markdown_text: str) -> str | None:
    match = _PYTHON_FENCE.search(markdown_text)
    if not match:
        return None
    body = match["body"].strip()
    if not body:
        return None
    return body + "\n"
