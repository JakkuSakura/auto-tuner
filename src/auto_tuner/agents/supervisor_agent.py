from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from auto_tuner.config import OpenRouterConfig
from auto_tuner.llm.openrouter import (
    PromptBundle,
    build_prompt_provider,
    openrouter_chat_completion,
)
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

    def generate_naive_solution(
        self,
        *,
        workspace_dir: Path,
        task_markdown: str,
        example_id: int,
        theme_hint: str,
    ) -> str: ...

    def grade_example(
        self,
        *,
        workspace_dir: Path | None = None,
        meta_prompt: str,
        grading_prompt: str,
        task: str,
        naive_solution: str,
    ) -> GradeResult: ...


class OpenRouterSupervisorAgent:
    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._prompt_provider = build_prompt_provider(config)

    @staticmethod
    def _system_messages() -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Output only the final answer. "
                    "Do not output analysis or reasoning. "
                    "Do not include any extra commentary."
                ),
            }
        ]

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
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        workspace_dir.mkdir(parents=True, exist_ok=True)

        prompt = _build_generation_request(
            meta_prompt=meta_prompt,
            generation_prompt=generation_prompt,
            example_id=example_id,
            theme_hint=theme_hint,
        )

        (workspace_dir / "task_generation_prompt.md").write_text(prompt)

        try:
            response = openrouter_chat_completion(
                self._config,
                model=self._config.prompt_model,
                max_tokens=1536,
                messages=[*self._system_messages(), {"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - boundary error record
            (workspace_dir / "task_generation_error.md").write_text(str(exc))
            raise RuntimeError(f"OpenRouter request failed during generation: {exc}") from exc

        (workspace_dir / "task_generation_output.md").write_text(response)

        task = _extract_task_markdown(response)

        task_path = workspace_dir / "task.md"
        task_path.write_text(task)

        return GeneratedTaskExample(
            task=task,
            generation_prompt=generation_prompt,
            task_path=task_path,
        )

    def generate_naive_solution(
        self,
        *,
        workspace_dir: Path,
        task_markdown: str,
        example_id: int,
        theme_hint: str,
    ) -> str:
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        workspace_dir.mkdir(parents=True, exist_ok=True)
        prompt = _build_naive_solution_request(
            task_markdown=task_markdown,
            example_id=example_id,
            theme_hint=theme_hint,
        )
        (workspace_dir / "naive_generation_prompt.md").write_text(prompt)
        try:
            response = openrouter_chat_completion(
                self._config,
                model=self._config.prompt_model,
                max_tokens=2048,
                messages=[*self._system_messages(), {"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - boundary error record
            (workspace_dir / "naive_generation_error.md").write_text(str(exc))
            raise RuntimeError(f"OpenRouter request failed during naive generation: {exc}") from exc

        (workspace_dir / "naive_generation_output.md").write_text(response)

        naive_solution = _extract_python_from_markdown(response)
        if naive_solution is None:
            raise RuntimeError(
                "Naive generation response did not contain a fenced ```python code block. "
                f"See: {workspace_dir / 'naive_generation_output.md'}"
            )
        if naive_solution.strip() in {"...", "# ..."}:
            raise RuntimeError(
                "Naive generation returned a placeholder ('...') instead of a real file. "
                f"See: {workspace_dir / 'naive_generation_output.md'}"
            )
        return naive_solution

    def grade_example(
        self,
        *,
        workspace_dir: Path | None = None,
        meta_prompt: str,
        grading_prompt: str,
        task: str,
        naive_solution: str,
    ) -> GradeResult:
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        prompt = _build_grading_request(
            meta_prompt=meta_prompt,
            grading_prompt=grading_prompt,
            task=task,
            naive_solution=naive_solution,
        )
        if workspace_dir is not None:
            workspace_dir.mkdir(parents=True, exist_ok=True)
            (workspace_dir / "grading_prompt.md").write_text(prompt)
        try:
            response = openrouter_chat_completion(
                self._config,
                model=self._config.grading_model,
                max_tokens=2048,
                messages=[*self._system_messages(), {"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - boundary error record
            if workspace_dir is not None:
                (workspace_dir / "grading_error.md").write_text(str(exc))
            raise RuntimeError(f"OpenRouter request failed during grading: {exc}") from exc

        if workspace_dir is not None:
            (workspace_dir / "grading_output.md").write_text(response)

        grade_payload = _extract_json_from_markdown(response)
        clean_solution = _extract_python_from_markdown(response)
        if clean_solution is None:
            raise ValueError("Grading response did not contain a fenced ```python code block.")

        return GradeResult(
            passed=bool(grade_payload.get("passed", False)),
            violations=list(grade_payload.get("violations", [])),
            severity=str(grade_payload.get("severity", "none")),
            suggestion=str(grade_payload.get("suggestion", "")),
            grading_prompt=grading_prompt,
            clean_solution=clean_solution,
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


def _build_naive_solution_request(*, task_markdown: str, example_id: int, theme_hint: str) -> str:
    return "\n".join(
        [
            "# auto-tuner naive solution request",
            "",
            f"- example_id: `{example_id}`",
            f"- theme_hint: `{theme_hint}`",
            "",
            "Write a *naive* baseline implementation for the task below.",
            "",
            "Rules (strict):",
            "- The naive solution should intentionally use at least one reflection/introspection pattern",
            "  (e.g. getattr / hasattr / __dict__ / vars) in a central part of the logic.",
            "- Keep the code small and realistic; avoid boilerplate and repeated dunder-method spam.",
            "- No placeholders like '...'.",
            "",
            "## Task",
            "",
            task_markdown.strip(),
            "",
            "## Output format (strict)",
            "",
            "Return Markdown with exactly 1 fenced code block:",
            "",
            "```python",
            "# naive_solution.py",
            "...",
            "```",
            "",
        ]
    )

_TASK_BLOCK = re.compile(r"###\s+task\.md\s*\n```markdown\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_task_markdown(markdown_text: str) -> str:
    match = _TASK_BLOCK.search(markdown_text)
    if not match:
        raise ValueError(
            "Generation response did not contain a ### task.md fenced ```markdown block."
        )
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
            "# full clean_solution.py file",
            "```",
            "",
            "Constraints for clean_solution.py:",
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
