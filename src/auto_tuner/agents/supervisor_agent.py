from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from auto_tuner.config import OpenRouterConfig
from auto_tuner.llm.openrouter import PromptBundle, build_prompt_provider


@dataclass(frozen=True)
class GeneratedWorkspaceExample:
    task: str
    naive_solution: str
    clean_solution: str
    generation_prompt: str
    task_path: Path
    naive_solution_path: Path
    clean_solution_path: Path
    agent_request_path: Path
    agent_response_path: Path


class SupervisorAgent(Protocol):
    def build_prompts(self, meta_prompt: str) -> PromptBundle: ...

    def generate_example(
        self,
        *,
        workspace_dir: Path,
        meta_prompt: str,
        generation_prompt: str,
        example_id: int,
        theme_hint: str,
    ) -> GeneratedWorkspaceExample: ...

    def write_refined_solution_after_training(
        self,
        *,
        workspace_dir: Path,
        task_path: Path,
        clean_solution_path: Path,
        meta_prompt: str,
        training_status: str,
        backend: str,
        output_dir: str,
    ) -> Path | None: ...


class OpenRouterSupervisorAgent:
    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._prompt_provider = build_prompt_provider(config)

    def build_prompts(self, meta_prompt: str) -> PromptBundle:
        return self._prompt_provider.build_prompts(meta_prompt)

    def generate_example(
        self,
        *,
        workspace_dir: Path,
        meta_prompt: str,
        generation_prompt: str,
        example_id: int,
        theme_hint: str,
    ) -> GeneratedWorkspaceExample:
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        workspace_dir.mkdir(parents=True, exist_ok=True)

        agent_request_path = workspace_dir / "agent_request.md"
        agent_response_path = workspace_dir / "agent_response.md"

        prompt = _build_generation_request(
            meta_prompt=meta_prompt,
            generation_prompt=generation_prompt,
            example_id=example_id,
            theme_hint=theme_hint,
        )
        agent_request_path.write_text(prompt)

        try:
            response = _openrouter_complete_markdown(self._config, prompt)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed during generation: {exc}") from exc

        agent_response_path.write_text(response)

        extracted = _extract_files_from_agent_markdown(response)
        task = extracted["task.md"]
        naive_solution = extracted["naive_solution.py"]
        clean_solution = extracted["clean_solution.py"]

        task_path = workspace_dir / "task.md"
        naive_solution_path = workspace_dir / "naive_solution.py"
        clean_solution_path = workspace_dir / "clean_solution.py"
        task_path.write_text(task)
        naive_solution_path.write_text(naive_solution)
        clean_solution_path.write_text(clean_solution)

        return GeneratedWorkspaceExample(
            task=task,
            naive_solution=naive_solution,
            clean_solution=clean_solution,
            generation_prompt=generation_prompt,
            task_path=task_path,
            naive_solution_path=naive_solution_path,
            clean_solution_path=clean_solution_path,
            agent_request_path=agent_request_path,
            agent_response_path=agent_response_path,
        )

    def write_refined_solution_after_training(
        self,
        *,
        workspace_dir: Path,
        task_path: Path,
        clean_solution_path: Path,
        meta_prompt: str,
        training_status: str,
        backend: str,
        output_dir: str,
    ) -> Path | None:
        if training_status != "completed":
            return None
        if not self._config.api_key:
            raise RuntimeError(
                "OpenRouter API key is required for post-training refinement. "
                "Set OPENROUTER_API_KEY or configure openrouter.api_key."
            )

        refinement_request_path = workspace_dir / "refinement_request.md"
        refinement_response_path = workspace_dir / "refinement_response.md"
        refined_solution_path = workspace_dir / "refined_solution.py"

        refinement_request = _build_refinement_request(
            meta_prompt=meta_prompt,
            task=task_path.read_text(),
            clean_solution=clean_solution_path.read_text(),
            backend=backend,
            output_dir=output_dir,
        )
        refinement_request_path.write_text(refinement_request)

        try:
            response = _openrouter_complete_markdown(self._config, refinement_request)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed during refinement: {exc}") from exc
        refinement_response_path.write_text(response)

        refined_solution = _extract_python_from_refinement(response)
        if refined_solution is None:
            raise ValueError(
                "Refinement response did not contain a fenced ```python code block."
            )
        refined_solution_path.write_text(refined_solution)

        (workspace_dir / "refinement.md").write_text(
            "\n".join(
                [
                    "# Post-training refinement",
                    "",
                    f"- backend: `{backend}`",
                    f"- output_dir: `{output_dir}`",
                    "",
                    "- Records: `refinement_request.md`, `refinement_response.md`.",
                    "",
                ]
            )
        )
        return refined_solution_path


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
            "Return Markdown with exactly 3 sections and exactly 3 fenced code blocks:",
            "",
            "### task.md",
            "```markdown",
            "...",
            "```",
            "",
            "### naive_solution.py",
            "```python",
            "...",
            "```",
            "",
            "### clean_solution.py",
            "```python",
            "...",
            "```",
            "",
            "Constraints:",
            "- The task must be meaningfully different from other examples.",
            "- The naive solution should demonstrate the anti-pattern (it may use dynamic access).",
            (
                "- The clean solution must satisfy the meta-prompt goal and avoid dynamic "
                "access patterns."
            ),
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


def _build_refinement_request(
    *,
    meta_prompt: str,
    task: str,
    clean_solution: str,
    backend: str,
    output_dir: str,
) -> str:
    return "\n".join(
        [
            "# auto-tuner post-training refinement request",
            "",
            f"- backend: `{backend}`",
            f"- output_dir: `{output_dir}`",
            "",
            "## Meta prompt goal",
            "",
            meta_prompt.strip(),
            "",
            "## Task",
            "",
            task.strip(),
            "",
            "## Current clean solution",
            "",
            "```python",
            clean_solution.rstrip(),
            "```",
            "",
            "## Output format (strict)",
            "",
            "Return a single fenced code block containing the full refined Python file:",
            "",
            "```python",
            "...",
            "```",
            "",
            "Constraints:",
            "- Preserve task intent and external behavior.",
            "- Improve clarity, explicitness, and validation as needed.",
            "- Do not use getattr(), hasattr(), __dict__, vars(), or dynamic dispatch tricks.",
            "",
        ]
    )


_REFINEMENT_BLOCK = re.compile(r"```python\s*\n(?P<body>.*?)\n```", re.DOTALL)


def _extract_python_from_refinement(markdown_text: str) -> str | None:
    match = _REFINEMENT_BLOCK.search(markdown_text)
    if not match:
        return None
    body = match["body"].strip()
    if not body:
        return None
    return body + "\n"


def _extract_files_from_agent_markdown(markdown_text: str) -> dict[str, str]:
    pattern = re.compile(
        r"###\s+(?P<name>task\.md|naive_solution\.py|clean_solution\.py)\s*\n"
        r"```(?P<lang>markdown|python)\s*\n(?P<body>.*?)\n```",
        re.DOTALL,
    )
    matches = {
        match["name"]: match["body"].strip() + "\n"
        for match in pattern.finditer(markdown_text)
    }
    expected = {"task.md", "naive_solution.py", "clean_solution.py"}
    missing = expected - set(matches)
    if missing:
        raise ValueError(f"Agent response missing sections: {sorted(missing)}")
    return matches
