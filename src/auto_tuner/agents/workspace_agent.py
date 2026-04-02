from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from auto_tuner.config import OpenRouterConfig


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


class WorkspaceAgent:
    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config

    def generate_example(
        self,
        *,
        workspace_dir: Path,
        meta_prompt: str,
        generation_prompt: str,
        example_id: int,
        theme_hint: str,
    ) -> GeneratedWorkspaceExample:
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

        response: str
        if self.config.api_key:
            try:
                response = _openrouter_complete_markdown(self.config, prompt)
            except httpx.HTTPError:
                response = _fallback_agent_response(
                    generation_prompt=generation_prompt,
                    example_id=example_id,
                    theme_hint=theme_hint,
                )
        else:
            response = _fallback_agent_response(
                generation_prompt=generation_prompt,
                example_id=example_id,
                theme_hint=theme_hint,
            )
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

        response: str
        if self.config.api_key:
            try:
                response = _openrouter_complete_markdown(self.config, refinement_request)
            except httpx.HTTPError:
                response = ""
        else:
            response = ""
        refinement_response_path.write_text(response)

        refined_solution: str | None = None
        if response.strip():
            refined_solution = _extract_python_from_refinement(response)
        if refined_solution is None:
            refined_solution = clean_solution_path.read_text()
        refined_solution_path.write_text(refined_solution)

        (workspace_dir / "refinement.md").write_text(
            "\n".join(
                [
                    "# Post-training refinement",
                    "",
                    f"- backend: `{backend}`",
                    f"- output_dir: `{output_dir}`",
                    "",
                    (
                        "This workspace currently writes `refined_solution.py` after training "
                        "completes."
                    ),
                    "- Records: `refinement_request.md`, `refinement_response.md`.",
                    (
                        "- If the LLM call fails or is unavailable, it falls back to "
                        "`clean_solution.py`."
                    ),
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


def _fallback_agent_response(*, generation_prompt: str, example_id: int, theme_hint: str) -> str:
    examples = _fallback_templates(generation_prompt)
    template = examples[(example_id - 1) % len(examples)]
    return "\n".join(
        [
            "### task.md",
            "```markdown",
            template.task.strip(),
            "```",
            "",
            "### naive_solution.py",
            "```python",
            template.naive_solution.strip(),
            "```",
            "",
            "### clean_solution.py",
            "```python",
            template.clean_solution.strip(),
            "```",
            "",
        ]
    )


@dataclass(frozen=True)
class _TemplateExample:
    task: str
    naive_solution: str
    clean_solution: str


def _fallback_templates(generation_prompt: str) -> list[_TemplateExample]:
    return [
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Refactor a tiny settings accessor to avoid dynamic attribute access."
            ),
            naive_solution=(
                "def get_setting(settings, name):\n"
                "    return getattr(settings, name)\n"
            ),
            clean_solution=(
                "def get_setting(settings, name):\n"
                "    mapping = {\n"
                "        'host': settings.host,\n"
                "        'port': settings.port,\n"
                "    }\n"
                "    return mapping[name]\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Replace getattr-based field selection in a serializer with explicit mapping."
            ),
            naive_solution=(
                "def serialize(user, field):\n"
                "    return {field: getattr(user, field)}\n"
            ),
            clean_solution=(
                "def serialize(user, field):\n"
                "    values = {\n"
                "        'id': user.id,\n"
                "        'email': user.email,\n"
                "    }\n"
                "    return {field: values[field]}\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Refactor a CLI flag reader to avoid reflection-style access and to "
                "validate inputs explicitly."
            ),
            naive_solution=(
                "def read_flag(args, name):\n"
                "    return getattr(args, name)\n"
            ),
            clean_solution=(
                "def read_flag(args, name):\n"
                "    flags = {\n"
                "        'dry_run': args.dry_run,\n"
                "        'verbose': args.verbose,\n"
                "    }\n"
                "    try:\n"
                "        return flags[name]\n"
                "    except KeyError as exc:\n"
                "        raise ValueError(f\"unknown flag: {name}\") from exc\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Replace getattr-driven JSON export in a dataclass-like model with "
                "explicit field selection."
            ),
            naive_solution=(
                "def to_json(obj, field_names):\n"
                "    return {name: getattr(obj, name) for name in field_names}\n"
            ),
            clean_solution=(
                "def to_json(obj, field_names):\n"
                "    allowed = {\n"
                "        'id': obj.id,\n"
                "        'name': obj.name,\n"
                "        'created_at': obj.created_at,\n"
                "    }\n"
                "    out = {}\n"
                "    for name in field_names:\n"
                "        out[name] = allowed[name]\n"
                "    return out\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Refactor a router that uses hasattr/getattr dispatch into an explicit "
                "command registry."
            ),
            naive_solution=(
                "def dispatch(handler, command):\n"
                "    if hasattr(handler, command):\n"
                "        return getattr(handler, command)()\n"
                "    raise ValueError('unknown command')\n"
            ),
            clean_solution=(
                "def dispatch(handler, command):\n"
                "    commands = {\n"
                "        'start': handler.start,\n"
                "        'stop': handler.stop,\n"
                "        'status': handler.status,\n"
                "    }\n"
                "    try:\n"
                "        fn = commands[command]\n"
                "    except KeyError as exc:\n"
                "        raise ValueError('unknown command') from exc\n"
                "    return fn()\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Refactor a metrics emitter to use explicit attribute access and validation."
            ),
            naive_solution=(
                "def emit_metric(obj, name):\n"
                "    value = getattr(obj, name)\n"
                "    return f\"{name}={value}\"\n"
            ),
            clean_solution=(
                "def emit_metric(obj, name):\n"
                "    if name == 'latency_ms':\n"
                "        value = obj.latency_ms\n"
                "    elif name == 'status_code':\n"
                "        value = obj.status_code\n"
                "    else:\n"
                "        raise ValueError(f\"unsupported metric: {name}\")\n"
                "    return f\"{name}={value}\"\n"
            ),
        ),
        _TemplateExample(
            task=(
                f"{generation_prompt}\n\n"
                "Replace getattr/hasattr-driven dispatch in a handler with an explicit registry."
            ),
            naive_solution=(
                "def handle(event, name):\n"
                "    if hasattr(event, name):\n"
                "        return getattr(event, name)()\n"
                "    raise ValueError('unknown')\n"
            ),
            clean_solution=(
                "def handle(event, name):\n"
                "    handlers = {\n"
                "        'created': event.created,\n"
                "        'deleted': event.deleted,\n"
                "    }\n"
                "    try:\n"
                "        handler = handlers[name]\n"
                "    except KeyError as exc:\n"
                "        raise ValueError('unknown') from exc\n"
                "    return handler()\n"
            ),
        ),
    ]
