from __future__ import annotations

import httpx


class _StubResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": self._content}}]}


def _agent_response_markdown() -> str:
    return "\n".join(
        [
            "# auto-tuner agent response",
            "",
            "### task.md",
            "```markdown",
            "Refactor a small function to avoid dynamic attribute access.",
            "```",
            "",
        ]
    )


def install_openrouter_stub(monkeypatch) -> None:
    original_post = httpx.Client.post

    def _post(self, url: str, headers=None, json=None, **kwargs):  # type: ignore[no-untyped-def]
        try:
            base_url = str(self.base_url)
        except AttributeError:
            base_url = ""
        if "openrouter.ai" not in base_url:
            return original_post(self, url, headers=headers, json=json, **kwargs)

        payload = json or {}
        messages = payload.get("messages") or []
        prompt = ""
        if messages and isinstance(messages, list):
            user_message = None
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "user":
                    user_message = message
                    break
            if user_message is None and messages:
                candidate = messages[0]
                user_message = candidate if isinstance(candidate, dict) else None
            if user_message is not None:
                prompt = str(user_message.get("content", ""))

        if "You design dataset-generation prompts" in prompt:
            content = "Write a concrete Python refactoring task prompt."
        elif "You design strict grading prompts" in prompt:
            content = "Grade the answer against the goal; fail on dynamic access."
        elif "# auto-tuner example generation request" in prompt:
            content = _agent_response_markdown()
        elif "# auto-tuner naive solution request" in prompt:
            content = "\n".join(
                [
                    "```python",
                    "def read_value(obj, key):",
                    "    return getattr(obj, key)",
                    "```",
                    "",
                ]
            )
        elif "# auto-tuner grading request" in prompt:
            content = "\n".join(
                [
                    "```json",
                    '{ "passed": true, "severity": "none", "violations": [], "suggestion": "" }',
                    "```",
                    "",
                    "```python",
                    "def read_value(obj):",
                    "    return obj.value",
                    "```",
                    "",
                ]
            )
        else:
            content = "ok"

        return _StubResponse(content)

    monkeypatch.setattr(httpx.Client, "post", _post, raising=True)
