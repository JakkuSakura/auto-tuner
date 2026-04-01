from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetExample(BaseModel):
    task: str
    naive_solution: str
    clean_solution: str
    generation_prompt: str = ""


class GradeResult(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
    severity: str = "none"
    suggestion: str = ""
    grading_prompt: str = ""


class DatasetRecord(BaseModel):
    prompt: str
    response: str

    def as_conversation(self) -> dict[str, list[dict[str, str]]]:
        return {
            "conversations": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response},
            ]
        }
