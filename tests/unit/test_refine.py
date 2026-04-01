from __future__ import annotations

from auto_tuner.models.dataset import DatasetExample, GradeResult
from auto_tuner.pipeline.refine import refine_examples


def test_refine_examples_passes_through() -> None:
    examples = [
        DatasetExample(
            task="task",
            naive_solution="bad",
            clean_solution="return obj.value",
        )
    ]
    grades = [GradeResult(passed=True)]

    refined = refine_examples(examples, grades)

    assert len(refined) == 1
    assert refined[0].clean_solution == "return obj.value"
