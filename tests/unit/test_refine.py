from __future__ import annotations

from auto_tuner.models.dataset import DatasetExample, GradeResult
from auto_tuner.pipeline.refine import refine_examples


def test_refine_examples_rewrites_getattr() -> None:
    examples = [
        DatasetExample(
            task="task",
            naive_solution="bad",
            clean_solution="return getattr(obj, 'value')",
        )
    ]
    grades = [GradeResult(passed=False, violations=["getattr("], severity="major", suggestion="")]

    refined = refine_examples(examples, grades)

    assert refined[0].clean_solution == "return obj.value"
