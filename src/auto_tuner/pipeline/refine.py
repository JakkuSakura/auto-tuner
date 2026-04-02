from __future__ import annotations

from auto_tuner.models.dataset import DatasetExample, GradeResult


def refine_examples(
    examples: list[DatasetExample], grades: list[GradeResult]
) -> list[DatasetExample]:
    return list(examples)
