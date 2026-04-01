from __future__ import annotations

from auto_tuner.models.dataset import DatasetExample, GradeResult


REPLACEMENTS = {
    "getattr(obj, 'value')": "obj.value",
    'getattr(obj, "value")': "obj.value",
    "hasattr(obj, 'value')": "True",
    ".__dict__": "",
    "vars(": "dict(",
}


def refine_examples(examples: list[DatasetExample], grades: list[GradeResult]) -> list[DatasetExample]:
    refined: list[DatasetExample] = []
    for example, grade in zip(examples, grades, strict=True):
        if grade.passed:
            refined.append(example)
            continue

        cleaned = example.clean_solution
        for source, target in REPLACEMENTS.items():
            cleaned = cleaned.replace(source, target)

        refined.append(
            DatasetExample(
                task=example.task,
                naive_solution=example.naive_solution,
                clean_solution=cleaned,
            )
        )
    return refined
