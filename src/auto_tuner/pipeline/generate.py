from __future__ import annotations

from auto_tuner.config import GenerationConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.models.dataset import DatasetExample


def generate_examples(config: GenerationConfig, prompts: PromptBundle) -> list[DatasetExample]:
    examples: list[DatasetExample] = []
    for index in range(config.sample_count):
        task = f"{prompts.generation_prompt} Example #{index + 1}."
        examples.append(
            DatasetExample(
                task=task,
                naive_solution=(
                    "def read_value(obj):\n"
                    "    return getattr(obj, 'value')\n"
                ),
                clean_solution=(
                    "def read_value(obj):\n"
                    "    return obj.value\n"
                ),
                generation_prompt=prompts.generation_prompt,
            )
        )
    return examples
