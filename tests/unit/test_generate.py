from __future__ import annotations

from auto_tuner.config import GenerationConfig
from auto_tuner.llm.openrouter import PromptBundle
from auto_tuner.pipeline.generate import generate_examples


def test_generate_examples_uses_generated_prompt() -> None:
    prompts = PromptBundle(
        meta_prompt="goal",
        generation_prompt="generated prompt",
        grading_prompt="grading prompt",
        source="fallback",
    )
    examples = generate_examples(GenerationConfig(sample_count=2), prompts)

    assert len(examples) == 2
    assert examples[0].generation_prompt == "generated prompt"
    assert examples[0].task.startswith("generated prompt")
