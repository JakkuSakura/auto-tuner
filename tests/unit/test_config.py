from __future__ import annotations

from pathlib import Path

import pytest

from auto_tuner.config import load_settings


def test_load_settings_uses_defaults_for_existing_example(sample_config_path: Path) -> None:
    settings = load_settings(sample_config_path)

    assert settings.training.backend == "fake"
    assert settings.generation.sample_count == 3
    assert settings.generation.auto_tune_prompt.startswith("Tune the model to rewrite code")
    assert settings.demo.example_models[0] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_load_settings_rejects_invalid_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[training]\nbackend = 'bad'\n")

    with pytest.raises(ValueError):
        load_settings(config_path)
