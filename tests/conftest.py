from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tuner.web.app import app
from tests.support.openrouter_stub import install_openrouter_stub


@pytest.fixture(autouse=True)
def _isolate_openrouter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    install_openrouter_stub(monkeypatch)


@pytest.fixture()
def sample_config_path() -> Path:
    return Path("tests/fixtures/sample_experiment.toml")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AUTO_TUNER_ARTIFACTS_DIR", str(tmp_path / ".artifacts"))
    return TestClient(app)
