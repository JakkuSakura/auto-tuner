from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from auto_tuner.web.app import app


@pytest.fixture()
def sample_config_path() -> Path:
    return Path("examples/sample_experiment.toml")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AUTO_TUNER_ARTIFACTS_DIR", str(tmp_path / ".artifacts"))
    return TestClient(app)
