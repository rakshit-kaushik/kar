"""Shared fixtures for the training tests. CPU/tokenizer only; nothing here touches the GPU."""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _artifacts() -> Path:
    from train.env import artifacts_dir

    return artifacts_dir()


@pytest.fixture(scope="session")
def artifacts() -> Path:
    return _artifacts()


@pytest.fixture(scope="session")
def base_4b(artifacts) -> Path:
    p = artifacts / "models" / "Qwen3-4B-8bit"
    if not (p / "tokenizer_config.json").exists():
        pytest.skip(f"Qwen3-4B-8bit not found under {p}")
    return p


@pytest.fixture(scope="session")
def base_06b(artifacts) -> Path:
    p = artifacts / "models" / "Qwen3-0.6B-8bit"
    if not (p / "config.json").exists():
        pytest.skip(f"Qwen3-0.6B-8bit not found under {p}")
    return p


@pytest.fixture(scope="session")
def tokenizer(base_4b):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(base_4b)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    from train.synth import write_dataset

    out = tmp_path_factory.mktemp("synth")
    write_dataset(out, n_train=20, n_valid=4, seed=0)
    return out


@pytest.fixture
def no_db(monkeypatch):
    """Make DATABASE_URL unusable so ledger writes fall back to the JSONL file."""
    monkeypatch.setenv("DATABASE_URL", "")


@pytest.fixture(autouse=True, scope="session")
def never_kill_real_ports():
    """No test may stop a real server: train.launch.stop_server refuses port kills under this guard
    (a rehearse test with a fake restart script killed the live :8000 on 2026-11-01 01:37)."""
    os.environ["R3_NEVER_KILL_PORTS"] = "1"
    yield
