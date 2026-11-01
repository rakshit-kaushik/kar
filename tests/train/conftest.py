"""Shared fixtures for the training tests. CPU/tokenizer only; nothing here touches the GPU."""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _real_artifacts() -> Path | None:
    from train.env import artifacts_dir, load_env

    load_env()
    try:
        return artifacts_dir()
    except RuntimeError:
        return None


REAL_ARTIFACTS = _real_artifacts()  # captured at import, before any test redirects R3_ARTIFACTS


@pytest.fixture(scope="session")
def artifacts() -> Path:
    """The real shared artifacts dir (models), opt-in by name; every other test sees a private one."""
    if REAL_ARTIFACTS is None:
        pytest.skip("R3_ARTIFACTS is not set")
    return REAL_ARTIFACTS


@pytest.fixture(autouse=True)
def private_artifacts(monkeypatch, tmp_path):
    """No test may reach the shared $R3_ARTIFACTS by default (its gpu.lock, runs, models): R3_ARTIFACTS points
    at a per-test directory, which S's scripts copied into tmp roots then use too. On 2026-11-01 07:14 the lock
    tests, run with R3_ARTIFACTS stripped from the scripts' environment, released session S's live GPU lock."""
    d = tmp_path / "artifacts"
    d.mkdir(exist_ok=True)
    monkeypatch.setenv("R3_ARTIFACTS", str(d))
    yield d


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


@pytest.fixture(autouse=True, scope="session")
def real_gpu_lock_untouched():
    """The shared gpu.lock must be byte-identical after the whole session: a test that acquired or released
    the real lock would otherwise pass silently while breaking another session's GPU window."""
    lock = (REAL_ARTIFACTS / "gpu.lock") if REAL_ARTIFACTS else None
    before = lock.read_bytes() if lock and lock.exists() else None
    yield
    after = lock.read_bytes() if lock and lock.exists() else None
    assert after == before, f"a test touched the shared GPU lock {lock}: before={before!r} after={after!r}"
