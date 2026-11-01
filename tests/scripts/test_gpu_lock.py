"""scripts/gpu_lock.sh must fail closed: an unreadable lock file (or a python3 that cannot run) is a held lock, never a free
one. Nov 1 07:14: a launcher with a stripped environment acquired the lock over a valid one because the read failure was
treated as "no lock". The script honours GPU_LOCK_FILE (set after .env is sourced) so tests never touch the real lock."""
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gpu_lock.sh"


def run(lock: pathlib.Path, *args, env_extra=None):
    env = {**os.environ, "GPU_LOCK_FILE": str(lock), **(env_extra or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


@pytest.fixture
def lock(tmp_path):
    assert "GPU_LOCK_FILE" in SCRIPT.read_text(), "gpu_lock.sh must honour GPU_LOCK_FILE so this test stays off the real lock"
    return tmp_path / "gpu.lock"


def test_acquire_refuses_a_valid_lock_of_another_owner(lock):
    assert run(lock, "acquire", "S", "10", "block").returncode == 0
    r = run(lock, "acquire", "T", "10", "batch")
    assert r.returncode == 1 and "held by S" in r.stdout
    assert run(lock, "status").returncode == 1


def test_unreadable_lock_file_is_treated_as_held(lock):
    lock.write_text("not json at all")
    s = run(lock, "status")
    assert s.returncode == 1 and "unreadable" in s.stdout.lower()
    assert run(lock, "acquire", "T", "10", "batch").returncode == 1
    assert lock.read_text() == "not json at all"


def test_missing_python_is_treated_as_held_not_free(lock, tmp_path):
    assert run(lock, "acquire", "S", "10", "block").returncode == 0
    bare = tmp_path / "bin"; bare.mkdir()
    for tool in ("bash", "date", "rm", "printf", "cat", "dirname", "sleep"):
        src = subprocess.run(["bash", "-c", f"command -v {tool}"], capture_output=True, text=True).stdout.strip()
        if src:
            (bare / tool).symlink_to(src)
    r = run(lock, "acquire", "T", "10", "batch", env_extra={"PATH": str(bare), "GPU_LOCK_PYTHON": str(bare / "no-such-python")})
    assert r.returncode == 1, r.stdout + r.stderr
    assert '"owner":"S"' in lock.read_text()


def test_release_is_owner_checked_and_expired_locks_are_free(lock):
    assert run(lock, "acquire", "S", "10", "block").returncode == 0
    assert run(lock, "release", "T").returncode == 1 and lock.exists()
    assert run(lock, "release", "S").returncode == 0 and not lock.exists()
    lock.write_text('{"owner":"T","since":1,"until":2,"purpose":"old"}')
    assert run(lock, "status").returncode == 0
