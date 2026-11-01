"""Repo root, `.env` loading and `$R3_ARTIFACTS` resolution shared by the training scripts."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return REPO_ROOT


def load_env(override: bool = False) -> None:
    """Load `.env` from the repo root (CONTRACT.md §0: `.env` is the single source of config)."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=override)


def artifacts_dir() -> Path:
    load_env()
    value = os.environ.get("R3_ARTIFACTS")
    if not value:
        raise RuntimeError("R3_ARTIFACTS is not set; copy .env.example to .env and fill it in")
    return Path(value).expanduser()


def expand(value: str) -> str:
    """Expand `${VAR}` / `$VAR` / `~` in a config string after loading `.env`."""
    load_env()
    return os.path.expanduser(os.path.expandvars(value))


def env_file_keys(path: Path | None = None) -> set[str]:
    """The keys defined in this worktree's `.env` (Conductor's copy of session S's): values this process
    carries that belong to the services' own `.env` in S's root and may be stale here."""
    p = Path(path) if path else REPO_ROOT / ".env"
    keys: set[str] = set()
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if key:
                keys.add(key)
    except OSError:
        pass
    return keys
