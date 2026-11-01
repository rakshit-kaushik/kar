"""Git hash, config hash, run ids and the versioned model id `r3-voice-<hash>`."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from train.env import REPO_ROOT


def git_info(repo: Path = REPO_ROOT) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "hash": run("rev-parse", "HEAD"),
            "short": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except Exception as e:  # not a git checkout
        return {"hash": None, "short": None, "branch": None, "dirty": None, "error": str(e)}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def model_id(prefix: str, *parts: str) -> str:
    """`<prefix>-<10 hex>`; the hash covers the resolved config, git hash, data manifest hash and run id."""
    return f"{prefix}-{hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()[:10]}"


def new_run_id(task: str) -> str:
    return f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
