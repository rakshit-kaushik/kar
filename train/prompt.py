"""The contract system prompt: schemas/system_prompt.md loaded byte-for-byte, `{today}` filled.

CONTRACT.md §0: one file for S, E and T; single placeholder `{today}` = `eval/sandbox/seed.json["clock_spoken"]`
("Monday, November second, 2026, ten in the morning"). The serving prompt must be a byte-for-byte
prefix of every training example, so the value comes from the same seed file everyone reads; the
file's trailing newline is kept. Legacy placeholders {weekday} {today_long} {now} are filled from the
sandbox clock if a prompt still has them; unknown braces are left alone. The path follows the
contract (schemas/system_prompt.md), then services/agent/system_prompt.md, or $R3_SYSTEM_PROMPT_PATH.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from train.env import load_env, repo_root

PLACEHOLDERS = ("today", "weekday", "today_long", "now")
CANDIDATE_PATHS = ("schemas/system_prompt.md", "services/agent/system_prompt.md")
SEED_PATH = repo_root() / "eval" / "sandbox" / "seed.json"
CONTRACT_CLOCK = {"clock": "2026-11-02T10:00", "clock_spoken": "Monday, November second, 2026, ten in the morning"}


def system_prompt_path() -> Path:
    load_env()
    override = os.environ.get("R3_SYSTEM_PROMPT_PATH")
    if override:
        return Path(override).expanduser()
    for rel in CANDIDATE_PATHS:
        p = repo_root() / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"no system prompt file at {CANDIDATE_PATHS}; set R3_SYSTEM_PROMPT_PATH")


SYSTEM_PROMPT_PATH = repo_root() / CANDIDATE_PATHS[0]  # the contract's location (for messages and manifests)


def sandbox_clock() -> dict:
    """`clock` (ISO) and `clock_spoken` from eval/sandbox/seed.json ($R3_SANDBOX_SEED overrides);
    the contract's stated clock when the seed file is not merged yet."""
    load_env()
    path = Path(os.environ.get("R3_SANDBOX_SEED") or SEED_PATH)
    if path.exists():
        seed = json.loads(path.read_text())
        return {"clock": seed["clock"], "clock_spoken": seed["clock_spoken"]}
    return dict(CONTRACT_CLOCK)


def system_prompt_template() -> str:
    return system_prompt_path().read_text()


def placeholder_values(now: dt.datetime | None = None, clock_spoken: str | None = None) -> dict[str, str]:
    clock = sandbox_clock()
    now = now or dt.datetime.fromisoformat(clock["clock"])
    return {
        "today": clock_spoken or clock["clock_spoken"],
        "weekday": now.strftime("%A"),
        "today_long": f"{now.strftime('%B')} {now.day}, {now.year}",
        "now": now.strftime("%H:%M"),
    }


def render_system_prompt(now: dt.datetime | None = None, template: str | None = None, **overrides: str) -> str:
    text = template if template is not None else system_prompt_template()
    values = placeholder_values(now)
    values.update(overrides)
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def system_prompt_sha256() -> str:
    return hashlib.sha256(system_prompt_path().read_bytes()).hexdigest()
