"""The contract system prompt: schemas/system_prompt.md loaded byte-for-byte, placeholders filled.

CONTRACT.md §0: one file for S, E and T with placeholders {weekday} {today_long} {today} {now}; the
serving prompt must be a byte-for-byte prefix of every training example, so everyone must fill the
placeholders the same way. Formats used here (to be confirmed with S and E):
    weekday    -> "Saturday"            (%A)
    today_long -> "November 1, 2026"    (%B D, YYYY)
    today      -> "2026-11-01"          (ISO)
    now        -> "10:15"               (%H:%M)
The file's trailing newline is kept (byte-for-byte). Placeholders absent from the file are simply not
filled (a prompt with only {today} works). The path follows the contract (schemas/system_prompt.md),
then services/agent/system_prompt.md, or $R3_SYSTEM_PROMPT_PATH.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path

from train.env import load_env, repo_root

PLACEHOLDERS = ("weekday", "today_long", "today", "now")
# first existing file wins; $R3_SYSTEM_PROMPT_PATH overrides (the human may move the file to services/agent/)
CANDIDATE_PATHS = ("schemas/system_prompt.md", "services/agent/system_prompt.md")


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


def system_prompt_template() -> str:
    return system_prompt_path().read_text()


def placeholder_values(now: dt.datetime) -> dict[str, str]:
    return {
        "weekday": now.strftime("%A"),
        "today_long": f"{now.strftime('%B')} {now.day}, {now.year}",
        "today": now.date().isoformat(),
        "now": now.strftime("%H:%M"),
    }


def render_system_prompt(now: dt.datetime, template: str | None = None) -> str:
    text = template if template is not None else system_prompt_template()
    for key, value in placeholder_values(now).items():
        text = text.replace("{" + key + "}", value)
    missing = [k for k in PLACEHOLDERS if "{" + k + "}" in text]
    if missing:
        raise ValueError(f"unfilled placeholders in system prompt: {missing}")
    return text


def system_prompt_sha256() -> str:
    return hashlib.sha256(system_prompt_path().read_bytes()).hexdigest()
