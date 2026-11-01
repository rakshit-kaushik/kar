"""The contract system prompt: schemas/system_prompt.md loaded byte-for-byte, placeholders filled.

CONTRACT.md §0: one file for S, E and T with placeholders {weekday} {today_long} {today} {now}; the
serving prompt must be a byte-for-byte prefix of every training example, so everyone must fill the
placeholders the same way. Formats used here (to be confirmed with S and E):
    weekday    -> "Saturday"            (%A)
    today_long -> "November 1, 2026"    (%B D, YYYY)
    today      -> "2026-11-01"          (ISO)
    now        -> "10:15"               (%H:%M)
The file's trailing newline is kept (byte-for-byte).
"""
from __future__ import annotations

import datetime as dt
import hashlib

from train.env import repo_root

SYSTEM_PROMPT_PATH = repo_root() / "schemas" / "system_prompt.md"
PLACEHOLDERS = ("weekday", "today_long", "today", "now")


def system_prompt_template() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


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
    return hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()
