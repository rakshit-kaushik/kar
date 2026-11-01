"""System prompt for the agent under test, rendered from the sandbox clock.

Source order (first that exists): an explicit path, then `schemas/system_prompt.md` (CONTRACT §0:
the one prompt S, E and T load byte-for-byte), then `services/agent/system_prompt.md` (S's earlier
live prompt, for worktrees that predate the shared file).
Placeholders filled when present: {today} {weekday} {today_long} {now} {clock}. In a template
that also uses {today_long}, {today} is the ISO date; otherwise it is seed.json["clock_spoken"].
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "eval" / "prompts"
PROMPT_CANDIDATES = (
    ROOT / "schemas" / "system_prompt.md",
    ROOT / "services" / "agent" / "system_prompt.md",
)


class _Safe(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def long_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def resolve_prompt_path(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"system prompt not found: {p}")
        return p
    for p in PROMPT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("no system prompt file found")


def render_system_prompt(
    clock: datetime, clock_spoken: str, path: str | Path | None = None
) -> tuple[str, str]:
    """-> (text, source path as a repo-relative string)."""
    p = resolve_prompt_path(path)
    tmpl = p.read_text()
    iso = clock.date().isoformat()
    fields = _Safe(
        today=iso if "{today_long}" in tmpl else clock_spoken,
        weekday=clock.strftime("%A"),
        today_long=long_date(clock.date()),
        now=clock.strftime("%H:%M"),
        clock=clock.strftime("%Y-%m-%dT%H:%M"),
        clock_spoken=clock_spoken,
    )
    try:
        rel = str(p.relative_to(ROOT))
    except ValueError:
        rel = str(p)
    return tmpl.format_map(fields).strip(), rel
