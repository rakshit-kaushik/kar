"""Prompt rendering for the agent under test and the simulated caller."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from eval.simulator.scenario import Scenario

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def long_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def render_agent_system(scenario: Scenario) -> str:
    tmpl = (PROMPT_DIR / "agent_system.md").read_text()
    return tmpl.format(
        weekday=scenario.today.strftime("%A"),
        today_long=long_date(scenario.today),
        today=scenario.today.isoformat(),
        now=scenario.now,
    ).strip()
