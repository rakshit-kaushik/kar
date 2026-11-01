"""Scenario spec for the caller simulator (CONTRACT §8 families).

A scenario names who is calling (persona + known facts), what they want (goal), how the
call should go wrong on purpose (corrections, detours, noise, injected tool failures),
the sandbox fixture it needs, and what the scorers should expect at the end.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from eval.sandbox import TOOL_NAMES

Family = Literal[
    "ordinary",
    "missing_information",
    "correction",
    "ambiguous_branch",
    "tool_timeout",
    "duplicate_request",
    "multi_intent_detour",
    "needs_person",
    "noisy_transcript",
]
FAMILIES: tuple[str, ...] = get_args(Family)


class Persona(BaseModel):
    name: str
    age: int | None = None
    style: str = "neutral"  # terse | chatty | anxious | impatient | confused | polite ...
    notes: str = ""


class Correction(BaseModel):
    """The caller changes a detail mid-call. `to` is what must win at the final action."""

    model_config = ConfigDict(populate_by_name=True)
    when: str
    field: str  # time | date | branch_id | service_type | card_last4 | phone | date_of_birth
    from_: str | None = Field(default=None, alias="from")
    to: str
    say: str | None = None


class Detour(BaseModel):
    when: str
    say: str
    resolution: str = ""


class InjectedFailure(BaseModel):
    tool: str
    nth_call: int = 1

    @field_validator("tool")
    @classmethod
    def _known_tool(cls, v: str) -> str:
        if v not in TOOL_NAMES:
            raise ValueError(f"unknown tool {v!r}; known: {TOOL_NAMES}")
        return v


class Fixture(BaseModel):
    customers: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    appointments: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []


class ToolStep(BaseModel):
    name: str
    args: dict[str, Any] = {}

    @field_validator("name")
    @classmethod
    def _known_tool(cls, v: str) -> str:
        if v not in TOOL_NAMES:
            raise ValueError(f"unknown tool {v!r}; known: {TOOL_NAMES}")
        return v


class ExpectedState(BaseModel):
    """Subset-match against Sandbox.dump_state() for the scenario's customer."""

    appointments: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    tickets: list[dict[str, Any]] = []


class Expected(BaseModel):
    final_state: ExpectedState = ExpectedState()
    tool_steps: list[ToolStep] = []
    must_not_call: list[str] = []
    handoff: bool = False
    handoff_reason: str | None = None
    max_turns: int = 12
    corrected: dict[str, str] = {}


class Scenario(BaseModel):
    id: str
    family: Family
    title: str = ""
    today: date
    now: str = "09:00"
    seed: int = 42
    customer_id: str | None = None
    known_facts: dict[str, Any] = {}
    persona: Persona
    goal: str
    script_hints: list[str] = []
    corrections: list[Correction] = []
    detours: list[Detour] = []
    noise_level: int = Field(0, ge=0, le=3)
    injected_failures: list[InjectedFailure] = []
    fixture: Fixture = Fixture()
    expected: Expected = Expected()
    tags: list[str] = []
    source: str | None = None  # file path, set by the loader

    def to_dict(self) -> dict:
        return json.loads(self.model_dump_json(by_alias=True))


SCENARIO_SUFFIXES = (".yaml", ".yml", ".json")


def load_scenario(path: str | Path) -> Scenario:
    p = Path(path)
    raw = p.read_text()
    data = json.loads(raw) if p.suffix == ".json" else yaml.safe_load(raw)
    s = Scenario.model_validate(data)
    s.source = str(p)
    return s


def load_scenarios(directory: str | Path) -> list[Scenario]:
    d = Path(directory)
    files = sorted(p for p in d.iterdir() if p.suffix in SCENARIO_SUFFIXES)
    out: list[Scenario] = []
    seen: dict[str, str] = {}
    for p in files:
        s = load_scenario(p)
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r} in {p} and {seen[s.id]}")
        seen[s.id] = str(p)
        out.append(s)
    return out
