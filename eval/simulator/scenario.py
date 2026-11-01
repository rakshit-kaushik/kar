"""Scenario spec: eval/scenarios/schema.json is the source of truth (human-owned).

Files are validated against the JSON schema first (jsonschema 4.26, Draft 2020-12), then parsed
into pydantic models for typed access. `validate_against_seed` backs `r3-eval validate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from eval.sandbox import TOOL_NAMES, Sandbox
from eval.sandbox import seed as S
from eval.sandbox.store import HANDOFF_REASONS, digits, parse_date, parse_slot

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
_VALIDATOR = Draft202012Validator(SCHEMA)

Family = Literal[
    "ordinary",
    "missing_info",
    "correction",
    "ambiguous_branch",
    "tool_timeout",
    "duplicate",
    "multi_intent",
    "needs_person",
    "noisy_transcript",
]
FAMILIES: tuple[str, ...] = get_args(Family)
Noise = Literal["none", "low", "high"]
InjectError = Literal[
    "timeout", "unavailable", "not_found", "conflict", "invalid_argument", "not_verified"
]
SCENARIO_SUFFIXES = (".json", ".yaml", ".yml")


class ScenarioError(ValueError):
    """A scenario file does not match eval/scenarios/schema.json."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Caller(_Strict):
    customer_id: str | None
    name: str
    persona: str
    knows: dict[str, Any]
    withholds: list[str] = []
    noise: Noise
    stt_quirks: list[str] = []


class ScriptItem(_Strict):
    trigger: str
    do: str


class Inject(_Strict):
    tool: str
    call_index: int = Field(ge=1)
    error: InjectError


class FinalState(_Strict):
    appointments_new: list[dict[str, Any]] = []
    appointments_changed: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    tickets: list[dict[str, Any]] = []
    no_other_changes: bool = False


class ToolCall(_Strict):
    tool: str
    args: dict[str, Any] = {}
    ok: bool | None = None


class Handoff(_Strict):
    required: bool
    reason: str | None = None


class Correction(_Strict):
    field: str
    value: str


class Expect(_Strict):
    final_state: FinalState
    tool_calls: list[ToolCall]
    forbidden_tools: list[str] = []
    max_successful_calls: dict[str, int] = {}
    handoff: Handoff
    max_turns: int = Field(ge=2, le=20)
    corrections_hold: list[Correction] = []
    must_mention: list[str] = []
    must_not_mention_before_verified: bool = True


class Scenario(_Strict):
    id: str
    family: Family
    title: str
    tests: str
    caller: Caller
    goal: str
    script: list[ScriptItem] = []
    inject: list[Inject] = []
    expect: Expect
    source: str | None = Field(default=None, exclude=True)

    def to_dict(self) -> dict:
        return json.loads(self.model_dump_json(exclude_none=False))


def validate_schema(data: Any, where: str = "<scenario>") -> None:
    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [
            f"{where}: " + "/".join(str(p) for p in e.absolute_path) + f": {e.message}"
            for e in errors[:5]
        ]
        raise ScenarioError("\n".join(lines))


def parse_scenario(data: Any, where: str = "<scenario>") -> Scenario:
    validate_schema(data, where)
    s = Scenario.model_validate(data)
    s.source = where if where != "<scenario>" else None
    return s


def load_scenario(path: str | Path) -> Scenario:
    p = Path(path)
    raw = p.read_text()
    data = json.loads(raw) if p.suffix == ".json" else yaml.safe_load(raw)
    return parse_scenario(data, str(p))


def load_scenarios(directory: str | Path) -> list[Scenario]:
    d = Path(directory)
    out: list[Scenario] = []
    seen: dict[str, str] = {}
    for p in sorted(x for x in d.iterdir() if x.suffix in SCENARIO_SUFFIXES):
        s = load_scenario(p)
        if s.id in seen:
            raise ValueError(f"duplicate scenario id {s.id!r} in {p} and {seen[s.id]}")
        seen[s.id] = str(p)
        out.append(s)
    return out


def validate_against_seed(s: Scenario, sb: Sandbox) -> list[str]:
    """Human-readable problems: customer/phone/dob, tool names, slots open, ids exist, cards."""
    problems: list[str] = []
    cust = None
    if s.caller.customer_id is not None:
        cust = sb.find_customer(s.caller.customer_id)
        if cust is None:
            problems.append(f"customer {s.caller.customer_id} is not in the seed")
        else:
            kp = s.caller.knows.get("phone")
            if kp and digits(kp) != cust.phone:
                problems.append(f"knows.phone {kp} differs from the record ({cust.phone})")
            kd = s.caller.knows.get("date_of_birth")
            if kd and parse_date(kd) != cust.date_of_birth:
                problems.append(
                    f"knows.date_of_birth {kd} differs from the record ({cust.date_of_birth})"
                )
    for tc in s.expect.tool_calls:
        if tc.tool not in TOOL_NAMES:
            problems.append(f"tool_calls: unknown tool {tc.tool}")
    for name in (
        list(s.expect.forbidden_tools)
        + list(s.expect.max_successful_calls)
        + [i.tool for i in s.inject]
    ):
        if name not in TOOL_NAMES:
            problems.append(f"unknown tool {name}")
    for a in s.expect.final_state.appointments_new:
        dt = parse_slot(a.get("slot_start"))
        if dt is None:
            problems.append(
                f"appointments_new: slot_start {a.get('slot_start')!r} is not YYYY-MM-DDTHH:MM"
            )
            continue
        if dt.strftime("%H:%M") not in S.slots_for(dt.date(), sb.seed):
            problems.append(f"appointments_new: {a['slot_start']} is outside hours or off the grid")
        elif not sb.slot_is_open(a.get("branch_id", ""), dt):
            problems.append(
                f"appointments_new: slot {a['slot_start']} at {a.get('branch_id')} is not open in the seed"
            )
        if a.get("branch_id") not in sb.branches:
            problems.append(f"appointments_new: unknown branch {a.get('branch_id')}")
        if a.get("service_type") not in sb.services:
            problems.append(f"appointments_new: unknown service {a.get('service_type')}")
        if a.get("customer_id") and sb.find_customer(a["customer_id"]) is None:
            problems.append(f"appointments_new: unknown customer {a['customer_id']}")
    for a in s.expect.final_state.appointments_changed:
        ap = sb.find_appointment(a.get("appointment_id"))
        if ap is None:
            problems.append(
                f"appointments_changed: appointment {a.get('appointment_id')} is not in the seed"
            )
        elif a.get("slot_start"):
            dt = parse_slot(a["slot_start"])
            if dt is None or not sb.slot_is_open(
                ap.branch_id, dt, ignore_appointment=ap.appointment_id
            ):
                problems.append(
                    f"appointments_changed: {a['slot_start']} is not open at {ap.branch_id}"
                )
    for c in s.expect.final_state.cards:
        owner = cust.customer_id if cust else None
        cards = sb.customer_cards(owner) if owner else sb.cards
        if not any(k.card_last4 == str(c.get("card_last4")) for k in cards):
            problems.append(
                f"cards: card ending {c.get('card_last4')} is not on the caller's account"
            )
    if (
        s.expect.handoff.required
        and s.expect.handoff.reason
        and s.expect.handoff.reason not in HANDOFF_REASONS
    ):
        problems.append(f"handoff.reason {s.expect.handoff.reason} is not a contract reason")
    if not s.expect.handoff.required and s.expect.handoff.reason:
        problems.append("handoff.reason given but handoff not required")
    return problems
