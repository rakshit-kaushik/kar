"""In-memory Marlow Mutual sandbox: one instance per episode, deterministic, resettable.

The eight contract tools run against this object (`Sandbox.call`). Every call is logged
with its args and result so the simulator can save the tool log and the scorers can
replay it. The same seed can be loaded into Postgres with `eval.sandbox.reset`.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import seed as S

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas" / "tools"
TIMEOUT_RESULT = {"error": "timeout", "message": "The request timed out."}


def load_tool_schemas() -> dict[str, dict]:
    """The contract's tool files, keyed by tool name (OpenAI function-calling format)."""
    out = {}
    for p in sorted(SCHEMA_DIR.glob("*.json")):
        spec = json.loads(p.read_text())
        out[spec["function"]["name"]] = spec
    return out


TOOL_SCHEMAS = load_tool_schemas()
TOOL_NAMES = list(TOOL_SCHEMAS)


@dataclass
class ToolLogEntry:
    idx: int
    tool: str
    args: Any
    result: dict
    ok: bool
    error: str | None
    injected: bool = False
    replay: bool = False
    ts: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "ok": self.ok,
            "error": self.error,
            "injected": self.injected,
            "replay": self.replay,
            "ts": self.ts,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class HandoffRecord:
    reason: str
    summary: str
    ts: str
    reason_valid: bool


def err(code: str, message: str) -> dict:
    return {"error": code, "message": message}


def validate_args(tool: str, args: Any, fallback: str) -> dict | None:
    """Check `args` against the tool's JSON schema. Returns an error dict or None.

    Uses `invalid_argument` when the tool lists it (CONTRACT §4), else `fallback`.
    """
    schema = TOOL_SCHEMAS[tool]["function"]["parameters"]
    code = (
        "invalid_argument"
        if "invalid_argument" in TOOL_SCHEMAS[tool]["function"]["description"]
        else fallback
    )
    if not isinstance(args, dict):
        return err(code, "Arguments must be an object.")
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args or args[key] in (None, ""):
            return err(code, f"Missing required argument: {key}")
    if schema.get("additionalProperties") is False:
        extra = [k for k in args if k not in props]
        if extra:
            return err(code, f"Unknown argument: {', '.join(sorted(extra))}")
    for key, val in args.items():
        spec = props.get(key)
        if spec is None:
            continue
        if spec.get("type") == "string" and not isinstance(val, str):
            return err(code, f"Argument {key} must be a string.")
        if "enum" in spec and val not in spec["enum"]:
            return err(code, f"Invalid {key}: {val!r}. Allowed: {', '.join(spec['enum'])}")
    return None


def parse_date(s: Any) -> date | None:
    if not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        return None


def parse_slot(s: Any) -> datetime | None:
    """'YYYY-MM-DDTHH:MM' (seconds tolerated, space tolerated) -> datetime, else None."""
    if not isinstance(s, str):
        return None
    t = s.strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    return None


def digits(s: Any) -> str:
    return re.sub(r"\D", "", s) if isinstance(s, str) else ""


class Sandbox:
    """Deterministic bank back end for one episode."""

    def __init__(
        self,
        seed: int = S.DEFAULT_SEED,
        today: date = S.DEFAULT_TODAY,
        fixture: dict | None = None,
        now: datetime | None = None,
    ):
        self.seed_int = seed
        self.today = today
        self.now = now or datetime.combine(today, datetime.min.time().replace(hour=9))
        data = S.generate_seed(seed=seed, today=today)
        self.customers: list[S.Customer] = data.customers
        self.cards: list[S.Card] = data.cards
        self.appointments: list[S.Appointment] = data.appointments
        self.tickets: list[S.Ticket] = []
        self.handoff: HandoffRecord | None = None
        self._next_appointment = 1 + max(int(a.appointment_id[2:]) for a in self.appointments)
        self._next_ticket = 5520
        self._next_customer = 1 + max(int(c.customer_id[2:]) for c in self.customers)
        self._fixture_customer_ids: set[str] = set()
        self.tool_log: list[ToolLogEntry] = []
        self._idem: dict[tuple[str, str], dict] = {}
        self._timeouts: dict[str, set[int]] = {}
        self._calls: dict[str, int] = {}
        self._touched_appointments: set[str] = set()
        self._handlers: dict[str, Callable[[Sandbox, dict], dict]] = {}
        from . import tools

        self._handlers = dict(tools.HANDLERS)
        if fixture:
            self.apply_fixture(fixture)
        self._snapshot()

    # ----- episode lifecycle -------------------------------------------------
    def _snapshot(self) -> None:
        self._baseline = copy.deepcopy(
            (
                self.customers,
                self.cards,
                self.appointments,
                self.tickets,
                self.handoff,
                self._next_appointment,
                self._next_ticket,
                self._next_customer,
            )
        )

    def reset(self) -> None:
        """Back to seed + fixture; clears the tool log, idempotency cache and call counters."""
        (
            self.customers,
            self.cards,
            self.appointments,
            self.tickets,
            self.handoff,
            self._next_appointment,
            self._next_ticket,
            self._next_customer,
        ) = copy.deepcopy(self._baseline)
        self.tool_log = []
        self._idem = {}
        self._calls = {}
        self._touched_appointments = set()

    def apply_fixture(self, fixture: dict) -> None:
        """Layer scenario-specific records over the seed, then re-baseline for reset().

        fixture keys (all optional):
          customers:    [{customer_id?, first_name, last_name?, phone, date_of_birth, branch_id,
                          cards?: ["4219", ...]}]
          cards:        [{customer_id, card_last4, status?: active|blocked, replacement?}]
          appointments: [{customer_id, branch_id, service_type, slot_start, status?}]
          unavailable:  [{branch_id, date, slots?: ["09:00", ...]}]   # default: whole day
        """
        for c in fixture.get("customers", []):
            cid = c.get("customer_id") or f"C-{self._next_customer}"
            if not c.get("customer_id"):
                self._next_customer += 1
            existing = self.find_customer(cid)
            dob = c["date_of_birth"]
            dob = dob if isinstance(dob, date) else date.fromisoformat(dob)
            if existing:
                self.customers.remove(existing)
                self.cards = [k for k in self.cards if k.customer_id != cid]
                self.appointments = [a for a in self.appointments if a.customer_id != cid]
            # keep phones unique: evict any seed customer holding this phone
            for other in [x for x in self.customers if x.phone == digits(c["phone"])]:
                self.customers.remove(other)
            self.customers.append(
                S.Customer(
                    cid,
                    c["first_name"],
                    c.get("last_name", "Doe"),
                    digits(c["phone"]),
                    dob,
                    c["branch_id"],
                )
            )
            self._fixture_customer_ids.add(cid)
            for l4 in c.get("cards", []):
                self.cards.append(S.Card(f"K-{len(self.cards):06d}", cid, str(l4)))
        for k in fixture.get("cards", []):
            self.cards = [
                x
                for x in self.cards
                if not (x.customer_id == k["customer_id"] and x.card_last4 == str(k["card_last4"]))
            ]
            self.cards.append(
                S.Card(
                    f"K-{len(self.cards):06d}",
                    k["customer_id"],
                    str(k["card_last4"]),
                    k.get("status", "active"),
                    k.get("replacement"),
                )
            )
        for a in fixture.get("appointments", []):
            dt = parse_slot(a["slot_start"])
            if dt is None:
                raise ValueError(f"fixture appointment slot_start not YYYY-MM-DDTHH:MM: {a}")
            self._evict_slot(a["branch_id"], dt)
            ap = S.Appointment(
                a.get("appointment_id") or f"A-{self._next_appointment}",
                a["customer_id"],
                a["branch_id"],
                a["service_type"],
                dt,
                a.get("status", "booked"),
            )
            if not a.get("appointment_id"):
                self._next_appointment += 1
            self.appointments.append(ap)
            self._fixture_customer_ids.add(a["customer_id"])
        fillers = [c for c in self.customers if c.filler]
        for u in fixture.get("unavailable", []):
            d = date.fromisoformat(u["date"]) if isinstance(u["date"], str) else u["date"]
            for hhmm in u.get("slots") or S.slots_for(d):
                dt = S.slot_dt(d, hhmm)
                if self.slot_is_open(u["branch_id"], dt):
                    self.appointments.append(
                        S.Appointment(
                            f"A-{self._next_appointment}",
                            fillers[self._next_appointment % len(fillers)].customer_id,
                            u["branch_id"],
                            "financial_review",
                            dt,
                        )
                    )
                    self._next_appointment += 1
        self._snapshot()

    def _evict_slot(self, branch_id: str, dt: datetime) -> None:
        self.appointments = [
            a
            for a in self.appointments
            if not (a.branch_id == branch_id and a.slot_start == dt and a.status == "booked")
        ]

    # ----- queries -----------------------------------------------------------
    def find_customer(self, customer_id: str) -> S.Customer | None:
        return next((c for c in self.customers if c.customer_id == customer_id), None)

    def find_customer_by_phone(self, phone: str) -> S.Customer | None:
        return next((c for c in self.customers if c.phone == phone), None)

    def find_appointment(self, appointment_id: str) -> S.Appointment | None:
        return next((a for a in self.appointments if a.appointment_id == appointment_id), None)

    def customer_appointments(self, customer_id: str) -> list[S.Appointment]:
        return sorted(
            (
                a
                for a in self.appointments
                if a.customer_id == customer_id
                and a.status == "booked"
                and a.slot_start.date() >= self.today
            ),
            key=lambda a: a.slot_start,
        )

    def customer_cards(self, customer_id: str) -> list[S.Card]:
        return [k for k in self.cards if k.customer_id == customer_id]

    def slot_is_open(self, branch_id: str, dt: datetime) -> bool:
        return not any(
            a.branch_id == branch_id and a.slot_start == dt and a.status == "booked"
            for a in self.appointments
        )

    def open_slots(self, branch_id: str, d: date) -> list[str]:
        out = []
        for hhmm in S.slots_for(d):
            dt = S.slot_dt(d, hhmm)
            if dt <= self.now:
                continue
            if self.slot_is_open(branch_id, dt):
                out.append(hhmm)
        return out

    def date_problem(self, d: date) -> str | None:
        """Why a date cannot be booked: closed / past / beyond horizon; None when fine."""
        if d < self.today:
            return "That date is in the past."
        if d > self.today + S.BOOKING_HORIZON:
            return f"Bookings open only {S.BOOKING_HORIZON.days} days ahead."
        if S.hours_for(d) is None:
            return "The branch is closed on Sundays."
        return None

    def new_appointment_id(self) -> str:
        aid = f"A-{self._next_appointment}"
        self._next_appointment += 1
        return aid

    def new_ticket_id(self) -> str:
        tid = f"T-{self._next_ticket}"
        self._next_ticket += 1
        return tid

    def touch(self, appointment_id: str) -> None:
        self._touched_appointments.add(appointment_id)

    # ----- failure injection -------------------------------------------------
    def inject_timeout(self, tool: str, nth_call: int = 1) -> None:
        """The nth call (1-based, counted per tool, replays included) returns `timeout`."""
        self._timeouts.setdefault(tool, set()).add(int(nth_call))

    # ----- dispatch ----------------------------------------------------------
    def call(self, name: str, args: Any) -> dict:
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        handler = self._handlers.get(name)
        if handler is None:
            result = err("invalid_argument", f"Unknown tool: {name}")
            self._log(name, args, result, ts, t0)
            return result
        self._calls[name] = self._calls.get(name, 0) + 1
        if self._calls[name] in self._timeouts.get(name, ()):
            result = dict(TIMEOUT_RESULT)
            self._log(name, args, result, ts, t0, injected=True)
            return result
        key = args.get("idempotency_key") if isinstance(args, dict) else None
        if isinstance(key, str) and key and (name, key) in self._idem:
            result = copy.deepcopy(self._idem[(name, key)])
            self._log(name, args, result, ts, t0, replay=True)
            return result
        result = handler(self, args)
        if (
            isinstance(key, str)
            and key
            and "idempotency_key" in TOOL_SCHEMAS[name]["function"]["parameters"]["properties"]
        ):
            self._idem[(name, key)] = copy.deepcopy(result)
        self._log(name, args, result, ts, t0)
        return result

    def reject(self, name: str, args: Any, message: str) -> dict:
        """Log a call the harness could not dispatch (e.g. unparseable arguments)."""
        result = err("invalid_argument", message)
        self._log(
            name,
            args,
            result,
            datetime.now().isoformat(timespec="milliseconds"),
            time.perf_counter(),
        )
        return result

    def _log(
        self,
        name: str,
        args: Any,
        result: dict,
        ts: str,
        t0: float,
        injected: bool = False,
        replay: bool = False,
    ) -> None:
        error = result.get("error") if isinstance(result, dict) else None
        self.tool_log.append(
            ToolLogEntry(
                idx=len(self.tool_log),
                tool=name,
                args=copy.deepcopy(args),
                result=result,
                ok=error is None,
                error=error,
                injected=injected,
                replay=replay,
                ts=ts,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 3),
            )
        )

    # ----- export ------------------------------------------------------------
    def dump_state(self) -> dict:
        """Final sandbox state for the episode JSON. Filler bookings are left out unless touched."""
        keep_cust = {
            c.customer_id for c in self.customers if not c.filler
        } | self._fixture_customer_ids
        apps = [
            a
            for a in self.appointments
            if a.customer_id in keep_cust or a.appointment_id in self._touched_appointments
        ]
        return {
            "today": self.today.isoformat(),
            "appointments": [
                {
                    "appointment_id": a.appointment_id,
                    "customer_id": a.customer_id,
                    "branch_id": a.branch_id,
                    "service_type": a.service_type,
                    "slot_start": a.slot_start.strftime("%Y-%m-%dT%H:%M"),
                    "status": a.status,
                }
                for a in sorted(apps, key=lambda a: (a.customer_id, a.slot_start))
            ],
            "cards": [
                {
                    "customer_id": k.customer_id,
                    "card_last4": k.card_last4,
                    "status": k.status,
                    "replacement": k.replacement,
                }
                for k in self.cards
                if k.customer_id in keep_cust
            ],
            "tickets": [
                {
                    "ticket_id": t.ticket_id,
                    "customer_id": t.customer_id,
                    "category": t.category,
                    "summary": t.summary,
                }
                for t in self.tickets
            ],
            "handoff": None
            if self.handoff is None
            else {
                "requested": True,
                "reason": self.handoff.reason,
                "summary": self.handoff.summary,
                "ts": self.handoff.ts,
                "reason_valid": self.handoff.reason_valid,
            },
            "tool_calls": len(self.tool_log),
        }
