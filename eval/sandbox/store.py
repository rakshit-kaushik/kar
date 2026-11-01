"""In-memory Marlow Mutual sandbox built from seed.json: one instance per episode.

The tools whose schema files exist in schemas/tools/ run against this object (`Sandbox.call`).
Every call is logged with args and result; `diff()` reports what changed against the seed, which
is what the final_state scorer compares with a scenario's `expect.final_state`.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import seed as S

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas" / "tools"
MESSAGES = {
    "timeout": "the request timed out",
    "unavailable": "the service is unavailable",
    "conflict": "conflict",
    "not_found": "not found",
    "invalid_argument": "invalid argument",
    "not_verified": "not verified",
}
HANDOFF_REASONS = (
    "suspected_fraud",
    "verification_failed",
    "caller_requested",
    "out_of_scope",
    "tool_failure",
)


def load_tool_schemas() -> dict[str, dict]:
    """The contract's tool files, keyed by tool name (OpenAI function-calling format)."""
    out = {}
    for p in sorted(SCHEMA_DIR.glob("*.json")):
        spec = json.loads(p.read_text())
        out[spec["function"]["name"]] = spec
    return out


TOOL_SCHEMAS = load_tool_schemas()
TOOL_NAMES = list(TOOL_SCHEMAS)
TICKET_CATEGORIES: list[str] = TOOL_SCHEMAS["create_ticket"]["function"]["parameters"][
    "properties"
]["category"]["enum"]


def err(code: str, message: str | None = None) -> dict:
    return {"error": code, "message": message or MESSAGES[code]}


def digits(s: Any) -> str:
    d = re.sub(r"\D", "", str(s or ""))
    return d[1:] if len(d) == 11 and d.startswith("1") else d


def parse_date(s: Any) -> date | None:
    try:
        return date.fromisoformat(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_slot(s: Any) -> datetime | None:
    """'YYYY-MM-DDTHH:MM' (seconds or a space tolerated) -> datetime, else None."""
    if not isinstance(s, str):
        return None
    t = s.strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    return None


def validate_args(tool: str, args: Any) -> dict | None:
    """Required keys present and enum values legal per the schema file; None when fine.

    Unknown extra keys are tolerated (the trimmed schemas have no additionalProperties).
    """
    if not isinstance(args, dict):
        return err("invalid_argument", "arguments must be an object")
    params = TOOL_SCHEMAS[tool]["function"]["parameters"]
    props = params.get("properties", {})
    missing = [k for k in params.get("required", []) if args.get(k) in (None, "")]
    if missing:
        return err("invalid_argument", f"missing required argument: {', '.join(missing)}")
    for key, val in args.items():
        spec = props.get(key)
        if spec and "enum" in spec and val is not None and val not in spec["enum"]:
            return err("invalid_argument", f"{key} must be one of {', '.join(spec['enum'])}")
    return None


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
    reason_valid: bool


class Sandbox:
    """Deterministic bank back end for one episode, built from seed.json."""

    def __init__(self, seed: dict | None = None, *, inject: list[dict] | None = None):
        from . import tools  # registers HANDLERS

        self.seed = copy.deepcopy(seed) if seed is not None else S.load_seed()
        self.clock: datetime = datetime.fromisoformat(self.seed["clock"])
        self.today: date = self.clock.date()
        self.clock_spoken: str = self.seed.get("clock_spoken", "")
        self.branches: dict = self.seed["branches"]
        self.services: dict = self.seed["services"]
        self.blocked: dict = self.seed.get("blocked_slots", {})
        self.customers: dict[str, S.Customer] = {}
        self.cards: list[S.Card] = []
        self.appointments: list[S.Appointment] = []
        for c in self.seed["customers"]:
            cid = c["customer_id"]
            self.customers[cid] = S.Customer(
                cid,
                c["first_name"],
                c["last_name"],
                c["phone"],
                date.fromisoformat(c["date_of_birth"]),
                c["branch_id"],
            )
            for k in c.get("cards", []):
                self.cards.append(
                    S.Card(
                        k["card_id"],
                        cid,
                        str(k["card_last4"]),
                        k.get("status", "active"),
                        k.get("replacement") or "none",
                    )
                )
            for a in c.get("appointments", []):
                self.appointments.append(
                    S.Appointment(
                        a["appointment_id"],
                        cid,
                        a["branch_id"],
                        a["service_type"],
                        parse_slot(a["slot_start"]),
                        a.get("status", "booked"),
                    )
                )
        self.tickets: list[S.Ticket] = []
        self.handoff: HandoffRecord | None = None
        self._next = {"A": 3001, "T": 5001}
        self.tools = [n for n in TOOL_NAMES if n in tools.HANDLERS]
        self._handlers = {n: tools.HANDLERS[n] for n in self.tools}
        self._faults: list[dict] = list(inject or [])
        self._calls: dict[str, int] = {}
        self._idem: dict[tuple[str, str], dict] = {}
        self.tool_log: list[ToolLogEntry] = []
        self._seed_appts = {a.appointment_id: (a.status, a.slot) for a in self.appointments}
        self._seed_cards = {
            (k.customer_id, k.card_last4): (k.status, k.replacement) for k in self.cards
        }
        self._baseline = copy.deepcopy(
            (self.cards, self.appointments, self.tickets, self.handoff, self._next)
        )

    # ----- episode lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        """Back to the seed; clears the log, idempotency cache and call counters (faults stay)."""
        self.cards, self.appointments, self.tickets, self.handoff, self._next = copy.deepcopy(
            self._baseline
        )
        self.tool_log, self._idem, self._calls = [], {}, {}

    def inject(self, faults: list[dict]) -> None:
        """scenario.inject: [{tool, call_index (1-based, per tool), error}] — replaces the list."""
        self._faults = [dict(f) for f in faults]

    def inject_timeout(self, tool: str, nth_call: int = 1) -> None:
        self._faults.append({"tool": tool, "call_index": int(nth_call), "error": "timeout"})

    # ----- queries -----------------------------------------------------------------------
    def find_customer(self, customer_id: Any) -> S.Customer | None:
        return self.customers.get(str(customer_id))

    def find_customer_by_phone(self, phone: Any) -> S.Customer | None:
        p = digits(phone)
        return next((c for c in self.customers.values() if c.phone == p), None) if p else None

    def find_appointment(self, appointment_id: Any) -> S.Appointment | None:
        return next((a for a in self.appointments if a.appointment_id == str(appointment_id)), None)

    def customer_appointments(self, customer_id: str) -> list[S.Appointment]:
        return sorted(
            (a for a in self.appointments if a.customer_id == customer_id and a.status == "booked"),
            key=lambda a: a.slot_start,
        )

    def customer_cards(self, customer_id: str) -> list[S.Card]:
        return [k for k in self.cards if k.customer_id == customer_id]

    def closed(self, d: date) -> bool:
        return S.hours_for(d, self.seed) is None

    def open_slots(
        self, branch_id: str, d: date, ignore_appointment: str | None = None
    ) -> list[str]:
        """Slots inside hours, after the clock, not booked at the branch, not blocked."""
        blocked = self.blocked.get(branch_id, {}).get(d.isoformat())
        if blocked == "all":
            return []
        taken = {
            a.slot
            for a in self.appointments
            if a.branch_id == branch_id
            and a.status == "booked"
            and a.appointment_id != ignore_appointment
        }
        taken |= {f"{d.isoformat()}T{b}" for b in (blocked or [])}
        out = []
        for hhmm in S.slots_for(d, self.seed):
            dt = S.slot_dt(d, hhmm)
            if dt > self.clock and dt.strftime("%Y-%m-%dT%H:%M") not in taken:
                out.append(hhmm)
        return out

    def slot_is_open(
        self, branch_id: str, dt: datetime, ignore_appointment: str | None = None
    ) -> bool:
        return dt.strftime("%H:%M") in self.open_slots(branch_id, dt.date(), ignore_appointment)

    def next_id(self, prefix: str) -> str:
        n = self._next[prefix]
        self._next[prefix] += 1
        return f"{prefix}-{n}"

    # ----- dispatch ----------------------------------------------------------------------
    def call(self, name: str, args: Any) -> dict:
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as e:
                return self._finish(
                    name, args, err("invalid_argument", f"bad arguments: {e}"), ts, t0
                )
        if args is None:
            args = {}
        handler = self._handlers.get(name)
        if handler is None:
            return self._finish(name, args, err("invalid_argument", f"unknown tool {name}"), ts, t0)
        if not isinstance(args, dict):
            return self._finish(
                name, args, err("invalid_argument", "arguments must be an object"), ts, t0
            )
        self._calls[name] = self._calls.get(name, 0) + 1
        for f in self._faults:
            if f.get("tool") == name and int(f.get("call_index", 1)) == self._calls[name]:
                return self._finish(
                    name, args, err(f.get("error", "timeout")), ts, t0, injected=True
                )
        key = args.get("idempotency_key")
        has_key = (
            isinstance(key, str) and key != ""
        )  # the tool layer (S) adds keys; the model-facing schemas omit them
        if has_key and (name, key) in self._idem:
            return self._finish(
                name, args, copy.deepcopy(self._idem[(name, key)]), ts, t0, replay=True
            )
        result = handler(self, args)
        if has_key and "error" not in result:
            self._idem[(name, key)] = copy.deepcopy(result)
        return self._finish(name, args, result, ts, t0)

    def reject(self, name: str, args: Any, message: str) -> dict:
        """Log a call the harness could not dispatch (e.g. unparseable arguments)."""
        return self._finish(
            name,
            args,
            err("invalid_argument", message),
            datetime.now().isoformat(timespec="milliseconds"),
            time.perf_counter(),
        )

    def _finish(
        self,
        name: str,
        args: Any,
        result: dict,
        ts: str,
        t0: float,
        injected: bool = False,
        replay: bool = False,
    ) -> dict:
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
        return result

    # ----- export ------------------------------------------------------------------------
    def diff(self) -> dict:
        """What changed against the seed, in the shape of expect.final_state."""
        new = [
            {
                "customer_id": a.customer_id,
                "branch_id": a.branch_id,
                "service_type": a.service_type,
                "slot_start": a.slot,
            }
            for a in self.appointments
            if a.appointment_id not in self._seed_appts and a.status == "booked"
        ]
        changed = [
            {"appointment_id": a.appointment_id, "status": a.status, "slot_start": a.slot}
            for a in self.appointments
            if a.appointment_id in self._seed_appts
            and (a.status, a.slot) != self._seed_appts[a.appointment_id]
        ]
        cards = [
            {"card_last4": k.card_last4, "status": k.status, "replacement": k.replacement}
            for k in self.cards
            if (k.status, k.replacement) != self._seed_cards.get((k.customer_id, k.card_last4))
        ]
        tickets = [{"customer_id": t.customer_id, "category": t.category} for t in self.tickets]
        return {
            "appointments_new": new,
            "appointments_changed": changed,
            "cards": cards,
            "tickets": tickets,
        }

    def dump_state(self) -> dict:
        return {
            "clock": self.clock.strftime("%Y-%m-%dT%H:%M"),
            "appointments": [
                {
                    "appointment_id": a.appointment_id,
                    "customer_id": a.customer_id,
                    "branch_id": a.branch_id,
                    "service_type": a.service_type,
                    "slot_start": a.slot,
                    "status": a.status,
                }
                for a in self.appointments
            ],
            "cards": [
                {
                    "customer_id": k.customer_id,
                    "card_last4": k.card_last4,
                    "status": k.status,
                    "replacement": k.replacement,
                }
                for k in self.cards
            ],
            "tickets": [
                {
                    "ticket_id": t.ticket_id,
                    "customer_id": t.customer_id,
                    "category": t.category,
                    "summary": t.summary,
                    "status": t.status,
                }
                for t in self.tickets
            ],
            "handoff": None
            if self.handoff is None
            else {
                "reason": self.handoff.reason,
                "summary": self.handoff.summary,
                "reason_valid": self.handoff.reason_valid,
            },
            "diff": self.diff(),
            "tool_calls": len(self.tool_log),
        }
