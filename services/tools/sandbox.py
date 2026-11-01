"""In-memory sandbox for Marlow Mutual Bank: the eight tools, implemented exactly per
eval/sandbox/seed.json["rules"]. One instance per call or eval episode; deterministic; supports
fault injection (`inject`) and a diff against the seed for scorers.

Error shape everywhere: {"error": <code>, "message": <text>} with codes from CONTRACT.md §4.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import pathlib
import re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MESSAGES = {"timeout": "the request timed out", "unavailable": "the service is unavailable", "conflict": "conflict", "not_found": "not found", "invalid_argument": "invalid argument", "not_verified": "not verified"}
HANDOFF_REASONS = ("suspected_fraud", "verification_failed", "caller_requested", "out_of_scope", "tool_failure")


def err(code: str, message: str | None = None) -> dict:
    return {"error": code, "message": message or MESSAGES[code]}


def _digits(s: Any) -> str:
    d = re.sub(r"\D", "", str(s or ""))
    return d[1:] if len(d) == 11 and d.startswith("1") else d


def _date(s: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def _slot(s: Any) -> dt.datetime | None:
    try:
        v = dt.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return None if v.tzinfo else v.replace(second=0, microsecond=0)


class Sandbox:
    TOOLS = ("lookup_account", "check_availability", "book_appointment", "reschedule_appointment", "cancel_appointment", "create_ticket", "request_handoff")

    def __init__(self, seed: dict) -> None:
        self.seed = seed
        self.clock: dt.datetime = dt.datetime.fromisoformat(seed["clock"])
        self.branches: dict = seed["branches"]
        self.services: dict = seed["services"]
        self.hours: dict = seed["hours"]
        self.slot_minutes: int = int(seed.get("slot_minutes", 30))
        self.blocked: dict = seed.get("blocked_slots", {})
        self.customers: dict[str, dict] = {}
        self.cards: dict[str, dict] = {}  # card_last4+customer -> card
        self.appointments: dict[str, dict] = {}
        for c in copy.deepcopy(seed["customers"]):
            cid = c["customer_id"]
            for k in c.pop("cards"):
                self.cards[f"{cid}:{k['card_last4']}"] = {**k, "customer_id": cid, "replacement": k.get("replacement", "none")}
            for a in c.pop("appointments"):
                self.appointments[a["appointment_id"]] = {**a, "customer_id": cid}
            self.customers[cid] = c
        self._seed_appts = copy.deepcopy(self.appointments)
        self.tickets: list[dict] = []
        self.handoff: dict | None = None
        self.taxonomy = set(json.loads((ROOT / "schemas/tools/create_ticket.json").read_text())["function"]["parameters"]["properties"]["category"]["enum"])
        self._idem: dict[tuple[str, str], dict] = {}
        self._next = {"A": 3001, "T": 5001}
        self._inject: list[dict] = []
        self._calls: dict[str, int] = {}
        self.log: list[dict] = []

    @classmethod
    def from_seed(cls, path: pathlib.Path | str) -> "Sandbox":
        return cls(json.loads(pathlib.Path(path).read_text()))

    # -- infrastructure -----------------------------------------------------------------------------
    def inject(self, faults: list[dict]) -> None:
        self._inject = list(faults)

    def _fault(self, tool: str) -> dict | None:
        self._calls[tool] = self._calls.get(tool, 0) + 1
        for f in self._inject:
            if f["tool"] == tool and f["call_index"] == self._calls[tool]:
                return err(f["error"])
        return None

    def _idempotent(self, tool: str, key: Any, fn) -> dict:
        k = (tool, str(key))
        if key and k in self._idem:
            return self._idem[k]
        result = fn()
        if key and "error" not in result:
            self._idem[k] = result
        return result

    def call(self, name: str, arguments: str | dict) -> dict:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            result = err("invalid_argument", f"bad arguments: {e}")
            self.log.append({"tool": name, "args": arguments, "result": result, "ok": False})
            return result
        if name not in self.TOOLS:
            result = err("invalid_argument", f"unknown tool {name}")
        else:
            try:
                result = getattr(self, name)(**args)
            except TypeError as e:
                result = err("invalid_argument", f"bad arguments for {name}: {e}")
        self.log.append({"tool": name, "args": args, "result": result, "ok": "error" not in result})
        return result

    # -- read tools -----------------------------------------------------------------------------------
    def lookup_account(self, phone: Any = None, date_of_birth: Any = None, **_: Any) -> dict:
        if (f := self._fault("lookup_account")):
            return f
        p = _digits(phone)
        cust = next((c for c in self.customers.values() if c["phone"] == p), None)
        if cust is None:
            return err("not_found", "no account matches that phone number")
        if str(date_of_birth or "").strip() != cust["date_of_birth"]:
            return err("not_verified", "date of birth does not match")
        cid = cust["customer_id"]
        return {
            "verified": True, "customer_id": cid, "first_name": cust["first_name"], "branch_id": cust["branch_id"],
            "appointments": [{"appointment_id": a["appointment_id"], "branch_id": a["branch_id"], "service_type": a["service_type"], "slot_start": a["slot_start"]}
                             for a in sorted(self.appointments.values(), key=lambda x: x["slot_start"]) if a["customer_id"] == cid and a["status"] == "booked"],
            "cards": [{"card_last4": k["card_last4"], "status": k["status"]} for k in self.cards.values() if k["customer_id"] == cid],
        }

    def _open_slots(self, branch_id: str, day: dt.date, ignore_appointment: str | None = None) -> list[str]:
        hours = self.hours.get(WEEKDAYS[day.weekday()])
        if not hours:
            return []
        start = dt.datetime.combine(day, dt.time.fromisoformat(hours[0]))
        end = dt.datetime.combine(day, dt.time.fromisoformat(hours[1]))
        blocked = self.blocked.get(branch_id, {}).get(day.isoformat())
        if blocked == "all":
            return []
        taken = {a["slot_start"] for a in self.appointments.values()
                 if a["branch_id"] == branch_id and a["status"] == "booked" and a["appointment_id"] != ignore_appointment}
        taken |= {f"{day.isoformat()}T{b}" for b in (blocked or [])}
        out, t = [], start
        while t < end:
            s = t.strftime("%Y-%m-%dT%H:%M")
            if s not in taken and t > self.clock:
                out.append(s)
            t += dt.timedelta(minutes=self.slot_minutes)
        return out

    def check_availability(self, branch_id: Any = None, service_type: Any = None, date: Any = None, **_: Any) -> dict:
        if (f := self._fault("check_availability")):
            return f
        day = _date(date)
        if branch_id not in self.branches or service_type not in self.services or day is None:
            return err("invalid_argument", "branch_id, service_type and date (YYYY-MM-DD) are required")
        if day < self.clock.date() or not self.hours.get(WEEKDAYS[day.weekday()]):
            return err("unavailable", "the branch is closed that day")
        return {"branch_id": branch_id, "date": day.isoformat(), "slots": [s[11:] for s in self._open_slots(branch_id, day)]}

    # -- write tools ------------------------------------------------------------------------------------
    def book_appointment(self, customer_id: Any = None, branch_id: Any = None, service_type: Any = None, slot_start: Any = None, idempotency_key: Any = None, **_: Any) -> dict:
        if (f := self._fault("book_appointment")):
            return f

        def do() -> dict:
            if customer_id not in self.customers:
                return err("not_found", "customer not found")
            when = _slot(slot_start)
            if branch_id not in self.branches or service_type not in self.services or when is None:
                return err("invalid_argument", "branch_id, service_type and slot_start (YYYY-MM-DDTHH:MM) are required")
            dup = next((a for a in self.appointments.values() if a["customer_id"] == customer_id and a["service_type"] == service_type
                        and a["status"] == "booked" and a["slot_start"][:10] == when.date().isoformat()), None)
            if dup:
                return err("conflict", f"already has {service_type} on {dup['slot_start'][:10]} at {dup['slot_start'][11:]} ({dup['appointment_id']})")
            key = when.strftime("%Y-%m-%dT%H:%M")
            if key not in self._open_slots(branch_id, when.date()):
                return err("unavailable", "that slot is not open")
            aid = f"A-{self._next['A']}"
            self._next["A"] += 1
            self.appointments[aid] = {"appointment_id": aid, "customer_id": customer_id, "branch_id": branch_id, "service_type": service_type, "slot_start": key, "status": "booked"}
            return {"appointment_id": aid, "branch_id": branch_id, "service_type": service_type, "slot_start": key, "status": "booked"}

        return self._idempotent("book_appointment", idempotency_key, do)

    def reschedule_appointment(self, appointment_id: Any = None, new_slot_start: Any = None, idempotency_key: Any = None, **_: Any) -> dict:
        if (f := self._fault("reschedule_appointment")):
            return f

        def do() -> dict:
            a = self.appointments.get(str(appointment_id))
            if a is None or a["status"] != "booked":
                return err("not_found", "no booked appointment with that id")
            when = _slot(new_slot_start)
            if when is None:
                return err("invalid_argument", "new_slot_start must be YYYY-MM-DDTHH:MM")
            key = when.strftime("%Y-%m-%dT%H:%M")
            if key not in self._open_slots(a["branch_id"], when.date(), ignore_appointment=a["appointment_id"]):
                return err("unavailable", "that slot is not open")
            a["slot_start"] = key
            return {"appointment_id": a["appointment_id"], "branch_id": a["branch_id"], "service_type": a["service_type"], "slot_start": key, "status": "booked"}

        return self._idempotent("reschedule_appointment", idempotency_key, do)

    def cancel_appointment(self, appointment_id: Any = None, idempotency_key: Any = None, **_: Any) -> dict:
        if (f := self._fault("cancel_appointment")):
            return f

        def do() -> dict:
            a = self.appointments.get(str(appointment_id))
            if a is None or a["status"] != "booked":
                return err("not_found", "no booked appointment with that id")
            a["status"] = "cancelled"
            return {"appointment_id": a["appointment_id"], "status": "cancelled"}

        return self._idempotent("cancel_appointment", idempotency_key, do)

    def create_ticket(self, customer_id: Any = None, category: Any = None, summary: Any = None, idempotency_key: Any = None, **_: Any) -> dict:
        if (f := self._fault("create_ticket")):
            return f

        def do() -> dict:
            if category not in self.taxonomy:
                return err("invalid_argument", "category is not in the taxonomy")
            if customer_id not in self.customers:
                return err("not_found", "customer not found")
            tid = f"T-{self._next['T']}"
            self._next["T"] += 1
            self.tickets.append({"ticket_id": tid, "customer_id": customer_id, "category": category, "summary": str(summary or ""), "status": "open"})
            return {"ticket_id": tid, "category": category, "status": "open"}

        return self._idempotent("create_ticket", idempotency_key, do)

    def request_handoff(self, reason: Any = None, summary: Any = None, **_: Any) -> dict:
        if (f := self._fault("request_handoff")):
            return f
        r = reason if reason in HANDOFF_REASONS else "out_of_scope"
        self.handoff = {"reason": r, "summary": str(summary or "")}
        return {"status": "transferring", "reason": r}

    # -- for scorers and the UI -----------------------------------------------------------------------
    def state(self) -> dict:
        return {"handoff": self.handoff, "appointments": list(self.appointments.values()), "cards": list(self.cards.values()), "tickets": list(self.tickets)}

    def diff(self) -> dict:
        new = [{"customer_id": a["customer_id"], "branch_id": a["branch_id"], "service_type": a["service_type"], "slot_start": a["slot_start"]}
               for aid, a in self.appointments.items() if aid not in self._seed_appts and a["status"] == "booked"]
        changed = [{"appointment_id": aid, "status": a["status"], "slot_start": a["slot_start"]}
                   for aid, a in self.appointments.items() if aid in self._seed_appts and (a["status"], a["slot_start"]) != (self._seed_appts[aid]["status"], self._seed_appts[aid]["slot_start"])]
        tickets = [{"customer_id": t["customer_id"], "category": t["category"]} for t in self.tickets]
        return {"appointments_new": new, "appointments_changed": changed, "tickets": tickets}
