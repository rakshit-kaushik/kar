"""Per-call session state (CONTRACT.md §5, schemas/state.json).

Verified facts come only from tool results; claimed facts come from what the caller said (as the
model passed them into tool arguments). The agent checkpoints `to_dict()` to Postgres `sessions`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

STAGES = ("greeting", "identifying", "intent", "collecting", "confirming", "executing", "closing", "handoff")
WRITE_TOOLS = {"book_appointment", "reschedule_appointment", "cancel_appointment", "create_ticket"}
INFRA_ERRORS = {"timeout", "unavailable"}
CLAIM_FIELDS = ("phone", "date_of_birth", "service_type", "branch_id", "date", "time", "card_last4", "appointment_ref")


def _empty_verified() -> dict[str, Any]:
    return {"customer_id": None, "first_name": None, "branch_id": None, "appointments": [], "cards": []}


def _empty_claimed() -> dict[str, Any]:
    return {k: None for k in CLAIM_FIELDS}


@dataclass
class CallState:
    call_id: str
    model_id: str
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    verified: dict[str, Any] = field(default_factory=_empty_verified)
    claimed: dict[str, Any] = field(default_factory=_empty_claimed)
    stage: str = "greeting"
    pending: dict[str, Any] | None = None
    completed: list[dict[str, Any]] = field(default_factory=list)
    handoff: dict[str, Any] | None = None
    verification_failures: int = 0
    tool_failures: int = 0
    business_errors: int = 0

    def _advance(self, stage: str) -> None:
        if self.stage == "handoff":
            return
        if STAGES.index(stage) > STAGES.index(self.stage):
            self.stage = stage

    def note_tool_call(self, tool: str, args: dict[str, Any]) -> None:
        for k in CLAIM_FIELDS:
            if args.get(k) not in (None, ""):
                self.claimed[k] = args[k]
        if tool in ("book_appointment", "reschedule_appointment") and args.get("slot_start") or args.get("new_slot_start"):
            slot = str(args.get("slot_start") or args.get("new_slot_start"))
            self.claimed["date"], self.claimed["time"] = slot[:10], slot[11:16]
        if args.get("appointment_id"):
            self.claimed["appointment_ref"] = args["appointment_id"]
        if tool == "lookup_account":
            self._advance("identifying")
        elif tool == "check_availability":
            self._advance("collecting")
        elif tool in WRITE_TOOLS:
            self.pending = {"tool": tool, "args": args, "read_back_spoken": None, "confirmed": None}
            self._advance("executing")

    def apply_tool_result(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        ok = "error" not in result
        if not ok:
            code = result.get("error")
            if code in INFRA_ERRORS:
                self.tool_failures += 1
            elif tool == "lookup_account":
                self.verification_failures += 1
            else:
                self.business_errors += 1
            return
        if tool == "lookup_account":
            self.verified = {k: result.get(k) for k in ("customer_id", "first_name", "branch_id")}
            self.verified["appointments"] = list(result.get("appointments") or [])
            self.verified["cards"] = list(result.get("cards") or [])
            self._advance("intent")
        elif tool in WRITE_TOOLS:
            self.completed.append({"tool": tool, "args": args, "result": result, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
            self.pending = None
            self._refresh_verified(tool, args, result)
            self._advance("closing")
        elif tool == "request_handoff":
            self.handoff = {"requested": True, "reason": result.get("reason") or args.get("reason"), "summary": str(args.get("summary") or "")}
            self.stage = "handoff"

    def _refresh_verified(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        appts = self.verified["appointments"]
        if tool == "book_appointment":
            appts.append({k: result[k] for k in ("appointment_id", "branch_id", "service_type", "slot_start") if k in result})
        elif tool == "reschedule_appointment":
            for a in appts:
                if a.get("appointment_id") == result.get("appointment_id"):
                    a["slot_start"] = result.get("slot_start")
        elif tool == "cancel_appointment":
            self.verified["appointments"] = [a for a in appts if a.get("appointment_id") != result.get("appointment_id")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified, "claimed": self.claimed, "stage": self.stage, "pending": self.pending,
            "completed": self.completed, "handoff": self.handoff, "model_id": self.model_id, "call_id": self.call_id, "started_at": self.started_at,
        }
