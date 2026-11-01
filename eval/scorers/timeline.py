"""Walk an episode's messages in order; map messages and tool-log entries to caller turns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from eval.simulator.episode import Episode


@dataclass
class ToolEvent:
    idx: int
    name: str
    args: Any
    result: dict

    @property
    def ok(self) -> bool:
        return isinstance(self.result, dict) and "error" not in self.result

    @property
    def error(self) -> str | None:
        return self.result.get("error") if isinstance(self.result, dict) else None


@dataclass
class Evidence:
    """Tool results seen so far, for "is this statement supported" checks."""

    events: list[ToolEvent] = field(default_factory=list)

    def ok(self, name: str) -> bool:
        return any(e.name == name and e.ok for e in self.events)

    def err(self, name: str, code: str, contains: str | None = None) -> bool:
        return any(
            e.name == name
            and e.error == code
            and (contains is None or contains in str(e.result.get("message", "")).lower())
            for e in self.events
        )

    def verified_payload(self) -> dict | None:
        for e in self.events:
            if e.name == "lookup_account" and e.ok and e.result.get("verified"):
                return e.result
        return None

    def lookup_with_appointments(self) -> bool:
        p = self.verified_payload()
        return bool(p and p.get("appointments"))

    def lookup_with_blocked_card(self) -> bool:
        p = self.verified_payload()
        return bool(p and any(c.get("status") == "blocked" for c in p.get("cards", [])))


def parse_args(tc: dict) -> Any:
    raw = (tc.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return raw


def walk(ep: Episode):
    """Yield (idx, kind, payload): kind in {"user", "assistant_text", "tool"}; greeting included."""
    pending: dict[str, tuple[str, Any]] = {}
    for idx, m in enumerate(ep.messages):
        role = m.get("role")
        if role == "user":
            yield idx, "user", m.get("content") or ""
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                pending[str(tc.get("id"))] = (
                    (tc.get("function") or {}).get("name") or "?",
                    parse_args(tc),
                )
            if m.get("content"):
                yield idx, "assistant_text", str(m["content"])
        elif role == "tool":
            name, args = pending.get(str(m.get("tool_call_id")), ("?", {}))
            try:
                result = json.loads(m.get("content") or "{}")
            except (TypeError, ValueError):
                result = {"error": "unparseable", "message": str(m.get("content"))}
            yield (
                idx,
                "tool",
                ToolEvent(
                    idx, name, args, result if isinstance(result, dict) else {"value": result}
                ),
            )


def message_turn(ep: Episode) -> dict[int, int]:
    """message index -> caller turn idx (1-based); the greeting/system map to 0."""
    out = {0: 0, 1: 0}
    for t in ep.turns:
        for i in t.get("message_indices", []):
            out[i] = t["idx"]
    return out


def tool_log_turn(ep: Episode) -> dict[int, int]:
    """tool_log idx -> caller turn idx. Tool calls are logged in the order the turns made them."""
    out, i = {}, 0
    for t in ep.turns:
        for _ in t.get("tool_calls", []):
            out[i] = t["idx"]
            i += 1
    return out
