"""Runs one tool call from the model against the sandbox, with the plumbing the contract requires:
idempotency keys on every write, tool_start/tool_end turn events, state updates, and errors returned
as JSON the model can speak about — never raised into the LLM loop."""
from __future__ import annotations

import json
import os
import pathlib
import time
import uuid
from typing import Any, Awaitable, Callable

from services.agent.state import INFRA_ERRORS, WRITE_TOOLS, CallState
from services.tools.sandbox import Sandbox, err

ROOT = pathlib.Path(__file__).resolve().parents[2]
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


def load_tool_schemas() -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted((ROOT / "schemas/tools").glob("*.json")):
        schema = json.loads(p.read_text())
        out[schema["function"]["name"]] = schema
    return out


def _valid_uuid(v: Any) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


class ToolExecutor:
    def __init__(self, sandbox: Sandbox, state: CallState, event_sink: EventSink | None = None, retry_infra_errors: bool | None = None) -> None:
        self.sandbox = sandbox
        self.state = state
        self._sink = event_sink
        self.schemas = load_tool_schemas()
        self.turn_idx = 0
        # one retry, same arguments and idempotency key, when a tool answers timeout/unavailable (state.INFRA_ERRORS): a transient
        # infrastructure error is the client's to absorb, not the caller's to authorize; the model sees only the final result. Both
        # attempts are tool_end events (attempt 1 and 2) with a tool_retry event between them. AGENT_TOOL_RETRY=0 turns it off.
        self.retry_infra_errors = (os.environ.get("AGENT_TOOL_RETRY", "1") != "0") if retry_infra_errors is None else retry_infra_errors

    async def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._sink is None:
            return
        await self._sink({"call_id": self.state.call_id, "turn_idx": self.turn_idx, "model_id": self.state.model_id, "event": event, "ts_ms": time.time_ns() // 1_000_000, "payload": payload})

    async def execute(self, name: str, raw_arguments: dict[str, Any] | str) -> str:
        t0 = time.perf_counter()
        args: dict[str, Any]
        result: dict[str, Any] | None = None
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be an object")
                args = parsed
            except (json.JSONDecodeError, ValueError) as e:
                args, result = {}, err("invalid_argument", f"arguments were not valid JSON: {e}")
        else:
            args = dict(raw_arguments or {})
        if result is None and name not in self.schemas:
            result = err("invalid_argument", f"unknown tool {name}")
        if result is None and name in WRITE_TOOLS and not _valid_uuid(args.get("idempotency_key")):
            args["idempotency_key"] = str(uuid.uuid4())

        await self._emit("tool_start", {"name": name})
        attempt = 1
        if result is None:
            self.state.note_tool_call(name, args)
            result = self.sandbox.call(name, args)
            self.state.apply_tool_result(name, args, result)
            if self.retry_infra_errors and result.get("error") in INFRA_ERRORS:
                await self._emit("tool_end", {"name": name, "ok": False, "error": result.get("error"), "ms": int((time.perf_counter() - t0) * 1000), "attempt": 1})
                await self._emit("tool_retry", {"name": name, "error": result.get("error")})
                attempt, t0 = 2, time.perf_counter()
                await self._emit("tool_start", {"name": name, "attempt": 2})
                result = self.sandbox.call(name, args)
                self.state.apply_tool_result(name, args, result)
        ok = "error" not in result
        payload: dict[str, Any] = {"name": name, "ok": ok}
        if not ok:
            payload["error"] = result.get("error")
        payload["ms"] = int((time.perf_counter() - t0) * 1000)
        if attempt == 2:
            payload["attempt"] = 2
        await self._emit("tool_end", payload)
        return json.dumps(result, separators=(",", ":"))
