"""Runs one tool call from the model against the sandbox, with the plumbing the contract requires:
idempotency keys on every write, tool_start/tool_end turn events, state updates, and errors returned
as JSON the model can speak about — never raised into the LLM loop."""
from __future__ import annotations

import json
import pathlib
import time
import uuid
from typing import Any, Awaitable, Callable

from services.agent.state import WRITE_TOOLS, CallState
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
    def __init__(self, sandbox: Sandbox, state: CallState, event_sink: EventSink | None = None) -> None:
        self.sandbox = sandbox
        self.state = state
        self._sink = event_sink
        self.schemas = load_tool_schemas()
        self.turn_idx = 0

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
        if result is None:
            self.state.note_tool_call(name, args)
            result = self.sandbox.call(name, args)
            self.state.apply_tool_result(name, args, result)
        ok = "error" not in result
        payload: dict[str, Any] = {"name": name, "ok": ok}
        if not ok:
            payload["error"] = result.get("error")
        payload["ms"] = int((time.perf_counter() - t0) * 1000)
        await self._emit("tool_end", payload)
        return json.dumps(result, separators=(",", ":"))
