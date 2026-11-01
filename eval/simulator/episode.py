"""One simulated call: scenario + agent under test + caller + fresh sandbox -> Episode JSON.

The agent and the caller are protocols so the loop is testable with scripted fakes.
Every assistant message is stored exactly as the model returned it (content and the
`tool_calls` objects), because these episodes become training data.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from eval.sandbox import Sandbox
from eval.simulator.prompts import render_agent_system
from eval.simulator.scenario import Scenario
from eval.simulator.stt_shape import shape

GREETING = (
    "Thanks for calling Marlow Mutual, this is the service line. "
    "Can I get the phone number on your account?"
)
OUTCOMES = ("completed", "handoff", "turn_cap", "abandoned", "error")


class AgentUnderTest(Protocol):
    async def respond(self, messages: list[dict]) -> dict:
        """Return the assistant message as a plain dict: {"role", "content", "tool_calls"?}."""


class Caller(Protocol):
    async def next_utterance(
        self, transcript: list[tuple[str, str]], turn_idx: int
    ) -> tuple[str, bool]:
        """Return (text, hangup). Empty text with hangup=True ends the call immediately."""


@dataclass
class Episode:
    episode_id: str
    scenario_id: str
    family: str
    model_id: str
    run_id: str
    outcome: str
    error: str | None
    started_at: str
    ended_at: str
    messages: list[dict]
    turns: list[dict]
    tool_log: list[dict]
    final_state: dict
    scenario: dict
    usage: dict
    n_agent_turns: int  # spoken agent turns incl. the greeting (= 1 + caller turns)
    n_model_requests: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, directory: str | Path, name: str | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name or self.scenario_id}.json"
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))
        return p

    @property
    def assistant_texts(self) -> list[str]:
        return [
            m["content"] for m in self.messages if m.get("role") == "assistant" and m.get("content")
        ]


def _parse_tool_call(tc: dict) -> tuple[str, Any, str | None]:
    """-> (name, args or raw string, parse_error)."""
    fn = tc.get("function") or {}
    name = fn.get("name") or "<missing>"
    raw = fn.get("arguments")
    if raw is None or raw == "":
        return name, {}, None
    if isinstance(raw, dict):
        return name, raw, None
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return name, raw, "Tool arguments were not valid JSON."
    return name, args, None


def _usage_of(obj: Any) -> dict:
    u = getattr(obj, "usage_totals", None)
    return dict(u) if isinstance(u, dict) else {}


async def run_episode(
    scenario: Scenario,
    agent: AgentUnderTest,
    caller: Caller,
    sandbox: Sandbox,
    *,
    model_id: str,
    run_id: str,
    max_turns: int = 20,
    max_tool_rounds: int = 4,
    system_prompt: str | None = None,
) -> Episode:
    sandbox.reset()
    for f in scenario.injected_failures:
        sandbox.inject_timeout(f.tool, f.nth_call)
    started = datetime.now()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt or render_agent_system(scenario)},
        {"role": "assistant", "content": GREETING},
    ]
    transcript: list[tuple[str, str]] = [("agent", GREETING)]
    turns: list[dict] = []
    latencies: list[float] = []
    outcome, error = "turn_cap", None
    n_model_requests = 0

    try:
        for turn_idx in range(1, max_turns + 1):
            raw_text, hangup = await caller.next_utterance(list(transcript), turn_idx)
            text = shape(raw_text, scenario.noise_level, seed=scenario.seed * 1000 + turn_idx)
            if not text.strip():
                if hangup:
                    outcome = "completed"
                    break
                text = "hello"  # silence; keep the call alive
            messages.append({"role": "user", "content": text})
            transcript.append(("caller", text))
            turn: dict = {
                "idx": turn_idx,
                "caller_text_raw": raw_text,
                "caller_text": text,
                "hangup": hangup,
                "assistant_text": "",
                "tool_calls": [],
                "message_indices": [len(messages) - 1],
            }
            turns.append(turn)
            handoff_called = False
            spoken: list[str] = []
            for round_idx in range(max_tool_rounds + 1):
                t0 = time.perf_counter()
                msg = await agent.respond(messages)
                latencies.append(round((time.perf_counter() - t0) * 1000, 1))
                msg = dict(msg)
                msg.setdefault("role", "assistant")
                messages.append(msg)
                turn["message_indices"].append(len(messages) - 1)
                n_model_requests += 1
                if msg.get("content"):
                    spoken.append(str(msg["content"]))
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    break
                for tc in tool_calls:
                    name, args, parse_error = _parse_tool_call(tc)
                    if parse_error:
                        result = sandbox.reject(name, args, parse_error)
                    else:
                        result = sandbox.call(name, args)
                    if name == "request_handoff" and result.get("handoff"):
                        handoff_called = True
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": json.dumps(result),
                        }
                    )
                    turn["message_indices"].append(len(messages) - 1)
                    turn["tool_calls"].append(
                        {"id": tc.get("id"), "name": name, "args": args, "result": result}
                    )
                if round_idx == max_tool_rounds:
                    outcome, error = (
                        "abandoned",
                        f"exceeded {max_tool_rounds} tool rounds in one turn",
                    )
                    break
            turn["assistant_text"] = " ".join(spoken).strip()
            if turn["assistant_text"]:
                transcript.append(("agent", turn["assistant_text"]))
            if outcome == "abandoned":
                break
            if handoff_called:
                outcome = "handoff"
                break
            if hangup:
                outcome = "completed"
                break
    except Exception as e:  # noqa: BLE001 — any failure is recorded on the episode, never raised
        outcome, error = "error", f"{type(e).__name__}: {e}"

    ended = datetime.now()
    usage = {"agent": _usage_of(agent), "caller": _usage_of(caller)}
    return Episode(
        episode_id=f"{scenario.id}-{uuid.uuid4().hex[:8]}",
        scenario_id=scenario.id,
        family=scenario.family,
        model_id=model_id,
        run_id=run_id,
        outcome=outcome,
        error=error,
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
        messages=messages,
        turns=turns,
        tool_log=[e.to_dict() for e in sandbox.tool_log],
        final_state=sandbox.dump_state(),
        scenario=scenario.to_dict(),
        usage=usage,
        n_agent_turns=1 + len(turns),
        n_model_requests=n_model_requests,
        latency_ms=latencies,
    )
