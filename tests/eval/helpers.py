"""Scripted agent/caller fakes and a schema-valid scenario factory shared by the eval tests."""

from __future__ import annotations

import json
import uuid

from eval.sandbox import Sandbox
from eval.simulator.episode import Episode, run_episode
from eval.simulator.scenario import Scenario, parse_scenario

DANA_LOOKUP = {"phone": "5550142219", "date_of_birth": "1988-03-12"}
BASE = {
    "id": "ordinary_01",
    "family": "ordinary",
    "title": "Dana books",
    "tests": "happy path",
    "caller": {
        "customer_id": "C-1001",
        "name": "Dana Whitfield",
        "persona": "Friendly, direct",
        "knows": {
            "phone": "5550142219",
            "phone_spoken": "five five five zero one four two two one nine",
            "date_of_birth": "1988-03-12",
            "first_name": "Dana",
            "cards": ["4219"],
        },
        "noise": "none",
    },
    "goal": "Book a mortgage consultation at Market Street on Tuesday November 3rd at 9:30.",
    "expect": {
        "final_state": {
            "appointments_new": [
                {
                    "customer_id": "C-1001",
                    "branch_id": "market_st",
                    "service_type": "mortgage_consultation",
                    "slot_start": "2026-11-03T09:30",
                }
            ],
            "no_other_changes": True,
        },
        "tool_calls": [
            {"tool": "lookup_account", "args": DANA_LOOKUP, "ok": True},
            {
                "tool": "check_availability",
                "args": {
                    "branch_id": "market_st",
                    "service_type": "mortgage_consultation",
                    "date": "2026-11-03",
                },
                "ok": True,
            },
            {
                "tool": "book_appointment",
                "args": {
                    "customer_id": "C-1001",
                    "branch_id": "market_st",
                    "service_type": "mortgage_consultation",
                    "slot_start": "2026-11-03T09:30",
                },
                "ok": True,
            },
        ],
        "handoff": {"required": False, "reason": None},
        "max_turns": 12,
    },
}


def scenario(**over) -> Scenario:
    """BASE with top-level overrides; `expect`/`caller` overrides are merged one level deep."""
    d = json.loads(json.dumps(BASE))
    for k, v in over.items():
        if k in ("expect", "caller") and isinstance(v, dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    return parse_scenario(d, "<test>")


def tc(name, args, id_=None):
    return {
        "id": id_ or f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def say(text):
    return {"role": "assistant", "content": text}


def tools(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class ScriptedAgent:
    def __init__(self, script, finish_reasons=None):
        self.script = list(script)
        self.seen = []
        self.finish_reasons = list(finish_reasons or [])
        self.last_finish_reason = None
        self.last_usage = {}

    async def respond(self, messages):
        self.seen.append([dict(m) for m in messages])
        self.last_finish_reason = self.finish_reasons.pop(0) if self.finish_reasons else "stop"
        if not self.script:
            return say("Goodbye.")
        return self.script.pop(0)


class ScriptedCaller:
    def __init__(self, lines):
        self.lines = [(line, False) if isinstance(line, str) else line for line in lines]
        self.seen = []
        self.last_raw = None

    async def next_utterance(self, transcript, turn_idx):
        self.seen.append(list(transcript))
        if not self.lines:
            return ("", True)
        text, hangup = self.lines.pop(0)
        self.last_raw = text
        return (text, hangup)


VERIFY = [
    say("Thanks. And your date of birth?"),
    tools(tc("lookup_account", DANA_LOOKUP)),
    say("Thanks Dana, you're verified. What can I help with today?"),
]
VERIFY_LINES = [
    "five five five zero one four two two one nine",
    "march twelfth nineteen eighty eight",
]


async def make_episode(scen: Scenario, agent_script, caller_lines, **kw) -> Episode:
    agent = kw.pop("agent", None) or ScriptedAgent(agent_script)
    return await run_episode(
        scen,
        agent,
        ScriptedCaller(caller_lines),
        Sandbox(),
        model_id="scripted",
        run_id="test",
        **kw,
    )
