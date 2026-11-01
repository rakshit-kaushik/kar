"""Scripted agent/caller fakes shared by simulator and scorer tests."""

from __future__ import annotations

import json
import uuid
from datetime import date

from eval.sandbox import Sandbox
from eval.simulator.episode import Episode, run_episode
from eval.simulator.scenario import Scenario

TODAY = date(2026, 11, 2)
DANA_FACTS = {"phone": "5550142219", "date_of_birth": "1988-03-12", "name": "Dana"}
LOOKUP_ARGS = {"phone": "5550142219", "date_of_birth": "1988-03-12"}


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
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    async def respond(self, messages):
        self.seen.append([dict(m) for m in messages])
        if not self.script:
            return say("Goodbye.")
        return self.script.pop(0)


class ScriptedCaller:
    def __init__(self, lines):
        self.lines = [(line, False) if isinstance(line, str) else line for line in lines]
        self.seen = []

    async def next_utterance(self, transcript, turn_idx):
        self.seen.append(list(transcript))
        if not self.lines:
            return ("", True)
        return self.lines.pop(0)


def scenario(**over) -> Scenario:
    base = {
        "id": "t-01",
        "family": "ordinary",
        "today": TODAY.isoformat(),
        "customer_id": "C-1042",
        "known_facts": DANA_FACTS,
        "persona": {"name": "Dana"},
        "goal": "test",
    }
    base.update(over)
    return Scenario.model_validate(base)


async def make_episode(scen: Scenario, agent_script, caller_lines, **kw) -> Episode:
    sb = Sandbox(seed=scen.seed, today=scen.today, fixture=scen.fixture.model_dump())
    return await run_episode(
        scen,
        ScriptedAgent(agent_script),
        ScriptedCaller(caller_lines),
        sb,
        model_id="scripted",
        run_id="test",
        **kw,
    )


def first_open_slot(scen: Scenario, branch="market_st", service="notary", day="2026-11-04"):
    sb = Sandbox(seed=scen.seed, today=scen.today, fixture=scen.fixture.model_dump())
    return sb.call(
        "check_availability", {"branch_id": branch, "service_type": service, "date": day}
    )["slots"]
