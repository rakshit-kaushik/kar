#!/usr/bin/env python
"""Headless dry run of the agent loop (no audio): LiveKit AgentSession with only the LLM and the
contract tools, driven by text turns through the real proxy. Repeats the ordinary booking flow N
times and reports: verified lookups, completed bookings, structural leaks, unsupported success
claims, per-turn latency. PASS = plumbing works (no leaks, at least one verified lookup); the rates
are information about the model, not a gate.

Usage: uv run python scripts/dry_run_agent.py [--repeats 3] [--base-url URL] [--model ID] [--quiet]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import statistics
import sys
import time

import httpx
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.plugins import openai as lk_openai
from openai import AsyncOpenAI

from services.agent.state import CallState
from services.tools.executor import ToolExecutor
from services.tools.livekit_tools import build_livekit_tools
from services.tools.sandbox import Sandbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SEED = json.loads((ROOT / "eval/sandbox/seed.json").read_text())
SYSTEM_PROMPT = (ROOT / "schemas/system_prompt.md").read_text().replace("{today}", SEED["clock_spoken"])
STRUCTURAL = ("<tool_call>", "</tool_call>", "<think>", "</think>", '{"name"', "<|im_", "[Calling tool", "[calling tool")
SUCCESS_CLAIM = re.compile(r"\b(verified|verification (is )?(successful|complete)|booked|is blocked|has been (booked|blocked|cancelled|canceled|moved))\b", re.IGNORECASE)

TURNS = [
    "hi i need to book an appointment",
    "five five five zero one four two two one nine",
    "march twelfth nineteen eighty eight",
    "i want a mortgage consultation next tuesday morning at market street",
    "nine thirty is fine",
    "yes",
]


class Data:
    def __init__(self, executor: ToolExecutor) -> None:
        self.executor = executor


async def episode(args, idx: int, quiet: bool, turns: list[str] | None = None, call_id: str | None = None) -> dict:
    """One text episode: `turns` (default TURNS) are the caller's lines; returns counts plus a per-turn record for replays."""
    turns = TURNS if turns is None else turns
    call_id = call_id or f"dry-run-{idx}"
    state = CallState(call_id=call_id, model_id=args.model)
    sandbox = Sandbox(SEED)
    executor = ToolExecutor(sandbox, state)
    api_key = os.environ.get("HOSTED_LLM_API_KEY", "local") if "anthropic" in args.base_url else "local"
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key, default_headers={"X-R3-Call-Id": call_id}, timeout=httpx.Timeout(60.0, connect=5.0))
    llm_kwargs = {"temperature": args.temperature} if args.temperature is not None else {}  # hosted Claude rejects temperature; local default is 0.2
    session = AgentSession(llm=lk_openai.LLM(model=args.model, client=client, parallel_tool_calls=False, **llm_kwargs), userdata=Data(executor))
    await session.start(Agent(instructions=SYSTEM_PROMPT, tools=build_livekit_tools(lambda ctx: ctx.userdata.executor)))

    out = {"verified": False, "booked": False, "leaks": 0, "unsupported_claims": 0, "turn_ms": [], "tool_calls": [], "turns": []}
    try:
        for user in turns:
            if not quiet:
                print(f"\nUSER: {user}")
            t0 = time.perf_counter()
            result = await session.run(user_input=user)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            out["turn_ms"].append(dt_ms)
            rec = {"user": user, "agent": [], "tools": [], "ms": dt_ms, "flags": []}
            for ev in result.events:
                item = ev.item
                if isinstance(item, FunctionCall):
                    out["tool_calls"].append(item.name)
                    rec["tools"].append({"name": item.name, "arguments": item.arguments})
                    if not quiet:
                        print(f"  -> {item.name}({item.arguments})")
                elif isinstance(item, FunctionCallOutput):
                    if rec["tools"]:
                        rec["tools"][-1]["output"] = item.output[:300]
                    if not quiet:
                        print(f"  <- {item.output[:140]}")
                elif isinstance(item, ChatMessage) and item.role == "assistant":
                    text = item.text_content or ""
                    leak = any(t in text for t in STRUCTURAL)
                    out["leaks"] += leak
                    claim = bool(SUCCESS_CLAIM.search(text))
                    unsupported = claim and not (state.verified["customer_id"] or state.completed)
                    out["unsupported_claims"] += unsupported
                    rec["agent"].append(text)
                    rec["flags"] += (["LEAK"] if leak else []) + (["UNSUPPORTED-CLAIM"] if unsupported else [])
                    if not quiet:
                        flags = (" LEAK" if leak else "") + (" UNSUPPORTED-CLAIM" if unsupported else "")
                        print(f"AGENT ({dt_ms} ms{flags}): {text}")
            out["turns"].append(rec)
    finally:
        await session.aclose()
    out["verified"] = state.verified["customer_id"] is not None
    out["booked"] = bool(sandbox.diff()["appointments_new"])
    out["verification_failures"] = state.verification_failures
    out["final_state"] = {"verified_customer": state.verified["customer_id"], "completed": list(state.completed) if isinstance(state.completed, (list, set, tuple)) else state.completed, "sandbox_diff": sandbox.diff()}
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None, help="omit to use the server default (0.2 locally); the hosted endpoint rejects it")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    eps = [await episode(args, i, args.quiet) for i in range(args.repeats)]
    n = len(eps)
    all_ms = [m for e in eps for m in e["turn_ms"]]
    print(f"\n== {args.model} @ {args.base_url}  episodes={n}")
    print(f"verified lookups: {sum(e['verified'] for e in eps)}/{n}   bookings: {sum(e['booked'] for e in eps)}/{n}   "
          f"failed lookups total: {sum(e['verification_failures'] for e in eps)}   leaks: {sum(e['leaks'] for e in eps)}   "
          f"unsupported claims: {sum(e['unsupported_claims'] for e in eps)}")
    print(f"turn latency (text, incl. tools): p50={statistics.median(all_ms):.0f} ms  max={max(all_ms)} ms")
    ok = sum(e["leaks"] for e in eps) == 0 and any(e["verified"] for e in eps)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
