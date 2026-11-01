#!/usr/bin/env python
"""Headless dry run of the agent loop (no audio): LiveKit AgentSession with only the LLM and the
contract tools, driven by text turns through the real proxy. Prints tool calls and replies, flags
structural leaks, exits 1 if the lookup never happened or anything structural reached the reply.

Usage: uv run python scripts/dry_run_agent.py [--base-url URL] [--model ID]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
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
SYSTEM_PROMPT = (ROOT / "services/agent/system_prompt.md").read_text().replace("{today}", SEED["clock_spoken"])
STRUCTURAL = ("<tool_call>", "</tool_call>", "<think>", "</think>", '{"name"', "<|im_", "[Calling tool", "[calling tool")

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


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    args = ap.parse_args()

    state = CallState(call_id="dry-run", model_id=args.model)
    sandbox = Sandbox(SEED)
    executor = ToolExecutor(sandbox, state)
    api_key = os.environ.get("HOSTED_LLM_API_KEY", "local") if "anthropic" in args.base_url else "local"
    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key, default_headers={"X-R3-Call-Id": "dry-run"}, timeout=httpx.Timeout(60.0, connect=5.0))
    session = AgentSession(llm=lk_openai.LLM(model=args.model, client=client, temperature=0.2, parallel_tool_calls=False), userdata=Data(executor))
    await session.start(Agent(instructions=SYSTEM_PROMPT, tools=build_livekit_tools(lambda ctx: ctx.userdata.executor)))

    leaks, tool_calls, ok = 0, [], True
    try:
        for user in TURNS:
            print(f"\nUSER: {user}")
            t0 = time.perf_counter()
            result = await session.run(user_input=user)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            for ev in result.events:
                item = ev.item
                if isinstance(item, FunctionCall):
                    tool_calls.append(item.name)
                    print(f"  -> {item.name}({item.arguments})")
                elif isinstance(item, FunctionCallOutput):
                    print(f"  <- {item.output[:140]}")
                elif isinstance(item, ChatMessage) and item.role == "assistant":
                    text = item.text_content or ""
                    leak = any(t in text for t in STRUCTURAL)
                    leaks += leak
                    print(f"AGENT ({dt_ms} ms{' LEAK' if leak else ''}): {text}")
    finally:
        await session.aclose()

    print("\nstate:", json.dumps({"stage": state.stage, "verified": state.verified["customer_id"], "completed": [c["tool"] for c in state.completed]}))
    print("sandbox diff:", json.dumps(sandbox.diff()))
    if "lookup_account" not in tool_calls:
        print("FAIL: lookup_account never called"); ok = False
    if leaks:
        print(f"FAIL: {leaks} structural leak(s) in replies"); ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
