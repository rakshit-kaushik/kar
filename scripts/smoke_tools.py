#!/usr/bin/env python
"""Streamed chat completion with the eight tools against LLM_BASE_URL.

Prints every delta kind as it arrives and reports: TTFT, whether tool calls came as OpenAI
`tool_calls` deltas or as raw text, any `<think>`/`<tool_call>` leakage, token usage.
Exit 1 on leakage or missing tool call. Usage: uv run python scripts/smoke_tools.py [--turn2]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TOOLS = [json.loads(p.read_text()) for p in sorted((ROOT / "schemas/tools").glob("*.json"))]
SYSTEM = (ROOT / "services/agent/system_prompt.md").read_text().replace("{today}", "Monday, November second, 2026, ten in the morning")


def messages(turn2: bool) -> list[dict]:
    m = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "hi i need to book an appointment"},
        {"role": "assistant", "content": "Happy to help. Can I get the phone number on your account?"},
        {"role": "user", "content": "five five five zero one four two two one nine"},
        {"role": "assistant", "content": "Thanks. And your date of birth?"},
        {"role": "user", "content": "march twelfth nineteen eighty eight"},
    ]
    if turn2:
        m += [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup_account", "arguments": json.dumps({"phone": "5550142219", "date_of_birth": "1988-03-12"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"verified": True, "customer_id": "C-1001", "first_name": "Dana", "branch_id": "market_st", "appointments": [], "cards": [{"card_last4": "4219", "status": "active"}]})},
        ]
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn2", action="store_true", help="include a completed lookup_account round trip; expects a spoken reply, not a tool call")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("HOSTED_LLM_API_KEY") if "anthropic" in (args.base_url or "") else "local")
    t0 = time.perf_counter()
    ttft = None
    text, tool_calls, n_chunks = [], {}, 0
    usage = None
    stream = client.chat.completions.create(
        model=args.model, messages=messages(args.turn2), tools=TOOLS, tool_choice="auto",
        stream=True, max_tokens=200, temperature=0.2, stream_options={"include_usage": True},
    )
    for chunk in stream:
        n_chunks += 1
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if ttft is None and (d.content or d.tool_calls):
            ttft = time.perf_counter() - t0
        if d.content:
            text.append(d.content)
        for tc in d.tool_calls or []:
            slot = tool_calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
    total = time.perf_counter() - t0
    content = "".join(text)

    print(f"base_url={args.base_url} model={args.model} turn2={args.turn2}")
    print(f"chunks={n_chunks} ttft={ttft and round(ttft, 3)}s total={round(total, 3)}s finish={finish!r}")
    if usage:
        print(f"usage: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
    print(f"content: {content!r}")
    for i, tc in sorted(tool_calls.items()):
        print(f"tool_call[{i}]: id={tc['id']} name={tc['name']} args={tc['arguments']}")

    leak = any(tag in content for tag in ("<tool_call>", "<think>", "</think>", "<|im_"))
    ok = True
    if leak:
        print("FAIL: structural tokens leaked into content"); ok = False
    if not args.turn2 and not tool_calls:
        print("FAIL: expected a tool_calls delta for lookup_account, got none"); ok = False
    if not args.turn2 and tool_calls:
        try:
            a = json.loads(tool_calls[0]["arguments"])
            print("parsed args:", a)
            if tool_calls[0]["name"] != "lookup_account":
                print("WARN: expected lookup_account, got", tool_calls[0]["name"])
        except json.JSONDecodeError as e:
            print("FAIL: tool arguments not valid JSON:", e); ok = False
    if args.turn2 and tool_calls:
        print("WARN: turn 2 produced a tool call instead of a spoken reply:", tool_calls)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
