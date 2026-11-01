#!/usr/bin/env python
"""Replay a recorded call's caller turns through any model, in text, with the real tools and a fresh sandbox per model,
for the side-by-side on stage (plan T+16-20, lane D). Everything shown is labelled "replay", never a live call.

  uv run python scripts/replay.py --call-id r3-g2-0116-011759-1 \
      --model base=r3-voice-base@http://127.0.0.1:8001/v1 --model hosted=claude-sonnet-5-5@$HOSTED_LLM_BASE_URL
  uv run python scripts/replay.py --turns-file demo/turns/ordinary.json --model ...   (a JSON list of caller lines)

Caller turns come from turn_events (`transcript` events with role user, final ones) or a JSON file. Output: a markdown
side-by-side and the raw JSON under $R3_ARTIFACTS/runs/replay-<stamp>/, plus the per-turn text latency per model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dry_run_agent import episode  # noqa: E402  (loads .env, system prompt, tools)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def caller_turns_from_db(call_id: str) -> list[str]:
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        rows = con.execute("select payload from turn_events where call_id=%s and event='transcript' order by ts_ms, id", (call_id,)).fetchall()
    turns = [r[0].get("text", "").strip() for r in rows if (r[0] or {}).get("role") == "user" and r[0].get("text", "").strip()]
    if not turns:  # the simulated caller logs its own lines when the agent's transcript is missing (network)
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            rows = con.execute("select payload from turn_events where call_id=%s and event='caller_speech_end' order by ts_ms, id", (call_id,)).fetchall()
        turns = [r[0].get("text", "").strip() for r in rows if (r[0] or {}).get("text")]
    return turns


def parse_model(spec: str) -> tuple[str, str, str]:
    name, rest = spec.split("=", 1)
    model, base_url = rest.split("@", 1)
    return name, model, base_url


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-id", default=None)
    ap.add_argument("--turns-file", default=None)
    ap.add_argument("--model", action="append", default=[], help="name=model@base_url (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.model:
        args.model = [f"base={os.environ.get('LOCAL_LLM_MODEL', 'r3-voice-base')}@{os.environ.get('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:8001/v1')}"]
        if os.environ.get("HOSTED_LLM_BASE_URL"):
            args.model.append(f"hosted={os.environ['HOSTED_LLM_MODEL']}@{os.environ['HOSTED_LLM_BASE_URL']}")
    if args.turns_file:
        turns = json.loads(pathlib.Path(args.turns_file).read_text()); source = args.turns_file
    elif args.call_id:
        turns = caller_turns_from_db(args.call_id); source = f"call {args.call_id}"
    else:
        print("give --call-id or --turns-file", file=sys.stderr); return 2
    if not turns:
        print(f"no caller turns found for {source}", file=sys.stderr); return 1
    stamp = time.strftime("%m%d-%H%M%S")
    out_dir = pathlib.Path(args.out or os.path.join(os.environ.get("R3_ARTIFACTS", os.path.expanduser("~/r3-artifacts")), "runs", f"replay-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"replay of {source}: {len(turns)} caller turns through {len(args.model)} model(s)")
    results = {}
    for spec in args.model:
        name, model, base_url = parse_model(spec)
        a = SimpleNamespace(base_url=base_url, model=model, temperature=None)
        t0 = time.perf_counter()
        try:
            results[name] = await episode(a, 0, quiet=True, turns=turns, call_id=f"replay-{name}-{stamp}")
            results[name]["model"] = model; results[name]["base_url"] = base_url; results[name]["wall_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:  # one model failing must not lose the others
            results[name] = {"error": repr(e)[:300], "model": model, "base_url": base_url}
            print(f"  {name}: FAILED {e!r}"[:200])
    # markdown side by side
    names = list(results)
    md = [f"# Replay of {source}", "", f"_Text replay with the contract tools and a fresh sandbox per model; labelled replay, not a live call. {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    md.append("| # | caller | " + " | ".join(names) + " |"); md.append("|---|---|" + "---|" * len(names))
    for i, user in enumerate(turns):
        cells = []
        for n in names:
            r = results[n]
            if "error" in r or i >= len(r.get("turns", [])):
                cells.append("_(no turn)_"); continue
            rec = r["turns"][i]
            tools = "; ".join(f"`{t['name']}`" for t in rec["tools"])
            text = " / ".join(rec["agent"]).replace("|", "\\|").replace("\n", " ")
            flags = " ".join(f"**{f}**" for f in rec["flags"])
            cells.append(f"{text} {('<br>tools: ' + tools) if tools else ''} {flags} <br><sub>{rec['ms']} ms</sub>".strip())
        md.append(f"| {i + 1} | {user.replace('|', '/')} | " + " | ".join(cells) + " |")
    md += ["", "| model | verified | booked | tool calls | leaks | unsupported claims | turn p50 ms | wall s |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = results[n]
        if "error" in r:
            md.append(f"| {n} ({r['model']}) | error | | | | | | |"); continue
        ms = sorted(r["turn_ms"]); p50 = ms[len(ms) // 2] if ms else 0
        md.append(f"| {n} ({r['model']}) | {r['verified']} | {r['booked']} | {', '.join(r['tool_calls']) or '-'} | {r['leaks']} | {r['unsupported_claims']} | {p50} | {r['wall_s']} |")
    (out_dir / "replay.md").write_text("\n".join(md) + "\n")
    (out_dir / "replay.json").write_text(json.dumps({"source": source, "turns": turns, "results": results}, indent=1, default=str))
    print("\n".join(md[-(len(names) + 2):]))
    print(f"\nwritten: {out_dir}/replay.md and replay.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
