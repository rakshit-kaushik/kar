"""HTTP-level load test of the model path (proxy :8001 -> server :8000), no audio vendors and no network.

Fires C concurrent streamed chat completions at the agent's real prompt size (system prompt + 7 tool schemas +
a mid-call conversation; the caller's phone digits vary per request so only the system prefix is cache-shared),
for C in --levels, --waves waves each. Reports TTFT and inter-token latency p50/p95 per level and the admission
limit: the largest C whose TTFT p50 <= --p50-budget and p95 <= --p95-budget (plan: 'the concurrency where p95
TTFT still meets budget becomes the admission limit'). --ledger writes load_text.cN.* rows.

  uv run python scripts/load_text.py --levels 1 2 4 8 --waves 3 --ledger --run-id load-text-base
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = json.loads((ROOT / "eval/sandbox/seed.json").read_text())
SYSTEM_PROMPT = (ROOT / "schemas/system_prompt.md").read_text().replace("{today}", SEED["clock_spoken"])
TOOLS = [json.loads(p.read_text()) for p in sorted((ROOT / "schemas/tools").glob("*.json"))]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def conversation(seed: int) -> list[dict]:
    """A mid-call prefix (turn 3 of the fixed Dana script), digits varied so prompts differ after the system prefix."""
    rnd = random.Random(seed)
    phone = " ".join(WORDS[rnd.randrange(10)] for _ in range(10))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Thanks for calling Marlow Mutual Bank. Can I get the phone number on your account?"},
        {"role": "user", "content": f"{phone}"},
        {"role": "assistant", "content": "Thank you. And your date of birth, please?"},
        {"role": "user", "content": "march twelfth, nineteen eighty eight"},
        {"role": "assistant", "content": "Thanks, Dana. How can I help you today?"},
        {"role": "user", "content": "i'd like to book a mortgage consultation for tuesday morning at the market street branch"},
    ]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs); k = (len(s) - 1) * p / 100.0; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def one(client, model: str, call_id: str, seed: int) -> dict:
    t0 = time.perf_counter(); first = None; last = None; n = 0; gaps = []
    stream = await client.chat.completions.create(model=model, messages=conversation(seed), tools=TOOLS, stream=True, max_tokens=96,
                                                  extra_headers={"X-R3-Call-Id": call_id}, stream_options={"include_usage": True})
    usage = None
    async for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d and (d.content or d.tool_calls):
            now = time.perf_counter(); n += 1
            if first is None:
                first = now
            else:
                gaps.append((now - last) * 1000)
            last = now
    total = time.perf_counter() - t0
    return {"call_id": call_id, "ttft_ms": (first - t0) * 1000 if first else None, "total_ms": total * 1000, "deltas": n,
            "itl_ms": statistics.median(gaps) if gaps else None, "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None)}


async def level(client, model: str, c: int, waves: int, tag: str) -> list[dict]:
    out = []
    for w in range(waves):
        res = await asyncio.gather(*[one(client, model, f"load-text-c{c}-{tag}-w{w}p{i}", seed=1000 * c + 10 * w + i) for i in range(c)], return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                out.append({"error": repr(r)[:200]})
            else:
                out.append(r)
        await asyncio.sleep(0.3)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "r3-voice-base"))
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--p50-budget", type=float, default=350.0); ap.add_argument("--p95-budget", type=float, default=600.0)
    ap.add_argument("--ledger", action="store_true"); ap.add_argument("--run-id", default="load-text"); ap.add_argument("--note", default="")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from openai import AsyncOpenAI
    import httpx
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get("LLM_API_KEY", "x"), timeout=httpx.Timeout(120.0, connect=5.0), max_retries=0)
    tag = time.strftime("%H%M%S")
    await one(client, args.model, f"load-text-warm-{tag}", seed=1)  # warm the system prefix
    rows = []; admission = None
    print(f"model {args.model} via {args.base_url}; {args.waves} waves per level; budgets p50 <= {args.p50_budget:.0f} ms, p95 <= {args.p95_budget:.0f} ms")
    print(f"{'conc':>5}{'n':>4}{'err':>4}{'ttft p50':>10}{'ttft p95':>10}{'ttft max':>10}{'itl p50':>9}{'itl p95':>9}{'total p50':>11}{'prompt':>8}{'compl':>7}")
    for c in args.levels:
        res = await level(client, args.model, c, args.waves, tag)
        ok = [r for r in res if "error" not in r and r["ttft_ms"] is not None]
        errs = len(res) - len(ok)
        ttft = [r["ttft_ms"] for r in ok]; itl = [r["itl_ms"] for r in ok if r["itl_ms"] is not None]; tot = [r["total_ms"] for r in ok]
        pt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]; ct = [r["completion_tokens"] for r in ok if r["completion_tokens"]]
        row = {"c": c, "n": len(res), "errors": errs, "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95), "ttft_max": max(ttft) if ttft else float("nan"),
               "itl_p50": pct(itl, 50), "itl_p95": pct(itl, 95), "total_p50": pct(tot, 50), "prompt": statistics.median(pt) if pt else None, "compl": statistics.median(ct) if ct else None}
        rows.append(row)
        print(f"{c:>5}{row['n']:>4}{errs:>4}{row['ttft_p50']:>10.0f}{row['ttft_p95']:>10.0f}{row['ttft_max']:>10.0f}{row['itl_p50']:>9.1f}{row['itl_p95']:>9.1f}{row['total_p50']:>11.0f}{(row['prompt'] or 0):>8.0f}{(row['compl'] or 0):>7.0f}")
        if errs == 0 and row["ttft_p50"] <= args.p50_budget and row["ttft_p95"] <= args.p95_budget:
            admission = c
        for r in res:
            if "error" in r:
                print(f"      error: {r['error']}")
    print(f"\nadmission limit (largest concurrency within budget): {admission if admission else 'none met the budget; use 1 and note it'}")
    if args.ledger:
        import psycopg
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
            for row in rows:
                for k in ("ttft_p50", "ttft_p95", "ttft_max", "itl_p50", "itl_p95", "total_p50"):
                    con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,'ms',true,%s)",
                                (args.run_id, args.model, f"load_text.c{row['c']}.{k}", row[k], f"{row['n']} requests, {row['errors']} errors, prompt ~{row['prompt']} tok, completion ~{row['compl']} tok; {args.note}"))
            con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,'load_text.admission_limit',%s,'calls',true,%s)",
                        (args.run_id, args.model, admission or 1, f"p50 <= {args.p50_budget:.0f} ms and p95 <= {args.p95_budget:.0f} ms; {args.note}"))
        print(f"ledger: rows under {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
