#!/usr/bin/env python
"""Base-model throughput benchmark on MLX (single stream, no server).

Measures: (a) TTFT, prefill and decode tok/s on the REAL prompt (system prompt + 8 tool schemas +
a 6-message conversation); (b) prefill/decode at 1K/2K/4K synthetic prompt tokens; (c) turn-2 TTFT
with a warm prompt cache, i.e. only the new tokens are prefilled, which is what a server's prefix
cache does. Writes runs/<id>/bench.json and cost_ledger rows.

Usage: uv run python scripts/bench_base.py [--model DIR] [--repeats 3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import subprocess
import time

import mlx.core as mx
from dotenv import load_dotenv
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
ART = pathlib.Path(os.environ.get("R3_ARTIFACTS", str(pathlib.Path.home() / "r3-artifacts")))


def tools() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((ROOT / "schemas/tools").glob("*.json"))]


def system_prompt() -> str:
    return (ROOT / "schemas/system_prompt.md").read_text().replace("{today}", json.loads((ROOT / "eval/sandbox/seed.json").read_text())["clock_spoken"])


def turn1_messages() -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": "hi i need to book an appointment"},
        {"role": "assistant", "content": "Happy to help. Can I get the phone number on your account?"},
        {"role": "user", "content": "five five five zero one four two two one nine"},
        {"role": "assistant", "content": "Thanks. And your date of birth?"},
        {"role": "user", "content": "march twelfth nineteen eighty eight"},
    ]


def render(tok, messages, with_tools=True) -> list[int]:
    return tok.apply_chat_template(
        messages, tools=tools() if with_tools else None, add_generation_prompt=True,
        enable_thinking=False, tokenize=True,
    )


def gen(model, tok, prompt_tokens, max_tokens, prompt_cache=None) -> dict:
    t0 = time.perf_counter()
    ttft = None
    text = []
    last = None
    for r in stream_generate(model, tok, prompt_tokens, max_tokens=max_tokens, prompt_cache=prompt_cache):
        if ttft is None:
            ttft = time.perf_counter() - t0
        text.append(r.text)
        last = r
    return {
        "ttft_s": round(ttft, 4),
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": round(last.prompt_tps, 1),
        "gen_tokens": last.generation_tokens,
        "gen_tps": round(last.generation_tps, 1),
        "peak_gb": round(last.peak_memory, 2),
        "finish": last.finish_reason,
        "text": "".join(text),
    }


def median(rows, key):
    return round(statistics.median(r[key] for r in rows), 4)


def synthetic_tokens(tok, n: int) -> list[int]:
    filler = (ROOT / "CONTRACT.md").read_text() * 8
    ids = tok.encode(filler)
    assert len(ids) >= n, f"filler too short for {n}"
    return ids[:n]


def common_prefix(a: list[int], b: list[int]) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ART / "models/Qwen3-4B-8bit"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-ledger", action="store_true")
    args = ap.parse_args()

    run_id = "bench-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ART / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    t0 = time.perf_counter()
    model, tok = load(args.model)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"loaded {args.model} in {load_s}s")

    # Prompt anatomy
    sys_only = len(render(tok, [{"role": "system", "content": system_prompt()}], with_tools=False))
    sys_tools = len(render(tok, [{"role": "system", "content": system_prompt()}], with_tools=True))
    p1 = render(tok, turn1_messages())
    tail = tok.decode(p1[-24:])
    print(f"prompt tokens: system={sys_only} system+tools={sys_tools} full turn-1={len(p1)}; tail={tail!r}")

    # Warmup
    gen(model, tok, p1, 8)

    # (a) real prompt, cold prefill
    real = [gen(model, tok, p1, 48) for _ in range(args.repeats)]
    print("REAL cold:", {k: median(real, k) for k in ("ttft_s", "prompt_tps", "gen_tps", "peak_gb")}, "| sample:", real[0]["text"][:80].replace("\n", " "))

    # (b) synthetic lengths
    synth = {}
    for n in (1024, 2048, 4096):
        toks = synthetic_tokens(tok, n)
        rows = [gen(model, tok, toks, 64) for _ in range(args.repeats)]
        synth[n] = {k: median(rows, k) for k in ("ttft_s", "prompt_tps", "gen_tps", "peak_gb")}
        print(f"SYNTH {n}:", synth[n])

    # (c) prefix cache: turn 1 fills the cache, turn 2 prefills only the new tokens
    cache = make_prompt_cache(model)
    r1 = gen(model, tok, p1, 48, prompt_cache=cache)
    msgs2 = turn1_messages() + [
        {"role": "assistant", "content": r1["text"].strip() or "Thanks, you're verified. What can I help with?"},
        {"role": "user", "content": "i need to see someone about a mortgage next week at market street"},
    ]
    p2 = render(tok, msgs2)
    cached_len = cache[0].offset if hasattr(cache[0], "offset") else None
    shared = common_prefix(p1, p2)
    warm = None
    if cached_len is not None and can_trim_prompt_cache(cache):
        trim_prompt_cache(cache, cached_len - shared)
        warm = gen(model, tok, p2[shared:], 48, prompt_cache=cache)
        warm["new_tokens"] = len(p2) - shared
        print(f"WARM turn-2: new_tokens={warm['new_tokens']} ttft={warm['ttft_s']} prompt_tps={warm['prompt_tps']} gen_tps={warm['gen_tps']} | {warm['text'][:80]!r}")
    else:
        print("WARM turn-2: cache not trimmable; skipped")

    result = {
        "run_id": run_id, "git": git, "model": args.model, "load_s": load_s,
        "mlx": mx.__version__, "device": str(mx.default_device()),
        "prompt_tokens": {"system": sys_only, "system_plus_tools": sys_tools, "turn1_full": len(p1), "turn2_full": len(p2), "turn2_new": (len(p2) - shared)},
        "real_cold": {k: median(real, k) for k in ("ttft_s", "prompt_tps", "gen_tps", "peak_gb")},
        "real_cold_samples": [r["text"] for r in real],
        "synthetic": {str(k): v for k, v in synth.items()},
        "warm_turn2": {k: warm[k] for k in ("ttft_s", "prompt_tps", "gen_tps", "new_tokens")} if warm else None,
        "template_tail": tail,
    }
    (out_dir / "bench.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'bench.json'}")

    if not args.no_ledger:
        import psycopg
        rows = [
            ("bench.prompt_tokens.system_plus_tools", sys_tools, "tokens"),
            ("bench.prompt_tokens.turn1_full", len(p1), "tokens"),
            ("bench.real_cold.ttft", result["real_cold"]["ttft_s"], "s"),
            ("bench.real_cold.prefill_tps", result["real_cold"]["prompt_tps"], "tok/s"),
            ("bench.real_cold.decode_tps", result["real_cold"]["gen_tps"], "tok/s"),
        ] + [(f"bench.synth{n}.{m}", synth[n][k], u) for n in synth for m, k, u in (("ttft", "ttft_s", "s"), ("prefill_tps", "prompt_tps", "tok/s"), ("decode_tps", "gen_tps", "tok/s"))]
        if warm:
            rows += [("bench.warm_turn2.ttft", warm["ttft_s"], "s"), ("bench.warm_turn2.new_tokens", warm["new_tokens"], "tokens")]
        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
            con.cursor().executemany(
                "insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,%s,true,%s)",
                [(run_id, "qwen3-4b-8bit-base", item, qty, unit, f"git {git}; single stream mlx_lm {mx.__version__}") for item, qty, unit in rows],
            )
        print(f"ledger: {len(rows)} rows under {run_id}")


if __name__ == "__main__":
    main()
