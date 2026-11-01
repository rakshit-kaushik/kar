"""Zero-shot routing baseline: classify the test tickets with any OpenAI-compatible endpoint.

    uv run python -m data.routing.baseline --tickets $R3_ARTIFACTS/datasets/tickets-v1/tickets.jsonl \
        --base-url <url> --model <id> --out $R3_ARTIFACTS/runs/routing-<id>/ [--split test] [--parallel 8]

Greedy, at most 8 completion tokens, label validated against the taxonomy (invalid -> needs_review),
scored with per-category precision/recall/F1, the costly-pair-weighted confusion, and coverage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from data.routing.prompt import ROUTING_SYSTEM_PROMPT
from data.routing.score import parse_label, score_predictions

ROOT = Path(__file__).resolve().parents[2]


async def classify(
    client: AsyncOpenAI, model: str, texts: list[str], parallel: int, max_tokens: int = 64
) -> list[str]:
    sem = asyncio.Semaphore(parallel)

    async def one(text: str) -> str:
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                            {"role": "user", "content": text},
                        ],
                    )
                    return r.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001 — retry, then count as needs_review
                    if attempt == 3:
                        print(f"request failed: {e}", file=sys.stderr)
                        return ""
                    await asyncio.sleep(2 * (attempt + 1))
        return ""

    return await asyncio.gather(*(one(t) for t in texts))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tickets", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="8 is the 4B batch-job budget; hosted models spend tokens before the label",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    a = build_parser().parse_args(argv)
    tickets = [
        json.loads(line) for line in Path(a.tickets).read_text().splitlines() if line.strip()
    ]
    tickets = [t for t in tickets if t.get("split", "test") == a.split]
    key = a.api_key or (
        os.environ.get("HOSTED_LLM_API_KEY")
        if a.base_url.rstrip("/") == os.environ.get("HOSTED_LLM_BASE_URL", "").rstrip("/")
        else "none"
    )
    client = AsyncOpenAI(base_url=a.base_url, api_key=key, max_retries=0)
    raws = asyncio.run(
        classify(client, a.model, [t["text"] for t in tickets], a.parallel, a.max_tokens)
    )
    preds = [parse_label(r) for r in raws]
    gold = [t["label"] for t in tickets]
    s = score_predictions(gold, preds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as fh:
        for t, raw, p in zip(tickets, raws, preds, strict=True):
            fh.write(
                json.dumps({"ticket_id": t["ticket_id"], "gold": t["label"], "pred": p, "raw": raw})
                + "\n"
            )
    (out / "scores.json").write_text(
        json.dumps({"model": a.model, "base_url": a.base_url, "split": a.split, **s}, indent=1)
    )
    lines = [
        f"# Routing zero-shot — `{a.model}` on {len(tickets)} {a.split} tickets",
        "",
        f"accuracy {s['accuracy']:.3f} · macro F1 {s['macro_f1']:.3f} · coverage {s['coverage']:.3f} · weighted error {s['weighted_error']}",
        "",
        "| category | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for lab, v in sorted(s["per_category"].items()):
        lines.append(
            f"| {lab} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |"
        )
    lines += ["", f"costly confusions (x3): {s['costly_confusions'] or 'none'}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
