"""Generate ~1,000 labeled synthetic tickets with the hosted model, verify the test split, write tickets.jsonl.

    uv run python -m data.routing.generate --per-label 84 --out $R3_ARTIFACTS/datasets/tickets-v1 [--parallel 12]

Per request: one category, one variation spec (length, tone, noise, channel), 6 tickets as a JSON
list; the label is the category asked for. Test split (100, stratified) gets a second pass: the
hosted model classifies each test ticket zero-shot with the shared routing prompt; disagreements are
flagged (`verify_agrees: false`) for the human's hand check. Training files come from T's
`python -m train.routing_format --tickets <out>/tickets.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from data.routing.prompt import DEFINITIONS, ROUTING_SYSTEM_PROMPT
from data.routing.score import parse_label
from data.routing.tickets import (
    LABELS,
    assign_ids,
    dedupe,
    parse_batch,
    split_tickets,
    variation_specs,
)

ROOT = Path(__file__).resolve().parents[2]
PER_REQUEST = 6
CONTRACT_EXAMPLES = {
    "card_lost_stolen": "lost my debit card; wallet stolen; card never arrived; found my card, unblock it; replacement card request",
    "card_declined": "declined at the grocery store; ATM won't give cash; card stopped working abroad; chip not reading; online purchase blocked",
    "unauthorized_transaction": "charge I don't recognize; someone used my card; account drained; subscription I never signed up for; duplicate charge from unknown merchant",
    "fee_dispute": "overdraft fee unfair; monthly fee should be waived; ATM fee refund; late fee on credit card; wire fee too high",
    "account_access": "locked out of app; forgot password; two-factor code not arriving; new phone, can't log in; username unknown",
    "statement_document": "need last year's statements; 1099 form; proof of funds letter; mortgage interest statement; void check",
    "profile_change": "moved, new address; change phone number; name change after marriage; update email; add joint owner",
    "loan_mortgage": "mortgage payoff amount; refinance rates; car loan payment date; HELOC application; pre-approval letter",
    "appointment": "book a notary; reschedule Thursday; cancel my appointment; which branch is open Saturday; see a banker about retirement",
    "transfer_payment": "wire not received; transfer pending for days; bill pay went to wrong payee; Zelle failed; recurring transfer stopped",
    "complaint": "rude teller; waited an hour; promised a call back and nobody did; branch closed early; app keeps crashing",
    "other": "foreign currency order; notary hours; check cashing policy; empty ticket; spam",
}
LENGTH_WORDS = {
    "short": "5 to 15 words",
    "medium": "20 to 45 words",
    "long": "60 to 120 words, with some irrelevant detail",
}


def gen_prompt(label: str, spec: dict, n: int) -> str:
    return (
        f"Write {n} different customer support tickets sent to Marlow Mutual Bank, a fictional retail bank. "
        f"Every ticket must belong to the category '{label}': {DEFINITIONS[label]}. Example topics: {CONTRACT_EXAMPLES[label]}.\n"
        f"Variation: length {LENGTH_WORDS[spec['length']]}; tone {spec['tone']}; channel {spec['channel']}; text style {spec['noise']}.\n"
        "Use realistic but fictional details (made-up names, last four digits, amounts, dates in November 2026); never real PII. "
        "Do not name the category or use its id. Vary the opening words across the tickets.\n"
        f"Answer with a JSON list of {n} strings and nothing else."
    )


async def _complete(
    client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int, sem: asyncio.Semaphore
) -> str:
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 — retry transient API errors, then give up on this batch
                if attempt == 3:
                    print(f"request failed: {e}", file=sys.stderr)
                    return ""
                await asyncio.sleep(2 * (attempt + 1))
    return ""


async def generate_tickets(
    client: AsyncOpenAI, model: str, per_label: int, seed: int, parallel: int
) -> list[dict]:
    sem = asyncio.Semaphore(parallel)
    batches = (per_label + PER_REQUEST - 1) // PER_REQUEST
    jobs = []
    for li, label in enumerate(LABELS):
        for bi, spec in enumerate(variation_specs(batches, seed * 100 + li)):
            jobs.append((label, spec, bi))
    outs = await asyncio.gather(
        *(
            _complete(
                client,
                model,
                [{"role": "user", "content": gen_prompt(label, spec, PER_REQUEST)}],
                1800,
                sem,
            )
            for label, spec, _ in jobs
        )
    )
    tickets = []
    for (label, spec, _), raw in zip(jobs, outs, strict=True):
        tickets.extend(parse_batch(raw, label, spec))
    return tickets


async def verify_tickets(
    client: AsyncOpenAI, model: str, tickets: list[dict], parallel: int
) -> list[dict]:
    sem = asyncio.Semaphore(parallel)
    msgs = [
        [
            {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
            {"role": "user", "content": t["text"]},
        ]
        for t in tickets
    ]
    outs = await asyncio.gather(*(_complete(client, model, m, 20, sem) for m in msgs))
    out = []
    for t, raw in zip(tickets, outs, strict=True):
        v = parse_label(raw)
        out.append({**t, "verify_label": v, "verify_agrees": v == t["label"]})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--per-label",
        type=int,
        default=84,
        help="tickets to request per category (12 x 84 ≈ 1,000)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--parallel", type=int, default=12)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument(
        "--extend",
        default=None,
        help="existing tickets.jsonl: keep it and its test split, append new train tickets",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(
        base_url=os.environ["HOSTED_LLM_BASE_URL"],
        api_key=os.environ["HOSTED_LLM_API_KEY"],
        max_retries=0,
    )
    model = os.environ["HOSTED_LLM_MODEL"]
    tickets = asyncio.run(generate_tickets(client, model, a.per_label, a.seed, a.parallel))
    n_raw = len(tickets)
    if a.extend:
        from data.routing.tickets import extend_tickets

        existing = [
            json.loads(line) for line in Path(a.extend).read_text().splitlines() if line.strip()
        ]
        tickets = extend_tickets(existing, dedupe(tickets))
        test = [x for x in tickets if x["split"] == "test"]
        verified = {x["ticket_id"]: x for x in test}  # verified when the base set was built
    else:
        tickets = split_tickets(assign_ids(dedupe(tickets)), test_size=a.test_size, seed=a.seed)
        test = [x for x in tickets if x["split"] == "test"]
        verified = {
            x["ticket_id"]: x for x in asyncio.run(verify_tickets(client, model, test, a.parallel))
        }
        tickets = [verified.get(x["ticket_id"], x) for x in tickets]
    for x in tickets:
        x.setdefault("source", "synthetic-hosted-v1")
    with (out / "tickets.jsonl").open("w") as fh:
        for t in tickets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    per_label = {lab: sum(1 for t in tickets if t["label"] == lab) for lab in LABELS}
    agree = sum(1 for x in test if verified[x["ticket_id"]].get("verify_agrees"))
    manifest = {
        "name": out.name,
        "files": {"tickets": "tickets.jsonl"},
        "sha256": {
            "tickets.jsonl": hashlib.sha256((out / "tickets.jsonl").read_bytes()).hexdigest()
        },
        "n_tickets": len(tickets),
        "n_raw": n_raw,
        "n_deduped": n_raw - len(tickets),
        "per_label": per_label,
        "split": {"test": len(test), "train": len(tickets) - len(test)},
        "test_verification": {
            "model": model,
            "agree": agree,
            "disagree": len(test) - agree,
            "note": "machine-checked only; the human's hand check of the 100 test tickets is pending",
        },
        "routing_prompt_sha256": hashlib.sha256(ROUTING_SYSTEM_PROMPT.encode()).hexdigest(),
        "generator": "data.routing.generate",
        "model": model,
        "seed": a.seed,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("n_tickets", "n_raw", "per_label", "split", "test_verification")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
