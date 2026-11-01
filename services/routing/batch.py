"""Ticket routing batch job (CONTRACT.md §9, plan 'Ticket routing track').

Pulls unlabeled tickets from Postgres, labels them with the routing model through mlx_lm batch generation
(greedy, at most 8 new tokens), validates every label against the taxonomy (invalid -> needs_review), writes
the labels back with their source, and records a cost_ledger row. The prompt is the one the routing adapter
is trained on (train/routing_format.py when present; an identical local copy otherwise, checked by a test).

  uv run python -m services.routing.batch --model-dir $R3_ARTIFACTS/models/Qwen3-4B-8bit --limit 200 [--dry-run]
  uv run python -m services.routing.batch --model-dir $R3_ARTIFACTS/models/r3-route-<hash> --source r3-route-<hash>

The job takes the GPU lock (scripts/gpu_lock.sh, owner R) for its run and never overlaps a live call window
by construction: the scheduler that calls it is responsible for the window; the lock is the guard.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCHEMA = json.loads((ROOT / "schemas" / "tools" / "create_ticket.json").read_text())
LABELS: list[str] = list(_SCHEMA["function"]["parameters"]["properties"]["category"]["enum"])
NEEDS_REVIEW = "needs_review"
MAX_NEW_TOKENS = 8

try:  # the training-side prompt is the source of truth once T's module is on main
    from train.routing_format import ROUTING_SYSTEM_PROMPT  # type: ignore
except Exception:  # pragma: no cover - exercised until the training module lands on main
    DEFINITIONS = {
        "card_lost_stolen": "card missing or taken; needs block or replacement (lost, stolen, never arrived, found it, replacement)",
        "card_declined": "card refused at a merchant or ATM while the card is in hand (declined, ATM, abroad, chip, online blocked)",
        "unauthorized_transaction": "charges the customer did not make; fraud (unrecognized charge, someone used my card, drained)",
        "fee_dispute": "customer disputes a fee the bank charged (overdraft, monthly, ATM, late, wire fee)",
        "account_access": "cannot log in or use online or mobile banking (locked out, password, two-factor, new phone)",
        "statement_document": "wants a statement, tax form or letter (statements, 1099, proof of funds, interest statement, void check)",
        "profile_change": "address, phone, name or email update; add a joint owner",
        "loan_mortgage": "questions on existing or new loans (payoff, refinance, car loan, HELOC, pre-approval)",
        "appointment": "wants to book, move or cancel a branch visit, or asks branch hours for one",
        "transfer_payment": "wires, transfers or bill pay not going through (pending, wrong payee, Zelle failed, recurring stopped)",
        "complaint": "dissatisfaction with service and no transactional ask (rude, waited, no call back, closed early, app crashes)",
        "other": "none of the above, or unreadable (currency order, notary hours, check cashing policy, empty, spam)",
    }
    ROUTING_SYSTEM_PROMPT = (
        "Classify the customer's support ticket into exactly one category. Answer with the category id only, nothing else.\n"
        "Categories:\n" + "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in LABELS)
    )


def validate_label(raw: str) -> str:
    """The model's text -> a taxonomy id, or needs_review. Tolerates whitespace, case, a stray <think> block and
    trailing punctuation; anything that is not exactly one id is invalid."""
    s = (raw or "").strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1]
    s = s.strip().strip(".,;:!\"'`").strip().lower()
    return s if s in LABELS else NEEDS_REVIEW


def build_messages(ticket_text: str) -> list[dict]:
    return [{"role": "system", "content": ROUTING_SYSTEM_PROMPT}, {"role": "user", "content": ticket_text}]


def label_tickets(tickets: list[dict], generate: Callable[[list[list[dict]]], list[str]], source: str) -> list[dict]:
    """tickets: [{ticket_id, text|summary}]; generate: batch of message lists -> raw texts (same order)."""
    msgs = [build_messages(str(t.get("text") or t.get("summary") or "").strip()) for t in tickets]
    raws = generate(msgs) if msgs else []
    assert len(raws) == len(tickets), "generator must return one text per ticket"
    return [{"ticket_id": t["ticket_id"], "label": validate_label(raw), "raw": raw, "label_source": source} for t, raw in zip(tickets, raws)]


# ---- model -----------------------------------------------------------------------------------------------
def mlx_generator(model_dir: str, batch_size: int = 16) -> Callable[[list[list[dict]]], list[str]]:
    from mlx_lm import batch_generate, load

    model, tokenizer = load(model_dir)

    def generate(batch_msgs: list[list[dict]]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(batch_msgs), batch_size):
            chunk = batch_msgs[i:i + batch_size]
            prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=False) for m in chunk]
            prompts = [p if isinstance(p, list) else tokenizer.encode(p) for p in prompts]
            resp = batch_generate(model, tokenizer, prompts, max_tokens=MAX_NEW_TOKENS, verbose=False)
            out.extend(resp.texts)
        return out

    return generate


# ---- storage ---------------------------------------------------------------------------------------------
def fetch_unlabeled(con, limit: int) -> list[dict]:
    rows = con.execute("select ticket_id, summary from tickets where label is null order by created_at limit %s", (limit,)).fetchall()
    return [{"ticket_id": r[0], "text": r[1]} for r in rows]


def write_labels(con, rows: list[dict]) -> int:
    with con.cursor() as cur:
        cur.executemany("update tickets set label=%s, label_source=%s where ticket_id=%s", [(r["label"], r["label_source"], r["ticket_id"]) for r in rows])
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--source", default=None, help="label_source written back (default: basename of model dir)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="label but do not write back")
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args()
    from dotenv import load_dotenv
    import psycopg
    load_dotenv(ROOT / ".env")
    source = args.source or os.path.basename(args.model_dir.rstrip("/"))
    lock = ROOT / "scripts" / "gpu_lock.sh"
    if not args.no_lock and subprocess.run([str(lock), "acquire", "R", "20", "routing batch"], capture_output=True, text=True).returncode != 0:
        print(subprocess.run([str(lock), "status"], capture_output=True, text=True).stdout.strip(), "-> routing job skipped", file=sys.stderr)
        return 3
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
            tickets = fetch_unlabeled(con, args.limit)
            if not tickets:
                print("no unlabeled tickets"); return 0
            t0 = time.perf_counter()
            rows = label_tickets(tickets, mlx_generator(args.model_dir, args.batch_size), source)
            secs = time.perf_counter() - t0
            invalid = sum(1 for r in rows if r["label"] == NEEDS_REVIEW)
            if not args.dry_run:
                write_labels(con, rows)
                con.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values ('routing-batch', %s, 'routing.tickets_labeled', %s, 'tickets', true, %s)",
                            (source, len(rows), f"invalid={invalid}; seconds={secs:.1f}; per_ticket_ms={1000 * secs / len(rows):.0f}"))
            print(f"labeled {len(rows)} tickets with {source} in {secs:.1f}s ({1000 * secs / len(rows):.0f} ms/ticket); needs_review={invalid}; {'dry run' if args.dry_run else 'written'}")
            for r in rows[:5]:
                print(f"  {r['ticket_id']}: {r['label']:<26} <- {r['raw'][:60]!r}")
    finally:
        if not args.no_lock:
            subprocess.run([str(lock), "release", "R"], capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
