"""Labeled tickets -> prompt -> label examples for the routing adapter (train/configs/routing.yaml).

One example = system prompt with the 12-category taxonomy (CONTRACT.md §9; the ids come from
schemas/tools/create_ticket.json, the machine-readable source of truth), the ticket text as the
user turn, and the label as the assistant turn (mlx_lm `messages` format, no tools). The target
renders as `<think>\\n\\n</think>\\n\\n<label><|im_end|>` with loss on the label only, and every label
plus `<|im_end|>` fits the batch job's 8-token greedy budget. Labels outside the taxonomy and empty
tickets are dropped and counted. Split is stratified by label.

    uv run python -m train.routing_format --tickets tickets.jsonl --out $R3_ARTIFACTS/datasets/routing-v1
    uv run python -m train.routing_format --from-db --out ...      # tickets table, label is not null
Session S's batch job can import ROUTING_SYSTEM_PROMPT and LABELS from here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from train.env import load_env, repo_root

_SCHEMA = json.loads((repo_root() / "schemas" / "tools" / "create_ticket.json").read_text())
LABELS: list[str] = list(_SCHEMA["function"]["parameters"]["properties"]["category"]["enum"])

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
assert set(DEFINITIONS) == set(LABELS), "taxonomy definitions out of sync with schemas/tools/create_ticket.json"

ROUTING_SYSTEM_PROMPT = (
    "Classify the customer's support ticket into exactly one category. Answer with the category id only, nothing else.\n"
    "Categories:\n" + "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in LABELS)
)


def format_tickets(tickets: list[dict]) -> tuple[list[dict], list[dict]]:
    examples, dropped = [], []
    for t in tickets:
        text = str(t.get("text") or t.get("summary") or "").strip()
        label = str(t.get("label") or "").strip()
        ticket_id = t.get("ticket_id")
        if not text:
            dropped.append({"ticket_id": ticket_id, "reason": "empty_text"})
            continue
        if label not in LABELS:
            dropped.append({"ticket_id": ticket_id, "reason": "invalid_label", "label": label})
            continue
        examples.append({
            "messages": [
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": label},
            ],
            "meta": {"ticket_id": ticket_id, "label": label},
        })
    return examples, dropped


def _split(examples: list[dict], valid_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for ex in examples:
        by_label.setdefault(ex["meta"]["label"], []).append(ex)
    train, valid = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_valid = int(round(len(group) * valid_fraction))
        valid += group[:n_valid]
        train += group[n_valid:]
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_routing_dataset(tickets: list[dict], out_dir: str | Path, valid_fraction: float = 0.1, seed: int = 0) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples, dropped = format_tickets(tickets)
    train, valid = _split(examples, valid_fraction, seed)
    files = {"train": "train.jsonl", "valid": "valid.jsonl"}
    for split, rows in (("train", train), ("valid", valid)):
        with (out / files[split]).open("w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    manifest = {
        "name": out.name,
        "version": "0.1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": "train/routing_format.py",
        "seed": seed,
        "format": "mlx_lm.messages",
        "task": "routing",
        "enable_thinking": False,
        "files": files,
        "sha256": {name: _sha256(out / name) for name in files.values()},
        "counts": {"train": len(train), "valid": len(valid)},
        "dropped": len(dropped),
        "dropped_reasons": dict(Counter(d["reason"] for d in dropped)),
        "labels": LABELS,
        "label_counts": {"train": dict(Counter(ex["meta"]["label"] for ex in train)),
                         "valid": dict(Counter(ex["meta"]["label"] for ex in valid))},
        "system_prompt_sha256": hashlib.sha256(ROUTING_SYSTEM_PROMPT.encode()).hexdigest(),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def read_tickets_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def read_tickets_db(database_url: str) -> list[dict]:
    import psycopg

    with psycopg.connect(database_url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("select ticket_id, summary, label from tickets where label is not null order by ticket_id")
        return [{"ticket_id": tid, "text": summary, "label": label} for tid, summary, label in cur.fetchall()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickets", type=Path, help="jsonl with ticket_id, text (or summary), label")
    src.add_argument("--from-db", action="store_true", help="labeled rows of the Postgres tickets table")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.from_db:
        import os

        load_env()
        tickets = read_tickets_db(os.environ["DATABASE_URL"])
    else:
        tickets = read_tickets_jsonl(a.tickets)
    path = write_routing_dataset(tickets, a.out, a.valid_fraction, a.seed)
    m = json.loads(path.read_text())
    print(f"wrote {m['counts']} examples, dropped {m['dropped']} {m['dropped_reasons']}; manifest {path}")
    print(f"next: uv run python -m train.loader --data {a.out} --model $R3_ARTIFACTS/models/Qwen3-4B-8bit --max-seq-length 1024 --no-require-tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
