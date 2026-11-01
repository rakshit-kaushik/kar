"""Load eval/sandbox/seed.json into the Postgres sandbox tables (customers, cards,
appointments, tickets) — the RUNBOOK's "reset of sandbox tables only".

    uv run python -m eval.sandbox.reset [--seed-file eval/sandbox/seed.json] [--dry-run]

Eval episodes do NOT use these tables (each episode has its own in-memory Sandbox);
this is for the live agent (session S) and the demo, so a call sees the same data.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from . import seed as S

TABLES_IN_DELETE_ORDER = ("tickets", "appointments", "cards", "customers")


def load_into_postgres(conn, seed: dict, clear: bool = True) -> dict[str, int]:
    """Replace the sandbox rows with the seed inside the caller's transaction. Returns counts."""
    customers, cards, appointments = [], [], []
    for c in seed["customers"]:
        cid = c["customer_id"]
        customers.append(
            (
                cid,
                c["first_name"],
                c["last_name"],
                c["phone"],
                date.fromisoformat(c["date_of_birth"]),
                c["branch_id"],
            )
        )
        for k in c.get("cards", []):
            cards.append(
                (
                    k["card_id"],
                    cid,
                    str(k["card_last4"]),
                    k.get("status", "active"),
                    k.get("replacement"),
                )
            )
        for a in c.get("appointments", []):
            appointments.append(
                (
                    a["appointment_id"],
                    cid,
                    a["branch_id"],
                    a["service_type"],
                    datetime.fromisoformat(a["slot_start"]),
                    a.get("status", "booked"),
                    None,
                )
            )
    with conn.cursor() as cur:
        if clear:
            for t in TABLES_IN_DELETE_ORDER:
                cur.execute(f"delete from {t}")
        cur.executemany(
            "insert into customers (customer_id, first_name, last_name, phone, date_of_birth, branch_id)"
            " values (%s, %s, %s, %s, %s, %s)",
            customers,
        )
        cur.executemany(
            "insert into cards (card_id, customer_id, card_last4, status, replacement) values (%s, %s, %s, %s, %s)",
            cards,
        )
        cur.executemany(
            "insert into appointments (appointment_id, customer_id, branch_id, service_type, slot_start, status,"
            " idempotency_key) values (%s, %s, %s, %s, %s, %s, %s)",
            appointments,
        )
        counts = {}
        for t in ("customers", "cards", "appointments", "tickets"):
            cur.execute(f"select count(*) from {t}")
            counts[t] = cur.fetchone()[0]
    return counts


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed-file", default=str(S.SEED_PATH))
    p.add_argument(
        "--dry-run", action="store_true", help="generate and count, do not touch Postgres"
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    a = p.parse_args(argv)
    data = S.load_seed(a.seed_file)
    if a.dry_run:
        print(
            f"clock={data['clock']} customers={len(data['customers'])}"
            f" cards={sum(len(c['cards']) for c in data['customers'])}"
            f" appointments={sum(len(c['appointments']) for c in data['customers'])}"
        )
        return 0
    if not a.database_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    import psycopg

    with psycopg.connect(a.database_url) as conn:
        counts = load_into_postgres(conn, data)
        conn.commit()
    print("sandbox tables reset:", " ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
