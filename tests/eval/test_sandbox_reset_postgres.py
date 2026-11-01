"""eval.sandbox.reset loads seed.json into the Postgres sandbox tables (rolled back here)."""

from datetime import date

import pytest

from eval.sandbox import load_seed
from eval.sandbox.reset import load_into_postgres

pytestmark = pytest.mark.db


def test_load_replaces_sandbox_tables_with_seed(rollback_conn):
    seed = load_seed()
    counts = load_into_postgres(rollback_conn, seed)
    n_cards = sum(len(c["cards"]) for c in seed["customers"])
    n_apps = sum(len(c["appointments"]) for c in seed["customers"])
    assert counts == {"customers": 10, "cards": n_cards, "appointments": n_apps, "tickets": 0}
    row = rollback_conn.execute(
        "select first_name, phone, date_of_birth, branch_id from customers where customer_id='C-1001'"
    ).fetchone()
    assert row == ("Dana", "5550142219", date(1988, 3, 12), "market_st")
    assert rollback_conn.execute("select status from cards where card_last4='9912'").fetchone() == (
        "blocked",
    )


def test_load_twice_is_idempotent(rollback_conn):
    seed = load_seed()
    load_into_postgres(rollback_conn, seed)
    counts = load_into_postgres(rollback_conn, seed)
    assert counts["customers"] == 10
