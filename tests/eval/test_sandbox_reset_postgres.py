"""eval.sandbox.reset loads the seed into the Postgres sandbox tables (rolled back here)."""

from datetime import date

import pytest

from eval.sandbox import generate_seed
from eval.sandbox.reset import load_into_postgres

pytestmark = pytest.mark.db


def test_load_replaces_sandbox_tables_with_seed(rollback_conn):
    data = generate_seed(seed=42, today=date(2026, 11, 2))
    counts = load_into_postgres(rollback_conn, data)
    assert counts == {
        "customers": len(data.customers),
        "cards": len(data.cards),
        "appointments": len(data.appointments),
        "tickets": 0,
    }
    row = rollback_conn.execute(
        "select first_name, phone, date_of_birth, branch_id from customers where customer_id='C-1042'"
    ).fetchone()
    assert row == ("Dana", "5550142219", date(1988, 3, 12), "market_st")
    n = rollback_conn.execute("select count(*) from appointments where status='booked'").fetchone()[
        0
    ]
    assert n == len(data.appointments)


def test_load_twice_is_idempotent(rollback_conn):
    data = generate_seed(seed=42, today=date(2026, 11, 2))
    load_into_postgres(rollback_conn, data)
    counts = load_into_postgres(rollback_conn, data)
    assert counts["customers"] == len(data.customers)
    assert rollback_conn.execute("select count(*) from customers").fetchone()[0] == len(
        data.customers
    )
