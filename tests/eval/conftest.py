import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def pytest_configure(config):
    config.addinivalue_line("markers", "db: needs a reachable Postgres at DATABASE_URL")
    config.addinivalue_line(
        "markers", "live: hits a real model endpoint (needs .env and R3_LIVE=1)"
    )


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    import psycopg

    try:
        with psycopg.connect(url, connect_timeout=3) as c:
            c.execute("select 1")
    except psycopg.Error as e:  # pragma: no cover
        pytest.skip(f"Postgres unreachable: {e}")
    return url


@pytest.fixture
def rollback_conn(database_url):
    """A connection whose work is always rolled back: safe against the shared r3 database."""
    import psycopg

    conn = psycopg.connect(database_url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
