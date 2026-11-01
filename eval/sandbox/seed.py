"""seed.json access and calendar helpers. The seed file is human-owned (never edited here).

`eval/sandbox/seed.json` fixes the clock (Monday 2026-11-02 10:00), branches, services, hours,
customers with cards and appointments, blocked slots, and the `rules` text every tool follows.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

SEED_PATH = Path(__file__).with_name("seed.json")
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@lru_cache(maxsize=4)
def _read(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_seed(path: str | Path = SEED_PATH) -> dict:
    """A fresh deep copy of the seed, so callers may mutate it."""
    return copy.deepcopy(_read(str(path)))


@dataclass
class Customer:
    customer_id: str
    first_name: str
    last_name: str
    phone: str
    date_of_birth: date
    branch_id: str


@dataclass
class Card:
    card_id: str
    customer_id: str
    card_last4: str
    status: str = "active"  # active | blocked
    replacement: str = "none"  # none | standard | express


@dataclass
class Appointment:
    appointment_id: str
    customer_id: str
    branch_id: str
    service_type: str
    slot_start: datetime
    status: str = "booked"  # booked | cancelled
    idempotency_key: str | None = None

    @property
    def slot(self) -> str:
        return self.slot_start.strftime("%Y-%m-%dT%H:%M")


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    category: str
    summary: str
    status: str = "open"
    idempotency_key: str | None = None


def hours_for(d: date, seed: dict) -> tuple[str, str] | None:
    h = seed["hours"].get(WEEKDAYS[d.weekday()])
    return (h[0], h[1]) if h else None


def slots_for(d: date, seed: dict) -> list[str]:
    """Every slot start ("HH:MM") inside the day's hours; [] when closed."""
    hours = hours_for(d, seed)
    if hours is None:
        return []
    cur = datetime.combine(d, time.fromisoformat(hours[0]))
    end = datetime.combine(d, time.fromisoformat(hours[1]))
    out = []
    while cur < end:
        out.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=int(seed.get("slot_minutes", 30)))
    return out


def slot_dt(d: date, hhmm: str) -> datetime:
    return datetime.combine(d, time.fromisoformat(hhmm))


_DEFAULT = _read(str(SEED_PATH))
DEFAULT_CLOCK = datetime.fromisoformat(_DEFAULT["clock"])
DEFAULT_TODAY = DEFAULT_CLOCK.date()
BRANCHES: dict[str, str] = {k: v["name"] for k, v in _DEFAULT["branches"].items()}
SERVICES: dict[str, dict] = _DEFAULT["services"]
CARD_REPLACEMENT: dict[str, dict] = _DEFAULT.get("card_replacement", {})  # removed with block_card
