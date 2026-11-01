"""Deterministic Marlow Mutual seed data (CONTRACT.md §1). Synthetic only, via Faker.

Everything here is a pure function of (seed, today) so a scenario file can name a
customer by phone + date of birth and get the same record on every machine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from faker import Faker

BRANCHES: dict[str, str] = {
    "market_st": "Market Street",
    "marlow_st": "Marlow Street",
    "marker_st": "Marker Street",
    "mill_rd": "Mill Road",
}

SERVICES: dict[str, dict] = {
    "account_opening": {"duration_min": 30, "fee": "free"},
    "mortgage_consultation": {"duration_min": 30, "fee": "free"},
    "loan_application": {"duration_min": 30, "fee": "free"},
    "financial_review": {"duration_min": 30, "fee": "free"},
    "notary": {"duration_min": 30, "fee": "$10 per document"},
    "safe_deposit_box": {"duration_min": 30, "fee": "$60 per year, small box"},
}

CARD_REPLACEMENT: dict[str, dict] = {
    "standard": {"eta": "5-7 business days", "fee_usd": 0},
    "express": {"eta": "2 business days", "fee_usd": 25},
}

TICKET_CATEGORIES = [
    "card_lost_stolen",
    "card_declined",
    "unauthorized_transaction",
    "fee_dispute",
    "account_access",
    "statement_document",
    "profile_change",
    "loan_mortgage",
    "appointment",
    "transfer_payment",
    "complaint",
    "other",
]

SLOT_MINUTES = 30
BOOKING_HORIZON = timedelta(days=21)
DEFAULT_TODAY = date(2026, 11, 2)  # a Monday; scenarios set their own `today`
DEFAULT_SEED = 42

# The contract's anchor customer (G1–G3). Always present, never a filler.
ANCHOR = {
    "customer_id": "C-1042",
    "first_name": "Dana",
    "last_name": "Whitfield",
    "phone": "5550142219",
    "date_of_birth": date(1988, 3, 12),
    "branch_id": "market_st",
    "cards": ["4219"],
}


@dataclass
class Customer:
    customer_id: str
    first_name: str
    last_name: str
    phone: str
    date_of_birth: date
    branch_id: str
    filler: bool = False  # filler customers only exist to occupy slots


@dataclass
class Card:
    card_id: str
    customer_id: str
    card_last4: str
    status: str = "active"  # active | blocked
    replacement: str | None = None


@dataclass
class Appointment:
    appointment_id: str
    customer_id: str
    branch_id: str
    service_type: str
    slot_start: datetime
    status: str = "booked"  # booked | cancelled
    idempotency_key: str | None = None


@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    category: str
    summary: str
    idempotency_key: str | None = None


@dataclass
class SeedData:
    seed: int
    today: date
    customers: list[Customer] = field(default_factory=list)
    cards: list[Card] = field(default_factory=list)
    appointments: list[Appointment] = field(default_factory=list)


def hours_for(d: date) -> tuple[str, str] | None:
    """Opening hours for a date: Mon–Fri 9–17, Sat 9–13, Sun closed."""
    wd = d.weekday()
    if wd < 5:
        return ("09:00", "17:00")
    if wd == 5:
        return ("09:00", "13:00")
    return None


def slots_for(d: date) -> list[str]:
    """All 30-minute slot start times ("HH:MM") for a date; [] when closed."""
    hours = hours_for(d)
    if hours is None:
        return []
    start = datetime.combine(d, time.fromisoformat(hours[0]))
    end = datetime.combine(d, time.fromisoformat(hours[1]))
    out, cur = [], start
    while cur < end:
        out.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=SLOT_MINUTES)
    return out


def slot_dt(d: date, hhmm: str) -> datetime:
    return datetime.combine(d, time.fromisoformat(hhmm))


def generate_seed(
    seed: int = DEFAULT_SEED,
    today: date = DEFAULT_TODAY,
    n_customers: int = 60,
    n_filler: int = 260,
    occupancy: float = 0.7,
) -> SeedData:
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)
    data = SeedData(seed=seed, today=today)

    phones: set[str] = {ANCHOR["phone"]}
    data.customers.append(
        Customer(
            ANCHOR["customer_id"],
            ANCHOR["first_name"],
            ANCHOR["last_name"],
            ANCHOR["phone"],
            ANCHOR["date_of_birth"],
            ANCHOR["branch_id"],
        )
    )
    for last4 in ANCHOR["cards"]:
        data.cards.append(Card(f"K-{rng.randint(100000, 999999)}", ANCHOR["customer_id"], last4))

    def new_phone() -> str:
        while True:
            p = fake.numerify("555#######")
            if p not in phones:
                phones.add(p)
                return p

    def new_dob() -> date:
        return today - timedelta(days=rng.randint(20 * 365, 85 * 365))

    branches = list(BRANCHES)
    cid = 1000
    for _ in range(n_customers - 1):
        if cid == 1042:
            cid += 1
        c = Customer(
            f"C-{cid}",
            fake.first_name(),
            fake.last_name(),
            new_phone(),
            new_dob(),
            rng.choice(branches),
        )
        cid += 1
        data.customers.append(c)
        last4s: set[str] = set()
        for _k in range(rng.choice([1, 1, 2])):
            while True:
                l4 = f"{rng.randint(0, 9999):04d}"
                if l4 not in last4s:
                    last4s.add(l4)
                    break
            data.cards.append(Card(f"K-{rng.randint(100000, 999999)}", c.customer_id, l4))

    fillers = []
    for _ in range(n_filler):
        c = Customer(
            f"C-{cid}",
            fake.first_name(),
            fake.last_name(),
            new_phone(),
            new_dob(),
            rng.choice(branches),
            filler=True,
        )
        cid += 1
        data.customers.append(c)
        fillers.append(c)

    # Appointments. One banker per branch: a (branch, slot) holds at most one booking.
    taken: set[tuple[str, datetime]] = set()
    aid = 7000
    days = [today + timedelta(days=i) for i in range(BOOKING_HORIZON.days + 1)]
    services = list(SERVICES)

    def book(cust: Customer, branch: str, dt: datetime, service: str) -> None:
        nonlocal aid
        data.appointments.append(Appointment(f"A-{aid}", cust.customer_id, branch, service, dt))
        aid += 1
        taken.add((branch, dt))

    # Real (non-anchor, non-filler) customers: about half hold one upcoming appointment.
    for c in data.customers[1:n_customers]:
        if rng.random() < 0.5:
            for _try in range(10):
                d = rng.choice(days[1:])
                slots = slots_for(d)
                if not slots:
                    continue
                dt = slot_dt(d, rng.choice(slots))
                if (c.branch_id, dt) not in taken:
                    book(c, c.branch_id, dt, rng.choice(services))
                    break

    # Fillers occupy roughly `occupancy` of every branch's remaining slots.
    for d in days:
        for hhmm in slots_for(d):
            dt = slot_dt(d, hhmm)
            for branch in branches:
                if (branch, dt) in taken:
                    continue
                if rng.random() < occupancy:
                    book(rng.choice(fillers), branch, dt, rng.choice(services))

    return data
