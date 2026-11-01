"""Seed data for the Marlow Mutual sandbox: deterministic, contract-shaped."""

from datetime import date, datetime

from eval.sandbox import seed as S


def test_four_branches_with_contract_names():
    assert list(S.BRANCHES) == ["market_st", "marlow_st", "marker_st", "mill_rd"]
    assert S.BRANCHES["marlow_st"] == "Marlow Street"
    assert S.BRANCHES["mill_rd"] == "Mill Road"


def test_catalog_matches_contract():
    assert set(S.SERVICES) == {
        "account_opening",
        "mortgage_consultation",
        "loan_application",
        "financial_review",
        "notary",
        "safe_deposit_box",
    }
    assert all(v["duration_min"] == 30 for v in S.SERVICES.values())
    assert S.SERVICES["notary"]["fee"] == "$10 per document"
    assert S.SERVICES["safe_deposit_box"]["fee"] == "$60 per year, small box"
    assert S.SERVICES["mortgage_consultation"]["fee"] == "free"
    assert S.CARD_REPLACEMENT["standard"] == {"eta": "5-7 business days", "fee_usd": 0}
    assert S.CARD_REPLACEMENT["express"] == {"eta": "2 business days", "fee_usd": 25}


def test_hours_weekday_saturday_sunday():
    assert S.hours_for(date(2026, 11, 2)) == ("09:00", "17:00")  # Monday
    assert S.hours_for(date(2026, 11, 7)) == ("09:00", "13:00")  # Saturday
    assert S.hours_for(date(2026, 11, 8)) is None  # Sunday closed


def test_slots_for_day_are_thirty_minutes():
    assert S.slots_for(date(2026, 11, 2))[:3] == ["09:00", "09:30", "10:00"]
    assert S.slots_for(date(2026, 11, 2))[-1] == "16:30"
    assert len(S.slots_for(date(2026, 11, 7))) == 8
    assert S.slots_for(date(2026, 11, 8)) == []


def test_anchor_customer_dana_from_contract():
    data = S.generate_seed(seed=42, today=date(2026, 11, 2))
    dana = next(c for c in data.customers if c.customer_id == "C-1042")
    assert (dana.first_name, dana.phone, dana.date_of_birth, dana.branch_id) == (
        "Dana",
        "5550142219",
        date(1988, 3, 12),
        "market_st",
    )
    cards = [k for k in data.cards if k.customer_id == "C-1042"]
    assert [k.card_last4 for k in cards] == ["4219"]
    assert cards[0].status == "active"


def test_seed_is_deterministic_and_phones_unique():
    a = S.generate_seed(seed=42, today=date(2026, 11, 2))
    b = S.generate_seed(seed=42, today=date(2026, 11, 2))
    assert [c.phone for c in a.customers] == [c.phone for c in b.customers]
    assert [x.appointment_id for x in a.appointments] == [x.appointment_id for x in b.appointments]
    phones = [c.phone for c in a.customers]
    assert len(set(phones)) == len(phones)
    assert all(len(p) == 10 and p.isdigit() for p in phones)
    c = S.generate_seed(seed=7, today=date(2026, 11, 2))
    assert [x.phone for x in c.customers if not x.filler] != [
        x.phone for x in a.customers if not x.filler
    ]


def test_seed_appointments_sit_inside_hours_and_horizon():
    today = date(2026, 11, 2)
    data = S.generate_seed(seed=42, today=today)
    assert data.today == today
    assert len(data.appointments) > 100
    for ap in data.appointments:
        assert isinstance(ap.slot_start, datetime)
        d = ap.slot_start.date()
        assert today <= d <= today + S.BOOKING_HORIZON
        assert ap.slot_start.strftime("%H:%M") in S.slots_for(d)
        assert ap.status == "booked"
    # one banker per branch: a (branch, slot) is never booked twice in the seed
    keys = [(ap.branch_id, ap.slot_start) for ap in data.appointments]
    assert len(keys) == len(set(keys))


def test_seed_leaves_open_slots_but_not_too_many():
    data = S.generate_seed(seed=42, today=date(2026, 11, 2))
    taken = {(ap.branch_id, ap.slot_start) for ap in data.appointments}
    d = date(2026, 11, 4)
    open_slots = [
        s
        for s in S.slots_for(d)
        if ("market_st", datetime.combine(d, datetime.strptime(s, "%H:%M").time())) not in taken
    ]
    assert 2 <= len(open_slots) <= 8


def test_customers_have_cards_and_sensible_dob():
    data = S.generate_seed(seed=42, today=date(2026, 11, 2))
    real = [c for c in data.customers if not c.filler]
    assert len(real) >= 40
    by_cust = {}
    for k in data.cards:
        by_cust.setdefault(k.customer_id, []).append(k)
    for c in real:
        assert 1 <= len(by_cust[c.customer_id]) <= 2
        assert all(
            len(k.card_last4) == 4 and k.card_last4.isdigit() for k in by_cust[c.customer_id]
        )
        assert 18 <= (data.today - c.date_of_birth).days // 365 <= 90
