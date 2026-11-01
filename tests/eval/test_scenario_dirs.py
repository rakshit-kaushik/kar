"""Every scenario file must load, fit the contract, and be consistent with the sandbox seed."""

from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from eval.sandbox import Sandbox
from eval.sandbox import seed as S
from eval.sandbox.store import parse_slot
from eval.simulator.scenario import FAMILIES, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DIRS = [
    d
    for d in (ROOT / "eval/scenarios/gen", ROOT / "eval/scenarios/test")
    if any(p.suffix in (".yaml", ".yml", ".json") for p in d.iterdir())
]


@pytest.mark.parametrize("directory", DIRS, ids=lambda d: d.name)
def test_three_scenarios_per_family(directory):
    scen = load_scenarios(directory)
    counts = Counter(s.family for s in scen)
    assert counts == {f: 3 for f in FAMILIES}, counts


def all_scenarios():
    return [s for d in DIRS for s in load_scenarios(d)]


@pytest.mark.parametrize("scen", all_scenarios(), ids=lambda s: s.id)
def test_scenario_is_consistent_with_sandbox(scen):
    sb = Sandbox(seed=scen.seed, today=scen.today, fixture=scen.fixture.model_dump())
    exp = scen.expected
    if scen.customer_id:
        cust = sb.find_customer(scen.customer_id)
        assert cust is not None, f"{scen.id}: customer {scen.customer_id} not in sandbox"
        kf = scen.known_facts
        if "phone" in kf:
            assert str(kf["phone"]) == cust.phone, f"{scen.id}: known phone differs from record"
        if "date_of_birth" in kf and scen.family != "correction":
            assert date.fromisoformat(str(kf["date_of_birth"])) == cust.date_of_birth
        # every booked appointment the customer starts with is accounted for in expected
        start_ids = {a.appointment_id for a in sb.customer_appointments(scen.customer_id)}
        listed_slots = {str(e.get("slot_start", "")) for e in exp.final_state.appointments}
        for a in sb.customer_appointments(scen.customer_id):
            slot = a.slot_start.strftime("%Y-%m-%dT%H:%M")
            moved = any(
                step.name == "reschedule_appointment"
                and step.args.get("appointment_id") == a.appointment_id
                for step in exp.tool_steps
            )
            assert slot in listed_slots or moved, (
                f"{scen.id}: seed appointment {a.appointment_id} {slot} not in expected"
            )
        for step in exp.tool_steps:
            aid = step.args.get("appointment_id")
            if aid:
                assert aid in start_ids, (
                    f"{scen.id}: {aid} is not an appointment of {scen.customer_id}"
                )
    for e in exp.final_state.appointments:
        if e.get("slot_start"):
            dt = parse_slot(e["slot_start"])
            assert dt and dt.strftime("%H:%M") in S.slots_for(dt.date()), (
                f"{scen.id}: slot outside hours"
            )
            if e.get("status", "booked") == "booked" and e.get("branch_id"):
                open_or_seeded = sb.slot_is_open(e["branch_id"], dt) or any(
                    a.customer_id == scen.customer_id and a.slot_start == dt
                    for a in sb.appointments
                )
                assert open_or_seeded, (
                    f"{scen.id}: expected slot {e['slot_start']} at {e['branch_id']} is not bookable"
                )
        if e.get("date"):
            d = date.fromisoformat(str(e["date"]))
            assert S.hours_for(d) is not None, f"{scen.id}: expected date is a Sunday"
            if e.get("branch_id") and e.get("status", "booked") == "booked":
                assert sb.open_slots(e["branch_id"], d), (
                    f"{scen.id}: no open slots at {e['branch_id']} on {d}"
                )
    for f in scen.fixture.free:
        d = date.fromisoformat(str(f["date"]))
        for hhmm in f.get("slots") or S.slots_for(d):
            assert hhmm in sb.open_slots(f["branch_id"], d), f"{scen.id}: free slot {hhmm} not open"
    for step in exp.tool_steps:
        if step.name == "request_handoff" and "reason" in step.args:
            assert exp.handoff is True
    assert exp.max_turns <= 20
