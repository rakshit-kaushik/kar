"""data.gen.scenarios: bank-driven, schema-valid generation scenarios that never reuse a test combination."""

import json
from collections import Counter
from pathlib import Path

from data.gen.scenarios import generate, test_signatures, write_dir
from eval.sandbox import Sandbox
from eval.simulator.scenario import FAMILIES, load_scenarios, parse_scenario, validate_against_seed

TEST_DIR = Path(__file__).resolve().parents[3] / "eval/scenarios/test"


def test_generate_is_deterministic_schema_valid_and_seed_consistent():
    a = generate(60, seed=1)
    b = generate(60, seed=1)
    assert [s["id"] for s in a] == [s["id"] for s in b] and [s["goal"] for s in a] == [
        s["goal"] for s in b
    ]
    assert len({s["id"] for s in a}) == 60
    sb = Sandbox()
    problems = {}
    for d in a:
        s = parse_scenario(d, d["id"])  # schema + pydantic
        p = validate_against_seed(s, sb)
        if p:
            problems[s.id] = p
    assert problems == {}
    fams = Counter(s["family"] for s in a)
    assert set(fams) == set(FAMILIES) and min(fams.values()) >= 3
    assert all(s["id"].startswith("gen_") for s in a)
    assert all(s["expect"]["final_state"].get("no_other_changes") is True for s in a)


def test_generation_never_reuses_a_test_signature():
    sigs = test_signatures(TEST_DIR)
    assert sigs and any(len(x) == 4 for x in sigs)
    for s in generate(120, seed=3):
        for a in s["expect"]["final_state"].get("appointments_new", []):
            assert (
                a["customer_id"],
                a["branch_id"],
                a["service_type"],
                a["slot_start"][:10],
            ) not in sigs
        for t in s["expect"]["final_state"].get("tickets", []):
            assert ("ticket", s["caller"]["customer_id"], t["category"]) not in sigs
        for c in s["expect"]["final_state"].get("appointments_changed", []):
            assert ("changed", c["appointment_id"]) not in sigs


def test_mix_weights_and_noise_distribution():
    mix = {"ordinary": 4, "correction": 1}
    out = generate(50, seed=5, mix=mix)
    fams = Counter(s["family"] for s in out)
    assert set(fams) == {"ordinary", "correction"} and fams["ordinary"] > fams["correction"]
    noise = Counter(s["caller"]["noise"] for s in generate(90, seed=7))
    assert set(noise) == {"none", "low", "high"} and noise["high"] >= 9
    for s in out:
        assert s["caller"]["knows"]["phone_spoken"].count(" ") == 9  # ten spoken digits
        assert s["caller"]["persona"]


def test_write_dir_round_trips_through_the_loader(tmp_path):
    out = generate(12, seed=2)
    paths = write_dir(out, tmp_path)
    assert len(paths) == 12 and all(p.suffix == ".json" for p in paths)
    loaded = load_scenarios(tmp_path)
    assert [s.id for s in loaded] == sorted(s["id"] for s in out)
    assert json.loads(paths[0].read_text())["id"] == out[0]["id"]


def test_short_profile_yields_short_call_scenarios_only():
    out = generate(40, seed=11, profile="short")
    fams = Counter(s["family"] for s in out)
    assert set(fams) <= {"ordinary", "needs_person", "duplicate"} and fams["ordinary"] >= 15
    for s in out:
        tools = [t["tool"] for t in s["expect"]["tool_calls"]]
        assert "book_appointment" not in tools and "reschedule_appointment" not in tools
        assert s["expect"]["max_turns"] <= 12
    sb = Sandbox()
    assert all(not validate_against_seed(parse_scenario(d, d["id"]), sb) for d in out)


def test_phrasing_bank_is_clean_and_seeds_personas():
    from data.gen.banks import load_bank

    bank = load_bank()
    assert len(bank) >= 200
    for u in bank:
        assert 3 <= len(u.split()) <= 18 and "\n" not in u
        assert not __import__("re").search(r"\d{7,}", u)  # no phone-like numbers from the corpora
    assert len(set(bank)) == len(bank)
    out = generate(10, seed=4)
    assert all("Phrasing seeds:" in s["caller"]["persona"] for s in out)
    assert all(len(s["caller"]["persona"]) < 400 for s in out)
    plain = generate(5, seed=4, phrasing=False)
    assert all("Phrasing seeds:" not in s["caller"]["persona"] for s in plain)
