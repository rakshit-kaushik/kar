"""data.gen.scenarios: bank-driven, schema-valid generation scenarios that never reuse a test combination."""

import json
from collections import Counter
from pathlib import Path

from data.gen.scenarios import generate, test_signatures, write_dir
from eval.sandbox import Sandbox
from eval.simulator.scenario import FAMILIES, load_scenarios, parse_scenario, validate_against_seed
from eval.simulator.stt_shape import spell_digits

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


def test_catalog_question_mentions_are_robust_substrings():
    out = generate(60, seed=12, profile="short")
    mentions = [tuple(s["expect"]["must_mention"]) for s in out if s["expect"].get("must_mention")]
    assert mentions and all("nine to five" not in m for m in mentions)
    assert all(all(len(x) <= 14 for x in m) for m in mentions)  # single words or short amounts


def test_double_fault_profile_expects_tool_failure_handoffs():
    out = generate(40, seed=21, profile="double_fault")
    assert len(out) == 40 and all(s["family"] == "tool_timeout" for s in out)
    tools_used = Counter(s["inject"][0]["tool"] for s in out)
    assert set(tools_used) == {
        "book_appointment",
        "check_availability",
        "reschedule_appointment",
        "create_ticket",
    }
    sb = Sandbox()
    for s in out:
        inj = s["inject"]
        assert [i["call_index"] for i in inj] == [1, 2] and len({i["tool"] for i in inj}) == 1
        assert all(i["error"] in ("timeout", "unavailable") for i in inj)
        e = s["expect"]
        assert e["handoff"] == {"required": True, "reason": "tool_failure"}
        assert e["final_state"] == {"no_other_changes": True}  # nothing written
        steps = [(t["tool"], t.get("ok")) for t in e["tool_calls"]]
        assert steps[0] == ("lookup_account", True) and steps[-1] == ("request_handoff", True)
        assert steps.count((inj[0]["tool"], False)) == 2
        assert not validate_against_seed(parse_scenario(s, s["id"]), sb)
    assert any(i["error"] == "unavailable" for s in out for i in s["inject"])


def test_double_fault_short_profile_uses_cancel_and_reschedule_only():
    out = generate(50, seed=31, profile="double_fault_short")
    assert len(out) == 50
    tools_used = Counter(s["inject"][0]["tool"] for s in out)
    assert set(tools_used) == {"cancel_appointment", "reschedule_appointment"}
    sb = Sandbox()
    for s in out:
        assert s["expect"]["handoff"] == {"required": True, "reason": "tool_failure"}
        assert s["expect"]["max_turns"] <= 10 and "Phrasing seeds" not in s["caller"]["persona"]
        assert [i["call_index"] for i in s["inject"]] == [1, 2]
        assert not s.get("script")
        assert not validate_against_seed(parse_scenario(s, s["id"]), sb)


def test_frontload_decline_profile_has_both_variants():
    out = generate(40, seed=41, profile="frontload_decline")
    assert len(out) == 40 and all(s["family"] == "ordinary" for s in out)
    front = [s for s in out if "in one sentence" in s["goal"]]
    decline = [s for s in out if "never mind" in s["goal"]]
    assert len(front) >= 12 and len(decline) >= 12 and len(front) + len(decline) == 40
    sb = Sandbox()
    for s in front:
        tools = [t["tool"] for t in s["expect"]["tool_calls"]]
        assert tools == ["lookup_account", "check_availability", "book_appointment"]
        assert s["expect"]["final_state"]["appointments_new"] and s["expect"]["max_turns"] <= 10
    for s in decline:
        assert "book_appointment" in s["expect"]["forbidden_tools"]
        assert s["expect"]["final_state"] == {"no_other_changes": True}
        assert [t["tool"] for t in s["expect"]["tool_calls"]] == [
            "lookup_account",
            "check_availability",
        ]
        assert s["script"] and "never mind" in s["script"][0]["do"]
    assert all(not validate_against_seed(parse_scenario(d, d["id"]), sb) for d in out)


def test_v51_profile_covers_the_five_data_gaps():
    out = generate(60, seed=51, profile="v51")
    assert len(out) == 60
    sb = Sandbox()
    assert all(not validate_against_seed(parse_scenario(d, d["id"]), sb) for d in out)
    by = {
        k: [s for s in out if s["id"].startswith(f"gen_{k}_")]
        for k in (
            "misstate_retry",
            "relative_booking",
            "two_intents_confirm",
            "self_correct_inline",
            "card_already_blocked",
        )
    }
    assert all(len(v) >= 5 for v in by.values()), {k: len(v) for k, v in by.items()}
    for s in by["misstate_retry"]:
        steps = s["expect"]["tool_calls"]
        assert steps[0]["tool"] == "lookup_account" and steps[0]["ok"] is False
        assert steps[1]["tool"] == "lookup_account" and steps[1]["ok"] is True
        assert steps[0]["args"]["date_of_birth"] != steps[1]["args"]["date_of_birth"]
        assert steps[0]["args"]["phone"] == steps[1]["args"]["phone"]
        # the sandbox really answers not_verified to the mis-stated date of birth
        assert sb.call("lookup_account", steps[0]["args"])["error"] == "not_verified"
        assert s["expect"]["handoff"]["required"] is False and "could not verify" in s["goal"]
        sb.reset()
    for s in by["relative_booking"]:
        goal = s["goal"].lower()
        assert "say the day only as" in goal
        assert "tomorrow" in goal or "this " in goal or "next " in goal or " of november" in goal
        assert [t["tool"] for t in s["expect"]["tool_calls"]] == [
            "lookup_account",
            "check_availability",
            "book_appointment",
        ]
    for s in by["two_intents_confirm"]:
        tools = [t["tool"] for t in s["expect"]["tool_calls"]]
        assert (
            tools[:3] == ["lookup_account", "check_availability", "book_appointment"]
            and tools[3] == "create_ticket"
        )
        assert (
            s["expect"]["final_state"]["appointments_new"] and s["expect"]["final_state"]["tickets"]
        )
        if s["expect"]["final_state"]["tickets"][0]["category"] == "card_lost_stolen":
            assert tools[-1] == "request_handoff" and s["expect"]["handoff"]["required"] is True
        assert s["script"] and "and also" in s["script"][0]["do"]
    for s in by["self_correct_inline"]:
        assert "no wait" in s["goal"] and s["expect"]["corrections_hold"]
        held = s["expect"]["corrections_hold"][0]
        booked = s["expect"]["final_state"]["appointments_new"][0]
        assert booked[held["field"]] == held["value"]
    for s in by["card_already_blocked"]:
        assert "already blocked" in s["goal"]
        tools = [t["tool"] for t in s["expect"]["tool_calls"]]
        assert tools == ["lookup_account", "create_ticket", "request_handoff"]
        assert s["expect"]["final_state"]["tickets"][0]["category"] == "card_lost_stolen"
        assert s["expect"]["handoff"]["required"] is True


def test_relative_date_phrases_from_the_seed_clock():
    from datetime import date

    from data.gen.scenarios import relative_date

    today = date(2026, 11, 2)  # Monday
    assert relative_date(date(2026, 11, 3), today) == "tomorrow"
    assert relative_date(date(2026, 11, 5), today) == "this thursday"
    assert relative_date(date(2026, 11, 10), today) == "next tuesday"
    assert (
        relative_date(date(2026, 11, 14), today, prefer_ordinal=True)
        == "the fourteenth of november"
    )


def test_v51_profile_includes_split_phone_number_scenarios():
    out = generate(100, seed=52, profile="v51")
    split = [s for s in out if s["id"].startswith("gen_split_identifier_")]
    assert len(split) >= 5
    sb = Sandbox()
    for s in split:
        steps = s["expect"]["tool_calls"]
        assert steps[0]["tool"] == "lookup_account" and steps[0]["ok"] is True
        phone = steps[0]["args"]["phone"]
        first, rest = s["script"][0]["do"], s["script"][1]["do"]
        # the caller says six digits, stops, and gives the last four only when asked for the rest
        assert spell_digits(phone[:6]) in first and spell_digits(phone[6:]) in rest
        assert "stop" in s["goal"] and s["family"] == "noisy_transcript"
        assert not validate_against_seed(parse_scenario(s, s["id"]), sb)
