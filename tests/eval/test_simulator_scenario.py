"""Scenario spec = eval/scenarios/schema.json (human-owned). Loading, validation, seed consistency."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from eval.sandbox import TOOL_NAMES, Sandbox
from eval.simulator.scenario import (
    FAMILIES,
    ScenarioError,
    load_scenario,
    load_scenarios,
    validate_against_seed,
)

TEST_DIR = Path(__file__).resolve().parents[2] / "eval/scenarios/test"
BASE = json.loads((TEST_DIR / "ordinary_01.json").read_text())


def test_all_27_test_scenarios_load_three_per_family():
    scen = load_scenarios(TEST_DIR)
    assert len(scen) == 27
    assert Counter(s.family for s in scen) == {f: 3 for f in FAMILIES}
    assert FAMILIES == (
        "ordinary",
        "missing_info",
        "correction",
        "ambiguous_branch",
        "tool_timeout",
        "duplicate",
        "multi_intent",
        "needs_person",
        "noisy_transcript",
    )
    s = next(x for x in scen if x.id == "correction_01")
    assert s.caller.noise == "none" and s.caller.knows["phone"] == "5550111123"
    assert s.script[0].trigger.startswith("agent reads back")
    assert s.expect.corrections_hold[0].field == "slot_start"
    assert s.expect.handoff.required is False and s.expect.max_turns == 14
    assert s.expect.final_state.no_other_changes is True
    assert (
        s.expect.tool_calls[2].ok is True
        and s.expect.tool_calls[2].args["slot_start"] == "2026-11-05T14:30"
    )
    assert s.source.endswith("correction_01.json")


def test_defaults_for_optional_keys():
    s = load_scenario(TEST_DIR / "ordinary_01.json")
    assert s.script == [] and s.inject == [] and s.expect.forbidden_tools == []
    assert s.expect.max_successful_calls == {} and s.expect.must_mention == []
    assert s.expect.must_not_mention_before_verified is True
    assert s.caller.withholds == [] and s.caller.stt_quirks == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d.__setitem__("family", "weird"), "family"),
        (lambda d: d.__setitem__("id", "Ordinary-1"), "id"),
        (lambda d: d.__setitem__("extra", 1), "extra"),
        (lambda d: d["expect"].pop("max_turns"), "max_turns"),
        (lambda d: d["caller"].__setitem__("noise", "medium"), "noise"),
        (
            lambda d: d.__setitem__(
                "inject", [{"tool": "book_appointment", "call_index": 0, "error": "timeout"}]
            ),
            "call_index",
        ),
        (lambda d: d["expect"].__setitem__("max_turns", 25), "max_turns"),
    ],
)
def test_schema_violations_are_rejected_with_the_path(tmp_path, mutate, needle):
    d = json.loads(json.dumps(BASE))
    mutate(d)
    p = tmp_path / "ordinary_01.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ScenarioError) as e:
        load_scenario(p)
    assert needle in str(e.value)


def test_yaml_is_accepted_and_duplicate_ids_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(BASE))
    (tmp_path / "b.json").write_text(json.dumps(BASE))
    with pytest.raises(ValueError, match="duplicate"):
        load_scenarios(tmp_path)
    (tmp_path / "b.json").unlink()
    assert load_scenarios(tmp_path)[0].id == "ordinary_01"


BLOCK_CARD_SCENARIOS = {
    "correction_03",
    "duplicate_02",
    "missing_info_02",
    "multi_intent_02",
    "needs_person_01",
    "needs_person_02",
    "needs_person_03",
    "tool_timeout_03",
}  # await the human's rewrite


def test_every_test_scenario_is_consistent_with_the_seed():
    sb = Sandbox()
    problems = {s.id: validate_against_seed(s, sb) for s in load_scenarios(TEST_DIR)}
    bad = {k: v for k, v in problems.items() if v}
    if "block_card" in TOOL_NAMES:
        assert bad == {}
    else:
        assert set(bad) == BLOCK_CARD_SCENARIOS
        assert all("block_card" in " ".join(v) for v in bad.values())


def _with(**changes):
    d = json.loads(json.dumps(BASE))
    for path, value in changes.items():
        cur = d
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = value
    return d


def test_seed_consistency_catches_bad_customer_phone_slot_and_tool(tmp_path):
    sb = Sandbox()
    cases = {
        "customer": _with(**{"caller.customer_id": "C-9999"}),
        "phone": _with(**{"caller.knows": {**BASE["caller"]["knows"], "phone": "5550000000"}}),
        "slot": _with(
            **{
                "expect.final_state": {
                    "appointments_new": [
                        {
                            "customer_id": "C-1001",
                            "branch_id": "marlow_st",
                            "service_type": "notary",
                            "slot_start": "2026-11-05T14:00",
                        }
                    ],
                    "no_other_changes": True,
                }
            }
        ),
        "tool": _with(**{"expect.forbidden_tools": ["make_coffee"]}),
        "appointment": _with(
            **{
                "expect.final_state": {
                    "appointments_changed": [{"appointment_id": "A-9999", "status": "cancelled"}],
                    "no_other_changes": True,
                }
            }
        ),
        "card": _with(
            **{
                "expect.final_state": {
                    "cards": [{"card_last4": "0000", "status": "blocked"}],
                    "no_other_changes": True,
                }
            }
        ),
    }
    for name, data in cases.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(data))
        problems = validate_against_seed(load_scenario(p), sb)
        assert problems, name
        assert name.split("_")[0] in " ".join(problems).lower() or name == "slot", (name, problems)
