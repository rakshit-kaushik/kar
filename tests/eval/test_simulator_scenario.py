"""Scenario spec: YAML/JSON -> validated Scenario with contract-shaped defaults."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from eval.simulator.scenario import FAMILIES, Scenario, load_scenario, load_scenarios

MINIMAL = {
    "id": "ordinary-01",
    "family": "ordinary",
    "title": "Dana books a mortgage consultation",
    "today": "2026-11-02",
    "customer_id": "C-1042",
    "known_facts": {"phone": "5550142219", "date_of_birth": "1988-03-12"},
    "persona": {"name": "Dana", "style": "terse"},
    "goal": "Book a mortgage consultation at Market Street on Wednesday the 4th, morning.",
    "expected": {
        "final_state": {
            "appointments": [
                {
                    "customer_id": "C-1042",
                    "branch_id": "market_st",
                    "service_type": "mortgage_consultation",
                    "date": "2026-11-04",
                    "status": "booked",
                }
            ]
        },
        "tool_steps": [
            {"name": "lookup_account"},
            {"name": "check_availability", "args": {"branch_id": "market_st"}},
            {"name": "book_appointment"},
        ],
        "handoff": False,
    },
}


def test_nine_families_from_contract():
    assert FAMILIES == (
        "ordinary",
        "missing_information",
        "correction",
        "ambiguous_branch",
        "tool_timeout",
        "duplicate_request",
        "multi_intent_detour",
        "needs_person",
        "noisy_transcript",
    )


def test_minimal_scenario_parses_with_defaults():
    s = Scenario.model_validate(MINIMAL)
    assert s.today == date(2026, 11, 2)
    assert s.noise_level == 0 and s.seed == 42 and s.now == "09:00"
    assert s.corrections == [] and s.detours == [] and s.injected_failures == []
    assert s.expected.max_turns == 12
    assert s.expected.handoff is False
    assert s.expected.tool_steps[1].args == {"branch_id": "market_st"}
    assert s.fixture.appointments == []


def test_bad_family_rejected():
    with pytest.raises(ValueError):
        Scenario.model_validate({**MINIMAL, "family": "weird"})


def test_noise_level_bounds_and_failure_tool_must_exist():
    with pytest.raises(ValueError):
        Scenario.model_validate({**MINIMAL, "noise_level": 4})
    with pytest.raises(ValueError):
        Scenario.model_validate({**MINIMAL, "injected_failures": [{"tool": "make_coffee"}]})
    s = Scenario.model_validate({**MINIMAL, "injected_failures": [{"tool": "book_appointment"}]})
    assert s.injected_failures[0].nth_call == 1


def test_correction_keeps_from_and_to():
    s = Scenario.model_validate(
        {
            **MINIMAL,
            "corrections": [
                {
                    "when": "after the read-back",
                    "field": "time",
                    "from": "14:00",
                    "to": "14:30",
                    "say": "actually make it two thirty",
                }
            ],
        }
    )
    assert s.corrections[0].from_ == "14:00" and s.corrections[0].to == "14:30"


def test_load_yaml_and_json_directory(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(MINIMAL))
    import json

    (tmp_path / "b.json").write_text(json.dumps({**MINIMAL, "id": "ordinary-02"}))
    (tmp_path / "notes.md").write_text("ignored")
    scen = load_scenarios(tmp_path)
    assert [s.id for s in scen] == ["ordinary-01", "ordinary-02"]
    assert load_scenario(tmp_path / "a.yaml").id == "ordinary-01"


def test_duplicate_ids_rejected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(MINIMAL))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(MINIMAL))
    with pytest.raises(ValueError, match="duplicate"):
        load_scenarios(tmp_path)
