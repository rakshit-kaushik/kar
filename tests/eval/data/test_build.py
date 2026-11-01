"""data.format.build: scored run dir -> train/valid jsonl + manifest + yield report."""

import hashlib
import json

import pytest
from helpers import VERIFY, VERIFY_LINES, ScriptedAgent, ScriptedCaller, say, scenario, tc, tools

from data.format.build import build_dataset, split_of
from eval.runner import run_scenarios

AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}
BOOK = {
    "customer_id": "C-1001",
    "branch_id": "market_st",
    "service_type": "mortgage_consultation",
    "slot_start": "2026-11-03T09:30",
    "idempotency_key": "k",
}
GOOD = VERIFY + [
    tools(tc("check_availability", AVAIL)),
    say("Nine thirty is open. Book the mortgage consultation at Market Street?"),
    tools(tc("book_appointment", BOOK)),
    say("Booked. Anything else?"),
    say("Thanks Dana, goodbye."),
]
LINES = VERIFY_LINES + ["tuesday nine thirty market street", "yes", ("no that's it", True)]


def scenarios():
    a = scenario(id="gen_ordinary_aa_00")
    b = scenario(id="gen_ordinary_aa_01", caller={"persona": "Terse"})
    c = scenario(
        id="gen_needs_person_aa_00",
        family="needs_person",
        expect={
            "handoff": {"required": True, "reason": "caller_requested"},
            "tool_calls": [{"tool": "request_handoff", "ok": True}],
            "final_state": {"no_other_changes": True},
        },
    )
    return [a, b, c]


def agent_factory():
    return ScriptedAgent(GOOD)  # the needs_person scenario fails its scorers with this script


def caller_factory(scen, seed):
    return ScriptedCaller(LINES)


@pytest.mark.asyncio
async def test_build_filters_failed_episodes_and_writes_dataset(tmp_path):
    run_dir = tmp_path / "run"
    await run_scenarios(
        scenarios(),
        agent_factory,
        caller_factory,
        out_dir=run_dir,
        run_id="g1",
        model_id="teacher",
        parallel=3,
        repeats=2,
    )
    out = tmp_path / "ds"
    stats = build_dataset(run_dir, out, name="smoke", fits=None, valid_frac=0.5, seed=0)
    assert (
        stats["episodes"]["generated"] == 6
        and stats["episodes"]["kept"] == 4
        and stats["episodes"]["scorer_failed"] == 2
    )
    assert stats["targets"]["generated"] == 12 and stats["targets"]["kept"] == 12
    assert (
        stats["targets"]["by_kind"]["tool_call"]["kept"] == 4
        and stats["targets"]["by_kind"]["closing"]["kept"] == 4
    )
    assert stats["targets"]["dropped"] == {"brevity": 0, "length": 0}
    assert stats["targets_per_conversation"] == {"mean": 3.0, "histogram": {"3": 4}}
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (out / "valid.jsonl").read_text().splitlines()]
    assert len(train) + len(valid) == 12 and train and valid
    # split is by scenario, never by example: a scenario's examples all land in one split
    assert not (
        {ex["meta"]["scenario_id"] for ex in train} & {ex["meta"]["scenario_id"] for ex in valid}
    )
    assert all(
        ex["messages"][-1]["role"] == "assistant" and isinstance(ex["messages"][-1]["content"], str)
        for ex in train
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["files"] == {"train": "train.jsonl", "valid": "valid.jsonl"}
    assert (
        m["sha256"]["train.jsonl"] == hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    )
    assert m["name"] == "smoke" and m["n_examples"] == {"train": len(train), "valid": len(valid)}
    assert set(m["provenance"]) >= {
        "generator_git_sha",
        "system_prompt_sha256",
        "tools_sha256",
        "stt_rule_version",
        "scenario_ids",
        "scorer_verdicts",
        "run_dirs",
        "built_at",
        "teacher_model_id",
    }
    assert m["provenance"]["scorer_verdicts"]["gen_needs_person_aa_00-s0"]["all_pass"] is False
    assert m["targets_per_conversation"] == stats["targets_per_conversation"] and m["drops"] == {
        "scorer_failed_episodes": 2,
        "brevity_targets": 0,
        "over_2048_targets": 0,
    }
    report = (out / "yield.md").read_text()
    assert "tool_call" in report and "kept" in report and "smoke" in report


def test_split_is_deterministic_by_scenario_and_family():
    a = split_of("gen_ordinary_aa_00", "ordinary", valid_frac=0.1, seed=0)
    assert a in ("train", "valid") and a == split_of(
        "gen_ordinary_aa_00", "ordinary", valid_frac=0.1, seed=0
    )
    ids = [f"gen_ordinary_aa_{i:02d}" for i in range(100)]
    frac = sum(split_of(i, "ordinary", 0.1, 0) == "valid" for i in ids) / 100
    assert 0.04 <= frac <= 0.2


@pytest.mark.asyncio
async def test_length_filter_is_applied_per_target_and_counted(tmp_path):
    run_dir = tmp_path / "run"
    await run_scenarios(
        scenarios()[:1],
        agent_factory,
        caller_factory,
        out_dir=run_dir,
        run_id="g2",
        model_id="teacher",
        parallel=1,
        repeats=1,
    )
    stats = build_dataset(
        run_dir,
        tmp_path / "ds",
        name="s",
        fits=lambda ex: ex["meta"]["kind"] != "closing",
        valid_frac=0.0,
        seed=0,
    )
    assert stats["targets"]["dropped"]["length"] == 1 and stats["targets"]["kept"] == 2
    assert stats["targets"]["by_kind"]["closing"] == {
        "generated": 1,
        "kept": 0,
        "brevity": 0,
        "length": 1,
    }


def test_generation_cli_parser():
    from data.gen.run import build_parser

    a = build_parser().parse_args(
        ["--n", "50", "--seed", "0", "--out", "/tmp/x", "--parallel", "20"]
    )
    assert (
        (a.n, a.seed, a.parallel, a.name) == (50, 0, 20, None)
        and a.mix is None
        and a.valid_frac == 0.1
    )
    b = build_parser().parse_args(
        ["--n", "10", "--out", "/tmp/y", "--mix", "ordinary=4,correction=1", "--no-length-check"]
    )
    assert b.mix == {"ordinary": 4.0, "correction": 1.0} and b.no_length_check is True


@pytest.mark.asyncio
async def test_build_reports_closing_coverage_and_accepts_several_run_dirs(tmp_path):
    r1, r2 = tmp_path / "run1", tmp_path / "run2"
    await run_scenarios(
        scenarios()[:1],
        agent_factory,
        caller_factory,
        out_dir=r1,
        run_id="g1",
        model_id="teacher",
        parallel=1,
        repeats=1,
    )
    await run_scenarios(
        scenarios()[1:2],
        agent_factory,
        caller_factory,
        out_dir=r2,
        run_id="g2",
        model_id="teacher",
        parallel=1,
        repeats=1,
    )
    stats = build_dataset(
        [r1, r2],
        tmp_path / "ds",
        name="two",
        fits=lambda ex: ex["meta"]["kind"] != "closing",
        valid_frac=0.0,
        seed=0,
    )
    assert stats["episodes"]["generated"] == 2 and stats["closing_coverage"] == 0.0
    m = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert m["closing_coverage"] == 0.0 and m["provenance"]["run_dirs"] == [str(r1), str(r2)]
    assert m["provenance"]["batches"] == {
        "g1": {"generated": 1, "kept": 1},
        "g2": {"generated": 1, "kept": 1},
    }
    stats2 = build_dataset(r1, tmp_path / "ds2", name="one", fits=None, valid_frac=0.0, seed=0)
    assert stats2["closing_coverage"] == 1.0


def test_build_cli_parser():
    from data.format.build import build_parser

    a = build_parser().parse_args(
        ["--runs", "/r1", "/r2", "--out", "/ds", "--name", "v1", "--no-length-check"]
    )
    assert (
        a.runs == ["/r1", "/r2"] and a.out == "/ds" and a.name == "v1" and a.no_length_check is True
    )
    assert a.valid_frac == 0.1 and a.seed == 0
