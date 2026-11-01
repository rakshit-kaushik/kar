"""r3-eval runner: repeats with seeds, summary with mean and spread, Postgres writes, CLI subcommands."""

import json
from pathlib import Path

import pytest
from helpers import (
    VERIFY,
    VERIFY_LINES,
    ScriptedAgent,
    ScriptedCaller,
    say,
    scenario,
    tc,
    tools,
)

from eval.runner import (
    build_parser,
    resolve_api_key,
    run_scenarios,
    summarize,
    validate_dir,
    write_db,
)
from eval.scorers import ADVISORY, WEIGHTS, ScorerResult

TEST_DIR = Path(__file__).resolve().parents[2] / "eval/scenarios/test"
AVAIL = {"branch_id": "market_st", "service_type": "mortgage_consultation", "date": "2026-11-03"}
BOOK = {
    "customer_id": "C-1001",
    "branch_id": "market_st",
    "service_type": "mortgage_consultation",
    "slot_start": "2026-11-03T09:30",
    "idempotency_key": "k",
}


def two_scenarios():
    a = scenario()
    b = scenario(
        id="needs_person_01",
        family="needs_person",
        expect={
            "handoff": {"required": True, "reason": "caller_requested"},
            "tool_calls": [{"tool": "request_handoff", "ok": True}],
            "final_state": {"no_other_changes": True},
        },
    )
    return [a, b]


def agent_factory():
    return ScriptedAgent(
        VERIFY
        + [
            tools(tc("check_availability", AVAIL)),
            say("Nine thirty is open, book it?"),
            tools(tc("book_appointment", BOOK)),
            say("Booked. Bye."),
        ]
    )


def caller_factory(scen, seed):
    return ScriptedCaller(VERIFY_LINES + ["tuesday nine thirty", "yes", ("bye", True)])


@pytest.mark.asyncio
async def test_run_scenarios_repeats_with_seeds_and_writes_everything(tmp_path):
    results = await run_scenarios(
        two_scenarios(),
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r-test",
        model_id="scripted",
        parallel=3,
        repeats=3,
    )
    assert len(results) == 6 and sorted({ep.repeat_seed for ep, _ in results}) == [0, 1, 2]
    files = sorted(p.name for p in (tmp_path / "episodes").iterdir())
    assert files == [
        f"{sid}-s{k}.json" for sid in ("needs_person_01", "ordinary_01") for k in range(3)
    ]
    data = json.loads((tmp_path / "episodes" / "ordinary_01-s2.json").read_text())
    assert data["repeat_seed"] == 2 and {s["scorer"] for s in data["scores"]} == set(WEIGHTS)
    rows = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * len(WEIGHTS)
    assert set(rows[0]) == {
        "run_id",
        "model_id",
        "scenario_id",
        "episode_id",
        "repeat_seed",
        "family",
        "scorer",
        "pass",
        "advisory",
        "turn_idx",
        "detail",
    }
    assert all(
        r["pass"] is False
        for r in rows
        if r["scenario_id"] == "needs_person_01" and r["scorer"] == "handoff"
    )
    md = (tmp_path / "summary.md").read_text()
    assert "needs_person" in md and "ordinary" in md and "r-test" in md and "seeds 0, 1, 2" in md


def fake_result(scenario_id, family, fails, seed=0):
    ep = type("E", (), {})()
    ep.scenario_id, ep.family, ep.episode_id, ep.repeat_seed = (
        scenario_id,
        family,
        f"{scenario_id}-s{seed}",
        seed,
    )
    ep.outcome, ep.turns, ep.error = "completed", [1, 2, 3], None
    ep.usage = {
        "agent": {"prompt_tokens": 10, "completion_tokens": 2, "requests": 1},
        "caller": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "requests": 1,
            "temperature_applied": False,
        },
    }
    ep.latency_ms, ep.n_model_requests, ep.n_truncated = [100.0], 1, 0
    ep.prompt_source, ep.first_prompt_tokens = "eval/prompts/system_prompt.md", 3000
    ep.started_at, ep.ended_at, ep.final_state = (
        "2026-11-01T10:00:00",
        "2026-11-01T10:01:00",
        {"x": 1},
    )
    scores = [
        ScorerResult(name, name not in fails, {"applicable": True}, advisory=name in ADVISORY)
        for name in WEIGHTS
    ]
    return ep, scores


def test_summary_mean_spread_and_ranked_failures():
    results = [
        fake_result("a", "ordinary", set(), 0),
        fake_result("a", "ordinary", {"brevity"}, 1),
        fake_result("a", "ordinary", {"no_success_claim_before_result", "brevity"}, 2),
        fake_result("b", "correction", {"no_success_claim_before_result"}, 0),
        fake_result("b", "correction", set(), 1),
        fake_result("b", "correction", set(), 2),
    ]
    md = summarize(results, run_id="r1", model_id="m1")
    fam = md.split("## Per family")[1].split("## Ranked")[0]
    assert (
        "| ordinary | 3 | 67% (0–100) |" in fam
    )  # all-pass per seed: 1,1,0 -> mean 67, spread 0–100
    assert "| correction | 3 | 67% (0–100) |" in fam
    ranked = md.split("## Ranked failures")[1]
    assert ranked.index("no_success_claim_before_result") < ranked.index("brevity")
    assert "| 2 | 5 | 10 |" in ranked and "| 2 | 0 | 0 |" in ranked
    assert "brevity (advisory)" in md
    assert "caller temperature" in md.lower() and "not applied" in md.lower()


@pytest.mark.db
def test_write_db_inserts_eval_results_sessions_and_ledger(rollback_conn):
    results = [
        fake_result("a", "ordinary", {"no_jargon"}),
        fake_result("b", "needs_person", set(), 1),
    ]
    counts = write_db(rollback_conn, results, run_id="r-db", model_id="m1")
    assert counts == {"eval_results": 2 * len(WEIGHTS), "sessions": 2, "cost_ledger": 5}
    row = rollback_conn.execute(
        "select family, scorer, detail from eval_results where run_id='r-db' and pass=false"
    ).fetchone()
    assert row[0] == "ordinary" and row[1] == "no_jargon"
    assert (
        row[2]["repeat_seed"] == 0 and row[2]["advisory"] is False and row[2]["applicable"] is True
    )
    sess = rollback_conn.execute(
        "select call_id, scenario_id, outcome, state from sessions where run_id='r-db' order by 1"
    ).fetchall()
    assert sess[0] == ("a-s0", "a", "completed", {"x": 1})


def test_validate_dir_reports_problems_per_file(tmp_path):
    assert validate_dir(TEST_DIR) == {}
    bad = json.loads((TEST_DIR / "ordinary_01.json").read_text())
    bad["caller"]["customer_id"] = "C-9999"
    (tmp_path / "ordinary_01.json").write_text(json.dumps(bad))
    (tmp_path / "broken.json").write_text(json.dumps({"id": "x"}))
    problems = validate_dir(tmp_path)
    assert set(problems) == {"ordinary_01.json", "broken.json"}
    assert any("C-9999" in p for p in problems["ordinary_01.json"])


def test_cli_subcommands_and_api_key_resolution(monkeypatch):
    p = build_parser()
    a = p.parse_args(
        [
            "run",
            "--base-url",
            "http://127.0.0.1:8001/v1",
            "--model",
            "r3-voice-base",
            "--scenarios",
            "eval/scenarios/test/",
            "--repeats",
            "3",
            "--out",
            "/tmp/x",
        ]
    )
    assert a.command == "run" and a.repeats == 3 and a.parallel == 4 and a.no_db is False
    assert a.caller_temperature == 0.3 and a.prompt is None
    v = p.parse_args(["validate", "eval/scenarios/test"])
    assert v.command == "validate" and v.directory == "eval/scenarios/test"
    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("HOSTED_LLM_API_KEY", "sk-hosted")
    assert resolve_api_key(None, "https://api.anthropic.com/v1") == "sk-hosted"
    assert resolve_api_key(None, "http://127.0.0.1:8001/v1") == "none"
    assert resolve_api_key("explicit", "https://api.anthropic.com/v1") == "explicit"


def test_summary_has_a_per_scenario_matrix():
    results = [
        fake_result("a", "ordinary", set(), 0),
        fake_result("a", "ordinary", {"handoff"}, 1),
        fake_result("a", "ordinary", set(), 2),
    ]
    md = summarize(results, run_id="r1", model_id="m1")
    sec = md.split("## Per scenario")[1].split("## Ranked")[0]
    assert "| a | ordinary | 2/3 | handoff (s1 t-) |" in sec


@pytest.mark.asyncio
async def test_rescore_rebuilds_scores_and_summary_from_saved_episodes(tmp_path):
    from eval.runner import rescore_dir

    results = await run_scenarios(
        two_scenarios(),
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r-test",
        model_id="scripted",
        parallel=2,
        repeats=2,
    )
    (tmp_path / "summary.md").write_text("stale")
    (tmp_path / "scores.jsonl").write_text("")
    n = rescore_dir(tmp_path, scenarios_dir=None)
    assert n == len(results) == 4
    rows = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * len(WEIGHTS)
    assert "## Per family" in (tmp_path / "summary.md").read_text()
    p = build_parser().parse_args(["rescore", str(tmp_path)])
    assert p.command == "rescore" and p.directory == str(tmp_path) and p.db is False
    assert build_parser().parse_args(["rescore", str(tmp_path), "--db"]).db is True


@pytest.mark.db
@pytest.mark.asyncio
async def test_rescore_with_db_replaces_the_runs_eval_results(tmp_path, rollback_conn):
    from eval.runner import rescore_dir

    results = await run_scenarios(
        two_scenarios(),
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r-rescore",
        model_id="scripted",
        parallel=2,
        repeats=1,
    )
    write_db(rollback_conn, results, run_id="r-rescore", model_id="scripted")
    rollback_conn.execute("update eval_results set pass = not pass where run_id='r-rescore'")
    n = rescore_dir(tmp_path, scenarios_dir=None, conn=rollback_conn)
    assert n == 2
    rows = rollback_conn.execute(
        "select scenario_id, scorer, pass from eval_results where run_id='r-rescore' and scorer='handoff' order by 1"
    ).fetchall()
    assert rows == [("needs_person_01", "handoff", False), ("ordinary_01", "handoff", True)]
    assert rollback_conn.execute(
        "select count(*) from eval_results where run_id='r-rescore'"
    ).fetchone()[0] == 2 * len(WEIGHTS)
