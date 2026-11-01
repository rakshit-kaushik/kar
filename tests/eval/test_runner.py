"""r3-eval runner: orchestration, summary markdown, Postgres writes, CLI defaults."""

import json

import pytest
from helpers import LOOKUP_ARGS, ScriptedAgent, ScriptedCaller, say, scenario, tc, tools

from eval.runner import build_parser, resolve_api_key, run_scenarios, summarize, write_db
from eval.scorers import WEIGHTS, ScorerResult

VERIFY = [
    say("Date of birth?"),
    tools(tc("lookup_account", LOOKUP_ARGS)),
    say("Verified. How can I help?"),
]
LINES = ["five five five zero one four two two one nine", "march twelfth nineteen eighty eight"]


def two_scenarios():
    a = scenario(id="ordinary-01", family="ordinary", expected={"max_turns": 6})
    b = scenario(id="needs-01", family="needs_person", expected={"handoff": True, "max_turns": 6})
    return [a, b]


def agent_factory():
    return ScriptedAgent(VERIFY + [say("Sure, I can help with that."), say("Goodbye.")])


def caller_factory(scen):
    return ScriptedCaller(LINES + ["i need a person", ("bye", True)])


@pytest.mark.asyncio
async def test_run_scenarios_writes_episodes_scores_and_summary(tmp_path):
    results = await run_scenarios(
        two_scenarios(),
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r-test",
        model_id="scripted",
        parallel=2,
        episodes_per_scenario=2,
    )
    assert len(results) == 4
    files = sorted(p.name for p in (tmp_path / "episodes").iterdir())
    assert files == [
        "needs-01-1.json",
        "needs-01-2.json",
        "ordinary-01-1.json",
        "ordinary-01-2.json",
    ]
    data = json.loads((tmp_path / "episodes" / "needs-01-1.json").read_text())
    assert data["run_id"] == "r-test" and data["model_id"] == "scripted"
    assert {s["scorer"] for s in data["scores"]} == set(WEIGHTS)
    rows = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * len(WEIGHTS)
    assert set(rows[0]) == {
        "run_id",
        "model_id",
        "scenario_id",
        "episode_id",
        "family",
        "scorer",
        "pass",
        "detail",
    }
    handoff_rows = [r for r in rows if r["scenario_id"] == "needs-01" and r["scorer"] == "handoff"]
    assert all(r["pass"] is False for r in handoff_rows)  # the scripted agent never hands off
    md = (tmp_path / "summary.md").read_text()
    assert "needs_person" in md and "ordinary" in md and "r-test" in md


def fake_result(scenario_id, family, fails):
    ep = type("E", (), {})()
    ep.scenario_id, ep.family, ep.episode_id = scenario_id, family, f"{scenario_id}-x"
    ep.outcome, ep.turns, ep.usage = (
        "completed",
        [1, 2, 3],
        {"agent": {"prompt_tokens": 10, "completion_tokens": 2, "requests": 1}, "caller": {}},
    )
    ep.latency_ms, ep.n_model_requests, ep.error = [100.0], 1, None
    scores = [ScorerResult(name, name not in fails, {}) for name in WEIGHTS]
    return ep, scores


def test_summary_ranks_failures_by_frequency_times_cost():
    results = [
        fake_result("a", "ordinary", {"brevity"}),
        fake_result("b", "ordinary", {"brevity", "no_unsupported_success_claim"}),
        fake_result("c", "correction", {"brevity", "no_unsupported_success_claim"}),
        fake_result("d", "correction", set()),
    ]
    md = summarize(results, run_id="r1", model_id="m1")
    assert "| ordinary |" in md and "| correction |" in md
    ranked = md.split("## Ranked failures")[1]
    pos_claim = ranked.index("no_unsupported_success_claim")
    pos_brev = ranked.index("brevity")
    assert pos_claim < pos_brev  # 2 x 5 = 10 beats 3 x 1 = 3
    assert "| 2 | 5 | 10 |" in ranked and "| 3 | 1 | 3 |" in ranked
    assert "all scorers pass" in md.lower()
    assert "50%" in md  # correction family: 1 of 2 all-pass


@pytest.mark.db
def test_write_db_inserts_eval_results_sessions_and_ledger(rollback_conn):
    results = [fake_result("a", "ordinary", {"brevity"}), fake_result("b", "needs_person", set())]
    for ep, _ in results:
        ep.started_at, ep.ended_at, ep.final_state = (
            "2026-11-01T10:00:00",
            "2026-11-01T10:01:00",
            {"x": 1},
        )
    counts = write_db(rollback_conn, results, run_id="r-db", model_id="m1")
    assert counts == {"eval_results": 2 * len(WEIGHTS), "sessions": 2, "cost_ledger": 5}
    n = rollback_conn.execute(
        "select count(*) from eval_results where run_id='r-db' and pass=false"
    ).fetchone()[0]
    assert n == 1
    row = rollback_conn.execute(
        "select family, scorer, detail from eval_results where run_id='r-db' and pass=false"
    ).fetchone()
    assert row[0] == "ordinary" and row[1] == "brevity" and row[2] == {}
    sess = rollback_conn.execute(
        "select scenario_id, outcome, state from sessions where run_id='r-db' order by 1"
    ).fetchall()
    assert sess[0] == ("a", "completed", {"x": 1})


def test_cli_defaults_and_api_key_resolution(monkeypatch):
    p = build_parser()
    a = p.parse_args(
        [
            "--base-url",
            "http://127.0.0.1:8001/v1",
            "--model",
            "r3-voice-base",
            "--scenarios",
            "eval/scenarios/test/",
            "--out",
            "/tmp/x",
        ]
    )
    assert (
        a.parallel == 4 and a.episodes_per_scenario == 1 and a.no_db is False and a.max_turns == 20
    )
    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://api.anthropic.com/v1")
    monkeypatch.setenv("HOSTED_LLM_API_KEY", "sk-hosted")
    assert resolve_api_key(None, "https://api.anthropic.com/v1") == "sk-hosted"
    assert resolve_api_key(None, "http://127.0.0.1:8001/v1") == "none"
    assert resolve_api_key("explicit", "https://api.anthropic.com/v1") == "explicit"
