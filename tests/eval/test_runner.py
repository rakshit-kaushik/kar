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
    assert counts == {"eval_results": 2 * len(WEIGHTS), "sessions": 2, "cost_ledger": 6}
    ledger = dict(
        rollback_conn.execute("select item, qty from cost_ledger where run_id='r-db'").fetchall()
    )
    assert (
        ledger["episodes"] == 2 and ledger["episodes_passed"] == 1
    )  # the no_jargon failure is non-advisory
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
    assert a.caller_temperature == 0.3 and a.prompt is None and a.retries == 3 and a.backoff == 1.0
    b = p.parse_args(
        [
            "run",
            "--base-url",
            "u",
            "--model",
            "m",
            "--scenarios",
            "d",
            "--out",
            "o",
            "--retries",
            "6",
            "--backoff",
            "5",
        ]
    )
    assert (b.retries, b.backoff) == (6, 5.0)
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


def test_compare_runs_side_by_side(tmp_path):
    from eval.runner import compare_runs

    def write_run(name, fails_by_scenario):
        d = tmp_path / name
        (d / "episodes").mkdir(parents=True)
        rows = []
        for sid, fam, fails in fails_by_scenario:
            for seed in range(2):
                for scorer in WEIGHTS:
                    rows.append(
                        {
                            "run_id": name,
                            "model_id": name,
                            "scenario_id": sid,
                            "episode_id": f"{sid}-s{seed}",
                            "repeat_seed": seed,
                            "family": fam,
                            "scorer": scorer,
                            "pass": scorer not in fails,
                            "advisory": scorer in ADVISORY,
                            "turn_idx": None,
                            "detail": {},
                        }
                    )
        (d / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return d

    a = write_run("hosted", [("o1", "ordinary", set()), ("c1", "correction", set())])
    b = write_run("base", [("o1", "ordinary", {"no_jargon"}), ("c1", "correction", {"brevity"})])
    md = compare_runs([a, b])
    assert "| family | hosted | base |" in md
    assert "| ordinary | 100% | 0% |" in md and "| correction | 100% | 100% |" in md
    assert "| all | 100% | 50% |" in md
    assert "no_jargon" in md.split("## Failures by scorer")[1]
    p = build_parser().parse_args(["compare", str(a), str(b)])
    assert p.command == "compare" and p.runs == [str(a), str(b)]


@pytest.mark.asyncio
async def test_run_scenarios_accepts_explicit_seeds(tmp_path):
    results = await run_scenarios(
        two_scenarios()[:1],
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r",
        model_id="scripted",
        parallel=2,
        seeds=[0, 2],
    )
    assert sorted(ep.repeat_seed for ep, _ in results) == [0, 2]
    assert sorted(p.name for p in (tmp_path / "episodes").iterdir()) == [
        "ordinary_01-s0.json",
        "ordinary_01-s2.json",
    ]
    p = build_parser().parse_args(
        [
            "run",
            "--base-url",
            "u",
            "--model",
            "m",
            "--scenarios",
            "d",
            "--out",
            "o",
            "--seeds",
            "0,2",
        ]
    )
    assert p.seeds == [0, 2] and p.repeats == 1


@pytest.mark.db
@pytest.mark.asyncio
async def test_rescore_with_db_replaces_the_runs_ledger_rows(tmp_path, rollback_conn):
    from eval.runner import rescore_dir

    results = await run_scenarios(
        two_scenarios(),
        agent_factory,
        caller_factory,
        out_dir=tmp_path,
        run_id="r-ledger",
        model_id="scripted",
        parallel=2,
        repeats=1,
    )
    write_db(rollback_conn, results, run_id="r-ledger", model_id="scripted")
    rollback_conn.execute(
        "insert into cost_ledger (run_id, model_id, item, qty, unit, measured) values ('r-ledger', 'scripted', 'episodes', 99, 'episode', true)"
    )
    rescore_dir(tmp_path, scenarios_dir=None, conn=rollback_conn)
    ledger = dict(
        rollback_conn.execute(
            "select item, qty from cost_ledger where run_id='r-ledger'"
        ).fetchall()
    )
    assert ledger["episodes"] == 2 and ledger["episodes_passed"] == 1 and len(ledger) == 6


@pytest.mark.asyncio
async def test_failures_dump_lists_failing_episodes_with_scorer_turn_and_context(tmp_path):
    from eval.runner import failures_report

    bad_agent = lambda: ScriptedAgent(
        VERIFY + [say("Done, you're booked for Tuesday at nine thirty."), say("Goodbye.")]
    )
    await run_scenarios(
        two_scenarios()[:1],
        bad_agent,
        caller_factory,
        out_dir=tmp_path,
        run_id="r",
        model_id="m",
        parallel=1,
        repeats=1,
    )
    md = failures_report(tmp_path)
    assert "## ordinary_01-s0" in md and "no_success_claim_before_result" in md and "turn 3" in md
    assert "C: " in md and "A: Done, you're booked" in md
    assert md.count("## ") == 1  # one failing episode
    good = tmp_path / "good"
    await run_scenarios(
        two_scenarios()[:1],
        agent_factory,
        caller_factory,
        out_dir=good,
        run_id="g",
        model_id="m",
        parallel=1,
        repeats=1,
    )
    assert "no failing episodes" in failures_report(good)
    p = build_parser().parse_args(["failures", str(tmp_path)])
    assert p.command == "failures" and p.directory == str(tmp_path)


def test_bucket_labels_from_scorer_evidence():
    from eval.runner import bucket_for

    assert bucket_for({"scorer": "valid_tool_calls", "detail": {}}, {}, None) == "model capability"
    assert (
        bucket_for(
            {
                "scorer": "tool_calls",
                "detail": {"first_missing": {"tool": "reschedule_appointment"}},
            },
            {},
            {"reschedule_appointment": 6},
        )
        == "missing example type"
    )
    assert (
        bucket_for(
            {"scorer": "tool_calls", "detail": {"first_missing": {"tool": "lookup_account"}}},
            {},
            {"lookup_account": 155},
        )
        == "model capability"
    )
    assert (
        bucket_for(
            {"scorer": "final_state", "detail": {"missing": {"appointments_new": [{}]}}},
            {"tool_calls": True},
            None,
        )
        == "application logic"
    )
    assert (
        bucket_for(
            {"scorer": "final_state", "detail": {"missing": {"appointments_new": [{}]}}},
            {"tool_calls": False},
            None,
        )
        == "missing example type"
    )
    assert (
        bucket_for(
            {"scorer": "handoff", "detail": {"required": True, "requested": False}}, {}, None
        )
        == "missing example type"
    )
    assert bucket_for({"scorer": "no_jargon", "detail": {}}, {}, None) == "missing example type"
    assert bucket_for({"scorer": "max_turns", "detail": {}}, {}, None) == "model capability"
    assert (
        bucket_for({"scorer": "calendar_consistency", "detail": {}}, {}, None) == "model capability"
    )
    assert bucket_for({"scorer": "anything", "detail": {}}, {"outcome": "error"}, None) == "harness"


@pytest.mark.asyncio
async def test_buckets_report_counts_and_verdict(tmp_path):
    from eval.runner import buckets_report

    bad_agent = lambda: ScriptedAgent(
        VERIFY + [say("Done, you're booked for Tuesday at nine thirty."), say("Goodbye.")]
    )
    await run_scenarios(
        two_scenarios(),
        bad_agent,
        caller_factory,
        out_dir=tmp_path,
        run_id="r",
        model_id="m",
        parallel=2,
        repeats=1,
    )
    md = buckets_report(
        tmp_path,
        target_counts={"lookup_account": 155, "check_availability": 569, "request_handoff": 194},
    )
    assert "## Bucket counts" in md and "missing example type" in md and "model capability" in md
    assert "## Per episode" in md and "ordinary_01-s0" in md and "needs_person_01-s0" in md
    assert "Verdict" in md and ("run 2 is justified" in md or "run 2 is not justified" in md)
    p = build_parser().parse_args(["buckets", str(tmp_path), "--dataset", "/tmp/ds"])
    assert p.command == "buckets" and p.directory == str(tmp_path) and p.dataset == "/tmp/ds"


def _ep_for_caller_fault(user_turns, lookups, knows=None, expected_ok=True):
    knows = knows or {"phone": "5550198823", "date_of_birth": "1975-07-04"}
    return {
        "scenario": {
            "caller": {"knows": knows},
            "expect": {"tool_calls": [{"tool": "lookup_account", "ok": expected_ok}]},
        },
        "messages": [{"role": "user", "content": u} for u in user_turns],
        "tool_log": [{"tool": "lookup_account", "args": a, "ok": ok} for a, ok in lookups],
    }


def test_caller_fault_detects_misstated_credentials():
    from eval.runner import caller_fault

    failed = [({"phone": "5550198823", "date_of_birth": "1975-03-04"}, False)]
    phone = "five five five zero one nine eight eight two three"
    wrong_dob = _ep_for_caller_fault([phone, "nineteen seventy five march fourth"], failed)
    assert caller_fault(wrong_dob) == "caller never stated the true date of birth"
    # the caller said it right (fillers between digits are STT noise, not a fault): agent fault
    right = _ep_for_caller_fault(
        [
            "five five five zero one nine like eight eight two three",
            "july fourth nineteen seventy five",
        ],
        failed,
    )
    assert caller_fault(right) is None
    # verification succeeded, or was not expected to: nothing to blame on the caller
    assert caller_fault(_ep_for_caller_fault([phone, "nineteen sixty"], [({}, True)])) is None
    assert (
        caller_fault(_ep_for_caller_fault([phone, "nineteen sixty"], failed, expected_ok=False))
        is None
    )
    # no lookup attempted at all: the agent's problem
    assert caller_fault(_ep_for_caller_fault([phone, "nineteen sixty"], [])) is None
    bad_phone = _ep_for_caller_fault(
        [
            "five five five zero one eight ate eight zero zero one",
            "july fourth nineteen seventy five",
        ],
        [({"phone": "5550188001", "date_of_birth": "1975-07-04"}, False)],
    )
    assert caller_fault(bad_phone) == "caller never stated the true phone number"
    # day as a cardinal, year as "two thousand one"
    knows = {"phone": "5550155678", "date_of_birth": "2001-09-09"}
    ok2 = _ep_for_caller_fault(
        ["five five five zero one five five six seven eight", "september nine two thousand one"],
        [({}, False)],
        knows=knows,
    )
    assert caller_fault(ok2) is None
    garbled = _ep_for_caller_fault(
        ["five five five zero one five five six seven eight", "twenty one september ninth"],
        [({}, False)],
        knows=knows,
    )
    assert caller_fault(garbled) == "caller never stated the true date of birth"


def test_bucket_for_caller_fault_is_harness():
    from eval.runner import bucket_for

    score = {"scorer": "tool_calls", "detail": {"first_missing": {"tool": "check_availability"}}}
    assert (
        bucket_for(
            score,
            {"caller_fault": "caller never stated the true date of birth"},
            {"check_availability": 3},
        )
        == "harness"
    )
    assert (
        bucket_for(score, {"caller_fault": None}, {"check_availability": 3})
        == "missing example type"
    )


def _write_run(d, model, episodes):
    """A minimal run dir: scores.jsonl rows for two scorers plus one episode file per episode."""
    (d / "episodes").mkdir(parents=True)
    with open(d / "scores.jsonl", "w") as f:
        for sid, seed, fails, ep in episodes:
            eid = f"{sid}-s{seed}-abc"
            for scorer in ("tool_calls", "handoff"):
                row = {
                    "run_id": "r",
                    "model_id": model,
                    "scenario_id": sid,
                    "episode_id": eid,
                    "repeat_seed": seed,
                    "family": sid.rsplit("_", 1)[0],
                    "scorer": scorer,
                    "pass": scorer not in fails,
                    "advisory": False,
                }
                f.write(json.dumps(row) + "\n")
            (d / "episodes" / f"{eid}.json").write_text(
                json.dumps({**ep, "episode_id": eid, "scenario_id": sid, "repeat_seed": seed})
            )


def test_compare_can_exclude_caller_fault_episodes_across_runs(tmp_path):
    from eval.runner import compare_runs

    phone = "five five five zero one nine eight eight two three"
    bad = _ep_for_caller_fault([phone, "nineteen sixty"], [({}, False)])
    good = _ep_for_caller_fault([phone, "july fourth nineteen seventy five"], [({}, True)])
    a, b = tmp_path / "a", tmp_path / "b"
    _write_run(
        a,
        "A",
        [
            ("ordinary_01", 0, {"tool_calls"}, bad),
            ("ordinary_01", 1, set(), good),
            ("ordinary_02", 0, set(), good),
        ],
    )
    _write_run(
        b,
        "B",
        [
            ("ordinary_01", 0, set(), good),
            ("ordinary_01", 1, set(), good),
            ("ordinary_02", 0, {"handoff"}, good),
        ],
    )
    assert "| all | 67% | 67% |" in compare_runs([a, b])
    md = compare_runs([a, b], exclude_caller_faults=True)
    # ordinary_01 seed 0 is dropped from both runs because the caller was at fault in run A
    assert "| all | 100% | 50% |" in md
    assert "ordinary_01-s0" in md and "1 scenario" in md
    p = build_parser().parse_args(["compare", str(a), str(b), "--exclude-caller-faults"])
    assert p.command == "compare" and p.exclude_caller_faults is True


def _ep_with_caller_lines(lines, farewell_hangup=False):
    msgs = []
    for u in lines:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": "I've opened your ticket. Goodbye."})
    return {
        "messages": msgs,
        "outcome": "turn_cap",
        "scenario": {"caller": {"knows": {}}, "expect": {}},
    }


def test_caller_stuck_detects_repeated_or_farewell_only_caller_lines():
    from eval.runner import caller_stuck

    assert (
        caller_stuck(_ep_with_caller_lines(["goodbye"] * 4))
        == "caller said goodbye without hanging up"
    )
    assert (
        caller_stuck(_ep_with_caller_lines(["yes please try again"] * 5))
        == "caller repeated the same line 5 times in a row"
    )
    assert (
        caller_stuck(_ep_with_caller_lines(["book a notary", "thursday", "four pm", "yes"])) is None
    )
    # two identical answers in a row are normal conversation, three are a stuck caller
    assert caller_stuck(_ep_with_caller_lines(["marker street", "marker street", "yes"])) is None


def test_bucket_for_max_turns_with_stuck_caller_is_harness():
    from eval.runner import bucket_for

    stuck = {"caller_stuck": "caller said goodbye without hanging up"}
    assert bucket_for({"scorer": "max_turns", "detail": {}}, stuck, None) == "harness"
    # only the turn-cap failure is the caller's doing; other scorers keep their own labels
    assert bucket_for({"scorer": "no_jargon", "detail": {}}, stuck, None) == "missing example type"


def test_agent_repeats_mirrors_the_loop_guard_rule():
    from eval.runner import LOOP_GUARD_HANDOFF_AFTER, agent_repeats

    def ep(*assistant_lines):
        msgs = []
        for line in assistant_lines:
            msgs.append({"role": "assistant", "content": line})
            msgs.append({"role": "user", "content": "yes"})
        return {"messages": msgs}

    # the live guard (AGENT_LOOP_GUARD_HANDOFF_AFTER=3) rephrases the first two consecutive repeats and
    # hands off at the third; a split caller utterance makes one re-ask legitimate
    assert LOOP_GUARD_HANDOFF_AFTER == 3
    assert agent_repeats(
        ep("What day is that?", "What day is that?", "what day is that", "x", "x")
    ) == {
        "rephrase": 3,
        "handoff": 0,
        "max_streak": 3,
    }
    assert agent_repeats(ep("Same.", "Same.", "Same.", "Same.", "Later.")) == {
        "rephrase": 2,
        "handoff": 1,
        "max_streak": 4,
    }
    assert agent_repeats(ep("Which branch?", "Which branch?", "Marker Street, got it.")) == {
        "rephrase": 1,
        "handoff": 0,
        "max_streak": 2,
    }
    assert agent_repeats(ep("One.", "Two.", "Three.")) == {
        "rephrase": 0,
        "handoff": 0,
        "max_streak": 1,
    }
    # tool-call-only turns (empty content) are not lines
    tool_turns = {"messages": [{"role": "assistant", "content": "", "tool_calls": [{}]}] * 3}
    assert agent_repeats(tool_turns) == {"rephrase": 0, "handoff": 0, "max_streak": 0}


def test_agent_repeats_exempts_a_repeated_farewell_like_the_live_guard():
    from eval.runner import agent_repeats

    msgs = []
    for line in [
        "I've booked it for Tuesday. Goodbye.",
        "I've booked it for Tuesday. Goodbye.",
        "Goodbye, take care.",
    ]:
        msgs += [{"role": "assistant", "content": line}, {"role": "user", "content": "goodbye"}]
    assert agent_repeats({"messages": msgs}) == {"rephrase": 0, "handoff": 0, "max_streak": 1}


def test_compare_disambiguates_runs_that_share_a_model_id(tmp_path):
    from eval.runner import compare_runs

    good = _ep_for_caller_fault(
        ["five five five zero one nine eight eight two three", "july fourth nineteen seventy five"],
        [({}, True)],
    )
    a, b = tmp_path / "hosted-caller", tmp_path / "local-caller"
    _write_run(a, "r3-voice-base", [("ordinary_01", 0, set(), good)])
    _write_run(b, "r3-voice-base", [("ordinary_01", 0, {"handoff"}, good)])
    md = compare_runs([a, b])
    assert "| family | r3-voice-base (hosted-caller) | r3-voice-base (local-caller) |" in md


def test_tool_calls_cause_splits_missing_call_from_wrong_args_and_outcome():
    from eval.runner import tool_calls_cause

    log = [
        {
            "idx": 0,
            "tool": "lookup_account",
            "args": {"phone": "5550142219", "date_of_birth": "1988-03-12"},
            "ok": True,
        },
        {
            "idx": 1,
            "tool": "book_appointment",
            "args": {
                "branch_id": "mill_rd",
                "service_type": "notary",
                "slot_start": "2026-11-05T16:00",
            },
            "ok": True,
        },
    ]
    missing = {
        "matched": [0],
        "first_missing": {
            "step": 1,
            "tool": "check_availability",
            "args": {"branch_id": "market_st"},
            "ok": True,
        },
    }
    assert tool_calls_cause(missing, log) == {
        "cause": "missing call",
        "tool": "check_availability",
        "fields": [],
    }
    wrong = {
        "matched": [0],
        "first_missing": {
            "step": 2,
            "tool": "book_appointment",
            "args": {
                "branch_id": "market_st",
                "service_type": "notary",
                "date": "2026-11-05",
                "time": "16:00",
            },
            "ok": True,
        },
    }
    assert tool_calls_cause(wrong, log) == {
        "cause": "wrong args",
        "tool": "book_appointment",
        "fields": ["branch_id"],
    }
    failed_log = [
        {
            "idx": 0,
            "tool": "lookup_account",
            "args": {"phone": "5550142219", "date_of_birth": "1988-03-12"},
            "ok": False,
            "error": "timeout",
        }
    ]
    outcome = {
        "matched": [],
        "first_missing": {
            "step": 0,
            "tool": "lookup_account",
            "args": {"phone": "5550142219", "date_of_birth": "1988-03-12"},
            "ok": True,
        },
    }
    assert tool_calls_cause(outcome, failed_log) == {
        "cause": "wrong outcome",
        "tool": "lookup_account",
        "fields": [],
    }
    assert tool_calls_cause({"matched": [], "first_missing": None}, log) is None


def test_failures_report_summarizes_tool_calls_causes(tmp_path):
    from eval.runner import failures_report

    (tmp_path / "episodes").mkdir()
    ep = {
        "episode_id": "ordinary_01-s0-abc",
        "family": "ordinary",
        "outcome": "completed",
        "scenario": {"goal": "book", "caller": {"knows": {}}, "expect": {}},
        "messages": [],
        "turns": [],
        "tool_log": [{"idx": 0, "tool": "lookup_account", "args": {}, "ok": True}],
        "scores": [
            {
                "scorer": "tool_calls",
                "passed": False,
                "advisory": False,
                "turn_idx": 2,
                "detail": {
                    "matched": [0],
                    "first_missing": {
                        "step": 1,
                        "tool": "check_availability",
                        "args": {},
                        "ok": True,
                    },
                },
            }
        ],
    }
    (tmp_path / "episodes" / "ordinary_01-s0-abc.json").write_text(json.dumps(ep))
    md = failures_report(tmp_path)
    assert "**tool_calls failures by cause**" in md
    assert "| missing call | check_availability | 1 |" in md
    assert "cause: missing call check_availability" in md


def test_caller_fault_judges_the_raw_reply_not_the_shaped_noise():
    from eval.runner import caller_fault

    # the simulator's STT shaping turned "eight zero" into "ate oh": intended noise, not the caller's fault
    ep = _ep_for_caller_fault(
        ["five five five zero one nine eight eight two three", "july fourth nineteen seventy five"],
        [({"phone": "5550198823", "date_of_birth": "1975-03-04"}, False)],
    )
    ep["messages"][0]["content"] = "five five five zero one nine ate ate two three"
    ep["turns"] = [
        {"caller_text_raw": "five five five zero one nine eight eight two three"},
        {"caller_text_raw": "july fourth nineteen seventy five"},
    ]
    assert caller_fault(ep) is None
    # the caller itself got the digits wrong: fault, whatever the shaping did
    ep["turns"][0]["caller_text_raw"] = "five five five zero one nine eight eight two"
    assert caller_fault(ep) == "caller never stated the true phone number"


def test_caller_fault_when_the_caller_never_gives_a_phone_number_at_all():
    from eval.runner import caller_fault

    # the greeting always asks for the phone; a caller who derails and never states it sank the call
    ep = _ep_for_caller_fault(["i just want to check my balance", "ok bye"], [])
    assert caller_fault(ep) == "caller never stated the true phone number"
    # but a missing date of birth with no lookup attempted may be the agent never asking: not the caller's fault
    ep = _ep_for_caller_fault(["five five five zero one nine eight eight two three", "hello"], [])
    assert caller_fault(ep) is None


def test_missing_call_is_bad_data_when_the_tool_was_common_in_prefixes_but_rare_as_a_target():
    from eval.runner import bucket_for

    score = {"scorer": "tool_calls", "detail": {"first_missing": {"tool": "check_availability"}}}
    # v2: the teacher called check_availability in 500 prefixes but it was the trained target 3 times
    assert (
        bucket_for(score, {}, {"check_availability": 3}, {"check_availability": 500}) == "bad data"
    )
    assert (
        bucket_for(score, {}, {"check_availability": 3}, {"check_availability": 4})
        == "missing example type"
    )
    assert bucket_for(score, {}, {"check_availability": 3}, None) == "missing example type"


def test_prefix_counts_from_dataset(tmp_path):
    from eval.runner import prefix_counts_from_dataset

    call = {"function": {"name": "check_availability", "arguments": "{}"}}
    ex = {
        "messages": [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "content": "{}"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "book_appointment", "arguments": "{}"}}],
            },
        ]
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(ex) + "\n" + json.dumps(ex) + "\n")
    assert prefix_counts_from_dataset(tmp_path) == {"check_availability": 2}


def test_one_strike_handoff_detects_a_transfer_after_a_single_not_verified():
    from eval.runner import one_strike_handoff

    def ep(lookups, handoff=True, line=""):
        log = [
            {"idx": i, "tool": "lookup_account", "args": {}, "ok": False, "error": "not_verified"}
            for i in range(lookups)
        ]
        if handoff:
            log.append(
                {
                    "idx": len(log),
                    "tool": "request_handoff",
                    "args": {"reason": "verification_failed"},
                    "ok": True,
                }
            )
        return {"tool_log": log, "messages": [{"role": "assistant", "content": line}]}

    assert one_strike_handoff(ep(1)) is True
    assert (
        one_strike_handoff(ep(1, handoff=False, line="I'm transferring you to a person now."))
        is True
    )
    assert one_strike_handoff(ep(2)) is False
    assert (
        one_strike_handoff(ep(1, handoff=False, line="Could you repeat your date of birth?"))
        is False
    )


def test_buckets_report_counts_one_strike_handoffs_as_a_data_gap(tmp_path):
    from eval.runner import buckets_report

    (tmp_path / "episodes").mkdir()
    ep = _ep_for_caller_fault(
        ["five five five zero one nine eight eight two three", "nineteen sixty"],
        [({"phone": "5550198823", "date_of_birth": "1960-01-01"}, False)],
    )
    ep["tool_log"][0].update(idx=0, error="not_verified")
    ep["tool_log"].append(
        {"idx": 1, "tool": "request_handoff", "args": {"reason": "verification_failed"}, "ok": True}
    )
    ep.update(
        episode_id="ordinary_01-s0-abc",
        family="ordinary",
        outcome="handoff",
        turns=[],
        scores=[
            {
                "scorer": "tool_calls",
                "passed": False,
                "advisory": False,
                "turn_idx": 0,
                "detail": {"first_missing": {"tool": "lookup_account"}},
            },
            {
                "scorer": "final_state",
                "passed": False,
                "advisory": False,
                "turn_idx": None,
                "detail": {},
            },
        ],
    )
    (tmp_path / "episodes" / "ordinary_01-s0-abc.json").write_text(json.dumps(ep))
    md = buckets_report(tmp_path, target_counts={"lookup_account": 700})
    assert "One-strike handoffs" in md and "ordinary_01-s0-abc" in md
    assert "| ordinary_01-s0-abc | ordinary | missing example type |" in md
    assert "| missing example type | 1 |" in md and "| harness | 0 |" in md


def test_failures_report_splits_claims_by_action(tmp_path):
    from eval.runner import failures_report

    (tmp_path / "episodes").mkdir()
    ep = {
        "episode_id": "correction_03-s0-abc",
        "family": "correction",
        "outcome": "completed",
        "scenario": {"goal": "x", "caller": {"knows": {}}, "expect": {}},
        "messages": [],
        "turns": [],
        "tool_log": [],
        "scores": [
            {
                "scorer": "no_success_claim_before_result",
                "passed": False,
                "advisory": False,
                "turn_idx": 4,
                "detail": {
                    "violations": [
                        {"idx": 13, "turn": 4, "action": "handoff", "text": "I'm transferring you"},
                        {"idx": 15, "turn": 5, "action": "book", "text": "I booked it"},
                    ]
                },
            }
        ],
    }
    (tmp_path / "episodes" / "correction_03-s0-abc.json").write_text(json.dumps(ep))
    md = failures_report(tmp_path)
    assert "**success claims by claimed action**: transfer 1, write 1" in md


def test_one_strike_handoff_uses_the_live_claim_guard_phrase_set():
    from eval.runner import one_strike_handoff

    def ep(line):
        return {
            "tool_log": [
                {
                    "idx": 0,
                    "tool": "lookup_account",
                    "args": {},
                    "ok": False,
                    "error": "not_verified",
                }
            ],
            "messages": [{"role": "assistant", "content": line}],
        }

    assert one_strike_handoff(ep("A specialist will take over from here.")) is True
    assert one_strike_handoff(ep("Let me put you through to a colleague.")) is True
    assert one_strike_handoff(ep("I'll hand you over to a banker now.")) is True
    assert one_strike_handoff(ep("Could you read me your date of birth again?")) is False
