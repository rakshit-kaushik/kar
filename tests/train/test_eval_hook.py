"""Task 5: given a model dir, start the server, wait for health, call r3-eval (stubbed until session E's
runner is merged), record the per-family table; plus the precision gate comparison."""
import json
import socket
import sys
from pathlib import Path

import pytest

from train.eval_hook import (
    collect_results,
    evaluate_model_dir,
    family_table,
    format_family_table,
    precision_gate,
    run_r3_eval,
    server_command,
    wait_for_health,
)

HERE = Path(__file__).resolve().parent
ROWS = [
    {"scenario_id": "ordinary-1", "family": "ordinary", "scorer": "final_state", "pass": True},
    {"scenario_id": "ordinary-1", "family": "ordinary", "scorer": "no_jargon", "pass": True},
    {"scenario_id": "ordinary-2", "family": "ordinary", "scorer": "final_state", "pass": False},
    {"scenario_id": "correction-1", "family": "correction", "scorer": "final_state", "pass": True},
]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _scenarios(tmp_path):
    d = tmp_path / "scenarios"
    d.mkdir()
    (d / "ordinary__booking.json").write_text(json.dumps({"id": "ordinary__booking", "family": "ordinary"}))
    (d / "correction__time.json").write_text(json.dumps({"id": "correction__time", "family": "correction"}))
    return d


def test_family_table_aggregates_pass_rates_per_family_and_scorer():
    t = family_table(ROWS)
    assert t["overall"] == {"n": 4, "passed": 3, "pass_rate": 75.0}
    assert t["families"]["ordinary"]["pass_rate"] == pytest.approx(66.67, abs=0.01)
    assert t["families"]["ordinary"]["scorers"]["final_state"] == {"n": 2, "passed": 1, "pass_rate": 50.0}
    assert t["families"]["correction"]["pass_rate"] == 100.0
    md = format_family_table(t)
    assert "ordinary" in md and "correction" in md and "overall" in md


def test_precision_gate_is_within_one_point_of_the_unfused_adapter():
    fused = family_table(ROWS)
    adapter = family_table(ROWS[:3] + [{**ROWS[3], "pass": False}])  # final_state 2/3 vs 1/3
    gate = precision_gate(fused, adapter, tolerance_points=1.0)
    assert gate["scorer"] == "final_state"  # task completion when the runner scored it
    assert gate["fused_pass_rate"] == 66.67 and gate["adapter_pass_rate"] == 33.33
    assert gate["delta_points"] == 33.34 and gate["pass"] is True  # fused better than adapter passes
    gate = precision_gate(adapter, fused, tolerance_points=1.0)
    assert gate["delta_points"] == -33.34 and gate["pass"] is False
    gate = precision_gate(fused, fused, tolerance_points=1.0, scorer="no_jargon")
    assert gate["pass"] is True and gate["scorer"] == "no_jargon"


def test_server_commands_use_the_installed_servers_with_thinking_off():
    fused = server_command(model="/m/r3-voice-abc", port=8010, name="r3-voice-abc")
    assert fused[:3] == ["vllm-mlx", "serve", "/m/r3-voice-abc"]
    assert "--served-model-name" in fused and "--tool-call-parser" not in fused  # the proxy parses tool calls, as in production
    assert '{"enable_thinking": false}' in fused
    adapter = server_command(model="/m/base", port=8010, name="r3-voice-abc", adapter="/m/r3-voice-abc/adapter")
    assert adapter[0] == "mlx_lm.server" and "--adapter-path" in adapter and '{"enable_thinking": false}' in adapter
    custom = server_command(model="/m/x", port=8011, name="n", template="my-serve {model} --port {port} --name {name}")
    assert custom == ["my-serve", "/m/x", "--port", "8011", "--name", "n"]


def test_wait_for_health_gives_up_quickly_on_a_closed_port():
    assert wait_for_health(f"http://127.0.0.1:{_free_port()}/v1", timeout=1.0) is False


def test_stub_runner_writes_rows_in_the_contract_shape_when_eval_runner_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    out = tmp_path / "out"
    res = run_r3_eval(base_url=f"http://127.0.0.1:{_free_port()}/v1", model="r3-voice-abc", scenarios=_scenarios(tmp_path),
                      out_dir=out, run_id="eval-t1")
    assert res["runner"] == "stub" and res["returncode"] == 0
    rows = collect_results(run_id="eval-t1", model_id="r3-voice-abc", out_dir=out, database_url=None)
    assert {r["family"] for r in rows} == {"ordinary", "correction"}
    assert all(r["pass"] is False and r["detail"]["stub"] is True for r in rows)  # no server answered
    assert {r["run_id"] for r in rows} == {"eval-t1"} and (out / "summary.md").exists()


def test_evaluate_model_dir_starts_the_server_waits_and_records_the_table(tmp_path, monkeypatch):
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    model_dir = tmp_path / "r3-voice-abc"
    model_dir.mkdir()
    table = evaluate_model_dir(model_dir, model_name="r3-voice-abc", run_id="eval-t2", scenarios=_scenarios(tmp_path),
                               out_dir=tmp_path / "eval", port=_free_port(), database_url=None, timeout=30)
    assert table["overall"]["pass_rate"] == 100.0 and set(table["families"]) == {"ordinary", "correction"}
    assert table["runner"] == "stub" and table["server"]["started"] is True and table["server"]["stopped"] is True
    assert (tmp_path / "eval" / "family_table.md").exists() and (tmp_path / "eval" / "family_table.json").exists()


def test_server_command_defaults_match_the_production_restart_script():
    from train.eval_hook import request_model_for

    fused = server_command(model="/m/r3-voice-abc", port=8010, name="r3-voice-abc")
    assert "--prefix-trie-cache" in fused and "--continuous-batching" not in fused and "--tool-call-parser" not in fused
    assert fused[fused.index("--default-temperature") + 1] == "0.2" and fused[fused.index("--max-tokens") + 1] == "256"
    assert request_model_for(fused, "r3-voice-abc") == "r3-voice-abc"
    adapter = server_command(model="/m/base", port=8010, name="r3-voice-abc", adapter="/m/r3-voice-abc/adapter")
    assert request_model_for(adapter, "r3-voice-abc") == "default_model"  # mlx_lm.server loads other names as repos


def test_run_r3_eval_passes_run_id_and_no_db(tmp_path, monkeypatch):
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    res = run_r3_eval(base_url=f"http://127.0.0.1:{_free_port()}/v1", model="m", scenarios=_scenarios(tmp_path),
                      out_dir=tmp_path / "o", run_id="rid", database_url=None)
    cmd = res["command"]
    assert cmd[cmd.index("--run-id") + 1] == "rid" and "--no-db" in cmd


def test_collect_results_prefers_rows_of_the_run_id(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    rows = [{"run_id": "rid", "model_id": "default_model", "scenario_id": "s", "family": "ordinary", "scorer": "x", "pass": True},
            {"run_id": "other", "model_id": "default_model", "scenario_id": "s", "family": "ordinary", "scorer": "x", "pass": False}]
    (out / "eval_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = collect_results(run_id="rid", model_id="r3-voice-abc", out_dir=out, database_url=None)
    assert [r["run_id"] for r in got] == ["rid"]


def test_evaluate_model_dir_scores_through_the_real_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    port = _free_port()
    table = evaluate_model_dir(tmp_path / "r3-voice-abc", model_name="r3-voice-abc", run_id="eval-t3", scenarios=_scenarios(tmp_path),
                               out_dir=tmp_path / "eval", port=port, database_url=None, timeout=60, through_proxy=True)
    assert table["overall"]["pass_rate"] == 100.0
    assert table["server"]["proxy"]["base_url"] == f"http://127.0.0.1:{port + 1}/v1" and table["server"]["proxy"]["stopped"] is True
    assert table["server"]["base_url"] == f"http://127.0.0.1:{port + 1}/v1"  # scored through the proxy
    rows = collect_results("eval-t3", "r3-voice-abc", tmp_path / "eval", None)
    assert rows[0]["detail"]["ms"] > 0


def test_side_proxy_runs_with_fallback_off_and_the_model_server_is_warmed_first(tmp_path, monkeypatch):
    from train.eval_hook import proxy_env

    env = proxy_env(upstream_port=8010, proxy_port=8011)
    assert env["PROXY_FALLBACK"] == "0" and env["PROXY_UPSTREAM"] == "http://127.0.0.1:8010" and env["PROXY_PORT"] == "8011"
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    port = _free_port()
    table = evaluate_model_dir(tmp_path / "m", model_name="m", run_id="eval-t4", scenarios=_scenarios(tmp_path),
                               out_dir=tmp_path / "eval", port=port, database_url=None, timeout=60, through_proxy=True)
    assert table["server"]["warmup"]["ok"] is True and table["server"]["warmup"]["ms"] >= 0
    assert table["server"]["warmup"]["url"] == f"http://127.0.0.1:{port}/v1"  # straight at the model server, not the proxy


def test_external_runner_command_template_and_cwd_are_honoured(tmp_path, monkeypatch):
    """E's runner lives on e/sim, not on main: $R3_EVAL_RUNNER runs it from E's worktree
    ($R3_EVAL_RUNNER_CWD) with the contract's arguments."""
    runner = tmp_path / "fake_r3_eval.py"
    runner.write_text(
        "import json, os, sys, pathlib\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "out = pathlib.Path(args['--out']); out.mkdir(parents=True, exist_ok=True)\n"
        "row = {'run_id': args['--run-id'], 'model_id': args['--model'], 'scenario_id': 's', 'family': 'ordinary', 'scorer': 'final_state', 'pass': True,\n"
        "       'detail': {'cwd': os.getcwd(), 'no_db': '--no-db' in sys.argv}}\n"
        "(out / 'eval_results.jsonl').write_text(json.dumps(row) + '\\n')\n")
    cwd = tmp_path / "e-worktree"
    cwd.mkdir()
    monkeypatch.setenv("R3_EVAL_RUNNER", f"{sys.executable} {runner}")
    monkeypatch.setenv("R3_EVAL_RUNNER_CWD", str(cwd))
    monkeypatch.delenv("R3_EVAL_FORCE_STUB", raising=False)
    res = run_r3_eval(base_url="http://127.0.0.1:1/v1", model="r3-voice-x", scenarios=tmp_path / "scen", out_dir=tmp_path / "out",
                      run_id="rid", database_url=None)
    assert res["runner"] == "r3-eval" and res["returncode"] == 0 and res["cwd"] == str(cwd)
    rows = collect_results("rid", "r3-voice-x", tmp_path / "out", None)
    assert rows[0]["pass"] is True and rows[0]["detail"] == {"cwd": str(cwd), "no_db": True}


def test_serve_only_keeps_the_side_stack_up_until_the_stop_file(tmp_path, monkeypatch):
    from train.eval_hook import serve_side_stack

    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    port = _free_port()
    out = tmp_path / "serve"
    stop = out / "STOP"
    import threading

    def stopper():
        import time as _t

        for _ in range(100):
            if (out / "serving.json").exists():
                break
            _t.sleep(0.2)
        stop.write_text("")

    threading.Thread(target=stopper, daemon=True).start()
    info = serve_side_stack(tmp_path / "base", "r3-voice-base", port=port, out_dir=out, minutes=5, poll_seconds=0.2)
    assert info["proxy"]["base_url"] == f"http://127.0.0.1:{port + 1}/v1" and info["warmup"]["ok"] is True
    assert info["stopped_by"] == "stop_file" and info["model_pid"] and info["proxy_pid"]
    served = json.loads((out / "serving.json").read_text())
    assert served["base_url"] == f"http://127.0.0.1:{port + 1}/v1"
    assert wait_for_health(info["proxy"]["base_url"], timeout=2) is False  # both are down again


E_ROWS = [  # the shape eval/runner.py writes to scores.jsonl (one episode, two scorers + one advisory)
    {"run_id": "r", "model_id": "m", "scenario_id": "ordinary_01", "episode_id": "e1", "repeat_seed": 1, "family": "ordinary",
     "scorer": "final_state", "pass": True, "advisory": False, "turn_idx": None, "detail": {}},
    {"run_id": "r", "model_id": "m", "scenario_id": "ordinary_01", "episode_id": "e1", "repeat_seed": 1, "family": "ordinary",
     "scorer": "no_success_claim_before_result", "pass": False, "advisory": False, "turn_idx": 3, "detail": {}},
    {"run_id": "r", "model_id": "m", "scenario_id": "ordinary_01", "episode_id": "e1", "repeat_seed": 1, "family": "ordinary",
     "scorer": "brevity", "pass": False, "advisory": True, "turn_idx": 2, "detail": {}},
    {"run_id": "r", "model_id": "m", "scenario_id": "correction_01", "episode_id": "e2", "repeat_seed": 1, "family": "correction",
     "scorer": "final_state", "pass": False, "advisory": False, "turn_idx": None, "detail": {}},
    {"run_id": "r", "model_id": "m", "scenario_id": "correction_01", "episode_id": "e2", "repeat_seed": 1, "family": "correction",
     "scorer": "tool_calls", "pass": True, "advisory": False, "turn_idx": None, "detail": {}},
]


def test_collect_results_reads_the_runners_scores_jsonl(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in E_ROWS) + "\n")
    rows = collect_results(run_id="r", model_id="m", out_dir=out, database_url=None)
    assert len(rows) == 5


def test_family_table_excludes_advisory_rows_and_reports_the_selection_metrics():
    t = family_table(E_ROWS)
    assert t["overall"]["n"] == 4 and t["advisory"]["n"] == 1 and t["advisory"]["failed"] == 1
    assert t["families"]["ordinary"]["n"] == 2 and "brevity" not in t["families"]["ordinary"]["scorers"]
    sel = t["selection"]
    assert sel["task_completion"] == 50.0 and sel["tool_calls"] == 100.0 and sel["unsupported_success_claims"] == 1
    assert sel["episodes"] == 2


def test_precision_gate_defaults_to_final_state_when_the_runner_scored_it():
    gate = precision_gate(family_table(E_ROWS), family_table(E_ROWS))
    assert gate["scorer"] == "final_state" and gate["pass"] is True and gate["delta_points"] == 0.0


def test_select_candidate_prefers_zero_unsupported_claims_then_task_completion(tmp_path):
    from train.select import select_candidates

    def table(name, tc, claims, tools=100.0):
        return {"model": name, "selection": {"task_completion": tc, "unsupported_success_claims": claims, "tool_calls": tools, "episodes": 27}}

    ranked = select_candidates([table("base", 40.0, 0), table("run1", 70.0, 2), table("run2", 65.0, 0)])
    assert [r["model"] for r in ranked] == ["run2", "base", "run1"]
    assert ranked[0]["rank"] == 1 and "2 unsupported" in ranked[2]["reason"]


def test_table_from_an_existing_runner_output(tmp_path):
    from train.eval_hook import table_from_scores

    out = tmp_path / "run"
    out.mkdir()
    (out / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in E_ROWS) + "\n")
    table = table_from_scores(out, model_name="r3-voice-base")
    assert table["model"] == "r3-voice-base" and table["run_id"] == "r" and table["selection"]["task_completion"] == 50.0
    assert (out / "family_table.json").exists() and (out / "family_table.md").exists()


def test_side_model_servers_run_with_queueing_admission(monkeypatch):
    """vllm-mlx 0.5.0's simple engine answers 503 to a second concurrent request unless
    VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait (S's live server sets it; so must the side ports)."""
    from train.eval_hook import server_env

    monkeypatch.delenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", raising=False)
    assert server_env()["VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION"] == "wait"
    monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "reject")
    assert server_env()["VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION"] == "reject"


def test_select_candidates_dedupes_tables_of_the_same_model():
    from train.select import select_candidates

    t = {"model": "m", "selection": {"task_completion": 50.0, "unsupported_success_claims": 0, "tool_calls": 10.0, "episodes": 27}}
    ranked = select_candidates([t, dict(t), {"model": "base", "selection": {"task_completion": 40.0, "unsupported_success_claims": 0, "tool_calls": 5.0, "episodes": 27}}])
    assert [r["model"] for r in ranked] == ["m", "base"] and [r["rank"] for r in ranked] == [1, 2]


def test_table_from_scores_can_exclude_the_caller_fault_episodes_listed_by_the_compare(tmp_path):
    from train.eval_hook import excluded_episode_prefixes, table_from_scores

    compare = tmp_path / "compare-excl-caller-faults.md"
    compare.write_text("Excluded 2 scenario x seed episode(s) in which the simulated caller mis-stated its own phone: ordinary_01-s1, correction_01-s1\n\n| x |\n")
    assert excluded_episode_prefixes(compare) == ["ordinary_01-s1", "correction_01-s1"]
    out = tmp_path / "run"
    out.mkdir()
    rows = [dict(r, episode_id=f"{r['scenario_id']}-s{r['repeat_seed']}-abc123") for r in E_ROWS]
    (out / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    table = table_from_scores(out, model_name="m", exclude_file=compare)
    assert table["excluded"]["prefixes"] == ["ordinary_01-s1", "correction_01-s1"] and table["excluded"]["rows_dropped"] == 5
    assert table["overall"]["n"] == 0 and (out / "family_table_excl.json").exists()
    table = table_from_scores(out, model_name="m")  # no exclusion: all 4 non-advisory rows
    assert table["overall"]["n"] == 4


def test_selection_has_all_pass_per_episode_and_g4_ordering():
    """G4 rule (plan owner's stand-in, 05:35): rank by all-pass rate (every non-advisory scorer) excluding
    caller faults, unsupported claims as the tie-breaker toward zero, then completion, then tool calls."""
    from train.select import select_candidates

    rows = [dict(r, episode_id=f"{r['scenario_id']}-s{r['repeat_seed']}-x") for r in E_ROWS]
    t = family_table(rows)
    assert t["selection"]["all_pass"] == 0.0  # e1 fails the claim scorer, e2 fails final_state
    rows2 = [dict(r, **{"pass": True}) for r in rows]
    assert family_table(rows2)["selection"]["all_pass"] == 100.0
    def tbl(name, ap, claims, tc):
        return {"model": name, "selection": {"all_pass": ap, "unsupported_success_claims": claims, "task_completion": tc, "tool_calls": 50.0, "episodes": 27}}
    ranked = select_candidates([tbl("base", 15.0, 0, 85.0), tbl("run1", 24.0, 6, 85.0), tbl("run2", 24.0, 1, 80.0)])
    assert [r["model"] for r in ranked] == ["run2", "run1", "base"]


def test_excluded_prefixes_are_read_from_the_excluded_line_anywhere_in_the_file(tmp_path):
    from train.eval_hook import excluded_episode_prefixes

    f = tmp_path / "final-compare-excl-caller-faults.md"
    f.write_text("# Final comparison: hosted baseline, base rerun, run 1 rerun, run 2\n\n"
                 "Excluded 3 scenario x seed episode(s) in which the simulated caller mis-stated its own phone or date of birth "
                 "in at least one run: ambiguous_branch_01-s0, duplicate_03-s2, tool_timeout_02-s1.\n\n| family | a | b |\n")
    assert excluded_episode_prefixes(f) == ["ambiguous_branch_01-s0", "duplicate_03-s2", "tool_timeout_02-s1"]
    g = tmp_path / "none.md"  # E's format when nothing is excluded: the list reads "none"
    g.write_text("Excluded 0 scenario x seed episode(s) in which the simulated caller mis-stated its own phone or date of birth "
                 "in at least one run: none\n| family |\n")
    assert excluded_episode_prefixes(g) == []
    h = tmp_path / "plain.md"
    h.write_text("Excluded 1 scenario x seed episode(s) ...: ordinary_02-s1\n")
    assert excluded_episode_prefixes(h) == ["ordinary_02-s1"]
