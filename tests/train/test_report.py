"""The training report: every run under $R3_ARTIFACTS/runs with a run.json, its eval table and the selection."""
import json
from pathlib import Path

from train.report import build_report, render_report


def _run(root: Path, run_id: str, task: str, status: str, model_id: str, tps: float, val: float, table: dict | None = None):
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "run_id": run_id, "task": task, "status": status, "model_id": model_id, "model_dir": f"/m/{model_id}",
        "config_path": f"train/configs/{task}.yaml", "config_hash": "c" * 64, "git": {"short": "abc1234", "dirty": False},
        "data": {"dir": "/d/x", "manifest_hash": "m" * 64, "stats": {"n": 88, "mean_tokens": 1780.0}},
        "sizing": {"iters": 50, "optimizer_steps": 4, "cut": None},
        "metrics": {"wall_seconds": 306.1, "total_tokens_per_second": tps, "peak_memory_gb": 11.4, "initial_val_loss": 2.23, "final_val_loss": val},
        "fuse": {"fuse_seconds": 5.6},
    }))
    if table:
        (d / "eval").mkdir()
        (d / "eval" / "family_table.json").write_text(json.dumps(table))
    return d


def test_report_lists_runs_candidates_and_the_selection(tmp_path):
    _run(tmp_path, "train-smoke-1", "voice", "done", "r3-voice-aaa", 360.5, 0.61,
         {"model": "r3-voice-aaa", "model_dir": "/m/r3-voice-aaa", "runner": "r3-eval", "scenarios": "eval/scenarios/test",
          "overall": {"n": 10, "passed": 7, "pass_rate": 70.0}, "families": {"ordinary": {"n": 10, "passed": 7, "pass_rate": 70.0, "scorers": {}}},
          "selection": {"task_completion": 60.0, "tool_calls": 90.0, "unsupported_success_claims": 0, "episodes": 27}})
    _run(tmp_path, "voice-run1", "voice", "done", "r3-voice-bbb", 900.0, 0.40,
         {"model": "r3-voice-bbb", "model_dir": "/m/r3-voice-bbb", "runner": "r3-eval", "scenarios": "eval/scenarios/test",
          "overall": {"n": 10, "passed": 8, "pass_rate": 80.0}, "families": {"ordinary": {"n": 10, "passed": 8, "pass_rate": 80.0, "scorers": {}}},
          "selection": {"task_completion": 75.0, "tool_calls": 95.0, "unsupported_success_claims": 1, "episodes": 27}})
    _run(tmp_path, "voice-run2-failed", "voice", "failed", "r3-voice-ccc", 0.0, 0.0)
    base = {"model": "r3-voice-base", "model_dir": "/m/base", "runner": "r3-eval", "scenarios": "eval/scenarios/test",
            "overall": {"n": 10, "passed": 4, "pass_rate": 40.0}, "families": {"ordinary": {"n": 10, "passed": 4, "pass_rate": 40.0, "scorers": {}}},
            "selection": {"task_completion": 40.0, "tool_calls": 70.0, "unsupported_success_claims": 0, "episodes": 27}}
    (tmp_path / "base_table.json").write_text(json.dumps(base))

    rep = build_report(tmp_path, extra_tables=[tmp_path / "base_table.json"])
    assert [r["run_id"] for r in rep["runs"]] == ["train-smoke-1", "voice-run1", "voice-run2-failed"]
    assert rep["selection"]["selected"]["model"] == "r3-voice-aaa"  # zero unsupported claims beats higher completion
    assert [c["model"] for c in rep["selection"]["ranked"]] == ["r3-voice-aaa", "r3-voice-base", "r3-voice-bbb"]
    md = render_report(rep)
    assert "train-smoke-1" in md and "r3-voice-bbb" in md and "failed" in md and "selected" in md.lower()
    assert "| r3-voice-aaa |" in md


def test_report_has_a_routing_row_from_zero_shot_scores_and_the_adapters_ledger_rows(tmp_path):
    for name, model, acc, f1, cov, err in (("routing-base-zeroshot-1", "r3-voice-base", 0.81, 0.8348, 0.94, 23),
                                           ("routing-hosted-zeroshot-1", "claude-sonnet-5-5", 0.99, 0.9936, 0.99, 1)):
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "scores.json").write_text(json.dumps({"model": model, "split": "test", "n": 100, "coverage": cov, "accuracy": acc,
                                                   "macro_f1": f1, "weighted_error": err}))
    _run(tmp_path, "routing-run1", "routing", "done", "r3-routing-abc", 600.0, 0.04)

    def ledger(run_id):  # stands in for the Postgres cost_ledger: S's batch job writes routing.* rows
        assert run_id == "routing-r3-routing-abc"
        return {"routing.accuracy": 0.91, "routing.macro_f1": 0.908, "routing.coverage": 0.98, "routing.weighted_error": 11, "routing.n": 100}

    rep = build_report(tmp_path, ledger_reader=ledger)
    rows = {r["model"]: r for r in rep["routing"]}
    assert rows["r3-routing-abc"]["accuracy"] == 0.91 and rows["r3-routing-abc"]["weighted_error"] == 11
    assert rows["r3-voice-base"]["macro_f1"] == 0.8348 and rows["claude-sonnet-5-5"]["accuracy"] == 0.99
    md = render_report(rep)
    assert "## Routing" in md and "| r3-routing-abc | adapter | 0.91 |" in md


def test_report_ranks_on_the_caller_fault_excluded_tables_when_any_exist_and_keeps_the_newest_per_model(tmp_path):
    from train.report import gather_tables

    sel = lambda ap, tc, claims: {"task_completion": tc, "tool_calls": 50.0, "unsupported_success_claims": claims, "episodes": 13, "all_pass": ap}
    # run 1: an old raw table in the training run dir, a newer excluded table in the eval run dir
    _run(tmp_path, "voice-run1", "voice", "done", "r3-voice-bbb", 900.0, 0.40,
         {"model": "r3-voice-bbb", "created_at": "2026-11-01T04:53:00+00:00", "selection": sel(None, 66.0, 7)})
    for rid, model, created, ap, tc, claims in (("r3-voice-bbb-eval-20261101-073000-localcaller", "r3-voice-bbb", "2026-11-01T08:00:00+00:00", 23.6, 85.5, 6),
                                                  ("r3-voice-bbb-eval-localcaller", "r3-voice-bbb", "2026-11-01T05:11:00+00:00", 30.0, 90.0, 1),
                                                  ("r3-voice-base-eval-localcaller", "r3-voice-base", "2026-11-01T05:11:00+00:00", 15.4, 84.6, 0)):
        d = tmp_path / "runs" / rid / "eval"
        d.mkdir(parents=True)
        (d / "family_table.json").write_text(json.dumps({"model": model, "run_id": rid, "created_at": created, "selection": sel(None, tc - 20, claims)}))
        (d / "family_table_excl.json").write_text(json.dumps({"model": model, "run_id": rid, "created_at": created, "selection": sel(ap, tc, claims),
                                                              "excluded": {"file": "/x/compare-excl-caller-faults.md", "prefixes": ["ordinary_02-s0"] * 26}}))
    (tmp_path / "models" / "r3-voice-bbb").mkdir(parents=True)

    tables, basis = gather_tables(tmp_path / "runs")
    assert basis == "caller-fault-excluded" and sorted(t["model"] for t in tables) == ["r3-voice-base", "r3-voice-bbb"]
    assert {t["model"]: t["run_id"] for t in tables}["r3-voice-bbb"] == "r3-voice-bbb-eval-20261101-073000-localcaller"  # newest, not best
    rep = build_report(tmp_path)
    assert rep["selection"]["basis"] == "caller-fault-excluded" and rep["selection"]["excluded_prefixes"] == 26
    assert [c["model"] for c in rep["selection"]["ranked"]] == ["r3-voice-bbb", "r3-voice-base"]
    assert rep["selection"]["selected"]["all_pass"] == 23.6 and rep["selection"]["selected"]["model_dir"] == str(tmp_path / "models" / "r3-voice-bbb")
    assert rep["runs"][0]["eval"]["all_pass"] == 23.6 and rep["runs"][0]["eval_run_id"] == "r3-voice-bbb-eval-20261101-073000-localcaller"
    md = render_report(rep)
    assert "caller-fault-excluded" in md and "| 1 | r3-voice-bbb | 23.6 | 6 |" in md
    # a paired precision gate next to the selected model's table is quoted
    gate_dir = tmp_path / "runs" / "r3-voice-bbb-eval-20261101-073000-localcaller" / "eval"
    (gate_dir / "precision_gate.json").write_text(json.dumps({"paired": True, "episodes_paired": 17, "fused": {"final_state": 14, "all_pass": 10, "unsupported_success_claims": 0},
                                                             "adapter": {"final_state": 14, "all_pass": 12, "unsupported_success_claims": 0}, "pass": True, "rule": "r"}))
    rep = build_report(tmp_path)
    assert rep["selection"]["gate"]["episodes_paired"] == 17
    assert "Precision gate (paired on 17" in render_report(rep) and "**PASS**" in render_report(rep)
