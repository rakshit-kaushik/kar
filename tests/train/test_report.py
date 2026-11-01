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
