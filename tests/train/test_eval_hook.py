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
    adapter = family_table(ROWS[:3] + [{**ROWS[3], "pass": False}])  # 50% vs 75%
    gate = precision_gate(fused, adapter, tolerance_points=1.0)
    assert gate["fused_pass_rate"] == 75.0 and gate["adapter_pass_rate"] == 50.0
    assert gate["delta_points"] == 25.0 and gate["pass"] is True  # fused better than adapter passes
    gate = precision_gate(adapter, fused, tolerance_points=1.0)
    assert gate["delta_points"] == -25.0 and gate["pass"] is False
    gate = precision_gate(fused, fused, tolerance_points=1.0, scorer="final_state")
    assert gate["pass"] is True and gate["scorer"] == "final_state"


def test_server_commands_use_the_installed_servers_with_thinking_off():
    fused = server_command(model="/m/r3-voice-abc", port=8010, name="r3-voice-abc")
    assert fused[:3] == ["vllm-mlx", "serve", "/m/r3-voice-abc"]
    assert "--served-model-name" in fused and "--tool-call-parser" in fused and "qwen3" in fused
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
