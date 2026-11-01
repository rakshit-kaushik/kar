"""Sizing and planning logic of the one-command launcher (no GPU, no server, no training here)."""
import json
import shutil
from pathlib import Path

import yaml
from mlx_lm.lora import yaml_loader

from train.launch import main, resolve_config
from train.runinfo import model_id

REPO = Path(__file__).resolve().parents[2]


def _recipe(name):
    return yaml.load((REPO / "train" / "configs" / name).read_text(), yaml_loader)


def test_one_epoch_is_len_train_at_batch_one_with_midpoint_eval_and_cosine_warmup():
    cfg, sizing = resolve_config(_recipe("voice_run1.yaml"), n_train=2400, mean_tokens=1200.0,
                                 data_dir=Path("/d"), run_dir=Path("/r"), window_minutes=None, tokens_per_sec=None)
    assert cfg["iters"] == 2400 and sizing["epoch_iters"] == 2400 and sizing["cut"] is None
    assert cfg["steps_per_eval"] == 1200 and cfg["save_every"] == 1200
    assert cfg["lr_schedule"] == {"name": "cosine_decay", "arguments": [1e-4, 150], "warmup": 4, "warmup_init": 0.0}
    assert sizing["optimizer_steps"] == 150 and sizing["warmup_steps"] == 4
    assert cfg["data"] == "/d" and cfg["adapter_path"] == "/r/adapter"
    assert "${" not in cfg["model"] and cfg["model"].endswith("Qwen3-4B-8bit")
    assert cfg["r3"]["task"] == "voice"


def test_fixed_iters_in_the_recipe_are_respected():
    cfg, sizing = resolve_config(_recipe("smoke.yaml"), n_train=100, mean_tokens=1200.0,
                                 data_dir=Path("/d"), run_dir=Path("/r"), window_minutes=10, tokens_per_sec=None)
    assert cfg["iters"] == 50 and sizing["fixed_iters"] is True
    assert cfg["steps_per_eval"] == 25 and cfg["save_every"] == 25
    assert cfg["lr_schedule"]["arguments"] == [1e-4, 4] and cfg["lr_schedule"]["warmup"] == 0


def test_window_sizing_cuts_conversation_count_to_fit_and_records_it():
    recipe = _recipe("voice_run1.yaml")
    cfg, sizing = resolve_config(recipe, n_train=2400, mean_tokens=1200.0, data_dir=Path("/d"), run_dir=Path("/r"),
                                 window_minutes=120, tokens_per_sec=300.0)
    # 120 min * 60 * 0.85 = 6120 s * 300 tok/s = 1.836M tokens / 1200 = 1530 examples -> 1520 (multiple of 16)
    assert cfg["iters"] == 1520 and sizing["cut"] == {"from": 2400, "to": 1520, "reason": "window"}
    assert cfg["steps_per_eval"] == 760 and cfg["lr_schedule"]["arguments"][1] == 95
    cfg, sizing = resolve_config(recipe, n_train=2400, mean_tokens=1200.0, data_dir=Path("/d"), run_dir=Path("/r"),
                                 window_minutes=120, tokens_per_sec=800.0)
    assert cfg["iters"] == 2400 and sizing["cut"] is None
    assert sizing["est_train_seconds"] == 2400 * 1200.0 / 800.0


def test_model_id_is_stable_and_prefixed():
    a = model_id("r3-voice", "cfg", "git", "manifest", "run1")
    assert a.startswith("r3-voice-") and len(a) == len("r3-voice-") + 10
    assert a == model_id("r3-voice", "cfg", "git", "manifest", "run1")
    assert a != model_id("r3-voice", "cfg", "git", "other", "run1")


def test_dry_run_writes_the_plan_and_touches_nothing_else(synth_dir, base_4b, tmp_path):
    recipe = _recipe("smoke.yaml")
    recipe["model"] = str(base_4b)
    cfg_path = tmp_path / "smoke_test.yaml"
    cfg_path.write_text(yaml.safe_dump(recipe))
    rc = main(["--config", str(cfg_path), "--data", str(synth_dir), "--artifacts", str(tmp_path / "art"),
               "--run-id", "dry1", "--dry-run"])
    assert rc == 0
    run_dir = tmp_path / "art" / "runs" / "dry1"
    resolved = yaml.load((run_dir / "config.resolved.yaml").read_text(), yaml_loader)
    assert resolved["iters"] == 50 and resolved["data"] == str(synth_dir)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "planned" and run["sizing"]["iters"] == 50
    assert run["data"]["manifest_hash"] and run["data"]["stats"]["n"] == 20
    assert run["git"]["hash"] and run["model_id"].startswith("r3-voice-")
    assert not (tmp_path / "art" / "models").exists()


def test_stop_server_terminates_whatever_listens_on_the_port(tmp_path):
    import socket
    import subprocess
    import sys
    import time

    from train.launch import _listening_pids, start_server, stop_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if _listening_pids(port):
                break
            time.sleep(0.1)
        assert proc.pid in _listening_pids(port)
        stop_server(port, timeout=10)
        assert _listening_pids(port) == [] and proc.wait(timeout=5) is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_start_server_reports_the_command_exit_code(monkeypatch):
    from train.launch import start_server

    monkeypatch.setenv("R3_SERVER_START_CMD", "exit 3")
    assert start_server() == 3
    monkeypatch.setenv("R3_SERVER_START_CMD", "true")
    assert start_server() == 0


def test_summarize_metrics_corrects_iterations_per_second_on_partial_report_windows(tmp_path):
    """mlx_lm 0.32.2 reports it/s = steps_per_report / window_time even when the last window
    (it == iters) is shorter than steps_per_report; the launcher re-derives it from the iteration delta."""
    import json

    from train.launch import summarize_metrics

    p = tmp_path / "metrics.jsonl"
    rows = [
        {"kind": "val", "iteration": 0, "val_loss": 1.0, "val_time": 1.0},
        {"kind": "train", "iteration": 10, "train_loss": 2.0, "learning_rate": 1e-4, "iterations_per_second": 1.0,
         "tokens_per_second": 50.0, "trained_tokens": 500, "peak_memory": 10.0},
        {"kind": "train", "iteration": 14, "train_loss": 1.9, "learning_rate": 1e-4, "iterations_per_second": 2.5,
         "tokens_per_second": 50.0, "trained_tokens": 700, "peak_memory": 11.0},  # 4 steps reported as 10/time
        {"kind": "val", "iteration": 13, "val_loss": 0.9, "val_time": 1.0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = summarize_metrics(p, {"wall_seconds": 20.0, "iters": 14}, batch=1, mean_tokens=1000.0, steps_per_report=10)
    assert m["iterations_per_second"] == 1.0  # (10 + 4) steps / (10 + 4) seconds
    assert m["total_tokens_per_second"] == 1000.0
    assert m["peak_memory_gb"] == 11.0 and m["final_val_loss"] == 0.9
