"""Sizing and planning logic of the one-command launcher (no GPU, no server, no training here)."""
import json
import shutil
from pathlib import Path

import pytest
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


def test_stop_server_terminates_whatever_listens_on_the_port(tmp_path, monkeypatch):
    import socket
    import subprocess
    import sys
    import time

    from train.launch import _listening_pids, start_server, stop_server

    monkeypatch.delenv("R3_NEVER_KILL_PORTS", raising=False)  # this test owns the fake listener it kills

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


def _fake_services_root(tmp_path, monkeypatch):
    """A services checkout: real gpu_lock.sh, fake gpu_flip.sh / restart.sh / probe_dispatch.py that log their calls."""
    import shutil
    import stat

    root = tmp_path / "services"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "gpu_lock.sh", root / "scripts" / "gpu_lock.sh")
    (root / "scripts" / "gpu_flip.sh").write_text('#!/bin/bash\necho "$1" >> "$(dirname "$0")/../flips.log"\n')
    (root / "scripts" / "restart.sh").write_text('#!/bin/bash\necho "restart owner=${R3_OWNER:-}" >> "$(dirname "$0")/../events.log"\necho up\n')
    (root / "scripts" / "server_stop.sh").write_text('#!/bin/bash\necho stop >> "$(dirname "$0")/../events.log"\n')
    (root / "scripts" / "probe_dispatch.py").write_text(
        "import pathlib\np = pathlib.Path(__file__).parent.parent / 'probes'\nn = int(p.read_text()) + 1 if p.exists() else 1\n"
        "p.write_text(str(n))\nprint('FAIL' if n < 2 else 'PASS')\n")
    for f in (root / "scripts").iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("R3_ARTIFACTS", str(tmp_path / "art"))
    (tmp_path / "art").mkdir(exist_ok=True)
    monkeypatch.delenv("R3_SERVER_START_CMD", raising=False)
    monkeypatch.delenv("R3_SERVER_PROBE_CMD", raising=False)
    return root


def test_acquire_gpu_waits_for_a_held_lock_and_releases_it(tmp_path, monkeypatch):
    from train.launch import acquire_gpu, gpu_lock, release_gpu

    root = _fake_services_root(tmp_path, monkeypatch)
    assert gpu_lock(root, "acquire", "S", "5", "load test")[0] == 0
    res = acquire_gpu(root, "T", 5, "smoke", wait_minutes=0)
    assert res["acquired"] is False and "held by S" in res["output"]
    assert gpu_lock(root, "release", "S")[0] == 0
    res = acquire_gpu(root, "T", 5, "smoke", wait_minutes=0)
    assert res["acquired"] is True
    assert gpu_lock(root, "status")[0] == 1  # held (by T)
    assert release_gpu(root, "T")[0] == 0 and gpu_lock(root, "status")[0] == 0


def test_probe_stack_retries_until_the_dispatch_probe_passes(tmp_path, monkeypatch):
    from train.launch import probe_stack

    root = _fake_services_root(tmp_path, monkeypatch)
    res = probe_stack(root, attempts=3, interval=0)
    assert res["ok"] is True and res["attempts"] == 2


def test_rehearse_cycles_the_server_through_the_services_root_under_the_lock(tmp_path, monkeypatch):
    root = _fake_services_root(tmp_path, monkeypatch)
    rc = main(["--rehearse", "--services-root", str(root), "--artifacts", str(tmp_path / "art"), "--run-id", "reh1",
               "--config", str(REPO / "train" / "configs" / "smoke.yaml"), "--data", "unused"])
    assert rc == 0
    assert (root / "flips.log").read_text().split() == ["hosted", "local"]
    assert (root / "events.log").read_text().splitlines() == ["stop", "restart owner=T"]  # restart.sh needs R3_OWNER=T while T holds the lock
    run = json.loads((tmp_path / "art" / "runs" / "reh1" / "run.json").read_text())
    assert run["status"] == "rehearsed" and run["lock"]["acquired"] is True and run["lock"]["released"] is True
    assert run["probe"]["ok"] is True and run["probe"]["attempts"] == 2
    from train.launch import gpu_lock

    assert gpu_lock(root, "status")[0] == 0  # free again


def test_stop_server_prefers_the_services_root_stop_script(tmp_path, monkeypatch):
    from train.launch import stop_server

    root = _fake_services_root(tmp_path, monkeypatch)
    monkeypatch.delenv("R3_SERVER_STOP_CMD", raising=False)
    stop_server(root=root)
    assert (root / "events.log").read_text().split() == ["stop"]


def test_stop_server_never_kills_ports_under_the_test_guard(tmp_path, monkeypatch):
    from train.launch import stop_server

    monkeypatch.delenv("R3_SERVER_STOP_CMD", raising=False)
    with pytest.raises(RuntimeError, match="R3_NEVER_KILL_PORTS"):
        stop_server(port=8000, root=tmp_path)  # no stop script in root -> would fall through to a port kill


def test_tail_metrics_skips_a_partially_written_last_line_and_does_not_repeat(tmp_path):
    from train.launch import tail_metrics

    p = tmp_path / "metrics.jsonl"
    p.write_text('{"kind": "val", "iteration": 0, "val_loss": 1.0, "val_time": 1.0}\n{"kind": "train", "iteration": 10, "train_lo')
    seen, printed = tail_metrics(p, 0)
    assert seen == 1 and [r["iteration"] for r in printed] == [0]
    p.write_text(p.read_text() + 'ss": 2.0, "learning_rate": 1e-4, "iterations_per_second": 0.5, "tokens_per_second": 50.0, "trained_tokens": 1, "peak_memory": 1.0}\n')
    seen, printed = tail_metrics(p, seen)
    assert seen == 2 and [r["iteration"] for r in printed] == [10]
    seen, printed = tail_metrics(p, seen)
    assert seen == 2 and printed == []


def test_rehearse_can_deploy_a_model_dir_through_restart_sh(tmp_path, monkeypatch):
    root = _fake_services_root(tmp_path, monkeypatch)
    (root / "scripts" / "restart.sh").write_text('#!/bin/bash\necho "restart owner=${R3_OWNER:-} model=${MODEL_DIR:-base}" >> "$(dirname "$0")/../events.log"\n')
    rc = main(["--rehearse", "--services-root", str(root), "--artifacts", str(tmp_path / "art"), "--run-id", "dep1",
               "--serve-model-dir", "/models/r3-voice-abc", "--config", str(REPO / "train" / "configs" / "smoke.yaml"), "--data", "-"])
    assert rc == 0
    assert (root / "events.log").read_text().splitlines()[-1] == "restart owner=T model=/models/r3-voice-abc"
    run = json.loads((tmp_path / "art" / "runs" / "dep1" / "run.json").read_text())
    assert run["served_model_dir"] == "/models/r3-voice-abc"


def test_sizing_counts_validation_passes_in_the_estimate():
    cfg, sizing = resolve_config(_recipe("voice_run1.yaml"), n_train=2044, mean_tokens=1771.0, data_dir=Path("/d"), run_dir=Path("/r"),
                                 window_minutes=120, tokens_per_sec=850.0, n_valid=244)
    assert cfg["val_batches"] == 96
    # 3 validation passes (it=1, midpoint, end) over 96 examples, forward-only at ~3x a training step's rate
    assert sizing["est_eval_seconds"] == round(3 * 96 * 1771.0 / 850.0 / 3, 1)
    assert sizing["est_total_seconds"] == round(sizing["est_train_seconds"] + sizing["est_eval_seconds"], 1)


def test_start_server_drops_the_worktrees_env_values_so_restart_sh_reads_its_own(monkeypatch, tmp_path):
    """Nov 1 07:10: the launcher's environment carried the worktree .env copy's LOCAL_LLM_MODEL=r3-voice-base and
    restart.sh (which lets environment values win over kar/.env) advertised the deployed run-1 weights as r3-voice-base."""
    import train.launch as launch

    seen = tmp_path / "seen"
    monkeypatch.setattr(launch, "env_file_keys", lambda: {"LOCAL_LLM_MODEL", "LLM_MODEL", "MODEL_DIR"})
    monkeypatch.setenv("LOCAL_LLM_MODEL", "r3-voice-base")
    monkeypatch.setenv("LLM_MODEL", "r3-voice-base")
    monkeypatch.setenv("MODEL_DIR", "/stale/dir")
    monkeypatch.setenv("R3_KEEP_ME", "yes")
    monkeypatch.setenv("R3_SERVER_START_CMD",
                       "printf '%s|%s|%s|%s|%s' \"${LOCAL_LLM_MODEL:-unset}\" \"${LLM_MODEL:-unset}\" \"$R3_OWNER\" "
                       f"\"${{MODEL_DIR:-unset}}\" \"$R3_KEEP_ME\" > {seen}")
    assert launch.start_server(model_dir="/m/r3-voice-x") == 0
    assert seen.read_text() == "unset|unset|T|/m/r3-voice-x|yes"
    assert launch.start_server() == 0
    assert seen.read_text() == "unset|unset|T|unset|yes"


def test_acquire_gpu_trusts_only_a_status_read_that_names_the_owner(tmp_path, monkeypatch):
    """A gpu_lock.sh whose acquire prints success while status still shows another holder (the fail-open read
    seen Nov 1 07:14) is not treated as acquired."""
    from train.launch import acquire_gpu

    root = tmp_path / "svc"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "gpu_lock.sh").write_text(
        "#!/usr/bin/env bash\ncase \"$1\" in acquire) echo \"gpu: acquired by $2 for $3 min ($4)\"; exit 0;; "
        "status) echo \"gpu: held by S for another 9 min (block)\"; exit 1;; release) echo released; exit 0;; esac\n")
    res = acquire_gpu(root, "T", 5, "smoke", wait_minutes=0, poll_seconds=0)
    assert res["acquired"] is False and res["attempts"] == 1


def test_served_model_is_read_from_the_process_command_line_not_the_served_name():
    """Nov 1 08:02: :8000 advertised r3-voice-730f8ba68e while its process had loaded r3-voice-206506525c."""
    from train.launch import parse_server_command, verify_served

    cmd = ("/x/.venv/bin/python scripts/vllm_mlx_serve.py serve /art/models/r3-voice-206506525c --host 127.0.0.1 --port 8000 "
           "--served-model-name r3-voice-730f8ba68e --prefix-trie-cache")
    got = parse_server_command(cmd)
    assert got == {"model_dir": "/art/models/r3-voice-206506525c", "served_name": "r3-voice-730f8ba68e", "adapter": None}
    ok, msg = verify_served("/art/models/r3-voice-730f8ba68e", {"pid": 1, "command": cmd, **got})
    assert ok is False and "requested /art/models/r3-voice-730f8ba68e but the server loaded /art/models/r3-voice-206506525c" in msg
    ok, _ = verify_served("/art/models/r3-voice-206506525c/", {"pid": 1, "command": cmd, **got})
    assert ok is True  # same directory, trailing slash
    ok, _ = verify_served(None, {"pid": 1, "command": cmd, **got})
    assert ok is True  # nothing requested: report only
    mlx = parse_server_command("mlx_lm.server --model /art/models/Qwen3-4B-8bit --adapter-path /art/models/r3-voice-x/adapter --port 8000")
    assert mlx == {"model_dir": "/art/models/Qwen3-4B-8bit", "served_name": None, "adapter": "/art/models/r3-voice-x/adapter"}
    assert verify_served("/art/models/x", None) == (False, "no model server listening")
