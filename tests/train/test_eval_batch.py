"""The eval batch's local-caller mode: the base is served on a second side pair as the simulated caller
and every candidate is scored with the runner pointed at it (fake servers, stub runner)."""
import json
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_local_caller_batch_serves_the_caller_pair_and_scores_candidates_with_it(tmp_path, monkeypatch):
    from train.eval_batch import main

    art = tmp_path / "art"
    (art / "models" / "fake-cand").mkdir(parents=True)
    (art / "models" / "Qwen3-4B-8bit").mkdir(parents=True)
    (tmp_path / "scen").mkdir()
    (tmp_path / "scen" / "ordinary__x.json").write_text(json.dumps({"id": "ordinary__x", "family": "ordinary"}))
    monkeypatch.setenv("R3_ARTIFACTS", str(art))
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    monkeypatch.delenv("R3_EVAL_RUNNER", raising=False)
    port = _free_port()
    while any(p in range(port, port + 4) for p in []):
        port = _free_port()
    rc = main(["--candidates", "fake-cand:1", "--local-caller", "--no-lock", "--port", str(port), "--scenarios", str(tmp_path / "scen"),
               "--services-root", str(tmp_path), "--no-db"])
    assert rc == 0
    batch = json.loads(next((art / "runs").glob("eval-batch-*/batch.json")).read_text())
    assert batch["caller"]["base_url"] == f"http://127.0.0.1:{port + 3}/v1" and batch["caller"]["stopped"] is True
    step = next(s for s in batch["steps"] if s["step"] == "eval")
    assert step["run_id"] == "fake-cand-eval-localcaller" and step["selection"]["episodes"] == 1
    assert "--caller-base-url" in step["runner_args"] and f"127.0.0.1:{port + 3}" in step["runner_args"]


def test_candidate_spec_parses_dir_repeats_and_optional_served_name():
    from train.eval_batch import parse_candidate

    assert parse_candidate("r3-voice-abc:3") == ("r3-voice-abc", 3, "r3-voice-abc")
    assert parse_candidate("Qwen3-4B-8bit:1:r3-voice-base") == ("Qwen3-4B-8bit", 1, "r3-voice-base")
    assert parse_candidate("r3-voice-abc") == ("r3-voice-abc", 3, "r3-voice-abc")


def test_gate_step_scores_the_unfused_adapter_on_the_base_and_writes_the_precision_gate(tmp_path, monkeypatch):
    from train.eval_batch import main

    art = tmp_path / "art"
    (art / "models" / "r3-voice-abc" / "adapter").mkdir(parents=True)
    (art / "models" / "Qwen3-4B-8bit").mkdir(parents=True)
    (tmp_path / "scen").mkdir()
    (tmp_path / "scen" / "ordinary__x.json").write_text(json.dumps({"id": "ordinary__x", "family": "ordinary"}))
    monkeypatch.setenv("R3_ARTIFACTS", str(art))
    monkeypatch.setenv("R3_EVAL_FORCE_STUB", "1")
    monkeypatch.setenv("R3_EVAL_SERVER_CMD", f"{sys.executable} {HERE / 'fake_server.py'} {{port}} {{name}}")
    port = _free_port()
    rc = main(["--candidates", "r3-voice-abc:1", "--gate", "r3-voice-abc", "--no-lock", "--port", str(port), "--scenarios", str(tmp_path / "scen"),
               "--services-root", str(tmp_path), "--no-db"])
    assert rc == 0
    batch = json.loads(next((art / "runs").glob("eval-batch-*/batch.json")).read_text())
    gate = next(s for s in batch["steps"] if s["step"] == "gate")
    assert gate["model_id"] == "r3-voice-abc" and gate["adapter"].endswith("r3-voice-abc/adapter") and gate["gate"]["pass"] is True
    assert (art / "runs" / "r3-voice-abc-adapter-eval" / "eval" / "family_table.json").exists()
    assert (art / "runs" / "r3-voice-abc-eval" / "eval" / "precision_gate.json").exists()
