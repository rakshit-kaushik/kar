"""One lock window after run 1: routing adapter training, candidate evals with session E's runner
(hosted caller), and the base served for E's baseline reruns, all on T's side ports.

    uv run python -m train.eval_batch --candidates r3-voice-62e6c53289:3 r3-voice-efc3a5e9f9:1 \
        [--routing-data $R3_ARTIFACTS/datasets/routing-v1] [--base-minutes 15] [--services-root <checkout>]

Steps (each recorded in $R3_ARTIFACTS/runs/eval-batch-<ts>/batch.json):
  1. gpu_lock.sh acquire T <minutes> "eval batch"
  2. optional: train/launch.py on routing.yaml with --no-lock --no-flip --no-server (beside the live server)
  3. for each candidate <model id>:<repeats>: serve it on :8010 (vllm-mlx, S's flags) + S's proxy on :8011,
     warm up, run `r3-eval run --repeats N` through the proxy (R3_EVAL_RUNNER from E's worktree), write
     runs/<model id>-eval/eval/family_table.{json,md}
  4. optional: serve the base on :8010/:8011 for --base-minutes (E's reruns; STOP file ends it early)
  5. release the lock; print the candidate ranking (train/select.py) including E's base baseline table
--local-caller (hosted credits out): the base is served on port+2/port+3 as the simulated caller for the
whole batch and every candidate is scored with `--caller-base-url http://127.0.0.1:<port+3>/v1
--caller-model r3-voice-base`; run ids end in -localcaller so the rows never mix with hosted-caller rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from train.env import REPO_ROOT, artifacts_dir, load_env
from train.eval_hook import evaluate_model_dir, serve_side_stack, table_from_scores
from train.launch import acquire_gpu, release_gpu, services_root
from train.select import select_candidates

LOCK_OWNER = "T"


def parse_candidate(spec: str) -> tuple[str, int, str]:
    """`<model dir name>[:<repeats>[:<served name>]]` -> (dir name, repeats, served name)."""
    parts = spec.split(":")
    d = parts[0]
    repeats = int(parts[1]) if len(parts) > 1 and parts[1] else 3
    name = parts[2] if len(parts) > 2 and parts[2] else d
    return d, repeats, name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="*", default=[], help="<model dir>[:<repeats>[:<served name>]] ... (none = routing and/or base serving only)")
    ap.add_argument("--routing-data", default=None, help="train the routing adapter first (routing.yaml) on this dataset")
    ap.add_argument("--base-minutes", type=float, default=0.0, help="serve the base on the side ports for E afterwards")
    ap.add_argument("--base-model", default=None, help="base model dir (default $R3_ARTIFACTS/models/Qwen3-4B-8bit)")
    ap.add_argument("--scenarios", default="eval/scenarios/test")
    ap.add_argument("--services-root", default=None)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--lock-minutes", type=int, default=None)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--base-table", default=None, help="E's base baseline run dir (scores.jsonl) to include in the ranking")
    ap.add_argument("--local-caller", action="store_true", help="serve the base on port+2/+3 as the simulated caller (credits out)")
    ap.add_argument("--no-db", action="store_true", help="runner writes no Postgres rows (tests)")
    a = ap.parse_args(argv)

    load_env()
    artifacts = artifacts_dir()
    root = services_root(a.services_root)
    base = Path(a.base_model or artifacts / "models" / "Qwen3-4B-8bit")
    batch_id = f"eval-batch-{time.strftime('%Y%m%d-%H%M%S')}"
    out = artifacts / "runs" / batch_id
    out.mkdir(parents=True, exist_ok=True)
    cands = [parse_candidate(c) for c in a.candidates]
    minutes = a.lock_minutes or (10 + (8 if a.routing_data else 0) + sum(5 + 25 * r for _, r, _ in cands) + int(a.base_minutes))
    record: dict = {"batch_id": batch_id, "candidates": cands, "steps": [], "started_at": time.time()}

    def save():
        (out / "batch.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

    caller_proc = None
    caller_args = ""
    suffix = ""
    if a.local_caller:
        caller_out = out / "caller-serve"
        cmd = [sys.executable, "-m", "train.eval_hook", "--model-dir", str(base), "--model-name", "r3-voice-base", "--port", str(a.port + 2),
               "--serve-only", "--minutes", str(minutes + 30), "--out", str(caller_out)]
        record["caller"] = {"command": cmd, "base_url": f"http://127.0.0.1:{a.port + 3}/v1", "started": True, "stopped": False}
        caller_args = f" --caller-base-url http://127.0.0.1:{a.port + 3}/v1 --caller-model r3-voice-base"
        suffix = "-localcaller"
    lock = None
    if not a.no_lock:
        lock = acquire_gpu(root, LOCK_OWNER, minutes, batch_id, wait_minutes=180)
        record["lock"] = lock
        save()
        print(f"[batch] lock: {lock['output']}", flush=True)
        if not lock["acquired"]:
            return 1
    try:
        if a.local_caller:
            caller_out.mkdir(parents=True, exist_ok=True)
            caller_log = (caller_out / "serve.log").open("a")
            caller_proc = subprocess.Popen(record["caller"]["command"], cwd=REPO_ROOT, stdout=caller_log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            deadline = time.time() + 900
            while time.time() < deadline and not (caller_out / "serving.json").exists():
                if caller_proc.poll() is not None:
                    raise RuntimeError(f"caller stack exited early; see {caller_out / 'serve.log'}")
                time.sleep(2)
            if not (caller_out / "serving.json").exists():
                raise RuntimeError("caller stack not up within 15 min")
            print(f"[batch] local caller up at {record['caller']['base_url']}", flush=True)
            save()
        if a.routing_data:
            cmd = [sys.executable, "-m", "train.launch", "--config", "train/configs/routing.yaml", "--data", a.routing_data,
                   "--run-id", "routing-run1", "--services-root", str(root), "--tokens-per-sec", "850", "--window-minutes", "20",
                   "--no-lock", "--no-flip", "--no-server"]
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
            rr = artifacts / "runs" / "routing-run1" / "run.json"
            rec = json.loads(rr.read_text()) if rr.exists() else {}
            record["steps"].append({"step": "routing", "rc": rc, "seconds": round(time.time() - t0), "model_id": rec.get("model_id"), "status": rec.get("status")})
            save()
        for dir_name, repeats, model_id in cands:
            model_dir = artifacts / "models" / dir_name
            run_id = f"{model_id}-eval{suffix}"
            runner_base = (os.environ.get("R3_EVAL_RUNNER") or "/Users/abgup/conductor/workspaces/kar/stockholm/.venv/bin/r3-eval run").split(" --repeats")[0].split(" --caller-base-url")[0]
            os.environ["R3_EVAL_RUNNER_CWD"] = os.environ.get("R3_EVAL_RUNNER_CWD") or "/Users/abgup/conductor/workspaces/kar/stockholm"
            runner_args = f" --repeats {repeats}{caller_args}"
            if not os.environ.get("R3_EVAL_FORCE_STUB"):
                os.environ["R3_EVAL_RUNNER"] = runner_base + runner_args  # one template per candidate: repeats differ
            t0 = time.time()
            step = {"step": "eval", "model_id": model_id, "model_dir": str(model_dir), "repeats": repeats, "run_id": run_id, "runner_args": runner_args}
            try:
                table = evaluate_model_dir(model_dir, model_name=model_id, run_id=run_id, scenarios=a.scenarios,
                                           out_dir=artifacts / "runs" / run_id / "eval", port=a.port,
                                           database_url=None if a.no_db else "env", timeout=900, through_proxy=True)
                step.update(selection=table["selection"], overall=table["overall"], runner=table["runner"], rc=table["returncode"])
                # mirror the table into the training run's directory when there is one
                for rj in (artifacts / "runs").glob("*/run.json"):
                    rec = _safe(rj)
                    if rec and rec.get("model_id") == model_id and rec.get("task") in ("voice", "routing"):
                        (rj.parent / "eval").mkdir(exist_ok=True)
                        (rj.parent / "eval" / "family_table.json").write_text(json.dumps(table, indent=2, default=str) + "\n")
            except Exception as e:  # noqa: BLE001
                step["error"] = f"{type(e).__name__}: {e}"
            step["seconds"] = round(time.time() - t0)
            record["steps"].append(step)
            save()
        if a.base_minutes > 0:
            info = serve_side_stack(base, "r3-voice-base", port=a.port, out_dir=out / "base-serve", minutes=a.base_minutes)
            record["steps"].append({"step": "base-serve", "stopped_by": info.get("stopped_by"), "minutes": a.base_minutes})
            save()
    finally:
        if caller_proc is not None:
            (out / "caller-serve" / "STOP").touch()
            try:
                caller_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(caller_proc.pid, 15)
            record["caller"]["stopped"] = True
        if lock and lock.get("acquired"):
            rc, _ = release_gpu(root, LOCK_OWNER)
            record["lock"]["released"] = rc == 0
        record["finished_at"] = time.time()
        save()
    tables = []
    for rj in (artifacts / "runs").glob("*/eval/family_table.json"):
        t = _safe(rj)
        if t and t.get("selection"):
            tables.append(t)
    if a.base_table:
        try:
            tables.append(table_from_scores(Path(a.base_table), model_name="r3-voice-base"))
        except FileNotFoundError as e:
            print(f"[batch] base table skipped: {e}")
    if tables:
        ranked = select_candidates(tables)
        record["ranking"] = ranked
        save()
        for r in ranked:
            print(f"[batch] #{r['rank']} {r['model']}: {r['reason']}")
    print(f"[batch] record {out / 'batch.json'}")
    return 0


def _safe(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
