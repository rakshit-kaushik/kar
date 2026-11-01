"""One command for a training window (RUNBOOK "GPU schedule"):

    uv run python -m train.launch --config train/configs/smoke.yaml --data $R3_ARTIFACTS/datasets/<name>

1. preflight: read the recipe, verify the dataset manifest + loss masks (train/loader.py), size
   `iters` to one epoch and to the GPU window from measured tokens/s, write
   $R3_ARTIFACTS/runs/<id>/{config.resolved.yaml, data_report.json, run.json}
2. take the GPU lock (`scripts/gpu_lock.sh acquire T <minutes> <run id>`, waiting while another
   session holds it), `scripts/gpu_flip.sh hosted` and stop the model server on :8000 (every other
   session keeps working against the hosted URL; the proxy falls back to the hosted model too)
3. train: train/lora_driver.py (= mlx_lm.lora train_model, clip 1.0, metrics every 10 steps)
4. train/fuse/fuse_quantize.py: adapter + 8-bit base -> fused 8-bit model under
   $R3_ARTIFACTS/models/<prefix>-<hash>/ with the unfused adapter kept beside it
5. `scripts/gpu_flip.sh local`, restart the stack (scripts/restart.sh or $R3_SERVER_START_CMD), confirm
   it with the dispatch probe (scripts/probe_dispatch.py, retried), release the lock: always, even when
   a step failed. `--services-root <checkout>` selects whose .env and scripts are touched (default
   $R3_SERVICES_ROOT or this checkout); `--rehearse` runs only this server cycle, no training.
6. cost_ledger rows + runs/<id>/run.json (config, git hash, data manifest hash, tokens/s, losses)
7. optionally `--eval`: train/eval_hook.py scores the new model with r3-eval

Sizing: one epoch = len(train.jsonl) micro-steps at batch 1. With `--window-minutes` and a
measured total tokens/s (`--tokens-per-sec`, or the latest run.json with one, e.g. the smoke run),
iters is cut to fit `window * safety_factor`; seq and epochs are never raised. Recipes with a fixed
`iters` (smoke.yaml) are left alone. Validation + a checkpoint happen at iters/2 and at the end.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from mlx_lm.lora import yaml_loader

from train.env import REPO_ROOT, artifacts_dir, expand, load_env
from train.ledger import write_ledger, write_run_json
from train.loader import check_dataset, load_manifest, load_tokenizer, manifest_hash, read_jsonl
from train.runinfo import config_hash, git_info, model_id, new_run_id

SERVER_PORT = 8000
LOCK_OWNER = "T"


def services_root(explicit: str | None = None) -> Path:
    """The checkout whose .env, scripts/gpu_flip.sh, scripts/restart.sh and scripts/gpu_lock.sh are used."""
    load_env()
    return Path(explicit or os.environ.get("R3_SERVICES_ROOT") or REPO_ROOT).expanduser().resolve()


def gpu_lock(root: Path, *args: str) -> tuple[int, str]:
    script = Path(root) / "scripts" / "gpu_lock.sh"
    if not script.exists():
        return 0, f"no {script}; GPU lock not enforced"
    out = subprocess.run(["bash", str(script), *[str(a) for a in args]], cwd=root, capture_output=True, text=True)
    return out.returncode, (out.stdout + out.stderr).strip()


def acquire_gpu(root: Path, owner: str, minutes: int, purpose: str, wait_minutes: float = 120.0, poll_seconds: float = 30.0) -> dict:
    """Acquire the GPU lock, polling while another session holds it (announce by message around this)."""
    t0 = time.time()
    deadline = t0 + wait_minutes * 60
    attempts = 0
    while True:
        attempts += 1
        rc, out = gpu_lock(root, "acquire", owner, str(minutes), purpose)
        if rc == 0:
            return {"acquired": True, "waited_s": round(time.time() - t0), "attempts": attempts, "output": out, "minutes": minutes}
        if time.time() >= deadline:
            return {"acquired": False, "waited_s": round(time.time() - t0), "attempts": attempts, "output": out, "minutes": minutes}
        if attempts == 1:
            print(f"[lock] {out}; waiting up to {wait_minutes:.0f} min", flush=True)
        time.sleep(poll_seconds)


def release_gpu(root: Path, owner: str) -> tuple[int, str]:
    rc, out = gpu_lock(root, "release", owner)
    print(f"[lock] {out}", flush=True)
    return rc, out


def probe_stack(root: Path, attempts: int = 3, interval: float = 20.0) -> dict:
    """After restart.sh: session S's dispatch probe (joins a room, expects the agent to greet). The
    worker's first job process takes ~20 s to spawn after a start, so retry. PASS is in stdout."""
    custom = os.environ.get("R3_SERVER_PROBE_CMD")
    script = Path(root) / "scripts" / "probe_dispatch.py"
    if not custom and not script.exists():
        return {"ok": None, "attempts": 0, "output": f"no {script}", "skipped": True}
    outputs = []
    for i in range(1, attempts + 1):
        if custom:
            proc = subprocess.run(custom, shell=True, cwd=root, capture_output=True, text=True)
        else:
            proc = subprocess.run([sys.executable, str(script)], cwd=root, capture_output=True, text=True)
        text = (proc.stdout + proc.stderr).strip()
        outputs.append(text[-400:])
        ok = proc.returncode == 0 and "PASS" in text
        print(f"[server] dispatch probe {i}/{attempts}: {'PASS' if ok else 'FAIL'}", flush=True)
        if ok:
            return {"ok": True, "attempts": i, "output": outputs[-1]}
        if i < attempts:
            time.sleep(interval)
    return {"ok": False, "attempts": attempts, "output": outputs[-1] if outputs else ""}


def load_recipe(path: str | Path) -> dict:
    return yaml.load(Path(path).read_text(), yaml_loader) or {}


def expand_strings(obj):
    if isinstance(obj, str):
        return expand(obj)
    if isinstance(obj, dict):
        return {k: expand_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_strings(v) for v in obj]
    return obj


def resolve_config(recipe: dict, *, n_train: int, mean_tokens: float, data_dir: Path, run_dir: Path,
                   window_minutes: float | None, tokens_per_sec: float | None) -> tuple[dict, dict]:
    """The recipe plus the run-sized keys mlx_lm.lora needs: iters, steps_per_eval, save_every,
    lr_schedule (cosine over the optimizer steps with linear warmup), data, adapter_path."""
    cfg = expand_strings(copy.deepcopy(recipe))
    r3 = cfg.get("r3") or {}
    batch = int(cfg.get("batch_size", 1))
    ga = int(cfg.get("grad_accumulation_steps", 1))
    epochs = int(r3.get("epochs", 1))
    epoch_iters = max(1, n_train // batch)
    fixed = recipe.get("iters") is not None
    iters = int(recipe["iters"]) if fixed else epoch_iters * epochs
    window = window_minutes if window_minutes is not None else r3.get("window_minutes")
    safety = float(r3.get("safety_factor", 0.85))
    budget_seconds = float(window) * 60.0 * safety if window is not None else None
    seconds_per_iter = (batch * mean_tokens / tokens_per_sec) if tokens_per_sec else None
    cut = None
    if not fixed and budget_seconds is not None and seconds_per_iter:
        max_iters = int(budget_seconds // seconds_per_iter)
        if max_iters < iters:
            new = max(ga, (max_iters // ga) * ga)
            cut = {"from": iters, "to": new, "reason": "window"}
            iters = new
    steps_per_eval = int(recipe.get("steps_per_eval") or max(1, iters // 2))
    save_every = int(recipe.get("save_every") or steps_per_eval)
    decay_steps = max(1, math.ceil(iters / ga))
    warmup = int(round(float(r3.get("warmup_ratio", 0.0)) * decay_steps))
    lr = float(cfg["learning_rate"])
    cfg.update(
        iters=iters,
        steps_per_eval=steps_per_eval,
        save_every=save_every,
        data=str(data_dir),
        adapter_path=str(Path(run_dir) / "adapter"),
        lr_schedule={"name": r3.get("schedule", "cosine_decay"), "arguments": [lr, decay_steps], "warmup": warmup, "warmup_init": 0.0},
    )
    sizing = {
        "n_train": n_train,
        "batch_size": batch,
        "grad_accumulation_steps": ga,
        "epochs": epochs,
        "epoch_iters": epoch_iters,
        "fixed_iters": fixed,
        "iters": iters,
        "optimizer_steps": decay_steps,
        "warmup_steps": warmup,
        "steps_per_eval": steps_per_eval,
        "mean_tokens_per_example": mean_tokens,
        "tokens_per_sec_used": tokens_per_sec,
        "window_minutes": window,
        "safety_factor": safety,
        "budget_seconds": budget_seconds,
        "est_train_seconds": (iters * seconds_per_iter) if seconds_per_iter else None,
        "cut": cut,
    }
    return cfg, sizing


def reference_tokens_per_sec(artifacts: Path, task: str, reference_run: str | None) -> tuple[float | None, str | None]:
    """Measured total tokens/s from a previous run.json (the smoke run sizes run 1)."""
    runs = artifacts / "runs"
    candidates = [runs / reference_run] if reference_run else sorted(runs.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if runs.exists() else []
    for d in candidates:
        p = d / "run.json"
        if not p.exists():
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tps = ((rec.get("metrics") or {}).get("total_tokens_per_second"))
        if tps and (reference_run or rec.get("task") == task):
            return float(tps), str(d)
    return None, None


def summarize_metrics(metrics_path: Path, train_summary: dict, batch: int, mean_tokens: float,
                      steps_per_report: int = 10) -> dict:
    train, val = [], []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                (train if rec["kind"] == "train" else val).append(rec)
    # mlx_lm 0.32.2 reports it/s as steps_per_report / window_time even for a short final window
    # (it == iters), so re-derive it from the iteration deltas: total steps / total training time.
    steps = time_s = 0.0
    prev = 0
    for r in train:
        n = r["iteration"] - prev
        prev = r["iteration"]
        if r.get("iterations_per_second"):
            steps += n
            time_s += steps_per_report / r["iterations_per_second"]
    it_per_s = (steps / time_s) if time_s else None
    out = {
        "wall_seconds": train_summary.get("wall_seconds"),
        "iters": train_summary.get("iters"),
        "reports": len(train),
        "final_train_loss": train[-1]["train_loss"] if train else None,
        "first_train_loss": train[0]["train_loss"] if train else None,
        "trained_loss_tokens": train[-1]["trained_tokens"] if train else None,
        "loss_tokens_per_second": (sum(r["tokens_per_second"] for r in train) / len(train)) if train else None,
        "iterations_per_second": it_per_s,
        "peak_memory_gb": max((r["peak_memory"] for r in train), default=None),
        "val": [{"iteration": r["iteration"], "val_loss": r["val_loss"]} for r in val],
        "initial_val_loss": val[0]["val_loss"] if val else None,
        "final_val_loss": val[-1]["val_loss"] if val else None,
    }
    if out["iterations_per_second"]:
        out["total_tokens_per_second"] = out["iterations_per_second"] * batch * mean_tokens
        out["total_tokens_processed_est"] = (train_summary.get("iters") or 0) * batch * mean_tokens
    return out


def ledger_rows(ctx: dict, metrics: dict, fuse_meta: dict | None, status: str) -> list[dict]:
    rows = [{
        "item": "train_run", "qty": 1, "unit": "run", "measured": True,
        "note": {"status": status, "task": ctx["task"], "config": ctx["config_path"], "config_hash": ctx["config_hash"],
                 "git": ctx["git"].get("hash"), "git_dirty": ctx["git"].get("dirty"), "data": str(ctx["data"]["dir"]),
                 "manifest_hash": ctx["data"]["manifest_hash"], "base_model": ctx["cfg"]["model"],
                 "iters": ctx["sizing"]["iters"], "cut": ctx["sizing"]["cut"], "model_dir": str(ctx["model_dir"])},
    }]
    def add(item, qty, unit, measured=True, note=None):
        if qty is not None:
            rows.append({"item": item, "qty": qty, "unit": unit, "measured": measured, "note": note})
    add("train_wall_time", metrics.get("wall_seconds"), "s")
    add("train_iters", metrics.get("iters"), "micro-steps")
    add("train_tokens_per_sec_total", metrics.get("total_tokens_per_second"), "tokens/s", note="prompt+target tokens, from it/s x mean example length")
    add("train_tokens_per_sec_loss", metrics.get("loss_tokens_per_second"), "tokens/s", note="mlx_lm tokens/s: loss (target) tokens only")
    add("train_tokens_total", metrics.get("total_tokens_processed_est"), "tokens", measured=False, note="iters x mean example length")
    add("train_loss_tokens", metrics.get("trained_loss_tokens"), "tokens")
    add("train_peak_memory", metrics.get("peak_memory_gb"), "GB")
    add("train_loss_final", metrics.get("final_train_loss"), "nats")
    add("val_loss_initial", metrics.get("initial_val_loss"), "nats")
    add("val_loss_final", metrics.get("final_val_loss"), "nats")
    if fuse_meta:
        add("fuse_quantize_wall_time", fuse_meta.get("fuse_seconds"), "s")
    add("gpu_hours_training", (metrics.get("wall_seconds") or 0) / 3600.0 if metrics.get("wall_seconds") else None, "h",
        note="unit_cost left for session D's machine-hour rate")
    return rows


def preflight(opts) -> dict:
    load_env()
    artifacts = Path(opts.artifacts).expanduser() if opts.artifacts else artifacts_dir()
    recipe = load_recipe(opts.config)
    r3 = recipe.get("r3") or {}
    task = r3.get("task", "voice")
    prefix = r3.get("model_prefix", f"r3-{task}")
    run_id = opts.run_id or new_run_id(task)
    run_dir = artifacts / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(opts.data)
    mhash = manifest_hash(manifest)
    model_path = expand(str(recipe["model"]))
    tokenizer = load_tokenizer(model_path)
    seq = int(recipe.get("max_seq_length", 2048))
    require_tools = task == "voice"
    train_examples = read_jsonl(manifest.files["train"])
    report = check_dataset(tokenizer, train_examples, seq, require_tools=require_tools)
    valid_report = None
    if "valid" in manifest.files:
        valid_report = check_dataset(tokenizer, read_jsonl(manifest.files["valid"]), seq, require_tools=require_tools)
    for name, rep in (("train", report), ("valid", valid_report)):
        if rep is not None and not rep.ok:
            for idx, errs in rep.failures[:20]:
                print(f"[preflight] {name} example {idx}: {'; '.join(errs)}")
            raise SystemExit(f"[preflight] {rep.n_failed} of {rep.n} {name} examples fail the mask/length checks; "
                             f"fix the data (train/loader.py prints the masks), never raise max_seq_length")

    tps, tps_source = (opts.tokens_per_sec, "--tokens-per-sec") if opts.tokens_per_sec else reference_tokens_per_sec(artifacts, task, opts.reference_run)
    cfg, sizing = resolve_config(recipe, n_train=len(train_examples), mean_tokens=report.stats["mean_tokens"],
                                 data_dir=manifest.dir, run_dir=run_dir, window_minutes=opts.window_minutes, tokens_per_sec=tps)
    sizing["tokens_per_sec_source"] = tps_source
    git = git_info()
    chash = config_hash(cfg)
    mid = model_id(prefix, chash, git.get("hash") or "nogit", mhash, run_id)
    model_dir = artifacts / "models" / mid
    cfg_path = run_dir / "config.resolved.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "data_report.json").write_text(json.dumps({
        "train": {"stats": report.stats, "checks": [m.checks for m in report.masks]},
        "valid": {"stats": valid_report.stats} if valid_report else None,
    }, indent=2))
    ctx = {
        "run_id": run_id, "run_dir": run_dir, "artifacts": artifacts, "task": task, "prefix": prefix,
        "config_path": str(opts.config), "config_hash": chash, "cfg": cfg, "recipe": recipe, "sizing": sizing,
        "git": git, "model_id": mid, "model_dir": model_dir,
        "data": {"dir": manifest.dir, "manifest_hash": mhash, "manifest": manifest.data, "stats": report.stats,
                 "n_valid": valid_report.n if valid_report else 0},
    }
    write_run_json(run_dir, run_record(ctx, "planned"))
    print(f"[plan] run {run_id} · task {task} · model id {mid}")
    print(f"[plan] base {cfg['model']}")
    print(f"[plan] data {manifest.dir} (manifest {mhash[:12]}…) · {report.stats['n']} train / {ctx['data']['n_valid']} valid · "
          f"mean {report.stats['mean_tokens']:.0f} tokens, max {report.stats['max_tokens']} (seq {seq}); scaffold "
          f"{report.stats['scaffold_tokens']} tokens")
    print(f"[plan] iters {sizing['iters']}{' (fixed)' if sizing['fixed_iters'] else ''} = {sizing['optimizer_steps']} optimizer steps "
          f"(grad accum {sizing['grad_accumulation_steps']}), warmup {sizing['warmup_steps']}, eval+checkpoint every {sizing['steps_per_eval']}")
    if tps:
        print(f"[plan] sized from {tps:.0f} total tokens/s ({tps_source}): est. {sizing['est_train_seconds']/60:.1f} min of training"
              + (f" within {sizing['window_minutes']} min x {sizing['safety_factor']}" if sizing["window_minutes"] else ""))
    elif not sizing["fixed_iters"]:
        print("[plan] no measured tokens/s (pass --tokens-per-sec or run the smoke first): one full epoch, unsized")
    if sizing["cut"]:
        print(f"[plan] CUT: window fits {sizing['cut']['to']} of {sizing['cut']['from']} examples; cutting conversation count "
              f"(consider 2 targets per conversation at the data stage instead); seq and epochs stay")
    print(f"[plan] git {git.get('short')}{' (dirty)' if git.get('dirty') else ''} · resolved config {cfg_path}")
    return ctx


def run_record(ctx: dict, status: str, **extra) -> dict:
    return {
        "run_id": ctx["run_id"], "status": status, "task": ctx["task"], "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_id": ctx["model_id"], "model_dir": str(ctx["model_dir"]), "base_model": ctx["cfg"]["model"],
        "config_path": ctx["config_path"], "config_hash": ctx["config_hash"], "resolved_config": ctx["cfg"],
        "git": ctx["git"], "data": {**ctx["data"], "dir": str(ctx["data"]["dir"])}, "sizing": ctx["sizing"],
        "command": sys.argv, **extra,
    }


def flip(mode: str, root: Path = REPO_ROOT) -> None:
    print(f"[server] {root}/scripts/gpu_flip.sh {mode}", flush=True)
    subprocess.run(["bash", "scripts/gpu_flip.sh", mode], cwd=root, check=True)


def _listening_pids(port: int) -> list[int]:
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()]


def stop_server(port: int = SERVER_PORT, timeout: float = 30.0) -> None:
    custom = os.environ.get("R3_SERVER_STOP_CMD")
    if custom:
        print(f"[server] stop: {custom}", flush=True)
        subprocess.run(custom, shell=True, cwd=REPO_ROOT)
        return
    pids = _listening_pids(port)
    if not pids:
        print(f"[server] nothing listening on :{port}", flush=True)
        return
    print(f"[server] stopping model server on :{port} (pids {pids})", flush=True)
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _listening_pids(port):
        time.sleep(1)
    for pid in _listening_pids(port):
        print(f"[server] pid {pid} still listening after {timeout:.0f}s, SIGKILL", flush=True)
        os.kill(pid, signal.SIGKILL)


def start_server(root: Path = REPO_ROOT) -> int:
    cmd = os.environ.get("R3_SERVER_START_CMD") or "bash scripts/restart.sh"
    print(f"[server] start in {root}: {cmd}", flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=root).returncode
    if rc != 0:
        print(f"[server] WARNING: start command exited {rc} (scripts/restart.sh is session S's); start the server by hand", flush=True)
    return rc


def train_step(ctx: dict, device: str) -> dict:
    run_dir = ctx["run_dir"]
    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "train.log"
    cmd = [sys.executable, "-m", "train.lora_driver", "--config", str(run_dir / "config.resolved.yaml"),
           "--metrics", str(metrics_path), "--device", device]
    print(f"[train] {' '.join(cmd)}\n[train] log {log_path}", flush=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "true"}
    seen = 0
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        while True:
            rc = proc.poll()
            if metrics_path.exists():
                lines = metrics_path.read_text().splitlines()
                for line in lines[seen:]:
                    r = json.loads(line)
                    if r["kind"] == "train":
                        print(f"[train] iter {r['iteration']:>6} loss {r['train_loss']:.4f} lr {r['learning_rate']:.2e} "
                              f"loss-tok/s {r['tokens_per_second']:.0f} it/s {r['iterations_per_second']:.3f} "
                              f"peak {r['peak_memory']:.1f}GB", flush=True)
                    else:
                        print(f"[train] iter {r['iteration']:>6} val_loss {r['val_loss']:.4f}", flush=True)
                seen = len(lines)
            if rc is not None:
                break
            time.sleep(2)
    if rc != 0:
        raise RuntimeError(f"training failed with exit {rc}; see {log_path}")
    return json.loads((run_dir / "train_summary.json").read_text())


def fuse_step(ctx: dict, device: str) -> dict:
    from train.fuse.fuse_quantize import fuse_and_quantize

    q = (ctx["cfg"].get("r3") or {}).get("quantize") or {}
    return fuse_and_quantize(
        base=ctx["cfg"]["model"], adapter=ctx["run_dir"] / "adapter", out_dir=ctx["model_dir"],
        bits=int(q.get("bits", 8)), group_size=int(q.get("group_size", 64)), device=device,
        log_path=ctx["run_dir"] / "fuse.log",
        extra_meta={"model_id": ctx["model_id"], "run_id": ctx["run_id"], "config_hash": ctx["config_hash"],
                    "git": ctx["git"], "manifest_hash": ctx["data"]["manifest_hash"], "task": ctx["task"]},
    )


def server_cycle(root: Path, opts, ctx: dict, work) -> dict:
    """lock -> flip hosted -> stop :8000 -> work() -> flip local -> restart -> probe -> release, with
    the restore steps in a finally block so nobody waits on a human. Returns what happened."""
    state = {"lock": None, "probe": None, "flipped": False, "stopped": False, "restarted": None, "error": None}
    minutes = int(opts.lock_minutes or math.ceil(((ctx.get("sizing") or {}).get("est_train_seconds") or 600) / 60) + 15)
    try:
        if not opts.no_lock:
            state["lock"] = acquire_gpu(root, LOCK_OWNER, minutes, ctx["run_id"], wait_minutes=opts.lock_wait_minutes)
            print(f"[lock] {state['lock']['output']}", flush=True)
            if not state["lock"]["acquired"]:
                raise RuntimeError(f"GPU lock not acquired within {opts.lock_wait_minutes} min: {state['lock']['output']}")
        if not opts.no_flip:
            flip("hosted", root)
            state["flipped"] = True
        if not opts.no_server:
            stop_server()
            state["stopped"] = True
        work()
    except BaseException as e:  # noqa: BLE001 - restore first, report after
        state["error"] = f"{type(e).__name__}: {e}"
        print(f"[launch] FAILED: {state['error']}", flush=True)
    finally:
        if state["flipped"]:
            try:
                flip("local", root)
            except Exception as e:  # noqa: BLE001
                print(f"[server] WARNING: gpu_flip.sh local failed: {e}", flush=True)
        if state["stopped"]:
            state["restarted"] = start_server(root)
            if state["restarted"] == 0:
                state["probe"] = probe_stack(root)
        if state["lock"] and state["lock"]["acquired"]:
            rc, _ = release_gpu(root, LOCK_OWNER)
            state["lock"]["released"] = rc == 0
    return state


def rehearse(opts) -> int:
    """The server cycle alone (no training): proves lock, flip, stop, restart and probe on the live stack."""
    load_env()
    root = services_root(opts.services_root)
    artifacts = Path(opts.artifacts).expanduser() if opts.artifacts else artifacts_dir()
    run_id = opts.run_id or new_run_id("rehearse")
    run_dir = artifacts / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"run_id": run_id, "sizing": {"est_train_seconds": 0}}
    t0 = time.time()
    state = server_cycle(root, opts, ctx, lambda: print("[rehearse] (no training)", flush=True))
    status = "rehearsed" if not state["error"] else "failed"
    write_run_json(run_dir, {"run_id": run_id, "status": status, "task": "rehearse", "services_root": str(root),
                             "started_at": t0, "finished_at": time.time(), "lock": state["lock"], "probe": state["probe"],
                             "restarted": state["restarted"], "error": state["error"], "command": sys.argv})
    print(f"[rehearse] {status}: lock {state['lock']} · restart rc {state['restarted']} · probe {state['probe']}")
    return 0 if status == "rehearsed" else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="train/configs/{smoke,voice_run1,routing}.yaml")
    ap.add_argument("--data", required=True, help="dataset directory with manifest.json")
    ap.add_argument("--services-root", default=None, help="checkout whose .env and scripts are used (default $R3_SERVICES_ROOT or this one)")
    ap.add_argument("--no-lock", action="store_true", help="skip scripts/gpu_lock.sh (dry runs beside the idle server)")
    ap.add_argument("--lock-minutes", type=int, default=None, help="lock duration; default: estimated training time + 15")
    ap.add_argument("--lock-wait-minutes", type=float, default=120.0, help="how long to wait for a held lock")
    ap.add_argument("--rehearse", action="store_true", help="server cycle only: lock, flip, stop, restart, probe, release")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--artifacts", default=None, help="override $R3_ARTIFACTS")
    ap.add_argument("--window-minutes", type=float, default=None, help="GPU window; default from the recipe's r3.window_minutes")
    ap.add_argument("--tokens-per-sec", type=float, default=None, help="measured total tokens/s; default: latest run.json of this task")
    ap.add_argument("--reference-run", default=None, help="run id whose measured tokens/s sizes this run")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="cpu = dry run off the GPU")
    ap.add_argument("--no-flip", action="store_true", help="do not touch .env (scripts/gpu_flip.sh)")
    ap.add_argument("--no-server", action="store_true", help="do not stop/restart the model server")
    ap.add_argument("--no-db", action="store_true", help="ledger to runs/<id>/ledger.jsonl only")
    ap.add_argument("--dry-run", action="store_true", help="preflight + plan only")
    ap.add_argument("--eval", action="store_true", help="after the restart, score the model with train/eval_hook.py")
    ap.add_argument("--scenarios", default="eval/scenarios/test", help="scenario dir for --eval")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    opts = parse_args(argv)
    if opts.rehearse:
        return rehearse(opts)
    ctx = preflight(opts)
    if opts.dry_run:
        print("[plan] dry run: nothing started")
        return 0
    run_dir = ctx["run_dir"]
    root = services_root(opts.services_root)
    results: dict = {"metrics": {}, "fuse": None}
    t_start = time.time()

    def work():
        write_run_json(run_dir, run_record(ctx, "training", started_at=t_start))
        summary = train_step(ctx, opts.device)
        results["metrics"] = summarize_metrics(run_dir / "metrics.jsonl", summary, ctx["sizing"]["batch_size"],
                                               ctx["data"]["stats"]["mean_tokens"], steps_per_report=int(ctx["cfg"].get("steps_per_report", 10)))
        write_run_json(run_dir, run_record(ctx, "fusing", started_at=t_start, metrics=results["metrics"]))
        results["fuse"] = fuse_step(ctx, opts.device)

    state = server_cycle(root, opts, ctx, work)
    status, error = ("done", None) if not state["error"] else ("failed", state["error"])
    metrics, fuse_meta = results["metrics"], results["fuse"]
    database_url = None if opts.no_db else os.environ.get("DATABASE_URL")
    rows = ledger_rows(ctx, metrics, fuse_meta, status)
    ledger = write_ledger(rows, ctx["run_id"], ctx["model_id"] if status == "done" else None, database_url, run_dir / "ledger.jsonl")
    record = run_record(ctx, status, started_at=t_start, finished_at=time.time(), wall_seconds=time.time() - t_start,
                        metrics=metrics, fuse=fuse_meta, ledger=ledger, error=error, services_root=str(root),
                        lock=state["lock"], probe=state["probe"], restarted=state["restarted"])
    write_run_json(run_dir, record)
    print(f"[launch] {status}: run.json {run_dir / 'run.json'} · ledger {'postgres + ' if ledger['db'] else ''}{ledger['jsonl']}"
          + (f" (db error: {ledger['error']})" if ledger["error"] else ""))
    if status != "done":
        return 1
    print(f"[launch] model {ctx['model_id']} at {ctx['model_dir']} · total tokens/s "
          f"{metrics.get('total_tokens_per_second', 0):.0f} · final val loss {metrics.get('final_val_loss')}")
    if opts.eval:
        from train.eval_hook import evaluate_model_dir

        table = evaluate_model_dir(ctx["model_dir"], model_name=ctx["model_id"], run_id=ctx["run_id"],
                                   scenarios=opts.scenarios, out_dir=run_dir / "eval")
        record["eval"] = table
        write_run_json(run_dir, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
