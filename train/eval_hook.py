"""Eval-on-checkpoint: given a model directory, start the server, wait for health, run `r3-eval`
(session E's runner; train/eval_stub.py until it is merged), record the per-family table.

    uv run python -m train.eval_hook --model-dir $R3_ARTIFACTS/models/r3-voice-<hash> \
        --scenarios eval/scenarios/test --out $R3_ARTIFACTS/runs/<id>/eval --run-id <id>
    uv run python -m train.eval_hook --model-dir ... --gate      # precision gate: fused vs adapter-on-base

Servers (installed versions, flags read from their argparse):
  fused 8-bit dir  -> vllm-mlx 0.5.0: `vllm-mlx serve <dir> --served-model-name <id> --continuous-batching
                      --enable-auto-tool-choice --tool-call-parser qwen3 --reasoning-parser qwen3
                      --default-chat-template-kwargs '{"enable_thinking": false}'`
  adapter on base  -> mlx_lm.server 0.32.2: `mlx_lm.server --model <base> --adapter-path <dir>/adapter
                      --chat-template-args '{"enable_thinking": false}'` (vllm-mlx has no adapter flag)
Override either with $R3_EVAL_SERVER_CMD, a template with {model} {port} {name} {adapter} placeholders.
Results come from Postgres `eval_results` (run_id / model_id) when the runner wrote there, else from
<out>/eval_results.jsonl. Selection is by this table on the test set, never by loss.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from train.env import REPO_ROOT, load_env

CHAT_KWARGS = '{"enable_thinking": false}'
DEFAULT_FUSED_CMD = (
    "vllm-mlx serve {model} --host 127.0.0.1 --port {port} --served-model-name {name} --continuous-batching "
    "--enable-auto-tool-choice --tool-call-parser qwen3 --reasoning-parser qwen3 --default-chat-template-kwargs '{chat_kwargs}'"
)
DEFAULT_ADAPTER_CMD = (
    "mlx_lm.server --model {model} --adapter-path {adapter} --host 127.0.0.1 --port {port} --chat-template-args '{chat_kwargs}'"
)


def server_command(model: str, port: int, name: str, adapter: str | None = None, template: str | None = None) -> list[str]:
    template = template or os.environ.get("R3_EVAL_SERVER_CMD") or (DEFAULT_ADAPTER_CMD if adapter else DEFAULT_FUSED_CMD)
    return shlex.split(template.format(model=model, port=port, name=name, adapter=adapter or "", chat_kwargs=CHAT_KWARGS))


def _get(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_health(base_url: str, timeout: float = 600.0, interval: float = 2.0, proc: subprocess.Popen | None = None) -> bool:
    root = base_url.rsplit("/v1", 1)[0]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if _get(f"{root}/health") or _get(f"{base_url.rstrip('/')}/models"):
            return True
        time.sleep(interval)
    return False


def start_server(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_server(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def runner_available() -> bool:
    if os.environ.get("R3_EVAL_FORCE_STUB"):
        return False
    try:
        return importlib.util.find_spec("eval.runner") is not None
    except (ImportError, ValueError):
        return False


def run_r3_eval(base_url: str, model: str, scenarios: str | Path, out_dir: str | Path, run_id: str, timeout: float = 3600.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = ["--base-url", base_url, "--model", model, "--scenarios", str(scenarios), "--out", str(out_dir)]
    if runner_available():
        exe = Path(sys.executable).with_name("r3-eval")
        cmd = [str(exe) if exe.exists() else "r3-eval", *common]
        runner = "r3-eval"
    else:
        cmd = [sys.executable, "-m", "train.eval_stub", *common, "--run-id", run_id]
        runner = "stub"
    log = out_dir / "eval.log"
    print(f"[eval] {runner}: {' '.join(cmd)}", flush=True)
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return {"runner": runner, "command": cmd, "returncode": proc.returncode, "log": str(log)}


def collect_results(run_id: str, model_id: str, out_dir: str | Path, database_url: str | None, since: float | None = None) -> list[dict]:
    rows: list[dict] = []
    if database_url:
        try:
            import psycopg

            with psycopg.connect(database_url, connect_timeout=5) as conn, conn.cursor() as cur:
                cur.execute(
                    "select run_id, model_id, scenario_id, family, scorer, pass, detail from eval_results "
                    "where model_id = %s and (run_id = %s or created_at >= %s) order by id",
                    (model_id, run_id, dt.datetime.fromtimestamp(since or 0, dt.timezone.utc)),
                )
                rows = [dict(zip(("run_id", "model_id", "scenario_id", "family", "scorer", "pass", "detail"), r)) for r in cur.fetchall()]
        except Exception as e:  # noqa: BLE001
            print(f"[eval] postgres eval_results unavailable ({type(e).__name__}: {e}); using jsonl", flush=True)
    if not rows:
        p = Path(out_dir) / "eval_results.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def _rate(passed: int, n: int) -> float:
    return round(100.0 * passed / n, 2) if n else 0.0


def family_table(rows: list[dict]) -> dict:
    fams: dict[str, dict] = {}
    for r in rows:
        fam = fams.setdefault(str(r.get("family")), {"n": 0, "passed": 0, "scorers": {}})
        sc = fam["scorers"].setdefault(str(r.get("scorer")), {"n": 0, "passed": 0})
        ok = 1 if r.get("pass") else 0
        fam["n"] += 1
        fam["passed"] += ok
        sc["n"] += 1
        sc["passed"] += ok
    for fam in fams.values():
        fam["pass_rate"] = _rate(fam["passed"], fam["n"])
        for sc in fam["scorers"].values():
            sc["pass_rate"] = _rate(sc["passed"], sc["n"])
    n = len(rows)
    passed = sum(1 for r in rows if r.get("pass"))
    return {"overall": {"n": n, "passed": passed, "pass_rate": _rate(passed, n)}, "families": dict(sorted(fams.items()))}


def format_family_table(table: dict) -> str:
    scorers = sorted({s for f in table["families"].values() for s in f["scorers"]})
    head = ["family", "rows", "pass %"] + scorers
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, fam in table["families"].items():
        cells = [name, str(fam["n"]), f"{fam['pass_rate']:.1f}"] + [
            f"{fam['scorers'][s]['pass_rate']:.0f}" if s in fam["scorers"] else "-" for s in scorers]
        lines.append("| " + " | ".join(cells) + " |")
    o = table["overall"]
    lines.append("| " + " | ".join(["**overall**", str(o["n"]), f"{o['pass_rate']:.1f}"] + ["" for _ in scorers]) + " |")
    return "\n".join(lines)


def _scorer_rate(table: dict, scorer: str | None) -> float:
    if not scorer:
        return table["overall"]["pass_rate"]
    n = sum(f["scorers"].get(scorer, {}).get("n", 0) for f in table["families"].values())
    passed = sum(f["scorers"].get(scorer, {}).get("passed", 0) for f in table["families"].values())
    return _rate(passed, n)


def precision_gate(fused: dict, adapter: dict, tolerance_points: float = 1.0, scorer: str | None = None) -> dict:
    """G3: the fused, re-quantized artifact must be within `tolerance_points` of the unfused adapter on
    the 8-bit base (the configuration that was trained). Better-than-adapter passes."""
    f, a = _scorer_rate(fused, scorer), _scorer_rate(adapter, scorer)
    delta = round(f - a, 2)
    return {"scorer": scorer or "overall", "fused_pass_rate": f, "adapter_pass_rate": a, "delta_points": delta,
            "tolerance_points": tolerance_points, "pass": delta >= -tolerance_points}


def evaluate_model_dir(model_dir: str | Path, model_name: str, run_id: str, scenarios: str | Path, out_dir: str | Path,
                       base_url: str | None = None, adapter: str | Path | None = None, base: str | Path | None = None,
                       port: int = 8010, database_url: str | None = "env", timeout: float = 600.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if database_url == "env":
        load_env()
        database_url = os.environ.get("DATABASE_URL")
    server: dict = {"started": False, "stopped": False, "command": None, "base_url": base_url}
    proc = None
    if base_url is None:
        cmd = server_command(model=str(base or model_dir), port=port, name=model_name, adapter=str(adapter) if adapter else None)
        base_url = f"http://127.0.0.1:{port}/v1"
        server.update(command=cmd, base_url=base_url, log=str(out_dir / "server.log"))
        print(f"[eval] starting server: {' '.join(cmd)}", flush=True)
        proc = start_server(cmd, out_dir / "server.log")
        server["started"] = True
        t0 = time.time()
        if not wait_for_health(base_url, timeout=timeout, proc=proc):
            stop_server(proc)
            raise RuntimeError(f"server did not become healthy within {timeout:.0f}s; see {out_dir / 'server.log'}")
        server["health_seconds"] = round(time.time() - t0, 1)
        print(f"[eval] healthy after {server['health_seconds']}s", flush=True)
    since = time.time()
    try:
        res = run_r3_eval(base_url, model_name, scenarios, out_dir, run_id)
        rows = collect_results(run_id, model_name, out_dir, database_url, since=since)
    finally:
        if proc is not None:
            stop_server(proc)
            server["stopped"] = True
    table = family_table(rows)
    table.update(model=model_name, model_dir=str(model_dir), adapter=str(adapter) if adapter else None, run_id=run_id,
                 scenarios=str(scenarios), runner=res["runner"], returncode=res["returncode"], server=server,
                 rows_source="postgres" if database_url and rows and "id" in rows[0] else "jsonl",
                 created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (out_dir / "family_table.json").write_text(json.dumps(table, indent=2, default=str) + "\n")
    md = f"# {model_name} on {scenarios} ({res['runner']})\n\n" + format_family_table(table) + "\n"
    if res["runner"] == "stub":
        md += "\n**STUB**: train/eval_stub.py ran, not r3-eval; the only scorer is `stub_server_responds`.\n"
    (out_dir / "family_table.md").write_text(md)
    print(md, flush=True)
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="versioned fused model dir ($R3_ARTIFACTS/models/<id>)")
    ap.add_argument("--model-name", default=None, help="served model id (default: the dir name)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--scenarios", default="eval/scenarios/test")
    ap.add_argument("--out", type=Path, default=None, help="default: <model-dir>/eval/<run-id>")
    ap.add_argument("--base-url", default=None, help="score an already running endpoint instead of starting one")
    ap.add_argument("--port", type=int, default=8010, help="port for the server this hook starts (not :8000, the live one)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--gate", action="store_true", help="also serve base + <model-dir>/adapter and apply the 1-point precision gate")
    ap.add_argument("--base", default=None, help="base model dir for --gate (default: r3.json base_model)")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--scorer", default=None, help="gate on one scorer (default: overall pass rate)")
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args(argv)
    name = a.model_name or a.model_dir.name
    run_id = a.run_id or f"eval-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = a.out or (a.model_dir / "eval" / run_id)
    db = None if a.no_db else "env"
    fused = evaluate_model_dir(a.model_dir, name, run_id, a.scenarios, out, base_url=a.base_url, port=a.port, database_url=db, timeout=a.timeout)
    if not a.gate:
        return 0
    meta = json.loads((a.model_dir / "r3.json").read_text()) if (a.model_dir / "r3.json").exists() else {}
    base = a.base or meta.get("base_model")
    if not base:
        raise SystemExit("--gate needs --base (no r3.json base_model)")
    adapter_table = evaluate_model_dir(a.model_dir, f"{name}-adapter", f"{run_id}-adapter", a.scenarios, out / "adapter",
                                       adapter=a.model_dir / "adapter", base=base, port=a.port, database_url=db, timeout=a.timeout)
    gate = precision_gate(fused, adapter_table, a.tolerance, a.scorer)
    (out / "precision_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"[gate] fused {gate['fused_pass_rate']} vs adapter-on-base {gate['adapter_pass_rate']} on {gate['scorer']}: "
          f"delta {gate['delta_points']:+.2f} points -> {'PASS' if gate['pass'] else 'FAIL'} (tolerance {a.tolerance})")
    return 0 if gate["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
