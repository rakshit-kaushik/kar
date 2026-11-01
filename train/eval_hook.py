"""Eval-on-checkpoint: given a model directory, start the server, wait for health, run `r3-eval`
(session E's runner; train/eval_stub.py until it is merged), record the per-family table.

    uv run python -m train.eval_hook --model-dir $R3_ARTIFACTS/models/r3-voice-<hash> \
        --scenarios eval/scenarios/test --out $R3_ARTIFACTS/runs/<id>/eval --run-id <id>
    uv run python -m train.eval_hook --model-dir ... --gate      # precision gate: fused vs adapter-on-base

Servers (installed versions, flags read from their argparse and from scripts/restart.sh):
  fused 8-bit dir  -> vllm-mlx 0.5.0 with session S's production flags: `vllm-mlx serve <dir> --served-model-name
                      <id> --prefix-trie-cache --default-chat-template-kwargs '{"enable_thinking": false}'
                      --default-temperature 0.2 --max-tokens 256` (no continuous batching, no tool parser:
                      the proxy handles tool calls, as in production)
  adapter on base  -> mlx_lm.server 0.32.2: `mlx_lm.server --model <base> --adapter-path <dir>/adapter
                      --chat-template-args '{"enable_thinking": false}'` (vllm-mlx has no adapter flag);
                      requests must name the model `default_model`, so the runner is given that name
By default session S's proxy (services.proxy.main) runs on port+1 in front of the model server
(PROXY_UPSTREAM / PROXY_PORT, PROXY_FALLBACK=0 so no turn is silently answered by the hosted model)
and the runner scores through it, exactly like the live :8000/:8001 stack, on side ports; the model
server gets one warm-up request straight first (cold start ~19 s). Override the model server with $R3_EVAL_SERVER_CMD, a template with {model}
{port} {name} {adapter} placeholders. Results come from Postgres `eval_results` (run_id, else model_id
since the start) when the runner wrote there, else from <out>/eval_results.jsonl. Selection is by this
table on the test set, never by loss.
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
    "vllm-mlx serve {model} --host 127.0.0.1 --port {port} --served-model-name {name} --prefix-trie-cache "
    "--default-chat-template-kwargs '{chat_kwargs}' --default-temperature 0.2 --max-tokens 256"
)
DEFAULT_ADAPTER_CMD = (
    "mlx_lm.server --model {model} --adapter-path {adapter} --host 127.0.0.1 --port {port} "
    "--chat-template-args '{chat_kwargs}' --temp 0.2 --max-tokens 256"
)


def request_model_for(cmd: list[str], name: str) -> str:
    """The `model` the runner must send: mlx_lm.server maps only `default_model` to its --model
    (+ --adapter-path) and tries to download any other name as a Hugging Face repo."""
    return "default_model" if cmd and Path(cmd[0]).name == "mlx_lm.server" else name


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


def server_env() -> dict:
    """Environment for a side model server: vllm-mlx 0.5.0's simple engine must queue concurrent
    requests (VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait, as in scripts/restart.sh) instead of
    answering 503, or a runner at parallel 4 loses most episodes."""
    env = dict(os.environ)
    env.setdefault("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "wait")
    return env


def start_server(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write(f"$ VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait {' '.join(cmd)}\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=server_env())


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
    if os.environ.get("R3_EVAL_RUNNER"):
        return True
    try:
        return importlib.util.find_spec("eval.runner") is not None
    except (ImportError, ValueError):
        return False


def runner_command() -> tuple[list[str], Path]:
    """The r3-eval executable and the directory to run it from. $R3_EVAL_RUNNER (a command, e.g.
    `/path/to/e-worktree/.venv/bin/r3-eval`) with $R3_EVAL_RUNNER_CWD (default: that checkout) lets
    the hook drive session E's runner from E's worktree before e/sim reaches main."""
    custom = os.environ.get("R3_EVAL_RUNNER")
    if custom:
        cmd = shlex.split(custom)
        cwd = Path(os.environ.get("R3_EVAL_RUNNER_CWD") or REPO_ROOT)
        return cmd, cwd
    exe = Path(sys.executable).with_name("r3-eval")
    return [str(exe) if exe.exists() else "r3-eval", "run"], REPO_ROOT  # E's CLI: `r3-eval run ...`


def run_r3_eval(base_url: str, model: str, scenarios: str | Path, out_dir: str | Path, run_id: str,
                timeout: float = 3600.0, database_url: str | None = None) -> dict:
    """`r3-eval --base-url --model --scenarios --out --run-id [--no-db]` (eval/runner.py's CLI), or
    train/eval_stub.py with the same arguments when the runner is not importable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = ["--base-url", base_url, "--model", model, "--scenarios", str(scenarios), "--out", str(out_dir), "--run-id", run_id]
    if not database_url:
        common.append("--no-db")
    cwd = REPO_ROOT
    if runner_available():
        exe, cwd = runner_command()
        cmd = [*exe, *common]
        runner = "r3-eval"
    else:
        cmd = [sys.executable, "-m", "train.eval_stub", *common]
        runner = "stub"
    log = out_dir / "eval.log"
    print(f"[eval] {runner} (cwd {cwd}): {' '.join(cmd)}", flush=True)
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return {"runner": runner, "command": cmd, "cwd": str(cwd), "returncode": proc.returncode, "log": str(log)}


def collect_results(run_id: str, model_id: str, out_dir: str | Path, database_url: str | None, since: float | None = None) -> list[dict]:
    rows: list[dict] = []
    if database_url:
        try:
            import psycopg

            with psycopg.connect(database_url, connect_timeout=5) as conn, conn.cursor() as cur:
                cur.execute(
                    "select run_id, model_id, scenario_id, family, scorer, pass, detail from eval_results "
                    "where run_id = %s or (model_id = %s and created_at >= %s) order by id",
                    (run_id, model_id, dt.datetime.fromtimestamp(since or 0, dt.timezone.utc)),
                )
                rows = [dict(zip(("run_id", "model_id", "scenario_id", "family", "scorer", "pass", "detail"), r)) for r in cur.fetchall()]
        except Exception as e:  # noqa: BLE001
            print(f"[eval] postgres eval_results unavailable ({type(e).__name__}: {e}); using jsonl", flush=True)
    if not rows:
        for name in ("scores.jsonl", "eval_results.jsonl"):  # E's runner writes scores.jsonl; the stub eval_results.jsonl
            p = Path(out_dir) / name
            if p.exists():
                rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                mine = [r for r in rows if r.get("run_id") == run_id]
                rows = mine or rows
                break
    return rows


def proxy_env(upstream_port: int, proxy_port: int) -> dict:
    """Environment for a side proxy: upstream = the side model server, hosted fallback OFF (an eval
    that silently fell back to the hosted model would not be scoring the candidate)."""
    return {**os.environ, "PROXY_UPSTREAM": f"http://127.0.0.1:{upstream_port}", "PROXY_PORT": str(proxy_port), "PROXY_FALLBACK": "0"}


def warm_up(model_url: str, request_model: str, timeout: float = 180.0) -> dict:
    """One chat completion with the contract tools straight at the model server before scoring: the
    first request after a start takes ~19 s (engine warmup + system-prompt cache) and would otherwise
    hit the runner's first turn."""
    from train.eval_stub import probe

    t0 = time.time()
    ok, detail = probe(model_url, request_model, timeout=timeout)
    return {"ok": ok, "ms": round(1000 * (time.time() - t0)), "url": model_url, **({} if ok else {"error": detail.get("error")})}


def start_proxy(upstream_port: int, proxy_port: int, log_path: Path) -> subprocess.Popen:
    """Session S's proxy (services/proxy) on a side port, fronting the side model server."""
    env = proxy_env(upstream_port, proxy_port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    log.write(f"$ PROXY_UPSTREAM={env['PROXY_UPSTREAM']} PROXY_PORT={proxy_port} python -m services.proxy.main\n")
    log.flush()
    return subprocess.Popen([sys.executable, "-m", "services.proxy.main"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def _rate(passed: int, n: int) -> float:
    return round(100.0 * passed / n, 2) if n else 0.0


TASK_COMPLETION = "final_state"
TOOL_ARGS = "tool_calls"
SUCCESS_CLAIM = "no_success_claim_before_result"


def family_table(rows: list[dict]) -> dict:
    """Pass rates per family and scorer over the non-advisory rows, plus the plan's selection metrics:
    task completion (final_state), tool-argument accuracy (tool_calls) and the count of unsupported
    success claims (failed no_success_claim_before_result rows). Advisory rows are counted apart."""
    advisory = [r for r in rows if r.get("advisory")]
    rows = [r for r in rows if not r.get("advisory")]
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
    table = {"overall": {"n": n, "passed": passed, "pass_rate": _rate(passed, n)}, "families": dict(sorted(fams.items())),
             "advisory": {"n": len(advisory), "failed": sum(1 for r in advisory if not r.get("pass"))}}
    table["selection"] = {
        "task_completion": _scorer_rate(table, TASK_COMPLETION) if _has_scorer(table, TASK_COMPLETION) else None,
        "tool_calls": _scorer_rate(table, TOOL_ARGS) if _has_scorer(table, TOOL_ARGS) else None,
        "unsupported_success_claims": sum(1 for r in rows if r.get("scorer") == SUCCESS_CLAIM and not r.get("pass")),
        "episodes": len({r.get("episode_id") or (r.get("scenario_id"), r.get("repeat_seed")) for r in rows}),
    }
    return table


def _has_scorer(table: dict, scorer: str) -> bool:
    return any(scorer in f["scorers"] for f in table["families"].values())


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
    sel = table.get("selection")
    if sel:
        lines.append("")
        lines.append(f"selection: task completion {sel['task_completion']} · tool calls {sel['tool_calls']} · "
                     f"unsupported success claims {sel['unsupported_success_claims']} · episodes {sel['episodes']}"
                     + (f" · advisory failures {table['advisory']['failed']}/{table['advisory']['n']}" if table.get("advisory") else ""))
    return "\n".join(lines)


def _scorer_rate(table: dict, scorer: str | None) -> float:
    if not scorer:
        return table["overall"]["pass_rate"]
    n = sum(f["scorers"].get(scorer, {}).get("n", 0) for f in table["families"].values())
    passed = sum(f["scorers"].get(scorer, {}).get("passed", 0) for f in table["families"].values())
    return _rate(passed, n)


def precision_gate(fused: dict, adapter: dict, tolerance_points: float = 1.0, scorer: str | None = None) -> dict:
    """G3: the fused, re-quantized artifact must be within `tolerance_points` of the unfused adapter on
    the 8-bit base (the configuration that was trained), on task completion (final_state) when the
    runner scored it. Better-than-adapter passes."""
    if scorer is None and _has_scorer(fused, TASK_COMPLETION) and _has_scorer(adapter, TASK_COMPLETION):
        scorer = TASK_COMPLETION
    f, a = _scorer_rate(fused, scorer), _scorer_rate(adapter, scorer)
    delta = round(f - a, 2)
    return {"scorer": scorer or "overall", "fused_pass_rate": f, "adapter_pass_rate": a, "delta_points": delta,
            "tolerance_points": tolerance_points, "pass": delta >= -tolerance_points}


def evaluate_model_dir(model_dir: str | Path, model_name: str, run_id: str, scenarios: str | Path, out_dir: str | Path,
                       base_url: str | None = None, adapter: str | Path | None = None, base: str | Path | None = None,
                       port: int = 8010, database_url: str | None = "env", timeout: float = 600.0,
                       through_proxy: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if database_url == "env":
        load_env()
        database_url = os.environ.get("DATABASE_URL")
    server: dict = {"started": False, "stopped": False, "command": None, "base_url": base_url, "proxy": None}
    request_model = model_name
    proc = proxy = None
    if base_url is None:
        cmd = server_command(model=str(base or model_dir), port=port, name=model_name, adapter=str(adapter) if adapter else None)
        request_model = request_model_for(cmd, model_name)
        model_url = f"http://127.0.0.1:{port}/v1"
        server.update(command=cmd, model_url=model_url, log=str(out_dir / "server.log"))
        print(f"[eval] starting server: {' '.join(cmd)}", flush=True)
        proc = start_server(cmd, out_dir / "server.log")
        server["started"] = True
        t0 = time.time()
        if not wait_for_health(model_url, timeout=timeout, proc=proc):
            stop_server(proc)
            raise RuntimeError(f"server did not become healthy within {timeout:.0f}s; see {out_dir / 'server.log'}")
        server["health_seconds"] = round(time.time() - t0, 1)
        print(f"[eval] healthy after {server['health_seconds']}s", flush=True)
        server["warmup"] = warm_up(model_url, request_model)
        print(f"[eval] warm-up straight at the server: {'ok' if server['warmup']['ok'] else 'FAILED'} in {server['warmup']['ms']} ms", flush=True)
        if not server["warmup"]["ok"]:
            stop_server(proc)
            raise RuntimeError(f"warm-up request failed: {server['warmup'].get('error')}; see {out_dir / 'server.log'}")
        base_url = model_url
        if through_proxy:
            proxy_port = port + 1
            proxy = start_proxy(port, proxy_port, out_dir / "proxy.log")
            proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
            server["proxy"] = {"port": proxy_port, "base_url": proxy_url, "started": True, "stopped": False, "log": str(out_dir / "proxy.log")}
            if not wait_for_health(proxy_url, timeout=60, proc=proxy):
                stop_server(proxy)
                stop_server(proc)
                raise RuntimeError(f"proxy did not become healthy on :{proxy_port}; see {out_dir / 'proxy.log'}")
            base_url = proxy_url
            print(f"[eval] proxy on :{proxy_port} -> :{port}; scoring through the proxy", flush=True)
        server["base_url"] = base_url
    since = time.time()
    try:
        res = run_r3_eval(base_url, request_model, scenarios, out_dir, run_id, database_url=database_url)
        rows = collect_results(run_id, request_model, out_dir, database_url, since=since)
    finally:
        if proxy is not None:
            stop_server(proxy)
            server["proxy"]["stopped"] = True
        if proc is not None:
            stop_server(proc)
            server["stopped"] = True
    table = family_table(rows)
    table.update(model=model_name, request_model=request_model, model_dir=str(model_dir), adapter=str(adapter) if adapter else None, run_id=run_id,
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


def serve_side_stack(model_dir: str | Path, model_name: str, port: int, out_dir: str | Path, minutes: float = 60.0,
                     adapter: str | Path | None = None, base: str | Path | None = None, poll_seconds: float = 5.0,
                     timeout: float = 600.0) -> dict:
    """Keep a model server + S's proxy (fallback off) up on side ports for another session's eval:
    writes <out>/serving.json (urls, pids) once warm, then waits until <out>/STOP exists or `minutes`
    pass, and stops both. Hold the GPU lock around it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    cmd = server_command(model=str(base or model_dir), port=port, name=model_name, adapter=str(adapter) if adapter else None)
    request_model = request_model_for(cmd, model_name)
    model_url = f"http://127.0.0.1:{port}/v1"
    proxy_port = port + 1
    proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    info = {"model_dir": str(model_dir), "model_name": model_name, "request_model": request_model, "command": cmd,
            "model_url": model_url, "proxy": {"port": proxy_port, "base_url": proxy_url}, "base_url": proxy_url,
            "out_dir": str(out_dir), "stop_file": str(stop_file), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    print(f"[serve] {' '.join(cmd)}", flush=True)
    proc = start_server(cmd, out_dir / "server.log")
    proxy = None
    try:
        if not wait_for_health(model_url, timeout=timeout, proc=proc):
            raise RuntimeError(f"model server not healthy within {timeout:.0f}s; see {out_dir / 'server.log'}")
        info["warmup"] = warm_up(model_url, request_model)
        if not info["warmup"]["ok"]:
            raise RuntimeError(f"warm-up failed: {info['warmup'].get('error')}")
        proxy = start_proxy(port, proxy_port, out_dir / "proxy.log")
        if not wait_for_health(proxy_url, timeout=60, proc=proxy):
            raise RuntimeError(f"proxy not healthy on :{proxy_port}; see {out_dir / 'proxy.log'}")
        info.update(model_pid=proc.pid, proxy_pid=proxy.pid, ready_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        (out_dir / "serving.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"[serve] up: score at {proxy_url} with model {request_model!r}; stop with `touch {stop_file}` or after {minutes:g} min", flush=True)
        deadline = time.time() + minutes * 60
        info["stopped_by"] = "timeout"
        while time.time() < deadline:
            if stop_file.exists():
                info["stopped_by"] = "stop_file"
                break
            if proc.poll() is not None or proxy.poll() is not None:
                info["stopped_by"] = "process_exit"
                break
            time.sleep(poll_seconds)
    finally:
        if proxy is not None:
            stop_server(proxy)
        stop_server(proc)
        info["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        (out_dir / "serving.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"[serve] down ({info.get('stopped_by', 'error')})", flush=True)
    return info


def excluded_episode_prefixes(compare_file: str | Path) -> list[str]:
    """The `<scenario>-s<seed>` ids E's compare-excl-caller-faults.md lists on its first line (episodes in
    which the simulated caller mis-stated its own credentials); rows whose episode_id starts with one are
    dropped from every candidate alike."""
    first = Path(compare_file).read_text().splitlines()[0] if Path(compare_file).exists() else ""
    if ":" not in first:
        return []
    return [tok.strip().rstrip(".") for tok in first.split(":", 1)[1].split(",") if tok.strip()]


def table_from_scores(run_dir: str | Path, model_name: str | None = None, run_id: str | None = None,
                      exclude_file: str | Path | None = None) -> dict:
    """family_table.{json,md} for a run directory the runner already wrote (e.g. E's base baseline),
    so it can join the candidate ranking (train/select.py, train/report.py). With `exclude_file`
    (E's compare-excl-caller-faults.md) the caller-fault episodes are dropped and the table is written
    as family_table_excl.{json,md}: the selection rule is applied to that view."""
    run_dir = Path(run_dir)
    rows = collect_results(run_id or "", model_name or "", run_dir, database_url=None)
    if not rows:
        raise FileNotFoundError(f"no scores.jsonl / eval_results.jsonl in {run_dir}")
    rid = run_id or rows[0].get("run_id")
    name = model_name or rows[0].get("model_id")
    excluded = None
    if exclude_file:
        prefixes = excluded_episode_prefixes(exclude_file)
        before = len(rows)
        rows = [r for r in rows if not any(str(r.get("episode_id") or "").startswith(p + "-") or str(r.get("episode_id") or "") == p for p in prefixes)]
        excluded = {"file": str(exclude_file), "prefixes": prefixes, "rows_dropped": before - len(rows)}
    table = family_table(rows)
    table.update(model=name, request_model=name, model_dir=None, adapter=None, run_id=rid, scenarios=None, runner="r3-eval",
                 returncode=0, server=None, rows_source=str(run_dir), excluded=excluded,
                 created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    stem = "family_table_excl" if excluded else "family_table"
    (run_dir / f"{stem}.json").write_text(json.dumps(table, indent=2, default=str) + "\n")
    (run_dir / f"{stem}.md").write_text(f"# {name} ({rid})" + (f" excluding {len(excluded['prefixes'])} caller-fault episodes" if excluded else "") + "\n\n"
                                       + format_family_table(table) + "\n")
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--from-scores"]:  # family table from an existing runner output: --from-scores <run dir> [--model-name N]
        sub = argparse.ArgumentParser()
        sub.add_argument("--from-scores", type=Path, required=True)
        sub.add_argument("--model-name", default=None)
        sub.add_argument("--run-id", default=None)
        sub.add_argument("--exclude-from", type=Path, default=None, help="E's compare-excl-caller-faults.md")
        a = sub.parse_args(argv)
        t = table_from_scores(a.from_scores, a.model_name, a.run_id, exclude_file=a.exclude_from)
        print(format_family_table(t))
        return 0
    ap.add_argument("--model-dir", required=True, type=Path, help="versioned fused model dir ($R3_ARTIFACTS/models/<id>)")
    ap.add_argument("--model-name", default=None, help="served model id (default: the dir name)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--scenarios", default="eval/scenarios/test")
    ap.add_argument("--out", type=Path, default=None, help="default: <model-dir>/eval/<run-id>")
    ap.add_argument("--base-url", default=None, help="score an already running endpoint instead of starting one")
    ap.add_argument("--port", type=int, default=8010, help="side port for the model server (the proxy takes port+1); the live stack stays on :8000/:8001")
    ap.add_argument("--no-proxy", action="store_true", help="score the model server directly instead of through services/proxy")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--gate", action="store_true", help="also serve base + <model-dir>/adapter and apply the 1-point precision gate")
    ap.add_argument("--base", default=None, help="base model dir for --gate (default: r3.json base_model)")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--scorer", default=None, help="gate on one scorer (default: overall pass rate)")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--serve-only", action="store_true", help="keep the side stack up for another session's eval (no scoring)")
    ap.add_argument("--minutes", type=float, default=60.0, help="--serve-only: stop after this many minutes (or when <out>/STOP exists)")
    a = ap.parse_args(argv)
    name = a.model_name or a.model_dir.name
    if a.serve_only:
        out = a.out or (a.model_dir / "serve")
        serve_side_stack(a.model_dir, name, port=a.port, out_dir=out, minutes=a.minutes, timeout=a.timeout)
        return 0
    run_id = a.run_id or f"eval-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = a.out or (a.model_dir / "eval" / run_id)
    db = None if a.no_db else "env"
    fused = evaluate_model_dir(a.model_dir, name, run_id, a.scenarios, out, base_url=a.base_url, port=a.port, database_url=db,
                               timeout=a.timeout, through_proxy=not a.no_proxy)
    if not a.gate:
        return 0
    meta = json.loads((a.model_dir / "r3.json").read_text()) if (a.model_dir / "r3.json").exists() else {}
    base = a.base or meta.get("base_model")
    if not base:
        raise SystemExit("--gate needs --base (no r3.json base_model)")
    adapter_table = evaluate_model_dir(a.model_dir, f"{name}-adapter", f"{run_id}-adapter", a.scenarios, out / "adapter",
                                       adapter=a.model_dir / "adapter", base=base, port=a.port, database_url=db, timeout=a.timeout,
                                       through_proxy=not a.no_proxy)
    gate = precision_gate(fused, adapter_table, a.tolerance, a.scorer)
    (out / "precision_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"[gate] fused {gate['fused_pass_rate']} vs adapter-on-base {gate['adapter_pass_rate']} on {gate['scorer']}: "
          f"delta {gate['delta_points']:+.2f} points -> {'PASS' if gate['pass'] else 'FAIL'} (tolerance {a.tolerance})")
    return 0 if gate["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
