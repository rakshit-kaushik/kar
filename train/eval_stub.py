"""Stand-in for `r3-eval` (eval/runner.py, session E) until that runner is merged.

Same CLI shape as CONTRACT.md §7:
    r3-eval --base-url <url> --model <id> --scenarios <dir> --out <dir>   (+ --run-id here)
Writes <out>/eval_results.jsonl rows (run_id, model_id, scenario_id, family, scorer, pass, detail)
and <out>/summary.md. The only thing it can score is whether the endpoint answers one chat
completion that carries the contract's tools (scorer `stub_server_responds`); every row has
detail.stub = true so a stub table can never pass for a real one. Never touches Postgres.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import datetime as dt

from train.env import repo_root
from train.prompt import render_system_prompt

SCORER = "stub_server_responds"


def list_scenarios(scenarios_dir: str | Path) -> list[dict]:
    d = Path(scenarios_dir)
    out = []
    for p in sorted(d.glob("*")) if d.exists() else []:
        if p.suffix not in (".json", ".yaml", ".yml") or p.name.startswith("."):
            continue
        data: dict = {}
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text()) or {}
            else:
                import yaml

                data = yaml.safe_load(p.read_text()) or {}
        except Exception:  # unreadable scenario: still a row, family unknown
            data = {}
        if not isinstance(data, dict):
            data = {}
        out.append({"scenario_id": str(data.get("id") or p.stem), "family": str(data.get("family") or p.stem.split("__")[0]), "path": str(p)})
    return out


def probe(base_url: str, model: str, timeout: float = 30.0) -> tuple[bool, dict]:
    tools = [json.loads(p.read_text()) for p in sorted((repo_root() / "schemas" / "tools").glob("*.json"))]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": render_system_prompt(dt.datetime.now())},
                     {"role": "user", "content": "yeah it's five five five zero one four two two one nine"}],
        "tools": tools, "max_tokens": 48, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        choice = (payload.get("choices") or [{}])[0].get("message", {})
        return True, {"ms": round(1000 * (time.perf_counter() - t0)), "content": (choice.get("content") or "")[:120],
                      "tool_calls": len(choice.get("tool_calls") or [])}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--no-db", action="store_true", help="accepted for CLI parity with r3-eval; the stub never writes to Postgres")
    a = ap.parse_args(argv)
    run_id = a.run_id or f"stub-{int(time.time())}"
    a.out.mkdir(parents=True, exist_ok=True)
    scenarios = list_scenarios(a.scenarios) or [{"scenario_id": "none", "family": "stub", "path": None}]
    ok, detail = probe(a.base_url, a.model, a.timeout)
    rows = [{"run_id": run_id, "model_id": a.model, "scenario_id": s["scenario_id"], "family": s["family"],
             "scorer": SCORER, "pass": bool(ok), "detail": {"stub": True, "note": "r3-eval not merged; train/eval_stub.py", **detail}}
            for s in scenarios]
    with (a.out / "eval_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fams = sorted({r["family"] for r in rows})
    md = ["# STUB eval (not a real score)", "", f"model `{a.model}` at `{a.base_url}`; server responded: **{ok}** {detail}", "",
          "| family | scenarios | server_responds |", "|---|---|---|"]
    md += [f"| {fam} | {sum(1 for r in rows if r['family'] == fam)} | {'yes' if ok else 'no'} |" for fam in fams]
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
