"""The training write-up: every run under $R3_ARTIFACTS/runs (run.json), its eval table when one
exists (runs/<id>/eval/family_table.json), and the candidate selection by the plan's rule.

    uv run python -m train.report --out $R3_ARTIFACTS/runs/TRAINING-REPORT.md [--table <family_table.json> ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from train.env import artifacts_dir
from train.select import select_candidates


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def ledger_routing_scores(run_id: str) -> dict:
    """routing.* rows that session S's batch job writes to cost_ledger for a routing model (run_id routing-<model id>)."""
    import os

    from train.env import load_env

    load_env()
    url = os.environ.get("DATABASE_URL")
    if not url:
        return {}
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("select item, qty from cost_ledger where run_id = %s and item like 'routing.%%'", (run_id,))
            return {item: float(qty) for item, qty in cur.fetchall()}
    except Exception:  # noqa: BLE001 - the report must render without Postgres
        return {}


def routing_rows(artifacts: Path, runs: list[dict], ledger_reader=ledger_routing_scores) -> list[dict]:
    """The routing row of the write-up: zero-shot baselines (E's scores.json) and every fused routing adapter
    (S's batch-job rows in the ledger), all on the same 100 test tickets."""
    rows = []
    for p in sorted((Path(artifacts) / "runs").glob("routing-*-zeroshot-*/scores.json")):
        d = _load(p) or {}
        rows.append({"model": d.get("model"), "kind": "zero-shot", "accuracy": d.get("accuracy"), "macro_f1": round(d["macro_f1"], 4) if d.get("macro_f1") is not None else None,
                     "coverage": d.get("coverage"), "weighted_error": d.get("weighted_error"), "n": d.get("n"), "source": str(p.parent.name)})
    for r in runs:
        if r.get("task") != "routing" or r.get("status") != "done":
            continue
        got = ledger_reader(f"routing-{r['model_id']}") or {}
        rows.append({"model": r["model_id"], "kind": "adapter", "accuracy": got.get("routing.accuracy"), "macro_f1": got.get("routing.macro_f1"),
                     "coverage": got.get("routing.coverage"), "weighted_error": got.get("routing.weighted_error"), "n": got.get("routing.n"),
                     "source": f"cost_ledger run_id routing-{r['model_id']}" if got else "not scored yet"})
    return rows


def build_report(artifacts: Path, extra_tables: list[Path] | None = None, tasks: tuple[str, ...] = ("voice", "routing", "rehearse"),
                 ledger_reader=ledger_routing_scores) -> dict:
    runs_dir = Path(artifacts) / "runs"
    runs, tables = [], []
    for d in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        rec = _load(d / "run.json")
        if not rec or rec.get("task") not in tasks or rec.get("task") == "rehearse":
            continue
        m = rec.get("metrics") or {}
        table = _load(d / "eval" / "family_table.json")
        runs.append({
            "run_id": rec.get("run_id"), "task": rec.get("task"), "status": rec.get("status"), "model_id": rec.get("model_id"),
            "model_dir": rec.get("model_dir"), "config": rec.get("config_path"), "git": (rec.get("git") or {}).get("short"),
            "data": (rec.get("data") or {}).get("dir"), "manifest_hash": ((rec.get("data") or {}).get("manifest_hash") or "")[:12],
            "n_train": ((rec.get("data") or {}).get("stats") or {}).get("n"), "iters": (rec.get("sizing") or {}).get("iters"),
            "optimizer_steps": (rec.get("sizing") or {}).get("optimizer_steps"), "cut": (rec.get("sizing") or {}).get("cut"),
            "wall_min": round((m.get("wall_seconds") or 0) / 60, 1), "tokens_per_sec": m.get("total_tokens_per_second"),
            "peak_gb": round(m["peak_memory_gb"], 1) if m.get("peak_memory_gb") is not None else None, "val_loss": [m.get("initial_val_loss"), m.get("final_val_loss")],
            "eval": table["selection"] if table else None, "error": rec.get("error"),
        })
        if table and table.get("selection"):
            tables.append(table)
    for p in extra_tables or []:
        t = _load(Path(p))
        if t and t.get("selection"):
            tables.append(t)
    ranked = select_candidates(tables) if tables else []
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "artifacts": str(artifacts), "runs": runs,
            "selection": {"selected": ranked[0] if ranked else None, "ranked": ranked},
            "routing": routing_rows(artifacts, runs, ledger_reader)}


def render_report(rep: dict) -> str:
    out = [f"# R3 training report ({rep['generated_at']})", "", "## Runs", "",
           "| run | task | status | model | data (n) | iters / opt steps | wall min | tokens/s | peak GB | val loss first -> last | eval: completion / tool calls / unsupported claims |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        ev = r["eval"]
        ev_txt = f"{ev['task_completion']} / {ev['tool_calls']} / {ev['unsupported_success_claims']}" if ev else "-"
        cut = f" (cut from {r['cut']['from']})" if r.get("cut") else ""
        vl = r["val_loss"]
        vl_txt = f"{vl[0]:.2f} -> {vl[1]:.2f}" if vl and vl[0] is not None and vl[1] is not None else "-"
        tps = f"{r['tokens_per_sec']:.0f}" if r.get("tokens_per_sec") else "-"
        out.append(f"| {r['run_id']} | {r['task']} | {r['status']}{(' (' + r['error'][:60] + ')') if r.get('error') else ''} | {r['model_id']} | "
                   f"{r['manifest_hash']} ({r['n_train']}) | {r['iters']}{cut} / {r['optimizer_steps']} | {r['wall_min']} | {tps} | "
                   f"{r['peak_gb'] if r['peak_gb'] is not None else '-'} | {vl_txt} | {ev_txt} |")
    out += ["", "## Candidates (test set, scored by r3-eval; zero unsupported success claims first, then task completion, then tool calls)", ""]
    if rep["selection"]["ranked"]:
        out += ["| rank | model | task completion | tool calls | unsupported claims | episodes |", "|---|---|---|---|---|---|"]
        out += [f"| {c['rank']} | {c['model']} | {c['task_completion']} | {c['tool_calls']} | {c['unsupported_success_claims']} | {c['episodes']} |"
                for c in rep["selection"]["ranked"]]
        s = rep["selection"]["selected"]
        out += ["", f"**Selected: {s['model']}** ({s['reason']}); model dir `{s.get('model_dir')}`."]
    else:
        out.append("No scored candidates yet.")
    if rep.get("routing"):
        out += ["", "## Routing (100 hand-check test tickets; adapter scored by the batch job, zero-shot by r3-eval)", "",
                "| model | kind | accuracy | macro F1 | coverage | weighted error | source |", "|---|---|---|---|---|---|---|"]
        for r in rep["routing"]:
            fmt = lambda v: "-" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
            out.append(f"| {r['model']} | {r['kind']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['coverage'])} | {fmt(r['weighted_error'])} | {r['source']} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--table", action="append", type=Path, default=[], help="extra family_table.json (e.g. the base model's)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    artifacts = Path(a.artifacts).expanduser() if a.artifacts else artifacts_dir()
    rep = build_report(artifacts, extra_tables=a.table)
    md = render_report(rep)
    print(md)
    if a.out:
        a.out.write_text(md)
        a.out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
