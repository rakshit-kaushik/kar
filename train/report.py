"""The training write-up: every run under $R3_ARTIFACTS/runs (run.json), its eval table when one
exists (runs/<id>/eval/family_table_excl.json when E's caller-fault exclusion was applied, else family_table.json),
and the candidate selection by the G4 rule on one basis for every model.

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


def _newest(a: dict | None, b: dict) -> dict:
    """Of two tables for the same model, the one written last (created_at, else the file's mtime recorded at load)."""
    if a is None:
        return b
    return b if (b.get("created_at") or "", b.get("_mtime") or 0) >= (a.get("created_at") or "", a.get("_mtime") or 0) else a


def _load_table(p: Path) -> dict | None:
    t = _load(p)
    if t and t.get("selection"):
        t["_mtime"] = p.stat().st_mtime
        t["_path"] = str(p)
    return t


def gather_tables(runs_dir: Path, extra_tables: list[Path] | None = None) -> tuple[list[dict], str]:
    """Every scored candidate under runs/*/eval, one table per model (the newest), all on the same basis:
    the caller-fault-excluded tables (family_table_excl.json, E's compare --exclude-caller-faults applied) when any
    exist, else the raw tables. Mixing the two bases would compare different episode sets."""
    excl: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    for d in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        for name, bucket in (("family_table_excl.json", excl), ("family_table.json", raw)):
            t = _load_table(d / "eval" / name)
            if t:
                bucket[t.get("model")] = _newest(bucket.get(t.get("model")), t)
    for p in extra_tables or []:
        t = _load_table(Path(p))
        if t:
            bucket = excl if t.get("excluded") else raw
            bucket[t.get("model")] = _newest(bucket.get(t.get("model")), t)
    if excl:
        return list(excl.values()), "caller-fault-excluded"
    return list(raw.values()), "raw"


def build_report(artifacts: Path, extra_tables: list[Path] | None = None, tasks: tuple[str, ...] = ("voice", "routing", "rehearse"),
                 ledger_reader=ledger_routing_scores) -> dict:
    runs_dir = Path(artifacts) / "runs"
    tables, basis = gather_tables(runs_dir, extra_tables)
    for t in tables:  # tables rebuilt from scores.jsonl carry no model dir; the fused artifact lives under models/<model id>
        if not t.get("model_dir") and (Path(artifacts) / "models" / str(t.get("model"))).is_dir():
            t["model_dir"] = str(Path(artifacts) / "models" / str(t.get("model")))
    by_model = {t.get("model"): t for t in tables}
    runs = []
    for d in sorted(runs_dir.glob("*")) if runs_dir.exists() else []:
        rec = _load(d / "run.json")
        if not rec or rec.get("task") not in tasks or rec.get("task") == "rehearse":
            continue
        m = rec.get("metrics") or {}
        table = by_model.get(rec.get("model_id"))
        runs.append({
            "run_id": rec.get("run_id"), "task": rec.get("task"), "status": rec.get("status"), "model_id": rec.get("model_id"),
            "model_dir": rec.get("model_dir"), "config": rec.get("config_path"), "git": (rec.get("git") or {}).get("short"),
            "data": (rec.get("data") or {}).get("dir"), "manifest_hash": ((rec.get("data") or {}).get("manifest_hash") or "")[:12],
            "n_train": ((rec.get("data") or {}).get("stats") or {}).get("n"), "iters": (rec.get("sizing") or {}).get("iters"),
            "optimizer_steps": (rec.get("sizing") or {}).get("optimizer_steps"), "cut": (rec.get("sizing") or {}).get("cut"),
            "wall_min": round((m.get("wall_seconds") or 0) / 60, 1), "tokens_per_sec": m.get("total_tokens_per_second"),
            "peak_gb": round(m["peak_memory_gb"], 1) if m.get("peak_memory_gb") is not None else None, "val_loss": [m.get("initial_val_loss"), m.get("final_val_loss")],
            "eval": table["selection"] if table else None, "eval_run_id": table.get("run_id") if table else None, "error": rec.get("error"),
        })
    ranked = select_candidates(tables) if tables else []
    excluded = next((t.get("excluded") for t in tables if t.get("excluded")), None) if basis == "caller-fault-excluded" else None
    gate = None
    if ranked:  # the precision gate written next to the selected model's eval table (paired when eval_hook --paired-gate ran)
        sel_table = by_model.get(ranked[0]["model"])
        gp = Path(sel_table["_path"]).with_name("precision_gate.json") if sel_table and sel_table.get("_path") else None
        gate = _load(gp) if gp and gp.exists() else None
    return {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "artifacts": str(artifacts), "runs": runs,
            "selection": {"selected": ranked[0] if ranked else None, "ranked": ranked, "basis": basis, "gate": gate,
                          "excluded_prefixes": len((excluded or {}).get("prefixes") or []), "exclude_file": (excluded or {}).get("file"),
                          "tables": [{"model": t.get("model"), "run_id": t.get("run_id"), "path": t.get("_path")} for t in tables]},
            "routing": routing_rows(artifacts, runs, ledger_reader)}


def render_report(rep: dict) -> str:
    out = [f"# R3 training report ({rep['generated_at']})", "", "## Runs", "",
           "| run | task | status | model | data (n) | iters / opt steps | wall min | tokens/s | peak GB | val loss first -> last | eval: all-pass / completion / tool calls / unsupported claims |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        ev = r["eval"]
        ev_txt = f"{ev.get('all_pass', '-')} / {ev['task_completion']} / {ev['tool_calls']} / {ev['unsupported_success_claims']}" if ev else "-"
        cut = f" (cut from {r['cut']['from']})" if r.get("cut") else ""
        vl = r["val_loss"]
        vl_txt = f"{vl[0]:.2f} -> {vl[1]:.2f}" if vl and vl[0] is not None and vl[1] is not None else "-"
        tps = f"{r['tokens_per_sec']:.0f}" if r.get("tokens_per_sec") else "-"
        out.append(f"| {r['run_id']} | {r['task']} | {r['status']}{(' (' + r['error'][:60] + ')') if r.get('error') else ''} | {r['model_id']} | "
                   f"{r['manifest_hash']} ({r['n_train']}) | {r['iters']}{cut} / {r['optimizer_steps']} | {r['wall_min']} | {tps} | "
                   f"{r['peak_gb'] if r['peak_gb'] is not None else '-'} | {vl_txt} | {ev_txt} |")
    sel = rep["selection"]
    basis = ("caller-fault-excluded episodes (E's compare --exclude-caller-faults; " + f"{sel.get('excluded_prefixes')} scenario x seed episodes dropped)"
             if sel.get("basis") == "caller-fault-excluded" else "all episodes (no caller-fault exclusion applied)")
    out += ["", "## Candidates (test set, scored by r3-eval; G4 rule: all-pass rate first, unsupported success claims as the tie-breaker toward zero, "
            "then task completion, then tool calls; never loss)", "", f"Basis: {basis}. One table per model, the newest.", ""]
    if sel["ranked"]:
        out += ["| rank | model | all-pass | unsupported claims | task completion | tool calls | episodes | eval run |", "|---|---|---|---|---|---|---|---|"]
        rid = {t["model"]: t["run_id"] for t in sel.get("tables", [])}
        out += [f"| {c['rank']} | {c['model']} | {c.get('all_pass') if c.get('all_pass') is not None else '-'} | {c['unsupported_success_claims']} | "
                f"{c['task_completion']} | {c['tool_calls']} | {c['episodes']} | {rid.get(c['model']) or '-'} |"
                for c in sel["ranked"]]
        s = sel["selected"]
        out += ["", f"**Selected: {s['model']}** ({s['reason']}); model dir `{s.get('model_dir')}`."]
        g = sel.get("gate")
        if g and g.get("paired"):
            out += ["", f"Precision gate (paired on {g['episodes_paired']} scenario x seed episodes both runs scored, same exclusions): fused final_state "
                    f"{g['fused']['final_state']} vs unfused adapter on the base {g['adapter']['final_state']}, all-pass {g['fused']['all_pass']} vs "
                    f"{g['adapter']['all_pass']}, unsupported claims {g['fused']['unsupported_success_claims']} vs {g['adapter']['unsupported_success_claims']}: "
                    f"**{'PASS' if g['pass'] else 'FAIL'}** ({g['rule']})."]
        elif g:
            out += ["", f"Precision gate (unpaired, {g.get('scorer')}): fused {g.get('fused_pass_rate')} vs adapter {g.get('adapter_pass_rate')}: "
                    f"**{'PASS' if g.get('pass') else 'FAIL'}**."]
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
