#!/usr/bin/env python
"""Assemble the write-up tables from Postgres and the run directories (plan T+22-24): eval runs per family (base,
candidate, hosted), latency per simulated-call batch (g1.* ledger rows), the HTTP load test, cost per call and per
successful task, routing (zero-shot summaries from E's runs + the batch job's ledger rows), and the targets table.

  uv run python scripts/writeup.py [--eval-run <run_id> ...] [--out path.md]
Defaults: every eval run whose id starts with baseline-, candidate- or eval- ; output $R3_ARTIFACTS/runs/writeup-<stamp>.md
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cost_report import eval_runs, live_calls, load_rates, price_group  # noqa: E402
from services.cost.report import mac_hourly_usd  # noqa: E402

ART = os.environ.get("R3_ARTIFACTS", os.path.expanduser("~/r3-artifacts"))
TARGETS = [  # (ledger metric, label, target text, lower is better)
    ("g1.eos_to_first_audio_heard.p50", "End of speech → first audio heard (caller clock) p50", "≤ 1000 ms"),
    ("g1.eos_to_first_audio_heard.p95", "End of speech → first audio heard p95", "≤ 1600 ms"),
    ("g1.eos_to_tts_audible.p50", "End of speech → first audible TTS (agent clock) p50", "≤ 1000 ms"),
    ("g1.transcription_delay.p50", "STT final after end of speech p50", "≤ 500 ms"),
    ("g1.llm_ttft_local.p50", "Model time to first token p50 (local)", "≤ 350 ms"),
    ("g1.tts_ttfb.p50", "TTS first byte p50", "≤ 200 ms"),
    ("g1.prompt_tokens.p50", "Prompt tokens per turn p50", "≤ 150 new tokens on turn 2+ (cache hit); total shown"),
    ("load_text.admission_limit", "Concurrent calls within budget", "4 proposed"),
]


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def fmt(v, nd=0):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def eval_table(con, run_ids: list[str]) -> str:
    """Per run: episodes passing every non-advisory scorer, by family (episode = detail.episode_id, else scenario_id)."""
    rows = con.execute("""
        with per_ep as (
          select run_id, model_id, family, coalesce(detail->>'episode_id', scenario_id) as ep,
                 bool_and(pass) filter (where coalesce(detail->>'advisory', 'false') = 'false') as ok
          from eval_results where run_id = any(%s) group by 1,2,3,4)
        select run_id, model_id, family, count(*), sum(ok::int) from per_ep group by 1,2,3 order by 1,3""", (run_ids,)).fetchall()
    if not rows:
        return "_no eval runs found_"
    fams = sorted({r[2] for r in rows}); runs = []
    data: dict[str, dict] = defaultdict(dict); model_of = {}
    for run_id, model, fam, n, ok in rows:
        data[run_id][fam] = (ok, n); model_of[run_id] = model
        if run_id not in runs:
            runs.append(run_id)
    header = ["run (model)"] + fams + ["all"]
    out = []
    for run_id in runs:
        cells = []; tot_ok = tot_n = 0
        for f in fams:
            ok, n = data[run_id].get(f, (0, 0)); tot_ok += ok or 0; tot_n += n
            cells.append(f"{ok}/{n}" if n else "-")
        caller = "caller: local base model" if "localcaller" in run_id else "caller: claude-sonnet-5-5"
        out.append([f"{run_id} ({model_of[run_id]}; {caller})"] + cells + [f"**{tot_ok}/{tot_n}** ({100 * tot_ok / tot_n:.0f}%)" if tot_n else "-"])
    return md_table(header, out) + ("\n\n_Cells: episodes passing every non-advisory scorer / episodes. Source: eval_results (E's runner). "
            "Runs marked 'caller: local base model' were scored with Qwen3-4B playing the caller while the hosted account was out of credits; "
            "that caller mis-states its own DOB or phone in about 20% of episodes, so the comparable numbers for those runs are E's "
            "`r3-eval compare --exclude-caller-faults` (same scenario x seed dropped from every run; Nov 1 run 1: 24% vs base 15% with the same local caller, 9% with the hosted caller, hosted 100%), "
            "never a raw row against a hosted-caller row._")


def ledger_metrics(con, run_like: str) -> dict[str, dict[str, float]]:
    rows = con.execute("select run_id, item, qty from cost_ledger where run_id like %s order by id", (run_like,)).fetchall()
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for run_id, item, qty in rows:
        out[run_id][item] = float(qty)  # the latest row wins
    return out


def latency_table(con) -> str:
    runs = {}
    for like in ("g1-%", "g2-%", "load-c%", "rehearsal-%-all"):  # rehearsal batches: the combined row only, not the six per-scenario rows
        runs.update(ledger_metrics(con, like))
    runs = {k: v for k, v in runs.items() if any(i.startswith("g1.") for i in v)}
    if not runs:
        return "_no simulated-call batches yet_"
    cols = [("g1.eos_to_first_audio_heard", "eos→audio heard"), ("g1.eos_to_tts_audible", "eos→TTS audible"), ("g1.transport", "transport"),
            ("g1.turn_taking_overhead", "turn-taking"), ("g1.llm_ttft_local", "LLM TTFT (local)"), ("g1.tts_ttfb", "TTS TTFB"), ("g1.prompt_tokens", "prompt tok")]
    header = ["batch", "calls (n turns)"] + [f"{c} p50/p95" for _, c in cols] + ["claims / loop guard / claim guard (calls)"]
    rows = []
    for run_id, m in sorted(runs.items()):
        n = m.get("g1.eos_to_first_audio_heard.n") or m.get("g1.llm_ttft_local.n")
        flags = f"{fmt(m.get('g1.unsupported_claim_calls'))} / {fmt(m.get('g1.loop_guard_calls'))} / {fmt(m.get('g1.claim_guard_calls'))} of {fmt(m.get('g1.calls'))}"
        rows.append([run_id, fmt(n)] + [f"{fmt(m.get(k + '.p50'))} / {fmt(m.get(k + '.p95'))}" for k, _ in cols] + [flags])
    return md_table(header, rows) + ("\n\n_Milliseconds (prompt tok = tokens). Caller-clock numbers include the Mac's network path; see RUNBOOK (hotspot). "
            "claims = calls with a spoken success claim and no matching tool result; loop guard = calls where the agent replaced a repeated line; "
            "claim guard = calls where the agent intervened on a claim the model made without a tool result: recorded request_handoff before an announced transfer, or held back a booked/moved/cancelled/ticket sentence and asked instead "
            "(both application interventions, identical for every model, never the model's own recovery). Known turn-taking limitation: the endpointing window "
            "(0.25-2.0 s) splits digit strings read with pauses into two caller turns; not tuned further in this sprint._")


def load_table(con) -> str:
    runs = ledger_metrics(con, "load-text-%")
    if not runs:
        return "_no HTTP load test yet_"
    rows = []
    for run_id, m in sorted(runs.items()):
        levels = sorted({int(re.match(r"load_text\.c(\d+)\.", k).group(1)) for k in m if re.match(r"load_text\.c(\d+)\.", k)})
        for c in levels:
            rows.append([run_id, c, fmt(m.get(f"load_text.c{c}.ttft_p50")), fmt(m.get(f"load_text.c{c}.ttft_p95")), fmt(m.get(f"load_text.c{c}.total_p50"))])
        rows.append([run_id, "admission limit", f"**{fmt(m.get('load_text.admission_limit'))}**", "", ""])
    return md_table(["run", "concurrent", "first token p50 ms", "p95 ms", "whole turn p50 ms"], rows) + "\n\n_Tool-call turns: vllm-mlx emits a tool call only when complete, so first token = whole turn; the SimpleEngine serializes requests._"


def cost_tables(con, rates: dict, eval_ids: list[str]) -> str:
    groups = live_calls(con, None)
    rows = [price_group(us, rates, m) for m, us in sorted(groups.items()) if us]
    parts = []
    if rows:
        parts.append(md_table(["served model", "calls", "ok", "min/call", "audio $/call", "LLM $/call", "total $/call", "$/successful task", "LLM basis"],
                              [[r["model"], r["calls"], r["successes"], fmt(r["avg_min"], 2), fmt(r["audio_usd"], 4), fmt(r["llm_usd"], 4), fmt(r["total_usd"], 4),
                                fmt(r["per_success_usd"], 4) if r["per_success_usd"] is not None else "n/a", r["llm_basis"]] for r in rows]))
    ev = eval_runs(con, eval_ids, rates)
    if ev:
        parts.append(md_table(["text-eval run", "model", "episodes", "passed", "LLM $", "$/episode", "$/successful episode", "basis"],
                              [[e["run_id"], e["model"], fmt(e["episodes"]), fmt(e["passed"]), fmt(e["llm_usd"], 4), fmt(e["per_episode_usd"], 4), fmt(e["per_success_usd"], 4) if e["per_success_usd"] else "n/a", e["basis"]] for e in ev]))
    parts.append(f"_Rates: cost_rates (list prices with source and date; Mac ${mac_hourly_usd(rates):.3f}/h = purchase / {rates['mac.amortization_hours']:.0f} h + power; admission limit {int(rates['admission.concurrent_calls'])}). Measured quantities, estimated bases marked._")
    return "\n\n".join(parts)


def rehearsal_table(con) -> str:
    """Per demo scenario from scripts/demo_check.py's ledger rows: passed/runs, calls where the loop or claim guard intervened, TTS vendor."""
    names = ("ordinary", "correction", "timeout", "interruption", "handoff", "handoff_person")
    runs: dict[str, dict[str, tuple[float, str]]] = {}
    for run_id, item, qty, note in con.execute("select run_id, item, qty, note from cost_ledger where run_id like 'rehearsal-%' and item like 'rehearsal.%' order by id").fetchall():
        runs.setdefault(run_id, {})[item] = (float(qty), note or "")
    rows = []
    for run_id, items in sorted(runs.items()):
        cells, vendors = [], set()
        for s in names:
            n = items.get(f"rehearsal.{s}.runs")
            if not n:
                cells.append("-")
                continue
            p = items.get(f"rehearsal.{s}.pass", (0, ""))[0]
            g = items.get(f"rehearsal.{s}.guarded", (0, ""))[0]
            cg = items.get(f"rehearsal.{s}.claim_guarded", (0, ""))[0]
            rt = items.get(f"rehearsal.{s}.retried", (0, ""))[0]
            po = items.get(f"rehearsal.{s}.policy", (0, ""))[0]
            marks = ([f"guard {int(g)}"] if g else []) + ([f"claim {int(cg)}"] if cg else []) + ([f"retry {int(rt)}"] if rt else []) + ([f"policy {int(po)}"] if po else [])
            cells.append(f"{int(p)}/{int(n[0])}" + (f" ({', '.join(marks)})" if marks else ""))
            m = re.search(r"tts=([^;]+)", n[1])
            if m:
                vendors.add(m.group(1))
        rows.append([run_id] + cells + ["/".join(sorted(vendors)) or "?"])
    note = ("\n\n_pass = expected final state reached with no success claim before its tool result, no ids spoken, no structural text; "
            "(guard n) = calls in which the agent's loop guard replaced a repeated line; (claim n) = calls in which the agent's claim guard intervened (recorded request_handoff before an announced transfer, or held back an unsupported booked/moved/cancelled/ticket sentence); (retry n) = calls in which the tool executor absorbed a transient fault with one retry on the same idempotency key, the model seeing only the final result; (policy n) = calls in which the application enforced the contract's fraud-ticket handoff rule after the model's reply, never the model's own transfer "
            "the model announced without the call; both application interventions counted apart from the model. "
            "Rows whose run id ends in -rules ran with the executor retry and the fraud-ticket policy rule on; earlier rows ran the model with the two guards only. "
            "Recordings per call under $R3_ARTIFACTS/runs/<rehearsal run>/audio; the simulated caller's recordings are labelled recorded, never live._")
    return (md_table(["rehearsal run (model, stamp)"] + list(names) + ["TTS"], rows) + note) if rows else "_no rehearsal yet_"


def routing_table(con) -> str:
    rows = []
    for path in sorted(glob.glob(os.path.join(ART, "runs", "routing-*zeroshot*", "summary.md"))):
        text = open(path).read()
        m = re.search(r"accuracy ([\d.]+) · macro F1 ([\d.]+) · coverage ([\d.]+) · weighted error (\d+)", text)
        title = re.search(r"`([^`]+)`", text)
        if m:
            rows.append([f"{title.group(1) if title else '?'} zero-shot ({os.path.basename(os.path.dirname(path))})", m.group(1), m.group(2), m.group(3), m.group(4), "E's runner"])
    for run_id, m in sorted(ledger_metrics(con, "routing-%").items()):
        if "routing.accuracy" in m:
            source = run_id[len("routing-"):] if run_id.startswith("routing-") else run_id
            kind = "base zero-shot" if "base" in source else "adapter"
            rows.append([f"{source} ({kind}, batch job)", fmt(m["routing.accuracy"], 3), fmt(m["routing.macro_f1"], 3), fmt(m["routing.coverage"], 3), fmt(m["routing.weighted_error"]), "services/routing/batch.py"])
    note = ("\n\n_Two decoding methods, compare rows within one: E's runner rows decode with a 64-token budget and a retry, parsed to one label; "
            "the batch-job rows (services/routing/batch.py) decode greedily with at most 8 tokens, validated against the taxonomy, invalid → needs_review. "
            "Same 100 hand-checked test tickets in both._")
    return md_table(["model", "accuracy", "macro F1", "coverage", "weighted error", "source"], rows) + note if rows else "_no routing results yet_"


def targets_table(con) -> str:
    base = ledger_metrics(con, "g1-sim-base").get("g1-sim-base", {})
    reh = ledger_metrics(con, "rehearsal-%-all")  # the final model's rehearsal, all calls combined; the latest batch wins
    reh_id, final = (list(reh.items())[-1] if reh else ("no rehearsal yet", {}))
    load = ledger_metrics(con, "load-text-%")
    adm = next((m.get("load_text.admission_limit") for m in load.values() if "load_text.admission_limit" in m), None)
    rows = []
    for key, label, target in TARGETS:
        if key == "load_text.admission_limit":
            rows.append([label, target, fmt(adm) if adm is not None else "-", fmt(adm) if adm is not None else "-"])
            continue
        b, f = base.get(key), final.get(key)
        rows.append([label, target, fmt(b) if b is not None else "-", fmt(f) if f is not None else "-"])
    return md_table(["measure", "proposed target", "measured (g1-sim-base, base model)", f"measured ({reh_id}, final model)"], rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-run", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        eval_ids = args.eval_run or [r[0] for r in con.execute("select distinct run_id from eval_results where run_id like 'baseline-%test%' or run_id like 'candidate-%' or run_id like 'eval-%' or run_id like '%-eval-%' order by 1").fetchall()]
        rates = load_rates(con)
        doc = [f"# R3 write-up tables — {time.strftime('%Y-%m-%d %H:%M')}", "",
               "## Task completion (eval runner, 27 hand-written test scenarios × repeats)", eval_table(con, eval_ids), "",
               "## Latency per simulated-call batch", latency_table(con), "",
               "## Load test (HTTP level)", load_table(con), "",
               "## Cost", cost_tables(con, rates, eval_ids), "",
               "## Ticket routing (100 hand-checked test tickets)", routing_table(con), "",
               "## Demo rehearsal (simulated caller, the five scripted scenarios plus the alternative handoff flow 5b; pass = expected final state, no claim without a tool result, no ids spoken)", rehearsal_table(con), "",
               "## Targets vs measured", targets_table(con), ""]
    out = pathlib.Path(args.out or os.path.join(ART, "runs", f"writeup-{time.strftime('%m%d-%H%M')}.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(doc))
    print("\n".join(doc)); print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
