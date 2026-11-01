"""Cost per successfully completed task, for every model path, from measured usage (plan: Demo/economics).

  uv run python scripts/cost_report.py                       # live calls: all calls with a call_end, grouped by model path
  uv run python scripts/cost_report.py --like 'r3-g1-%'      # one batch of calls
  uv run python scripts/cost_report.py --eval-run baseline-hosted-text-test-1 --eval-run baseline-base-text-test-1
  uv run python scripts/cost_report.py --set-admission 4     # record the load test's admission limit (cost_rates, kind=measured)
  uv run python scripts/cost_report.py --ledger --run-id cost-1   # write cost.* rows to cost_ledger

Audio vendors are priced from measured minutes and characters at list price (cost_rates, kind=list). The LLM on the
local path is the Mac's hourly cost divided by calls per hour at the admission limit (estimated, kind=assumption);
the hosted path is measured tokens at list price. Text-eval runs (E's ledger rows) are priced the same way per episode.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.cost.report import HOSTED_MODEL, call_usage, local_llm_seconds, mac_hourly_usd, price_call  # noqa: E402


def load_rates(con) -> dict:
    return {k: float(v) for k, v in con.execute("select key, value from cost_rates").fetchall()}


def path_of(model_id: str) -> str:
    return "hosted" if model_id == HOSTED_MODEL else model_id


def live_calls(con, like: str | None) -> dict[str, list[dict]]:
    """Per served model (sessions.model_id): priced usage per call that reached call_end."""
    q = "select call_id, model_id from sessions where (ended_at is not null or call_id in (select call_id from turn_events where event='call_end'))"
    args: tuple = ()
    if like:
        q += " and call_id like %s"; args = (like,)
    calls = con.execute(q, args).fetchall()
    out: dict[str, list[dict]] = defaultdict(list)
    for call_id, model_id in calls:
        rows = con.execute("select call_id, turn_idx, model_id, event, ts_ms, payload from turn_events where call_id=%s order by ts_ms, id", (call_id,)).fetchall()
        if not rows:
            continue
        u = call_usage([tuple(r) for r in rows]); u["call_id"] = call_id
        out[model_id or "unknown"].append(u)
    return out


def price_group(usages: list[dict], rates: dict, path_model: str) -> dict:
    avg_min = sum(u["duration_min"] for u in usages) / len(usages)
    path = "hosted" if path_model == HOSTED_MODEL else "local"
    priced = [price_call(u, rates, path=path, avg_call_min=avg_min) for u in usages]
    # tokens that went to the hosted model inside a local call (fallback turns) are priced at list on top
    extra = 0.0
    if path == "local":
        for u in usages:
            t = u["tokens"].get(HOSTED_MODEL)
            if t:
                extra += t["prompt"] / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.input_usd_per_mtok"] + t["completion"] / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.output_usd_per_mtok"]
    n = len(usages); ok = sum(1 for u in usages if u["success"])
    audio = sum(p["audio_usd"] for p in priced); llm = sum(p["llm_usd"] for p in priced) + extra
    return {"model": path_model, "path": path, "calls": n, "successes": ok, "avg_min": avg_min, "audio_usd": audio / n, "llm_usd": llm / n,
            "total_usd": (audio + llm) / n, "per_success_usd": (audio + llm) / ok if ok else None, "llm_basis": priced[0]["llm_basis"], "fallback_llm_usd": extra / n}


def eval_runs(con, run_ids: list[str], rates: dict) -> list[dict]:
    """Text-eval runs from E's ledger rows: episodes, agent_prompt_tokens, agent_completion_tokens (+ episodes_passed when present)."""
    out = []
    for run_id in run_ids:
        items = {r[0]: (r[1], float(r[2])) for r in con.execute("select item, model_id, sum(qty) from cost_ledger where run_id=%s group by item, model_id", (run_id,)).fetchall()}
        if "episodes" not in items:
            print(f"  {run_id}: no 'episodes' row in cost_ledger", file=sys.stderr); continue
        model = items["episodes"][0]; n = items["episodes"][1]
        passed = items.get("episodes_passed", (None, None))[1]
        if passed is None:
            r = con.execute("select count(distinct scenario_id) filter (where pass), count(distinct scenario_id) from eval_results where run_id=%s", (run_id,)).fetchone()
            passed = float(r[0]) if r and r[1] else None
        pt = items.get("agent_prompt_tokens", (None, 0.0))[1]; ct = items.get("agent_completion_tokens", (None, 0.0))[1]
        span = con.execute("select extract(epoch from max(created_at) - min(created_at)) from eval_results where run_id=%s", (run_id,)).fetchone()[0]
        if model == HOSTED_MODEL:
            llm = pt / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.input_usd_per_mtok"] + ct / 1e6 * rates[f"anthropic.{HOSTED_MODEL}.output_usd_per_mtok"]; basis = "measured tokens at list price"
        else:
            secs = local_llm_seconds(pt, ct, rates) if "mac.prefill_tok_s" in rates else (float(span) if span else 0.0)
            llm = mac_hourly_usd(rates) * secs / 3600.0
            basis = f"estimated: Mac hourly cost x {secs / 60:.1f} min of GPU time from tokens at measured prefill/decode rates"
        out.append({"run_id": run_id, "model": model, "episodes": n, "passed": passed, "prompt_tokens": pt, "completion_tokens": ct, "llm_usd": llm,
                    "per_episode_usd": llm / n if n else None, "per_success_usd": (llm / passed) if passed else None, "basis": basis})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", default=None, help="call_id pattern for live calls (SQL LIKE)")
    ap.add_argument("--eval-run", action="append", default=[], help="text-eval run_id (repeatable)")
    ap.add_argument("--set-admission", type=int, default=None)
    ap.add_argument("--ledger", action="store_true"); ap.add_argument("--run-id", default="cost-report")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
        if args.set_admission is not None:
            con.execute("update cost_rates set value=%s, kind='measured', as_of=current_date, note='concurrency where p95 TTFT met budget in the load test (scripts/g1_block.sh)' where key='admission.concurrent_calls'", (args.set_admission,))
            print(f"admission.concurrent_calls = {args.set_admission}")
        rates = load_rates(con)
        hourly = mac_hourly_usd(rates)
        print(f"Mac hourly cost: ${hourly:.3f} (= {rates['mac.purchase_usd']:.0f} / {rates['mac.amortization_hours']:.0f} h + {rates['mac.power_kw']} kW x ${rates['electricity_usd_per_kwh']}/kWh); admission limit {int(rates['admission.concurrent_calls'])} concurrent calls")
        groups = live_calls(con, args.like)
        rows = [price_group(us, rates, m) for m, us in sorted(groups.items()) if us]
        if rows:
            print("\nLive calls (audio vendors measured at list; local LLM = Mac hourly / calls per hour at the admission limit):")
            print(f"{'model':<28}{'calls':>6}{'ok':>4}{'min/call':>9}{'audio$':>9}{'llm$':>9}{'total$':>9}{'$/success':>11}  llm basis")
            for r in rows:
                ps = f"{r['per_success_usd']:.4f}" if r["per_success_usd"] is not None else "n/a"
                print(f"{r['model']:<28}{r['calls']:>6}{r['successes']:>4}{r['avg_min']:>9.2f}{r['audio_usd']:>9.4f}{r['llm_usd']:>9.4f}{r['total_usd']:>9.4f}{ps:>11}  {r['llm_basis']}" + (f" (+${r['fallback_llm_usd']:.4f} hosted fallback tokens)" if r["fallback_llm_usd"] else ""))
        evals = eval_runs(con, args.eval_run, rates)
        if evals:
            print("\nText-eval runs (LLM only; no audio vendors):")
            print(f"{'run_id':<34}{'model':<22}{'eps':>5}{'pass':>6}{'llm$':>9}{'$/episode':>11}{'$/success':>11}  basis")
            for e in evals:
                ps = f"{e['per_success_usd']:.4f}" if e["per_success_usd"] else "n/a"; pe = f"{e['per_episode_usd']:.4f}" if e["per_episode_usd"] else "n/a"
                passed = f"{e['passed']:.0f}" if e["passed"] is not None else "?"
                print(f"{e['run_id']:<34}{e['model']:<22}{e['episodes']:>5.0f}{passed:>6}{e['llm_usd']:>9.4f}{pe:>11}{ps:>11}  {e['basis']}")
        if args.ledger:
            with con.cursor() as cur:
                for r in rows:
                    for item, val, measured in (("cost.audio_per_call_usd", r["audio_usd"], True), ("cost.llm_per_call_usd", r["llm_usd"], r["path"] == "hosted"), ("cost.total_per_call_usd", r["total_usd"], False), ("cost.per_success_usd", r["per_success_usd"], False)):
                        if val is not None:
                            cur.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,'USD',%s,%s)", (args.run_id, r["model"], item, val, measured, f"{r['calls']} calls, {r['successes']} ok; {r['llm_basis']}"))
                for e in evals:
                    for item, val in (("cost.eval_llm_per_episode_usd", e["per_episode_usd"]), ("cost.eval_llm_per_success_usd", e["per_success_usd"])):
                        if val is not None:
                            cur.execute("insert into cost_ledger(run_id, model_id, item, qty, unit, measured, note) values (%s,%s,%s,%s,'USD',%s,%s)", (args.run_id, e["model"], item, val, e["model"] == HOSTED_MODEL, f"{e['run_id']}: {e['basis']}"))
            print(f"\nledger: cost.* rows written under run_id {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
