"""r3-eval: score any OpenAI-compatible endpoint on a scenario directory, or validate scenarios.

    uv run r3-eval run --base-url http://127.0.0.1:8001/v1 --model r3-voice-base \
        --scenarios eval/scenarios/test/ --repeats 3 --out $R3_ARTIFACTS/runs/<id>/ [--parallel 4]
    uv run r3-eval validate eval/scenarios/test/
    uv run r3-eval rescore $R3_ARTIFACTS/runs/<id>/   # re-score saved episodes after scorer changes

`run`: every scenario x repeats (caller seeds 0..N-1, caller temperature 0.3), fresh Sandbox per
episode, hosted caller vs. the agent URL, twelve scorers. Writes <out>/episodes/*.json,
<out>/scores.jsonl, <out>/summary.md (per-family mean and per-run spread, ranked failures) and,
unless --no-db, rows to eval_results / sessions / cost_ledger in Postgres.
`validate`: schema + seed consistency for every file in a directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.sandbox import Sandbox
from eval.scorers import ADVISORY, WEIGHTS, ScorerResult, all_pass, score_episode
from eval.scorers.normalize import args_match
from eval.simulator.clients import LLMCaller, OpenAIAgent, make_client
from eval.simulator.episode import AgentUnderTest, Caller, Episode, run_episode
from eval.simulator.farewell import is_farewell
from eval.simulator.scenario import (
    SCENARIO_SUFFIXES,
    Scenario,
    load_scenario,
    load_scenarios,
    validate_against_seed,
)
from eval.simulator.stt_shape import cardinal, ordinal, spell_digits, year_words

ROOT = Path(__file__).resolve().parents[1]
Result = tuple[Episode, list[ScorerResult]]
AgentFactory = Callable[[], AgentUnderTest]
CallerFactory = Callable[[Scenario, int], Caller]


# ----- orchestration -----------------------------------------------------------------------


def save_scored(ep: Episode, scores: list[ScorerResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ep.to_dict()
    data["scores"] = [s.to_dict() for s in scores]
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def score_rows(ep: Episode, scores: list[ScorerResult]) -> list[dict]:
    return [
        {
            "run_id": ep.run_id,
            "model_id": ep.model_id,
            "scenario_id": ep.scenario_id,
            "episode_id": ep.episode_id,
            "repeat_seed": ep.repeat_seed,
            "family": ep.family,
            "scorer": s.scorer,
            "pass": s.passed,
            "advisory": s.advisory,
            "turn_idx": s.turn_idx,
            "detail": s.detail,
        }
        for s in scores
    ]


async def run_scenarios(
    scenarios: list[Scenario],
    agent_factory: AgentFactory,
    caller_factory: CallerFactory,
    *,
    out_dir: str | Path,
    run_id: str,
    model_id: str,
    parallel: int = 4,
    repeats: int = 1,
    prompt_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    seeds: list[int] | None = None,
) -> list[Result]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    scores_path.write_text("")
    sem = asyncio.Semaphore(max(1, parallel))
    results: list[Result] = []

    async def one(sc: Scenario, seed: int) -> None:
        async with sem:
            ep = await run_episode(
                sc,
                agent_factory(),
                caller_factory(sc, seed),
                Sandbox(),
                model_id=model_id,
                run_id=run_id,
                repeat_seed=seed,
                prompt_path=prompt_path,
            )
            scores = score_episode(ep, sc)
            name = f"{sc.id}-s{seed}"
            save_scored(ep, scores, out / "episodes" / f"{name}.json")
            with scores_path.open("a") as fh:
                for row in score_rows(ep, scores):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append((ep, scores))
            if progress:
                failed = [s.scorer for s in scores if not s.passed and not s.advisory]
                progress(
                    f"{name:34} {ep.outcome:10} turns={len(ep.turns):2} "
                    f"{'ALL PASS' if not failed else 'FAIL ' + ','.join(failed)}"
                    + (f"  [{ep.error}]" if ep.error else "")
                )

    seed_list = list(seeds) if seeds is not None else list(range(repeats))
    await asyncio.gather(*(one(sc, seed) for sc in scenarios for seed in seed_list))
    results.sort(key=lambda r: (r[0].scenario_id, r[0].repeat_seed))
    (out / "summary.md").write_text(summarize(results, run_id=run_id, model_id=model_id))
    return results


# ----- summary -----------------------------------------------------------------------------


def _pct(n: float, d: float) -> str:
    return "-" if d == 0 else f"{round(100 * n / d)}%"


def _mean_spread(results: list[Result]) -> str:
    """All-pass rate per repeat seed -> 'mean% (min–max)'; one seed -> 'mean%'."""
    by_seed: dict[int, list[bool]] = defaultdict(list)
    for ep, scores in results:
        by_seed[ep.repeat_seed].append(all_pass(scores))
    rates = [sum(v) / len(v) for v in by_seed.values() if v]
    if not rates:
        return "-"
    mean = sum(rates) / len(rates)
    if len(rates) == 1:
        return f"{round(100 * mean)}%"
    return f"{round(100 * mean)}% ({round(100 * min(rates))}–{round(100 * max(rates))})"


def summarize(results: list[Result], *, run_id: str, model_id: str) -> str:
    scorers = list(WEIGHTS)
    by_family: dict[str, list[Result]] = defaultdict(list)
    for ep, scores in results:
        by_family[ep.family].append((ep, scores))
    n = len(results)
    seeds = sorted({ep.repeat_seed for ep, _ in results})
    passed_all = sum(1 for _, s in results if all_pass(s))

    def col(name: str) -> str:
        return f"{name} (advisory)" if name in ADVISORY else name

    lines = [
        f"# r3-eval summary — run `{run_id}`",
        "",
        (
            f"model: `{model_id}` · episodes: {n} · scenarios: {len({ep.scenario_id for ep, _ in results})}"
            f" · repeats: {len(seeds)} (seeds {', '.join(map(str, seeds))}) · generated "
            f"{datetime.now().isoformat(timespec='seconds')}"
        ),
        "",
        (
            f"**All non-advisory scorers pass: {passed_all}/{n} ({_pct(passed_all, n)}); "
            f"mean over repeats {_mean_spread(results)}**"
        ),
        "",
        (
            "Per family: all-pass rate as mean over repeats (min–max across repeats); scorer columns are "
            "pass rates over the episodes where the scorer applies."
        ),
        "",
        "## Per family",
        "",
        "| family | episodes | all pass | " + " | ".join(col(s) for s in scorers) + " |",
        "|---|---|---|" + "---|" * len(scorers),
    ]
    for fam in sorted(by_family):
        rs = by_family[fam]
        cells = []
        for name in scorers:
            app = [next(x for x in s if x.scorer == name) for _, s in rs]
            app = [x for x in app if x.detail.get("applicable", True)]
            cells.append(_pct(sum(1 for x in app if x.passed), len(app)))
        lines.append(f"| {fam} | {len(rs)} | {_mean_spread(rs)} | " + " | ".join(cells) + " |")
    by_scenario: dict[str, list[Result]] = defaultdict(list)
    for ep, scores in results:
        by_scenario[ep.scenario_id].append((ep, scores))
    lines += [
        "",
        "## Per scenario",
        "",
        "| scenario | family | all-pass seeds | failing scorers (seed, turn) |",
        "|---|---|---|---|",
    ]
    for sid in sorted(by_scenario):
        rs = sorted(by_scenario[sid], key=lambda r: r[0].repeat_seed)
        ok = sum(1 for _, sc in rs if all_pass(sc))
        fails_txt = ", ".join(
            f"{x.scorer} (s{ep.repeat_seed} t{x.turn_idx if x.turn_idx is not None else '-'})"
            for ep, sc in rs
            for x in sc
            if not x.passed and not x.advisory
        )
        lines.append(f"| {sid} | {rs[0][0].family} | {ok}/{len(rs)} | {fails_txt} |")
    lines += [
        "",
        "## Ranked failures (frequency × cost)",
        "",
        "| scorer | failures | weight | cost | example scenarios (turn) |",
        "|---|---|---|---|---|",
    ]
    fails: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for ep, scores in results:
        for s in scores:
            if not s.passed:
                fails[s.scorer] += 1
                tag = f"{ep.scenario_id}" + (f" (t{s.turn_idx})" if s.turn_idx is not None else "")
                if len(examples[s.scorer]) < 4 and tag not in examples[s.scorer]:
                    examples[s.scorer].append(tag)
    ranked = sorted(fails.items(), key=lambda kv: (-kv[1] * WEIGHTS[kv[0]], -kv[1], kv[0]))
    for name, cnt in ranked:
        lines.append(
            f"| {col(name)} | {cnt} | {WEIGHTS[name]} | {cnt * WEIGHTS[name]} | {', '.join(examples[name])} |"
        )
    if not ranked:
        lines.append("| (none) | 0 | - | 0 | |")
    outcomes = Counter(ep.outcome for ep, _ in results)
    turns = [len(ep.turns) for ep, _ in results]
    lat = sorted(x for ep, _ in results for x in ep.latency_ms)
    agent_pt = sum(ep.usage.get("agent", {}).get("prompt_tokens", 0) for ep, _ in results)
    agent_ct = sum(ep.usage.get("agent", {}).get("completion_tokens", 0) for ep, _ in results)
    reqs = sum(ep.n_model_requests for ep, _ in results)
    trunc = sum(getattr(ep, "n_truncated", 0) for ep, _ in results)
    first_pt = [
        ep.first_prompt_tokens for ep, _ in results if getattr(ep, "first_prompt_tokens", None)
    ]
    sources = sorted({getattr(ep, "prompt_source", "") for ep, _ in results})
    temp_applied = [ep.usage.get("caller", {}).get("temperature_applied") for ep, _ in results]
    lines += [
        "",
        "## Episodes",
        "",
        f"- outcomes: {dict(outcomes)}",
        (
            f"- caller turns per episode: mean {sum(turns) / n:.1f}, max {max(turns)}"
            if n
            else "- no episodes"
        ),
        (
            f"- agent model requests: {reqs}; prompt tokens {agent_pt}; completion tokens {agent_ct}; "
            f"truncated replies (finish_reason=length): {trunc}"
        ),
        (
            f"- agent request latency ms: p50 {lat[len(lat) // 2]:.0f}, "
            f"p95 {lat[max(0, int(len(lat) * 0.95) - 1)]:.0f}"
            if lat
            else "- no latency samples"
        ),
        (
            f"- system prompt: {', '.join(s for s in sources if s)}; first-request prompt tokens "
            f"(prompt + tools + greeting + first caller turn, endpoint tokenizer): "
            f"{round(sum(first_pt) / len(first_pt)) if first_pt else 'n/a'}"
        ),
        (
            "- caller temperature: requested but NOT applied — the endpoint rejected `temperature`; "
            "repeats differ only through the seed in the caller prompt"
            if any(t is False for t in temp_applied)
            else "- caller temperature: applied as requested"
        ),
    ]
    errors = [(ep.scenario_id, ep.repeat_seed, ep.error) for ep, _ in results if ep.error]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {sid} s{seed}: {err}" for sid, seed, err in errors]
    return "\n".join(lines) + "\n"


# ----- Postgres -----------------------------------------------------------------------------


def _insert_eval_results(
    cur, ep: Episode, scores: list[ScorerResult], run_id: str, model_id: str
) -> int:
    from psycopg.types.json import Jsonb

    cur.executemany(
        "insert into eval_results (run_id, model_id, scenario_id, family, scorer, pass, detail)"
        " values (%s, %s, %s, %s, %s, %s, %s)",
        [
            (
                run_id,
                model_id,
                ep.scenario_id,
                ep.family,
                s.scorer,
                s.passed,
                Jsonb(
                    {
                        **s.detail,
                        "repeat_seed": ep.repeat_seed,
                        "advisory": s.advisory,
                        "turn_idx": s.turn_idx,
                        "episode_id": ep.episode_id,
                    }
                ),
            )
            for s in scores
        ],
    )
    return len(scores)


def _ledger_rows(results: list[Result], run_id: str, model_id: str) -> list[tuple]:
    usage = {
        "agent_prompt_tokens": 0,
        "agent_completion_tokens": 0,
        "caller_prompt_tokens": 0,
        "caller_completion_tokens": 0,
    }
    for ep, _ in results:
        for who in ("agent", "caller"):
            u = ep.usage.get(who, {}) or {}
            usage[f"{who}_prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            usage[f"{who}_completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
    rows = [
        (run_id, model_id, "episodes", len(results), "episode", True, "r3-eval run"),
        (
            run_id,
            model_id,
            "episodes_passed",
            sum(1 for _, sc in results if all_pass(sc)),
            "episode",
            True,
            "episodes passing every non-advisory scorer",
        ),
    ]
    rows += [
        (run_id, model_id, item, qty, "token", True, "r3-eval measured usage")
        for item, qty in usage.items()
    ]
    return rows


def _insert_ledger(cur, results: list[Result], run_id: str, model_id: str) -> int:
    rows = _ledger_rows(results, run_id, model_id)
    cur.executemany(
        "insert into cost_ledger (run_id, model_id, item, qty, unit, measured, note) values (%s, %s, %s, %s, %s, %s, %s)",
        rows,
    )
    return len(rows)


def write_db(conn, results: list[Result], *, run_id: str, model_id: str) -> dict[str, int]:
    from psycopg.types.json import Jsonb

    counts = {"eval_results": 0, "sessions": 0, "cost_ledger": 0}
    with conn.cursor() as cur:
        for ep, scores in results:
            counts["eval_results"] += _insert_eval_results(cur, ep, scores, run_id, model_id)
            cur.execute(
                "insert into sessions (call_id, model_id, started_at, ended_at, outcome, state, scenario_id, run_id)"
                " values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict (call_id) do nothing",
                (
                    ep.episode_id,
                    model_id,
                    datetime.fromisoformat(ep.started_at),
                    datetime.fromisoformat(ep.ended_at),
                    ep.outcome,
                    Jsonb(ep.final_state),
                    ep.scenario_id,
                    run_id,
                ),
            )
            counts["sessions"] += 1
        counts["cost_ledger"] += _insert_ledger(cur, results, run_id, model_id)
    return counts


# ----- validate ------------------------------------------------------------------------------


def validate_dir(directory: str | Path) -> dict[str, list[str]]:
    """{file name: problems} for every scenario file that fails the schema or the seed checks."""
    sb = Sandbox()
    problems: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for p in sorted(x for x in Path(directory).iterdir() if x.suffix in SCENARIO_SUFFIXES):
        try:
            s = load_scenario(p)
        except Exception as e:  # noqa: BLE001 — report every broken file, keep going
            problems[p.name] = [str(e)]
            continue
        issues = validate_against_seed(s, sb)
        if s.id in seen:
            issues.append(f"duplicate id {s.id} (also in {seen[s.id]})")
        seen[s.id] = p.name
        if issues:
            problems[p.name] = issues
    return problems


# ----- rescore -------------------------------------------------------------------------------


def rescore_dir(directory: str | Path, scenarios_dir: str | Path | None = None, conn=None) -> int:
    """Re-run the scorers over <dir>/episodes/*.json; rewrite scores.jsonl and summary.md.

    With `conn`, the run's eval_results and cost_ledger rows are replaced from the saved episodes (sessions stay)."""
    from eval.simulator.scenario import parse_scenario

    out = Path(directory)
    by_id = {s.id: s for s in load_scenarios(scenarios_dir)} if scenarios_dir else {}
    results: list[Result] = []
    for p in sorted((out / "episodes").glob("*.json")):
        data = json.loads(p.read_text())
        ep = Episode.from_dict(data)
        sc = by_id.get(ep.scenario_id) or parse_scenario(data["scenario"], f"{p.name}#scenario")
        scores = score_episode(ep, sc)
        save_scored(ep, scores, p)
        results.append((ep, scores))
    results.sort(key=lambda r: (r[0].scenario_id, r[0].repeat_seed))
    with (out / "scores.jsonl").open("w") as fh:
        for ep, scores in results:
            for row in score_rows(ep, scores):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    run_id = results[0][0].run_id if results else out.name
    model_id = results[0][0].model_id if results else "?"
    (out / "summary.md").write_text(summarize(results, run_id=run_id, model_id=model_id))
    if conn is not None and results:
        with conn.cursor() as cur:
            cur.execute("delete from eval_results where run_id = %s", (run_id,))
            for ep, scores in results:
                _insert_eval_results(cur, ep, scores, run_id, model_id)
            cur.execute("delete from cost_ledger where run_id = %s", (run_id,))
            _insert_ledger(cur, results, run_id, model_id)
    return len(results)


# ----- compare -------------------------------------------------------------------------------


def compare_runs(run_dirs: list[str | Path], exclude_caller_faults: bool = False) -> str:
    """Side-by-side all-pass rates per family for several run directories (reads scores.jsonl).
    With `exclude_caller_faults`, a scenario x seed episode in which the simulated caller mis-stated
    its credentials in any run (see `caller_fault`) is dropped from every run, so candidates are
    compared on the same calls."""
    runs: list[tuple[str, dict[str, dict]]] = []  # (label, episode_id -> {"family", "fails": set})
    excluded: set[str] = set()
    for d in run_dirs:
        d = Path(d)
        episodes: dict[str, dict] = {}
        label = d.name
        for line in (d / "scores.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            label = r.get("model_id") or label
            e = episodes.setdefault(
                r["episode_id"],
                {
                    "family": r["family"],
                    "key": f"{r.get('scenario_id')}-s{r.get('repeat_seed')}",
                    "fails": set(),
                    "advisory_fails": set(),
                },
            )
            if not r["pass"]:
                (e["advisory_fails"] if r.get("advisory") else e["fails"]).add(r["scorer"])
        if exclude_caller_faults:
            for p in (d / "episodes").glob("*.json"):
                ep = json.loads(p.read_text())
                if caller_fault(ep):
                    excluded.add(f"{ep.get('scenario_id')}-s{ep.get('repeat_seed')}")
        runs.append((label, episodes))
    if excluded:
        runs = [
            (label, {eid: e for eid, e in eps.items() if e["key"] not in excluded})
            for label, eps in runs
        ]
    labels = [label for label, _ in runs]
    if len(set(labels)) < len(labels):  # same model id scored twice (e.g. hosted vs local caller)
        runs = [
            (f"{label} ({Path(d).parent.name if Path(d).name == 'eval' else Path(d).name})", eps)
            if labels.count(label) > 1
            else (label, eps)
            for (label, eps), d in zip(runs, run_dirs)
        ]
    families = sorted({e["family"] for _, eps in runs for e in eps.values()})
    lines = []
    if exclude_caller_faults:
        lines += [
            f"Excluded {len(excluded)} scenario x seed episode(s) in which the simulated caller mis-stated "
            f"its own phone or date of birth in at least one run: {', '.join(sorted(excluded)) or 'none'}",
            "",
        ]
    lines += [
        "| family | " + " | ".join(label for label, _ in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]

    def rate(eps: dict, fam: str | None) -> str:
        sel = [e for e in eps.values() if fam is None or e["family"] == fam]
        return _pct(sum(1 for e in sel if not e["fails"]), len(sel))

    for fam in families:
        lines.append(f"| {fam} | " + " | ".join(rate(eps, fam) for _, eps in runs) + " |")
    lines.append("| all | " + " | ".join(rate(eps, None) for _, eps in runs) + " |")
    lines += [
        "",
        "## Failures by scorer (episodes failing, non-advisory; advisory in brackets)",
        "",
        "| scorer | " + " | ".join(label for label, _ in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    for scorer in WEIGHTS:
        cells = []
        for _, eps in runs:
            n = sum(1 for e in eps.values() if scorer in e["fails"])
            adv = sum(1 for e in eps.values() if scorer in e["advisory_fails"])
            cells.append(f"[{adv}]" if scorer in ADVISORY else str(n))
        lines.append(f"| {scorer} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----- failures dump (analysis loop: every failing episode, the scorer, the turn, the context) -----


def _render_turns(ep: dict, turns: list[int]) -> list[str]:
    lines = []
    for t in ep.get("turns", []):
        if t["idx"] not in turns:
            continue
        lines.append(f"C: {t['caller_text']}")
        for c in t.get("tool_calls", []):
            err = c["result"].get("error") if isinstance(c.get("result"), dict) else None
            lines.append(
                f"  -> {c['name']}({json.dumps(c['args'])[:160]})"
                + (f"  <- error {err}" if err else "  <- ok")
            )
        if t.get("assistant_text"):
            lines.append(f"A: {t['assistant_text']}")
    return lines


def tool_calls_cause(detail: dict, tool_log: list[dict]) -> dict | None:
    """Why the tool_calls scorer failed, from its first missing expected step: the tool was never
    called after the last matched step ("missing call"), it was called with other argument values
    ("wrong args", with the differing fields), or it was called with the right values but the wrong
    outcome ("wrong outcome", e.g. expected ok and got an error)."""
    fm = (detail or {}).get("first_missing")
    if not fm:
        return None
    matched = detail.get("matched") or []
    after = max(matched) if matched else -1
    candidates = [
        c
        for c in tool_log
        if c.get("tool") == fm["tool"] and c.get("idx", 0) > after and not c.get("replay")
    ]
    if not candidates:
        return {"cause": "missing call", "tool": fm["tool"], "fields": []}
    best = min(candidates, key=lambda c: len(args_match(fm.get("args") or {}, c.get("args"))))
    fields = sorted(args_match(fm.get("args") or {}, best.get("args")))
    if fields:
        return {"cause": "wrong args", "tool": fm["tool"], "fields": fields}
    return {"cause": "wrong outcome", "tool": fm["tool"], "fields": []}


def failures_report(run_dir: str | Path) -> str:
    """Markdown: for each failing episode, the non-advisory scorers that failed with their turn, the
    detail, and the transcript around the first failing turn."""
    out = [f"# Failing episodes in `{Path(run_dir).name}`", ""]
    n = 0
    causes: Counter[tuple[str, str]] = Counter()
    fields: Counter[str] = Counter()
    claims: Counter[str] = Counter()
    body: list[str] = []
    for p in sorted((Path(run_dir) / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        fails = [sc for sc in ep.get("scores", []) if not sc["passed"] and not sc.get("advisory")]
        if not fails:
            continue
        n += 1
        out = body
        header = f"family {ep['family']} · outcome {ep['outcome']}" + (
            f" · error {ep['error']}" if ep.get("error") else ""
        )
        out += [
            f"## {ep['episode_id']}",
            header,
            f"goal: {ep.get('scenario', {}).get('goal', '')[:200]}",
            "",
        ]
        first = None
        for sc in fails:
            turn = sc.get("turn_idx")
            if first is None and turn is not None:
                first = turn
            detail = json.dumps(sc.get("detail", {}), ensure_ascii=False)
            out.append(
                f"- **{sc['scorer']}** at turn {turn if turn is not None else '-'}: {detail[:300]}"
            )
            if sc["scorer"] == "no_success_claim_before_result":
                for v in (sc.get("detail") or {}).get("violations") or []:
                    claims["transfer" if v.get("action") == "handoff" else "write"] += 1
            if sc["scorer"] == "tool_calls":
                cause = tool_calls_cause(sc.get("detail") or {}, ep.get("tool_log") or [])
                if cause:
                    causes[(cause["cause"], cause["tool"])] += 1
                    for f in cause["fields"]:
                        fields[f] += 1
                    out.append(
                        f"  - cause: {cause['cause']} {cause['tool']}"
                        + (f" [{', '.join(cause['fields'])}]" if cause["fields"] else "")
                    )
        out.append("")
        if first:
            window = sorted({t for t in (first - 1, first, first + 1) if t > 0})
        else:
            window = [t["idx"] for t in ep.get("turns", [])][-3:]
        out += _render_turns(ep, window) or ["(no turns)"]
        out.append("")
    head = [f"# Failing episodes in `{Path(run_dir).name}`", ""]
    if causes:
        head += [
            "**tool_calls failures by cause**",
            "",
            "First missing expected step: never called after the last matched step (missing call), called with "
            "other argument values (wrong args), or called with the right values and the wrong outcome.",
            "",
            "| cause | tool | episodes |",
            "|---|---|---|",
        ]
        head += [f"| {c} | {t} | {k} |" for (c, t), k in causes.most_common()]
        if fields:
            head += [
                "",
                "Wrong-args fields: " + ", ".join(f"{f} {k}" for f, k in fields.most_common()),
            ]
        head.append("")
    if claims:
        head += [
            "**success claims by claimed action**: "
            + ", ".join(f"{k} {claims[k]}" for k in ("transfer", "write") if claims.get(k)),
            "",
        ]
    out = head + body
    if n == 0:
        out.append("no failing episodes")
    return "\n".join(out) + "\n"


# ----- failure buckets (plan T+16: bad data / missing example type / context truncation / model capability /
# application logic; "harness" for transport errors) ----------------------------------------------------

BUCKETS = (
    "bad data",
    "missing example type",
    "context truncation",
    "model capability",
    "application logic",
    "harness",
)
LOW_TARGETS = 30  # a tool with fewer supervised targets than this in the training set is a data gap
DATA_SCORERS = {
    "handoff",
    "no_jargon",
    "no_success_claim_before_result",
    "verified_before_account_facts",
    "corrections_hold",
    "must_mention",
    "forbidden_tools",
}


_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _spoken_digits(text: str) -> str:
    """Every digit the caller spoke, in order; filler words between digits are dropped."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return "".join(_DIGIT_WORDS.get(t, t if t.isdigit() else "") for t in toks)


def _dob_spoken(text: str, iso: str) -> bool:
    """True when month, day and year of `iso` all appear in `text` in some spoken form."""
    y, m, d = (int(x) for x in iso.split("-"))
    t = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
    month = _MONTHS[m - 1]
    days = {ordinal(d), cardinal(d), str(d), f"{d:02d}", spell_digits(f"{d:02d}")}
    years = {year_words(y), spell_digits(str(y)), str(y), cardinal(y)}
    if 2001 <= y <= 2009:  # "twenty one" is 2021, not 2001
        years |= {f"two thousand and {cardinal(y - 2000)}", f"twenty oh {cardinal(y - 2000)}"}
    has = lambda w: f" {w} " in t
    return (
        (has(month) or has(month[:3])) and any(has(w) for w in days) and any(has(w) for w in years)
    )


def caller_fault(ep: dict) -> str | None:
    """Why the simulated caller, not the agent, sank verification. None when the caller stated the
    true credentials (then a failed lookup is the agent's fault), when no lookup was attempted,
    when one succeeded, or when the scenario never expected verification to succeed."""
    sc = ep.get("scenario") or {}
    expected = (sc.get("expect") or {}).get("tool_calls") or []
    if not any(c.get("tool") == "lookup_account" and c.get("ok", True) for c in expected):
        return None
    lookups = [t for t in ep.get("tool_log") or [] if t.get("tool") == "lookup_account"]
    if any(t.get("ok") for t in lookups):
        return None
    knows = (sc.get("caller") or {}).get("knows") or {}
    # judge what the caller meant to say (the reply before STT shaping); the shaping is the harness's
    # intended noise and the agent is expected to cope with it
    raw = [t.get("caller_text_raw") for t in ep.get("turns") or [] if t.get("caller_text_raw")]
    said = " ".join(
        raw or [m.get("content") or "" for m in ep.get("messages") or [] if m.get("role") == "user"]
    )
    phone = re.sub(r"\D", "", str(knows.get("phone") or ""))
    if phone and phone not in _spoken_digits(said):
        return "caller never stated the true phone number"  # the greeting always asks for it
    if not lookups:
        return None  # no lookup attempted: the agent may never have asked for the date of birth
    dob = knows.get("date_of_birth")
    if dob and not _dob_spoken(said, str(dob)):
        return "caller never stated the true date of birth"
    return None


def caller_stuck(ep: dict) -> str | None:
    """A simulated caller that keeps saying goodbye without hanging up, or repeats one line three or
    more times in a row, makes the turn cap the harness's doing rather than the agent's."""
    lines = [
        (m.get("content") or "").strip().lower()
        for m in ep.get("messages") or []
        if m.get("role") == "user"
    ]
    if sum(1 for line in lines if is_farewell(line)) >= 2:
        return "caller said goodbye without hanging up"
    best = run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a and a == b else 1
        best = max(best, run)
    if best >= 3:
        return f"caller repeated the same line {best} times in a row"
    return None


def _normalize_line(text: str) -> str:
    """Same normalization as the agent's loop guard (services/agent/loopguard.normalize)."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


# The live guard exempts a repeated farewell (a closing line said twice to a caller who keeps saying
# goodbye); same pattern as services/agent/loopguard, applied to the normalized previous line.
_GUARD_FAREWELL = re.compile(
    r"\b(goodbye|good bye|bye|take care|have a (good|great|nice) (day|one|evening|afternoon|morning))\b"
)


def agent_repeats(ep: dict) -> dict:
    """How often S's application-side loop guard would have intervened on this transcript: the
    first consecutive identical assistant line is swallowed and rephrased, the second becomes a
    handoff that ends the call. The runner talks to the model directly, so it never sees the guard;
    this is the same rule applied after the fact. Tool-call-only turns are not lines."""
    lines = [
        _normalize_line(m.get("content") or "")
        for m in ep.get("messages") or []
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]
    rephrase = handoff = 0
    best = run = 1 if lines else 0
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b and not _GUARD_FAREWELL.search(a) else 1
        best = max(best, run)
        if run == 2:
            rephrase += 1
        elif run == 3:
            handoff += 1
            break
    return {"rephrase": rephrase, "handoff": handoff, "max_streak": best}


def bucket_for(
    score: dict, ep_info: dict, target_counts: dict | None, prefix_counts: dict | None = None
) -> str:
    """First-pass label for one failing scorer of one episode, from scorer evidence alone.
    `target_counts` / `prefix_counts`: how often each tool was the trained target / appeared in a
    training prefix (the dataset the candidate trained on)."""
    if ep_info.get("outcome") == "error":
        return "harness"
    if ep_info.get("caller_fault"):
        return "harness"
    if ep_info.get("caller_stuck") and score["scorer"] == "max_turns":
        return "harness"
    name, d = score["scorer"], score.get("detail", {}) or {}
    if ep_info.get("long_context") and name in ("tool_calls", "final_state", "corrections_hold"):
        return "context truncation"
    if name in ("valid_tool_calls", "max_turns", "calendar_consistency", "max_successful_calls"):
        return "model capability"
    if name == "tool_calls":
        tool = (d.get("first_missing") or {}).get("tool")
        if tool and target_counts is not None and target_counts.get(tool, 0) < LOW_TARGETS:
            # the teacher made the call all the time but the selection never trained it: the data
            # that exists taught the wrong thing (target mix), not a type of example that is absent
            if prefix_counts and prefix_counts.get(tool, 0) >= LOW_TARGETS:
                return "bad data"
            return "missing example type"
        return "model capability"
    if name == "final_state":
        return "application logic" if ep_info.get("tool_calls") else "missing example type"
    if name in DATA_SCORERS:
        return "missing example type"
    return "model capability"


def target_counts_from_dataset(dataset_dir: str | Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for split in ("train", "valid"):
        f = Path(dataset_dir) / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            for tc in json.loads(line)["messages"][-1].get("tool_calls") or []:
                counts[(tc.get("function") or {}).get("name") or "?"] += 1
    return dict(counts)


def prefix_counts_from_dataset(dataset_dir: str | Path) -> dict[str, int]:
    """How often each tool is called inside a training prefix (context the model saw, not a target)."""
    counts: Counter[str] = Counter()
    for split in ("train", "valid"):
        f = Path(dataset_dir) / f"{split}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            for m in json.loads(line)["messages"][:-1]:
                for tc in m.get("tool_calls") or []:
                    counts[(tc.get("function") or {}).get("name") or "?"] += 1
    return dict(counts)


# Same phrase set as the live claim guard (services/agent/claimguard.TRANSFER_RE), applied to the
# normalized line, so the post-hoc count and the live `claim_guard` events mean the same thing.
_TRANSFER_LINE = re.compile(
    r"\b(transferr?ing you|transfer you|hand(?:ing)? you (?:over|off)|connect(?:ing)? you (?:to|with)|"
    r"put(?:ting)? you through|(?:a |my |our )?(?:specialist|colleague|team member|representative|agent|person) "
    r"(?:will|is going to|can) (?:take over|take it from here|help|be with you|assist)|"
    r"let me get (?:a|my) (?:colleague|specialist))\b"
)


def one_strike_handoff(ep: dict) -> bool:
    """The agent handed off (or said it was transferring) after a single not_verified lookup. The
    contract asks for a second attempt first; the generated data never produced not_verified, so this
    is a model gap that data can fix even when the caller caused the failed lookup."""
    lookups = [
        t
        for t in ep.get("tool_log") or []
        if t.get("tool") == "lookup_account" and not t.get("replay")
    ]
    if len(lookups) != 1 or lookups[0].get("ok"):
        return False
    err = lookups[0].get("error") or (lookups[0].get("result") or {}).get("error")
    if err != "not_verified":
        return False
    if any(t.get("tool") == "request_handoff" for t in ep.get("tool_log") or []):
        return True
    return any(
        _TRANSFER_LINE.search(_normalize_line(m.get("content") or ""))
        for m in ep.get("messages") or []
        if m.get("role") == "assistant"
    )


def buckets_report(
    run_dir: str | Path, target_counts: dict | None = None, dataset_dir: str | Path | None = None
) -> str:
    prefix_counts = None
    if dataset_dir is not None:
        prefix_counts = prefix_counts_from_dataset(dataset_dir)
        if target_counts is None:
            target_counts = target_counts_from_dataset(dataset_dir)
    rows = []
    caller_faults: list[tuple[str, str]] = []
    stuck_callers: list[tuple[str, str]] = []
    one_strikes: list[str] = []
    repeaters: list[tuple[str, dict]] = []
    counts: Counter[str] = Counter()
    for p in sorted((Path(run_dir) / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        scores = ep.get("scores", [])
        fails = [sc for sc in scores if not sc["passed"] and not sc.get("advisory")]
        if not fails:
            continue
        fail_turn = min(
            (sc["turn_idx"] for sc in fails if sc.get("turn_idx") is not None), default=None
        )
        info = {
            "outcome": ep["outcome"],
            "tool_calls": next(
                (sc["passed"] for sc in scores if sc["scorer"] == "tool_calls"), False
            ),
            "long_context": bool(fail_turn and fail_turn >= 8),
            "caller_fault": caller_fault(ep),
            "caller_stuck": caller_stuck(ep),
            "one_strike_handoff": one_strike_handoff(ep),
        }
        if info["one_strike_handoff"]:
            one_strikes.append(ep["episode_id"])
        if info["caller_stuck"] and any(sc["scorer"] == "max_turns" for sc in fails):
            stuck_callers.append((ep["episode_id"], info["caller_stuck"]))
        rep = agent_repeats(ep)
        if rep["rephrase"] or rep["handoff"]:
            repeaters.append((ep["episode_id"], rep))
        if info["caller_fault"]:
            caller_faults.append((ep["episode_id"], info["caller_fault"]))
        labelled = [
            (sc["scorer"], bucket_for(sc, info, target_counts, prefix_counts), sc.get("turn_idx"))
            for sc in fails
        ]
        primary = max(labelled, key=lambda x: WEIGHTS.get(x[0], 0))[1]
        if info["one_strike_handoff"]:
            primary = "missing example type"  # root cause: no retry after not_verified
        counts[primary] += 1
        rows.append((ep["episode_id"], ep["family"], primary, labelled))
    lines = [
        f"# Failure buckets — `{Path(run_dir).name}` (first pass by session E from scorer evidence; the human's labels win)",
        "",
        f"Failing episodes: {len(rows)}. Target counts per tool from the training set: {target_counts or 'n/a'}; "
        f"calls inside training prefixes: {prefix_counts or 'n/a'}",
        "",
        f"One-strike handoffs (the agent transferred after a single not_verified lookup; the contract wants a second "
        f"attempt and the training data never produced not_verified, so the episode is bucketed as missing example "
        f"type even when the caller caused the failed lookup): {len(one_strikes)}"
        + ("".join(f"\n- {eid}" for eid in one_strikes) if one_strikes else ""),
        "",
        f"Caller faults (the simulated caller mis-stated its own phone or date of birth, so verification failed "
        f"before the agent could be judged; bucketed as harness): {len(caller_faults)}"
        + ("".join(f"\n- {eid}: {why}" for eid, why in caller_faults) if caller_faults else ""),
        "",
        f"Stuck callers (the simulated caller kept saying goodbye without hanging up, or repeated one line three or more "
        f"times in a row; their max_turns failure is bucketed as harness): {len(stuck_callers)}"
        + ("".join(f"\n- {eid}: {why}" for eid, why in stuck_callers) if stuck_callers else ""),
        "",
        f"Adapter repetition (S's loop guard would have intervened: first identical line rephrased, second a handoff; "
        f"a repeated farewell is exempt; exact match after normalization, so one-word variants such as "
        f"'Which day' -> 'Which date' are not counted, same as the live guard): "
        f"{len(repeaters)} failing episodes, {sum(r['rephrase'] for _, r in repeaters)} rephrases, "
        f"{sum(r['handoff'] for _, r in repeaters)} handoffs"
        + (
            "".join(f"\n- {eid}: streak {r['max_streak']}" for eid, r in repeaters)
            if repeaters
            else ""
        ),
        "",
        "## Bucket counts",
        "",
        "| bucket | episodes |",
        "|---|---|",
    ]
    for b in BUCKETS:
        lines.append(f"| {b} | {counts.get(b, 0)} |")
    data_fixable = counts.get("missing example type", 0) + counts.get("bad data", 0)
    model_side = len(rows) - data_fixable - counts.get("harness", 0)
    verdict = (
        "run 2 is justified"
        if data_fixable >= max(3, 0.1 * max(1, len(rows) - counts.get("harness", 0)))
        else "run 2 is not justified"
    )
    lines += [
        "",
        (
            f"**Verdict:** {verdict}: {data_fixable} failing episodes sit in data-fixable buckets, {model_side} in "
            f"model/application buckets, {counts.get('harness', 0)} were harness errors. The plan's rule: run 2 only if a "
            "bucket that data can fix is also one the test set punishes (every failing test episode counts)."
        ),
        "",
        "## Per episode",
        "",
        "| episode | family | bucket | failing scorers (bucket, turn) |",
        "|---|---|---|---|",
    ]
    for eid, fam, primary, labelled in rows:
        detail = "; ".join(f"{sc} ({b}, t{t if t is not None else '-'})" for sc, b, t in labelled)
        lines.append(f"| {eid} | {fam} | {primary} | {detail} |")
    return "\n".join(lines) + "\n"


# ----- CLI -----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r3-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="score an endpoint on a scenario directory")
    r.add_argument("--base-url", required=True, help="agent under test (OpenAI-compatible)")
    r.add_argument("--model", required=True)
    r.add_argument(
        "--api-key",
        default=None,
        help="default: HOSTED_LLM_API_KEY for the hosted URL, else 'none'",
    )
    r.add_argument("--scenarios", required=True, help="directory of scenario files")
    r.add_argument("--out", required=True, help="run directory, e.g. $R3_ARTIFACTS/runs/<id>/")
    r.add_argument(
        "--repeats", type=int, default=1, help="episodes per scenario (caller seeds 0..N-1)"
    )
    r.add_argument(
        "--seeds",
        type=lambda v: [int(x) for x in v.split(",")],
        default=None,
        help="explicit caller seeds, e.g. 0,2 (overrides --repeats)",
    )
    r.add_argument("--run-id", default=None, help="default: <model>-<timestamp>")
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument(
        "--temperature", type=float, default=None, help="agent temperature; only sent when given"
    )
    r.add_argument("--caller-temperature", type=float, default=0.3)
    r.add_argument("--retries", type=int, default=3, help="transport retries per model request")
    r.add_argument("--backoff", type=float, default=1.0, help="seconds; doubles per retry")
    r.add_argument("--caller-base-url", default=None, help="default: HOSTED_LLM_BASE_URL")
    r.add_argument("--caller-model", default=None, help="default: HOSTED_LLM_MODEL")
    r.add_argument(
        "--prompt",
        default=None,
        help="system prompt file (default order: services/agent, schemas, eval/prompts)",
    )
    r.add_argument("--no-db", action="store_true", help="skip Postgres writes")
    r.add_argument("--database-url", default=None, help="default: DATABASE_URL")
    r.add_argument("--only", default=None, help="regex on scenario id")
    v = sub.add_parser("validate", help="check scenario files against schema.json and the seed")
    v.add_argument("directory")
    rs = sub.add_parser("rescore", help="re-run the scorers over a run directory's saved episodes")
    rs.add_argument("directory")
    rs.add_argument(
        "--scenarios",
        default=None,
        help="scenario dir to re-read expectations from (default: as saved)",
    )
    rs.add_argument(
        "--db", action="store_true", help="replace the run's eval_results rows in Postgres"
    )
    rs.add_argument("--database-url", default=None)
    c = sub.add_parser(
        "compare", help="side-by-side all-pass rates per family for several run directories"
    )
    c.add_argument("runs", nargs="+")
    c.add_argument(
        "--exclude-caller-faults",
        action="store_true",
        help="drop scenario x seed episodes where the simulated caller mis-stated its credentials in any run",
    )
    f = sub.add_parser(
        "failures", help="dump every failing episode with scorer, turn and transcript context"
    )
    f.add_argument("directory")
    f.add_argument("--out", default=None, help="write the markdown here (default: stdout)")
    b = sub.add_parser(
        "buckets",
        help="first-pass failure buckets (data / model / application / harness) for a run directory",
    )
    b.add_argument("directory")
    b.add_argument(
        "--dataset",
        default=None,
        help="training dataset dir; per-tool target counts decide data gaps",
    )
    b.add_argument("--out", default=None)
    return p


def resolve_api_key(explicit: str | None, base_url: str) -> str:
    if explicit:
        return explicit
    if base_url.rstrip("/") == (os.environ.get("HOSTED_LLM_BASE_URL") or "").rstrip("/"):
        return os.environ.get("HOSTED_LLM_API_KEY") or "none"
    return os.environ.get("R3_AGENT_API_KEY") or "none"


def _cmd_validate(a: argparse.Namespace) -> int:
    problems = validate_dir(a.directory)
    n = len([p for p in Path(a.directory).iterdir() if p.suffix in SCENARIO_SUFFIXES])
    if not problems:
        print(f"{a.directory}: {n} scenario files valid")
        return 0
    for name, issues in problems.items():
        print(f"{name}:")
        for i in issues:
            print(f"  - {i}")
    print(f"{len(problems)} of {n} files have problems", file=sys.stderr)
    return 1


def _cmd_run(a: argparse.Namespace) -> int:
    scenarios = load_scenarios(a.scenarios)
    if a.only:
        scenarios = [s for s in scenarios if re.search(a.only, s.id)]
    if not scenarios:
        print(f"no scenarios in {a.scenarios}", file=sys.stderr)
        return 2
    run_id = (
        a.run_id or f"{re.sub(r'[^A-Za-z0-9._-]+', '-', a.model)}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    out = Path(a.out)
    caller_url = a.caller_base_url or os.environ["HOSTED_LLM_BASE_URL"]
    caller_model = a.caller_model or os.environ["HOSTED_LLM_MODEL"]
    agent_client = make_client(a.base_url, resolve_api_key(a.api_key, a.base_url))
    caller_client = make_client(caller_url, resolve_api_key(None, caller_url))

    def agent_factory() -> OpenAIAgent:
        return OpenAIAgent(
            client=agent_client,
            model=a.model,
            temperature=a.temperature,
            retries=a.retries,
            backoff_s=a.backoff,
        )

    def caller_factory(sc: Scenario, seed: int) -> LLMCaller:
        return LLMCaller(
            sc,
            client=caller_client,
            model=caller_model,
            temperature=a.caller_temperature,
            seed=seed,
        )

    print(
        f"run {run_id}: {len(scenarios)} scenarios x {a.repeats} repeats -> {out}", file=sys.stderr
    )
    results = asyncio.run(
        run_scenarios(
            scenarios,
            agent_factory,
            caller_factory,
            out_dir=out,
            run_id=run_id,
            model_id=a.model,
            parallel=a.parallel,
            repeats=a.repeats,
            prompt_path=a.prompt,
            seeds=a.seeds,
            progress=lambda line: print(line, file=sys.stderr),
        )
    )
    if not a.no_db:
        import psycopg

        url = a.database_url or os.environ.get("DATABASE_URL")
        if not url:
            print(
                "DATABASE_URL not set; skipping Postgres (use --no-db to silence)", file=sys.stderr
            )
        else:
            with psycopg.connect(url) as conn:
                counts = write_db(conn, results, run_id=run_id, model_id=a.model)
                conn.commit()
            print(f"postgres: {counts}", file=sys.stderr)
    md = (out / "summary.md").read_text()
    print(md.split("## Episodes")[0])
    print(f"summary: {out / 'summary.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    a = build_parser().parse_args(argv)
    if a.command == "validate":
        return _cmd_validate(a)
    if a.command == "compare":
        print(compare_runs(a.runs, exclude_caller_faults=a.exclude_caller_faults))
        return 0
    if a.command == "buckets":
        md = buckets_report(a.directory, dataset_dir=a.dataset)
        if a.out:
            Path(a.out).write_text(md)
            print(f"buckets -> {a.out}")
        else:
            print(md)
        return 0
    if a.command == "failures":
        md = failures_report(a.directory)
        if a.out:
            Path(a.out).write_text(md)
            print(f"failures -> {a.out}")
        else:
            print(md)
        return 0
    if a.command == "rescore":
        if a.db:
            import psycopg

            with psycopg.connect(a.database_url or os.environ["DATABASE_URL"]) as conn:
                n = rescore_dir(a.directory, a.scenarios, conn=conn)
                conn.commit()
            print("eval_results rows replaced", file=sys.stderr)
        else:
            n = rescore_dir(a.directory, a.scenarios)
        print(f"rescored {n} episodes -> {Path(a.directory) / 'summary.md'}")
        return 0
    return _cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
