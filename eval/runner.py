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
from eval.simulator.clients import LLMCaller, OpenAIAgent, make_client
from eval.simulator.episode import AgentUnderTest, Caller, Episode, run_episode
from eval.simulator.scenario import (
    SCENARIO_SUFFIXES,
    Scenario,
    load_scenario,
    load_scenarios,
    validate_against_seed,
)

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

    await asyncio.gather(*(one(sc, seed) for sc in scenarios for seed in range(repeats)))
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


def write_db(conn, results: list[Result], *, run_id: str, model_id: str) -> dict[str, int]:
    from psycopg.types.json import Jsonb

    counts = {"eval_results": 0, "sessions": 0, "cost_ledger": 0}
    with conn.cursor() as cur:
        for ep, scores in results:
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
            counts["eval_results"] += len(scores)
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
        rows = [(run_id, model_id, "episodes", len(results), "episode", True, "r3-eval run")]
        rows += [
            (run_id, model_id, item, qty, "token", True, "r3-eval measured usage")
            for item, qty in usage.items()
        ]
        cur.executemany(
            "insert into cost_ledger (run_id, model_id, item, qty, unit, measured, note) values (%s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
        counts["cost_ledger"] += len(rows)
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


def rescore_dir(directory: str | Path, scenarios_dir: str | Path | None = None) -> int:
    """Re-run the scorers over <dir>/episodes/*.json; rewrite scores.jsonl and summary.md. No DB."""
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
    return len(results)


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
    r.add_argument("--run-id", default=None, help="default: <model>-<timestamp>")
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument(
        "--temperature", type=float, default=None, help="agent temperature; only sent when given"
    )
    r.add_argument("--caller-temperature", type=float, default=0.3)
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
        return OpenAIAgent(client=agent_client, model=a.model, temperature=a.temperature)

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
    if a.command == "rescore":
        n = rescore_dir(a.directory, a.scenarios)
        print(f"rescored {n} episodes -> {Path(a.directory) / 'summary.md'}")
        return 0
    return _cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
