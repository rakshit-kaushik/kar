"""r3-eval: score any OpenAI-compatible endpoint on a scenario directory.

    uv run r3-eval --base-url http://127.0.0.1:8001/v1 --model r3-voice-base \
        --scenarios eval/scenarios/test/ --out $R3_ARTIFACTS/runs/<id>/ [--parallel 4]

Per episode: fresh Sandbox(fixture) -> hosted caller vs. agent under test -> nine scorers.
Writes <out>/episodes/*.json, <out>/scores.jsonl, <out>/summary.md and, unless --no-db,
rows to eval_results / sessions / cost_ledger in Postgres.
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
from eval.scorers import WEIGHTS, ScorerResult, score_episode
from eval.simulator.clients import LLMCaller, OpenAIAgent, make_client
from eval.simulator.episode import AgentUnderTest, Caller, Episode, run_episode
from eval.simulator.scenario import Scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[1]
Result = tuple[Episode, list[ScorerResult]]


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
            "family": ep.family,
            "scorer": s.scorer,
            "pass": s.passed,
            "detail": s.detail,
        }
        for s in scores
    ]


async def run_scenarios(
    scenarios: list[Scenario],
    agent_factory: Callable[[], AgentUnderTest],
    caller_factory: Callable[[Scenario], Caller],
    *,
    out_dir: str | Path,
    run_id: str,
    model_id: str,
    parallel: int = 4,
    episodes_per_scenario: int = 1,
    max_turns: int = 20,
    progress: Callable[[str], None] | None = None,
) -> list[Result]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    scores_path.write_text("")
    sem = asyncio.Semaphore(max(1, parallel))
    results: list[Result] = []

    async def one(sc: Scenario, k: int) -> None:
        async with sem:
            sb = Sandbox(seed=sc.seed, today=sc.today, fixture=sc.fixture.model_dump())
            ep = await run_episode(
                sc,
                agent_factory(),
                caller_factory(sc),
                sb,
                model_id=model_id,
                run_id=run_id,
                max_turns=max_turns,
            )
            scores = score_episode(ep, sc)
            name = sc.id if episodes_per_scenario == 1 else f"{sc.id}-{k}"
            save_scored(ep, scores, out / "episodes" / f"{name}.json")
            with scores_path.open("a") as fh:
                for row in score_rows(ep, scores):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append((ep, scores))
            if progress:
                failed = [s.scorer for s in scores if not s.passed]
                progress(
                    f"{name:40} {ep.outcome:10} turns={len(ep.turns):2} "
                    f"{'ALL PASS' if not failed else 'FAIL ' + ','.join(failed)}"
                )

    await asyncio.gather(
        *(one(sc, k) for sc in scenarios for k in range(1, episodes_per_scenario + 1))
    )
    results.sort(key=lambda r: (r[0].scenario_id, r[0].episode_id))
    (out / "summary.md").write_text(summarize(results, run_id=run_id, model_id=model_id))
    return results


# ----- summary -----------------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return "-" if d == 0 else f"{round(100 * n / d)}%"


def summarize(results: list[Result], *, run_id: str, model_id: str) -> str:
    scorers = list(WEIGHTS)
    by_family: dict[str, list[Result]] = defaultdict(list)
    for ep, scores in results:
        by_family[ep.family].append((ep, scores))
    n = len(results)
    all_pass = sum(1 for _, s in results if all(x.passed for x in s))
    lines = [
        f"# r3-eval summary — run `{run_id}`",
        "",
        (
            f"model: `{model_id}` · episodes: {n} · scenarios: {len({ep.scenario_id for ep, _ in results})}"
            f" · generated {datetime.now().isoformat(timespec='seconds')}"
        ),
        "",
        f"**All scorers pass: {all_pass}/{n} ({_pct(all_pass, n)})**",
        "",
        "## Per family",
        "",
        "| family | episodes | all scorers pass | " + " | ".join(scorers) + " |",
        "|---|---|---|" + "---|" * len(scorers),
    ]
    for fam in sorted(by_family):
        rs = by_family[fam]
        ap = sum(1 for _, s in rs if all(x.passed for x in s))
        cells = [
            _pct(sum(1 for _, s in rs if next(x for x in s if x.scorer == name).passed), len(rs))
            for name in scorers
        ]
        lines.append(f"| {fam} | {len(rs)} | {_pct(ap, len(rs))} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Ranked failures (frequency × cost)",
        "",
        "| scorer | failures | weight | cost | example scenarios |",
        "|---|---|---|---|---|",
    ]
    fails: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for ep, scores in results:
        for s in scores:
            if not s.passed:
                fails[s.scorer] += 1
                if len(examples[s.scorer]) < 4 and ep.scenario_id not in examples[s.scorer]:
                    examples[s.scorer].append(ep.scenario_id)
    ranked = sorted(fails.items(), key=lambda kv: (-kv[1] * WEIGHTS[kv[0]], -kv[1], kv[0]))
    for name, cnt in ranked:
        lines.append(
            f"| {name} | {cnt} | {WEIGHTS[name]} | {cnt * WEIGHTS[name]} | {', '.join(examples[name])} |"
        )
    if not ranked:
        lines.append("| (none) | 0 | - | 0 | |")
    outcomes = Counter(ep.outcome for ep, _ in results)
    turns = [len(ep.turns) for ep, _ in results]
    lat = sorted(x for ep, _ in results for x in ep.latency_ms)
    agent_pt = sum(ep.usage.get("agent", {}).get("prompt_tokens", 0) for ep, _ in results)
    agent_ct = sum(ep.usage.get("agent", {}).get("completion_tokens", 0) for ep, _ in results)
    reqs = sum(ep.n_model_requests for ep, _ in results)
    lines += [
        "",
        "## Episodes",
        "",
        f"- outcomes: {dict(outcomes)}",
        f"- caller turns per episode: mean {sum(turns) / n:.1f}, max {max(turns)}"
        if n
        else "- no episodes",
        f"- agent model requests: {reqs}; prompt tokens {agent_pt}; completion tokens {agent_ct}",
        f"- agent request latency ms: p50 {lat[len(lat) // 2]:.0f}, p95 {lat[int(len(lat) * 0.95) - 1 if len(lat) > 1 else 0]:.0f}"
        if lat
        else "- no latency samples",
    ]
    errors = [(ep.scenario_id, ep.error) for ep, _ in results if ep.error]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {sid}: {err}" for sid, err in errors]
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
                        Jsonb(s.detail),
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
            "insert into cost_ledger (run_id, model_id, item, qty, unit, measured, note)"
            " values (%s, %s, %s, %s, %s, %s, %s)",
            rows,
        )
        counts["cost_ledger"] += len(rows)
    return counts


# ----- CLI -----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r3-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", required=True, help="agent under test (OpenAI-compatible)")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--api-key",
        default=None,
        help="default: HOSTED_LLM_API_KEY for the hosted URL, else 'none'",
    )
    p.add_argument("--scenarios", required=True, help="directory of .yaml/.json scenarios")
    p.add_argument("--out", required=True, help="run directory, e.g. $R3_ARTIFACTS/runs/<id>/")
    p.add_argument("--run-id", default=None, help="default: <model>-<timestamp>")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--episodes-per-scenario", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=None, help="only sent when given")
    p.add_argument("--caller-base-url", default=None, help="default: HOSTED_LLM_BASE_URL")
    p.add_argument("--caller-model", default=None, help="default: HOSTED_LLM_MODEL")
    p.add_argument("--no-db", action="store_true", help="skip Postgres writes")
    p.add_argument("--database-url", default=None, help="default: DATABASE_URL")
    p.add_argument("--only", default=None, help="regex on scenario id")
    return p


def resolve_api_key(explicit: str | None, base_url: str) -> str:
    if explicit:
        return explicit
    if base_url.rstrip("/") == (os.environ.get("HOSTED_LLM_BASE_URL") or "").rstrip("/"):
        return os.environ.get("HOSTED_LLM_API_KEY") or "none"
    return os.environ.get("R3_AGENT_API_KEY") or "none"


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    a = build_parser().parse_args(argv)
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

    def caller_factory(sc: Scenario) -> LLMCaller:
        return LLMCaller(sc, client=caller_client, model=caller_model, max_turns=a.max_turns)

    print(
        f"run {run_id}: {len(scenarios)} scenarios x {a.episodes_per_scenario} -> {out}",
        file=sys.stderr,
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
            episodes_per_scenario=a.episodes_per_scenario,
            max_turns=a.max_turns,
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


if __name__ == "__main__":
    sys.exit(main())
