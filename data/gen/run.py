"""Generate scenarios, run them with the hosted teacher as agent, filter and build a dataset.

    uv run python -m data.gen.run --n 50 --seed 0 --out $R3_ARTIFACTS/datasets/r3-voice-gen-smoke-1 \
        [--parallel 20] [--mix ordinary=4,correction=1,...] [--name r3-voice-gen-smoke-1] [--no-length-check]

Layout under --out: scenarios/ (the generated scenario files), run/ (episodes, scores, summary),
and the dataset files train.jsonl, valid.jsonl, manifest.json, yield.md at the top level.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from data.format.build import build_dataset
from data.gen.scenarios import generate, write_dir
from eval.runner import run_scenarios
from eval.simulator.clients import LLMCaller, OpenAIAgent, make_client
from eval.simulator.scenario import Scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[2]


def parse_mix(s: str | None) -> dict[str, float] | None:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, required=True, help="scenarios to generate (one episode each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None, help="dataset name (default: the --out directory name)")
    p.add_argument(
        "--mix", type=parse_mix, default=None, help="family weights, e.g. ordinary=4,correction=1"
    )
    p.add_argument("--parallel", type=int, default=20)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--teacher-base-url", default=None, help="default: HOSTED_LLM_BASE_URL")
    p.add_argument("--teacher-model", default=None, help="default: HOSTED_LLM_MODEL")
    p.add_argument(
        "--no-length-check", action="store_true", help="skip the 2,048-token filter (no tokenizer)"
    )
    p.add_argument("--scenarios-only", action="store_true", help="write the scenarios and stop")
    p.add_argument(
        "--profile",
        default="default",
        choices=[
            "default",
            "short",
            "double_fault",
            "double_fault_short",
            "frontload_decline",
            "v51",
        ],
        help="short = status/cancel/handoff calls only",
    )
    p.add_argument(
        "--no-build", action="store_true", help="run the episodes, skip the dataset build"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    name = a.name or out.name
    scenarios = generate(a.n, seed=a.seed, mix=a.mix, profile=a.profile)
    write_dir(scenarios, out / "scenarios")
    loaded = load_scenarios(out / "scenarios")
    print(f"{len(loaded)} scenarios -> {out / 'scenarios'}", file=sys.stderr)
    if a.scenarios_only:
        return 0
    url = a.teacher_base_url or os.environ["HOSTED_LLM_BASE_URL"]
    model = a.teacher_model or os.environ["HOSTED_LLM_MODEL"]
    key = (
        os.environ.get("HOSTED_LLM_API_KEY")
        if url.rstrip("/") == os.environ.get("HOSTED_LLM_BASE_URL", "").rstrip("/")
        else None
    )
    teacher_client = make_client(url, key)
    caller_client = make_client(
        os.environ["HOSTED_LLM_BASE_URL"], os.environ.get("HOSTED_LLM_API_KEY")
    )

    def agent_factory() -> OpenAIAgent:
        return OpenAIAgent(client=teacher_client, model=model)

    def caller_factory(sc: Scenario, seed: int) -> LLMCaller:
        return LLMCaller(
            sc,
            client=caller_client,
            model=os.environ["HOSTED_LLM_MODEL"],
            seed=a.seed * 1000 + seed,
        )

    asyncio.run(
        run_scenarios(
            loaded,
            agent_factory,
            caller_factory,
            out_dir=out / "run",
            run_id=name,
            model_id=model,
            parallel=a.parallel,
            repeats=1,
            progress=lambda line: print(line, file=sys.stderr),
        )
    )
    if a.no_build:
        return 0
    fits = None
    if not a.no_length_check:
        from data.format.length import LengthChecker

        fits = LengthChecker().fits
    stats = build_dataset(
        out / "run", out, name=name, fits=fits, valid_frac=a.valid_frac, seed=a.seed
    )
    print((out / "yield.md").read_text())
    print(f"dataset: {out} ({stats['episodes']['kept']} episodes kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
