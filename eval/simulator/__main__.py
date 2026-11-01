"""Run one scenario live and print the transcript.

    uv run python -m eval.simulator eval/scenarios/test/ordinary_01.json \
        [--base-url URL --model ID] [--seed 0] [--prompt PATH] [--out DIR]

Defaults: agent = HOSTED_LLM_* (text baseline), caller = HOSTED_LLM_* at temperature 0.3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval.sandbox import Sandbox
from eval.simulator.clients import LLMCaller, OpenAIAgent, make_client
from eval.simulator.episode import run_episode
from eval.simulator.scenario import load_scenario


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scenario")
    p.add_argument("--base-url", default=os.environ.get("HOSTED_LLM_BASE_URL"))
    p.add_argument("--model", default=os.environ.get("HOSTED_LLM_MODEL"))
    p.add_argument(
        "--api-key",
        default=None,
        help="defaults to HOSTED_LLM_API_KEY for the hosted URL, else 'none'",
    )
    p.add_argument("--caller-base-url", default=os.environ.get("HOSTED_LLM_BASE_URL"))
    p.add_argument("--caller-model", default=os.environ.get("HOSTED_LLM_MODEL"))
    p.add_argument("--seed", type=int, default=0, help="caller variation seed")
    p.add_argument(
        "--prompt", default=None, help="system prompt file (default: see eval/simulator/prompts.py)"
    )
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    hosted_key = os.environ.get("HOSTED_LLM_API_KEY")
    agent_key = a.api_key or (
        hosted_key if a.base_url == os.environ.get("HOSTED_LLM_BASE_URL") else None
    )
    scenario = load_scenario(a.scenario)
    agent = OpenAIAgent(client=make_client(a.base_url, agent_key), model=a.model)
    caller = LLMCaller(
        scenario,
        client=make_client(a.caller_base_url, hosted_key),
        model=a.caller_model,
        seed=a.seed,
    )
    ep = asyncio.run(
        run_episode(
            scenario,
            agent,
            caller,
            Sandbox(),
            model_id=a.model,
            run_id="adhoc",
            repeat_seed=a.seed,
            prompt_path=a.prompt,
        )
    )
    for m in ep.messages[1:]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                print(f"  -> {tc['function']['name']}({tc['function']['arguments']})")
            if m.get("content"):
                print(f"A: {m['content']}")
        elif m["role"] == "user":
            print(f"C: {m['content']}")
        elif m["role"] == "tool":
            print(f"  <- {m['content'][:160]}")
    print(
        f"\noutcome={ep.outcome} error={ep.error} caller_turns={len(ep.turns)} requests={ep.n_model_requests} "
        f"truncated={ep.n_truncated} prompt={ep.prompt_source} ({ep.system_prompt_chars} chars, "
        f"first prompt_tokens={ep.first_prompt_tokens})"
    )
    print(f"usage={json.dumps(ep.usage)}")
    print(f"diff={json.dumps(ep.final_state['diff'])}")
    if a.out:
        print("saved", ep.save(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
