# R3

A phone agent for a fictional retail bank (Marlow Mutual): it verifies callers, books, moves and cancels branch appointments, opens tickets, and hands off to a person. The model is a LoRA adapter on Qwen3-4B, trained and served locally with MLX on one Mac. Speech runs through LiveKit with Deepgram speech-to-text and Deepgram or Cartesia text-to-speech. Built in a 24-hour sprint, Oct 31 – Nov 1, 2026.

## Start here

1. **[`results/`](results/README.md)**: what came out. Scores against the base and hosted models, latency, rehearsal, cost, and lessons learned.
2. **[`CONTRACT.md`](CONTRACT.md)**: the spec everything was built against: workflow, tools, policy, test families, schemas.
3. **[`RUNBOOK.md`](RUNBOOK.md)**: how to run the stack, plus a time-ordered log of decisions and incidents during the sprint.

The build plans (`master-build-plan.md`, `r3-24h-build-plan.md`) were written before the contract and describe a field-service business. The contract switched the workflow to bank branch appointments. Where they disagree, **`CONTRACT.md` wins**.

## Result in one line

The fine-tuned model (run 2, `r3-voice-730f8ba68e`) passed every scorer on 62% of test episodes, against 12% for the base model and 100% for the hosted model, with no unsupported success claims. Latency halved, but was still above target. Details and caveats are in [`results/`](results/README.md).

## Layout

| Path | What |
|---|---|
| `schemas/` | system prompt, the seven tool schemas, state and event schemas |
| `services/` | proxy (tool-call normalizer, hosted fallback), LiveKit agent, tools, demo helpers |
| `eval/` | caller simulator, sandbox, deterministic scorers, `r3-eval` runner; `eval/scenarios/test/` holds the 27 hand-written test scenarios |
| `data/` | training-data generation, STT shaping, formatting, splits |
| `train/` | mlx_lm.lora configs, run launcher, fuse/quantize, eval-on-checkpoint, selection |
| `scripts/` | restart, deploy, serving proof, GPU lock, simulated calls, rehearsal, write-up and cost reports |
| `demo/` | the scripted demo scenarios |
| `tests/` | `uv run pytest tests` |

Models, datasets and raw eval runs are not in git. They live in `$R3_ARTIFACTS` (`~/r3-artifacts`), so the numbers in `results/` can't be re-derived from this repo alone.

## Run

```sh
uv sync
cp .env.example .env            # fill in the keys
scripts/restart.sh              # Postgres check, model server, proxy, agent
uv run r3-eval run --base-url http://127.0.0.1:8001/v1 --model <id> \
    --scenarios eval/scenarios/test/ --repeats 3 --out <dir>
```
