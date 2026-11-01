You are session S (serving) of the R3 sprint. Work only in this worktree, branch `s/skeleton`.

Read first: `CONTRACT.md` (all), `schemas/`, `RUNBOOK.md`, then "T+1 to T+4, session S: walking skeleton" and "T+4 to T+18, session S" in `master-build-plan.md`. You own the Mac GPU in the windows the schedule marks S; never run a model process in a T window.

## Exit criterion (G1, T+4)
Ten live browser calls complete `lookup_account` on the untuned Qwen3-4B-8bit, nothing structural (`<tool_call>`, `<think>`, JSON) is ever spoken, and per-stage p50/p95 for every event in `schemas/events.json` are in `turn_events`, with the baseline row in `cost_ledger`.

## Build, in order
1. Server: vllm-mlx serving `$R3_ARTIFACTS/models/Qwen3-4B-8bit` on 8000 with continuous batching and prefix cache; stream a chat completion with the seven tools; confirm tool calls arrive as OpenAI `tool_calls` deltas. If they arrive as raw text, build `services/proxy/` on 8001 (one hour, permanent): parses Qwen3 `<tool_call>` while streaming, strips `<think>`, forwards everything else. Thinking disabled via chat-template flag AND the proxy guard.
2. `services/tools/`: the eight tools as LiveKit function tools wrapping session E's sandbox (`eval/sandbox/`; stub with deterministic data if not merged), with idempotency keys and `schemas/tools/*.json` as the schema source.
3. `services/agent/`: LiveKit `AgentSession` — Deepgram STT, Cartesia TTS (`CARTESIA_*` from `.env`), Silero VAD, LiveKit turn detector, OpenAI LLM plugin at `LLM_BASE_URL`; system prompt = the contract's §1 catalog + §2 policy; state per `schemas/state.json`; every event in `schemas/events.json` written to `turn_events`; session checkpointed to `sessions`. LiveKit's example web client for the browser.
4. Ten calls, baseline row, G1 report.

## Rules
- Read installed sources in `.venv/lib/python3.12/site-packages/` (`vllm_mlx`, `mlx_lm`, `livekit/agents`, `livekit/plugins/*`) before calling any API; report version mismatches; never invent flags.
- Files you may touch: `services/**`, `scripts/restart.sh`, `db/migrations/002+`, `tests/services/**`. Not `eval/`, `train/`, `schemas/`, `CONTRACT.md`.
- Hand load tests and benchmark sweeps to subagents.
