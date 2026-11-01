# R3 RUNBOOK

## Processes and ports
| Process | Port | Start | Check |
| --- | --- | --- | --- |
| Postgres 15 (Homebrew, native) | 5432 | already running as a brew service | `psql $DATABASE_URL -c 'select 1'` |
| Model server (vllm-mlx) | 8000 | `scripts/restart.sh` (S fills in) | `curl :8000/v1/models` |
| Proxy (tool-call delta normalizer, `<think>` guard, hosted fallback) | 8001 | same | `curl :8001/health` (shows upstream and fallback) |
| LiveKit agent worker | — | `uv run python -m services.agent.main dev` | logs show `registered worker` |
| UI | 3000 | external Astra branch | browser |

`.env` is the single source of config. `LLM_BASE_URL` / `LLM_MODEL` are the active endpoint; `scripts/gpu_flip.sh hosted|local` rewrites them.

## Server configuration (decided at G1 by measurement, Nov 1)
`vllm-mlx serve <model> --prefix-trie-cache --default-chat-template-kwargs '{"enable_thinking": false}'` — the single-request engine with the mlx-lm prefix trie cache. Spoken-turn TTFT 0.10 s, 13 ms/token; the system prompt + tool schemas (1,948 tokens) are reused on every request ("System KV cache HIT"). Continuous batching (`--continuous-batching`, with or without `--use-paged-cache`) gave 0.58–0.75 s TTFT because its prefix cache almost never hit on repeated prompts. Revisit only if the T+8 load test shows queueing at 4 calls.

The proxy on 8001 is permanent: vllm-mlx 0.5.0 emits each streamed tool call twice (different ids) and LiveKit would execute both; the proxy dedupes, strips `<think>` and `[Calling tool…]` preambles, converts leaked raw `<tool_call>` text, and writes `llm_request/llm_first_token/llm_done` to `turn_events` keyed by the agent's `X-R3-Call-Id` header.

## Hosted fallback (proxy, Nov 1)
Per turn, the proxy sends the request to the local server and waits for the first token. If the server returns an error or no first token arrives within `PROXY_FIRST_TOKEN_TIMEOUT_MS` (default 2000; `.env`), the same request is re-issued to `HOSTED_LLM_BASE_URL` as `HOSTED_LLM_MODEL` before any byte reaches the agent, so the switch happens only at a turn boundary and call state is untouched. Events: `llm_error` then `fallback {reason: upstream_error|first_token_timeout, from_model, to_model}`; the turn's `llm_first_token`/`llm_done` carry the hosted `model_id`, and the response has `X-R3-Model`. `PROXY_FALLBACK=0` disables it. The fallback rate is a reported metric (plan §Measurements); the timeout is recalibrated from the T+8 load test (p99 TTFT at the admission limit). `scripts/restart.sh` prewarms the server directly on :8000 so the ~19 s cold start never trips the fallback.

## Prompt and tools (contract commit aa7893a)
The agent, the dry run, the bench and the smoke test all load `schemas/system_prompt.md` (shared byte-for-byte with E and T; `{today}` = `seed.json["clock_spoken"]`) and the seven tools in `schemas/tools/`. `block_card` is cut: lost or stolen card = `create_ticket(card_lost_stolen)` + `request_handoff`.

## Calls and reports
- `uv run python scripts/join_token.py <room>` → browser link; the agent auto-joins. `~/r3-artifacts/g1_links.txt` has ten.
- `uv run python scripts/dry_run_agent.py --repeats 4 --quiet` → text-only loop check through the proxy (no audio).
- `uv run python scripts/g1_report.py [--ledger]` → per-stage p50/p95 and per-call outcomes from `turn_events`.
- `uv run python scripts/bench_base.py` → single-stream MLX throughput; `scripts/smoke_tools.py [--turn2]` → streamed tool-call check.

## GPU schedule (hours from T+0) — the schedule is law
Without a human arbitrating, `scripts/gpu_lock.sh` is the schedule: `acquire <owner> <minutes> [purpose]` before a training run, benchmark or load test; `release <owner>` after; `status` to look. Live calls on the served model need no lock. Say so in a cross-session message when you take or return it. Stop the server only with `scripts/server_stop.sh` (kills the :8000 listener, never connected clients; a bare `lsof -ti:8000 | xargs kill` once took E's eval client down). `scripts/restart.sh --agent-only` restarts the LiveKit worker alone, touches nothing shared and needs no lock.

Call checks: `uv run python scripts/probe_dispatch.py` joins a room and confirms the agent is dispatched and greets (run after every agent restart and before handing out links). The agent now ends its job and deletes the room when the caller leaves, so a rejoin never meets a leftover agent.
| Window | Owner | Everyone else points at |
| --- | --- | --- |
| T+1–4 | S skeleton | local |
| T+4–7 | S benchmarks, prefix cache, KV persistence | local |
| T+7–8 | T smoke adapter (G2) | hosted |
| T+8–10 | S load tests | local |
| T+10–12 | T run 1 | hosted |
| T+12–13 | eval run 1, precision gate | local |
| T+13–13.5 | T routing adapter | hosted |
| T+13.5–18 | S deploy, fallback, routing job | local |
| T+18–19.5 | T run 2 (only if labels justify) | hosted |
| T+19.5–24 | eval, G4 at T+20.5, rehearsal | local |

## Gates
G1 T+4 live call + tool lookup on base, latency in Postgres · G2 T+8 smoke adapter end to end · G3 T+13 candidate deployed under versioned id · G4 T+20.5 final model frozen.

## Cut order when behind
routing track → speculative decoding → run 2 → Next.js/Astra UI extras → cost-ledger automation → interruption slice → audio baseline.

## Artifacts
`$R3_ARTIFACTS` = `~/r3-artifacts`: `models/` (base, adapters, fused), `datasets/` (seeds: taskmaster, sgd; generated JSONL + manifests), `runs/` (training and eval runs by id).

## Recovery
- Rollback a model: point the server at the previous `models/r3-voice-<hash>` directory and restart.
- Fallback on the fly: `scripts/gpu_flip.sh hosted`.
- Postgres reset of sandbox tables only: `eval/sandbox/reset.py` (E fills in).
