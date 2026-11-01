# R3 RUNBOOK

## Processes and ports
| Process | Port | Start | Check |
| --- | --- | --- | --- |
| Postgres 15 (Homebrew, native) | 5432 | already running as a brew service | `psql $DATABASE_URL -c 'select 1'` |
| Model server (vllm-mlx) | 8000 | `scripts/restart.sh` (S fills in) | `curl :8000/v1/models` |
| Proxy (tool-call delta normalizer, `<think>` guard, hosted fallback) | 8001 | same | `curl :8001/health` (shows upstream and fallback) |
| LiveKit agent worker | — | `uv run python -m services.agent.main dev` | logs show `registered worker` |
| UI | 3000 | external Astra branch | browser |

`.env` is the single source of config. `LLM_BASE_URL` / `LLM_MODEL` are the active endpoint; `scripts/gpu_flip.sh hosted|local` rewrites them. T's run launcher executes gpu_flip, server_stop, restart and the probe from this checkout's working tree (`--services-root`), so the tree must stay runnable at all times; stage risky edits in a copy. Check `git branch --show-current` before committing: on Nov 1 this checkout had been switched to `d/demo` (the user's UI branch; the untracked `ui/` files are theirs) and five lane-S commits landed there before main was fast-forwarded.

## Server configuration (decided at G1 by measurement, Nov 1)
`vllm-mlx serve <model> --prefix-trie-cache --default-chat-template-kwargs '{"enable_thinking": false}'` — the single-request engine with the mlx-lm prefix trie cache. Spoken-turn TTFT 0.10 s, 13 ms/token; the system prompt + tool schemas (1,948 tokens) are reused on every request ("System KV cache HIT"). Continuous batching (`--continuous-batching`, with or without `--use-paged-cache`) gave 0.58–0.75 s TTFT because its prefix cache almost never hit on repeated prompts.

**Load test (Nov 1, `scripts/load_text.py`, HTTP level through the proxy, real prompt size 1,575 tokens, tool-call turns):** the SimpleEngine serializes generation, so a second concurrent request waits a full turn. A tool-call turn takes 1.1–1.2 s end to end (49 tokens at ~41 tok/s; vllm-mlx emits a tool call only when it is complete, so for tool turns "first token" = whole turn); at 2 concurrent the second request's first token comes at ~2.4 s, at 4 the fourth at ~4.8 s. **Admission limit = 1 concurrent call** (`cost_rates.admission.concurrent_calls`, ledger run `load-text-base`). Consequences: `PROXY_FIRST_TOKEN_TIMEOUT_MS=3000` (2000 turned ordinary queued turns into hosted fallbacks), `PROXY_ADMISSION_LIMIT=1` (a turn arriving while one local generation is in flight goes to the hosted model when available, event `admission`). Continuous batching is the only route to several concurrent local calls and costs 0.5 s of TTFT at 1 call; the demo is one live call at a time, so it stays off.

The proxy on 8001 is permanent: vllm-mlx 0.5.0 emits each streamed tool call twice (different ids) and LiveKit would execute both; the proxy dedupes, strips `<think>` and `[Calling tool…]` preambles, converts leaked raw `<tool_call>` text, and writes `llm_request/llm_first_token/llm_done` to `turn_events` keyed by the agent's `X-R3-Call-Id` header.

## Hosted fallback (proxy, Nov 1)
Per turn, the proxy sends the request to the local server and waits for the first token. If the server returns an error or no first token arrives within `PROXY_FIRST_TOKEN_TIMEOUT_MS` (default 2000; `.env`), the same request is re-issued to `HOSTED_LLM_BASE_URL` as `HOSTED_LLM_MODEL` before any byte reaches the agent, so the switch happens only at a turn boundary and call state is untouched. Events: `llm_error` then `fallback {reason: upstream_error|first_token_timeout, from_model, to_model}`; the turn's `llm_first_token`/`llm_done` carry the hosted `model_id`, and the response has `X-R3-Model`. `PROXY_FALLBACK=0` disables it. Hosted requests go without `temperature`, `top_p`, `max_tokens` and `max_completion_tokens`: the hosted endpoint rejects the first two and its hidden reasoning consumes the token budget (a 96-token budget returned `finish_reason=length` with no visible token; E saw the same on routing). A hosted 4xx (bad key, no credit: on Nov 1 the Anthropic account ran out of credit mid-sprint) opens a 10-minute breaker: the abandoned local request is re-issued without a timeout, the proxy waits on local instead of falling back while the breaker is open, and `/health` shows `fallback.available=false` with the reason; event `fallback_unavailable`. The fallback rate is a reported metric (plan §Measurements); the timeout is recalibrated from the T+8 load test (p99 TTFT at the admission limit). `scripts/restart.sh` prewarms the server directly on :8000 so the ~19 s cold start never trips the fallback.

## Prompt and tools (contract commit aa7893a)
The agent, the dry run, the bench and the smoke test all load `schemas/system_prompt.md` (shared byte-for-byte with E and T; `{today}` = `seed.json["clock_spoken"]`) and the seven tools in `schemas/tools/`. `block_card` is cut: lost or stolen card = `create_ticket(card_lost_stolen)` + `request_handoff`.

## Calls and reports
- `uv run python scripts/sim_caller.py --calls N [--concurrency C] [--script fixed|llm] [--prefix r3-sim]` → audio-level simulated caller (Cartesia voice "Daniel" into a LiveKit room, waits for the agent to be audibly quiet, measures end of caller speech → first audible agent frame). `--prepare` caches the fixed Dana lines; `--listener` adds a loopback client that measures capture → heard for the caller's own speech (pure transport). Caller events: `caller_speech_start/_end`, `caller_first_audio`, `caller_summary` under the room name.
- `SMOKE_MODEL_DIR=$R3_ARTIFACTS/models/r3-voice-<hash> scripts/g1_block.sh` → the whole GPU block under the lock: G2 serve + call, 10 G1 calls, load tests 1/2/4/8, turn-taking A/B; reports and ledger rows per step in `$R3_ARTIFACTS/runs/g1-block-*/`.
- Agent turn-taking knobs (`AGENT_VAD_MIN_SILENCE`, `AGENT_ENDPOINT_MIN/MAX`, `AGENT_PREEMPTIVE`, see .env.example) are read at agent start; `scripts/restart.sh --agent-only` applies them without touching the server. VAD 0.3 s split spoken phone numbers at their pauses (the turn detector accepted "five five five zero one four" as a complete turn); do not go below 0.4.
- `nohup scripts/agent_watchdog.sh &` → probes dispatch every 5 min, restarts the worker alone when the process is gone or two probes fail with no caller in any room (`scripts/rooms.py --busy`). Log: `$R3_ARTIFACTS/logs/watchdog.log`.
- `uv run python -m services.routing.batch --model-dir <routing model dir> [--limit N] [--dry-run]` → labels unlabeled tickets (greedy, 8 tokens, validated; invalid → needs_review), writes labels + a ledger row; takes the GPU lock as owner R and skips if held.
- `uv run python scripts/load_text.py --levels 1 2 4 8 --waves 3 [--ledger --run-id load-text-<model>]` → HTTP-level load test through the proxy, no audio vendors; TTFT / inter-token p50/p95 per concurrency and the admission limit (p50 ≤ 350 ms and p95 ≤ 600 ms). Run it on every deployed model.
- `uv run python scripts/cost_report.py [--like 'r3-g1-%'] [--eval-run <run_id> ...] [--set-admission N] [--ledger --run-id cost-<tag>]` → cost per call and per successfully completed task for every served model (audio vendors from measured minutes/characters at list price; local LLM = Mac hourly cost / calls per hour at the admission limit, estimated; hosted = measured tokens at list price) plus E's text-eval runs from their ledger rows. Rates live in Postgres `cost_rates` (migration 002: kind list / assumption / measured, source URL and date per row; override one row to change an assumption). Mac rate: $4,099 / 26,280 h (3 years around the clock) + 0.10 kW × $0.15 = $0.171/h.
- `uv run python scripts/replay.py --call-id <room> | --turns-file demo/turns/<name>.json --model name=model@base_url ...` → a recorded call's caller turns replayed in text through any model with the real tools and a fresh sandbox per model; side-by-side markdown + JSON under `runs/replay-*`. Labelled replay on stage, never a live call.
- `demo/scenarios.md` + `demo/turns/*.json` → the five scripted demo scenarios (ordinary, correction, tool timeout, interruption, handoff) with expected tool calls and final state; `sim_caller.py --lines-file demo/turns/<name>.json` speaks them. A room name (or `--prefix`) containing `--fault-<tool>-<error>[-<n>]` injects a sandbox fault for the live timeout demo (services/agent/faults.py).
- `uv run python -m services.routing.batch --model-dir $R3_ARTIFACTS/models/r3-routing-<hash> --gold-file $R3_ARTIFACTS/datasets/tickets-v1/tickets.jsonl` → labels the unlabeled tickets and scores the test split (accuracy, macro F1, coverage, 3x-weighted costly confusions) → ledger `routing.*`; `scripts/load_tickets.py <jsonl> --split test|train|all` loads E's tickets (category = gold). Prompt pinned byte-for-byte to the trained adapter (test).
- `scripts/backup.sh` → git bundle of all branches, pg_dump of r3, .env snapshot, deploy state (`BACKUP_MODELS=1` adds the deployed model dir) under `$R3_ARTIFACTS/backups/<stamp>/`. No second disk on this Mac: copy that directory elsewhere when one exists.
- `scripts/restart.sh --proxy-only` restarts the proxy alone (no lock needed, like `--agent-only`); use it to roll out proxy changes while another session holds the GPU.
- Network: the Mac was on a cellular hotspot on Nov 1 (en0 = 192.0.0.2 CLAT). TCP RTT to LiveKit median ~100 ms with 400 ms spikes; the Python rtc receiver then buffers 0.3–1.1 s. Caller-side "first audio heard" includes that; the agent-side `eos → tts_first_audible` does not. The demo needs wired or stable Wi-Fi. At 01:20 on Nov 1 the hotspot degraded for 10+ minutes (Deepgram websockets dropping every ~10 s, `STTError`, LiveKit signaling timeouts): calls lost their transcripts while the model server answered in 0.45 s, so the G1 block was stopped and the GPU handed to training; `g1_report` counts `error` events per call so a network-invalidated batch is visible.
- Clocks: DST fell back at 02:00 on Nov 1, so wall-clock times between 01:00 and 02:00 occur twice in logs; compare `ts_ms` epochs in `turn_events`, not log times.
- `uv run python scripts/join_token.py <room>` → browser link; the agent auto-joins. `~/r3-artifacts/g1_links.txt` has ten.
- `uv run python scripts/dry_run_agent.py --repeats 4 --quiet` → text-only loop check through the proxy (no audio).
- `uv run python scripts/g1_report.py [--like 'r3-g1-%'] [--ledger --run-id ...]` → per-stage p50/p95 (turn-taking overhead, LLM TTFT local/hosted, decode tok/s, first audible TTS on the agent clock, first audio heard on the caller clock, transport) and per-call outcomes from `turn_events`.
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

### Prefix cache: never hit until Nov 1 02:50 (root cause + fix)
Evidence: server.log showed `prefix_trie_cache=True` at start, zero "Prefix trie cache HIT" lines in 290 live turns, and
10-token spoken turns taking 0.46-0.55 s server-side (= full prefill of ~1.5K tokens). Cause: vllm-mlx 0.5.0 keys mlx-lm's
LRUPromptCache by the model object; an MLX nn.Module is a dict subclass and unhashable, so every insert and lookup raised
TypeError inside a try that counts a "skip" and logs at DEBUG. The System KV cache (system prefix only) hit only on
[system, user] prewarm requests: live prompts start with the assistant greeting, so the system-prefix check fails there.
Fix: services/server/trie_keyfix.py keys the trie by id(model); scripts/vllm_mlx_serve.py applies it and runs the CLI;
restart.sh starts the server through it with `--prefix-trie-cache-size 8 --prefix-trie-cache-memory-mb 2048`.
Expected effect (tokenizer experiment, tests/services/test_trie_keyfix.py): consecutive turns share all but the dropped
empty `<think>` block, so turn 2+ prefills 30-150 new tokens instead of 1.5-2.3K. Verify after a restart: a 2-turn
request pair against :8000 must log "Prefix trie cache HIT (stream_chat): reusing N tokens, prefilling M new" on the
second request (scripts/deploy_block.sh does this).

### Hosted fallback is a hedge (Nov 1, 02:20)
A local turn with no first token after `PROXY_FIRST_TOKEN_TIMEOUT_MS` (3000) no longer gets cancelled: the proxy starts the
hosted request beside it and serves whichever streams a token first (`fallback` event with `hedged: true`, then
`fallback_cancelled` on the loser). Reason: the engine finishes a dropped request anyway, so the old cancel-and-reissue
cost 5-8 s per turn when the hosted side then failed (credits exhausted, call r3-g1-0145-015841-1 turn 3). With the
breaker open (hosted 4xx) the proxy simply waits for local. `g1_report` counts a fallback only when the hosted stream was
served (`llm_first_token.source = hosted`).
