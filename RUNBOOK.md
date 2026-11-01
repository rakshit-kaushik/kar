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

### Nov 1 05:42: claim guard (reviewer item from the run-1 failure labels)
In 5 of 81 text episodes the candidate said "I've opened your ticket and I'm transferring you" right after create_ticket and
called request_handoff a turn late or never (the contract counts a transfer announced without the tool as an unsupported
claim; the same pattern as live call 2). `services/agent/claimguard.py` watches the streamed reply in `R3Agent.llm_node`
(outside the loop guard): the chunk that completes a transfer phrase waits until request_handoff has been recorded through
the executor (reason from the call state: suspected_fraud after an unauthorized_transaction ticket, verification_failed
after failed verification with no verified customer, else out_of_scope), then the sentence goes to the TTS. `claim_guard`
turn event {claimed, reason}; its own column (`cguard`) beside `guard` and `claims` in g1_report, `claim=` in demo_check,
ledger items g1.claim_guard_calls / g1.claim_guard_interventions and rehearsal.<scenario>.claim_guarded; the write-up's
latency and rehearsal tables show it. Identical for base, adapter and hosted; reported as the agent's intervention, never
as the model's recovery. `AGENT_CLAIM_GUARD=0` turns it off. Two side fixes: the live claims counter
(`services/common/claims.py`) now treats "I'm transferring you" as a claim about request_handoff and checks every claim in
a sentence (it used to stop at the first, so a supported ticket claim hid the transfer after it); the state schema was not
widened for the verification count (`state_view()` adds it for the guard only). Reviewer's second observation, for the
scripts: on a single not_verified the candidate transfers instead of asking again, so no demo scenario mis-states the
date of birth (none does: the five turn scripts give phone and DOB cleanly). `scripts/dry_run_agent.py --real-agent`
exercises the wiring in text.

### Nov 1 05:01: loop guard; demo rehearsal tooling
- Loop guard (services/agent/loopguard.py, in R3Agent.llm_node): a reply is held only while it is still identical to the previous
  assistant line; an exact repeat is swallowed and a rephrased re-ask is spoken, a second consecutive repeat becomes the handoff
  line plus request_handoff(out_of_scope) through the executor. `loop_guard` turn events {action, repeated, repeats}.
  AGENT_LOOP_GUARD=0 turns it off (agent-only restart). Why: E's eval found 6/81 run-1 episodes repeating a line 3+ times and the
  live calls showed it in 2 of 4; application logic is the cheapest fix and keeps T's eval serve config unchanged.
- Rehearsal: `RUNS=5 scripts/rehearsal.sh` (lock S, ~50 min) runs the five demo scenarios through the simulated caller with
  recordings, scores each call with scripts/demo_check.py (services/demo/check.py), latency + cost into the ledger under
  rehearsal-<model>-<stamp>; the write-up has a rehearsal table. Found on the way: a `--fault-...` spec in a sim_caller prefix was
  never parsed (batch suffix appended after it), so the timeout scenario had never been injected through sim_caller; fixed
  (services/sim/naming.py). measure_block.sh: INT_TURN (default 5) and TIMEOUT_CALL=1.

### Nov 1 04:49: filler line before tool-call turns (reviewer decision on the G3 numbers)
- Tool-call turns on the candidate have no audio for ~2 s (model writes the tool call 748 ms, tool runs, model speaks the result,
  TTS). The agent now says `AGENT_FILLER` ("One moment.") the instant a tool starts: once per caller turn, never while it is
  already speaking, never on spoken turns, not added to the chat context (the model never sees it). services/agent/filler.py,
  hook in services/tools/livekit_tools.py (`on_tool_start`), wired in services/agent/main.py (`_tool_start`). Empty
  AGENT_FILLER switches it off; agent-only restart to apply (`scripts/restart.sh --agent-only`).
- LiveKit 1.8.10 ordering checked in site-packages: the reply's speech handle marks generation done before the tool task is
  awaited (agent_activity.py 4023 vs 4061), so a say() from inside a tool plays while the tool runs and the tool reply follows.
- Every filler logs a `filler_start` turn event (tool, text). g1_report shows "filler before tool calls: n turns · filler audible ·
  first reply audible" (agent clock) under "first audio heard", which includes the filler; ledger metrics eos_to_filler_audible,
  eos_to_reply_audible, filler_turns. The report also labels the TTS first-byte line with the vendor from call_end.tts_vendor.
- Live test pending the first GPU-free slot (not during T's eval batch): `CALLS=3 TAG=filler scripts/measure_block.sh`, compare
  eos→first audio heard on tool-call turns with the 04:32 block (g1-sim-r3-voice-206506525c).

### Nov 1 04:32-04:38: candidate r3-voice-206506525c, clean numbers (GPU free, Deepgram TTS, 3 calls + 1 interruption)
- eos→first audio heard p50 1466 / p95 2592 ms (base 3060 / 12006; targets 1000 / 1600); turn-taking overhead p50 494 ms;
  LLM TTFT spoken turns p50 164 ms (base 722), tool-call turns p50 748 ms (base 1013); trie "reusing 1707 tokens, prefilling 29
  new" on turn 2+; Deepgram TTS first byte 241 ms (Cartesia was 188); transport 256 ms. Cost $0.043/call, $0.065/successful task.
- 3/3 lookups, 2/3 bookings, 0 structural leaks, 0 fallbacks. Two candidate failure modes seen (ledger run g1-sim-r3-voice-206506525c):
  (a) when the caller gives day, time of day and branch in one utterance ("Tuesday morning at the Market Street branch"), the
  model sometimes asks "Which day would you like..." again and loops (call 2 and the interruption call; call 1 asked for the time
  and booked); (b) call 2 ended with "I booked your mortgage consultation..." with no book_appointment call: an unsupported
  success claim, the contract's prohibited case. g1_report now counts these (`unsupported_claim_calls`); E's runner scores it in text.
- The interruption test cannot fire on the candidate: its turn-4 reply is too short to cut into (the base read slots for 28 s).
- Hedge timeout lowered to PROXY_FIRST_TOKEN_TIMEOUT_MS=2000 (tool-call p95 1068 ms).
- Side servers (T's eval batches) must start with VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=wait, as restart.sh does: without it
  vllm-mlx 0.5.0's simple engine answers 503 to a second concurrent request (T's 04:22 batch died of it; rerun 04:34).

### Nov 1 04:22-04:40: candidate deployed, trie verified, Cartesia out of credits
- r3-voice-206506525c (run 1 on v2) deployed by scripts/deploy_block.sh; probe PASS; previous = Qwen3-4B-8bit (`deploy_model.sh --rollback`).
- Prefix trie verified in production after the key fix: 2-turn probe at :8000 logged "Prefix trie cache HIT: reusing 1464 tokens,
  prefilling 45 new" (turn 1, shared prefix from the prewarm) and "reusing 1505 tokens, prefilling 30 new" (turn 2). Turn 1
  (10 tokens) 242 ms end to end vs 560-1070 ms before; turn 2 (tool call, 1535-token prompt) 573 ms vs ~1000 ms.
- All 4 simulated calls in the block failed with "no audio frames were pushed" from Cartesia: `curl tts/bytes` returned
  402 quota_exceeded ("requires approximately 19 credits but you have 2 remaining"). Fix: `AGENT_TTS=deepgram` in .env
  (Deepgram Aura-2 on the STT key, services/agent/tts.py); agent restarted alone; probe PASS. The user decides whether to
  top up Cartesia for the demo voice; the cost ledger prices TTS characters by the vendor recorded in call_end.tts_vendor.
- Clean candidate latency numbers wait for the next GPU-free window (T's eval batch holds the lock to ~07:00); one functional
  call ran during the batch (ledger run g1-sim-r3-voice-206506525c-contended, latency not comparable).

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

- `scripts/deploy_block.sh $R3_ARTIFACTS/models/r3-voice-<hash>`: the ≤15-min hand-back block under lock S: deploy, probe, 2-turn trie probe (expects a "Prefix trie cache HIT" log line), 3 simulated calls + 1 interruption call, g1_report + cost into the ledger, release.
- `scripts/routing_scheduler.sh` (MODEL_DIR=...): labels unlabeled tickets whenever the GPU lock is free. Not running during the demo: the lock is free while live calls are served, and a second 4B model competes with them. Run the batch job on demand in a free window instead.

### Hosted fallback is a hedge (Nov 1, 02:20)
A local turn with no first token after `PROXY_FIRST_TOKEN_TIMEOUT_MS` (3000) no longer gets cancelled: the proxy starts the
hosted request beside it and serves whichever streams a token first (`fallback` event with `hedged: true`, then
`fallback_cancelled` on the loser). Reason: the engine finishes a dropped request anyway, so the old cancel-and-reissue
cost 5-8 s per turn when the hosted side then failed (credits exhausted, call r3-g1-0145-015841-1 turn 3). With the
breaker open (hosted 4xx) the proxy simply waits for local. `g1_report` counts a fallback only when the hosted stream was
served (`llm_first_token.source = hosted`).
