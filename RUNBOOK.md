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

### Nov 1 08:02-08:30: deploy block on run 2, a loop-guard bug, and the claim guard widened
The deploy block on r3-voice-730f8ba68e (08:02-08:10, ~/r3-artifacts/runs/deploy-r3-voice-730f8ba68e-1101-0802) served the final
model under its own name, confirmed the trie hit (1 new token prefilled on turn 2) and completed 3 of 3 ordinary bookings, but
both scenario checks FAILED and the loop guard fired 5 times in 3 ordinary calls and 3 times in each scenario call. Root cause,
agent side, not the model: LiveKit's preemptive generation runs the LLM node on the interim transcript and again on the final
one; the loop guard counted the discarded first generation as the previous reply, so every regenerated turn looked like an exact
repeat ("Sorry, I may have missed that" in place of the model's first question); with the split phone number that reached 3
repeats and a false loop-guard handoff at the first caller turn of the handoff call. The 07:11 block's false re-asks were the same
bug, not only the split number. Fix 695d64f, then 08:30 after the verification call showed the second variant (a reply cancelled
because the caller talked on, "Five five five zero one four" / "two two one nine", never spoken yet counted as the previous line):
before each generation the guard reads the last assistant message in the chat context, which is what the caller actually heard;
if it is not what the guard yielded last time, or was interrupted, the previous generation never reached the caller, and the new
line is compared against what was heard with the repeat count reset. A spoken repeat still escalates: rephrase, rephrase, handoff
(the rephrase entering the chat context counts as heard). The two 08:02 scenario failures
count as the guard bug in every table (the timeout call never reached check_availability; the handoff call never reached lookup).

The interruption call of the same block had one model-side unsupported success claim: "I booked your mortgage consultation" after
the caller answered the read-back with "No, that's everything, goodbye", with no book_appointment call. The reviewer approved
widening the claim guard (08:15) to completed-action claims: a sentence claiming a finished write (booked, moved/rescheduled,
cancelled, ticket opened) with no successful result of the matching tool in the call state is held back whole and replaced by a
question ("I haven't booked anything yet. Would you like me to go ahead and book it?"); read-backs, questions and future tense pass
untouched; text is held only to the sentence boundary (Deepgram's TTS buffers to the sentence anyway). Same `claim_guard` event
and column, with `kind` (transfer | booked | moved | rescheduled | cancelled | ticket). The transcript then carries the question,
so the claims counter does not count the swallowed claim; the claim_guard column does. Both scenario checks were re-run on the
fixed build before the rehearsal (results below).

### Nov 1 08:23-08:45: verification calls, a scoring race, two scenario changes
Verification on the fixed build (~/r3-artifacts/runs/verify-r3-voice-730f8ba68e-1101-0823): the widened claim guard fired live once,
correctly (the model said "I've booked your mortgage consultation" after the caller declined; held, replaced by the question; no
claim spoken; the sentence hold cost no latency: first audible TTS p50 1124 ms vs 1318 in the deploy block). Loop guard 0 in both
scenario calls; the ordinary call exposed the second variant of the regeneration bug (fixed 3a3b5c7, above).

Scoring race: the agent writes `sessions.state` once, at call end, about 3 s after the caller leaves; `scripts/demo_check.py` ran
right after the simulated caller returned and read the opening checkpoint ("0 bookings" for a booked call, "tickets []" for an
opened ticket). It now waits for `ended_at` (DEMO_CHECK_WAIT_S, default 45 s) before scoring. The deploy block's scenario verdicts
were affected the same way; its calls were anyway broken by the loop-guard bug. Re-scored on the final state, the verification
calls fail for one reason each: timeout "fault not injected" (the model never called check_availability), handoff "no handoff
requested" (ticket opened, no transfer).

Scenario changes (8ce45bf, demo/scenarios.md): the timeout fault moved from check_availability to book_appointment, because run 2
books directly once the caller has named a time (0 check_availability calls in 7 live bookings), so the old fault was never
consumed; and 5b `handoff_person` (Dana asks for a person, then insists; expected request_handoff with reason caller_requested,
nothing written) was added as the reviewer's alternative handoff flow, since the direct fraud report (5) opened the ticket but did
not transfer in its one live call on run 2. The rehearsal runs six scenarios; the demo uses whichever handoff flow passes.

### Nov 1 08:40-09:25: rehearsal on the final model, and the executor retry
Rehearsal `rehearsal-r3-voice-730f8ba68e-1101-0840` (six scenarios x 5, Deepgram caller voice, recordings under its run dir). Early
rows: ordinary 3/5 (the model re-asked the day the caller had given in 2 calls; the loop guard caught the resulting repeat and the
claim guard held a false "I booked" at the scripted "no, that's everything" in both; nothing unsupported was spoken), correction 5/5,
timeout 0/5: after the injected book_appointment timeout the model asks "Would you like me to try again?" and then never calls
book_appointment again when the caller says yes (repeats the apology, or drifts); two calls ended in a transfer line without the
tool, which the transfer guard made real. The final model does not retry a failed write on request.

Response, same category and conditions as the guards (identical for every model, its own count, never shown as the model's
recovery): `services/tools/executor.py` retries a tool once when the sandbox answers timeout or unavailable (state.INFRA_ERRORS),
same arguments and idempotency key; the model sees only the final result; events: tool_end attempt 1 (ok false), tool_retry,
tool_start/tool_end attempt 2; `AGENT_TOOL_RETRY=0` turns it off; `scripts/demo_check.py` prints `retry=` per call and writes
`rehearsal.<scenario>.retried`. The agent was restarted with it only after the rehearsal's lock was released; the timeout scenario
was then re-run five times and both rows are reported. The handoff reason after an infrastructure failure is now tool_failure
(`state_view` carries tool_failures and business_errors; the timeout had been counted as an infrastructure failure, so the earlier
business_errors rule never fired).

### Nov 1 07:58: gate-fail fallback ready
If run 2's fused 8-bit artifact fails the precision gate, `scripts/serve_unfused.sh $R3_ARTIFACTS/models/r3-voice-730f8ba68e`
(under lock S) replaces the vllm-mlx server with mlx_lm.server 0.32.2 serving the unfused adapter on the trained base
(Qwen3-4B-8bit, the exact configuration the gate run used), labels proxy and agent `r3-voice-730f8ba68e-unfused`, and pins
the proxy's upstream name to `default_model` (mlx_lm.server answers only to that name). One request at a time, no prefix
cache: expect base-like TTFT (~0.7-1.0 s on tool-call turns). Rollback: `scripts/deploy_model.sh --rollback`.

### Nov 1 07:11-07:31: hand-back block, lock overlap, served-name mismatch, caller TTS fallback
What happened, in order. (1) T's hand-back restart (07:10) served run 1's weights under the name `r3-voice-base` and started
the proxy and agent with LLM_MODEL=r3-voice-base: T's launcher carried its worktree's stale .env in its environment, and
restart.sh lets environment values win over kar/.env (by design, for command-line overrides). My 07:11 block therefore
logged run 1 as r3-voice-base; relabeled afterwards with `scripts/relabel_model.py` (816 rows: sessions, turn_events,
ledger run ids g1-sim/g1-interrupt/cost-r3-voice-206506525c; dir renamed measure-r3-voice-206506525c-1101-0711-contended).
(2) Lock overlap 07:14-07:17: T's test suite ran gpu_lock.sh against the real lock file (its stripped environment lost
R3_ARTIFACTS, so the script fell back to $HOME/r3-artifacts), rewrote and released lock S, and later released T's own
lock; T's batch acquired at 07:14:27 while my block ran. Separately, my 05:14 `pkill` of a stuck block had left its
`gpu_lock.sh wait` child alive, and that orphan took lock S at 07:11:1x for nobody. gpu_lock.sh is now fail-closed
(unreadable lock = held), writes atomically (temp + rename), honours GPU_LOCK_FILE for tests, and `wait` gives up when its
parent is gone (tests/scripts/test_gpu_lock.py). Block numbers: calls 1-2 clean, call 3 and the interruption call
contended by T's side servers; indicative only (below). (3) 07:24-07:30 live turns broken: my clean proxy restart sent
the candidate's name to a server advertising r3-voice-base, and vllm-mlx answers 404 to any other name; restart.sh's
proxy check failed and exited before its agent step, so the agent kept the base label. Fix: the proxy sends local
requests under the single id from the server's /v1/models and keeps the caller's id on every label and event
(`/health` shows `upstream_model`; LOCAL_LLM_UPSTREAM_NAME overrides). Verified 07:31: proxy check ok, one-token turn
200, probe session labeled r3-voice-206506525c. (4) Both scenario calls (timeout, handoff) aborted before the first word:
the simulated caller synthesizes new lines through Cartesia, which is out of credits (402). `scripts/sim_caller.py` now
falls back to Deepgram Aura (SIM_TTS=auto: Cartesia first so cached lines keep their voice; `aura-2-orion-en` for new
lines; SIM_TTS=deepgram forces one voice, which the rehearsal uses); all five demo scripts are pre-synthesized
(`--prepare --lines-file`). (5) Loop guard escalation moved from the 2nd to the 3rd exact repeat
(AGENT_LOOP_GUARD_HANDOFF_AFTER=3): in call 1 the caller's phone number was split by turn detection into two utterances,
the model asked for the date of birth after each fragment, and the guard handed off a recoverable call (request_handoff
recorded, call continued, booking then claimed without the tool).

Indicative numbers, run 1 with filler + loop guard, 3 calls + interruption (calls 3-4 contended): eos→first audio heard
p50 1492 / p95 2528 ms (04:32 clean block 1466 / 2592); filler before tool calls 4 turns, filler audible p50 1555 ms,
first reply audible p50 2063 ms (the filler fills ~0.5 s of the tool-call wait); turn-taking 494; LLM TTFT spoken 139 /
tool-call 820; Deepgram TTS first byte 170; 3/3 lookups, 1/3 bookings; unsupported claims 2 calls (both "I booked..."
without book_appointment, the known run-1 failure mode); loop guard 5 interventions in 3 calls (threshold 2 at the time);
claim guard 0. The clean filler A/B and the claim guard's live call run in the deploy block of the final candidate.

### Nov 1 05:42: claim guard (reviewer item from the run-1 failure labels)
In 6 of 81 text episodes (E's failures.md: "transfer 6"; the reviewer's label pass counted 5) the candidate said "I've opened
your ticket and I'm transferring you" right after create_ticket and called request_handoff a turn late or never (the contract counts a transfer announced without the tool as an unsupported
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
exercises the wiring in text. Not a guard case: the 16 one-strike handoffs in run 1 (transfer after a single not_verified
instead of a retry, request_handoff correctly called) are a missing example type, covered by E's v5.1 misstate-retry scenarios.

### Nov 1 05:01: loop guard; demo rehearsal tooling
(07:33: escalation moved to the third exact repeat, AGENT_LOOP_GUARD_HANDOFF_AFTER=3, reviewer-approved; the motivating
call and the reasoning are in the 07:11-07:31 section. E recounts the post-hoc interventions at 3 so the write-up's guard
column is on one rule. Known turn-taking limitation, not tuned further: the 0.25-2.0 s endpointing window splits digit
strings read with pauses; the demo scripts now read the phone number in one breath.)
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
