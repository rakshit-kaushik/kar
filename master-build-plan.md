# R3 build plan — one person, MLX, cascaded voice

Oct 31, 2026 · @Ab Gup

## Scope and decisions locked

R3 is one person, one Mac and three or four coding-agent sessions building, in 24 hours, a voice agent for one field-service workflow and a ticket router, both as LoRA adapters on Qwen3-4B, trained and served with MLX. The demo compares base, adapted and hosted models on task completion, per-stage latency and cost per successful task. The budget that matters is not tokens but two things that don't parallelize: your judgment (the contract, the test scenarios, the data spot-check, the failure labels) and the Mac GPU (it trains or it serves, never both).

| Decision | Choice | Reason |
| --- | --- | --- |
| Base model | Qwen3-4B instruct, thinking disabled | the one 4B-class model with a native tool-call format worth training on |
| Training | mlx\_lm.lora on the 8-bit base; LoRA r32 / alpha 64 / dropout 0.05 on attention + MLP projections; seq 2,048 | train and serve on the same precision; no bitsandbytes on Metal; 2,048 halves run time and the workflow fits |
| Training budget | one 10-minute smoke run, one \~2-hour run, one optional \~1.5-hour rerun, one 20-minute routing run | that is all the GPU time a 24-hour sprint can give to training |
| Serving | vllm-mlx or oMLX, OpenAI-compatible, continuous batching + prefix cache; pick one by T+4 | stock mlx\_lm.server is one request at a time |
| Stand-in model | a hosted model behind the same OpenAI-compatible interface | every session keeps working while the GPU trains; also the fallback and the baseline |
| Voice pipeline | cascaded: Deepgram STT → model → Cartesia TTS, inside LiveKit Agents AgentSession | VAD, turn detection, interruption handling, TTS chunking and function calling come built in |
| Audio entry | browser only; no Twilio | one vendor fewer; phone adds nothing to the model work |
| Data | 600–1,000 simulator-generated conversations, scorer-filtered, 50 spot-checked by you; at most 3 training targets per conversation | the correction pass is the part that doesn't scale, so the filter does most of it |
| Ticket routing | short label generation (greedy, ≤ 8 tokens, validated) as a batch job with mlx\_lm directly; first thing cut if behind | prefill-only cost; isolated from live calls |
| Adapters | one per task, fused and quantized to 8-bit, each its own model instance | no multi-LoRA batching on MLX; keeps batch traffic away from voice latency |
| Hardware | one Mac, 64 GB unified memory or more | memory bandwidth and prefill compute are the limits, not capacity |
| Packaging | native processes (uv) for the model server and the agent; Docker only for Postgres | Metal is not exposed inside Docker on macOS |
| Dropped | MLflow, OpenTelemetry + Prometheus + Grafana, Redis worker, S3, launchd, Twilio | JSON logs, git tags and local disk cover a sprint |

Out of scope: phone access, speech-to-speech models, serving several adapters from one process, a second business workflow, a second analysis iteration unless the first run finishes by T+12.

## Target architecture

A call runs browser mic → LiveKit room → AgentSession (Silero VAD, LiveKit turn detector, Deepgram STT) → local model server over HTTP → tools in the Python API process → Cartesia TTS → browser playback. Everything the model sees and says is logged per turn with stage timestamps.

| Component | Choice | Role |
| --- | --- | --- |
| Model server | vllm-mlx or oMLX serving the fused 8-bit voice model | streaming chat completions with OpenAI-style tool\_calls; prefix cache for system prompt + tool schemas; batching across calls |
| Agent runtime | LiveKit Agents (Python), one AgentSession per call | audio session, VAD, turn boundaries, interruptions, TTS streaming, function-tool dispatch |
| Tools | Python functions with strict schemas, deterministic sandbox data | lookup\_account, check\_availability, book\_appointment, reschedule\_appointment, create\_ticket, request\_handoff; idempotency keys on anything that writes |
| Session state | in-memory per call, checkpointed to Postgres | verified facts (from tools) kept apart from claimed facts (from the caller); stage; pending and completed actions; handoff status |
| Eval runner + simulator | Python CLI | caller simulator + fresh sandbox per episode; scores any OpenAI-compatible endpoint with the same tools |
| Data pipeline | Python scripts, versioned JSONL + manifest | generation, correction pass, STT shaping, chat-template rendering, per-turn unrolling, splits |
| Training | mlx\_lm.lora → mlx\_lm.fuse → quantize | adapters per task; runs overnight, never while serving |
| Routing job | mlx\_lm batch generation in a worker | labels tickets from Postgres in batches on a schedule |
| Demo UI | Next.js | transcript, stage, verified vs claimed facts, tool calls, model selector, per-stage latency, outcome, cost |
| Store | Postgres (Docker) | customers, tickets, sessions, turn events, eval results, cost ledger |
| Observability | structured JSON logs + one notebook | per-call trace from speech end to playback; mx memory stats and macmon for the machine |

## Before T+0: setup

None of this needs judgment, so none of it is inside the 24 hours. Do it the evening before; the downloads are the long pole, so they start first. The clock starts when you open `CONTRACT.md` to write it.

- [ ] Accounts and keys in a `.env`: LiveKit Cloud project (URL, key, secret); Deepgram; Cartesia; the hosted model used for the simulator caller, generation, fallback and baseline. Check that hosted model's rate tier against 20 parallel generation episodes; raise it now if it is the default tier
- [ ] Machine: current macOS; Xcode command line tools; Homebrew; uv; an arm64 Python 3.12 (not Rosetta); Node 22; OrbStack or Docker Desktop for Postgres; git; Conductor installed and signed in through your Claude Code login
- [ ] Downloads, started first: `mlx-community/Qwen3-4B-8bit`; `mlx-community/Qwen3-0.6B-8bit` for the draft-model test; the LiveKit turn-detector and Silero VAD files via the agent's `download-files` command; LiveKit's example web client
- [ ] Shared artifacts directory outside every worktree, for example `~/r3-artifacts/{models,datasets,runs}`, named by one env var; worktrees hold code only
- [ ] One venv per worktree with uv and a shared uv cache; `mlx-lm`, `vllm-mlx`, `livekit-agents` with the deepgram, cartesia, silero, turn-detector and openai plugins, `psycopg`; versions pinned in a lock file the first time they resolve
- [ ] Three checks before bed: `mlx_lm.generate` on the 4B prints tokens per second; vllm-mlx answers a streamed chat completion with a tool definition; the LiveKit agent runs in console mode (local mic and speaker, no browser) against the local server. Whatever fails here is the first task of session S, not a surprise at T+2
- [ ] Repo scaffold: `CONTRACT.md`, `RUNBOOK.md`, `schemas/` (tools, events, state), `services/` (server proxy, agent, tools), `eval/` (simulator, scorers, runner), `data/`, `train/`, `ui/`, `db/`, `scripts/` (restart, GPU flip), `.env.example`; ports fixed in the contract (server 8000, proxy 8001, UI 3000, Postgres 5432)
- [ ] Conductor: one workspace per session from the same repo, a setup script that copies `.env` and runs `uv sync`, a run script per workspace; the opening prompt for each session drafted as a file in `prompts/` so T+1 is paste, not compose
- [ ] Postgres up with the migrations for sessions, turn events, eval results and the cost ledger; one row in the ledger with the machine's name, chip and memory

## The 24 hours: sessions in parallel, the GPU on a schedule

Four agent sessions run side by side from T+1, each in its own git worktree, all reading one contract you write in the first hour. They never wait on each other except at the four gates, and they never wait on the GPU: the LLM endpoint is a URL in config, and during a training window that URL points at the hosted model.

Critical path: contract (T+0–1) → simulator and scorers (T+1–5) → data (T+5–9) → training run 1 (T+10–12) → eval and deploy (T+12–14) → demo on the candidate (T+14 on). Serving, routing and the demo UI sit off that path and fill the gaps.

- Session S, serving: MLX server, LiveKit agent, tools wiring, adapter serve chain, load tests, deploy, fallback
- Session E, eval and data: simulator, scorers, test-set formatting, generation, filtering, STT shaping, splits, mask checks
- Session T, training: mlx\_lm.lora config, fuse and quantize scripts, run launcher, eval-on-checkpoint, routing adapter
- Session D, demo: Next.js UI over Postgres, cost ledger, scenario scripts, replay
- You: the contract, the test scenarios, the data spot-check, listening to candidate calls, failure labeling, rehearsal; and a review pass over every session roughly every 45 minutes

Two sessions instead of four is a legitimate cut: S absorbs D, T absorbs E. The critical path doesn't move, since E feeds T in sequence anyway, and your review load halves. What you give up is the UI being built in parallel (LiveKit's playground covers the demo) and some overlap between generation and training scripts (generation runs in the background, so the merged session writes the scripts while it runs). Either way, every session hands background work to subagents: generation batches, load tests, benchmark sweeps.

Gates, where every session syncs on main:

1. G1, T+4: a live browser call completes a tool lookup on the untuned base model, nothing structural is spoken, per-stage latency is in Postgres.
2. G2, T+8: a 50-step smoke adapter goes train → fuse → quantize → serve → LiveKit call → scored by the runner. Run 1 may not start before this.
3. G3, T+13: the first candidate is deployed under a versioned id, within one point of the unfused adapter on the 8-bit base on the test set, holding the latency targets at 4 simulated calls.
4. G4, T+20.5: final model chosen and frozen; after this only rehearsal, recordings and the write-up.

&#91;embedded content: 24-hour GPU schedule · 4 training windows, 4 gates\]

Your own fixed-time work is small in hours and sits exactly where the sessions need it; everything else is review passes. The GPU gives training about 4 hours of the 24, and the agent sessions never notice, since they talk to a URL.

## T+0 to T+1, you alone: the contract

Exit criterion: one `CONTRACT.md` in the repo that every session reads before writing a line, plus the schemas as files. No agent starts before this exists; an hour here saves the merge hell at T+8. Workflow: service appointment booking for a field-service business. Intents: book, reschedule, cancel, status, pricing question, speak to a person. Exceptions: unknown account, no availability, ambiguous address, duplicate booking, tool timeout. Prohibited: confirming anything a tool has not returned, prices outside the price list, bookings outside service hours.

- [ ] Workflow spec, one page: callers, intents, what the agent may act on, what counts as done, exceptions, prohibited actions, handoff rules, five example dialogues (three good, two bad) with expected tool calls and final state
- [ ] Conversation policy, ten rules: one question per turn; confirm address and time before any write; one or two sentences per reply; numbers, times and addresses written as they should be spoken; what to say on a correction, a tool failure, a handoff
- [ ] Tool schemas as JSON files: lookup\_account, check\_availability, book\_appointment, reschedule\_appointment, create\_ticket, request\_handoff; roughly 60 tokens each; named error codes (not\_found, conflict, timeout, unavailable); idempotency key on every write
- [ ] Session-state schema: verified facts (tool-sourced) apart from claimed facts (caller-sourced); stage; pending action; completed actions; handoff status
- [ ] Event schema for the turn log: speech end, STT final, model request, first token, tool start and end, TTS first audio, playback end, interruption, with call id and model id on every row
- [ ] Interfaces: the model is always an OpenAI-compatible URL from config; the simulator, the agent and the eval runner all consume the same tool files; Postgres tables for sessions, turn events, eval results, cost ledger
- [ ] Repo layout with one worktree per session and the decisions table from this doc copied in; the banned shortcuts written down: no vLLM, no PEFT, no Docker for model processes, no packing, no thinking mode
- [ ] Routing taxonomy, if routing survives: at most 12 categories, a definition and five examples each, and which confusions are costly

The contract is the only thing the generator, the scorers, the agent and the demo scripts all read; an ambiguity here shows up four times later, in four worktrees.

## T+1 to T+4, session S: walking skeleton

Exit criterion (G1, T+4): ten live browser calls complete an account lookup on the untuned base model, nothing structural is spoken, per-stage p50/p95 are in Postgres. This window answers the questions the rest of the day depends on: real TTFT at your prompt size, how slow prefill is on this Mac, and which server to keep.

Order matters: the server and the LiveKit agent are the two integration risks, so they go first and in parallel (session S can itself be two sessions for this window); the tools and the logging are routine and come after.

- [ ] Install mlx-lm; pull Qwen3-4B 8-bit; record single-stream decode tok/s and prefill tok/s at 1K, 2K and 4K prompt tokens into the ledger
- [ ] vllm-mlx with continuous batching and the paged cache; stream a chat completion with tools; confirm tool calls arrive as OpenAI tool\_calls deltas, not raw `<tool_call>` text; try oMLX if it fails. If neither parses Qwen3 tool calls while streaming, a thin proxy that does (one hour) becomes a permanent part of the stack
- [ ] Thinking off end to end (chat-template flag or extra body); strip any `<think>` block in the proxy as a guard
- [ ] LiveKit Agents AgentSession: Deepgram STT, Cartesia TTS, Silero VAD, LiveKit turn detector, LLM plugin pointed at the local URL; LiveKit's example web client for the browser
- [ ] Tools from the contract's JSON files, deterministic sandbox data, wired as function tools; lookup\_account first, the rest by T+4
- [ ] Postgres (Docker): sessions, turn events per the contract's event schema, eval results, cost ledger
- [ ] Ten calls on the base model; per-stage p50/p95 and the prompt token count into the ledger as the baseline row

What G1 decides: the server, the system prompt size, and whether the live demo path can run on the Mac or needs the hosted model with the adapter shown in replay and text eval.

## T+1 to T+5, session E: simulator, scorers, baselines

Exit criterion (T+5): the runner scores any OpenAI-compatible URL on the test set with the contract's tools, and a baseline table exists for base-in-text and hosted-in-text with a ranked failure list. The simulator built here is the data generator from T+5 on, so it is built once, as the evaluator. You write the test scenarios yourself at T+5 to T+6 (27 of them, three per family) while the session runs baselines on a first draft set.

- [ ] Caller simulator: a scenario spec (persona, goal, known facts, planned corrections and detours, noise level) drives a hosted model playing the caller; the agent under test runs against a fresh sandbox per episode; episodes end on completion, handoff or a turn cap
- [ ] Deterministic scorers: final sandbox state matches the expected state; tool name and normalized arguments per step; no success claim in assistant text before the matching tool result; turn count within budget; handoff when required and never when not; corrected details still correct at the final action
- [ ] Scenario families: ordinary, missing information, correction, ambiguous address, tool timeout, duplicate request, multi-intent detour, needs a person, noisy transcript. Your 27 hand-written scenarios are the test set and are never used for generation
- [ ] Baselines by T+5: Qwen3-4B base in text and one hosted model in text, identical prompts and tools; through audio on 10 live calls once G1 is done. Per-family scores and the top failure modes ranked by frequency times cost go in the ledger
- [ ] Human rubric, three lines: usefulness of the question asked, spoken brevity, naturalness; you rate 20 episodes of the candidate at T+12

If routing survives, the same session adds a 100-ticket hand-checked test set and a zero-shot baseline with the taxonomy in the prompt, at T+6, in under an hour.

## T+5 to T+9, session E: data

Exit criterion (T+9): 600–1,000 scorer-passing conversations with provenance, STT-shaped caller turns, frozen splits, a manifest, and loss masks verified on 20 decoded examples; your spot-check of 50 episodes done at T+9 to T+10 before run 1 starts. Generation is API-bound, not GPU-bound, so it runs while session S has the Mac.

- [ ] Generation: scenario templates × personas × detours through the simulator with the sandbox tools in the loop, so every tool result in the data is real; a strong hosted model plays the agent with the conversation policy in its prompt; 20 episodes in parallel; generation scenarios never overlap the 27 test scenarios
- [ ] Filter instead of correct: any episode that fails a deterministic scorer is dropped; the mix is rebalanced after filtering, about 40% ordinary and the rest spread over the failure families, weighted by the T+5 failure list. Expect to generate 1.5× what you keep
- [ ] STT shaping from the G1 transcripts: a rule set for what Deepgram emits (casing, punctuation, number and address formatting, filler words), plus homophone swaps and dropped words at modest rates, applied to simulated caller turns; 50 turns through a real TTS → Deepgram round trip to check the rules against reality
- [ ] Interruptions: a slice where an assistant turn is cut mid-sentence and the context carries the truncated text; the target treats a half-spoken confirmation as unconfirmed
- [ ] Format: Qwen3 chat template with tool definitions; at most three training targets per conversation, chosen to cover a tool call, a clarification and the closing turn, each as prefix → that assistant turn so prompt masking puts loss on exactly that turn, tool-call tokens included, tool results never; decode 20 masks by hand
- [ ] Splits by scenario family and seed persona; the 27 hand-written scenarios are the test set, never generated or augmented; manifest with generator version, policy version, STT rule version, scorer verdicts per episode
- [ ] Your spot-check, T+9 to T+10: 50 episodes read end to end; if more than 5 are wrong in a way the scorers missed, add the missing scorer and refilter before run 1, not after

Size check: 800 conversations × 3 targets × about 1,200 tokens is roughly 3M tokens per epoch. That is the number that sets run 1's length, and the reason targets per conversation are capped.

## T+1 to T+20, session T: training

Exit criterion: a candidate chosen by the eval runner on the test set, with a reproducible config, logs and a fused 8-bit artifact under a versioned id. The session writes scripts from T+1 and owns the GPU only in the four windows on the schedule: smoke at T+7, run 1 at T+10, routing at T+13, run 2 at T+18 if the failure labels justify it. Every run is sized from the smoke run's measured tokens per second so it ends inside its window.

- [ ] T+1 to T+6: mlx\_lm.lora config, data loader that reads the manifest, mlx\_lm.fuse and quantize scripts, a run launcher that writes config, git hash, data manifest hash and metrics to the ledger, and an eval-on-checkpoint hook that starts the server and calls the runner
- [ ] T+7, the smoke adapter (G2 at T+8): 50 steps on 100 examples → fuse → quantize → serve → one LiveKit call → scored by the runner. Record tokens per second; that number sizes run 1
- [ ] Run 1 config: Qwen3-4B 8-bit base loaded quantized; LoRA r32, alpha 64, dropout 0.05 on q, k, v, o, gate, up, down; seq 2,048; micro-batch 1 (2 if the smoke run shows the memory); gradient accumulation 16; lr 1e-4, cosine, 3% warmup; gradient checkpointing on; clipping 1.0; no packing; one epoch, eval at the end and at the midpoint checkpoint
- [ ] No sweep. If run 2 happens, it changes one thing: the data (targeted additions from the failure labels), not the hyperparameters, unless run 1's loss curve is flat or diverging, in which case lr moves one step and nothing else
- [ ] Selection by the runner on the test set: task completion, tool-argument accuracy, zero unsupported success claims; never by loss
- [ ] Precision gate at T+12: the fused and requantized artifact within one point of the unfused adapter on the 8-bit base (the configuration that was trained) on task completion; if it fails, serve the unfused adapter on the 8-bit base and accept the slower tokens
- [ ] T+13, routing adapter: same recipe at seq 1,024 on about 1,000 labeled tickets, one epoch, 20 minutes; skip without discussion if the clock is behind

While a run is going, the server is down and every other session's config points at the hosted URL; the launcher flips it back and restarts the server when the run ends, so nobody waits on a human for that.

## T+4 to T+18, session S: serving and deployment

Exit criterion (G3 at T+13, then T+18): the candidate is served under a versioned id, holds the latency targets at 4 simulated calls, falls back to the hosted model on timeout at a turn boundary, and the routing batch job runs on a schedule that never overlaps a live call. The session owns the GPU in the benchmark, load-test, eval and deploy windows and works on scripts against the hosted URL during training windows.

- [ ] T+4 to T+7, benchmarks: prefill and decode tok/s at 1K/2K/4K; prefix cache hit confirmed for system prompt + tool schemas (prefill tokens on turn 2 should be only the new transcript plus tool result); per-call KV persistence across turns checked the same way; KV cache quantization left off unless memory shows up as a limit
- [ ] T+7 to T+8 (G2): the smoke adapter's serve chain: fuse → quantize → load → versioned id → health check → prewarm request; one LiveKit call on it
- [ ] T+8 to T+10, load tests: a simulated-call generator with turn lengths and pauses taken from the G1 traces; 1/2/4/8 concurrent calls; TTFT, inter-token latency and first-audio p50/p95 into the ledger; the concurrency where p95 TTFT still meets budget becomes the admission limit
- [ ] Speculative decoding with a Qwen3-0.6B draft: one test at 1 and 4 calls during the load window; keep only if TTFT or inter-token latency improves at the demo's concurrency; otherwise off and forgotten
- [ ] T+13.5 to T+16, deploy: the chosen candidate under `r3-voice-<hash>`; precision gate result recorded; admission control (reject or route to hosted above the limit); fallback to the hosted URL on a first-token timeout or server error, at a turn boundary, with state preserved and the event logged; rollback is the previous model directory
- [ ] T+14 to T+16, routing batch job: a worker that pulls unlabeled tickets from Postgres, runs mlx\_lm batch generation on the routing model, validates labels, writes them back; scheduled only in windows the schedule marks as free
- [ ] T+16 to T+18: a restart script that brings up Postgres, server, agent and UI in order and verifies each; a frozen `.env`; the model directory and ledger copied to a second disk

The hosted fallback is also the demo's insurance: if the Mac path misses the latency budget on the day, the live call runs hosted and the adapter is shown through replay and the text eval, with the numbers side by side.

## T+12 to T+20, you and session E: the one analysis loop

There is room for one loop, not three, so it is decided by a rule rather than by taste: run 2 happens only if, by T+17, the failure labels show at least one bucket that data can fix and that the test set punishes. Otherwise run 1 is the final model and the time goes to rehearsal.

- [ ] T+12 to T+13, you: listen to 10 live calls on the candidate and rate 20 simulated episodes on the three-line rubric; note anything the scorers miss
- [ ] T+13 to T+16, session E: run the full test set and the dev set on the candidate; dump every failing episode with the scorer that failed it and the turn where it went wrong
- [ ] T+16 to T+18, you: label each failure as one of bad data, missing example type, context truncation, model capability, application logic. Thirty minutes per bucket is the cap; the count per bucket is the output
- [ ] Fix in the cheapest place first: application logic and tool schemas (minutes, no retrain), then 150–300 targeted conversations for the top data bucket (session E, one hour of generation), then run 2 on the merged data with the same config
- [ ] T+19.5 to T+20.5: run 2 through the runner and the precision gate; the better of run 1 and run 2 on the test set is the final model (G4); the other stays on disk as the rollback

Capability failures (the 4B simply can't do it) are not fixed in this sprint; they go in the write-up as the argument for the next base model, with the hosted model's score on the same scenarios beside them.

## T+6 to T+24, session D: demo, economics, rehearsal

Exit criterion: five scripted scenarios run five times each on the frozen build with a recording of each, a UI that shows the model's decisions as they happen, and a cost per successful task computed from measured usage for all three model paths.

- [ ] T+6 to T+12, the UI over Postgres: live transcript; workflow stage; verified facts beside claimed facts; tool calls with arguments and results; model selector (base, adapter, hosted) that changes the agent's LLM URL; per-stage latency bars for the current turn; outcome. LiveKit's playground is the fallback if this slips
- [ ] T+12 to T+16, cost ledger: per call, Deepgram minutes, Cartesia characters, LiveKit minutes, LLM tokens; the local path costed as the Mac's hourly cost divided by measured calls per hour at the admission limit; the hosted path at list price; measured and estimated marked apart; cost per successfully completed task as the headline
- [ ] T+16 to T+20, replay: a recorded transcript replayed through any model with the same tools, for the side-by-side on stage
- [ ] T+20.5 to T+22, scenario scripts: ordinary booking; a correction mid-flow; a tool timeout with recovery; an interruption; a handoff. Each written as what the caller says, what the agent should do, what the final state must be
- [ ] T+22 to T+24, you: each scenario five times on the frozen build; the backup recording of each; the write-up with the baseline table, the candidate table, the latency table and the cost table

Everything on stage is labeled live, recorded or scripted; a replay shown as a live call is the one thing this demo must never do.

## Ticket routing track

Routing is the controlled comparison the original plan wanted (same base, same recipe, a task with deterministic scoring), and it costs about three hours of agent time and 20 minutes of GPU. It is also the first thing cut, since the demo stands without it.

- [ ] T+1, you: the taxonomy in the contract, at most 12 categories, a definition and five examples each, and the costly confusions named
- [ ] T+6, session E: about 1,000 synthetic tickets generated from the definitions with variation in length, tone and noise, labeled at generation time; 100 hand-checked by you form the test set; a zero-shot baseline with the taxonomy in the prompt, for the base and the hosted model
- [ ] T+13, session T: the routing adapter, seq 1,024, one epoch, 20 minutes; fused and quantized like the voice model
- [ ] T+14, session S: the batch job; output is the label string, greedy, 8 tokens at most, validated against the taxonomy; anything invalid becomes `needs_review`
- [ ] Scoring: per-category precision, recall and F1; the confusion matrix weighted by the costly pairs; coverage, meaning the share of tickets that got a valid label

Per-ticket cost is one prefill of about 500 tokens, so the economics row for routing is close to free on the Mac and is worth showing beside the voice numbers.

## Measurements and targets

Targets are proposals until G1; the ones marked "after baseline" are set from the T+5 baseline table and written into the ledger then. Every number comes from the turn-events table or the eval runner, never from a vendor dashboard.

| Measure | Measured where | Proposed target |
| --- | --- | --- |
| End of speech to first audio | turn events | p50 ≤ 1.0 s, p95 ≤ 1.6 s |
| STT final after end of speech | turn events | ≤ 0.5 s (vendor; endpointing setting) |
| Model time to first token | turn events | p50 ≤ 0.35 s at 4 concurrent calls |
| Prefill tokens per turn | server log | ≤ 150 on turns 2 and later |
| Inter-token latency | server log | ≤ 50 ms at 4 calls; speech needs about 5 tokens/s |
| TTS first audio | turn events | ≤ 0.2 s |
| Concurrent calls within targets | load test | 4 |
| Task completion, test set | eval runner | adapter above base; within a margin of hosted set after baseline |
| Tool-argument accuracy, ordinary family | eval runner | ≥ 95% |
| Unsupported success claims | eval runner | 0 |
| Stale-detail errors after a correction | eval runner | 0 on the correction family |
| Handoff precision and recall | eval runner | after baseline |
| Routing macro F1 | eval runner | after baseline |
| Cost per successfully completed task | cost ledger | computed for all three paths; no target |
| Fallback rate | turn events | reported, no target |

## Risks, and the cut order

The sprint fails in one of three ways: the skeleton isn't up by T+4, run 1 doesn't fit its window, or the candidate isn't better than base on the test set. Each has a pre-decided response so the decision takes a minute, not an hour.

| Risk | Signal | Response |
| --- | --- | --- |
| Tool-call streaming not parsed by the server | raw `<tool_call>` text reaches TTS at T+2 | the proxy, one hour, becomes permanent; don't try a third server |
| Prefill too slow for the voice budget | G1 TTFT p50 above 0.6 s at 1 call | 2K context cap, tool schemas trimmed, prefix cache confirmed; if still over at T+10, the live demo path is hosted and the adapter is shown in replay and text |
| Skeleton not up by T+4 | G1 missed | sessions E, T and D keep going against the hosted URL; S gets a second session; G1 slips to T+6 and the load-test window shrinks |
| Run 1 too slow for its window | smoke run tok/s below plan | cut targets per conversation from 3 to 2, then conversations to 600; never raise seq or epochs |
| Candidate not better than base | T+12 test-set table | check masks and chat template first (the usual cause), then the data mix; run 2 is the one retry; if still flat, the demo shows the eval and the latency work and says so |
| Synthetic data teaches the generator's voice | spot-check finds stiff or verbose turns | tighten the policy prompt, regenerate the ordinary slice only, refilter |
| Thinking tokens leak | `<think>` in any transcript | template flag plus the proxy guard; both, not one |
| One Mac, no second machine | any GPU contention | the schedule is law; a run that overlaps a load test invalidates both |
| You run out | T+14 onward | the T+14 to T+16 window has nothing that needs you; put a break there, and let the sessions run |

Cut order when behind, top first: routing track; speculative decoding test; run 2; the Next.js UI (LiveKit's playground instead); the cost ledger automation (a spreadsheet by hand); the interruption slice of the data; the audio baseline (text baselines stay).

## Working with the agent sessions

Four sessions in parallel is the right number: more than that and your review pass becomes the bottleneck, fewer and the GPU schedule has idle windows. The agent is interchangeable; the setup below is what decides whether the day works.

- Contract first, then sessions. Each session's opening prompt is the contract plus one sentence naming its worktree, its exit criteria and the files it may touch. A session that needs to change a shared schema stops and asks; it does not edit the contract
- Conductor is the right tool for this layout: one workspace per session, each a git worktree with its own branch, terminal, diff and merge flow, on Claude Code or Codex through the login you already have. What it does not do is coordinate: merges are yours, the GPU schedule is enforced by the launcher script, and anything shared (Postgres, the model server, the artifacts directory, ports) lives outside the worktrees and is named in the contract
- Pin and point. mlx-lm, vllm-mlx, LiveKit Agents and the vendor SDKs all change monthly, and every agent will invent flags for them. Pin versions in the first hour and tell each session to read the installed source (site-packages) for the API it is about to call before writing against it, and to say when the installed version differs from what it expected
- Eval runner as the target. From T+5 every task for sessions S, T and E is phrased as a scenario to make pass or a number to move; the runner is how the agent checks its own work without you
- Half-hour tasks with a written exit. Anything bigger gets split; anything without an exit criterion is not started
- Your review pass is a diff read plus the exit check, not a conversation. Roughly every 45 minutes per session; a session that is stuck twice on the same thing gets a smaller task, not more context
- The judgment stays with you: the contract, the test scenarios, the data spot-check, the failure labels and the model choice. Agents generate, filter, wire and measure; they do not decide what the model should learn
- A `CONTRACT.md` and the decisions table in the repo root, plus a one-page `RUNBOOK.md` with the restart script, the GPU schedule and the cut order, so a session started at T+15 has the same context as one started at T+1
