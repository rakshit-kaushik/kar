# R3 — 24-hour build plan (condensed)

> **Note (Nov 1, 2026):** this plan predates the contract. The workflow became branch appointment booking for a fictional retail bank (Marlow Mutual), and the machine has 36 GB, not 64. `CONTRACT.md` is authoritative; results are in `results/`.

Living version with checklists, the GPU schedule drawing, targets and risks: https://claude.ai/artifact/X8x6CZrHjiQ9DZvUD9xSKz

## What R3 is
One person, one Mac, 2–4 coding-agent sessions, 24 hours. A voice agent for one field-service workflow (service appointment booking) and a ticket router, both as LoRA adapters on Qwen3-4B, trained and served with MLX. Demo compares base, adapted and hosted models on task completion, per-stage latency and cost per successful task.

The two things that don't parallelize: your judgment (contract, test scenarios, data spot-check, failure labels) and the Mac GPU (trains or serves, never both).

## Decisions locked
- Base: Qwen3-4B instruct, thinking disabled (only 4B-class model with a native tool-call format).
- Training: mlx_lm.lora on the 8-bit base; LoRA r32 / alpha 64 / dropout 0.05 on attn + MLP projections; seq 2,048; lr 1e-4; one epoch; no packing; no sweep.
- Training budget: 10-min smoke (T+7), ~2 h run 1 (T+10), 20-min routing (T+13), optional ~1.5 h run 2 (T+18).
- Serving: vllm-mlx or oMLX (OpenAI-compatible, continuous batching, prefix cache); pick by T+4. Stock mlx_lm.server is serial.
- Stand-in: a hosted model behind the same OpenAI-compatible URL; used while the GPU trains, as the fallback, and as the baseline.
- Voice: cascaded Deepgram STT → model → Cartesia TTS inside LiveKit Agents AgentSession (VAD, turn detection, interruptions, TTS chunking, function calling built in). Browser only; no Twilio.
- Data: 600–1,000 simulator-generated conversations, scorer-filtered, 50 spot-checked by hand; ≤3 training targets per conversation (prefix → assistant turn; loss on tool-call tokens, never tool results); caller turns STT-shaped from the G1 transcripts.
- Routing: short label generation (greedy, ≤8 tokens, validated) as a batch job via mlx_lm directly; first thing cut.
- Adapters: one per task, fused + quantized to 8-bit, each its own model instance (no multi-LoRA batching on MLX). Precision gate compares the fused/requantized artifact against the unfused adapter on the 8-bit base (the trained configuration).
- Hardware: one Mac, ≥64 GB unified memory. Bandwidth and prefill compute are the limits, not capacity.
- Packaging: native processes (uv); Docker only for Postgres (Metal not exposed in Docker).
- Dropped: MLflow, OTel/Prometheus/Grafana, Redis worker, S3, launchd, Twilio.

## Before T+0: setup (outside the 24 h; no judgment needed)
- Keys in `.env`: LiveKit Cloud (URL/key/secret), Deepgram, Cartesia, hosted model (check its rate tier against 20 parallel generation episodes).
- Machine: current macOS, Xcode CLT, Homebrew, uv, arm64 Python 3.12, Node 22, OrbStack/Docker for Postgres, git, Conductor signed in via Claude Code login.
- Downloads first: mlx-community/Qwen3-4B-8bit, Qwen3-0.6B-8bit (draft test), LiveKit turn-detector + Silero files, LiveKit example web client.
- Shared artifacts dir outside worktrees (`~/r3-artifacts/{models,datasets,runs}`), one env var. Worktrees hold code only.
- One venv per worktree (uv, shared cache): mlx-lm, vllm-mlx, livekit-agents[deepgram,cartesia,silero,turn-detector,openai], psycopg; pin in a lock file.
- Three checks: `mlx_lm.generate` prints tok/s; vllm-mlx streams a chat completion with a tool definition; LiveKit agent runs in console mode against the local server.
- Repo scaffold: CONTRACT.md, RUNBOOK.md, schemas/ (tools, events, state), services/ (proxy, agent, tools), eval/, data/, train/, ui/, db/, scripts/ (restart, GPU flip), .env.example; ports fixed (server 8000, proxy 8001, UI 3000, Postgres 5432).
- Conductor: one workspace per session, setup script copies .env and runs `uv sync`; opening prompts drafted in `prompts/`.
- Postgres up with migrations (sessions, turn events, eval results, cost ledger).

## GPU schedule (hours from start)
| Window | GPU owner | Everyone else |
| --- | --- | --- |
| T+1–4 | S: skeleton | local URL |
| T+4–7 | S: benchmarks, prefix cache, KV persistence | local |
| T+7–8 | T: smoke adapter → fuse → quantize → serve (G2) | hosted during the run |
| T+8–10 | S: load tests 1/2/4/8 calls | local |
| T+10–12 | T: run 1 | hosted |
| T+12–13 | eval run 1, precision gate | local |
| T+13–13.5 | T: routing adapter | hosted |
| T+13.5–18 | S: deploy candidate, fallback, routing job | local |
| T+18–19.5 | T: run 2 (only if failure labels justify it) | hosted |
| T+19.5–24 | eval, final pick (G4 at T+20.5), serve for rehearsal | local |

Your fixed-time work: contract T+0–1; test scenarios T+5–6 (27, three per family); data spot-check T+9–10; listen to 10 candidate calls T+12–13; failure labeling T+16–18; rehearsal + write-up T+22–24. T+14–16 needs nothing from you.

## Gates
- G1 T+4: live browser call completes a tool lookup on the base model; nothing structural spoken; per-stage latency in Postgres.
- G2 T+8: smoke adapter through train → fuse → quantize → serve → LiveKit → scored by the runner. No real run before this.
- G3 T+13: candidate deployed under a versioned id, within 1 point of the unfused adapter on the 8-bit base on the test set, latency targets held at 4 simulated calls.
- G4 T+20.5: final model frozen; after this only rehearsal, recordings, write-up.

## Sessions and exits
- S (serving): skeleton by T+4; server pick, tool-call stream parsing as OpenAI tool_calls deltas (thin proxy if neither server does it), thinking off + `<think>` guard; benchmarks; adapter serve chain; load tests; admission limit; hosted fallback at turn boundaries on first-token timeout; routing batch job; restart script.
- E (eval + data): caller simulator + deterministic scorers (final sandbox state, tool name/args, no success claim before tool result, turn budget, handoff correctness, corrected details retained) by T+5; text baselines (base, hosted) + ranked failure list; generation T+5–9 with sandbox tools in the loop; filter instead of correct; STT shaping; interruption slice; ≤3 targets per conversation; masks decoded by hand; splits by family and persona; 27 hand-written scenarios are the test set, never generated.
- T (training): scripts T+1–6 (config, loader, fuse/quantize, launcher that logs config/git/manifest hashes, eval-on-checkpoint); smoke T+7; run 1 T+10 sized from smoke tok/s; selection by the runner, never loss; precision gate; routing T+13; run 2 changes data only.
- D (demo): Next.js over Postgres (transcript, stage, verified vs claimed facts, tool calls, model selector, per-stage latency, outcome); cost ledger per call (Deepgram, Cartesia, LiveKit, LLM tokens; local path = Mac hourly cost ÷ calls/hour at the admission limit; hosted at list); replay through any model; five scripted scenarios × 5 runs, recorded; everything labeled live / recorded / scripted.

Two-session variant (serving vs training): S absorbs D, T absorbs E. Critical path unchanged (E feeds T in sequence); review load halves; the UI loses parallel attention (LiveKit playground covers the demo). Every session hands background work (generation batches, load tests, benchmark sweeps) to subagents.

## Proposed targets (set the rest after the T+5 baseline)
End of speech → first audio p50 ≤ 1.0 s, p95 ≤ 1.6 s · STT final ≤ 0.5 s · model TTFT p50 ≤ 0.35 s at 4 calls · prefill tokens ≤ 150 on turns 2+ · inter-token ≤ 50 ms at 4 calls · TTS first audio ≤ 0.2 s · 4 concurrent calls · tool-arg accuracy ≥ 95% (ordinary) · unsupported success claims 0 · stale-detail errors 0 (correction family).

## Cut order when behind
Routing track → speculative decoding test → run 2 → Next.js UI (LiveKit playground instead) → cost-ledger automation → interruption slice → audio baseline.

## Pre-decided responses
- Tool-call streaming unparsed: proxy, one hour, permanent.
- Prefill too slow (G1 TTFT p50 > 0.6 s at 1 call): 2K cap, trimmed schemas, cache confirmed; still over at T+10 → live demo path hosted, adapter shown in replay + text eval.
- Run 1 too slow: targets/conversation 3 → 2, then conversations → 600; never raise seq or epochs.
- Candidate not better than base: check masks and chat template first, then data mix; run 2 is the one retry.
- Thinking leaks: template flag + proxy guard, both.
