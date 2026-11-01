# Results

## Task quality (text eval, 27 scenarios × 3 seeds)

All four models were scored in the final batch of Nov 1, 07:25 (`eval-batch-20261101-072547`), on the same 26 removed caller-fault episodes. Base ran one seed per scenario.

| | Base | Run 1 | **Run 2** | Hosted |
|---|---|---|---|---|
| Every scorer passes | 11.8% | 16.4% | **61.8%** | 100% |
| Task completion | 58.8% | 76.4% | **85.5%** | 100% |
| Tool-call accuracy | 52.9% | 20.0% | **70.9%** | 100% |
| Unsupported success claims | 0 | 7 | **0** | 0 |
| Re-asked a given detail (advisory) | 1 | 19 | **0** | 0 |
| Episodes scored | 17 | 55 | 55 | 81 |
| Raw, nothing removed | 2/27 (7%) | 10/81 (12%) | **38/81 (47%)** | 81/81 |

Hosted was scored with the hosted caller, the others with the local 4B caller. The hosted row is a ceiling, not a like-for-like comparison.

**By family, run 2.** Raw counts include caller faults; the "removed" column excludes them.

| Family | Raw | Caller faults removed |
|---|---|---|
| ordinary | 6/9 | 100% |
| missing_info | 7/9 | 86% |
| correction | 7/9 | 78% |
| needs_person | 6/9 | 67% |
| ambiguous_branch | 6/9 | 62% |
| tool_timeout | 3/9 | 50% |
| multi_intent | 2/9 | 12% |
| duplicate | 0/9 | 0% (1 episode left) |
| noisy_transcript | 1/9 | 0% (1 episode left) |

For `duplicate` and `noisy_transcript`, 8 of 9 episodes were caller faults, so **those two families have effectively no clean measurement.**

**What run 2 fixed and what it didn't.** Failing episodes per scorer:

| Scorer | Run 1 | Run 2 |
|---|---|---|
| `tool_calls` | 44 | 16 |
| `final_state` | 13 | 8 |
| `handoff` | 10 | 9 |
| `no_success_claim_before_result` | 7 | 0 |
| `calendar_consistency` | 0 | 4 |

Run 2's 43 failing episodes, labelled by cause: missing example type 23, harness (mostly caller faults) 15, model capability 4, application logic 1. The largest remaining pattern was a missing handoff after a ticket or a double fault (10 episodes), followed by 7 missing lookups and 4 handoffs that weren't needed.

**Precision gate (passed).** Paired comparison on the same 17 episodes:

| | Fused 8-bit | Unfused adapter on the 8-bit base |
|---|---|---|
| Final state correct | 14 | 14 |
| Every scorer passes | 10 | 12 |
| Unsupported claims | 0 | 0 |

The serving path made a difference of about 1 to 3 episodes out of 17, which is within noise. The same fused weights scored 13/17 through mlx_lm.server and 10/17 through vllm-mlx.

## Latency (live calls, one at a time)

Run 1's column is the 30-call rehearsal (run 1's weights, 08:43–09:55). Run 2's column is its redeploy check at 10:52: only 3 calls (18 turns), so its p95 is rough. Both include the prefix-cache fix (Nov 1, 02:50).

| Measure | Target | Base | Run 1 (30 calls) | Run 2 (3 calls) |
|---|---|---|---|---|
| End of speech → first audio heard, p50 / p95 | 1.0 / 1.6 s | 3.06 / 12.0 s | **1.57 / 2.88 s** | **1.63 / 2.31 s** |
| End of speech → first audible TTS (agent clock), p50 | 1.0 s | 2.58 s | 1.27 s | 1.32 s |
| STT final after end of speech, p50 | 0.5 s | 0.29 s | 0.27 s | 0.28 s |
| Model time to first token (spoken turns), p50 | 0.35 s | 0.97 s | **0.15 s** | **0.16 s** |
| Model time to a whole tool call, p50 | — | — | — | 0.76 s |
| New tokens prefilled on turn 2+ | ≤ 150 | 1,400–2,300 | 1–45 | trie hit confirmed |
| TTS first byte, p50 | 0.2 s | 0.19 s | 0.16 s | 0.17 s |
| Concurrent calls within budget | 4 | 1 | **1** | **1** |

In run 2's three redeploy calls, every lookup and booking completed, with no unsupported claims, no guard interventions and no hosted fallbacks.

The biggest single gain was the prefix cache: it had never hit until a one-file key fix. What's left:

- Turns that call a tool wait about 0.7–1 s, because vllm-mlx sends a tool call only once it's complete.
- Turn detection plus STT take about 0.5 s.
- The phone hotspot adds about 0.3 s per turn.

vllm-mlx 0.5.0 handles one request at a time, so the admission limit is 1 call.

## Live rehearsal (simulated caller over LiveKit, 5 calls per scenario)

⚠ These calls ran on **run 1's weights** (served under run 2's name) with the agent's loop guard and claim guard on.

| Scenario | Model + guards | With two agent rules added | Failure seen |
|---|---|---|---|
| Correction mid-flow | 5/5 | — | — |
| Caller asks twice for a person | 5/5 | — | — |
| Ordinary booking | 3/5 | — | re-asked a weekday the caller had given |
| Interruption | 2/5 | 2/5 | barge-ins lost by STT; misread date of birth, then a transfer after one strike |
| Fraud report, then transfer | 1/5 | 5/5 (fraud-transfer rule) | right ticket every time, never transferred itself |
| Booking timeout, then retry | 0/5 | 5/5 (one-retry rule) | asked to retry, then never retried |

Overall 16/30 with the model and guards alone. **No unsupported success claim was spoken in 40 calls.** The loop guard stepped in on 4 calls and the claim guard on 7, including holding back two false "I booked" sentences.

**Run 2 rehearsal (its real weights, started 11:02, in progress when this was written):** ordinary booking **5/5**, with no guard or rule interventions. Run 1 scored 3/5 on the same scenario. The other scenarios' results will land in `$R3_ARTIFACTS/runs/rehearsal-r3-voice-730f8ba68e-1101-1102-rules/`.

## Cost

Measured minutes and characters at list price. The local model is costed as the Mac's hourly cost ($0.171/h) divided by calls per hour at an admission limit of 1, so that part is an estimate.

| | Per call / episode | Per success |
|---|---|---|
| Live call, rehearsal (30 calls) | $0.043 | $0.081 per passing call |
| Text eval, local model | $0.0003 | $0.0007 (run 2), $0.0045 (base) |
| Text eval, hosted model | $0.055 | $0.055 |

On a live call, about 90% of the cost is audio services (speech-to-text, text-to-speech, transport), not the model. In the text eval, the local model is about 180× cheaper per episode than hosted, and fine-tuning makes each success about 6× cheaper than base.
