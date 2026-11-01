# R3 results — voice agent (Oct 31 – Nov 1, 2026)

What we built, how we measured it, what came out, and what to do differently.

| File | What's in it |
|---|---|
| [setup.md](setup.md) | The eval (test set, simulator, scorers), the training data, the training runs |
| [results.md](results.md) | Task quality, latency, live rehearsal, cost |
| [lessons.md](lessons.md) | What went wrong (training and operations) and what to keep in mind next time |

## Headline

| | Base Qwen3-4B | Run 1 | **Run 2 (final)** | Hosted (Sonnet 5.5) |
|---|---|---|---|---|
| Test set, every scorer passes | 12% | 16% | **62%** | 100% |
| Task completion | 59% | 76% | **85%** | 100% |
| Tool-call accuracy | 53% | 20% | **71%** | 100% |
| Unsupported success claims | 0 | 7 | **0** | 0 |

27 hand-written scenarios × 3 seeds, same simulated caller for the three local models, episodes where the caller got its own details wrong removed (55 left; 17 for base). Details and caveats in [results.md](results.md).

- **Fine-tuning worked:** run 2 is about five times base on the strict measure and never claimed an action it hadn't taken.
- **It is not close to hosted:** duplicate requests, noisy transcripts and multi-intent calls are still mostly failing.
- **Latency improved but missed target:** end of speech to first audio 1.57 s p50 / 2.88 s p95 against 1.0 / 1.6 s, one call at a time against a target of four.
- **Final model:** `r3-voice-730f8ba68e` (run 2). Rollback: `r3-voice-206506525c` (run 1).

> **Which weights the live numbers are from.** From 08:02 to 10:52 on Nov 1 the server ran run 1's weights under run 2's name (see [lessons.md](lessons.md)), so the 16/30 rehearsal, its latency and its cost are run 1's. Run 2 was redeployed at 10:52 with a serving check of the loaded weights; its first live numbers are in [results.md](results.md), and its rehearsal was still running when this was written (11:10). The text-eval numbers above are unaffected: they used explicit model folders.

Raw sources: `$R3_ARTIFACTS/runs/` (`SESSION-T-FINAL-REPORT.md`, `report-S-1101-0940.md`, `writeup-1101-0954.md`, `eval-batch-20261101-072547/`).
