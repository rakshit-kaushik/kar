# eval — simulator, scorers and the `r3-eval` analysis loop

The runner scores any OpenAI-compatible URL on a scenario directory with the contract's tools: a simulated
caller (hosted model, or any OpenAI-compatible URL via `--caller-base-url`) talks to the agent under test
against a fresh sandbox per episode; deterministic scorers judge the saved episode. Everything the agent said
and every `tool_calls` object it returned is kept verbatim in `episodes/*.json`.

```
uv run r3-eval run --base-url <url> --model <id> --scenarios eval/scenarios/test/ --repeats 3 --out <dir>
uv run r3-eval validate <scenario-dir>           # schema + seed consistency of scenario files
uv run r3-eval rescore <run-dir> [--scenarios eval/scenarios/test] [--db]   # re-score saved episodes, no model calls
uv run r3-eval compare <run-dir>... [--exclude-caller-faults]               # side-by-side all-pass per family, scorer rows
uv run r3-eval failures <run-dir> [--out failures.md]                       # every failing episode, scorer, turn, transcript window
uv run r3-eval buckets <run-dir> --dataset <training set> [--out buckets.md] # first-pass failure buckets (plan T+16)
```

## Scorers (`eval/scorers`)

One scorer per `expect` key (final_state, tool_calls, forbidden_tools, max_successful_calls, handoff, max_turns,
corrections_hold, must_mention) plus the generic ones: no_success_claim_before_result, verified_before_account_facts,
no_jargon, valid_tool_calls, calendar_consistency. Advisory, never failing an episode: brevity (two sentences,
35 words) and reasked_given_slot (the agent asked for a branch, day, time or service the caller had already given;
questions offering two or more candidates are clarifications; slots reset after a completed write).

## What the analysis commands add to the scorers

- `compare --exclude-caller-faults` drops every scenario x seed in which the simulated caller mis-stated its own
  phone or date of birth in any of the compared runs (`eval.runner.caller_fault`: verification expected, no lookup
  succeeded, the caller's raw reply never contained the true value; a caller that never gives a phone number at all
  counts too). The first line names the excluded ids; other sessions parse it. Runs that share a model id are
  labelled with their directory name.
- `failures.md` splits tool_calls failures by cause (missing call / wrong args with the fields / wrong outcome) and
  success claims by claimed action (transfer vs write).
- `buckets.md` labels each failing episode (bad data / missing example type / context truncation / model capability /
  application logic / harness) from scorer evidence and the training set's target and prefix counts: a missing call
  whose tool was common in prefixes but rare as a target is bad data (target mix); a handoff after a single
  not_verified is a missing example type even in a caller-fault episode; caller faults and stuck callers (one line
  repeated three times, or goodbye without hanging up) are harness. It also lists adapter repetition under the live
  loop guard's rule (rephrase each exact repeat, handoff at the third; farewells exempt) and unverified write attempts.
  The human's labels override every first-pass label.

Scripts used at the gates live in the worktree's `.context/` (`run1_analysis.sh`, `final_compare.sh`).
