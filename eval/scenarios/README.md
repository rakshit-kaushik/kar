# Scenarios

Format: `schema.json` (human-owned, JSON Schema 2020-12). `test/` holds the 27 test scenarios
(three per family; drafted by session S, owned by the human; no session writes there; never used
for generation). `gen/` holds generation scenarios for the data phase; they must not reuse a test
scenario's persona + goal + correction combination.

Every episode starts from `eval/sandbox/seed.json` (clock Monday 2026-11-02 10:00; fixed customers,
cards, appointments, blocked slots). Expected states are therefore exact.

Check a directory (schema + seed consistency: customer exists, phone/DOB match, slots open,
appointment ids exist, tool names valid):

    uv run r3-eval validate eval/scenarios/test/

Score an endpoint (every scenario 3 times, caller seeds 0/1/2); before any run against the local
server take the GPU lock (`scripts/gpu_lock.sh acquire E <minutes> "<purpose>"`) and tell session S:

    uv run r3-eval run --base-url <url> --model <id> --scenarios eval/scenarios/test/ --repeats 3 \
        --out $R3_ARTIFACTS/runs/<id>/ [--parallel 4 --retries 6 --backoff 5]
    uv run r3-eval rescore $R3_ARTIFACTS/runs/<id>/ --db --scenarios eval/scenarios/test/   # after scorer changes
    uv run r3-eval compare $R3_ARTIFACTS/runs/<a>/ $R3_ARTIFACTS/runs/<b>/                   # side-by-side table

Scorers, one per `expect` key: `final_state` (diff vs seed, `no_other_changes`), `tool_calls`
(ordered subsequence, subset args, `ok`), `forbidden_tools`, `max_successful_calls`, `handoff`,
`max_turns`, `corrections_hold`, `must_mention`; always on: `no_success_claim_before_result`,
`verified_before_account_facts` (unless `must_not_mention_before_verified: false`), `no_jargon`,
and `brevity` (advisory: reported, never fails an episode). Each result names the caller turn
where it failed. Costs for the ranked failure list: `eval/scorers/__init__.py` (`WEIGHTS`).

Run one scenario live and read the transcript:

    uv run python -m eval.simulator eval/scenarios/test/correction_01.json --seed 1
