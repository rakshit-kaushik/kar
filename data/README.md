# data/ — generation, shaping, formatting, splits (session E, T+5 to T+9)

Rules agreed before generation starts (2026-11-01):

- **Generator = the eval simulator.** `eval.simulator.episode.run_episode` with the hosted model as
  the agent (prompt `schemas/system_prompt.md`, the 7 contract tools) against a fresh `Sandbox()`;
  every tool result in the data is real. Scenarios come from `eval/scenarios/gen/` and must not
  reuse a test scenario's persona + goal + correction combination.
- **Filter, do not correct.** Any episode that fails a non-advisory scorer is dropped
  (`eval.scorers.score_episode` + `all_pass`). Expect to generate ~1.5x what is kept.
- **Length check is per training example, not per episode.** A training example is
  prefix (system + tools + conversation so far) -> one assistant target turn. The closing turns of
  long calls are the ones that overflow 2,048 (S measured re-prefill growing from 16 to ~720 new
  tokens per turn). Drop only the overflowing target; keep the episode's tool-call and clarification
  targets. Use T's `train/loader.py` (`render_and_mask(tokenizer, ex, max_seq_length)`, dataset
  check with `n_over_seq`) on branch `r3-training-scripts` rather than a second renderer; agree the
  example shape with T first.
- **At most three targets per episode**, chosen to cover a tool call, a clarification and the
  closing turn; loss on exactly that turn, tool-call tokens included, tool results never.
- **Repeats and sampling.** The hosted endpoint returns 400 for `temperature` and `top_p`; vary
  phrasing with the caller `seed` only (the runner records `temperature_applied=false`).
- **STT shaping** of caller turns via `eval.simulator.stt_shape.shape` (rule set to be checked
  against G1 transcripts and a TTS -> Deepgram round trip of 50 turns).
- **Provenance.** Manifest per dataset: generator version (git sha), prompt file hash, scenario ids,
  scorer verdicts per episode, STT rule version; splits by scenario family and seed persona.
  Persona and paraphrase banks seed from `$R3_ARTIFACTS/datasets/{taskmaster (TM-1..TM-4), sgd}`;
  utterances used for test paraphrases never seed training personas.
