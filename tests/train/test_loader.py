"""Exit check for task 2: the loader reads a manifest, proves the chat template renders with
thinking disabled, and puts the loss on exactly the target assistant turn."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from train.loader import (
    EMPTY_THINK,
    check_dataset,
    format_mask,
    load_manifest,
    manifest_hash,
    read_jsonl,
    render_and_mask,
    validate_example,
)

REPO = Path(__file__).resolve().parents[2]


def _examples(synth_dir):
    return read_jsonl(synth_dir / "train.jsonl")


def _first(examples, kind):
    return next(e for e in examples if e["meta"]["target_kind"] == kind)


def test_manifest_lists_files_and_hashes_verify(synth_dir):
    m = load_manifest(synth_dir)
    assert set(m.files) == {"train", "valid"}
    assert m.files["train"] == synth_dir / "train.jsonl"
    assert len(manifest_hash(m)) == 64


def test_tampered_file_fails_hash_check(synth_dir, tmp_path):
    import shutil

    d = tmp_path / "ds"
    shutil.copytree(synth_dir, d)
    with (d / "train.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(d)


def test_validate_example_catches_template_breaking_shapes(synth_dir):
    ok = _first(_examples(synth_dir), "tool_call")
    assert validate_example(ok) == []
    bad = json.loads(json.dumps(ok))
    bad["messages"][-1]["content"] = None
    assert any("content" in e for e in validate_example(bad))
    bad = json.loads(json.dumps(ok))
    bad["messages"].append({"role": "tool", "content": "{}"})
    assert any("assistant" in e for e in validate_example(bad))
    bad = json.loads(json.dumps(ok))
    del bad["tools"]
    assert any("tools" in e for e in validate_example(bad))
    assert validate_example(bad, require_tools=False) == []


def test_loss_covers_exactly_the_target_turn_with_tool_calls(tokenizer, synth_dir):
    ex = _first(_examples(synth_dir), "tool_call")
    me = render_and_mask(tokenizer, ex, max_seq_length=2048)
    assert me.errors == []
    assert me.prefix_text.endswith("<|im_start|>assistant\n")
    assert me.target_text.startswith(EMPTY_THINK)  # thinking disabled, rendered by the template
    for call in ex["messages"][-1]["tool_calls"]:
        assert f'<tool_call>\n{{"name": "{call["function"]["name"]}"' in me.target_text
    assert "<tool_response>" not in me.target_text
    assert me.target_text.endswith("<|im_end|>\n")
    assert me.n_target == me.n_tokens - me.offset
    assert all(me.checks.values()), me.checks


def test_tool_results_in_the_prefix_are_never_in_the_loss(tokenizer, synth_dir):
    ex = _first(_examples(synth_dir), "closing")
    assert any(m["role"] == "tool" for m in ex["messages"])
    me = render_and_mask(tokenizer, ex, max_seq_length=2048)
    assert "<tool_response>" in me.prefix_text
    assert "<tool_response>" not in me.target_text
    assert me.checks["no_tool_result_in_loss"] and me.checks["prefix_aligned"]


def test_serving_prompt_with_thinking_off_is_a_prefix_of_the_training_sequence(tokenizer, synth_dir):
    ex = _first(_examples(synth_dir), "clarification")
    me = render_and_mask(tokenizer, ex, max_seq_length=2048)
    served = tokenizer.apply_chat_template(
        ex["messages"][:-1], tools=ex["tools"], add_generation_prompt=True, enable_thinking=False
    )
    assert me.tokens[: len(served)] == list(served)
    assert me.checks["serve_prompt_is_prefix"]


def test_check_dataset_reports_stats_and_passes_on_synth(tokenizer, synth_dir):
    rep = check_dataset(tokenizer, _examples(synth_dir), max_seq_length=2048)
    assert rep.n == 20 and rep.n_failed == 0
    assert rep.stats["max_tokens"] <= 2048
    assert 0 < rep.stats["mean_target_tokens"] < rep.stats["mean_tokens"]
    assert rep.stats["total_tokens"] == sum(m.n_tokens for m in rep.masks)


def test_over_length_examples_fail_instead_of_being_truncated_silently(tokenizer, synth_dir):
    rep = check_dataset(tokenizer, _examples(synth_dir), max_seq_length=64)
    assert rep.n_failed == rep.n
    assert all(not m.checks["fits_seq"] for m in rep.masks)


def test_format_mask_shows_boundary_and_loss_region(tokenizer, synth_dir):
    ex = _first(_examples(synth_dir), "tool_call")
    text = format_mask(render_and_mask(tokenizer, ex, max_seq_length=2048), index=3)
    assert "LOSS" in text and "masked" in text.lower() and "<tool_call>" in text
    assert "<|im_start|>assistant" in text


def test_cli_prints_20_masks_and_exits_zero(synth_dir, base_4b):
    out = subprocess.run(
        [sys.executable, "-m", "train.loader", "--data", str(synth_dir), "--model", str(base_4b), "--n", "20"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.count("=== example") == 20
    assert "ALL CHECKS PASSED" in out.stdout
