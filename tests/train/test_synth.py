"""Exit check for the 20-example synthetic dataset used by every dry run."""
import hashlib
import json
from pathlib import Path

from train.synth import write_dataset


def _read(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_writes_20_train_4_valid_and_a_manifest(synth_dir):
    train, valid = _read(synth_dir / "train.jsonl"), _read(synth_dir / "valid.jsonl")
    assert len(train) == 20 and len(valid) == 4
    m = json.loads((synth_dir / "manifest.json").read_text())
    assert m["files"] == {"train": "train.jsonl", "valid": "valid.jsonl"}
    assert m["counts"] == {"train": 20, "valid": 4}
    for name, digest in m["sha256"].items():
        assert hashlib.sha256((synth_dir / name).read_bytes()).hexdigest() == digest
    assert m["format"] == "mlx_lm.messages+tools" and m["enable_thinking"] is False


def test_examples_are_contract_shaped(synth_dir):
    names = set()
    kinds = set()
    for ex in _read(synth_dir / "train.jsonl"):
        assert [t["function"]["name"] for t in ex["tools"]] == sorted(
            t["function"]["name"] for t in ex["tools"]
        )
        assert len(ex["tools"]) == 8
        assert ex["messages"][0]["role"] == "system"
        assert ex["messages"][-1]["role"] == "assistant"
        kinds.add(ex["meta"]["target_kind"])
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                assert isinstance(msg["content"], str)  # Qwen3 template cannot take null content
                for call in msg.get("tool_calls", []):
                    json.loads(call["function"]["arguments"])  # arguments are compact JSON strings
                    names.add(call["function"]["name"])
            if msg["role"] == "tool":
                assert isinstance(msg["content"], str)
    assert kinds == {"tool_call", "clarification", "closing"}
    assert {"lookup_account", "check_availability", "book_appointment"} <= names


def test_tool_results_precede_targets_and_never_end_a_prefix_with_a_tool_call(synth_dir):
    for ex in _read(synth_dir / "train.jsonl"):
        roles = [m["role"] for m in ex["messages"]]
        assert roles[-1] == "assistant"
        # a tool call turn in the prefix is always followed by its tool result
        for i, msg in enumerate(ex["messages"][:-1]):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                assert ex["messages"][i + 1]["role"] == "tool"


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(a, n_train=20, n_valid=4, seed=0)
    write_dataset(b, n_train=20, n_valid=4, seed=0)
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()
    assert (a / "valid.jsonl").read_bytes() == (b / "valid.jsonl").read_bytes()
