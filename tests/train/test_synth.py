"""Exit check for the 20-example synthetic dataset used by every dry run (7-tool contract)."""
import hashlib
import json
from pathlib import Path

from train.prompt import render_system_prompt
from train.synth import SYNTH_NOW, write_dataset


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
    assert m["system_prompt"]["path"].endswith("system_prompt.md") and len(m["tools"]) == len(list((Path(__file__).resolve().parents[2] / "schemas" / "tools").glob("*.json")))


def test_examples_are_contract_shaped(synth_dir):
    prompt = render_system_prompt(SYNTH_NOW)
    names, kinds = set(), set()
    for ex in _read(synth_dir / "train.jsonl"):
        tool_names = [t["function"]["name"] for t in ex["tools"]]
        assert tool_names == sorted(tool_names) and len(tool_names) >= 7
        assert ex["messages"][0] == {"role": "system", "content": prompt}
        assert ex["messages"][1]["role"] == "user"  # the greeting and phone question were already spoken
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
    assert {"lookup_account", "check_availability", "book_appointment", "create_ticket", "request_handoff"} <= names
    assert True


def test_every_tool_call_in_a_prefix_is_followed_by_its_results(synth_dir):
    for ex in _read(synth_dir / "train.jsonl"):
        msgs = ex["messages"]
        for i, msg in enumerate(msgs[:-1]):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                n = len(msg["tool_calls"])
                assert [m["role"] for m in msgs[i + 1 : i + 1 + n]] == ["tool"] * n
                assert [m["tool_call_id"] for m in msgs[i + 1 : i + 1 + n]] == [c["id"] for c in msg["tool_calls"]]


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_dataset(a, n_train=20, n_valid=4, seed=0)
    write_dataset(b, n_train=20, n_valid=4, seed=0)
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()
    assert (a / "valid.jsonl").read_bytes() == (b / "valid.jsonl").read_bytes()


def test_tools_and_lost_card_flow_follow_the_schema_directory(synth_dir):
    from train.synth import TOOLS_DIR

    expected = sorted(p.stem for p in TOOLS_DIR.glob("*.json"))
    rows = _read(synth_dir / "train.jsonl") + _read(synth_dir / "valid.jsonl")
    assert all([t["function"]["name"] for t in ex["tools"]] == expected for ex in rows)
    called = {c["function"]["name"] for ex in rows for m in ex["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    assert called <= set(expected)
    assert ("block_card" in called) == ("block_card" in expected)  # the lost-card scenarios switch with the contract
