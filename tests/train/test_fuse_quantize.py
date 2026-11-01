import json

import pytest

from train.fuse.fuse_quantize import layout, verify_model_dir


def _fake_model(d, bits=8):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "quantization": {"bits": bits, "group_size": 64}}))
    (d / "model.safetensors").write_bytes(b"")
    (d / "tokenizer_config.json").write_text("{}")


def test_verify_model_dir_accepts_an_8bit_model_with_weights_and_tokenizer(tmp_path):
    _fake_model(tmp_path / "m")
    info = verify_model_dir(tmp_path / "m", bits=8)
    assert info["bits"] == 8 and info["group_size"] == 64 and info["weight_files"] == ["model.safetensors"]


def test_verify_model_dir_rejects_other_precisions_and_missing_weights(tmp_path):
    _fake_model(tmp_path / "four", bits=4)
    with pytest.raises(ValueError, match="8"):
        verify_model_dir(tmp_path / "four", bits=8)
    _fake_model(tmp_path / "noweights")
    (tmp_path / "noweights" / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_model_dir(tmp_path / "noweights", bits=8)


def test_layout_puts_the_served_model_at_the_top_and_keeps_the_adapter(tmp_path):
    out = tmp_path / "r3-voice-abc"
    lay = layout(out)
    assert lay["fused"] == out and lay["adapter"] == out / "adapter" and lay["meta"] == out / "r3.json"


def test_sync_tokenizer_files_restores_the_base_tokenizer_byte_for_byte(base_4b, tmp_path):
    from train.fuse.fuse_quantize import sync_tokenizer_files, verify_tokenizer_parity
    from tiny_model import make_tiny_qwen3_8bit  # re-saves the tokenizer through transformers

    tiny = make_tiny_qwen3_8bit(tmp_path / "tiny", base_4b)
    assert (tiny / "tokenizer.json").read_bytes() != (base_4b / "tokenizer.json").read_bytes()
    assert (tiny / "chat_template.jinja").exists() and not (tiny / "vocab.json").exists()
    res = sync_tokenizer_files(base_4b, tiny)
    assert "vocab.json" in res["copied"] and res["removed"] == ["chat_template.jinja"]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
        assert (tiny / name).read_bytes() == (base_4b / name).read_bytes()
    assert verify_tokenizer_parity(base_4b, tiny) is True
