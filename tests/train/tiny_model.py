"""A tiny random-init Qwen3 (2 layers, hidden 64) with the REAL Qwen3-4B tokenizer and chat template,
quantized to 8-bit the way the base is. MLX's CPU quantized matmul runs at scalar speed, so this is
what makes an off-GPU end-to-end dry run (lora -> fuse -> 8-bit -> versioned dir) finish in minutes.
Losses on it mean nothing; every file format and code path is the real one."""
from __future__ import annotations

from pathlib import Path

CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 151936,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": True,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}


def make_tiny_qwen3_8bit(out_dir: Path, tokenizer_dir: Path, seed: int = 0) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_map
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import load_tokenizer, quantize_model, save

    mx.set_default_device(mx.cpu)
    mx.random.seed(seed)
    config = dict(CONFIG)
    model = Model(ModelArgs.from_dict(config))
    model.update(tree_map(lambda p: p.astype(mx.bfloat16), model.parameters()))
    model, config = quantize_model(model, config, group_size=64, bits=8)
    tokenizer = load_tokenizer(Path(tokenizer_dir))
    out_dir = Path(out_dir)
    save(out_dir, Path(tokenizer_dir), model, tokenizer, config)
    return out_dir


if __name__ == "__main__":
    import sys

    print(make_tiny_qwen3_8bit(Path(sys.argv[1]), Path(sys.argv[2])))
