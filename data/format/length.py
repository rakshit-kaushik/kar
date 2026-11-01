"""The 2,048-token check per training example, counted the way mlx_lm/T's loader renders it.

Uses mlx_lm 0.32.2 `load_tokenizer` (a TokenizerWrapper around the Hugging Face tokenizer) and
`apply_chat_template(messages, tools=..., tokenize=True, return_dict=False)`; the full sequence is
prefix + target, the prefix alone is rendered with add_generation_prompt and thinking disabled,
exactly as train/loader.py does. Loading the tokenizer needs only the model directory's tokenizer
files (no weights).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL_DIR = (
    Path(os.environ.get("R3_ARTIFACTS", "~/r3-artifacts")).expanduser() / "models" / "Qwen3-4B-8bit"
)


class LengthChecker:
    def __init__(self, model_path: str | Path | None = None, max_seq: int = 2048):
        from mlx_lm.utils import load_tokenizer

        self.model_path = Path(model_path or DEFAULT_MODEL_DIR)
        self.tokenizer = load_tokenizer(self.model_path)
        self.max_seq = max_seq

    def _template(self, messages: list[dict], tools: list[dict] | None, **kw) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=False, **kw
            )
        )

    def n_tokens(self, example: dict) -> int:
        return len(self._template(example["messages"], example.get("tools")))

    def prefix_tokens(self, example: dict) -> int:
        return len(
            self._template(
                example["messages"][:-1],
                example.get("tools"),
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    def fits(self, example: dict) -> bool:
        return self.n_tokens(example) <= self.max_seq
