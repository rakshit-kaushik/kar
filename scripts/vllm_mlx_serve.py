#!/usr/bin/env python
"""Start vllm-mlx with the prefix-trie key fix applied (services/server/trie_keyfix.py), same arguments as `vllm-mlx`.
restart.sh: .venv/bin/python scripts/vllm_mlx_serve.py serve <model> --prefix-trie-cache ..."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from services.server import trie_keyfix  # noqa: E402

trie_keyfix.apply()
from vllm_mlx.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
