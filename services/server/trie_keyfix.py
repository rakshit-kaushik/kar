"""Make vllm-mlx's SimpleEngine prefix trie work.

vllm-mlx 0.5.0 passes the model object as the key of mlx-lm's LRUPromptCache (``fetch_nearest_cache(model, tokens)``,
``insert_cache(model, ...)``). An MLX ``nn.Module`` is a dict subclass and unhashable, so every insert and lookup raised
``TypeError: unhashable type`` inside a ``try`` that counts a "skip" and logs at DEBUG. Result in production (Nov 1):
``prefix_trie_cache=True`` in the banner, zero hits in 290 live turns, every turn prefilling the whole 1.5-2.3K-token
prompt (0.4-0.55 s). The trie itself handles our pattern (the next turn's prompt shares all but the dropped empty
``<think>`` block with the stored prompt+generation, see tests/services/test_trie_keyfix.py).

``apply()`` wraps the two engine methods so the trie is keyed by ``id(model)`` when the model is unhashable. Used by
scripts/vllm_mlx_serve.py, the launcher restart.sh starts the server with. Pinned to vllm-mlx 0.5.0; re-check on upgrade.
"""
from __future__ import annotations

import functools
import logging

log = logging.getLogger("r3.trie_keyfix")


def _key(model):
    try:
        hash(model)
        return model
    except TypeError:
        return id(model)


def apply() -> None:
    from vllm_mlx.engine.simple import SimpleEngine

    if getattr(SimpleEngine._fetch_prefix_trie_cache, "_r3_keyfix", False):
        return
    orig_fetch, orig_insert = SimpleEngine._fetch_prefix_trie_cache, SimpleEngine._insert_prefix_trie_cache

    @functools.wraps(orig_fetch)
    def fetch(self, model, tokens, **kw):
        return orig_fetch(self, _key(model), tokens, **kw)

    @functools.wraps(orig_insert)
    def insert(self, model, cache_key, prompt_cache):
        return orig_insert(self, _key(model), cache_key, prompt_cache)

    fetch._r3_keyfix = insert._r3_keyfix = True
    SimpleEngine._fetch_prefix_trie_cache, SimpleEngine._insert_prefix_trie_cache = fetch, insert
    log.info("prefix trie keyed by id(model) (vllm-mlx 0.5.0 unhashable-module fix)")
