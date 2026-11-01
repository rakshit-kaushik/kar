"""vllm-mlx 0.5.0 keys mlx-lm's LRUPromptCache by the model object; an MLX nn.Module is a dict subclass and unhashable,
so every prefix-trie insert and lookup raised TypeError (logged at DEBUG only) and the cache never served a turn.
services/server/trie_keyfix.py makes the engine key the trie by id(model). CPU-only: no model weights."""
import threading

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


def fake_cache(n: int) -> list:
    cs = []
    for _ in range(2):
        c = KVCache(); c.update_and_fetch(mx.zeros((1, 1, n, 4)), mx.zeros((1, 1, n, 4))); mx.eval(c.keys); cs.append(c)
    return cs


def bare_engine():
    from vllm_mlx.engine.simple import SimpleEngine
    eng = SimpleEngine.__new__(SimpleEngine)
    eng._prefix_trie_cache_enabled = True; eng._prefix_trie_cache_size = 32; eng._prefix_trie_cache_memory_mb = None
    eng._prefix_trie_cache = None; eng._prefix_trie_cache_lock = threading.Lock(); eng._is_mllm = False
    eng._prefix_trie_cache_stats = {"lookups": 0, "hits": 0, "misses": 0, "skips": 0, "inserts": 0, "tokens_saved": 0}
    return eng


def test_next_turn_hits_the_stored_prompt_plus_generation_with_a_module_as_key():
    from services.server import trie_keyfix
    trie_keyfix.apply()
    mx.set_default_device(mx.cpu)
    eng, model = bare_engine(), nn.Linear(2, 2)
    stored = list(range(1404)) + [9001, 9002, 9003, 9004] + list(range(2000, 2010))  # prompt + think block + reply
    next_prompt = list(range(1404)) + list(range(2000, 2010)) + list(range(3000, 3020))  # history drops the think block
    eng._insert_prefix_trie_cache(model, stored, fake_cache(len(stored)))
    cache, rest, saved = eng._fetch_prefix_trie_cache(model, next_prompt, minimum_tokens_saved=0)
    assert eng._prefix_trie_cache_stats["inserts"] == 1 and eng._prefix_trie_cache_stats["skips"] == 0
    assert cache is not None and len(rest) == 30 and saved == 1404


def test_apply_is_idempotent():
    from services.server import trie_keyfix
    trie_keyfix.apply(); trie_keyfix.apply()
    from vllm_mlx.engine.simple import SimpleEngine
    assert getattr(SimpleEngine._fetch_prefix_trie_cache, "_r3_keyfix", False)
