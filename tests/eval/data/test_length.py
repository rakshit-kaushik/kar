"""data.format.length: the 2,048-token check per example with the Qwen3 tokenizer (as T's loader renders)."""

import os
from pathlib import Path

import pytest
from helpers import VERIFY, VERIFY_LINES, make_episode, say, scenario

from data.format.targets import select_targets, to_example

MODEL_DIR = (
    Path(os.environ.get("R3_ARTIFACTS", os.path.expanduser("~/r3-artifacts")))
    / "models"
    / "Qwen3-4B-8bit"
)
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer.json").exists(), reason="Qwen3 tokenizer not downloaded"
)


@pytest.fixture(scope="module")
def checker():
    from data.format.length import LengthChecker

    return LengthChecker(MODEL_DIR)


@pytest.mark.asyncio
async def test_counts_tokens_like_the_trainer_and_fits_2048(checker):
    ep = await make_episode(scenario(), VERIFY + [say("Bye.")], VERIFY_LINES + [("bye", True)])
    chosen, _ = select_targets(ep)
    ex = to_example(ep, chosen[0])
    n = checker.n_tokens(ex)
    assert 1400 <= n <= 1900, (
        n
    )  # prompt + 7 tools + greeting + two caller turns (T measured ~1,471 for the scaffold)
    assert checker.prefix_tokens(ex) < n
    assert checker.fits(ex) is True and checker.max_seq == 2048


@pytest.mark.asyncio
async def test_long_example_does_not_fit(checker):
    filler = say("I can help with that. " * 60)
    ep = await make_episode(
        scenario(), VERIFY + [filler, say("Bye.")], VERIFY_LINES + ["ok", ("bye", True)]
    )
    chosen, _ = select_targets(ep)
    closing = next(t for t in chosen if t.kind == "closing")
    ex = to_example(ep, closing)
    assert checker.fits(ex) is False and checker.n_tokens(ex) > 2048


def test_checker_is_a_fits_callback_for_select_targets(checker):
    assert callable(checker.fits)
