"""Loop guard (E's finding: 6 of 81 run-1 episodes repeat a line 3+ times; seen live in 2 of 4 calls): a reply is held only
while it is still identical to the previous assistant line; an exact repeat is swallowed and rephrased, a second one hands off."""
from services.agent.loopguard import HANDOFF_LINE, REPHRASE, LoopGuard, TurnFilter, guarded_stream, normalize

PREV = "Which day would you like the mortgage consultation?"


class Delta:
    def __init__(self, content=None, tool_calls=()):
        self.content, self.tool_calls = content, list(tool_calls)


class Chunk:  # duck-typed livekit.agents.llm.ChatChunk
    def __init__(self, content=None, tool_calls=()):
        self.delta = Delta(content, tool_calls)


async def agen(items):
    for it in items:
        yield it


async def collect(stream):
    return [c async for c in stream]


async def no_loop(action, repeated):
    raise AssertionError(f"unexpected intervention {action}")


def test_normalize_ignores_case_punctuation_and_spacing():
    assert normalize(" Which day,  would you like?? ") == normalize("which day would you like")


def test_turn_filter_holds_while_identical_and_reports_the_repeat_at_the_end():
    f = TurnFilter(normalize(PREV))
    assert f.feed("Which day") == "hold"
    assert f.feed(" would you like the mortgage consultation?") == "hold"  # equal so far: only the end of the stream tells
    assert f.finish() == "repeat"


def test_turn_filter_releases_on_the_first_differing_word():
    assert TurnFilter(normalize(PREV)).feed("Which time") == "release"


def test_turn_filter_releases_a_longer_reply_that_starts_with_the_previous_line():
    f = TurnFilter(normalize("Thanks."))
    assert f.feed("Thanks.") == "hold"
    assert f.feed(" What's your date of birth?") == "release"


def test_turn_filter_releases_a_shorter_reply_at_the_end():
    f = TurnFilter(normalize(PREV))
    assert f.feed("Which day?") == "hold"
    assert f.finish() == "release"


def test_turn_filter_without_a_previous_line_releases_at_once():
    assert TurnFilter(None).feed("Hello") == "release"


async def test_guarded_stream_passes_a_normal_reply_through_in_order_and_remembers_it():
    guard = LoopGuard()
    chunks = [Chunk("Thanks"), Chunk(". What's your date of birth?")]
    assert await collect(guarded_stream(agen(chunks), guard, no_loop)) == chunks
    assert guard.prev == normalize("Thanks. What's your date of birth?") and guard.repeats == 0


async def test_guarded_stream_swallows_an_exact_repeat_rephrases_then_hands_off():
    guard, calls = LoopGuard(), []

    async def on_loop(action, repeated):
        calls.append((action, repeated))

    line = [Chunk("Which day"), Chunk(" would you like the mortgage consultation?")]
    assert await collect(guarded_stream(agen(line), guard, on_loop)) == line  # first time: spoken as is
    assert await collect(guarded_stream(agen(line), guard, on_loop)) == [REPHRASE]  # repeat: replaced
    assert calls == [("rephrase", PREV)] and guard.repeats == 1
    assert await collect(guarded_stream(agen(line), guard, on_loop)) == [HANDOFF_LINE]  # still stuck: hand off
    assert calls[-1] == ("handoff", PREV) and guard.repeats == 2
    fresh = [Chunk("Tuesday at nine thirty, Market Street. Shall I book it?")]
    assert await collect(guarded_stream(agen(fresh), guard, on_loop)) == fresh and guard.repeats == 0  # a new line resets


async def test_guarded_stream_lets_tool_calls_through_while_holding_text():
    guard = LoopGuard()
    guard.prev = normalize(PREV)
    tool, c1, c3 = Chunk(None, tool_calls=["call"]), Chunk("Which day"), Chunk(" works for you?")
    assert await collect(guarded_stream(agen([c1, tool, c3]), guard, no_loop)) == [tool, c1, c3]


async def test_guarded_stream_is_a_pass_through_when_disabled():
    guard = LoopGuard(enabled=False)
    guard.prev = normalize(PREV)
    line = [Chunk(PREV)]
    assert await collect(guarded_stream(agen(line), guard, no_loop)) == line


def test_env_switch(monkeypatch):
    monkeypatch.setenv("AGENT_LOOP_GUARD", "0")
    assert LoopGuard.from_env().enabled is False
    monkeypatch.delenv("AGENT_LOOP_GUARD")
    assert LoopGuard.from_env().enabled is True


async def test_a_repeated_farewell_is_not_a_loop():
    # the caller keeps saying goodbye without hanging up: saying goodbye twice is right, a rephrase would be absurd
    guard = LoopGuard()
    bye = [Chunk("Thanks for calling Marlow Mutual Bank. Goodbye, Dana.")]
    assert await collect(guarded_stream(agen(bye), guard, no_loop)) == bye
    assert await collect(guarded_stream(agen(bye), guard, no_loop)) == bye
    assert guard.repeats == 0
