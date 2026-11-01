"""Claim guard (reviewer, from the run-1 failure labels): the model says "I'm transferring you" without calling request_handoff
in 5 of 81 episodes; the agent records the handoff through the executor before the sentence completes and logs claim_guard."""
from services.agent.claimguard import ClaimGuard, claim_guarded_stream, handoff_reason, state_view, transfer_claimed


class Delta:
    def __init__(self, content=None, tool_calls=()):
        self.content, self.tool_calls = content, list(tool_calls)


class Chunk:
    def __init__(self, content=None, tool_calls=()):
        self.delta = Delta(content, tool_calls)


async def agen(items):
    for it in items:
        yield it


def test_transfer_phrases():
    assert transfer_claimed("I've opened your ticket and I'm transferring you to a specialist.")
    assert transfer_claimed("A specialist will take over from here.")
    assert transfer_claimed("Let me get a colleague to help you.")
    assert not transfer_claimed("Let me check availability for Tuesday.")
    assert not transfer_claimed("You can transfer money between your accounts online.")


def test_handoff_reason_from_the_call_state():
    fraud = {"completed": [{"tool": "create_ticket", "args": {"category": "unauthorized_transaction"}, "result": {"status": "open"}}], "verification_failures": 0, "verified": {"customer_id": "C-1002"}}
    assert handoff_reason(fraud) == "suspected_fraud"
    assert handoff_reason({"completed": [], "verification_failures": 2, "verified": {"customer_id": None}}) == "verification_failed"
    assert handoff_reason({"completed": [], "verification_failures": 0, "verified": {"customer_id": "C-1001"}}) == "out_of_scope"


async def test_stream_records_the_handoff_before_the_completing_chunk_and_only_once():
    order, chunks = [], [Chunk("I've opened your ticket"), Chunk(" and I'm transferring"), Chunk(" you to a specialist.")]

    async def on_claim(text):
        order.append(("claim", text))

    out = []
    async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: False, on_claim):
        order.append(("chunk", c))
        out.append(c)
    assert out == chunks
    # "transferring you" completes on the third chunk: the handoff is recorded before that chunk is yielded, never before
    assert order == [("chunk", chunks[0]), ("chunk", chunks[1]), ("claim", "I've opened your ticket and I'm transferring you to a specialist."), ("chunk", chunks[2])]


async def test_stream_leaves_a_turn_alone_when_the_handoff_already_exists_or_when_disabled():
    chunks = [Chunk("I'm transferring you now.")]

    async def boom(text):
        raise AssertionError("should not intervene")

    assert [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: True, boom)] == chunks
    assert [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(enabled=False), lambda: False, boom)] == chunks
    tool = Chunk(None, tool_calls=["request_handoff"])
    assert [c async for c in claim_guarded_stream(agen([tool]), ClaimGuard(), lambda: False, boom)] == [tool]


def test_env_switch(monkeypatch):
    monkeypatch.setenv("AGENT_CLAIM_GUARD", "0")
    assert ClaimGuard.from_env().enabled is False
    monkeypatch.delenv("AGENT_CLAIM_GUARD")
    assert ClaimGuard.from_env().enabled is True


def test_state_view_carries_the_verification_failure_count_without_widening_the_schema():
    from services.agent.state import CallState
    s = CallState(call_id="c", model_id="m")
    s.verification_failures = 2
    assert "verification_failures" not in s.to_dict()
    assert state_view(s)["verification_failures"] == 2 and handoff_reason(state_view(s)) == "verification_failed"
