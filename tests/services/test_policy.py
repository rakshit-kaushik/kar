"""Contract handoff rule as application logic (reviewer question, Nov 1 09:20): after a fraud-category ticket the agent must transfer.
In the 08:40 rehearsal the final model opened the right ticket 5 of 5 and transferred 0 of 5 on its own."""
from services.agent.policy import POLICY_LINE, HandoffPolicy, policy_stream, required_handoff


class Delta:
    def __init__(self, content=None, tool_calls=()):
        self.content, self.tool_calls = content, list(tool_calls)


class Chunk:
    def __init__(self, content=None, tool_calls=()):
        self.delta = Delta(content, tool_calls)


async def agen(items):
    for it in items:
        yield it


FRAUD_TICKET = {"tool": "create_ticket", "args": {"customer_id": "C-1002", "category": "unauthorized_transaction", "summary": "x"}, "result": {"status": "open"}}
LOST_CARD = {"tool": "create_ticket", "args": {"customer_id": "C-1002", "category": "card_lost_stolen", "summary": "x"}, "result": {"status": "open"}}
FEE_TICKET = {"tool": "create_ticket", "args": {"customer_id": "C-1002", "category": "fee_dispute", "summary": "x"}, "result": {"status": "open"}}


def test_required_handoff_after_a_fraud_category_ticket_and_only_then():
    assert required_handoff({"completed": [FRAUD_TICKET], "handoff": None}) == "suspected_fraud"
    assert required_handoff({"completed": [LOST_CARD], "handoff": None}) == "suspected_fraud"
    assert required_handoff({"completed": [FEE_TICKET], "handoff": None}) is None
    assert required_handoff({"completed": [], "handoff": None}) is None
    assert required_handoff({"completed": [FRAUD_TICKET], "handoff": {"requested": True, "reason": "suspected_fraud"}}) is None  # already done


async def test_the_transfer_line_is_appended_once_after_the_reply_and_the_handoff_is_recorded_first():
    chunks = [Chunk("I've opened your ticket "), Chunk("for the unauthorized charge.")]
    order = []

    async def on_policy(reason):
        order.append(("policy", reason))

    out = []
    async for c in policy_stream(agen(chunks), HandoffPolicy(), lambda: {"completed": [FRAUD_TICKET], "handoff": None}, on_policy):
        order.append(("chunk", c))
        out.append(c)
    assert out == chunks + [" " + POLICY_LINE]
    assert order[-2:] == [("policy", "suspected_fraud"), ("chunk", " " + POLICY_LINE)]  # recorded before the sentence is spoken


async def test_nothing_is_appended_when_the_handoff_exists_or_the_reply_calls_a_tool_or_the_rule_is_off():
    chunks = [Chunk("I've opened your ticket.")]

    async def boom(reason):
        raise AssertionError("should not intervene")

    done = lambda: {"completed": [FRAUD_TICKET], "handoff": {"requested": True, "reason": "suspected_fraud"}}
    assert [c async for c in policy_stream(agen(chunks), HandoffPolicy(), done, boom)] == chunks
    pending = lambda: {"completed": [FRAUD_TICKET], "handoff": None}
    tool = Chunk(None, tool_calls=["request_handoff"])
    assert [c async for c in policy_stream(agen([tool]), HandoffPolicy(), pending, boom)] == [tool]  # the model is acting this turn
    assert [c async for c in policy_stream(agen(chunks), HandoffPolicy(enabled=False), pending, boom)] == chunks


def test_env_switch(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_HANDOFF", "0")
    assert HandoffPolicy.from_env().enabled is False
    monkeypatch.setenv("AGENT_POLICY_HANDOFF", "1")
    assert HandoffPolicy.from_env().enabled is True
