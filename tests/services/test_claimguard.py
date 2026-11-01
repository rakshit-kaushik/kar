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


# --- completed-action claims (reviewer, Nov 1 08:15): a booking/reschedule/cancel/ticket claim with no matching successful write in
# the call state is held at the sentence boundary and replaced by a question; read-backs, questions and future tense pass untouched.
from services.agent.claimguard import action_claimed, replacement_line


def test_action_claim_phrases_are_completed_actions_only():
    assert action_claimed("Your mortgage consultation is booked at the Market Street branch on Tuesday.") == ("booked", "book_appointment")
    assert action_claimed("I booked your mortgage consultation for Tuesday at nine thirty.") == ("booked", "book_appointment")
    assert action_claimed("You're booked for Tuesday, November third, at nine thirty.") == ("booked", "book_appointment")
    assert action_claimed("I've moved your appointment to Thursday at two.") == ("moved", "reschedule_appointment")
    assert action_claimed("Your appointment has been rescheduled to Thursday.") == ("rescheduled", "reschedule_appointment")
    assert action_claimed("I've cancelled your appointment for Tuesday.") == ("cancelled", "cancel_appointment")
    assert action_claimed("I've opened an unauthorized transaction ticket for the charge.") == ("ticket", "create_ticket")
    assert action_claimed("Your ticket has been created.") == ("ticket", "create_ticket")
    # read-backs, questions, future tense and offers are not claims
    assert action_claimed("To confirm: a mortgage consultation at the Market Street branch on Tuesday at nine thirty. Shall I book it?") is None
    assert action_claimed("I'll book that for you now.") is None
    assert action_claimed("I'll open an unauthorized transaction ticket for a charge of two hundred dollars. Shall I go ahead?") is None
    assert action_claimed("Would you like me to move it to Thursday?") is None
    assert action_claimed("That slot is already booked; nine thirty is open.") is None
    assert action_claimed("Which day would you like the mortgage consultation?") is None


def test_replacement_line_is_a_question_per_action():
    assert replacement_line("book_appointment").endswith("?") and "booked" in replacement_line("book_appointment")
    assert replacement_line("reschedule_appointment").endswith("?") and replacement_line("cancel_appointment").endswith("?")
    assert replacement_line("create_ticket").endswith("?") and "ticket" in replacement_line("create_ticket")


async def test_an_unsupported_booking_claim_is_swallowed_whole_and_replaced_by_the_question():
    chunks = [Chunk("Your mortgage "), Chunk("consultation is booked at the Market Street branch "), Chunk("on Tuesday at nine thirty. "), Chunk("Goodbye, Dana.")]
    seen = []

    async def on_action_claim(kind, tool, sentence):
        seen.append((kind, tool, sentence))

    out = [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: False, lambda t: None, completed_tools=lambda: set(), on_action_claim=on_action_claim)]
    assert out == [replacement_line("book_appointment")]  # nothing of the claim or of the rest of the reply reaches the TTS
    assert seen == [("booked", "book_appointment", "Your mortgage consultation is booked at the Market Street branch on Tuesday at nine thirty.")]


async def test_a_booking_claim_backed_by_a_successful_booking_passes_untouched():
    chunks = [Chunk("Your mortgage consultation is booked "), Chunk("for Tuesday at nine thirty. Goodbye, Dana.")]

    async def boom(*a):
        raise AssertionError("should not intervene")

    out = [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: False, boom, completed_tools=lambda: {"book_appointment"}, on_action_claim=boom)]
    assert out == chunks


async def test_read_backs_and_questions_stream_untouched_sentence_by_sentence():
    chunks = [Chunk("To confirm: a mortgage consultation on Tuesday at nine thirty. "), Chunk("Shall I "), Chunk("book it?")]
    pulled, first_out_at = 0, None

    async def source():
        nonlocal pulled
        for c in chunks:
            pulled += 1
            yield c

    async def boom(*a):
        raise AssertionError("should not intervene")

    out = []
    async for c in claim_guarded_stream(source(), ClaimGuard(), lambda: False, boom, completed_tools=lambda: set(), on_action_claim=boom):
        if first_out_at is None:
            first_out_at = pulled
        out.append(c)
    assert out == chunks
    assert first_out_at == 1  # the first sentence is released as soon as it is complete, not at the end of the reply


async def test_a_tool_call_chunk_flushes_the_held_text_first():
    tool = Chunk(None, tool_calls=["book_appointment"])
    chunks = [Chunk("One moment"), tool]

    async def boom(*a):
        raise AssertionError("should not intervene")

    out = [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: False, boom, completed_tools=lambda: set(), on_action_claim=boom)]
    assert out == chunks


async def test_transfer_and_action_guards_work_together_in_one_reply():
    chunks = [Chunk("I've opened your ticket"), Chunk(" and I'm transferring you to a specialist.")]
    order = []

    async def on_claim(text):
        order.append(("transfer", text))

    async def on_action_claim(kind, tool, sentence):
        order.append(("action", tool))

    out = [c async for c in claim_guarded_stream(agen(chunks), ClaimGuard(), lambda: False, on_claim, completed_tools=lambda: set(), on_action_claim=on_action_claim)]
    # the ticket claim is unsupported: the whole sentence is replaced; the transfer phrase in the same sentence still records the handoff
    assert out == [replacement_line("create_ticket")]
    assert order == [("transfer", "I've opened your ticket and I'm transferring you to a specialist."), ("action", "create_ticket")]
