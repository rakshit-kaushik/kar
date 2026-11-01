"""Simulated caller that repeats a fact the agent asks for again, as a person would (rehearsal artefact, Nov 1: the fixed-line
caller fell a turn behind whenever the model re-asked the weekday or a credential)."""
import pytest

from services.demo.caller_script import AnsweringScript

LINES = ["five five five zero one four two two one nine", "march twelfth, nineteen eighty eight",
         "i'd like to book a mortgage consultation for tuesday morning at the market street branch", "nine thirty is fine", "yes, that's right"]
FACTS = {"phone": "five five five zero one four two two one nine", "dob": "march twelfth, nineteen eighty eight", "day": "tuesday", "time": "nine thirty", "branch": "market street"}


async def test_a_re_asked_fact_is_repeated_and_the_script_does_not_advance():
    b = AnsweringScript(LINES, FACTS)
    t = []
    for _ in range(3):  # phone, dob, booking request
        line = await b.next_line(t); t.append(("caller", line))
    t.append(("agent", "Which day would you like the mortgage consultation?"))
    assert await b.next_line(t) == "tuesday" and b.last_reask == "day"
    t.append(("caller", "tuesday")); t.append(("agent", "Which time works for you on Tuesday?"))
    assert await b.next_line(t) == "nine thirty is fine" and b.last_reask is None  # time was never given: the script answers


async def test_a_first_time_question_and_an_unknown_fact_take_the_next_scripted_line():
    b = AnsweringScript(LINES, FACTS)
    t = [("agent", "What's the phone number on your account?")]
    assert await b.next_line(t) == LINES[0]  # nothing given yet: not a re-ask
    t += [("caller", LINES[0]), ("agent", "And your date of birth?")]
    assert await b.next_line(t) == LINES[1]


async def test_a_re_asked_credential_is_repeated_with_punctuation_ignored_and_capped():
    b = AnsweringScript(["five five five, zero one four, two two one nine", "march twelfth, nineteen eighty eight", "book please"], FACTS, max_reasks=2)
    t = [("caller", "five five five, zero one four, two two one nine"), ("caller", "march twelfth, nineteen eighty eight")]
    b.i = 2
    t.append(("agent", "Could you read me the phone number again, one digit at a time?"))
    assert await b.next_line(t) == FACTS["phone"]
    t += [("caller", FACTS["phone"]), ("agent", "Could you give me your date of birth once more?")]
    assert await b.next_line(t) == FACTS["dob"]
    t += [("caller", FACTS["dob"]), ("agent", "Could you read me the phone number again?")]
    assert await b.next_line(t) == "book please"  # cap reached: the script moves on
    assert b.reasks == 2


from services.demo.caller_script import role_for  # noqa: E402


def test_transcript_role_comes_from_the_transcribed_track_not_from_the_attribute_being_present():
    # LiveKit Agents publishes both its own speech and the caller's transcription with lk.transcribed_track_id set; only the
    # caller's carries the caller's own microphone track id. The Nov 1 09:59 run labelled every line "caller" and the adaptive
    # caller never saw an agent question.
    assert role_for({"lk.transcribed_track_id": "TR_mic1"}, {"TR_mic1"}) == "caller"
    assert role_for({"lk.transcribed_track_id": "TR_agent9"}, {"TR_mic1"}) == "agent"
    assert role_for({}, {"TR_mic1"}) == "agent"
    assert role_for({"lk.transcribed_track_id": "TR_mic1"}, set()) == "agent"  # own track unknown yet: never claim it as the caller's


# Nov 1, run 2 redeployed: the fixed lines were tuned on run 1's flow (re-ask the day, ask the time, read back). Run 2 proposes a
# slot straight after the request and says goodbye after booking, so the scripted caller accepted a time it never wanted and
# kept talking after the call was over. A person states their own fact against a different proposal, does not repeat a fact
# already given, and hangs up after the agent's goodbye.
TIMEOUT_LINES = ["five five five zero one four two two one nine", "march twelfth, nineteen eighty eight",
                 "can you book me a mortgage consultation for tuesday morning at market street", "sure, go ahead", "nine thirty is fine",
                 "yes", "yes, please try again", "thanks, goodbye"]
ORDINARY_LINES = LINES + ["no, that's everything. thank you, goodbye"]
DANA = {**FACTS, "service": "mortgage consultation"}


async def _after_request(lines):
    b = AnsweringScript(lines, DANA)
    t = []
    for _ in range(3):
        t.append(("caller", await b.next_line(t)))
    return b, t


async def test_a_proposal_that_contradicts_the_callers_time_gets_the_fact_and_the_script_waits():
    b, t = await _after_request(TIMEOUT_LINES)
    t.append(("agent", "To confirm: a mortgage consultation at Market Street on Tuesday, November third, at ten in the morning. Shall I book it?"))
    assert await b.next_line(t) == "nine thirty, please" and b.last_reask == "time:corrected"
    t += [("caller", "nine thirty, please"), ("agent", "To confirm: a mortgage consultation at Market Street on Tuesday, November third, at nine thirty in the morning. Shall I book it?")]
    assert await b.next_line(t) == "sure, go ahead" and b.last_reask is None


async def test_a_range_of_times_gets_the_callers_time():
    b, t = await _after_request(ORDINARY_LINES)
    t.append(("agent", "To confirm: a mortgage consultation at Market Street on Tuesday, November third, at any time from nine in the morning to four thirty in the afternoon."))
    assert await b.next_line(t) == "nine thirty, please"


async def test_a_different_day_branch_or_service_gets_the_fact_but_a_list_of_options_does_not():
    b, t = await _after_request(ORDINARY_LINES)
    t.append(("agent", "To confirm: an account opening at Market Street on Tuesday, November third, at nine thirty in the morning. Shall I book it?"))
    assert await b.next_line(t) == "mortgage consultation, please" and b.last_reask == "service:corrected"
    t.append(("agent", "To confirm: a mortgage consultation at Mill Road on Wednesday, November fourth, at nine thirty in the morning. Shall I book it?"))
    assert (await b.next_line(t)).endswith(", please")
    b2, t2 = await _after_request(ORDINARY_LINES)
    t2.append(("agent", "Which branch would you like: Market Street, Marlow Street, Marker Street, or Mill Road?"))
    assert await b2.next_line(t2) == "market street" and b2.last_reask == "branch"  # a re-ask of a given fact, not a contradiction


async def test_a_re_asked_service_is_repeated():
    b, t = await _after_request(ORDINARY_LINES)
    t.append(("agent", "Which service would you like: account opening, mortgage consultation, loan application, or notary?"))
    assert await b.next_line(t) == "mortgage consultation" and b.last_reask == "service"


async def test_a_scripted_line_that_only_repeats_a_given_fact_is_skipped():
    b, t = await _after_request(ORDINARY_LINES)
    t.append(("agent", "To confirm: a mortgage consultation at Market Street on Tuesday, November third, at ten in the morning. Shall I book it?"))
    first = await b.next_line(t)
    assert first == "nine thirty, please"
    t.append(("caller", first))
    t.append(("agent", "To confirm: a mortgage consultation at Market Street on Tuesday, November third, at nine thirty in the morning. Shall I book it?"))
    assert await b.next_line(t) == "yes, that's right"  # "nine thirty is fine" says nothing new


async def test_the_agents_goodbye_gets_the_last_line_and_ends_the_call():
    b, t = await _after_request(ORDINARY_LINES)
    t.append(("agent", "I've booked your mortgage consultation at Market Street on Tuesday, November third, at nine thirty in the morning. Goodbye, Dana!"))
    assert await b.next_line(t) == "no, that's everything. thank you, goodbye" and b.last_reask == "farewell"
    t.append(("caller", "no, that's everything. thank you, goodbye"))
    assert await b.next_line(t) is None
