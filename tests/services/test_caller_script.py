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
