"""Deterministic STT shaping of caller text (CONTRACT §3: lowercase, no punctuation, digits as words)."""

from eval.simulator.stt_shape import number_words, shape


def test_level0_lowercases_and_strips_punctuation():
    assert (
        shape("Hi, it's Dana. I need a Notary!", level=0, seed=1) == "hi it's dana i need a notary"
    )


def test_phone_number_spoken_digit_by_digit():
    assert shape("my number is 555-014-2219", 0, 1) == (
        "my number is five five five zero one four two two one nine"
    )
    assert shape("(555) 014 2219", 0, 1) == "five five five zero one four two two one nine"


def test_year_and_small_numbers_as_words():
    assert number_words("1988") == "nineteen eighty eight"
    assert number_words("2026") == "twenty twenty six"
    assert number_words("12") == "twelve"
    assert number_words("30") == "thirty"
    assert number_words("45") == "forty five"
    assert number_words("100") == "one hundred"
    assert number_words("4219") == "four two one nine"  # card digits: spelled out


def test_dates_times_and_ordinals():
    assert shape("March 12th 1988", 0, 1) == "march twelfth nineteen eighty eight"
    assert shape("Thursday at 2:30", 0, 1) == "thursday at two thirty"
    assert shape("at 9:00 or 11:00", 0, 1) == "at nine o'clock or eleven o'clock"
    assert shape("the 4th of November", 0, 1) == "the fourth of november"
    assert shape("card ending 4219", 0, 1) == "card ending four two one nine"


def test_level0_is_clean_and_deterministic():
    t = "i need to see someone about a mortgage next week"
    assert shape(t, 0, 1) == t
    assert shape(t, 2, 7) == shape(t, 2, 7)


def test_higher_levels_add_fillers_and_homophones_only():
    t = "i would like to book for two people on the fourth"
    out = shape(t, 3, 3)
    assert out != t
    allowed_extra = {"um", "uh", "like", "you", "know", "so", "yeah"}
    homophones = {"two": {"to", "too"}, "to": {"two", "too"}, "for": {"four"}, "fourth": {"forth"}}
    orig = t.split()
    for w in out.split():
        assert w in orig or w in allowed_extra or any(w in alts for alts in homophones.values()), w


def test_level_three_drops_or_repeats_but_keeps_most_words():
    t = "i want to cancel my appointment on friday at ten in the morning please"
    out = shape(t, 3, 11).split()
    kept = sum(1 for w in t.split() if w in out)
    assert kept >= 9
