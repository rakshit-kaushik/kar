"""The contract system prompt loader: schemas/system_prompt.md byte-for-byte, `{today}` filled from the
sandbox clock (eval/sandbox/seed.json clock_spoken), tolerant of legacy placeholders and of a moved file."""
import datetime as dt

from train.prompt import PLACEHOLDERS, placeholder_values, render_system_prompt, sandbox_clock, system_prompt_path

CLOCK_SPOKEN = "Monday, November second, 2026, ten in the morning"


def test_default_path_is_the_contract_file_and_today_comes_from_the_sandbox_clock():
    assert system_prompt_path().name == "system_prompt.md"
    assert sandbox_clock() == {"clock": "2026-11-02T10:00", "clock_spoken": CLOCK_SPOKEN}
    text = render_system_prompt()
    assert f"Today is {CLOCK_SPOKEN}." in text
    assert text == system_prompt_path().read_text().replace("{today}", CLOCK_SPOKEN)  # byte-for-byte otherwise
    assert not any("{" + k + "}" in text for k in PLACEHOLDERS)


def test_legacy_placeholders_derive_from_the_clock_and_can_be_overridden(tmp_path, monkeypatch):
    values = placeholder_values()
    assert values["today"] == CLOCK_SPOKEN and values["weekday"] == "Monday"
    assert values["today_long"] == "November 2, 2026" and values["now"] == "10:00"
    p = tmp_path / "prompt.md"
    p.write_text("It is {weekday}, {today_long}; {today}; {now}. Unknown {other} stays.\n")
    monkeypatch.setenv("R3_SYSTEM_PROMPT_PATH", str(p))
    assert system_prompt_path() == p
    assert render_system_prompt(now=dt.datetime(2026, 11, 5, 14, 30)) == (
        f"It is Thursday, November 5, 2026; {CLOCK_SPOKEN}; 14:30. Unknown {{other}} stays.\n")
    assert render_system_prompt(today="X") == "It is Monday, November 2, 2026; X; 10:00. Unknown {other} stays.\n"
