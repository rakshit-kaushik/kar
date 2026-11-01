"""The contract system prompt loader follows the file the human chooses and fills whatever placeholders it has."""
import datetime as dt

from train.prompt import PLACEHOLDERS, placeholder_values, render_system_prompt, system_prompt_path

NOW = dt.datetime(2026, 11, 1, 10, 15)


def test_default_path_is_the_contract_file_and_renders_every_placeholder():
    p = system_prompt_path()
    assert p.exists() and p.name == "system_prompt.md"
    text = render_system_prompt(NOW)
    assert "{" not in text or all("{" + k + "}" not in text for k in PLACEHOLDERS)
    assert placeholder_values(NOW) == {"weekday": "Sunday", "today_long": "November 1, 2026", "today": "2026-11-01", "now": "10:15"}


def test_env_override_and_subset_of_placeholders(tmp_path, monkeypatch):
    p = tmp_path / "prompt.md"
    p.write_text("Today is {today}. No other placeholders.\n")
    monkeypatch.setenv("R3_SYSTEM_PROMPT_PATH", str(p))
    assert system_prompt_path() == p
    assert render_system_prompt(NOW) == "Today is 2026-11-01. No other placeholders.\n"
