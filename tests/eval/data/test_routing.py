"""Routing track: synthetic ticket parsing/splitting and the deterministic routing scorer."""

import json

import pytest

from data.routing.score import COSTLY_PAIRS, parse_label, score_predictions
from data.routing.tickets import (
    LABELS,
    assign_ids,
    dedupe,
    parse_batch,
    split_tickets,
    variation_specs,
)


def test_labels_come_from_the_tool_schema():
    assert len(LABELS) == 12 and "unauthorized_transaction" in LABELS and LABELS[-1] == "other"


def test_parse_batch_accepts_json_list_or_fenced_and_labels_by_construction():
    raw = '```json\n["my card got declined at the store twice", "ATM wont give me cash with my debit card"]\n```'
    out = parse_batch(
        raw,
        label="card_declined",
        spec={"length": "short", "tone": "annoyed", "noise": "clean", "channel": "chat"},
    )
    assert [t["label"] for t in out] == ["card_declined", "card_declined"]
    assert (
        out[0]["text"] == "my card got declined at the store twice" and out[0]["tone"] == "annoyed"
    )
    assert parse_batch("not json at all", label="other", spec={}) == []
    assert (
        parse_batch(json.dumps([{"text": "x"}, "", "   ", 42]), label="other", spec={}) == []
    )  # only non-empty strings


def test_dedupe_is_case_and_space_insensitive_and_keeps_first():
    a = [
        {"text": "Lost my card", "label": "card_lost_stolen"},
        {"text": "lost  my card ", "label": "card_lost_stolen"},
        {"text": "Lost my card!", "label": "card_lost_stolen"},
    ]
    assert [t["text"] for t in dedupe(a)] == ["Lost my card", "Lost my card!"]


def test_assign_ids_and_split_holds_out_100_stratified():
    tickets = [
        {"text": f"ticket {i} {label}", "label": label} for i in range(90) for label in LABELS
    ]  # 1080
    tickets = assign_ids(tickets)
    assert tickets[0]["ticket_id"] == "R-00001" and tickets[-1]["ticket_id"] == "R-01080"
    split = split_tickets(tickets, test_size=100, seed=0)
    counts = {s: sum(1 for t in split if t["split"] == s) for s in ("train", "test")}
    assert counts["test"] == 100 and counts["train"] == 980
    per_label = {
        label: sum(1 for t in split if t["split"] == "test" and t["label"] == label)
        for label in LABELS
    }
    assert min(per_label.values()) >= 7 and max(per_label.values()) <= 10
    assert split_tickets(tickets, test_size=100, seed=0) == split


def test_variation_specs_are_deterministic_and_cover_lengths_tones_noise():
    specs = variation_specs(n=24, seed=1)
    assert len(specs) == 24 and specs == variation_specs(n=24, seed=1)
    assert {s["length"] for s in specs} == {"short", "medium", "long"}
    assert {s["noise"] for s in specs} >= {"clean", "typos", "no punctuation or caps"}
    assert all({"length", "tone", "noise", "channel"} <= set(s) for s in specs)


def test_parse_label_validates_against_the_taxonomy():
    assert parse_label("fee_dispute") == "fee_dispute"
    assert parse_label("  Fee_Dispute\n") == "fee_dispute"
    assert parse_label("The category is fee_dispute.") == "fee_dispute"
    assert parse_label("refund") == "needs_review" and parse_label("") == "needs_review"
    assert parse_label("fee_dispute or complaint") == "needs_review"  # two labels -> invalid


def test_score_predictions_per_category_and_costly_confusions():
    gold = [
        "fee_dispute",
        "fee_dispute",
        "unauthorized_transaction",
        "complaint",
        "card_lost_stolen",
        "other",
    ]
    pred = [
        "fee_dispute",
        "unauthorized_transaction",
        "unauthorized_transaction",
        "other",
        "card_declined",
        "needs_review",
    ]
    s = score_predictions(gold, pred)
    assert s["coverage"] == pytest.approx(5 / 6)
    assert s["accuracy"] == pytest.approx(2 / 6)
    assert (
        s["per_category"]["fee_dispute"]["precision"] == 1.0
        and s["per_category"]["fee_dispute"]["recall"] == 0.5
    )
    assert s["per_category"]["unauthorized_transaction"]["precision"] == 0.5
    assert 0 < s["macro_f1"] < 1
    assert s["costly_confusions"] == {
        "fee_dispute->unauthorized_transaction": 1,
        "complaint->other": 1,
        "card_lost_stolen->card_declined": 1,
    }
    assert (
        s["weighted_error"] == 3 * 3 + 1
    )  # three costly confusions at weight 3, one needs_review at weight 1
    assert ("unauthorized_transaction", "fee_dispute") in COSTLY_PAIRS and (
        "fee_dispute",
        "unauthorized_transaction",
    ) in COSTLY_PAIRS
    assert s["confusion"]["fee_dispute"]["unauthorized_transaction"] == 1


def test_routing_prompt_matches_session_t_byte_for_byte():
    import subprocess

    from data.routing.prompt import ROUTING_SYSTEM_PROMPT

    try:
        src = subprocess.check_output(
            ["git", "show", "r3-training-scripts:train/routing_format.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pytest.skip("branch r3-training-scripts not available")
    ns: dict = {}
    body = src.split("ROUTING_SYSTEM_PROMPT = (")[1].split("\n)\n")[0]
    defs = src.split("DEFINITIONS = {")[1].split("\n}\n")[0]
    exec(  # noqa: S102
        "LABELS = "
        + repr(LABELS)
        + "\nDEFINITIONS = {"
        + defs
        + "\n}\nROUTING_SYSTEM_PROMPT = ("
        + body
        + "\n)",
        ns,
    )
    assert ns["ROUTING_SYSTEM_PROMPT"] == ROUTING_SYSTEM_PROMPT


def test_gen_prompt_and_parsers():
    from data.routing.generate import build_parser, gen_prompt

    p = gen_prompt(
        "fee_dispute",
        {"length": "short", "tone": "annoyed", "noise": "typos", "channel": "chat"},
        6,
    )
    assert "6" in p and "fee_dispute" in p and "typos" in p and "JSON list" in p
    a = build_parser().parse_args(["--out", "/tmp/t", "--per-label", "10"])
    assert a.per_label == 10 and a.test_size == 100
    from data.routing.baseline import build_parser as bp

    b = bp().parse_args(["--tickets", "t.jsonl", "--base-url", "u", "--model", "m", "--out", "o"])
    assert b.split == "test" and b.parallel == 8
