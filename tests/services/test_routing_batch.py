"""Ticket routing batch job (CONTRACT.md §9): label generation is greedy, at most 8 tokens, and validated
against the taxonomy; anything invalid becomes needs_review. The prompt must be the one the routing adapter
was trained on (train/routing_format.py), so the job reuses that module's system prompt when it exists."""
import json
import pathlib

from services.routing.batch import LABELS, NEEDS_REVIEW, build_messages, label_tickets, validate_label

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_labels_come_from_the_create_ticket_schema():
    schema = json.loads((ROOT / "schemas/tools/create_ticket.json").read_text())
    assert LABELS == list(schema["function"]["parameters"]["properties"]["category"]["enum"])
    assert len(LABELS) == 12 and "other" in LABELS and NEEDS_REVIEW not in LABELS


def test_validate_label_accepts_exact_ids_and_tolerates_whitespace_case_and_trailing_punctuation():
    assert validate_label("card_lost_stolen") == "card_lost_stolen"
    assert validate_label("  Fee_Dispute.\n") == "fee_dispute"
    assert validate_label("<think>\n\n</think>\n\nappointment") == "appointment"


def test_validate_label_rejects_anything_outside_the_taxonomy():
    assert validate_label("fraud") == NEEDS_REVIEW
    assert validate_label("unauthorized_transaction or fee_dispute") == NEEDS_REVIEW
    assert validate_label("") == NEEDS_REVIEW


def test_messages_carry_the_taxonomy_and_the_ticket_text_only():
    msgs = build_messages("Someone charged my card at a store I have never been to")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "unauthorized_transaction" in msgs[0]["content"] and "Answer with the category id only" in msgs[0]["content"]
    assert msgs[1]["content"] == "Someone charged my card at a store I have never been to"


def test_label_tickets_maps_generator_output_to_validated_rows_with_source():
    tickets = [{"ticket_id": "T-1", "text": "lost my debit card"}, {"ticket_id": "T-2", "text": "overdraft fee unfair"}, {"ticket_id": "T-3", "text": "???"}]
    seen = []

    def fake_generate(prompts_messages):
        seen.append(len(prompts_messages))
        return ["card_lost_stolen", "fee_dispute.", "I think this is spam"]

    rows = label_tickets(tickets, fake_generate, source="r3-route-test")
    assert seen == [3]
    assert rows == [
        {"ticket_id": "T-1", "label": "card_lost_stolen", "raw": "card_lost_stolen", "label_source": "r3-route-test"},
        {"ticket_id": "T-2", "label": "fee_dispute", "raw": "fee_dispute.", "label_source": "r3-route-test"},
        {"ticket_id": "T-3", "label": NEEDS_REVIEW, "raw": "I think this is spam", "label_source": "r3-route-test"},
    ]


def test_routing_prompt_is_the_one_the_adapter_was_trained_with():
    """train/routing_format.py (T) and services/routing/batch.py (S) must stay byte-identical: the adapter learned this exact
    system turn. The sha is the system_prompt_sha256 recorded in $R3_ARTIFACTS/datasets/routing-v1/manifest.json."""
    import hashlib
    from services.routing.batch import ROUTING_SYSTEM_PROMPT
    assert hashlib.sha256(ROUTING_SYSTEM_PROMPT.encode()).hexdigest() == "be3b34cafbef512daa4ea4d1dba2c9ce30ca862f703d4d908f215fe029b4d65f"
