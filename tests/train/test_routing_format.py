"""Task 6: labeled tickets -> prompt -> label examples (<= 8 generated tokens) for routing.yaml."""
import json
from pathlib import Path

from train.loader import EMPTY_THINK, check_dataset, read_jsonl, render_and_mask
from train.routing_format import LABELS, ROUTING_SYSTEM_PROMPT, format_tickets, write_routing_dataset

REPO = Path(__file__).resolve().parents[2]
TICKETS = [
    {"ticket_id": "T-1", "text": "my debit card never arrived and i think it was taken from the mailbox", "label": "card_lost_stolen"},
    {"ticket_id": "T-2", "text": "declined at the grocery store twice today, card is in my hand", "label": "card_declined"},
    {"ticket_id": "T-3", "text": "there is a charge from a merchant i have never heard of", "label": "unauthorized_transaction"},
    {"ticket_id": "T-4", "text": "overdraft fee seems unfair, i was only short for an hour", "label": "fee_dispute"},
    {"ticket_id": "T-5", "text": "locked out of the app after the update", "label": "account_access"},
    {"ticket_id": "T-6", "text": "need last year's statements for my taxes", "label": "statement_document"},
    {"ticket_id": "T-7", "text": "moved last month, please update my address", "label": "profile_change"},
    {"ticket_id": "T-8", "text": "what is my mortgage payoff amount", "label": "loan_mortgage"},
    {"ticket_id": "T-9", "text": "need to reschedule thursday's notary appointment", "label": "appointment"},
    {"ticket_id": "T-10", "text": "wire sent monday has not arrived", "label": "transfer_payment"},
    {"ticket_id": "T-11", "text": "waited an hour at the branch and nobody helped", "label": "complaint"},
    {"ticket_id": "T-12", "text": "do you sell euros", "label": "other"},
    {"ticket_id": "T-13", "text": "some text", "label": "refund"},  # not in the taxonomy
    {"ticket_id": "T-14", "text": "   ", "label": "other"},  # empty
]


def test_labels_are_the_contract_taxonomy_from_the_create_ticket_schema():
    enum = json.loads((REPO / "schemas" / "tools" / "create_ticket.json").read_text())["function"]["parameters"]["properties"]["category"]["enum"]
    assert LABELS == enum and len(LABELS) == 12
    for label in LABELS:
        assert label in ROUTING_SYSTEM_PROMPT


def test_format_tickets_makes_prompt_to_label_examples_and_drops_invalid_ones():
    examples, dropped = format_tickets(TICKETS)
    assert len(examples) == 12 and [d["ticket_id"] for d in dropped] == ["T-13", "T-14"]
    ex = examples[0]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == ROUTING_SYSTEM_PROMPT
    assert ex["messages"][2]["content"] == "card_lost_stolen" and "tools" not in ex
    assert ex["meta"] == {"ticket_id": "T-1", "label": "card_lost_stolen"}


def test_every_label_fits_the_eight_token_generation_budget(tokenizer):
    for label in LABELS:
        assert len(tokenizer.encode(label)) + 1 <= 8, label  # label + <|im_end|>


def test_examples_render_with_loss_on_the_label_only(tokenizer):
    examples, _ = format_tickets(TICKETS[:3])
    me = render_and_mask(tokenizer, examples[2], max_seq_length=1024)
    assert me.target_text == EMPTY_THINK + "unauthorized_transaction<|im_end|>\n"
    assert all(me.checks.values()), me.checks
    rep = check_dataset(tokenizer, examples, 1024, require_tools=False)
    assert rep.ok and rep.stats["max_tokens"] <= 1024


def test_write_routing_dataset_splits_by_label_and_writes_a_manifest(tmp_path):
    tickets = [dict(t, ticket_id=f"{t['ticket_id']}-{i}") for i in range(5) for t in TICKETS[:12]]  # 60 valid
    manifest = write_routing_dataset(tickets, tmp_path / "routing", valid_fraction=0.2, seed=0)
    m = json.loads(manifest.read_text())
    assert m["counts"] == {"train": 48, "valid": 12} and m["format"] == "mlx_lm.messages" and m["dropped"] == 0
    assert m["label_counts"]["train"]["other"] == 4 and m["label_counts"]["valid"]["other"] == 1
    train = read_jsonl(tmp_path / "routing" / "train.jsonl")
    assert len(train) == 48 and all(len(ex["messages"]) == 3 for ex in train)
    assert set(m["sha256"]) == {"train.jsonl", "valid.jsonl"}
