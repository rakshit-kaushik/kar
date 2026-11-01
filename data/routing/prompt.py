"""The routing system prompt, byte-identical to train.routing_format.ROUTING_SYSTEM_PROMPT (session T),
so the training data, S's batch job and the zero-shot baselines share one prompt. The test suite
compares this text with T's module on branch r3-training-scripts."""

from __future__ import annotations

from data.routing.tickets import LABELS

DEFINITIONS = {
    "card_lost_stolen": "card missing or taken; needs block or replacement (lost, stolen, never arrived, found it, replacement)",
    "card_declined": "card refused at a merchant or ATM while the card is in hand (declined, ATM, abroad, chip, online blocked)",
    "unauthorized_transaction": "charges the customer did not make; fraud (unrecognized charge, someone used my card, drained)",
    "fee_dispute": "customer disputes a fee the bank charged (overdraft, monthly, ATM, late, wire fee)",
    "account_access": "cannot log in or use online or mobile banking (locked out, password, two-factor, new phone)",
    "statement_document": "wants a statement, tax form or letter (statements, 1099, proof of funds, interest statement, void check)",
    "profile_change": "address, phone, name or email update; add a joint owner",
    "loan_mortgage": "questions on existing or new loans (payoff, refinance, car loan, HELOC, pre-approval)",
    "appointment": "wants to book, move or cancel a branch visit, or asks branch hours for one",
    "transfer_payment": "wires, transfers or bill pay not going through (pending, wrong payee, Zelle failed, recurring stopped)",
    "complaint": "dissatisfaction with service and no transactional ask (rude, waited, no call back, closed early, app crashes)",
    "other": "none of the above, or unreadable (currency order, notary hours, check cashing policy, empty, spam)",
}
assert set(DEFINITIONS) == set(LABELS)

ROUTING_SYSTEM_PROMPT = (
    "Classify the customer's support ticket into exactly one category. Answer with the category id only, nothing else.\n"
    "Categories:\n" + "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in LABELS)
)
