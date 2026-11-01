"""Routing scoring (CONTRACT §9): accuracy, macro F1 over the 12 categories, coverage (valid labels), and the
confusion count weighted 3x for the costly pairs; needs_review is never a class, only a miss."""
from services.routing.score import COSTLY, score

GOLD = ["card_lost_stolen", "card_declined", "unauthorized_transaction", "fee_dispute", "complaint", "other", "appointment", "appointment"]


def test_perfect_predictions():
    s = score(GOLD, list(GOLD))
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0 and s["coverage"] == 1.0 and s["weighted_error"] == 0


def test_costly_confusions_weigh_three_and_needs_review_is_a_miss_not_a_class():
    pred = ["card_declined",            # lost/stolen -> declined: costly (3)
            "card_declined",
            "fee_dispute",              # fraud -> fee dispute: costly (3)
            "fee_dispute",
            "other",                    # complaint -> other: costly (3)
            "complaint",                # other -> complaint: ordinary (1)
            "needs_review",             # invalid: a miss (1), not a class
            "appointment"]
    s = score(GOLD, pred)
    assert s["n"] == 8 and s["correct"] == 3 and s["accuracy"] == 3 / 8
    assert s["coverage"] == 7 / 8
    assert s["weighted_error"] == 3 + 3 + 3 + 1 + 1
    assert s["costly"] == {"card_lost_stolen->card_declined": 1, "unauthorized_transaction->fee_dispute": 1, "complaint->other": 1}
    assert set(s["per_label"]) == set(COSTLY_LABELS := {"card_lost_stolen", "card_declined", "unauthorized_transaction", "fee_dispute", "complaint", "other", "appointment"}) | {"account_access", "statement_document", "profile_change", "loan_mortgage", "transfer_payment"}
    assert 0 < s["macro_f1"] < 1


def test_costly_pairs_match_the_contract():
    assert ("unauthorized_transaction", "fee_dispute") in COSTLY and ("fee_dispute", "unauthorized_transaction") in COSTLY
    assert ("card_lost_stolen", "card_declined") in COSTLY and ("card_declined", "card_lost_stolen") in COSTLY
    assert ("complaint", "other") in COSTLY and ("other", "complaint") not in COSTLY
