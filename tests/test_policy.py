import math

import pytest
from pydantic import TypeAdapter, ValidationError

from classification import MODEL_PATH, Classification, Classifier
from context_api import RUNBOOKS, STATUS_RECORDS
from context_tools import Found
from policy import Decision, after_lookup, before_lookup


DELIVERY = Classification(category="delivery", raw_category="delivery", confidence_score=0.99,
                          reason="model_prediction")
STATUS = Found(tool="status", attempts=1, evidence=STATUS_RECORDS["A17", "delivery"])
TEXT = "Where is my card?"


@pytest.mark.parametrize(("report", "category", "urgency"), [
    ("There's an extra charge on my statement.", "billing_dispute", "normal"),
    ("I didn't authorize this purchase.", "security", "urgent"),
    ("My account was hacked.", "security", "urgent"),
    ("All my transaction history has disappeared.", "data_loss", "urgent"),
    ("My records are now missing.", "data_loss", "urgent"),
    ("Someone is using my card.", "security", "urgent"),
    ("My card hasn't arrived and someone used it.", "security", "urgent"),
    ("There was no duplicate charge but my data is gone.", "data_loss", "urgent"),
    ("The prior billing dispute is resolved and someone else used my card.", "security", "urgent"),
    ("Last year my card was stolen and the incident is still unresolved.", "security", "urgent"),
])
def test_active_risk_overrides_confident_delivery(report, category, urgency):
    decision = before_lookup(TEXT + " " + report, DELIVERY, 0.48)
    assert decision.category == category and decision.urgency == urgency
    assert decision.reason_codes == [category] and decision.confidence_score is None
    assert decision.recommended_action == "request_review" and decision.needs_human_review
    assert decision.draft is None and decision.evidence_ids == [] and decision.internal_next_step


@pytest.mark.parametrize("background", [
    "I wasn't charged twice.", "There is no unauthorized activity.", "My card was not stolen.",
    "My previous billing dispute is now resolved.", "The previously resolved security incident.",
    "I don't see any unauthorized charges.",
])
def test_inactive_risk_mentions_allow_lookup(background):
    assert before_lookup(background + " " + TEXT, DELIVERY, 0.48) is None


@pytest.mark.parametrize("raw", ["billing_dispute", "security"])
def test_risk_prediction_precedes_abstention(raw):
    prediction = Classification(category="unknown", raw_category=raw, confidence_score=0.4,
                                reason="below_threshold")
    decision = before_lookup(TEXT, prediction, 0.48)
    assert decision.category == raw and decision.reason_codes == [raw]


def test_multiple_risks_retain_all_reasons_and_highest_urgency():
    decision = before_lookup("I was charged twice and all my data is gone.", DELIVERY, 0.48)
    assert decision.reason_codes == ["data_loss", "billing_dispute"] and decision.urgency == "urgent"


def test_frozen_threshold_is_inclusive():
    threshold = Classifier.load(MODEL_PATH).threshold
    just_below = math.nextafter(threshold, 0.0)
    assert before_lookup(TEXT, DELIVERY.model_copy(update={"confidence_score": just_below}),
                         threshold).reason_codes == ["below_threshold"]
    assert before_lookup(TEXT, DELIVERY.model_copy(update={"confidence_score": threshold}), threshold) is None


@pytest.mark.parametrize(("reason", "expected"), [
    ("empty_features", "unknown_intent"), ("model_unavailable", "model_unavailable"),
    ("below_threshold", "below_threshold"),
])
def test_uncertainty_requests_manual_classification(reason, expected):
    decision = before_lookup(TEXT, Classification(category="unknown", reason=reason), 0.48)
    assert decision.reason_codes == [expected] and decision.internal_next_step
    assert decision.draft is None


def test_only_matching_approved_evidence_can_supply_draft_text():
    assert after_lookup(DELIVERY, TEXT, "B42", STATUS).reason_codes == ["account_mismatch"]
    assert after_lookup(DELIVERY, TEXT, "A17", None).reason_codes == ["missing_evidence"]
    unapproved = STATUS.evidence.model_copy(update={"status": "guaranteed tomorrow"})
    altered = STATUS.model_copy(update={"evidence": unapproved})
    assert after_lookup(DELIVERY, TEXT, "A17", altered).reason_codes == ["unsupported_status"]
    book = RUNBOOKS[0].model_copy(update={"approved_next_step": "Send your PIN."})
    lookup = Found(tool="runbook", attempts=1, evidence=book)
    assert after_lookup(DELIVERY, "How long is delivery?", None, lookup).reason_codes == ["unapproved_runbook"]
    assert after_lookup(DELIVERY, TEXT, "A17", lookup).reason_codes == ["conflicting_evidence"]


@pytest.mark.parametrize(("reply", "invalid"), [
    (True, {"evidence_ids": []}), (True, {"draft": " "}), (True, {"needs_human_review": True}),
    (True, {"recommended_action": "send_reply"}), (True, {"confidence_score": 1.1}),
    (False, {"reason_codes": []}), (False, {"internal_next_step": " "}),
    (False, {"draft": "Your card has arrived."}),
])
def test_decision_schema_rejects_inconsistent_results(reply, invalid):
    decision = after_lookup(DELIVERY, TEXT, "A17", STATUS if reply else None)
    with pytest.raises(ValidationError):
        TypeAdapter(Decision).validate_python(decision.model_dump() | invalid)
