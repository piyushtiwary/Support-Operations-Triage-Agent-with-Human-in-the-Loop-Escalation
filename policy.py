"""Deterministic review rules and evidence-bound reply templates."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from classification import Classification, ModelCategory, normalize
from context_api import RUNBOOKS, Nonblank, RunbookEvidence, StatusEvidence
from context_tools import Found, Lookup, Unavailable, general_timing


class DecisionFields(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    category: ModelCategory | Literal["technical_issue", "unknown", "data_loss"]
    urgency: Literal["normal", "urgent"]
    confidence_score: float | None = Field(ge=0, le=1)


class PreparedReply(DecisionFields):
    recommended_action: Literal["prepare_reply"]
    needs_human_review: Literal[False]
    reason_codes: list[Nonblank]
    evidence_ids: list[Nonblank] = Field(min_length=1)
    draft: Nonblank
    internal_next_step: None


class ReviewDecision(DecisionFields):
    recommended_action: Literal["request_review"]
    needs_human_review: Literal[True]
    reason_codes: list[Nonblank] = Field(min_length=1)
    evidence_ids: list[Nonblank]
    draft: None
    internal_next_step: Nonblank


Decision = Annotated[PreparedReply | ReviewDecision, Field(discriminator="recommended_action")]

# Each match is evaluated in its own clause. Negative assertions such as
# "I didn't authorize this payment" are themselves positive risk signals.
PAYMENT = r"(?:payments?|transactions?|purchases?|charges?|withdrawals?|transfers?)"
UNRECOGNIZED = r"(?:didn t|did not|don t|do not|never)"
RISK_PATTERNS = {
    "security": re.compile(
        rf"\b(?:unauthori[sz]ed (?:use|activity|{PAYMENT})|(?:unrecogni[sz]ed|fraudulent) {PAYMENT}"
        rf"|{PAYMENT} (?:that )?i {UNRECOGNIZED} (?:make|recogni[sz]e|authori[sz]e|approve)"
        rf"|i {UNRECOGNIZED} (?:make|recogni[sz]e|authori[sz]e|approve) (?:this |that |the |a |these )?{PAYMENT}"
        r"|someone (?:else )?(?:used|has used|is using) (?:it|my (?:card|account))"
        r"|someone (?:else )?(?:has )?(?:accessed|logged into) my account"
        r"|(?:card|account|credentials?) (?:has been |was |is |got |may be |might be )?(?:now )?(?:lost|stolen|compromised|hacked)"
        r"|(?:lost|stolen|compromised|hacked) (?:my |debit |credit )*(?:card|account)"
        r"|(?:stole|hacked) my (?:card|account)|security (?:incident|breach)|identity theft|fraud"
        r"|(?:card|account) (?:was |has been )?used without my (?:permission|consent))\b"
    ),
    "data_loss": re.compile(
        r"\b(?:data loss|(?:lost|missing|deleted|erased) (?:all |my |account |transaction )*(?:data|history|records)"
        r"|(?:data|records|transactions|transaction history) (?:has |have |was |is |are |were |been |went |now )*"
        r"(?:missing|lost|gone|deleted|erased|disappeared))\b"
    ),
    "billing_dispute": re.compile(
        rf"\b(?:charged (?:me )?twice|double charged|(?:duplicate|extra|incorrect|unexpected|double) {PAYMENT}"
        rf"|dispute (?:this |that |a |the |my )?{PAYMENT}|(?:billing|payment|charge) dispute"
        rf"|billed twice|overcharged)\b"
    ),
}
CLAUSE_BREAK = re.compile(r"[.!?;,:\n]+|\b(?:but|however|except)\b", re.I)
DENIAL = re.compile(
    r"\b(?:no(?: evidence of)?|not|never|without|wasn t|isn t|weren t|don t|didn t|haven t seen|ruled out)"
    r"(?: any| a| an| been| being| getting| see| seeing| have| reporting| suspect| a case of)* $"
)
RESOLVED_AFTER = re.compile(
    r"^(?: last (?:year|month|week))?(?: (?:and|it|that|which|is|was|has|been|now|already|since|fully))* resolved\b"
)


def active_risk(text: str) -> list[str]:
    found = set()
    for fragment in CLAUSE_BREAK.split(text):
        clause = normalize(fragment)
        for risk, pattern in RISK_PATTERNS.items():
            for match in pattern.finditer(clause):
                before, after = clause[:match.start()], clause[match.end():]
                if DENIAL.search(before) or re.match(r" (?:did not|didn t|never) (?:occur|happen)\b", after):
                    continue
                # A resolved previous incident is background. An unresolved
                # incident is actionable even if it happened in the past.
                if RESOLVED_AFTER.match(after) or re.search(r"\b(?:previously|already) resolved $", before):
                    continue
                found.add(risk)
    return [risk for risk in RISK_PATTERNS if risk in found]


RISK_STEPS = {
    "security": "Send to the security team to verify the reported incident and account activity.",
    "data_loss": "Ask operations to verify the affected records and investigate the reported data loss.",
    "billing_dispute": "Ask billing to verify the disputed charge against the account ledger.",
}


def review(prediction: Classification, reasons: list[str], step: str, *, category=None) -> ReviewDecision:
    category = category or prediction.category
    return ReviewDecision(
        category=category, urgency="urgent" if category in {"security", "data_loss"} else "normal",
        confidence_score=prediction.confidence_score if category == prediction.category else None,
        recommended_action="request_review", needs_human_review=True, reason_codes=reasons,
        evidence_ids=[], draft=None, internal_next_step=step,
    )


def before_lookup(text: str, prediction: Classification, threshold: float | None) -> ReviewDecision | None:
    risks = active_risk(text)
    for label in (prediction.category, prediction.raw_category):
        if label in RISK_STEPS and label not in risks:
            risks.append(label)
    if risks:
        ordered = [risk for risk in RISK_STEPS if risk in risks]
        return review(prediction, ordered, " ".join(RISK_STEPS[r] for r in ordered), category=ordered[0])
    if prediction.reason == "model_unavailable":
        return review(prediction, ["model_unavailable"], "Classify the ticket manually.")
    if prediction.reason == "below_threshold" or (
        prediction.confidence_score is not None and threshold is not None
        and prediction.confidence_score < threshold
    ):
        return review(prediction, ["below_threshold"], "Read the request and confirm the intended category.")
    if prediction.category == "unknown":
        return review(prediction, ["unknown_intent"], "Clarify the request and choose a supported category.")
    if prediction.confidence_score is None and prediction.reason != "known_bug":
        return review(prediction, ["missing_score"], "Verify the classification before preparing a reply.")
    return None


STATUS_SENTENCES = {
    ("delivery", "dispatched"): "Your card has been dispatched.",
    ("delivery", "processing"): "Your card is being processed.",
    ("transfer", "pending"): "Your transfer is pending.",
}
# Frozen, revisioned runbook copy supplies the approved next step for status
# replies as well as the allowlist for runbook responses from the context tool.
APPROVED_BOOKS = {book.topic: book for book in RUNBOOKS}
VERIFY_STEPS = {
    "delivery": "Verify the card dispatch status manually before replying.",
    "transfer": "Verify the transfer status manually before replying.",
    "technical_issue": "Verify the approved workaround in the mobile app runbook.",
}


def after_lookup(prediction: Classification, text: str, account_ref: str | None, lookup: Lookup | None) -> Decision:
    category = prediction.category
    expects_book = category == "technical_issue" or general_timing(text)
    step = VERIFY_STEPS[category]
    if expects_book and category != "technical_issue":
        step = f"Verify the approved {category} timing guidance in the runbook before replying."
    if lookup is None:
        return review(prediction, ["missing_evidence"], step)
    if not isinstance(lookup, Found):
        reason = f"{lookup.tool}_unavailable" if isinstance(lookup, Unavailable) else lookup.reason
        return review(prediction, [reason], step)

    evidence = lookup.evidence
    topic = prediction.rule_id if category == "technical_issue" else category
    book = APPROVED_BOOKS.get(topic)
    if evidence.topic != topic or book is None:
        return review(prediction, ["topic_mismatch"], step)
    if isinstance(evidence, RunbookEvidence) and expects_book and lookup.tool == "runbook":
        if evidence != book:
            return review(prediction, ["unapproved_runbook"], step)
        sentence = evidence.guidance
    elif isinstance(evidence, StatusEvidence) and not expects_book and lookup.tool == "status":
        if account_ref != evidence.account_ref:
            return review(prediction, ["account_mismatch"], step)
        sentence = STATUS_SENTENCES.get((category, evidence.status))
        if sentence is None:
            return review(prediction, ["unsupported_status"], step)
    else:
        return review(prediction, ["conflicting_evidence"], step)

    return PreparedReply(
        category=category, urgency="normal", confidence_score=prediction.confidence_score,
        recommended_action="prepare_reply", needs_human_review=False, reason_codes=["evidence_verified"],
        evidence_ids=[f"{evidence.source_id}@{evidence.revision}"],
        draft=f"{sentence} {book.approved_next_step}", internal_next_step=None,
    )
