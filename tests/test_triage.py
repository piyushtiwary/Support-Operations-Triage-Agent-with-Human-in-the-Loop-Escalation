import asyncio
import sqlite3
from contextlib import closing

import httpx
import pytest
from pydantic import TypeAdapter

from classification import Classification, Classifier
from context_api import RUNBOOKS, STATUS_RECORDS
from context_tools import Found
from policy import Decision


pytestmark = pytest.mark.anyio
CARD = "Where is my card?"
MIXED = "Where is my card? I see a payment I didn't make."


async def submit(client, text, account="A17"):
    response = await client.post("/tickets", json={"external_id": "example", "text": text, "account_ref": account})
    assert response.status_code == 201
    return "/tickets/" + response.json()["id"]


async def triage_and_read(client, url):
    response = await client.post(url + "/triage")
    assert response.status_code == 200
    stored = (await client.get(url)).json()
    decision = TypeAdapter(Decision).validate_python(response.json())
    assert stored["decision"] == decision.model_dump()
    assert stored["state"] == ("pending_review" if decision.needs_human_review else "draft_ready")
    return decision, stored


@pytest.mark.parametrize(("text", "account", "evidence_id", "draft"), [
    (CARD, "A17", "card-A17@v1", "Your card has been dispatched. Check the dispatch date in card tracking."),
    (CARD, "B42", "card-B42@v1", "Your card is being processed. Check the dispatch date in card tracking."),
    ("My bank transfer is still pending", "A17", "transfer-A17@v1",
     "Your transfer is pending. Check the transfer details for its submission date."),
    ("How long does card delivery take?", None, "card-delivery@v1",
     RUNBOOKS[0].guidance + " " + RUNBOOKS[0].approved_next_step),
    ("My app freezes when I open card tracking.", None, "mobile-card-tracking@v1",
     RUNBOOKS[2].guidance + " Open card tracking in the web app."),
])
async def test_prepared_reply_cites_ticket_evidence(client, text, account, evidence_id, draft):
    url = await submit(client, text, account)
    decision, stored = await triage_and_read(client, url)
    assert decision.recommended_action == "prepare_reply" and decision.reason_codes == ["evidence_verified"]
    assert decision.draft == draft and decision.evidence_ids == [evidence_id]
    evidence = stored["lookup"]["evidence"]
    assert evidence_id == f"{evidence['source_id']}@{evidence['revision']}"
    if account:
        assert evidence["account_ref"] == account
    if decision.category == "technical_issue":
        assert decision.confidence_score is None


def unexpected_lookup(request):
    pytest.fail(f"Unexpected tool call to {request.url}")


@pytest.mark.parametrize(("text", "reason"), [
    (MIXED, "security"), ("My card hasn't arrived and someone used it", "security"),
    ("I was charged twice", "billing_dispute"), ("My data was deleted", "data_loss"),
    ("xqzv !!!", "unknown_intent"), ("Can I buy travel insurance?", "below_threshold"),
])
async def test_early_review_skips_tools(database, running_client, text, reason):
    async with running_client(database, context_transport=httpx.MockTransport(unexpected_lookup)) as client:
        url = await submit(client, text)
        decision, stored = await triage_and_read(client, url)
        assert decision.recommended_action == "request_review" and reason in decision.reason_codes
        assert decision.draft is None and decision.internal_next_step and decision.evidence_ids == []
        assert stored.get("lookup") is None
        if text == MIXED:
            assert decision.urgency == "urgent" and decision.category == "security"
            assert stored["classification"]["category"] == "delivery"


async def test_runtime_classifier_failure(database, running_client, monkeypatch):
    def fail(self, text):
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(Classifier, "predict", fail)
    async with running_client(database, context_transport=httpx.MockTransport(unexpected_lookup)) as client:
        decision, _ = await triage_and_read(client, await submit(client, CARD))
        assert decision.reason_codes == ["model_unavailable"] and decision.draft is None


@pytest.mark.parametrize(("account", "status", "reason", "attempts"), [
    (None, None, "missing_account_ref", 0), ("A17", 503, "status_unavailable", 2),
    ("A17", 404, "not_found", 1), ("A17", 200, "account_mismatch", 1),
])
async def test_context_failure_persists_manual_step(database, running_client, account, status, reason, attempts):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json=STATUS_RECORDS["B42", "delivery"].model_dump())

    async with running_client(database, context_transport=httpx.MockTransport(handler)) as client:
        url = await submit(client, CARD, account)
        decision, stored = await triage_and_read(client, url)
        assert decision.reason_codes == [reason] and decision.recommended_action == "request_review"
        assert decision.draft is None and decision.evidence_ids == []
        assert "dispatch status manually" in decision.internal_next_step
        assert stored["lookup"]["attempts"] == len(requests) == attempts
    async with running_client(database, context_transport=httpx.MockTransport(unexpected_lookup)) as client:
        assert (await triage_and_read(client, url)) == (decision, stored)


async def test_existing_step_three_evidence_is_reused(database, running_client):
    evidence = Found(tool="status", attempts=1, evidence=STATUS_RECORDS["A17", "delivery"])
    prediction = Classification(category="delivery", raw_category="delivery", confidence_score=0.8,
                                reason="model_prediction")
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""CREATE TABLE tickets (id TEXT PRIMARY KEY, external_id TEXT UNIQUE, text TEXT,
                   account_ref TEXT, state TEXT, classification TEXT, lookup TEXT)""")
        db.execute("INSERT INTO tickets VALUES ('old', 'old', ?, 'A17', 'classified', ?, ?)",
                   (CARD, prediction.model_dump_json(), evidence.model_dump_json()))
    async with running_client(database, context_transport=httpx.MockTransport(unexpected_lookup)) as client:
        decision, stored = await triage_and_read(client, "/tickets/old")
        assert decision.evidence_ids == ["card-A17@v1"] and stored["lookup"] == evidence.model_dump()


async def test_concurrent_triage_commits_one_consistent_result(database, running_client):
    # Make both requests read the undecided ticket, then return conflicting tool outcomes.
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request):
        if not entered.is_set():
            entered.set()
            await release.wait()
            return httpx.Response(404)
        return httpx.Response(200, json=STATUS_RECORDS["A17", "delivery"].model_dump())

    async with running_client(database, context_transport=httpx.MockTransport(handler)) as client:
        url = await submit(client, CARD)
        first = asyncio.create_task(client.post(url + "/triage"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            decision, stored = await triage_and_read(client, url)
        finally:
            release.set()
            other = await first
        assert decision.recommended_action == "prepare_reply" and stored["lookup"]["outcome"] == "found"
        assert other.json() == decision.model_dump() and (await client.get(url)).json() == stored
