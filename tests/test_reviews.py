import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

import httpx
import pytest


pytestmark = pytest.mark.anyio
MIXED = {"external_id": "mixed", "text": "Where is my card? I see a payment I didn't make."}
ROUTINE = {"external_id": "delivery", "text": "Where is my card?", "account_ref": "A17"}
CONFIRM = {"explanation": "Checked the report and confirmed the security handoff."}


async def submit(client, payload):
    response = await client.post("/tickets", json=payload)
    assert response.status_code == 201
    return response.json()["id"]


async def read_queue(client, state="pending"):
    response = await client.get("/reviews", params={"state": state})
    assert response.status_code == 200
    return response.json()


async def test_confirm_and_keep_routine_draft(client):
    mixed = await submit(client, MIXED)
    routine = await submit(client, ROUTINE)
    assert await read_queue(client) == []
    decisions = await asyncio.gather(*(client.post(f"/tickets/{mixed}/triage") for _ in range(4)))
    original = decisions[0].json()
    assert all(r.status_code == 200 and r.json() == original for r in decisions)
    draft = (await client.post(f"/tickets/{routine}/triage")).json()
    pending, = await read_queue(client)
    assert pending["id"] == mixed and pending["text"] == MIXED["text"]
    assert pending["decision"] == original
    assert original["reason_codes"] == ["security"] and original["internal_next_step"]
    assert original["evidence_ids"] == []

    result = await client.post(f"/reviews/{mixed}/resolve", json=CONFIRM)
    assert result.status_code == 200
    resolution = result.json()
    assert resolution == {
        **CONFIRM, "reviewer_id": "demo-reviewer", "outcome": "confirmed",
        "resolved_at": resolution["resolved_at"],
        **{k: original[k] for k in ("category", "urgency", "internal_next_step")},
    }
    assert datetime.fromisoformat(resolution["resolved_at"]).utcoffset() == timedelta(0)
    assert await read_queue(client) == []
    resolved, = await read_queue(client, "resolved")
    assert resolved["state"] == "reviewed" and resolved["resolution"] == resolution
    assert resolved["decision"] == original
    assert (await client.get(f"/tickets/{routine}")).json()["decision"] == draft
    assert draft["draft"].startswith("Your card has been dispatched.")
    for id in (routine, "absent"):
        assert (await client.post(f"/reviews/{id}/resolve", json=CONFIRM)).status_code == 404


async def test_correction_survives_restart_and_replays(database, running_client):
    correction = {
        "reviewer_id": "operator-2", "explanation": "Customer confirmed a duplicate charge.",
        "category": "billing_dispute", "urgency": "normal",
        "internal_next_step": "Ask billing to reconcile the two entries.",
    }
    async with running_client(database) as client:
        id = await submit(client, MIXED)
        original = (await client.post(f"/tickets/{id}/triage")).json()
        url = f"/reviews/{id}/resolve"
        result = await client.post(url, json=correction)
        assert result.status_code == 200
        resolution = result.json()
        assert resolution["outcome"] == "corrected"
        assert all(resolution[k] == v for k, v in correction.items())
    async with running_client(database) as client:
        assert (await client.post(url, json=correction)).json() == resolution
        assert (await client.post(url, json=correction | {"urgency": "urgent"})).status_code == 409
        assert (await client.post(f"/tickets/{id}/triage")).json() == original
        ticket = (await client.get(f"/tickets/{id}")).json()
        assert ticket["state"] == "reviewed" and ticket["resolution"] == resolution
        assert ticket["decision"] == original
        assert (await client.post("/tickets", json=MIXED)).json() == ticket
        assert await read_queue(client) == []


@pytest.mark.parametrize("invalid", [
    {"explanation": " \t "}, {"reviewer_id": ""}, {"urgency": "high"}, {"category": "other"},
    {"internal_next_step": " "}, {"outcome": "confirmed"},
])
async def test_invalid_resolution_leaves_review_pending(client, invalid):
    id = await submit(client, MIXED)
    await client.post(f"/tickets/{id}/triage")
    before = await read_queue(client)
    assert (await client.post(f"/reviews/{id}/resolve", json=CONFIRM | invalid)).status_code == 422
    assert len(before) == 1 and await read_queue(client) == before


async def test_competing_reviewers_accept_one_resolution(client):
    id = await submit(client, MIXED)
    original = (await client.post(f"/tickets/{id}/triage")).json()
    url = f"/reviews/{id}/resolve"
    submissions = [CONFIRM | {"reviewer_id": reviewer} for reviewer in ("first", "second")]
    responses = await asyncio.gather(*(client.post(url, json=s) for s in submissions))
    assert sorted(r.status_code for r in responses) == [200, 409]
    winner = next(r.json() for r in responses if r.status_code == 200)
    stored, = await read_queue(client, "resolved")
    assert stored["resolution"] == winner and stored["decision"] == original
    assert await read_queue(client) == []


async def test_failed_enqueue_rolls_back_and_can_retry(database, running_client):
    async with running_client(database, raise_app_exceptions=False) as client:
        id = await submit(client, MIXED)
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("""CREATE TRIGGER fail_enqueue BEFORE INSERT ON reviews
                          BEGIN SELECT RAISE(ABORT, 'injected enqueue failure'); END""")
        assert (await client.post(f"/tickets/{id}/triage")).status_code == 500
        ticket = (await client.get(f"/tickets/{id}")).json()
        assert ticket["state"] == "received" and "decision" not in ticket and "classification" not in ticket
        assert await read_queue(client) == []
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("DROP TRIGGER fail_enqueue")
        retry = await client.post(f"/tickets/{id}/triage")
        pending, = await read_queue(client)
        assert retry.status_code == 200 and retry.json() == pending["decision"]
        assert pending["state"] == "pending_review"


async def test_backfill_step_four_and_show_available_evidence(database, running_client):
    record = {"source_id": "card-A17", "revision": "v2", "topic": "delivery",
              "account_ref": "A17", "status": "awaiting_inspection"}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=record))
    async with running_client(database, context_transport=transport) as client:
        id = await submit(client, ROUTINE)
        original = (await client.post(f"/tickets/{id}/triage")).json()
        assert original["reason_codes"] == ["unsupported_status"]
    # Recreate the Step 4 schema with a persisted escalation and no queue.
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("DROP TABLE reviews")
    for _ in range(2):
        async with running_client(database) as client:
            pending, = await read_queue(client)
            assert pending["id"] == id and pending["decision"] == original
            assert pending["lookup"]["evidence"] == record
    async with running_client(database) as client:
        assert (await client.post(f"/reviews/{id}/resolve", json=CONFIRM)).status_code == 200
        assert await read_queue(client) == []
