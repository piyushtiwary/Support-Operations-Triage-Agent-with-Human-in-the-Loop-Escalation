import asyncio
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from context_api import STATUS_RECORDS
from classification import Classification
from context_tools import lookup_context


pytestmark = pytest.mark.anyio
CARD = "Where is my card?"
MIXED = "Where is my card? I see a payment I didn't make."
LOCAL_USAGE = {"source": "local", "provider_calls": 0, "input_tokens": 0, "output_tokens": 0,
               "estimated_provider_cost_usd": 0}


async def submit(client, text=CARD, account="A17"):
    response = await client.post("/tickets", json={"external_id": "case", "text": text, "account_ref": account})
    assert response.status_code == 201
    return "/tickets/" + response.json()["id"]


async def history(client, url):
    response = await client.get(url + "/runs")
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize(("scenario", "text", "account", "codes", "reason", "statuses"), [
    ("routine", CARD, "A17", ["evidence_verified"], None, [200]),
    ("risk", MIXED, "A17", ["security"], None, []),
    ("recovered", CARD, "A17", ["evidence_verified"], None, [503, 200]),
    ("outage", CARD, "A17", ["status_unavailable"], "connection_error", [None]),
    ("missing", CARD, None, ["missing_account_ref"], "missing_account_ref", []),
    ("exception", CARD, "A17", ["tool_error"], "tool_error", [None]),
])
async def test_explain_decision_from_audit(database, running_client, scenario, text, account, codes, reason, statuses):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if scenario == "outage":
            raise httpx.ConnectError(text, request=request)
        if scenario == "exception":
            raise RuntimeError(text)
        if scenario == "recovered" and len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=STATUS_RECORDS["A17", "delivery"].model_dump())

    async with running_client(database, context_transport=httpx.MockTransport(handler)) as client:
        url = await submit(client, text, account)
        assert await history(client, url) == []
        assert (await client.get("/tickets/absent/runs")).status_code == 404
        assert (await client.post(url + "/triage")).status_code == 200
        run, = await history(client, url)
        assert run["status"] == "completed" and run["error_code"] is None
        assert run["ticket_id"] == url.split("/")[-1] and run["retry_of_run_id"] is None
        assert run["classifier_version"].startswith("sha256:") and run["ruleset_version"]
        assert run["elapsed_ms"] >= 0
        assert all(datetime.fromisoformat(run[k]).tzinfo for k in ("started_at", "finished_at"))
        assert run["usage"] == LOCAL_USAGE and text not in str(run)
        assert run["reviewer_event"] is None
        decision = run["decision"]
        assert decision["reason_codes"] == codes
        attempts = run["attempts"]
        assert [a["http_status"] for a in attempts] == statuses
        assert [a["number"] for a in attempts] == list(range(1, len(calls) + 1))
        assert all(a["tool"] == "status" and 0 <= a["elapsed_ms"] <= run["elapsed_ms"] for a in attempts)
        if scenario == "risk":
            assert run["classification"]["category"] == "delivery" and run["lookup"] is None
            assert decision["urgency"] == "urgent" and decision["category"] == "security"
        else:
            assert run["lookup"]["reason"] == reason and run["lookup"]["attempts"] == len(calls)
        if codes == ["evidence_verified"]:
            assert decision["recommended_action"] == "prepare_reply" and "dispatched" in decision["draft"]
            assert attempts[-1]["outcome"] == "found"
            assert decision["evidence_ids"] == attempts[-1]["evidence_ids"] == ["card-A17@v1"]
        else:
            assert decision["recommended_action"] == "request_review" and decision["draft"] is None
            assert decision["internal_next_step"]
        if scenario == "recovered":
            assert attempts[0]["outcome"] == "unavailable" and attempts[0]["reason"] == "service_unavailable"
    async with running_client(database) as client:
        assert (await client.post(url + "/triage")).json() == decision
        assert await history(client, url) == [run]


async def test_reviewer_changes_join_original_run(database, running_client):
    async with running_client(database) as client:
        url = await submit(client, MIXED)
        await client.post(url + "/triage")
        original, = await history(client, url)
        payload = {"category": "billing_dispute", "urgency": "normal", "reviewer_id": "operator-2",
                   "internal_next_step": "Reconcile both charges.", "explanation": MIXED}
        for _ in range(2):
            resolved = await client.post("/reviews/" + original["ticket_id"] + "/resolve", json=payload)
            assert resolved.status_code == 200
    async with running_client(database) as client:
        reviewed, = await history(client, url)
        event = reviewed["reviewer_event"]
        assert event == {"reviewer_id": "operator-2", "outcome": "corrected", "resolved_at": resolved.json()["resolved_at"],
                         "category": "billing_dispute", "urgency": "normal",
                         "changed_fields": ["category", "urgency", "internal_next_step"]}
        assert reviewed == original | {"reviewer_event": event} and MIXED not in str(reviewed)


async def test_crash_during_lookup_and_linked_retry(database, running_client):
    # Stop a real worker inside its GET. No lifespan cleanup or exception handler runs.
    script = """
import asyncio, os, sys
from pathlib import Path
import httpx
from main import create_app
def crash(request):
    os._exit(23)
async def main():
    app = create_app(Path(sys.argv[1]), context_transport=httpx.MockTransport(crash))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            ticket = await client.post("/tickets", json={"external_id":"case","text":"Where is my card?","account_ref":"A17"})
            await client.post("/tickets/" + ticket.json()["id"] + "/triage")
asyncio.run(main())
"""
    crashed = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, str(database)],
                                      cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=45)
    assert crashed.returncode == 23, crashed.stderr.decode(errors="replace")
    async with running_client(database) as client:
        replay = await client.post("/tickets", json={"external_id": "case", "text": CARD, "account_ref": "A17"})
        url = "/tickets/" + replay.json()["id"]
        interrupted, = await history(client, url)
        assert interrupted["status"] == "interrupted" and interrupted["error_code"] == "worker_stopped"
        assert interrupted["elapsed_ms"] >= 0 and interrupted["finished_at"] is None
        assert interrupted["decision"] is None and interrupted["usage"] == LOCAL_USAGE
        attempt, = interrupted["attempts"]
        assert attempt["tool"] == "status" and attempt["number"] == 1 and attempt["outcome"] == "interrupted"
        decision = (await client.post(url + "/triage")).json()
        first, retry = await history(client, url)
        assert first == interrupted and retry["retry_of_run_id"] == first["run_id"]
        assert retry["run_id"] != first["run_id"] and retry["status"] == "completed"
        assert retry["decision"] == decision and decision["recommended_action"] == "prepare_reply"


async def test_live_worker_and_concurrent_winner(database, running_client):
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(request):
        entered.set()
        await release.wait()
        return httpx.Response(503)

    async with running_client(database, context_transport=httpx.MockTransport(delayed)) as first:
        url = await submit(first)
        active = asyncio.create_task(first.post(url + "/triage"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            async with running_client(database) as second:
                current, = await history(second, url)
                assert current["status"] == "running" and current["attempts"][0]["outcome"] == "started"
                winner = (await second.post(url + "/triage")).json()
        finally:
            release.set()
            response = await active
        assert response.json() == winner
        loser, published = await history(first, url)
        assert loser["status"] == "superseded" and loser["decision"] is None
        assert published["status"] == "completed" and published["decision"] == winner


async def test_cancelled_request_can_retry_in_same_worker(database, running_client):
    entered = asyncio.Event()

    async def handler(request):
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json=STATUS_RECORDS["A17", "delivery"].model_dump())

    async with running_client(database, context_transport=httpx.MockTransport(handler)) as client:
        url = await submit(client)
        task = asyncio.create_task(client.post(url + "/triage"))
        await asyncio.wait_for(entered.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        interrupted, = await history(client, url)
        assert interrupted["status"] == "interrupted" and interrupted["error_code"] == "request_cancelled"
        assert interrupted["attempts"][0]["outcome"] == "interrupted"
        assert (await client.post(url + "/triage")).status_code == 200
        _, retry = await history(client, url)
        assert retry["retry_of_run_id"] == interrupted["run_id"] and retry["status"] == "completed"


async def test_pre_audit_review_migrates_without_inventing_measurements(database, running_client):
    async with running_client(database) as client:
        url = await submit(client, MIXED)
        original = (await client.post(url + "/triage")).json()
        await client.post("/reviews/" + url.split("/")[-1] + "/resolve", json={"explanation": "Confirmed."})
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("ALTER TABLE reviews DROP COLUMN run_id")
        db.execute("DROP TABLE run_attempts")
        db.execute("DROP TABLE runs")
    for _ in range(2):
        async with running_client(database) as client:
            run, = await history(client, url)
            assert run["status"] == "legacy" and run["decision"] == original
            assert run["started_at"] is run["elapsed_ms"] is run["classifier_version"] is None
            assert run["reviewer_event"]["outcome"] == "confirmed" and run["attempts"] == []


async def test_audit_write_after_success_does_not_exhaust_tool_deadline():
    now = 0.0

    async def record(event):
        nonlocal now
        now += 2  # Each local write alone would exceed the entire tool deadline.

    def handler(request):
        return httpx.Response(200, json=STATUS_RECORDS["A17", "delivery"].model_dump())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        prediction = Classification(category="delivery", reason="model_prediction", confidence_score=0.8)
        result = await lookup_context(client, prediction, CARD, "A17", clock=lambda: now, on_attempt=record)
        assert result.outcome == "found" and result.attempts == 1 and now == 4


async def test_failed_commit_has_typed_outcome_and_retry_link(database, running_client):
    async with running_client(database) as client:
        url = await submit(client, MIXED)
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("""CREATE TRIGGER fail_review BEFORE INSERT ON reviews
                          BEGIN SELECT RAISE(ABORT, 'private diagnostic'); END""")
        response = await client.post(url + "/triage")
        assert response.status_code == 500
        failed, = await history(client, url)
        assert failed["status"] == "failed" and failed["error_code"] == "storage_error"
        assert failed["decision"] is failed["reviewer_event"] is None
        assert "private diagnostic" not in str(failed)
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("DROP TRIGGER fail_review")
        assert (await client.post(url + "/triage")).status_code == 200
        first, retry = await history(client, url)
        assert first == failed and retry["retry_of_run_id"] == failed["run_id"]
