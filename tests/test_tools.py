import asyncio
import sqlite3
from contextlib import closing

import httpx
import pytest

import context_tools
from classification import Classification
from context_api import RUNBOOKS, STATUS_RECORDS
from context_tools import lookup_context


pytestmark = pytest.mark.anyio
DELIVERY = Classification(category="delivery", raw_category="delivery", reason="model_prediction",
                          confidence_score=0.9)
CARD = STATUS_RECORDS["A17", "delivery"].model_dump()


class Clock:
    def __init__(self):
        self.now = 0.0

    async def sleep(self, seconds):
        self.now += seconds


async def lookup(handler, *, text="Where is my card?", clock=None, sleep=None):
    clock = clock or Clock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 base_url="http://test") as client:
        return await lookup_context(client, DELIVERY, text, "A17", clock=lambda: clock.now,
                                    sleep=sleep or clock.sleep)


async def test_retry_503_then_success():
    clock = Clock()
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET" and request.url.path == "/mock/status"
        assert dict(request.url.params) == {"account_ref": "A17", "topic": "delivery"}
        assert set(request.extensions["timeout"].values()) == {0.5}
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=CARD)

    result = await lookup(handler, clock=clock)
    assert result.attempts == len(requests) == 2 and clock.now == 0.1
    assert result.evidence.model_dump() == CARD


async def test_repeated_timeout():
    clock = Clock()
    attempts = []

    def handler(request):
        attempts.append(clock.now)
        clock.now += request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout("timed out", request=request)

    result = await lookup(handler, clock=clock)
    assert result.outcome == "unavailable" and result.reason == "timeout" and result.attempts == 2
    assert attempts == pytest.approx([0, 0.6]) and clock.now == pytest.approx(1.1)


async def test_repeated_503_stops_after_one_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503)

    result = await lookup(handler)
    assert result.outcome == "unavailable" and result.reason == "service_unavailable"
    assert len(calls) == result.attempts == 2


@pytest.mark.parametrize(("response", "text", "reason"), [
    (httpx.Response(404), "Where is my card?", "not_found"),
    (httpx.Response(500), "Where is my card?", "http_error"),
    (httpx.Response(200, content=b"{"), "Where is my card?", "invalid_schema"),
    (httpx.Response(200, json=CARD | {"revision": 1}), "Where is my card?", "invalid_schema"),
    (httpx.Response(200, json=CARD | {"account_ref": "B42"}), "Where is my card?", "account_mismatch"),
    (httpx.Response(200, json=CARD | {"topic": "transfer"}), "Where is my card?", "topic_mismatch"),
    (httpx.Response(200, json=[]), "How long does card delivery take?", "empty_match"),
    (httpx.Response(200, json=[RUNBOOKS[1].model_dump()]), "How long is delivery?", "topic_mismatch"),
    (httpx.Response(200, json=[book.model_dump() for book in RUNBOOKS]),
     "How long is delivery?", "conflicting_evidence"),
])
async def test_terminal_failure(response, text, reason):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    result = await lookup(handler, text=text)
    assert result.reason == reason and len(calls) == result.attempts == 1
    assert "evidence" not in result.model_dump()


@pytest.mark.parametrize(("wake_at", "expected_attempts"), [(1.4, 2), (1.6, 1)])
async def test_total_deadline_includes_backoff_and_slow_read(wake_at, expected_attempts):
    clock = Clock()
    caps = []

    def handler(request):
        caps.append(request.extensions["timeout"]["read"])
        if len(caps) == 1:
            return httpx.Response(503)
        clock.now += 0.2
        return httpx.Response(200, json=CARD)

    async def delayed_wakeup(seconds):
        assert seconds == 0.1
        clock.now = wake_at

    result = await lookup(handler, clock=clock, sleep=delayed_wakeup)
    assert result.reason == "deadline_exceeded" and result.attempts == expected_attempts
    assert caps == pytest.approx([0.5, 0.1][:expected_attempts])


async def test_outer_deadline_cancels_blocked_backoff(monkeypatch):
    monkeypatch.setattr(context_tools, "LOOKUP_SECONDS", 0.05)
    monkeypatch.setattr(context_tools, "BACKOFF_SECONDS", 0.01)
    blocked = asyncio.Event()

    async def stuck_sleep(seconds):
        await blocked.wait()

    async with asyncio.timeout(1):
        result = await lookup(lambda _: httpx.Response(503), sleep=stuck_sleep)
    assert result.reason == "deadline_exceeded" and result.attempts == 1


async def test_loopback_read_is_bounded_across_chunks():
    body = STATUS_RECORDS["A17", "delivery"].model_dump_json().encode()
    connections = []

    async def serve(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode())
            for offset in range(0, len(body), 10):
                writer.write(body[offset:offset + 10])
                await writer.drain()
                await asyncio.sleep(0.1)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    def accept(reader, writer):
        connections.append(asyncio.create_task(serve(reader, writer)))

    async with await asyncio.start_server(accept, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
                async with asyncio.timeout(2):
                    result = await lookup_context(client, DELIVERY, "Where is my card?", "A17")
            assert result.reason == "timeout" and result.attempts == len(connections) == 2
        finally:
            for task in connections:
                task.cancel()
            await asyncio.gather(*connections, return_exceptions=True)


@pytest.mark.parametrize(("text", "account", "tool", "field", "expected"), [
    ("Where is my card?", "A17", "status", "status", "dispatched"),
    ("My bank transfer is still pending", "A17", "status", "status", "pending"),
    ("How long does card delivery take?", None, "runbook", "source_id", "card-delivery"),
    ("How long does a bank transfer take?", "A17", "runbook", "source_id", "bank-transfer"),
    ("My app freezes when I open card tracking.", None, "runbook", "approved_next_step",
     "Open card tracking in the web app."),
    ("Where is my card?", None, "status", "reason", "missing_account_ref"),
    ("Where is my card?", "unknown", "status", "reason", "not_found"),
])
async def test_stored_ticket_lookup(database, running_client, text, account, tool, field, expected):
    async with running_client(database) as client:
        submission = {"external_id": "scenario", "text": text, "account_ref": account}
        ticket = (await client.post("/tickets", json=submission)).json()
        url = f"/tickets/{ticket['id']}"
        decision = (await client.post(url + "/triage")).json()
        stored = (await client.get(url)).json()
        assert stored["decision"] == decision
        result = stored["lookup"]
        assert result["tool"] == tool
        assert result["attempts"] == (0 if expected == "missing_account_ref" else 1)
        if field == "reason":
            assert result["outcome"] == "missing_context" and result["reason"] == expected
            assert "evidence" not in result
        else:
            evidence = result["evidence"]
            assert result["outcome"] == "found" and evidence[field] == expected
            assert evidence["source_id"] and evidence["revision"] == "v1"

    def unexpected_lookup(request):
        pytest.fail(f"Replay attempted {request.url.path}")

    async with running_client(database, context_transport=httpx.MockTransport(unexpected_lookup)) as client:
        assert (await client.post(url + "/triage")).json() == decision
        assert (await client.get(url)).json() == stored


async def test_disconnected_status_is_persisted(database, running_client):
    calls = []

    def disconnected(request):
        calls.append(request)
        raise httpx.ConnectError("offline", request=request)

    async with running_client(database, context_transport=httpx.MockTransport(disconnected)) as client:
        ticket = (await client.post("/tickets", json={
            "external_id": "outage", "text": "Where is my card?", "account_ref": "A17",
        })).json()
        url = f"/tickets/{ticket['id']}"
        assert (await client.post(url + "/triage")).status_code == 200
    async with running_client(database) as client:
        assert (await client.get(url)).json()["lookup"] == {
            "tool": "status", "attempts": 1, "outcome": "unavailable", "reason": "connection_error"
        }
    assert len(calls) == 1


async def test_enrich_existing_step_two_classification(database, running_client):
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""CREATE TABLE tickets (id TEXT PRIMARY KEY, external_id TEXT UNIQUE,
                      text TEXT, account_ref TEXT, state TEXT, classification TEXT)""")
        db.execute("""INSERT INTO tickets VALUES ('old', 'old', 'Where is my card?', 'A17',
                      'classified', ?)""", (DELIVERY.model_dump_json(),))
    async with running_client(database) as client:
        assert (await client.post("/tickets/old/triage")).json()["recommended_action"] == "prepare_reply"
        stored = (await client.get("/tickets/old")).json()
        assert stored["classification"] == DELIVERY.model_dump()
        assert stored["lookup"]["evidence"] == CARD
