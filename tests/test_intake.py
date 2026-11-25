import asyncio
import sqlite3
from contextlib import closing

import pytest


pytestmark = pytest.mark.anyio

DELIVERY = {"external_id": "delivery-001", "text": "Where is my card?", "account_ref": "A17"}
MIXED = {"external_id": "mixed-001", "text": "My card hasn't arrived and someone used it"}


@pytest.mark.parametrize("payload", [DELIVERY, MIXED])
async def test_submit_retrieve_and_replay_after_restart(database, payload, running_client):
    async with running_client(database) as first:
        response = await first.post("/tickets", json=payload)
        assert response.status_code == 201
        ticket = response.json()
        assert ticket["id"]
        assert ticket == {"account_ref": None, **payload, "id": ticket["id"], "state": "received"}

    async with running_client(database) as restarted:
        fetched = await restarted.get(f"/tickets/{ticket['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == ticket
        replay = await restarted.post("/tickets", json=payload)
        assert replay.status_code == 200
        assert replay.json() == ticket


@pytest.mark.parametrize("changes", [
    {"text": ""}, {"text": " \n\t"}, {"text": "x" * 10_001}, {"text": 123},
    {"external_id": " "}, {"account_ref": " "}, {"unexpected": True},
], ids=["empty", "whitespace", "oversized", "wrong-type", "blank-id", "blank-account", "unknown-field"])
async def test_invalid_submission(client, changes):
    assert (await client.post("/tickets", json=DELIVERY | changes)).status_code == 422
    assert (await client.post("/tickets", json=DELIVERY)).status_code == 201


async def test_max_length_preserves_whitespace(client):
    text = " " + "x" * 9998 + "\n"
    response = await client.post("/tickets", json=DELIVERY | {"text": text})
    assert response.status_code == 201
    assert response.json()["text"] == text


async def test_unknown_ticket(client):
    assert (await client.get("/tickets/no-such-id")).status_code == 404


@pytest.mark.parametrize("changes", [{"text": "Where is my card? "}, {"account_ref": "B42"}, {"account_ref": None}])
async def test_conflict_preserves_original(client, changes):
    original = (await client.post("/tickets", json=DELIVERY)).json()
    conflict = await client.post("/tickets", json=DELIVERY | changes)
    assert conflict.status_code == 409
    assert (await client.get(f"/tickets/{original['id']}")).json() == original


async def test_concurrent_replays_store_one_ticket(client, database):
    results = await asyncio.gather(*(client.post("/tickets", json=DELIVERY) for _ in range(12)))
    assert sorted(result.status_code for result in results) == [200] * 11 + [201]
    assert all(result.json() == results[0].json() for result in results)
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT count(*) FROM tickets").fetchone()[0] == 1
