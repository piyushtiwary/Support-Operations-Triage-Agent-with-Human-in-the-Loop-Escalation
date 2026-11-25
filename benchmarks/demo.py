"""Rehearse the three interview tickets on an isolated database through the real HTTP routes."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from benchmarks.prepare import read_json
from benchmarks.run import SeededContext
from main import create_app


async def demo(database):
    context = SeededContext()
    app = create_app(database, context_transport=context)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://demo"
    ) as client:
        async def submit(external_id, text):
            ticket = (await client.post("/tickets", json={
                "external_id": external_id, "text": text, "account_ref": "A17"})).raise_for_status().json()
            url = "/tickets/" + ticket["id"]
            (await client.post(url + "/triage")).raise_for_status()
            return url, (await client.get(url)).raise_for_status().json()

        _, card = await submit("demo-card", "Where is my card?")
        assert card["state"] == "draft_ready"
        print("Routine ticket", card["decision"]["draft"], card["decision"]["evidence_ids"])
        print(json.dumps(card["lookup"]["evidence"], indent=2))

        url, risk = await submit("demo-mixed", "Where is my card? I see a payment I didn't make.")
        pending = (await client.get("/reviews?state=pending")).raise_for_status().json()
        assert [ticket["id"] for ticket in pending] == [risk["id"]]
        print("Mixed ticket", risk["classification"]["category"], "prediction;", risk["decision"]["reason_codes"],
              "review reasons")
        resolved = (await client.post("/reviews/" + risk["id"] + "/resolve", json={
            "reviewer_id": "demo-operator", "explanation": "Possible unauthorized payment takes priority.",
            "internal_next_step": "Ask security to verify the reported payment before discussing dispatch.",
        })).raise_for_status().json()
        audit, = (await client.get(url + "/runs")).raise_for_status().json()
        assert resolved["outcome"] == "corrected" and audit["decision"] == risk["decision"]
        assert (await client.get("/reviews?state=pending")).raise_for_status().json() == []
        print("Resolved", json.dumps(audit["reviewer_event"], indent=2))

        context.fault = "503"
        url, outage = await submit("demo-outage", "Where is my card?")
        audit, = (await client.get(url + "/runs")).raise_for_status().json()
        statuses = [a["http_status"] for a in audit["attempts"]]
        assert statuses == [503, 503] and outage["decision"]["draft"] is None and outage["state"] == "pending_review"
        print("Status outage", statuses, audit["decision"]["reason_codes"], audit["decision"]["internal_next_step"])

    release = read_json(Path(__file__).with_name("results") / "summary.json")
    public = release["quality"]["public"]["correct_routine_drafts"]
    print(f"Release benchmark: {public['numerator']}/{public['denominator']} routine public drafts usable.")
    for name, quality in release["quality"].items():
        risk = quality["risk_reviews"]
        print(f"{name}: {risk['denominator'] - risk['numerator']}/{risk['denominator']} high-risk cases missed.")
    times = release["public_effort"]
    print(f"Simulated saving: {times['base']['saving_percent']}% base, "
          f"{times['slower_review']['saving_percent']}% with slower review.")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="triage-demo-") as temp:
        asyncio.run(demo(Path(temp) / "demo.sqlite3"))
