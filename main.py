"""Local ticket intake and triage API. Run with uv run --locked uvicorn main:app.

Mocks run in-process by default. To exercise a real connection or disconnect it,
run uvicorn context_api:app --port 8001 and set TRIAGE_CONTEXT_URL to
http://127.0.0.1:8001 before starting the ticket API.
"""

import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from classification import MODEL_PATH, Classification, Classifier, classify
from context_api import router as context_router
from context_tools import Lookup, lookup_context


LOOKUP_ADAPTER = TypeAdapter(Lookup)


class TicketSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    external_id: str = Field(min_length=1, examples=["delivery-001"])
    text: str = Field(min_length=1, max_length=10_000, examples=["Where is my card?"])
    account_ref: str | None = Field(default=None, min_length=1, examples=["A17"])

    @field_validator("external_id", "text", "account_ref")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Provide at least one non-whitespace character")
        return value


class Ticket(TicketSubmission):
    id: str
    state: Literal["received", "classified"]
    classification: Classification | None = None
    lookup: Lookup | None = None


def ticket_from_row(row: sqlite3.Row) -> Ticket:
    fields = dict(row)
    result = fields.pop("classification")
    if result is not None:
        fields["classification"] = Classification.model_validate_json(result)
    lookup = fields.pop("lookup")
    if lookup is not None:
        fields["lookup"] = LOOKUP_ADAPTER.validate_json(lookup)
    return Ticket(**fields)


def create_app(db_path: Path = Path("tickets.sqlite3"), *, model_path: Path = MODEL_PATH,
               context_base_url: str | None = None,
               context_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    db_path = db_path.resolve()

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with closing(connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY NOT NULL,
                    external_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    account_ref TEXT,
                    state TEXT NOT NULL DEFAULT 'received'
                )
            """)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(tickets)")}
            if "classification" not in columns:
                db.execute("ALTER TABLE tickets ADD COLUMN classification TEXT")
            if "lookup" not in columns:
                db.execute("ALTER TABLE tickets ADD COLUMN lookup TEXT")
        app.state.classifier = Classifier.load(model_path)
        base_url = context_base_url or os.getenv("TRIAGE_CONTEXT_URL")
        transport = context_transport
        if transport is None and not base_url:
            transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(base_url=base_url or "http://local-context",
                                     transport=transport, trust_env=False) as client:
            app.state.context_client = client
            yield

    app = FastAPI(title="Support & Operations Triage", lifespan=lifespan)
    app.include_router(context_router)

    @app.post(
        "/tickets",
        status_code=201,
        response_model_exclude_unset=True,
        responses={
            200: {"model": Ticket, "description": "Identical submission replayed"},
            409: {"description": "External ID already used with different content"},
        },
    )
    def submit_ticket(submission: TicketSubmission, response: Response) -> Ticket:
        """Create a ticket or recover it by replaying the original submission."""
        with closing(connect()) as db, db:
            inserted = db.execute(
                """
                INSERT INTO tickets (id, external_id, text, account_ref)
                VALUES (?, ?, ?, ?) ON CONFLICT(external_id) DO NOTHING
                """,
                (str(uuid4()), submission.external_id, submission.text, submission.account_ref),
            ).rowcount
            row = db.execute(
                "SELECT * FROM tickets WHERE external_id = ?", (submission.external_id,)
            ).fetchone()
            ticket = ticket_from_row(row)
            if not inserted:
                if (ticket.text, ticket.account_ref) != (submission.text, submission.account_ref):
                    raise HTTPException(409, "External ID already used with different content")
                response.status_code = 200
            return ticket

    def find_ticket(db: sqlite3.Connection, ticket_id: str) -> Ticket:
        row = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Ticket not found")
        return ticket_from_row(row)

    @app.get("/tickets/{id}", response_model_exclude_unset=True,
             responses={404: {"description": "Ticket not found"}})
    def get_ticket(id: str) -> Ticket:
        """Retrieve a ticket by the ID returned at intake."""
        with closing(connect()) as db:
            return find_ticket(db, id)

    def save_triage(ticket: Ticket, prediction: Classification, lookup: Lookup | None) -> Classification:
        with closing(connect()) as db, db:
            # Publish both together; the first completed lookup wins on concurrent replay.
            cursor = db.execute(
                """UPDATE tickets SET state = 'classified', classification = ?, lookup = ?
                   WHERE id = ? AND classification IS ? AND lookup IS NULL""",
                (prediction.model_dump_json(), lookup.model_dump_json() if lookup else None,
                 ticket.id, ticket.classification.model_dump_json() if ticket.classification else None),
            )
            if cursor.rowcount:
                return prediction
            return find_ticket(db, ticket.id).classification

    @app.post("/tickets/{id}/triage", responses={404: {"description": "Ticket not found"}})
    async def triage_ticket(id: str) -> Classification:
        """Classify and look up relevant context. GET the ticket to inspect the stored lookup."""
        ticket = await run_in_threadpool(get_ticket, id)
        prediction = ticket.classification
        if ticket.lookup is not None:
            return prediction
        if prediction is None or (prediction.reason == "model_unavailable" and app.state.classifier):
            prediction = await run_in_threadpool(classify, ticket.text, app.state.classifier)
        lookup = await lookup_context(app.state.context_client, prediction, ticket.text, ticket.account_ref)
        if prediction == ticket.classification and lookup is None:
            return prediction
        return await run_in_threadpool(save_triage, ticket, prediction, lookup)

    return app


app = create_app()
