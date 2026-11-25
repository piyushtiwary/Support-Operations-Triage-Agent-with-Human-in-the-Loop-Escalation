"""Local ticket intake and triage API. Run with uv run --locked uvicorn main:app.

Mocks run in-process by default. To exercise a real connection or disconnect it,
run uvicorn context_api:app --port 8001 and set TRIAGE_CONTEXT_URL to
http://127.0.0.1:8001 before starting the ticket API.
"""

import os
import sqlite3
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from classification import MODEL_PATH, Classification, Classifier, classify
from context_api import Nonblank, router as context_router
from context_tools import Lookup, lookup_context
from policy import Category, Decision, PreparedReply, ReviewDecision, Urgency, after_lookup, before_lookup


LOOKUP_ADAPTER = TypeAdapter(Lookup)
DECISION_ADAPTER = TypeAdapter(Decision)


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


class ReviewSubmission(BaseModel):
    """Explain the review. Omit a correction to keep the original value."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_id: Nonblank = Field(default="demo-reviewer", description="Local demo identity")
    explanation: Nonblank
    category: Category | None = None
    urgency: Urgency | None = None
    internal_next_step: Nonblank | None = None


class ReviewResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_id: Nonblank
    explanation: Nonblank
    resolved_at: datetime
    outcome: Literal["confirmed", "corrected"]
    category: Category
    urgency: Urgency
    internal_next_step: Nonblank


class Ticket(TicketSubmission):
    id: str
    state: Literal["received", "classified", "draft_ready", "pending_review", "reviewed"]
    classification: Classification | None = None
    lookup: Lookup | None = None
    decision: Decision | None = None
    resolution: ReviewResolution | None = None


TICKET_WITH_REVIEW = """
    SELECT tickets.*, reviews.resolution FROM tickets
    LEFT JOIN reviews ON reviews.ticket_id = tickets.id
"""


def ticket_from_row(row: sqlite3.Row) -> Ticket:
    fields = dict(row)
    result = fields.pop("classification")
    if result is not None:
        fields["classification"] = Classification.model_validate_json(result)
    lookup = fields.pop("lookup")
    if lookup is not None:
        fields["lookup"] = LOOKUP_ADAPTER.validate_json(lookup)
    decision = fields.pop("decision")
    if decision is not None:
        fields["decision"] = DECISION_ADAPTER.validate_json(decision)
    resolution = fields.pop("resolution")
    if resolution is not None:
        fields["resolution"] = ReviewResolution.model_validate_json(resolution)
    return Ticket(**fields)


def create_app(db_path: Path = Path("tickets.sqlite3"), *, model_path: Path = MODEL_PATH,
               context_base_url: str | None = None,
               context_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    db_path = db_path.resolve()

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
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
            if "decision" not in columns:
                db.execute("ALTER TABLE tickets ADD COLUMN decision TEXT")
            db.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    ticket_id TEXT PRIMARY KEY NOT NULL REFERENCES tickets(id),
                    resolution TEXT
                )
            """)
            # Escalations saved by Step 4 already have a decision, so triage
            # replays bypass the write path. Enqueue them during the upgrade.
            db.execute("""
                INSERT INTO reviews (ticket_id)
                SELECT id FROM tickets WHERE state = 'pending_review'
                ON CONFLICT(ticket_id) DO NOTHING
            """)
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
                TICKET_WITH_REVIEW + " WHERE external_id = ?", (submission.external_id,)
            ).fetchone()
            ticket = ticket_from_row(row)
            if not inserted:
                if (ticket.text, ticket.account_ref) != (submission.text, submission.account_ref):
                    raise HTTPException(409, "External ID already used with different content")
                response.status_code = 200
            return ticket

    def find_ticket(db: sqlite3.Connection, ticket_id: str) -> Ticket:
        row = db.execute(TICKET_WITH_REVIEW + " WHERE tickets.id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Ticket not found")
        return ticket_from_row(row)

    @app.get("/tickets/{id}", response_model_exclude_unset=True,
             responses={404: {"description": "Ticket not found"}})
    def get_ticket(id: str) -> Ticket:
        """Retrieve a ticket by the ID returned at intake."""
        with closing(connect()) as db:
            return find_ticket(db, id)

    def save_triage(ticket_id: str, prediction: Classification, lookup: Lookup | None,
                    decision: Decision) -> Decision:
        state = "draft_ready" if isinstance(decision, PreparedReply) else "pending_review"
        with closing(connect()) as db, db:
            # The winning transaction publishes the decision and its inputs together.
            cursor = db.execute(
                """UPDATE tickets SET state = ?, classification = ?, lookup = ?, decision = ?
                   WHERE id = ? AND decision IS NULL""",
                (state, prediction.model_dump_json(), lookup.model_dump_json() if lookup else None,
                 decision.model_dump_json(), ticket_id),
            )
            if cursor.rowcount:
                if decision.needs_human_review:
                    db.execute("INSERT INTO reviews (ticket_id) VALUES (?)", (ticket_id,))
                return decision
            return find_ticket(db, ticket_id).decision

    @app.post("/tickets/{id}/triage", responses={404: {"description": "Ticket not found"}})
    async def triage_ticket(id: str) -> Decision:
        """Prepare an evidence-backed draft or request review. Replays return the stored decision."""
        ticket = await run_in_threadpool(get_ticket, id)
        if ticket.decision is not None:
            return ticket.decision
        model = app.state.classifier
        prediction = ticket.classification
        if prediction is None or (prediction.reason == "model_unavailable" and model):
            prediction = await run_in_threadpool(classify, ticket.text, model)
        decision = before_lookup(ticket.text, prediction, model.threshold if model else None)
        lookup = ticket.lookup
        if decision is None:
            if lookup is None:
                lookup = await lookup_context(app.state.context_client, prediction, ticket.text, ticket.account_ref)
            decision = after_lookup(prediction, ticket.text, ticket.account_ref, lookup)
        return await run_in_threadpool(save_triage, ticket.id, prediction, lookup, decision)

    @app.get("/reviews", response_model_exclude_unset=True)
    def list_reviews(state: Literal["pending", "resolved"] = "pending") -> list[Ticket]:
        """List review tickets with their original reasons, lookup evidence and suggested next step."""
        ticket_state = "pending_review" if state == "pending" else "reviewed"
        with closing(connect()) as db:
            rows = db.execute(TICKET_WITH_REVIEW + """
                WHERE reviews.ticket_id IS NOT NULL AND tickets.state = ? ORDER BY reviews.rowid
            """, (ticket_state,))
            return [ticket_from_row(row) for row in rows]

    @app.post("/reviews/{id}/resolve", responses={
        404: {"description": "No review for this ticket ID"},
        409: {"description": "A different resolution was already accepted"},
    })
    def resolve_review(id: str, submission: ReviewSubmission) -> ReviewResolution:
        """Use the ticket ID from the queue. Confirm with an explanation or include corrected fields."""
        with closing(connect()) as db, db:
            # Serialize the read and both writes, including across app instances.
            db.execute("BEGIN IMMEDIATE")
            ticket = find_ticket(db, id)
            original = ticket.decision
            if not isinstance(original, ReviewDecision):
                raise HTTPException(404, "No review for this ticket")
            category = submission.category or original.category
            urgency = submission.urgency or original.urgency
            step = submission.internal_next_step or original.internal_next_step
            confirmed = (category, urgency, step) == (original.category, original.urgency, original.internal_next_step)
            existing = ticket.resolution
            resolution = ReviewResolution(
                reviewer_id=submission.reviewer_id, explanation=submission.explanation,
                resolved_at=existing.resolved_at if existing else datetime.now(UTC),
                outcome="confirmed" if confirmed else "corrected",
                category=category, urgency=urgency, internal_next_step=step,
            )
            if existing is not None:
                if existing != resolution:
                    raise HTTPException(409, "A different resolution was already accepted")
                return existing
            updated = db.execute("UPDATE reviews SET resolution = ? WHERE ticket_id = ? AND resolution IS NULL",
                                 (resolution.model_dump_json(), id))
            if updated.rowcount != 1:
                raise HTTPException(404, "No pending review for this ticket")
            db.execute("UPDATE tickets SET state = 'reviewed' WHERE id = ?", (id,))
            return resolution

    return app


app = create_app()
