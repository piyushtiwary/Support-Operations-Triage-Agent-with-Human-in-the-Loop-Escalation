"""Local ticket intake API. Run with uv run --locked uvicorn main:app."""

import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    state: Literal["received"]


def create_app(db_path: Path = Path("tickets.sqlite3")) -> FastAPI:
    db_path = db_path.resolve()

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with closing(connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY NOT NULL,
                    external_id TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    account_ref TEXT,
                    state TEXT NOT NULL DEFAULT 'received'
                )
            """)
        yield

    app = FastAPI(title="Support & Operations Triage", lifespan=lifespan)

    @app.post(
        "/tickets",
        status_code=201,
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
            ticket = Ticket(**row)
            if not inserted:
                if (ticket.text, ticket.account_ref) != (submission.text, submission.account_ref):
                    raise HTTPException(409, "External ID already used with different content")
                response.status_code = 200
            return ticket

    @app.get("/tickets/{id}", responses={404: {"description": "Ticket not found"}})
    def get_ticket(id: str) -> Ticket:
        """Retrieve a ticket by the ID returned at intake."""
        with closing(connect()) as db:
            row = db.execute("SELECT * FROM tickets WHERE id = ?", (id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Ticket not found")
        return Ticket(**row)

    return app


app = create_app()
