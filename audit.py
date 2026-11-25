"""Durable, text-free diagnostics for local triage runs."""

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from classification import Classification
from context_tools import LOOKUP_ADAPTER, Found, Lookup, Tool, ToolAttempt, evidence_ids
from policy import RULESET_VERSION, Category, Decision, Urgency


class LookupSummary(BaseModel):
    tool: Tool
    source: Literal["live", "stored"]
    outcome: Literal["found", "failed", "missing_context", "unavailable"]
    reason: str | None
    attempts: int
    evidence_ids: list[str]

    @classmethod
    def from_result(cls, lookup: Lookup, *, stored: bool):
        return cls(tool=lookup.tool, source="stored" if stored else "live", outcome=lookup.outcome,
                   reason=None if isinstance(lookup, Found) else lookup.reason,
                   attempts=lookup.attempts, evidence_ids=evidence_ids(lookup))


class ProviderUsage(BaseModel):
    source: Literal["local"] = "local"
    provider_calls: Literal[0] = 0
    input_tokens: Literal[0] = 0
    output_tokens: Literal[0] = 0
    estimated_provider_cost_usd: Literal[0] = 0


class ReviewerEvent(BaseModel):
    reviewer_id: str
    resolved_at: datetime
    outcome: Literal["confirmed", "corrected"]
    category: Category
    urgency: Urgency
    changed_fields: list[str]


class AuditRun(BaseModel):
    run_id: str
    ticket_id: str
    retry_of_run_id: str | None
    status: Literal["running", "completed", "superseded", "failed", "interrupted", "legacy"]
    classifier_version: str | None
    ruleset_version: str | None
    started_at: datetime | None
    finished_at: datetime | None
    elapsed_ms: float | None = Field(ge=0, description="Monotonic processing time through the last checkpoint.")
    error_code: str | None
    classification: Classification | None
    lookup: LookupSummary | None
    attempts: list[ToolAttempt]
    decision: Decision | None
    usage: ProviderUsage
    reviewer_event: ReviewerEvent | None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditLog:
    def __init__(self, path: Path, connect: Callable[[], sqlite3.Connection]):
        self.path = path
        self.connect = connect

    def owner_path(self, owner: str) -> Path:
        return self.path.with_name(f"{self.path.name}-{UUID(owner)}.owner")

    @contextmanager
    def worker(self):
        # An OS-backed SQLite reservation disappears even on a hard process exit.
        # Starting a second worker must never interrupt the first worker's runs.
        owner = str(uuid4())
        path = self.owner_path(owner)
        with closing(sqlite3.connect(path)) as lease:
            lease.execute("BEGIN IMMEDIATE")
            try:
                yield owner
            finally:
                # Serialize release/removal with other workers probing this file.
                with closing(self.connect()) as db, db:
                    db.execute("BEGIN IMMEDIATE")
                    lease.close()
                    path.unlink(missing_ok=True)

    def recover(self, db: sqlite3.Connection):
        owners = db.execute("SELECT DISTINCT owner FROM runs WHERE status = 'running'").fetchall()
        for row in owners:
            path = self.owner_path(row["owner"])
            if path.exists():
                with closing(sqlite3.connect(path, timeout=0)) as lease:
                    try:
                        lease.execute("BEGIN IMMEDIATE")
                    except sqlite3.OperationalError as exc:
                        if exc.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                            continue
                        raise
                path.unlink(missing_ok=True)
            db.execute("""UPDATE runs SET status = 'interrupted', error_code = 'worker_stopped'
                          WHERE status = 'running' AND owner = ?""", (row["owner"],))
            db.execute("""UPDATE run_attempts SET event = json_set(event, '$.outcome', 'interrupted')
                          WHERE run_id IN (SELECT run_id FROM runs WHERE owner = ? AND status = 'interrupted')
                          AND json_extract(event, '$.outcome') = 'started'""", (row["owner"],))

    def initialize(self, db: sqlite3.Connection):
        db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY NOT NULL,
                ticket_id TEXT NOT NULL REFERENCES tickets(id),
                retry_of_run_id TEXT REFERENCES runs(run_id),
                owner TEXT, status TEXT NOT NULL,
                classifier_version TEXT, ruleset_version TEXT,
                started_at TEXT, finished_at TEXT, elapsed_ms REAL,
                error_code TEXT, classification TEXT, lookup TEXT,
                usage TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS runs_by_ticket ON runs(ticket_id)")
        db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS published_run ON runs(ticket_id)
                      WHERE status IN ('completed', 'legacy')""")
        db.execute("""CREATE TABLE IF NOT EXISTS run_attempts (
                      run_id TEXT NOT NULL REFERENCES runs(run_id), number INTEGER NOT NULL,
                      event TEXT NOT NULL, PRIMARY KEY (run_id, number))""")
        columns = {r["name"] for r in db.execute("PRAGMA table_info(reviews)")}
        if "run_id" not in columns:
            db.execute("ALTER TABLE reviews ADD COLUMN run_id TEXT REFERENCES runs(run_id)")
        # Earlier releases saved the decision and final lookup, but no run clock or attempt trace.
        legacy = db.execute("""SELECT id, classification, lookup FROM tickets
            WHERE decision IS NOT NULL AND NOT EXISTS (SELECT 1 FROM runs WHERE ticket_id = tickets.id
                                                       AND status IN ('completed', 'legacy'))""")
        for row in legacy:
            lookup = LOOKUP_ADAPTER.validate_json(row["lookup"]) if row["lookup"] else None
            summary = LookupSummary.from_result(lookup, stored=True) if lookup else None
            db.execute("""INSERT INTO runs (run_id, ticket_id, status, classification, lookup, usage)
                          VALUES (?, ?, 'legacy', ?, ?, ?)""",
                       (str(uuid4()), row["id"], row["classification"],
                        summary.model_dump_json() if summary else None, ProviderUsage().model_dump_json()))
        db.execute("""UPDATE reviews SET run_id = (SELECT run_id FROM runs WHERE
                      runs.ticket_id = reviews.ticket_id AND status IN ('completed', 'legacy'))
                      WHERE run_id IS NULL""")
        self.recover(db)

    def start(self, db: sqlite3.Connection, run_id: str, ticket_id: str, owner: str, version: str):
        db.execute("""INSERT INTO runs (run_id, ticket_id, owner, status, classifier_version,
                      ruleset_version, started_at, elapsed_ms, usage, retry_of_run_id)
                      VALUES (?, ?, ?, 'running', ?, ?, ?, 0, ?, (SELECT run_id FROM runs
                      WHERE ticket_id = ? AND status IN ('interrupted', 'failed') ORDER BY rowid DESC LIMIT 1))""",
                   (run_id, ticket_id, owner, version, RULESET_VERSION, utc_now(),
                    ProviderUsage().model_dump_json(), ticket_id))

    def prediction(self, run_id: str, prediction: Classification, started: float):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE runs SET classification = ?, elapsed_ms = ? WHERE run_id = ?",
                       (prediction.model_dump_json(), (time.monotonic() - started) * 1000, run_id))

    def attempt(self, run_id: str, event: ToolAttempt, started: float):
        with closing(self.connect()) as db, db:
            db.execute("""INSERT INTO run_attempts VALUES (?, ?, ?) ON CONFLICT(run_id, number)
                          DO UPDATE SET event = excluded.event""", (run_id, event.number, event.model_dump_json()))
            db.execute("UPDATE runs SET elapsed_ms = ? WHERE run_id = ?",
                       ((time.monotonic() - started) * 1000, run_id))

    def stop(self, run_id: str, started: float, status: str, error_code: str):
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE runs SET status = ?, error_code = ?, finished_at = ?, elapsed_ms = ?
                          WHERE run_id = ? AND status = 'running'""",
                       (status, error_code, utc_now(), (time.monotonic() - started) * 1000, run_id))
            db.execute("""UPDATE run_attempts SET event = json_set(event, '$.outcome', 'interrupted')
                          WHERE run_id = ? AND json_extract(event, '$.outcome') = 'started'""", (run_id,))

    def history(self, db: sqlite3.Connection, ticket_id: str) -> list[AuditRun] | None:
        # Read all parts from one SQLite snapshot, including any concurrent reviewer update.
        db.execute("BEGIN")
        if not db.execute("SELECT 1 FROM tickets WHERE id = ?", (ticket_id,)).fetchone():
            return None
        rows = db.execute("""SELECT runs.*, tickets.decision, reviews.resolution FROM runs
            LEFT JOIN tickets ON tickets.id = runs.ticket_id AND runs.status IN ('completed', 'legacy')
            LEFT JOIN reviews ON reviews.run_id = runs.run_id WHERE runs.ticket_id = ? ORDER BY runs.rowid""",
                          (ticket_id,))
        result = []
        for row in rows:
            fields = dict(row)
            del fields["owner"]
            for key in ("classification", "lookup", "usage", "decision"):
                if fields[key] is not None:
                    fields[key] = json.loads(fields[key])
            fields["attempts"] = [ToolAttempt.model_validate_json(a["event"]) for a in db.execute(
                "SELECT event FROM run_attempts WHERE run_id = ? ORDER BY number", (row["run_id"],))]
            resolution = fields.pop("resolution")
            event = None
            if resolution:
                review = json.loads(resolution)
                changed = [k for k in ("category", "urgency", "internal_next_step")
                           if review[k] != fields["decision"][k]]
                event = ReviewerEvent(reviewer_id=review["reviewer_id"], resolved_at=review["resolved_at"],
                                      outcome=review["outcome"], category=review["category"],
                                      urgency=review["urgency"], changed_fields=changed)
            result.append(AuditRun(**fields, reviewer_event=event))
        return result
