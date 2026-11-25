"""Read-only local fixtures. Can also run separately with uvicorn context_api:app."""

from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StringConstraints


Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
StatusTopic = Literal["delivery", "transfer"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_id: Nonblank
    revision: Nonblank


class StatusEvidence(Evidence):
    account_ref: Nonblank
    topic: StatusTopic
    status: Nonblank


class RunbookEvidence(Evidence):
    topic: Literal["delivery", "transfer", "card-tracking-freeze-v1"]
    guidance: Nonblank
    approved_next_step: Nonblank


STATUS_RECORDS = {
    ("A17", "delivery"): StatusEvidence(source_id="card-A17", revision="v1",
        account_ref="A17", topic="delivery", status="dispatched"),
    ("A17", "transfer"): StatusEvidence(source_id="transfer-A17", revision="v1",
        account_ref="A17", topic="transfer", status="pending"),
    ("B42", "delivery"): StatusEvidence(source_id="card-B42", revision="v1",
        account_ref="B42", topic="delivery", status="processing"),
}

RUNBOOKS = (
    RunbookEvidence(source_id="card-delivery", revision="v1", topic="delivery",
        guidance="Standard card delivery usually takes 5 to 7 business days after dispatch.",
        approved_next_step="Check the dispatch date in card tracking."),
    RunbookEvidence(source_id="bank-transfer", revision="v1", topic="transfer",
        guidance="Standard bank transfers usually take 1 to 3 business days.",
        approved_next_step="Check the transfer details for its submission date."),
    RunbookEvidence(source_id="mobile-card-tracking", revision="v1", topic="card-tracking-freeze-v1",
        guidance="The mobile app can freeze when opening the card tracking screen.",
        approved_next_step="Open card tracking in the web app."),
)

router = APIRouter(prefix="/mock", tags=["Mock context tools"])


@router.get("/status")
async def get_status(account_ref: str, topic: StatusTopic) -> StatusEvidence:
    """Read the account record for one product."""
    record = STATUS_RECORDS.get((account_ref, topic))
    if record is None:
        raise HTTPException(404, "Account record not found")
    return record


@router.get("/runbooks")
async def search_runbooks(q: str) -> list[RunbookEvidence]:
    """Search the topic, guidance and approved next step by keyword."""
    words = q.casefold().split()
    matches = []
    for book in RUNBOOKS:
        content = f"{book.topic} {book.guidance} {book.approved_next_step}".casefold()
        if words and all(word in content for word in words):
            matches.append(book)
    return matches


app = FastAPI(title="Mock support context")
app.include_router(router)
