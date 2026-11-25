"""Select a read-only tool and fetch evidence within one lookup deadline."""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from classification import Classification, normalize
from context_api import RunbookEvidence, StatusEvidence


ATTEMPT_SECONDS = 0.5
LOOKUP_SECONDS = 1.5
BACKOFF_SECONDS = 0.1
RUNBOOK_LIST = TypeAdapter(list[RunbookEvidence])
Tool = Literal["status", "runbook"]


class LookupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool: Tool
    attempts: int = Field(ge=0, le=2)


class Found(LookupResult):
    outcome: Literal["found"] = "found"
    evidence: StatusEvidence | RunbookEvidence


class Failed(LookupResult):
    outcome: Literal["failed"] = "failed"
    reason: Literal["conflicting_evidence", "account_mismatch", "topic_mismatch",
                    "invalid_schema", "http_error"]


class MissingContext(LookupResult):
    outcome: Literal["missing_context"] = "missing_context"
    reason: Literal["missing_account_ref", "not_found", "empty_match"]


class Unavailable(LookupResult):
    outcome: Literal["unavailable"] = "unavailable"
    reason: Literal["connection_error", "service_unavailable", "timeout", "deadline_exceeded"]


Lookup = Found | Failed | MissingContext | Unavailable


def general_timing(text: str) -> bool:
    words = normalize(text)
    if re.search(r"\b(how long|how many days)\b", words) and re.search(
        r"\b(usually|typically|normally|on average)\b", words
    ):
        return True
    if re.search(r"\b(pending|overdue|late|dispatched|tracking|hasn t arrived|how long has my)\b", words):
        return False
    return bool(re.search(
        r"\b(how long|how many (?:business |working )?days|"
        r"(?:delivery|transfer) (?:time|times|estimate|estimates)|"
        r"(?:usual|typical|standard) timeframe)\b", words
    ))


def read_response(response: httpx.Response, tool: Tool, topic: str, account_ref: str | None,
                  attempt: int) -> Lookup:
    if response.status_code == 404:
        return MissingContext(tool=tool, attempts=attempt, reason="not_found")
    if response.status_code == 503:
        return Unavailable(tool=tool, attempts=attempt, reason="service_unavailable")
    if not response.is_success:
        return Failed(tool=tool, attempts=attempt, reason="http_error")
    try:
        if tool == "status":
            evidence = StatusEvidence.model_validate_json(response.content)
            if evidence.account_ref != account_ref:
                return Failed(tool=tool, attempts=attempt, reason="account_mismatch")
        else:
            matches = RUNBOOK_LIST.validate_json(response.content, strict=True)
            if not matches:
                return MissingContext(tool=tool, attempts=attempt, reason="empty_match")
            if len(matches) != 1:
                return Failed(tool=tool, attempts=attempt, reason="conflicting_evidence")
            evidence = matches[0]
    except ValidationError:
        return Failed(tool=tool, attempts=attempt, reason="invalid_schema")
    if evidence.topic != topic:
        return Failed(tool=tool, attempts=attempt, reason="topic_mismatch")
    return Found(tool=tool, attempts=attempt, evidence=evidence)


async def lookup_context(client: httpx.AsyncClient, prediction: Classification, text: str,
                         account_ref: str | None, *, clock: Callable[[], float] = time.monotonic,
                         sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> Lookup | None:
    topic = prediction.category
    if topic == "technical_issue" and prediction.rule_id:
        topic = prediction.rule_id
        tool = "runbook"
    elif topic in {"delivery", "transfer"}:
        tool = "runbook" if general_timing(text) else "status"
    else:
        return None

    if tool == "status":
        if account_ref is None:
            return MissingContext(tool=tool, attempts=0, reason="missing_account_ref")
        path, params = "/mock/status", {"topic": topic, "account_ref": account_ref}
    else:
        path, params = "/mock/runbooks", {"q": topic}

    deadline = clock() + LOOKUP_SECONDS
    attempts = 0
    try:
        async with asyncio.timeout(LOOKUP_SECONDS):
            while attempts < 2:
                remaining = deadline - clock()
                if remaining <= 0:
                    break
                attempts += 1
                cap = min(ATTEMPT_SECONDS, remaining)
                attempt_deadline = clock() + cap
                try:
                    # HTTPX timeouts apply per phase. Also bound the entire GET,
                    # including a response body that arrives in small chunks.
                    async with asyncio.timeout(cap):
                        response = await client.get(path, params=params, timeout=cap)
                    if clock() >= attempt_deadline:
                        raise TimeoutError
                    result = read_response(response, tool, topic, account_ref, attempts)
                    retry = response.status_code == 503
                except (httpx.TimeoutException, TimeoutError):
                    result = Unavailable(tool=tool, attempts=attempts, reason="timeout")
                    retry = True
                except httpx.RequestError:
                    result = Unavailable(tool=tool, attempts=attempts, reason="connection_error")
                    retry = False

                remaining = deadline - clock()
                if remaining <= 0:
                    break
                if not retry or attempts == 2:
                    return result
                if remaining <= BACKOFF_SECONDS:
                    break
                await sleep(BACKOFF_SECONDS)
    except TimeoutError:
        pass
    return Unavailable(tool=tool, attempts=attempts, reason="deadline_exceeded")
