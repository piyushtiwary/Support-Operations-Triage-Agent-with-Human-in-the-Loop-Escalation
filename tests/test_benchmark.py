from copy import deepcopy

import pytest

from benchmarks.prepare import DEFAULT_MANIFEST, inventory, json_lines, load_manifest, read_json
from benchmarks.run import replay
from benchmarks.scoring import effort, quality, ratio, review_signature, score


def test_frozen_inventory_matches_release():
    assert inventory() == read_json(DEFAULT_MANIFEST)
    paths = load_manifest(DEFAULT_MANIFEST)
    cases = json_lines(paths["cases"])
    reviews = json_lines(DEFAULT_MANIFEST.with_name("wording.jsonl"))
    assert len(reviews) == len(cases) == len({r["id"] for r in reviews}) == 64
    assert {r["id"] for r in reviews} == {c["id"] for c in cases}


def hand_scored():
    # Independent of the classifier and the release verdict file.
    case = {"id": "good", "account_ref": "A17"}
    gold = {"rubric": "card", "group": "public", "risks": [], "category": "delivery"}
    rubric = {"action": "prepare_reply", "evidence_id": "card-A17@v1", "status": "dispatched",
              "fact": "Your card was dispatched.", "next_step": "Check tracking."}
    evidence = {"source_id": "card-A17", "revision": "v1", "account_ref": "A17", "topic": "delivery",
                "status": "dispatched"}
    good = {"recommended_action": "prepare_reply", "category": "delivery", "evidence_ids": ["card-A17@v1"],
            "draft": "Your card was dispatched. Check tracking."}
    escalation = {"recommended_action": "request_review", "category": "delivery", "draft": None}
    wrong = good | {"draft": good["draft"] + " It will arrive tomorrow."}
    rows = []
    for name, decision in (("good", good), ("unnecessary", escalation), ("missed", good), ("incorrect", wrong)):
        missed = name == "missed"
        expected = gold | {"category": "security", "risks": ["security"]} if missed else gold
        contract = {"action": "request_review", "next_step": "Verify the incident."} if missed else rubric
        obs = {"decision": decision, "queue_items": int(name == "unnecessary"), "lookup": {"evidence": evidence}}
        verdict = {"reviewer": "test", "signature": review_signature(decision), "passes": name == "good"}
        rows.append(score(case | {"id": name}, expected, contract, obs, verdict))
    return rows


def test_hand_scored_quality_and_exclusive_effort():
    rows = hand_scored()
    stats = quality(rows)
    assert stats["correct_routine_drafts"] == stats["draft_precision"] == stats["unnecessary_reviews"] == ratio(1, 3)
    assert stats["risk_reviews"] == ratio(0, 1)
    assert stats["by_risk"]["mixed"] == {"numerator": 0, "denominator": 0, "percent": None}
    assert stats["durable_reviews"] == ratio(1, 1)
    assert stats["case_passes"] == ratio(1, 4)
    times = effort(rows)
    assert times["case_mix"] == {"verification": 1, "escalation": 1, "rework": 2}
    assert times["base"]["manual_seconds"] == times["slower_review"]["manual_seconds"] == 480
    assert (times["base"]["assisted_seconds"], times["base"]["saving_percent"]) == (420, 12.5)
    assert (times["slower_review"]["assisted_seconds"], times["slower_review"]["saving_percent"]) == (480, 0)


def test_no_decision_missing_queue_and_changed_evidence_fail_closed():
    case = {"id": "interrupted", "account_ref": "A17"}
    gold = {"rubric": "security", "group": "public", "risks": ["security"], "category": "security"}
    rubric = {"action": "request_review", "next_step": "Investigate."}
    obs = {"decision": None, "lookup": None, "queue_items": 0}
    failed = score(case, gold, rubric, obs, None)
    assert quality([failed])["no_decision"] == 1
    assert quality([failed])["risk_reviews"] == ratio(0, 1)
    times = effort([failed])
    assert times["base"]["assisted_seconds"] == times["slower_review"]["assisted_seconds"] == 150
    assert times["base"]["saving_percent"] == -25
    lost = score(case, gold, rubric, obs | {"decision": {"recommended_action": "request_review",
                 "category": "security"}}, None)
    assert "queue_mismatch" in lost["issues"] and not lost["queued"] and lost["effort_bucket"] == "rework"

    good = deepcopy(hand_scored()[0])
    decision = good["decision"]
    contract = {"action": "prepare_reply", "evidence_id": "card-A17@v1", "status": "dispatched",
                "fact": "Your card was dispatched.", "next_step": "Check tracking."}
    good["lookup"]["evidence"]["status"] = "processing"
    gold = gold | {"category": "delivery", "risks": []}
    row = score(case, gold, contract, {"decision": decision, "queue_items": 0, "lookup": good["lookup"]},
                good["review"])
    assert "wrong_evidence" in row["issues"] and not row["usable"]
    assert "unreviewed_wording" in score(case, gold, contract, obs | {
        "decision": decision | {"draft": "Different copy."}}, good["review"])["issues"]


@pytest.mark.anyio
async def test_api_replay_is_reproducible_with_fresh_databases(tmp_path):
    paths = load_manifest(DEFAULT_MANIFEST)
    selected = {"reliability-02", "reliability-17", "reliability-21"}
    cases = [c for c in json_lines(paths["cases"]) if c["id"] in selected]
    first, _ = await replay(cases, tmp_path / "first.db")
    second, _ = await replay(cases, tmp_path / "second.db")
    assert first == second
    assert first["reliability-02"]["decision"]["evidence_ids"] == ["card-delivery@v1"]
    assert first["reliability-21"]["queue_items"] == 1 and first["reliability-21"]["attempts"] == []
    outage = first["reliability-17"]
    assert outage["queue_items"] == 1 and outage["decision"]["draft"] is None
    assert [a["http_status"] for a in outage["attempts"]] == [503, 503]
