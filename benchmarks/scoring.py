"""Apply the frozen draft contract and account for each operator action exactly once."""

import hashlib
import json
from collections import Counter


def review_signature(decision):
    # A different score with identical presented wording does not need a new reading.
    content = {k: v for k, v in (decision or {}).items() if k != "confidence_score"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score(case, gold, rubric, observation, review):
    decision = observation["decision"] or {}
    action = decision.get("recommended_action")
    drafted = action == "prepare_reply"
    queued = action == "request_review" and observation["queue_items"] == 1
    issues = []
    if action != rubric["action"]:
        issues.append("wrong_route" if action else "no_decision")
    if decision.get("category") != gold["category"]:
        issues.append("wrong_category")
    if observation["queue_items"] != int(action == "request_review"):
        issues.append("queue_mismatch")
    if drafted:
        evidence = (observation["lookup"] or {}).get("evidence") or {}
        ref = f"{evidence.get('source_id')}@{evidence.get('revision')}"
        if "status" in rubric:
            supported = (evidence.get("account_ref") == case["account_ref"]
                         and evidence.get("status") == rubric["status"] and evidence.get("topic") == gold["category"])
        else:
            supported = (evidence.get("guidance") == rubric.get("fact")
                         and evidence.get("approved_next_step") == rubric["next_step"])
        if (decision["evidence_ids"] != [ref] or ref != rubric.get("evidence_id") or not supported):
            issues.append("wrong_evidence")
        # The release has a closed vocabulary of approved facts and instructions.
        # Exact sentences also exclude added dates, guarantees and unsupported promises.
        if decision["draft"] != f"{rubric.get('fact')} {rubric['next_step']}":
            issues.append("unsupported_or_incomplete_copy")
    if not review or review["signature"] != review_signature(decision):
        issues.append("unreviewed_wording")
        review = None
    elif not review["passes"]:
        issues.append("wording_or_omission")
    usable = drafted and not issues
    if not action or (drafted and not usable) or (action == "request_review" and not queued):
        bucket = "rework"
    elif queued:
        bucket = "escalation"
    else:
        bucket = "verification"
    return {
        "id": case["id"], "cohort": "public" if gold["group"] == "public" else "reliability",
        "group": gold["group"], "gold_category": gold["category"], "gold_action": rubric["action"],
        "rubric": gold["rubric"], "risks": gold["risks"], "fault": case.get("fault"),
        **observation, "review": review, "issues": issues, "drafted": drafted, "queued": queued,
        "usable": usable, "effort_bucket": bucket,
    }


def ratio(n, d):
    return {"numerator": n, "denominator": d, "percent": round(100 * n / d, 2) if d else None}


def quality(rows):
    routine = [r for r in rows if r["gold_action"] == "prepare_reply"]
    drafts = [r for r in rows if r["drafted"]]
    risks = [r for r in rows if r["risks"]]
    reviews = [r for r in rows if (r["decision"] or {}).get("recommended_action") == "request_review"]
    faults = [r for r in rows if r["fault"]]
    def risk_capture(group):
        return ratio(sum(r["queued"] for r in group), len(group))
    return {
        "correct_routine_drafts": ratio(sum(r["usable"] for r in routine), len(routine)),
        "draft_precision": ratio(sum(r["usable"] for r in drafts), len(drafts)),
        "risk_reviews": risk_capture(risks),
        "by_risk": {label: risk_capture([r for r in risks if label in r["risks"]])
                    for label in ("billing_dispute", "security", "data_loss", "mixed")},
        "unnecessary_reviews": ratio(sum((r["decision"] or {}).get("recommended_action") == "request_review"
                                        for r in routine), len(routine)),
        "durable_reviews": risk_capture(reviews),
        # All four injected faults yield no valid evidence. Any customer draft
        # in these cases conservatively fails the unsupported-claim gate.
        "fault_drafts_without_evidence": ratio(sum(r["drafted"] and not r["usable"] for r in faults), len(faults)),
        "case_passes": ratio(sum(not r["issues"] for r in rows), len(rows)),
        "no_decision": sum(r["decision"] is None for r in rows),
    }


def effort(rows):
    mix = Counter(r["effort_bucket"] for r in rows)
    manual = 120 * len(rows)
    result = {"case_mix": {k: mix[k] for k in ("verification", "escalation", "rework")}}
    for name, verify, escalate in (("base", 20, 100), ("slower_review", 30, 150)):
        assisted = mix["verification"] * verify + mix["escalation"] * escalate + mix["rework"] * 150
        result[name] = {"seconds_per_case": {"manual": 120, "verification": verify, "escalation": escalate,
                                            "rework": 150}, "manual_seconds": manual, "assisted_seconds": assisted,
                        "saving_percent": ratio(manual - assisted, manual)["percent"]}
    return result
