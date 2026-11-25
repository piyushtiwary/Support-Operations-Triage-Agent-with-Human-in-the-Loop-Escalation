"""Evaluate the frozen public slice and replay support tickets through the production API."""

import argparse
import asyncio
import json
import os
import platform
import sqlite3
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import numpy as np

from benchmarks.prepare import DEFAULT_MANIFEST, digest, json_lines, load_manifest, read_json, split_rows
from benchmarks.scoring import effort, quality, ratio, score
from classification import MODEL_PATH, Classifier, classify
from context_api import STATUS_RECORDS, app as context_app
from main import create_app
from policy import RULESET_VERSION
from training.fit import LABELS, baseline, summarize


class SeededContext(httpx.AsyncBaseTransport):
    """Only the fault selector reaches the transport. Healthy calls hit the real seeded routes."""

    def __init__(self):
        self.healthy = httpx.ASGITransport(app=context_app)
        self.fault = None

    async def handle_async_request(self, request):
        if self.fault == "503":
            return httpx.Response(503)
        if self.fault == "timeout":
            raise httpx.ReadTimeout("Seeded timeout", request=request)
        if self.fault == "malformed":
            return httpx.Response(200, content=b"{")
        if self.fault == "wrong_account":
            return httpx.Response(200, json=STATUS_RECORDS["B42", "delivery"].model_dump())
        if self.fault is not None:
            raise ValueError(f"Unknown fault: {self.fault}")
        return await self.healthy.handle_async_request(request)


def classify_public(test, model):
    rows = []
    for item in test:
        result = classify(item["text"], model)
        rows.append({"id": item["id"], "intent": item["intent"], "gold": item["category"],
                     "keyword": baseline(item["text"]), "model": result.category, "raw": result.raw_category,
                     "confidence_score": result.confidence_score, "reason": result.reason})
    gold = [r["gold"] for r in rows]
    metrics = {}
    for name in ("keyword", "raw", "model"):
        stats = summarize(gold, [r[name] or "unknown" for r in rows])
        counts = stats["confusion_counts"]
        stats["accuracy"] = ratio(stats["correct"], len(rows))
        stats["per_class"] = {}
        for label in LABELS[:-1]:
            support = sum(counts[label].values())
            stats["per_class"][label] = {
                "support": support, "recall": ratio(counts[label][label], support),
                "precision": ratio(counts[label][label], sum(line[label] for line in counts.values())),
            }
        metrics[name] = stats
    return rows, metrics


async def replay(cases, database):
    transport = SeededContext()
    app = create_app(database, context_transport=transport, context_base_url="http://seeded")
    observed, elapsed = {}, {}
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://triage"
    ) as client:
        for case in cases:
            transport.fault = case.get("fault")
            result = await client.post("/tickets", json={"external_id": case["id"], "text": case["text"],
                                                         "account_ref": case["account_ref"]})
            result.raise_for_status()
            url = "/tickets/" + result.json()["id"]
            start = time.perf_counter()
            response = await client.post(url + "/triage")
            elapsed[case["id"]] = (time.perf_counter() - start) * 1000
            ticket = (await client.get(url)).raise_for_status().json()
            runs = (await client.get(url + "/runs")).raise_for_status().json()
            decision = ticket.get("decision")
            if response.is_success and response.json() != decision:
                raise ValueError(f"API and persisted decision disagree for {case['id']}")
            run = runs[-1] if runs else {}
            observed[case["id"]] = {
                "http_status": response.status_code, "state": ticket["state"], "decision": decision,
                "classification": ticket.get("classification"), "lookup": ticket.get("lookup"),
                "error": run.get("error_code"), "usage": run.get("usage"),
                "attempts": [{k: v for k, v in a.items() if k != "elapsed_ms"} for a in run.get("attempts", [])],
            }
        pending = (await client.get("/reviews?state=pending")).raise_for_status().json()
        queue = Counter(item["external_id"] for item in pending)
        for case_id, row in observed.items():
            row["queue_items"] = queue[case_id]
    return observed, elapsed


def save_lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


async def evaluate(manifest, output, reviews_path):
    paths = load_manifest(manifest)
    test = split_rows(paths["test"])
    texts = {row["id"]: row["text"] for row in test}
    cases = [c if "text" in c else {**c, "text": texts[c["id"]]} for c in json_lines(paths["cases"])]
    gold = {k["id"]: k for k in json_lines(paths["key"])}
    rubrics = read_json(paths["rubrics"])
    reviews = {r["id"]: r for r in json_lines(reviews_path)}
    model = Classifier.load(MODEL_PATH)
    if model is None:
        raise ValueError("Cannot evaluate without the frozen classifier")
    predictions, classification = classify_public(test, model)
    timings = defaultdict(list)
    with TemporaryDirectory(prefix="triage-benchmark-") as temp:
        for index in range(6):
            observations, elapsed = await replay(cases, Path(temp) / f"pass-{index}.sqlite3")
            if index == 0:
                reference = observations
                continue
            if observations != reference:
                changed = [k for k in reference if observations[k] != reference[k]]
                raise ValueError(f"Non-timing outputs changed in pass {index}: {changed}")
            for case_id, duration in elapsed.items():
                timings[case_id].append(duration)
    rows = [score(c, gold[c["id"]], rubrics[gold[c["id"]]["rubric"]], reference[c["id"]], reviews.get(c["id"]))
            for c in cases]
    cohorts = {name: [r for r in rows if r["cohort"] == name] for name in ("public", "reliability")}
    latency = {}
    for faulted in (False, True):
        samples = [ms for row in rows if bool(row["fault"] and row["attempts"]) == faulted
                   for ms in timings[row["id"]]]
        latency["faulted" if faulted else "healthy"] = {
            "n": len(samples), "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95))}
    usage = [r["usage"] for r in rows if r["usage"] is not None]
    cost = sum(u["estimated_provider_cost_usd"] for u in usage)
    summary = {
        "manifest_sha256": digest(manifest), "wording_reviews_sha256": digest(reviews_path),
        "seed": read_json(manifest)["seed"], "revision": read_json(manifest)["revision"],
        "classifier_version": model.version, "threshold": model.threshold, "ruleset": RULESET_VERSION,
        "environment": {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
                        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                        "sqlite": sqlite3.sqlite_version, "packages": {name: version(name) for name in
                        ("scikit-learn", "numpy", "scipy", "fastapi", "pydantic", "httpx", "uvicorn", "pytest")}},
        "classification": classification, "quality": {k: quality(v) for k, v in cohorts.items()},
        "public_effort": effort(cohorts["public"]), "warmup_passes": 1, "measured_passes": 5, "latency": latency,
        "usage": {"tickets_with_audit": len(usage), "tickets": len(rows),
                  **{k: sum(u[k] for u in usage) for k in ("provider_calls", "input_tokens", "output_tokens")},
                  "total_usd": cost, "usd_per_audited_ticket": cost / len(usage) if usage else None},
    }
    output.mkdir(parents=True, exist_ok=True)
    save_lines(output / "classification.jsonl", predictions)
    save_lines(output / "cases.jsonl", [r | {"latency_ms": timings[r["id"]]} for r in rows])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(predictions)} classification rows, {len(rows)} workflow rows. Results in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviews", type=Path, help="Case-by-case wording verdicts for these decisions")
    parser.add_argument("--output", type=Path, default=Path("benchmark-output"))
    args = parser.parse_args()
    asyncio.run(evaluate(args.manifest, args.output, args.reviews or args.manifest.with_name("wording.jsonl")))
