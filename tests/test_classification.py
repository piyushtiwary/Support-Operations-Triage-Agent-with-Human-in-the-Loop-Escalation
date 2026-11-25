import asyncio
import json
import sqlite3
from collections import Counter
from contextlib import closing

import numpy as np
import pytest
from pydantic import ValidationError

from classification import MODEL_PATH, Classification, Classifier, classify, normalize
from training.fit import baseline, fit, read_split, summarize
from training.prepare import CATEGORIES, DATA


@pytest.fixture(scope="module")
def model():
    classifier = Classifier.load(MODEL_PATH)
    assert classifier is not None
    return classifier


def test_frozen_splits_and_pipeline(model):
    train, dev, test = (read_split(name) for name in ("train", "dev", "test"))
    assert all(r["id"].startswith("train-") for r in train + dev)
    assert all(r["id"].startswith("test-") for r in test)
    assert len({r["id"] for r in train + dev + test}) == len(train + dev + test)
    assert Counter(r["intent"] for r in test) == {intent: 40 for intent in CATEGORIES}
    for row in train + dev + test:
        assert CATEGORIES[row["intent"]] == row["category"]
    train_keys, dev_keys, test_keys = ({normalize(r["text"]) for r in rows}
                                      for rows in (train, dev, test))
    assert train_keys.isdisjoint(dev_keys | test_keys) and dev_keys.isdisjoint(test_keys)

    refit = fit(train)
    texts = [r["text"] for r in dev]
    assert refit["tfidf"].vocabulary_ == model.pipeline["tfidf"].vocabulary_
    np.testing.assert_allclose(refit.predict_proba(texts), model.pipeline.predict_proba(texts),
                               rtol=1e-7, atol=1e-9)
    report = json.loads((DATA.parent / "development.json").read_text())
    gold = [r["category"] for r in dev]
    assert report["training_count"] == len(train)
    assert report["threshold"] == model.threshold
    assert report["keyword_baseline"] == summarize(gold, [baseline(t) for t in texts])
    assert report["thresholded_model"] == summarize(gold, [classify(t, model).category for t in texts])


@pytest.mark.parametrize(("text", "label"), [
    ("Where is my card?", "delivery"),
    ("How long does a bank transfer take?", "transfer"),
    ("My card was stolen", "security"),
    ("I was charged twice for a purchase", "billing_dispute"),
])
def test_model_predictions(model, text, label):
    prediction = classify(text, model)
    assert prediction.category == prediction.raw_category == label
    assert model.threshold <= prediction.confidence_score <= 1
    assert prediction.rule_id is None


@pytest.mark.parametrize("score", [-0.001, 1.001, float("nan")])
def test_score_validation(score):
    with pytest.raises(ValidationError):
        Classification(category="delivery", reason="model_prediction", confidence_score=score)


def test_invalid_label():
    with pytest.raises(ValidationError):
        Classification(category="mortgage", reason="model_prediction")


def test_uncertainty_and_exact_rule(model):
    empty = classify("xqzv !!!", model)
    assert (empty.category, empty.reason, empty.raw_category, empty.confidence_score) == (
        "unknown", "empty_features", None, None
    )
    phrase = "  My APP freezes when I open card tracking! "
    known = classify(phrase, None)
    assert (known.category, known.reason, known.rule_id, known.confidence_score) == (
        "technical_issue", "known_bug", "card-tracking-freeze-v1", None
    )
    assert classify(phrase + "and my card was stolen", model).rule_id is None


@pytest.mark.anyio
@pytest.mark.parametrize(("text", "category"), [
    ("How long does a bank transfer take?", "transfer"),
    ("Can I buy travel insurance?", "unknown"),
    ("Where is my card?", "delivery"),
    ("My app freezes when I open card tracking.", "technical_issue"),
    # This mixed ticket currently predicts delivery. Risk routing belongs to step 4.
    ("My card hasn't arrived and someone used it", "delivery"),
])
async def test_triage_replay_after_restart(database, running_client, text, category):
    submission = {"external_id": "scenario", "text": text}
    async with running_client(database) as client:
        assert (await client.post("/tickets/missing/triage")).status_code == 404
        ticket = (await client.post("/tickets", json=submission)).json()
        url = f"/tickets/{ticket['id']}"
        responses = await asyncio.gather(*(client.post(url + "/triage") for _ in range(4)))
        prediction = responses[0].json()
        assert all(r.status_code == 200 and r.json() == prediction for r in responses)
        assert prediction["category"] == category
        if category == "unknown":
            assert prediction["reason"] == "below_threshold"
            assert prediction["raw_category"] in CATEGORIES.values()
            assert prediction["confidence_score"] is not None

    async with running_client(database, model_path=database.with_suffix(".missing")) as restarted:
        stored = (await restarted.get(url)).json()
        assert {key: value for key, value in stored.items() if key != "lookup"} == {
            **ticket, "state": "classified", "classification": prediction
        }
        assert (await restarted.post(url + "/triage")).json() == prediction
        assert (await restarted.post("/tickets", json=submission)).json() == stored


@pytest.mark.anyio
async def test_existing_database_and_failed_model_recovery(database, tmp_path, running_client):
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("""CREATE TABLE tickets (id TEXT PRIMARY KEY, external_id TEXT UNIQUE,
                      text TEXT, account_ref TEXT, state TEXT DEFAULT 'received')""")
        db.execute("""INSERT INTO tickets (id, external_id, text)
                      VALUES ('old', 'legacy', 'How long does a bank transfer take?')""")
    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"invalid model")
    url = "/tickets/old"
    async with running_client(database, model_path=corrupt) as broken:
        prediction = (await broken.post(url + "/triage")).json()
        assert prediction["category"] == "unknown" and prediction["reason"] == "model_unavailable"
        assert prediction["confidence_score"] is None
        assert (await broken.get(url)).json()["classification"] == prediction
    async with running_client(database) as repaired:
        prediction = (await repaired.post(url + "/triage")).json()
        assert prediction["category"] == "transfer"
        assert (await repaired.get(url)).json()["classification"] == prediction
