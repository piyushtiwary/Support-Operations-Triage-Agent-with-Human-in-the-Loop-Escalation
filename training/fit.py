"""Fit the local classifier and select its review threshold on development data."""

import csv
import json
import pickle
import re
from typing import get_args

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from classification import MODEL_PATH, ModelCategory, normalize
from training.prepare import DATA, SEED

LABELS = [*get_args(ModelCategory), "unknown"]
KEYWORDS = {
    "security": ["stolen", "lost", "compromised", "fraud", "someone used"],
    "billing_dispute": ["twice", "extra charge", "charged extra", "duplicate", "double"],
    "transfer": ["transfer", "recipient", "beneficiary"],
    "delivery": ["delivery", "arrive", "arrived", "dispatch", "tracking", "where is my card"],
}
# Project-authored development negatives, separate from BANKING77 and the final fixtures.
UNSUPPORTED_DEV = [
    "Can I get a quote for home insurance?",
    "I want to open a mortgage account",
    "Can you recommend a retirement savings plan?",
    "How do I change the colour of my debit card?",
    "Is there a fee to trade shares?",
    "Does the app support cryptocurrency trading?",
    "Can my credit limit be increased?",
    "How do I book a hotel with reward points?",
]


def baseline(text):
    text = normalize(text)
    for category, phrases in KEYWORDS.items():
        if any(re.search(r"\b" + phrase + r"\b", text) for phrase in phrases):
            return category
    return "unknown"


def read_split(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fit(training):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("logistic", LogisticRegression(C=4, max_iter=500, random_state=SEED)),
    ])
    return pipeline.fit([row["text"] for row in training], [row["category"] for row in training])


def summarize(gold, predictions):
    counts = confusion_matrix(gold, predictions, labels=LABELS)
    return {
        "confusion_counts": {
            label: dict(zip(LABELS, row.tolist())) for label, row in zip(LABELS, counts)
        },
        "correct": sum(a == b for a, b in zip(gold, predictions, strict=True)),
        "total": len(gold),
        "macro_f1_four_categories": f1_score(gold, predictions, labels=LABELS[:-1],
                                            average="macro", zero_division=0),
    }


def main():
    train = read_split("train")
    dev = read_split("dev")
    pipeline = fit(train)
    texts = [row["text"] for row in dev]
    gold = [row["category"] for row in dev]
    features = pipeline.named_steps["tfidf"].transform(texts + UNSUPPORTED_DEV)
    probabilities = pipeline.named_steps["logistic"].predict_proba(features)
    raw = pipeline.classes_[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    nonempty = features.getnnz(axis=1) > 0
    calibration_gold = gold + ["unknown"] * len(UNSUPPORTED_DEV)

    def predictions(threshold):
        return [label if present and score >= threshold else "unknown"
                for present, label, score in zip(nonempty, raw, scores, strict=True)]

    candidates = {}
    for percent in range(30, 96):
        threshold = percent / 100
        candidates[threshold] = f1_score(calibration_gold, predictions(threshold),
                                         labels=LABELS, average="macro", zero_division=0)
    threshold = max(candidates, key=candidates.get)
    chosen = predictions(threshold)
    report = {
        "sklearn_version": sklearn.__version__, "training_count": len(train),
        "threshold": threshold, "threshold_selection": {
            "criterion": "Maximize five-label macro-F1 on BANKING77 dev plus authored negatives. "
                         "Search 0.30 to 0.95 in 0.01 steps and break ties toward lower thresholds.",
            "macro_f1": candidates[threshold], "authored_negatives": len(UNSUPPORTED_DEV),
            "negatives_classified_unknown": chosen[len(dev):].count("unknown"),
        },
        "keyword_baseline": summarize(gold, [baseline(text) for text in texts]),
        "raw_model": summarize(gold, predictions(0)[:len(dev)]),
        "thresholded_model": summarize(gold, chosen[:len(dev)]),
    }
    MODEL_PATH.write_bytes(pickle.dumps((pipeline, threshold), protocol=5))
    (MODEL_PATH.parent / "development.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
