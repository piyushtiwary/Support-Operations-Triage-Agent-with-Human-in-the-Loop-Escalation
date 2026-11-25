"""Local, provisional intent classification for support tickets."""

import logging
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ModelCategory = Literal["delivery", "transfer", "billing_dispute", "security"]
MODEL_PATH = Path(__file__).resolve().parent / "data" / "classifier.pkl"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


# Exact, whole-ticket variants of one project-authored mobile app issue.
BUG_PHRASES = {
    "the mobile app freezes on the card tracking screen": "card-tracking-freeze-v1",
    "my app freezes when i open card tracking": "card-tracking-freeze-v1",
}


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    category: ModelCategory | Literal["technical_issue", "unknown"]
    raw_category: ModelCategory | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    rule_id: str | None = None
    reason: Literal["model_prediction", "below_threshold", "empty_features",
                    "known_bug", "model_unavailable"]


@dataclass(frozen=True)
class Classifier:
    pipeline: Pipeline
    threshold: float

    @classmethod
    def load(cls, path: Path) -> "Classifier | None":
        try:
            # The application owns this local artifact. Ticket data never selects a path.
            with path.open("rb") as stream:
                pipeline, threshold = pickle.load(stream)
            if (
                not isinstance(pipeline, Pipeline)
                or not isinstance(pipeline.named_steps.get("tfidf"), TfidfVectorizer)
                or not isinstance(pipeline.named_steps.get("logistic"), LogisticRegression)
                or len(pipeline["tfidf"].vocabulary_) != pipeline["logistic"].n_features_in_
                or set(pipeline.classes_) != set(get_args(ModelCategory))
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("Invalid classification artifact")
            return cls(pipeline, threshold)
        except Exception:
            logging.getLogger(__name__).exception("Could not load local intent classifier")
            return None

    def predict(self, text: str) -> Classification:
        features = self.pipeline.named_steps["tfidf"].transform([text])
        if not features.nnz:
            return Classification(category="unknown", reason="empty_features")
        estimator = self.pipeline.named_steps["logistic"]
        scores = estimator.predict_proba(features)[0]
        index = scores.argmax()
        raw = str(estimator.classes_[index])
        score = float(scores[index])
        accepted = score >= self.threshold
        return Classification(
            category=raw if accepted else "unknown",
            raw_category=raw,
            confidence_score=score,
            reason="model_prediction" if accepted else "below_threshold",
        )


def classify(text: str, model: Classifier | None) -> Classification:
    rule = BUG_PHRASES.get(normalize(text))
    if rule:
        return Classification(category="technical_issue", rule_id=rule, reason="known_bug")
    if model is None:
        return Classification(category="unknown", reason="model_unavailable")
    return model.predict(text)
