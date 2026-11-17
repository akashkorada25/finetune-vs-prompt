"""Classic baseline: TF-IDF features (words and character n-grams) + logistic regression."""
from __future__ import annotations

import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Example


def build() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=3000))])


def train(examples: list[Example]) -> tuple[Pipeline, float]:
    """Returns the fitted model and the training time in seconds."""
    model = build()
    start = time.perf_counter()
    model.fit([e.text for e in examples], [e.label for e in examples])
    return model, time.perf_counter() - start
