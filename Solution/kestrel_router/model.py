"""Model definitions: every candidate is an sklearn Pipeline that takes raw
request records (request_text + channel/product/warranty) and returns one of
the seven teams.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC

from .data import META_COLS
from .features import PolicyFlags, add_derived

SEED = 42


def _features(policy: bool) -> ColumnTransformer:
    """A = full-text word + char n-grams + metadata.
    B2 adds the last clause as its own text input plus the policy flags."""
    blocks = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text_clean"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=40000), "text_clean"),
        ("meta", OneHotEncoder(handle_unknown="ignore"), META_COLS),
    ]
    if policy:
        blocks += [
            ("last", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "last_clause"),
            ("policy", PolicyFlags(), ["text_clean", "last_clause", "product_family"]),
        ]
    return ColumnTransformer(blocks, sparse_threshold=1.0)


def _classifier(name: str):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(C=4.0, max_iter=3000)
    if name == "linsvm":
        # Calibrated so it also returns probabilities (needed for approach C).
        return CalibratedClassifierCV(LinearSVC(C=0.5), method="sigmoid", cv=3)
    if name == "nb":
        return ComplementNB(alpha=0.3)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=10, colsample_bytree=0.3,
                              random_state=SEED, verbose=-1, n_jobs=-1)
    raise ValueError(name)


CANDIDATES = ["majority", "nb", "logreg", "linsvm", "lgbm"]


def build(name: str, policy: bool = False) -> Pipeline:
    return Pipeline([
        ("prep", FunctionTransformer(add_derived)),
        ("features", _features(policy)),
        ("clf", _classifier(name)),
    ])
