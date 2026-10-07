"""Text structure and policy features (approach B2).

The request texts follow a pattern: optional greeting, one or two
"intents" separated by a comma, optional filler ("pls call back", "paid on
upi", an order number). The EDA showed three things a bag-of-words model
cannot see on its own:

1. When a request states two intents, the team that closes it follows the
   LAST one (~99% of two-intent rows), so word order matters.
2. "I paid" is not a billing problem (policy §3); the bot routes it to
   Billing anyway.
3. About 15% of requests name no intent at all ("please call back
   regarding my purifier"). Nobody can route these from the text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import clean_text

GREETING_RE = re.compile(
    r"^(?:hi|hello team|hello|sir|namaste|good morning|urgent|pls help|help)\s*[,:\-]*\s*"
)
FILLER_RE = re.compile(
    r"\b(?:order idnum|reg no idnum|idnum|pls call back|asap|thanks|thank you|kindly resolve|"
    r"very disappointed|paid on upi|already paid in full|paid in full|payment done|"
    r"paid by emi|paid via card|i paid extra for this)\b"
)
PRODUCT_RE = re.compile(
    r"\b(?:air fryer|fryer|mixer grinder|mixer|grinder|water purifier|purifier|"
    r"induction cooktop|cooktop|room heater|heater|ceiling fan|fan|robot vacuum|"
    r"vacuum|product|machine|appliance)\b"
)
TEXT_PRODUCTS = {
    "Water Purifier": r"purifier",
    "Air Fryer": r"fryer",
    "Mixer Grinder": r"mixer|grinder",
    "Induction Cooktop": r"cooktop|induction",
    "Room Heater": r"heater",
    "Ceiling Fan": r"\bfan\b",
    "Robot Vacuum": r"vacuum",
}

# Request "shapes" with no intent in them. Found in the EDA: each of these
# has a final-team purity of ~20% (close to random over 7 teams), versus
# ~98% for every other shape.
VAGUE_SHAPES = [
    "someone contact me about <p>",
    "service request for <p>",
    "complaint about <p>",
    "please call back regarding <p>",
    "issue with <p>",
    "not happy with <p>",
    "<p> query",
    "help <p>",
    "<p> problem",
    "need help with my <p>",
]

# Keyword groups taken from teams.csv "handles" and ops-policy §3. They are
# used both as model features and to write the reasons staff will read.
INTENTS = {
    "payment_problem": r"invoice|gst|double charg|charged twice|refund|emi conversion|coupon|"
                       r"payment deducted|corrected bill|wrong bill|cashback|payment failed",
    "paid_mention": r"\bpaid\b|payment done",
    "fault": r"leak|error code|not working|not turning on|noise|burnt|tripping|gone blank|"
             r"stopped working|smell|not heating|not cooling|not spinning",
    "spares": r"filter|candle|membrane|spare|\bjar\b|brush|blade|\bamc\b|consumable|\bkit\b",
    "install": r"install|demo\b|wall mounting|installer",
    "return": r"return|damaged|missing parts|wrong model|scratched|box was open|exchange|"
              r"replacement|cancel",
    "warranty": r"warranty|shield|claim",
    "advice": r"how to|which .* right|difference between|power consumption|safe for kids|"
              r"recipe|how much|how long",
}
INTENT_RES = {k: re.compile(v) for k, v in INTENTS.items()}


def strip_greeting(t: str) -> str:
    return GREETING_RE.sub("", t)


def content_clauses(t: str) -> list[str]:
    """Comma-separated parts of a cleaned request that say something.

    Parts that are only filler ("paid on upi", "pls call back") are dropped, so
    "installer did not turn up, paid on upi" counts as one request, not two.
    """
    parts = [p.strip() for p in strip_greeting(t).split(",") if p.strip()]
    kept = [p for p in parts if re.sub(r"[^a-z]", "", FILLER_RE.sub(" ", p))]
    return kept or parts


def last_clause(t: str) -> str:
    """The last meaningful comma-separated part of a cleaned request."""
    parts = content_clauses(t)
    return parts[-1] if parts else t


def shape(t: str) -> str:
    """Reduce a cleaned request to its shape: no greeting, filler or product."""
    t = strip_greeting(t)
    t = FILLER_RE.sub(" ", t)
    t = re.sub(r"[^a-z ]", " ", t)
    t = PRODUCT_RE.sub("<p>", t)
    return " ".join(t.split())


def is_vague(t: str) -> bool:
    return shape(t) in VAGUE_SHAPES


def text_product(t: str) -> str | None:
    hits = [p for p, rx in TEXT_PRODUCTS.items() if re.search(rx, t)]
    return hits[0] if len(hits) == 1 else None


def intent_flags(t: str) -> dict[str, bool]:
    return {k: bool(rx.search(t)) for k, rx in INTENT_RES.items()}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Columns every model variant uses. Works on raw records (service) too."""
    out = df.copy()
    if "text_clean" not in out:
        out["text_clean"] = out["request_text"].map(clean_text)
    out["last_clause"] = out["text_clean"].map(last_clause)
    return out


class PolicyFlags(BaseEstimator, TransformerMixin):
    """Binary features from the policy and the request structure.

    For each intent group: is it present anywhere, and is it present in the
    last clause. Plus: paid-but-not-a-payment-problem, vague, two-intent,
    and whether the product named in the text disagrees with product_family.
    """

    def fit(self, X, y=None):
        return self

    def _row(self, text, last, product_family):
        full = intent_flags(text)
        tail = intent_flags(last)
        f = {f"any_{k}": v for k, v in full.items()}
        f.update({f"last_{k}": v for k, v in tail.items()})
        f["paid_not_payment"] = full["paid_mention"] and not full["payment_problem"]
        f["vague"] = is_vague(text)
        f["two_intents"] = len(content_clauses(text)) > 1
        tp = text_product(text)
        f["product_mismatch"] = tp is not None and tp != product_family
        return f

    def transform(self, X):
        rows = [
            self._row(t, l, p)
            for t, l, p in zip(X["text_clean"], X["last_clause"], X["product_family"])
        ]
        self.feature_names_ = list(rows[0].keys()) if rows else []
        return np.array([[float(v) for v in r.values()] for r in rows])

    def get_feature_names_out(self, input_features=None):
        dummy = self._row("x", "x", "x")
        return np.array(list(dummy.keys()))
