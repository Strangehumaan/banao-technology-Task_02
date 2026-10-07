"""Turn a prediction into reasons a Kestrel service-desk employee can read."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_derived, content_clauses, intent_flags, is_vague, text_product

CLARIFY_QUESTION = (
    "To get you to the right team, is this about: 1) something not working, "
    "2) installation or a demo, 3) a payment, invoice or refund, 4) a damaged, wrong "
    "or incomplete delivery or a return, 5) warranty or Kestrel Shield, 6) filters or "
    "spare parts, or 7) how to use the product?"
)

FLAG_TEXT = {
    "paid_not_payment": "Mentions paying, but the problem is not the payment itself, so not Billing (policy §3).",
    "vague": "The message doesn't say what the problem is.",
    "two_intents": "Two requests in one message; these are closed by the team for the last one.",
    "product_mismatch": "Text names a different product than the product field; the text was used.",
    "last_fault": "Describes a fault (needs a technician).",
    "last_spares": "Asks about filters / spare parts.",
    "last_install": "About installation, demo or wall-mounting.",
    "last_return": "About a damaged, wrong or incomplete delivery, or a return.",
    "last_warranty": "About warranty / Kestrel Shield.",
    "last_advice": "A usage or buying question, no fault reported.",
    "last_payment_problem": "The problem is the payment itself (invoice, GST, refund, EMI, coupon).",
}


def _feature_names(pipe) -> np.ndarray:
    return pipe.named_steps["features"].get_feature_names_out()


def _coef(pipe) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return clf.coef_
    # CalibratedClassifierCV(LinearSVC): average the fold estimators.
    return np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)


# Words that never explain a routing decision on their own.
_UNINFORMATIVE = set(
    "a an and the of for to in on my is it this with me i you be not pls please call back "
    "asap thanks thank kindly resolve very disappointed hi hello team sir namaste good morning "
    "urgent help idnum order reg no upi paid via card in full already emi by done payment "
    "air fryer mixer grinder water purifier induction cooktop room heater ceiling fan robot "
    "vacuum product machine appliance did up from regarding about need want".split()
)
# Flags already covered by the notes written in explain().
_NOTE_FLAGS = {"paid_not_payment", "vague", "two_intents", "product_mismatch"}


def top_drivers(pipe, record: pd.DataFrame, cls_idx: int, k_words: int = 3) -> list[str]:
    """The intent flags and key words that pushed this request to the predicted team."""
    X = pipe.named_steps["features"].transform(pipe.named_steps["prep"].transform(record))
    x = np.asarray(X.todense()).ravel() if hasattr(X, "todense") else np.asarray(X).ravel()
    coef = _coef(pipe)
    # Contribution relative to the average class: what makes THIS team win.
    contrib = x * (coef[cls_idx] - coef.mean(axis=0))
    names = _feature_names(pipe)
    intents, words = [], []
    for i in np.argsort(-contrib):
        if contrib[i] <= 0:
            break
        block, _, feat = names[i].partition("__")
        if block == "policy" and feat in FLAG_TEXT and feat not in _NOTE_FLAGS:
            if FLAG_TEXT[feat] not in intents:
                intents.append(FLAG_TEXT[feat])
        elif block in ("word", "last") and len(words) < k_words:
            # character n-grams and one-hot metadata are skipped: not readable
            if any(tok not in _UNINFORMATIVE for tok in feat.split()):
                if not any(feat in w or w in feat for w in words):
                    words.append(feat)
    out = intents[:2]
    if words:
        out.append("Key words: " + ", ".join(f'"{w}"' for w in words))
    return out


def explain(pipe, record: pd.DataFrame, threshold: float) -> dict:
    """Full response for one request (a single-row DataFrame of raw fields)."""
    proba = pipe.predict_proba(record)[0]
    classes = list(pipe.classes_)
    order = np.argsort(-proba)
    best, second = order[0], order[1]
    conf = float(proba[best])

    d = add_derived(record).iloc[0]
    text, last = d["text_clean"], d["last_clause"]
    flags = intent_flags(text)
    notes = []
    if is_vague(text):
        notes.append(FLAG_TEXT["vague"])
    if flags["paid_mention"] and not flags["payment_problem"]:
        notes.append(FLAG_TEXT["paid_not_payment"])
    if len(content_clauses(text)) > 1:
        notes.append(f'Two requests in one message; routed on the last one: "{last}".')
    tp = text_product(text)
    if tp and tp != record["product_family"].iloc[0]:
        notes.append(f"Text is about a {tp}, but the product field says "
                     f"{record['product_family'].iloc[0]}; the text was used.")

    clarify = conf < threshold
    return {
        "team": classes[best],
        "confidence": round(conf, 3),
        "action": "clarify_first" if clarify else "route",
        "runner_up": {"team": classes[second], "confidence": round(float(proba[second]), 3)},
        "reasons": notes + top_drivers(pipe, record, best),
        "clarifying_question": CLARIFY_QUESTION if clarify else None,
        "all_teams": {c: round(float(p), 3) for c, p in sorted(zip(classes, proba), key=lambda t: -t[1])},
    }
