"""Time-split evaluation, the clarify-first rule (approach C) and rupee maths.

All numbers here come from out-of-time predictions: a model is always
trained on months strictly before the months it is scored on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import data as D
from .features import is_vague
from .model import build

# ---- Costs (ops-policy §4 plus two stated assumptions) ----------------------
TRANSFER_RS = 305          # §4: handling time per transfer
EXTRA_CONTACT_RS = 260     # §4: extra customer contact per misroute
LICENCE_RS_YEAR = 320_000  # §4: vendor bot licence
REQUESTS_PER_MONTH = 720   # measured: 10,822 / 15 months (PS form says ~700 orders)
CLARIFY_RS = 60            # ASSUMPTION: one quick-reply question, ~1/5 of a transfer's handling
CLARIFY_RESIDUAL = 0.10    # ASSUMPTION: share of clarified requests still misrouted


def misroute_cost(df: pd.DataFrame) -> float:
    """Average rupee cost of one misrouted request, from the resolution log.

    A misroute (first != final) costs its transfers plus one extra contact.
    """
    wrong = df[df["first_team"] != df["final_team"]]
    wrong = wrong[wrong["transfers"] > 0]          # ignore the impossible rows
    return TRANSFER_RS * wrong["transfers"].mean() + EXTRA_CONTACT_RS


@dataclass
class FoldResult:
    name: str
    fold: tuple[str, str]
    frame: pd.DataFrame          # one row per eval request
    proba: np.ndarray            # class probabilities (or None)
    classes: list[str]
    fit_seconds: float


def run_fold(name: str, df: pd.DataFrame, fold, policy=False, drop_bad=False) -> FoldResult:
    train, ev = D.time_split(df, *fold)
    if drop_bad:
        train = D.drop_impossible(train)          # B1: training rows only
    pipe = build(name, policy=policy)
    t0 = time.perf_counter()
    pipe.fit(train, train["final_team"])
    secs = time.perf_counter() - t0
    pred = pipe.predict(ev)
    proba = pipe.predict_proba(ev) if hasattr(pipe, "predict_proba") else None
    frame = ev[["request_id", "request_text", "text_clean", "product_family", "channel",
                "warranty_status", "team_label", "first_team", "final_team",
                "transfers", "impossible", "pingpong"]].copy()
    frame["pred"] = pred
    frame["vague"] = frame["text_clean"].map(is_vague)
    if proba is not None:
        frame["confidence"] = proba.max(axis=1)
    return FoldResult(name, fold, frame, proba, list(pipe.classes_), secs)


def bot_frame(df: pd.DataFrame, fold) -> pd.DataFrame:
    _, ev = D.time_split(df, *fold)
    frame = ev[["request_id", "text_clean", "team_label", "final_team"]].copy()
    frame["pred"] = frame["team_label"]
    frame["vague"] = frame["text_clean"].map(is_vague)
    return frame


def scores(frame: pd.DataFrame) -> dict:
    y, p = frame["final_team"], frame["pred"]
    v = frame["vague"]
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "acc_clear": accuracy_score(y[~v], p[~v]),
        "acc_vague": accuracy_score(y[v], p[v]),
        "n": len(frame),
    }


def bootstrap_ci(frame: pd.DataFrame, n_boot=2000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    correct = (frame["final_team"] == frame["pred"]).to_numpy()
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))


# ---- Approach C: clarify first when confidence is low ----------------------
def clarify_threshold(m_cost: float, q_cost=CLARIFY_RS, residual=CLARIFY_RESIDUAL) -> float:
    """Ask instead of guess when expected misroute cost > cost of asking.

    Guess:  (1 - p) * M        Ask:  Q + residual * M
    Ask iff p < 1 - residual - Q / M.  Holds if probabilities are calibrated.
    """
    return 1 - residual - q_cost / m_cost


def cost_per_request(frame: pd.DataFrame, tau: float, m_cost: float,
                     q_cost=CLARIFY_RS, residual=CLARIFY_RESIDUAL) -> dict:
    ask = frame["confidence"] < tau if tau > 0 else pd.Series(False, index=frame.index)
    wrong = frame["final_team"] != frame["pred"]
    auto_wrong = (wrong & ~ask).sum()
    cost = auto_wrong * m_cost + ask.sum() * (q_cost + residual * m_cost)
    return {
        "tau": tau,
        "clarify_share": ask.mean(),
        "auto_accuracy": 1 - wrong[~ask].mean() if (~ask).any() else np.nan,
        "misroutes_per_100": 100 * (auto_wrong + residual * ask.sum()) / len(frame),
        "rs_per_request": cost / len(frame),
    }


def monthly_rupees(rs_per_request: float) -> float:
    return rs_per_request * REQUESTS_PER_MONTH
