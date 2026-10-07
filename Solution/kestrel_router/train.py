"""Train the chosen model on all labelled data and write predictions.csv.

    uv run python -m kestrel_router.train

The model choice (logistic regression + B1 + B2) and the clarify threshold
come from notebooks/02_modeling.ipynb. This script only refits and predicts.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from . import data as D
from . import evaluate as E
from .model import build

warnings.filterwarnings("ignore", category=UserWarning)

SOLUTION = Path(__file__).resolve().parents[1]
ARTIFACTS = SOLUTION / "artifacts"
MODEL_PATH = ARTIFACTS / "router.joblib"
PREDICTIONS_PATH = SOLUTION / "predictions.csv"

FINAL = {"model": "logreg", "policy": True, "drop_bad": True}


def holdout_check(df: pd.DataFrame) -> dict:
    """Re-run the headline holdout so the saved metadata matches the model."""
    r = E.run_fold(FINAL["model"], df, D.HOLDOUT, policy=FINAL["policy"], drop_bad=FINAL["drop_bad"])
    s = E.scores(r.frame)
    lo, hi = E.bootstrap_ci(r.frame)
    return {**{k: round(v, 4) for k, v in s.items() if k != "n"}, "n": s["n"],
            "accuracy_ci95": [round(lo, 4), round(hi, 4)],
            "bot_accuracy": round(E.scores(E.bot_frame(df, D.HOLDOUT))["accuracy"], 4),
            "agreement_with_bot_label": round(float((r.frame.pred == r.frame.team_label).mean()), 4)}


def train_and_save(verbose: bool = True) -> dict:
    df = D.load_train()
    m_cost = E.misroute_cost(df)
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config": FINAL,
        "train_rows": int((~df["impossible"]).sum()),
        "dropped_impossible_rows": int(df["impossible"].sum()),
        "train_period": [str(df.created_at.min().date()), str(df.created_at.max().date())],
        "misroute_cost_rs": round(m_cost, 1),
        "clarify_threshold": round(E.clarify_threshold(m_cost), 3),
        "assumptions": {"clarify_cost_rs": E.CLARIFY_RS, "clarify_residual_misroute": E.CLARIFY_RESIDUAL},
    }
    if verbose:
        print("holdout check (train < 2026-04, score Apr-Jun 2026)...")
    meta["holdout"] = holdout_check(df)

    train = D.drop_impossible(df) if FINAL["drop_bad"] else df
    pipe = build(FINAL["model"], policy=FINAL["policy"])
    pipe.fit(train, train["final_team"])

    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, MODEL_PATH)
    (ARTIFACTS / "model_meta.json").write_text(json.dumps(meta, indent=2))

    test = D.load_test()
    preds = pd.DataFrame({"request_id": test["request_id"], "team": pipe.predict(test)})
    _validate(preds, test)
    preds.to_csv(PREDICTIONS_PATH, index=False)

    proba = pipe.predict_proba(test).max(axis=1)
    summary = {
        "rows": len(preds),
        "team_counts": preds["team"].value_counts().to_dict(),
        "clarify_first_share": round(float((proba < meta["clarify_threshold"]).mean()), 4),
    }
    (ARTIFACTS / "test_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print(json.dumps(meta["holdout"], indent=2))
        print(f"wrote {PREDICTIONS_PATH} ({len(preds)} rows) and {MODEL_PATH}")
    return meta


def _validate(preds: pd.DataFrame, test: pd.DataFrame) -> None:
    sample = pd.read_csv(D.DATA_DIR / "sample_submission.csv")
    assert list(preds.columns) == list(sample.columns), "columns differ from sample_submission"
    assert len(preds) == len(test) and preds["request_id"].is_unique, "one row per request_id"
    assert set(preds["request_id"]) == set(sample["request_id"]), "ids differ from sample_submission"
    assert set(preds["team"]) <= set(D.TEAMS), f"unknown team: {set(preds['team']) - set(D.TEAMS)}"


if __name__ == "__main__":
    train_and_save()
