"""Loading, cleaning and splitting the Kestrel data pack.

Everything that touches the raw files lives here so the notebooks, the
training script and the service all see the data the same way.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KESTREL_DATA_DIR", ROOT / "Data"))

# Policy §5: renamed on 15 Jan 2026, responsibilities unchanged. The test
# period (Jul-Sep 2026) is entirely after the rename, so we predict new names.
RENAMES = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}
TEAMS = [
    "Repairs",
    "Installs & Demo",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]

META_COLS = ["channel", "product_family", "warranty_status"]

# Rolling 3-month backtests: train on everything before `start`, test on the
# following three months. The last fold is the headline holdout because it
# sits right before the hidden test period.
FOLDS = [
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
]
HOLDOUT = FOLDS[-1]


def canon_team(s: pd.Series) -> pd.Series:
    return s.replace(RENAMES)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
_ID_RE = re.compile(r"\b(?:KO|SR)\d{4,}\b", re.I)


_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]+")


def _fix_run(run: str) -> str:
    for _ in range(3):
        try:
            fixed = run.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if fixed == run:
            break
        run = fixed
    return run


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 read as cp1252 (legacy Zoho export), up to three layers deep.

    Applied per run of non-ASCII characters, because the same row can hold
    both a double-encoded ellipsis and a genuine accent ("hélp").
    """
    return _NON_ASCII_RE.sub(lambda m: _fix_run(m.group()), text)


def clean_text(text: str) -> str:
    """Normalise a request so legacy and CRM rows look alike.

    - repairs mojibake ("â€¦" -> "…")
    - strips accents ("urgént" -> "urgent", "namasté" -> "namaste")
    - masks order / registration numbers, which carry no routing signal
    - lowercases and squeezes whitespace
    """
    text = fix_mojibake(str(text))
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _ID_RE.sub(" idnum ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def mask_ids(text: str) -> str:
    """For anything we display: hide order and registration numbers."""
    return _ID_RE.sub("[id]", str(text))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _check_data_dir() -> None:
    if not (DATA_DIR / "train.csv").exists():
        raise FileNotFoundError(
            f"Data pack not found in {DATA_DIR}. Copy the Kestrel files there "
            "or set KESTREL_DATA_DIR."
        )


def load_train() -> pd.DataFrame:
    """train.csv joined to resolution_log.csv, with canonical team names."""
    _check_data_dir()
    tr = pd.read_csv(DATA_DIR / "train.csv")
    rl = pd.read_csv(DATA_DIR / "resolution_log.csv")
    df = tr.merge(rl, on="request_id", how="left", validate="one_to_one")
    df["created_at"] = pd.to_datetime(df["created_at_ist"])
    for col in ["team_label", "first_team", "final_team"]:
        df[col] = canon_team(df[col])
    df["text_clean"] = df["request_text"].map(clean_text)
    # §3 says a request only changes team via a transfer, so a different
    # final team with zero transfers cannot have happened as recorded.
    df["impossible"] = (df["first_team"] != df["final_team"]) & (df["transfers"] == 0)
    # Moved away and came back: first == final but transfers > 0.
    df["pingpong"] = (df["first_team"] == df["final_team"]) & (df["transfers"] > 0)
    return df


def load_test() -> pd.DataFrame:
    _check_data_dir()
    te = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    te["created_at"] = pd.to_datetime(te["created_at_ist"])
    te["text_clean"] = te["request_text"].map(clean_text)
    return te


def time_split(df: pd.DataFrame, start: str, end: str):
    """Train = everything before `start`; eval = [start, end)."""
    train = df[df["created_at"] < start]
    evalset = df[(df["created_at"] >= start) & (df["created_at"] < end)]
    return train, evalset


def drop_impossible(df: pd.DataFrame) -> pd.DataFrame:
    """B1: remove contradictory rows. Apply to TRAINING data only."""
    return df[~df["impossible"]]
