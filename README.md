# Kestrel Home: service-request router

Routes a new service request to one of Kestrel's seven teams, says how sure it is and why, and flags requests it isn't sure about. It runs locally and makes no paid API calls.

| | |
|---|---|
| Accuracy vs the team that actually closed the request (Apr–Jun 2026 holdout) | **85.8%** (vendor bot: 76.7%) |
| Expected accuracy on `predictions.csv` | **~85.5%** (84–87%) |
| Run cost | **Rs 0 per prediction**, no paid API |

## Run it (clean machine)

You need Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/), plus the Kestrel data pack. The data is not in this repo (ops-policy §10).

1. **Put the data pack in `Data/`** at the repo root (next to `Solution/`): `train.csv`, `test_unlabelled.csv`, `resolution_log.csv`, `teams.csv`, `sample_submission.csv`.

2. **Install the dependencies** (from the repo root):

   ```bash
   uv sync
   ```

3. **Train the model and write `predictions.csv`** (about 15–20 seconds):

   ```bash
   uv run --directory Solution python -m kestrel_router.train
   ```

4. **Start the service**, then open http://localhost:8000:

   ```bash
   uv run --directory Solution uvicorn service.app:app --port 8000
   ```

Notes:
- If you skip step 3, the service trains itself on first start.
- If the data pack is missing, the service still starts: `/health` reports `degraded`, and `/route` returns a polite 503 explaining what to do. It doesn't crash.
- Without uv: `pip install -r Solution/requirements.txt`, then run the same commands with `python` from inside `Solution/`.
- If the data lives somewhere else, set `KESTREL_DATA_DIR`.

## The endpoint

`POST /route` takes one record:

```json
{
  "request_text": "installer did not turn up for room heater, paid on upi",
  "channel": "whatsapp",
  "product_family": "Room Heater",
  "warranty_status": "in_warranty"
}
```

It returns:

```json
{
  "team": "Installs & Demo",
  "confidence": 0.999,
  "action": "route",
  "runner_up": {"team": "Repairs", "confidence": 0.001},
  "reasons": [
    "Mentions paying, but the problem is not the payment itself, so not Billing (policy §3).",
    "About installation, demo or wall-mounting.",
    "Key words: \"turn\", \"installer did\""
  ],
  "clarifying_question": null,
  "all_teams": {"Installs & Demo": 0.999, "Repairs": 0.001, "...": 0.0}
}
```

When `action` is `"clarify_first"`, the model's confidence is below 0.82; this is mostly messages that don't say what the problem is. The best guess is still returned, together with a suggested question an agent could ask. Treat it as advisory: whether asking at intake is practical for Kestrel hasn't been tested.

Other endpoints:
- `GET /health`: whether a model is loaded, and its validated accuracy.
- `GET /`: the screen.

## What's where

| Path | What |
|---|---|
| `predictions.csv` | Deliverable 1: one team per test `request_id` |
| `notebooks/eda_simple.ipynb` | The short version: 9 charts covering what I tried, what I changed, and what I threw away (used in the screen recording) |
| `notebooks/01_eda.ipynb` | What's in the data, what's wrong with it, and the decisions that follow |
| `notebooks/02_modeling.ipynb` | Model comparison, B/C experiments, error analysis, score forecast, rupees (deliverable 3) |
| `EVIDENCE.md` | Short written summary of the evidence: how well it works and how often it doesn't |
| `memo_to_ritu.md` | Deliverable 4: one-page memo to Ritu |
| `submission-form.md` | Deliverable 6: the completed form |
| `recording_script.md` | Outline for the 3-minute screen recording (deliverable 5) |
| `kestrel_router/` | Data cleaning, features, models, evaluation, explanations, training |
| `service/` | FastAPI app (`app.py`) and the screen (`static/index.html`) |
| `tests/`, `conftest.py` | 9 smoke tests for text cleaning, the endpoint and the polite-failure path (`conftest.py` lets pytest import the package) |
| `artifacts/` | Trained model and metadata (git-ignored; rebuilt by step 3) |

Run the tests:

```bash
uv run --directory Solution pytest -q
```

Re-run the notebooks:

```bash
uv run --directory Solution/notebooks jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_modeling.ipynb
```

The second notebook takes ~5 minutes (it includes LightGBM).

## Three decisions to know about

1. **The target is `final_team`, not `team_label`.** `team_label` is the vendor bot's own guess: it equals `first_team` in 100% of rows. A model that matched it at 90% would copy a router that is wrong 23% of the time.
2. **The renamed teams are merged.** Installations became Installs & Demo and Consumables became Filters & Consumables (policy §5). Predictions use the current names.
3. **126 contradictory rows are dropped from training only.** In these the team changed with zero transfers, which policy §3 makes impossible. They stay in every evaluation set, so the accuracy figures are honest.
