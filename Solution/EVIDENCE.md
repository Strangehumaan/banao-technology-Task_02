# Evidence: how well it works, and how often it doesn't

Full working is in [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb); the data checks behind it are in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). This page summarises them.

## What "right" means

A prediction is right when it names the team that **actually closed** the request (`final_team` in the resolution log, with the renamed teams merged). The bot's `team_label` is not used as the answer: it is the bot's own first guess, and it agrees with the final team only 77.2% of the time.

## How it was tested

The tests are **out-of-time**: the model only ever sees months before the ones it is scored on, just as it will in real use.

| Training data | Scored on | Rows scored | Our model | Vendor bot |
|---|---|---|---|---|
| Apr–Sep 2025 | Oct–Dec 2025 | 2,199 | 85.2% | 77.5% |
| Apr 2025–Dec 2025 | Jan–Mar 2026 | 2,168 | 86.1% | 77.5% |
| Apr 2025–Mar 2026 | **Apr–Jun 2026** (headline) | 2,135 | **85.8%** | **76.7%** |

- On the headline quarter, the 95% bootstrap interval for accuracy is 84.4% to 87.3%.
- Macro-F1 is 85.6% (bot: 78.0%).
- The 126 contradictory rows were removed from training only. They stay in every scored quarter, so these figures include the label noise the hidden test will also have.

## How often it fails, and on what (headline quarter, 303 errors in 2,135)

| Kind of request | Share of requests | Our accuracy | Bot accuracy | Share of our errors |
|---|---|---|---|---|
| Clear: the message says what the problem is | 86% | **96.5%** | 86.3% | 21% |
| Vague: "please call back regarding my purifier" | 14% | 21.5% | 18.5% | **79%** |

**Vague requests.** About one request in seven names no problem at all. In history these ended up spread across all seven teams, with no team above about 22%, so neither a model nor a person can route them from the text. These requests make up four in five of our errors.

**Clear requests.** The roughly 3.5% of clear requests it gets wrong break down into:
- **About half are contradictory records:** the final team disagrees with an unambiguous message, e.g. "need warranty certificate" closed by Repairs with no transfer.
- **The rest are mostly mixed messages** where a real problem is followed by a vague tail ("box was open, purifier scratched, please call back regarding purifier"). The model leans on the last part, which here says nothing.

**By team.** Recall ranges from 82% to 90% across all seven teams. No team is systematically mis-learned.

## What it is worth

**Cost of one misroute: Rs 722.** In history a misroute took 1.51 transfers on average (Rs 305 each) plus one extra customer contact (Rs 260), per policy §4.

**On the headline quarter** (Apr–Jun 2026, 2,135 requests):

| | Misroutes | Transfers | Misrouting cost |
|---|---|---|---|
| Vendor bot | 498 | 706 (actual, from the log) | Rs 3.59 lakh |
| Our model | 303 | ~459 (estimated at 1.51 per misroute) | Rs 2.19 lakh |

Of the bot's 498 misroutes, our model gets **240** right. It introduces **45** new mistakes on requests the bot got right. Both get **258** wrong, mostly vague requests. The net result is **195 fewer misroutes**.

**Per year, at ~720 requests a month:**

| | Misroutes / month | Misrouting cost / month | Saving vs bot / year |
|---|---|---|---|
| Vendor bot (today) | 168 | Rs 1.21 lakh | — |
| Our model | 102 | Rs 0.74 lakh | **Rs 5.7 lakh**, plus the Rs 3.2 lakh licence = **Rs 8.9 lakh** |

These figures use only the policy costs and Kestrel's own data, with no assumptions.

## Optional: the low-confidence flag (not counted in the saving)

When the model's confidence is below 0.82, the service still returns its best guess, but marks the request as low confidence and suggests one question an agent could ask. This affects about 1 request in 6, mostly vague ones.

- **The confidence scores are reliable:** calibration error is 0.04, meaning that when the model says 90% sure it is right about 90% of the time.
- **Auto-routed requests are very accurate:** requests above the threshold are routed correctly ~98% of the time.

**Whether asking a question at intake is practical for Kestrel is untested.** The threshold also rests on two assumptions that are not in the pack: Rs 60 per question, and 10% still misrouted after asking. So none of this is included in the numbers above. If a trial showed it works, the modelling notebook estimates it could save roughly another Rs 1.5–7 lakh a year; the exact figure depends entirely on those assumptions.

## The pushback, tested

A model trained to copy the bot's labels (what "match those labels at 90%" literally asks for) matched the bot **96.3%** of the time on Apr–Jun 2026. It was right about the final team **76.7%** of the time, exactly the same as the bot. Hitting the 90% "match" bar would have bought nothing except the licence saving.

## Reproduce

```bash
uv run --directory Solution python -m kestrel_router.train
```

This re-checks the headline quarter, retrains on all labelled data and writes `predictions.csv`.

```bash
uv run --directory Solution pytest -q
```

These are the endpoint and failure-path tests (9 tests).
