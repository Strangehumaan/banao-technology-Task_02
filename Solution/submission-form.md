# Submission form: Task 2, Kestrel Home service-request routing (Variant B)



---

### What did you build, and what business decision does it support? State the number and the rupees.

**What I built:** a request router that replaces the vendor bot. It takes one request (message, channel, product, warranty) and returns:
- the team, with a confidence score;
- plain-language reasons ("Mentions paying, but the problem is not the payment, so not Billing (policy §3)");
- a low-confidence flag when it is unsure (mostly messages that don't state a problem).

It ships as a FastAPI endpoint with one screen, `predictions.csv`, two notebooks and a memo. It makes no paid calls.

**Decision it supports:** don't renew the Rs 3.2 lakh/year bot; switch to the router after a two-week shadow run.

**Number:** **85.8%** of requests go to the team that actually closed them, versus **76.7%** for the bot (Apr–Jun 2026, out-of-time). On requests that state their problem: 96.5%.

**Rupees:** a misroute costs about Rs 722: 1.51 transfers × Rs 305, plus Rs 260 for the extra contact (§4). At ~720 requests a month:
- **Misroutes:** 168 → 102 a month. On the Apr–Jun 2026 holdout that is 498 → 303.
- **Saving:** **Rs 5.7 lakh/year** in misrouting cost. With the licence, **~Rs 8.9 lakh/year**.
- **No assumptions:** these use only the policy costs and Kestrel's data.
- **Run cost:** Rs 0 per request.

### What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

**~85.5% accuracy (likely range 84–87%), scored against the final team.**

**Why accuracy:** the client's own bar is a match rate, and every misroute costs the same ~Rs 722, so plain accuracy maps directly to rupees. Macro-F1 tracks it within 0.5 points (85.6% on the holdout), so class imbalance doesn't distort it.

**How I estimated it:**
- Three rolling out-of-time quarters, each trained only on earlier months: 85.2%, 86.1% and 85.8%. The last one, Apr–Jun 2026, sits right before the test period. Its 95% bootstrap interval is 84.4% to 87.3%.
- The test quarter has the same mix of difficulty: 14.5% vague requests, versus 14.2% in the holdout.
- Test is all CRM, so the legacy encoding problems don't apply.
- The final model is trained on three more months than the holdout model.

So I expect the same or slightly better. The ceiling is ~86%: about 15% of requests contain no routable information, and in history those were closed by a near-random team.

**If the hidden outcomes are actually the bot's `team_label`:** my predictions agree with it 79.3% of the time, so expect ~79%. I chose `final_team` deliberately; see the pushback below.

### How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

**Validation:** out-of-time only. Three rolling folds (train before Oct 2025 / Jan 2026 / Apr 2026, score the next quarter).
- Impossible rows were dropped from training, but kept in every scored quarter.
- Nothing from the resolution log is ever an input; it is only used as the target.
- Five models were compared on identical folds before choosing; the full table is in `notebooks/02_modeling.ipynb`.

**Error rate:** 14.2% on Apr–Jun 2026 (303 of 2,135).

**What it gets wrong:**
- **79% of errors are vague requests.** "please call back regarding cooktop", "fryer problem": nobody can route these from the text. Best possible is ~22%; we get 21.5%. The service marks these as low confidence so agents know the guess is weak.
- **10% are rows whose recorded final team contradicts an unambiguous text** ("need warranty certificate" closed by Repairs with 0 transfers). This is label noise.
- **12% are genuine mistakes on clear requests**, 1.9% of them. Mostly a real problem followed by a vague tail ("box was open, purifier scratched, please call back regarding purifier"): the "last thing said wins" signal points at the empty part.

Recall per team is 82–90%; no team is systematically wrong.

**Service:** 9 automated tests cover text cleaning, the endpoint contract, input validation, the "paid is not Billing" rule, a comma before "paid on upi" (a bug found by testing the live endpoint), the vague → low-confidence path, and the polite 503 when no model is loaded. I also checked it by hand in the browser.

### Did you change, narrow, or push back on the client's ask? What, when, and why.

Yes, three times, all at the start, before any modelling.

1. **"Match those labels at 90%."** I changed the target from `team_label` to `final_team`.
   - Tanmay's email and the data both show `team_label` is the bot's first guess. It equals `first_team` in 100% of rows, and it is right only 77.2% of the time.
   - I tested the literal ask. A model trained on `team_label` matched the bot 96.3%, clearing the bar easily, and was right about the final team only 76.7% of the time, the same as the bot. Switching the bot off that way saves the licence and nothing else.
2. **The 90% bar.** Against real outcomes ~86% is the ceiling, because ~15% of requests carry no routing information. I report that honestly instead of chasing 90%, and measure success as fewer misroutes than the bot (and rupees).
3. **"Switch the bot off."** I recommend a two-week shadow run on live requests first. The data is templated, so real messages may be harder.

### What is wrong with what you are handing us, or with the data we handed you?

**In what I'm handing over:**
- **Templated text.** The text looks generated from templates. The model is fitted to that, so accuracy on real, messier messages (Hinglish, typos, long emails) is unknown. The shadow run is how to find out.
- **The low-confidence threshold rests on assumptions.** The 0.82 cut-off uses two figures I assumed (Rs 60 per question, 10% still misrouted after asking); the pack has neither. Whether Kestrel can ask customers a question at intake is also untested. So the flag is advisory only and isn't part of any headline number.
- **Hand-written keyword lists.** The policy features (fault / spares / install / "paid" words) are hand-written from `teams.csv` and the policy. New phrasings will be missed by them, though the n-gram model still sees the words.
- **Reasons are approximate.** They come from model weights plus keyword rules. They are a sound summary, not a proof, and occasionally list a generic phrase.
- **"Last thing said wins" fails on a vague tail.** It misroutes messages like "X is broken, please call me about X".
- **Dropping the 126 rows didn't help.** It measured as neutral (±0.1 pt). I kept it as a data-quality rule, not for accuracy.
- **Team names.** `predictions.csv` uses the current names (Installs & Demo, Filters & Consumables), because every test row is after the 15 Jan 2026 rename. If your key uses the old names, map them back.
- **Not production-hardened.** There is no authentication, no drift monitoring and no automated retraining. The model file isn't in the repo (§10), so it must be rebuilt with one command; the service does this itself on first start if the data is present.

**In the data:**
- **`team_label` is described as ground truth but isn't.** It is the bot's queue at creation.
- **126 impossible rows:** `first_team ≠ final_team` with `transfers = 0`, which §3 makes impossible. Both systems, every month.
- **351 ping-pong rows:** `first_team = final_team` with 1–2 transfers (A → B → A). The labels look right; the transfers were wasted, ~Rs 1.1 lakh of handling time over 15 months.
- **`product_family` disagrees with the product named in the text in 17.5% of rows** (17.2% in test). Example: "spare blade set for fan" filed under Robot Vacuum. I trusted the text.
- **480 legacy Zoho rows have mojibake and stray accents** ("â€¦", double-encoded "Ã¢â‚¬Â¦", "urgént"). Test has none. Some rows carry both a double-encoded character and a real accent, which breaks naive whole-string fixes.
- **Legacy `resolved_at` is UTC, not IST (§9).** 1,140 rows appear resolved before they were created; +5:30 fixes all of them.
- **106 texts appear more than once with different final teams.**
- **The final team is itself shaped by the bot.** Agents sometimes close what reached them, so the "truth" carries some of the bot's routing bias.
- **Minor inconsistencies:**
  - Tanmay says the resolution log excludes the latest requests, but every train row has one; 15 resolve after 30 Jun.
  - `teams.csv` lists the old names in its `team` column.
  - `sample_submission.csv` is all "Repairs".
  - "About 700 orders a month" vs ~720 requests a month in the data.
  - Some texts carry two order numbers.

### What did you deliberately leave out, and why that rather than something else?

- **An LLM in the product.**
  - The brief requires it to start without a paid key.
  - Policy §10 limits sharing customer text with vendors that aren't approved.
  - Farhan asked for no bill that grows per request.
  - Most importantly, it can't fix the main error source: a message that contains no problem.
  - I considered it but did not test it: I don't expect it to gain accuracy here, and it would add cost and risk.
- **Transformer embeddings / fine-tuning and AutoML.** The five models I tried landed within 1.5 points of each other, under a ~86% ceiling set by the data. A heavier model would add install weight and harm explainability and the "starts on a clean machine" requirement, for no measurable gain.
- **A fixed fallback team for vague requests.** It helped two quarters and hurt the latest one (−0.4 pt), so it isn't stable. The model's own best guess is kept, with a low-confidence flag.
- **Docker, auth, monitoring and a retraining scheduler.** "A small thing that runs beats a large thing that does not." `uv sync` plus two commands is enough here; these are next steps after the shadow run.
- **Turnaround-time analysis.** Not asked, and it would need the UTC fix. I noted the fix and left the analysis out.

### Anything you built or found that nobody asked for?

- **"The last thing said wins."** In requests with two intents, the closing team follows the **last** one 99.1% of the time. A bag-of-words model can't see word order, so I added the last clause as its own input. This is a large part of B's +1.2 points.
- **A low-confidence flag, with a suggested question.** The threshold (0.82) is derived from the policy's rupee costs, not tuned. Asking customers one question on unclear requests could add savings, but whether Kestrel can do that at intake is untested. It is offered as an option for the trial and **not counted** in any number above.
- **Workload view for Ritu's headcount plan.** The bot's queues overstate Repairs (+24%), Billing (+26%) and Filters (+37%), and understate Installs, Returns and Warranty. I included real monthly volumes per team and the predicted Jul–Sep volumes.
- **The "copy the bot" experiment**, which settles the 90%-match question with numbers.
- **The ping-pong cost** (~350 wasted transfers), and policy-citing reasons in every response.
- **The polite failure path** (503 with instructions, `/health` reports degraded) and tests for it.

### What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.


**Tool:** Claude Code (Claude Opus 5.5) in the Claude desktop app, as a pair programmer. No AI is used inside the product.

**Where it helped:**
- Reading the PS, policy and emails, and listing every hint and contradiction before any code was written.
- Writing the EDA that found the `team_label` = bot issue, the impossible and ping-pong rows, the vague shapes, and the last-intent rule.
- Writing the package, the notebooks, the FastAPI service and screen, the tests and these documents.

**Where it wasted time:**
- **pandas 3.0 changed `groupby().apply`** (grouping columns are excluded), which broke two notebook cells and needed reruns, each ~5 minutes because of LightGBM.
- **The first mojibake fix** failed on rows mixing a double-encoded character with a real accent.
- **Narrative numbers drafted before the notebook ran** (e.g. "~60% of errors are vague") were wrong and had to be corrected to the real outputs (79%).
- **Browser screenshots of the UI kept timing out**, so the UI was checked through the page text instead.

**Thrown away:**
- LightGBM (no gain, ~30× slower), Naive Bayes, and the vague-request fallback team.
- An LLM router, considered and rejected.
- A plain whole-string mojibake fix.

**Cost:** 2000

**Screen recording:** https://drive.google.com/file/d/1vf454VWkMNBpAhLs89IYPS3E9iMe7FBb/view?usp=sharing

### Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **Train on `final_team` (from `resolution_log.csv`), never on `team_label`.** `team_label` is the old bot's guess. Training on it rebuilds the bot: 96% "match", 77% right. Merge the old team names into the new ones (this is done in `kestrel_router/data.py`).
2. **To refresh the router:**
   - Drop the new export into `Data/`.
   - Run `uv run --directory Solution python -m kestrel_router.train`.
   - Check `Solution/artifacts/model_meta.json`: the holdout accuracy should be ~85% and well above `bot_accuracy`.
   - Restart the service.
   - Data and model files must never go to git (§10); `.gitignore` already handles this.
3. **Vague requests (~15% of traffic) cause ~80% of the errors, and no model can fix them.** The text has nothing to route on. The service marks them as low confidence (below 0.82). That cut-off comes from two assumed costs in `kestrel_router/evaluate.py` (`CLARIFY_RS = 60`, `CLARIFY_RESIDUAL = 0.10`). Treat the flag as advisory until the service desk confirms that asking customers a question at intake is practical. If it is, measure the real cost, update the two constants and retrain.

### Honest hours spent. 2.5.

### GitHub repo link

https://github.com/Strangehumaan/banao-technology-Task_02

### What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders a month)? Show the arithmetic. If you used no paid calls, say so.

**No paid calls.** The model is a 0.6 MB scikit-learn file that runs locally on CPU. The arithmetic below uses 700 requests a month, as the question says. The data shows ~720 a month, which doesn't change any conclusion.

**One prediction:**
```
API fees:  Rs 0     (no model API is called)
Compute:   ~50 ms of CPU on a 2019 laptop (i5-9300H), including the reasons
```

**One month:**
```
Variable cost:  700 requests × Rs 0 = Rs 0
Compute:        700 × 0.05 s ≈ 35 seconds of CPU
Retraining:     ~20 seconds of CPU
```

**Hosting** is the only real cost. It is a flat fee that doesn't grow with requests, which meets Farhan's condition. Two options:
```
(a) Existing office PC or server                    = Rs 0 extra
(b) Small cloud VM (1 vCPU, 1–2 GB RAM), $5–10/month
    $5 × Rs 88 = Rs 440;  $10 × Rs 88 = Rs 880       ≈ Rs 450–900 a month
    (typical market price, not a quote; Rs 88 per $ assumed)
```

The model needs well under 1 GB of RAM, so option (a) is enough.

**Per-request cost, including hosting:**
```
(a) Rs 0 ÷ 700                  = Rs 0 per request
(b) Rs 450–900 ÷ 700            ≈ Rs 0.65–1.30 per request
```

**Compared with the bot:**
```
Licence:      Rs 3,20,000 a year
Per month:    Rs 3,20,000 ÷ 12  ≈ Rs 26,700
Per request:  Rs 26,700 ÷ 700   ≈ Rs 38
```

So the router costs **Rs 0 to ~Rs 1.30 per request** against **~Rs 38** for the bot. That is **Rs 0–900 a month** against **~Rs 26,700**.
