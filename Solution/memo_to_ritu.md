# Memo: replacing the routing bot

**To:** Ritu Deshpande, Head of D2C Operations
**Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni
**Re:** Routing bot renewal

---

## The decision

**Don't renew the bot. Replace it with the new router after a two-week trial alongside it.**

I changed the target. The labels in the export are the bot's own first guesses, not where requests really belonged; Tanmay's note confirms this. A tool that "matched those labels at 90%" would copy the bot, mistakes included. We built one to test this: it matched the bot 96% of the time and routed no better than it. So we measured the new router against the team that actually closed each request.

## The number

| | Right team, first time |
|---|---|
| Vendor bot today | **77%** (about 1 in 4 misrouted) |
| New router | **86%** (tested on April–June 2026 requests it had never seen) |

We expect about 85–86% on the July–September requests you are scoring.

90% isn't reachable against real outcomes, by anyone. About one request in seven says nothing about the problem: "please call me about my purifier", Meenal's pile. History shows those end up spread across all seven teams. On requests that do describe the problem, the router is right 96–97% of the time.

## The rupees (about 720 requests a month)

Each misrouted request costs about **Rs 720**: 1.5 transfers at Rs 305, plus one extra customer call at Rs 260 (your policy §4).

| | Misroutes a month | Cost a year |
|---|---|---|
| Bot today | ~170 | Rs 14.5 lakh in misrouting + Rs 3.2 lakh licence |
| New router | ~100 | Rs 8.8 lakh in misrouting, no licence |
| **Saving** | **~65 fewer** | **Rs 5.7 lakh + Rs 3.2 lakh licence = about Rs 8.9 lakh a year** |

These figures use only your policy costs and Kestrel's own data. There are no assumptions in them.

**For Farhan:** the router costs **Rs 0 per request**. There are no AI fees, and the bill doesn't grow with volume. It runs on any office computer or existing server.

## Also useful for headcount

The bot's queues overstate Repairs and Billing, because those are where its wrong guesses land. They understate Installs, Returns and Warranty. Real monthly workload (April–June 2026):

| Repairs | Returns | Installs | Product Advice | Warranty | Billing | Filters |
|---|---|---|---|---|---|---|
| 166 | 109 | 107 | 90 | 86 | 85 | 69 |

Plan headcount on these, not on the bot's queue sizes.

## What to do next week

1. **Start the two-week trial.** Run the router alongside the bot on live requests. Customers still go where the bot sends them; we just record where the router would have sent them, then compare both with where each request finished. If the bot's renewal date falls before the trial ends, ask the vendor for a one-month extension rather than a year.
2. **Ask Tanmay for two fixes.** Correct the 126 records where a request changed team with no transfer recorded, and fix the product field, which disagrees with what the customer wrote about 1 time in 6. Then schedule a monthly refresh of the router on the latest closed requests (one command, a few seconds).
3. **Optional, with Meenal: look at the unclear requests.** The router marks requests it isn't sure about, mostly the "please call me" ones. During the trial, check whether agents could resolve these faster by asking one short question first. If that turns out to be practical, it could cut misroutes further. We have not counted it in the saving above.
