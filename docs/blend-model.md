# The evidence-weighted model (started 2026-10-13)

## Why this exists

Owner, Oct-12: "Let's stop trying to backtest and just work off the signals we have … give me
something that at least has some value and weights based on the data we have available to us,
both market data and the data and trends we have analyzed from the stock universe we have been
following." Then: "Build it, add it to the lab, and let me track it against the AI Portfolio. Maybe
create a weekly routine to review the two and identify any noticeable gaps between them."

This is **not** a backtested or proven strategy. Fifteen pre-registered tests (docs/score-v3-plan.md,
docs/compact-portfolio-plan.md, docs/conviction-portfolio-plan.md) showed no stock-picking design that
free data can prove beats SPY. The model instead weights each part by **how much evidence stands
behind it**, using two independent sources:

1. **Research:** 2016–2026 point-in-time tests of every signal on ~1,800 large and ~550 small companies.
2. **Live:** the Jul–Oct signal check on the ~4,500 stocks the app tracks (docs/signal-check-2026-10-05.md)
   and the AI Portfolio's own record since Apr-7 (+25.0% vs SPY +18.0%, a third of the gains from DELL).

Where both sources agree, the part gets weight. Where neither shows anything, it gets none.

## The rules

**Market switch (how much to own)** — strongest evidence: the A5 rule passed 1928–93 and 1994–2026,
and the live book made money on trend days (+16 bps/day) and lost on chop (−10 bps/day).
- S&P above its 200-day average AND its 12-month return beats T-bills → up to **10** stocks.
- One of the two → up to **5** stocks; idle cash parked in SPY.
- Neither → **no new stocks**; holdings ride their stops; the SPY parking is sold.
- Read from the Lab's daily A5 decision; if missing, computed; if that fails, treated as "one of two".

**What to buy** — weighted CANSLIM 2.0 signals, ranked within large caps and within small caps:

| Signal | Weight | Evidence |
|---|---|---|
| Earnings-beat streak | 17.5% | research t 4.1 (strongest, steadiest); live C positive |
| Size of the latest beat | 17.5% | research t 3.3; live C positive |
| Analyst coverage | 20% | research t 2.9; live I the best signal (+45 bps/5d) |
| Low days-to-cover (short pressure) | 20% | research t −4.3 (lower is better) |
| Return on equity | 15% | research t 2.9; live A flat |
| Buybacks (shrinking share count) | 10% | research t 2.8 |
| Breakouts, new highs, relative strength, volume, the classic total score | 0% | no signal or negative in both sources |

- Buy from the **top 10%** of its universe, best first.
- **Not stretched:** no stock more than 15% above its 50-day average (live: the classic 72+ band
  trailed 60–72 because high scores arrived after the run-up).
- **Max 3 per sector.**
- **Small caps** ($250M–$1B, ≥ $1M/day traded — the band where the score carried over in testing):
  max 3 of the 10 slots, ranked among themselves; **none, and nothing under $2B, while IWM is
  below its 50-day average** (62% of the live chop losses were small caps lagging).
- **Equal size** (portfolio ÷ 10), fully invested when the switch says 10 — no 5–60% regime cash
  reserve (the AI Portfolio's would hold back exactly one slot; Oct-13 first cycle bought 9, owner
  approved the exemption the same day). Idle cash only follows the market switch above. No extra weight for the top score at entry: the very top of the
  ranking did not beat the merely good (conviction test, Oct-12).
- No re-buy within 5 days of selling.

**After the buy — identical to the AI Portfolio** (live evidence: partial profits + take-profit
+$5,246, pre-earnings protective stops +$1,710, the 7–8% stop's average loss −8.0% passed its check):
ATR-widened 7% stop (6% in bearish tape), trailing stops by gain tier, partial profits at +25/+40/+60%
while the score holds, +75% take-profit when the score fades, pyramids into winners up to 25% of the
book, the drawdown circuit breaker. The one change: the score-exit and partial-profit checks read the
model's score (sell below the 45th percentile), and the classic score-crash rule is off (it reads the
classic score's history).

## How it runs and how it is compared

- **Shadow book** `shadow_model_v1` (strategy profile `blend_v1`, engine code `backend/blend_trader.py`):
  same harness as every shadow arm — trades in the same intraday cycles as the AI Portfolio, at live
  quotes, with the same stop job. $25,000 from Oct-13.
- **Lab:** "Model · Evidence-weighted" next to "AI Portfolio" (the live book, measured from Oct-13),
  on the comparison chart, with positions, trades and a daily log (`backend/lab_mirror.py`).
- **Weekly review** (Fridays 6:55 PM CT, push + the Lab card; `/api/lab/weekly-review` any time):
  returns this week and since start vs SPY, how much each has in stocks, holdings in common, trades and
  stops, and flags: a weekly return gap ≥ 2 points, an invested gap ≥ 25 points, no stocks in common,
  2+ stops in a week, the small-cap brake on.

## What to expect

Big differences week to week and year to year; a single stock can swing the comparison (as DELL did
for the AI Portfolio). If the model adds value, it should show mostly as (a) sitting out downtrends
and (b) fewer stretched-chart losers — not as picking the perfect stock. A few months is not enough
to call it either way; the weekly review is for spotting **gaps and malfunctions**, not a verdict.
