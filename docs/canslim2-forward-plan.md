# CANSLIM 2.0 forward test — pre-registration (2026-10-10, before any trade)

**Why forward:** every result so far came from 2016–2026 history that we had already looked
at (10 trials; best out-of-sample result +0.7%/yr, below the luck bar — docs/score-v3-plan.md).
Forward paper trading uses prices nobody has seen and cannot benefit from survivorship: each
trade only knows the stocks and data that exist on its day. It is clean evidence, but slow.

**Owner (Oct-10):** "seeing it build a portfolio based on these signals and trade in and out
based on exit rules would be valuable information going forward… not attached to just 8 stocks.
Whatever makes the most sense."

## The score (frozen)

CANSLIM 2.0 as built in `backend/canslim2.py` on 2026-10-10: six signals, equal weights
(beat streak, surprise %, ROE, buybacks, −days to cover, analyst coverage), ranked daily over
the universe price > $5, market cap ≥ $1B, 20-day dollar volume ≥ $5M. Parity with the research
code: Spearman 0.963 (`research/pit/c2_parity.py`). Any change to the score or the rules below is
a NEW strategy with its own start date, never an edit to these.

## Strategy 1 — `canslim2_tilt` (500-stock tilt; the research's exact portfolio)

The 500 largest in the universe, weight ∝ market cap × 2 × score percentile among them,
rebalanced every 20 sessions, 19 bps per unit of one-way turnover, marked daily on
dividend-adjusted closes. Starts at the 2026-10-12 close with $25,000.

## Strategy 2 — `canslim2_picks` (20-stock picks with exit rules)

Simulated at closing prices after each session's scores (≈ 17:20 ET), $25,000 start, 2026-10-12.

1. **Sell** a holding when (a) its score percentile is below **70**, or (b) it left the universe /
   has no score, or (c) its value is **≤ −15%** vs its cost (dividends included).
2. **Buy** while fewer than **20** holdings: the highest-scored stocks with score percentile
   ≥ **90**, not held, not sold today, at most **5 per sector**; each buy is min(cash,
   equity / 20). Equal-weight at entry, no rebalancing between entries; no pyramiding.
3. **Costs:** 10 bps per buy and per sell (≈ the tilt's 19 bps round trip).
4. **Prices:** daily marks on dividend-adjusted closes; order records show raw closes.

## What we compare, and when

- Both vs **SPY total return** (chain-linked, `lab.chain_spy_adj`), and picks vs tilt
  (does concentration help or hurt?).
- **Reviews:** 2027-01 (3 months: mechanics only — trades, costs, turnover, data gaps),
  2027-04 (6 months), 2027-10 (12 months). Expect noise: research tracking error ≈ 2.6%/yr
  for the tilt and far more for 20 stocks, so 12 months cannot confirm a 0.7%/yr edge. A
  12-month result is reported with its luck range (random-score versions of the same rules,
  recomputed on the same dates), not as a verdict.
- **Mechanics alarms (bugs, not evidence):** a missed daily mark on a trading day, > 20% of
  holdings without a close, turnover above 100% per month for picks.

## Honest expectation

Research: the tilt ≈ +0.7%/yr over SPY out of sample (6 of 8 years). Every concentrated list
(25–30 names) trailed SPY 2019–26 by several points a year — the mega-cap gap — and the
signals favour steady, profitable companies over the explosive big winners. The most likely
outcome for picks is trailing SPY; that answer is still worth having on fresh data.

## Strategy 3 — AI Portfolio pivot: `canslim2_picks_live` (added 2026-10-10, before activation)

Owner (Oct-10): approved pivoting the AI Portfolio to CANSLIM 2.0 after the binding Oct-21
readout, trading the way the AI Portfolio does (intraday, live prices, Alpaca paper mirror).

- **Activation:** automatically at the first trading cycle on/after **2026-10-22** (config
  `canslim2_pivot`): every real user portfolio (user_id > 0) on `nostate_cs_bear` flips to
  `canslim2_picks_live`, and new portfolios default to it (owner policy Aug-21: all portfolios
  run the same strategy). Shadow arms (sandbox ids < 0) are never touched.
- **Rules = Strategy 2's**, executed live: up to 20 positions, buys from the top 10% (max 5
  per sector), equal dollars = portfolio value / 20, sells when the score percentile < 70, the
  stock leaves the universe, or price ≤ cost × 0.85 (checked by the intraday stop job and by
  each trade cycle). No trailing stops, no take-profit, no pyramids, no SPY sweep, no cash
  reserve, no circuit breaker, no buy throttle, no seeds, no ML veto, no correction-zone rule.
  No same-day re-buy of a name sold that day.
- **Execution:** every trade cycle (~90 min, market hours) at the live quote; the Alpaca paper
  mirror copies each trade; its resting hard stop sits at cost × 0.85.
- **Transition:** existing holdings are judged by the same sell rules at the first cycle
  (kept only if their CANSLIM 2.0 percentile is ≥ 70 and they are above −15%).
- **Comparison:** vs SPY total return and vs Strategy 2 (same rules, close-only). Live vs
  close-only measures what intraday execution adds or costs. Reviews on the same calendar.
- **Backtester:** not run here (`backtester.py` refuses the engine); the research test of
  these signals is research/pit (docs/score-v3-plan.md).
