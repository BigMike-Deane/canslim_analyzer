# Conviction portfolio (concentrated, small caps allowed) — pre-registration

Written 2026-10-12 ~9:00 PM CT, **before any number from this test was computed** (the compact
batch, docs/compact-portfolio-plan.md, had just failed).

## Why

Owner, Oct-12 (answering whether the AI Portfolio should switch to 20 picks): "It doesn't have to
be 14 … If the confidence for a stock is higher then you could load up on that one and have 6 or 8
or 9 I don't care. Figure out a model that works and will generate revenue OR at the very least has
the best chance to over the long term. I'm talking a decent amount of revenue, not just copy paste
the SPY. That's why I originally included some small cap. I think there's some opportunity there."

So: a **concentrated** book (≈ 6–14 names), **more money in higher-conviction names**, **small caps
allowed**, judged on long-run growth vs SPY. Two things must be true for that to be the best-chance
design, and they are tested separately:

- **A. Conviction pays:** the very top of the CANSLIM 2.0 ranking earns more than the merely good
  (otherwise loading up adds risk, not return). Tested across every stock and date — far more
  statistical power than one portfolio path.
- **B. The concentrated portfolio itself** beats SPY and beats random portfolios built the same way.

Mechanism prior (not evidence): the score leans to quality (ROE), earnings-beat streaks and buybacks;
the published size effect mostly appears in small caps *after* removing junk (Asness et al., "Size
Matters, If You Control Your Junk"). The score's only test on never-used data carried over to
$250M–$1B (t 2.06).

## Universe and score (frozen, no choices left)

- **Core:** the v3 panel (close > $5, market cap ≥ $1B, 20-day $vol ≥ $5M) — all names, not just the
  500 largest. **Small:** $250M–$1B, close > $5, 20-day $vol ≥ $1M (the band built by
  `c2_smallcap.build()`).
- **Score:** the frozen walk-forward CANSLIM 2.0 formula — each test year's signal set and signs as
  selected on earlier data (`report_v8_real.json`, `v5b`), mean of signed per-date percentile ranks,
  ranks computed **within each band**. Conviction = the stock's score percentile within its band.
- Test period 2019-01 → 2026 (walk-forward years), every 20 sessions, 20-session total returns,
  SPY total return benchmark. Delisted mid-window = 0% (known limitation).

## Part A — does conviction pay? (gate A)

Per trade date, mean 20-session return minus SPY TR for buckets of within-band percentile:
top 1%, 1–2%, 2–5%, 5–10%, 10–20%, below 20%; for core, small and pooled. Newey-West t over dates.

**Gate A passes** if, pooled: top-2% (top 1% + 1–2%) mean excess > 0 with NW t ≥ 2.0, AND top-2%
mean > the 5–10% bucket mean, AND top-2% mean > 0 in both 2019–22 and 2023–26.

## Part B — the conviction portfolio (trial 15)

- Every 20 sessions, candidates = core + small, ranked by within-band percentile.
- **10 holdings**, max 3 per sector. A holding is kept while it stays in its universe at ≥ 95th
  percentile; empty slots fill from the top of the ranking.
- **Conviction weights:** holdings ordered by current percentile, weight ∝ (11 − rank):
  18.2%, 16.4%, 14.5%, … 1.8%. No index core, no cash.
- **Costs:** 40 bps per one-way trade for names under $1B, 19 bps otherwise.

**Gates (all four):**
1. CAGR > SPY TR.
2. Beats SPY in ≥ 5 of 8 calendar years.
3. Active return > luck bar: 95th pct of the best of **15** (ledger after this trial) random
   versions — same universe, slots, keep rule, weights and costs, scores replaced by a random number
   per company (300 runs).
4. Max drawdown ≤ SPY's + 20 pp (owner accepts concentration risk; wider than the +10 pp used for
   index-like designs, fixed here before any number).

**Reporting only:** 6 and 14 holdings; equal weights; core-only and small-only pools; with a 30%
SPY core; random median; rolling 12-month beat rate.

## Decision rule

- **A and B pass:** the conviction portfolio becomes the AI Portfolio's Oct-22 strategy (replacing
  the 20-picks pivot), pre-registered again with its live rules.
- **A passes, B fails gate 3:** conviction is real on average but 7.5 years cannot separate a
  10-stock book from luck — I report the expected edge and the risk range, and the owner decides
  with that in front of them.
- **A fails:** loading up on top scores is not supported; concentrated stock picking is not the
  best-chance design; recommend against it.

**Honest expectation (before running):** the luck band for a 10-stock small/large book is very wide
(random books can differ by ±5–10 pp/yr over 7.5 years), so gate 3 is very hard. Part A is the more
informative answer.
