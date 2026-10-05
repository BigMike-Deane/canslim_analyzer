# Phase 1 signal check — 2026-10-05

**Question:** does the CANSLIM score sort future winners from losers?
**Answer so far:** no, not over this window. The test can't tell a score that
doesn't work from one that went against the market regime. That makes phase 2
(point-in-time history) the deciding test.

## Data
- `stock_scores`, last scan per stock per weekday, 2026-07-07 → 2026-10-05
  (Labor Day excluded). Liquid universe: price > $3, market cap ≥ $300M
  (~3,250 names/day). Forward return from the same table's `current_price`,
  winsorized to [−90%, +100%]. Rows with an unchanged price are dropped as stale
  (13% of the lowest-scored decile before the liquidity filter).
- Excess = stock return − equal-weight universe return on the same date, which
  removes market moves.
- Caveats: market cap is today's value, not as of each date (minor). Names that
  dropped out of the scan have no forward price (survivorship, small here).

## Results
**Top-minus-bottom quintile, 5-day forward, 12 non-overlapping windows**

| Signal | Mean spread | t | Jul 7–Aug 11 | Aug 12 → |
|---|---|---|---|---|
| I | +45 bps | +1.31 | +53 | +37 |
| C | +15 | +0.39 | 0 | +30 |
| A | −4 | −0.08 | −4 | −4 |
| S | −36 | −0.56 | −75 | +3 |
| N | −70 | −0.78 | −119 | −21 |
| **total** | **−49** | **−0.92** | **−119** | **+20** |
| L | −75 | −1.20 | −143 | −7 |

**What the app actually buys (score ≥ 72), 20-day excess vs universe**

| Band | Jul 7–Aug 11 | Aug 12 → |
|---|---|---|
| 72+ | **−297 bps** (37% beat) | +74 (55%) |
| 60–72 | −185 (38%) | **+146** (58%) |
| <60 | +35 | −26 |

## Read
1. **Regime-dominated.** Entries before mid-August: high scores lost by 2–4%
   over 20 days, a junk/reversal rally. Entries after: high scores won.
   The momentum parts of the score (L, N, S) flip sign with the regime. That
   repeats the Jul-22 audit ("L/S/N inverted at 14d").
2. **I and C are the steadiest.** I has the best spread and is positive in both
   halves. Neither is significant on 12 windows.
3. **72+ trails 60–72 in BOTH halves** (−112 bps and −72 bps at 20d). The buy
   threshold picks the most extended names. This agrees with the Aug-27 Chop Lab
   finding (the bleed comes from held names 10–25% extended). It is the first
   concrete hypothesis for phase 2.
4. **No significance.** 13 weeks is about 2 regimes. Neither "the score works"
   nor "the score is useless" can be concluded. The stop rule ("stop if
   top-scored names don't beat the rest") is **not passed but not decisively
   failed either**. Phase 2 decides it.

## Phase 2 hypotheses to test on point-in-time history (pre-registered here)
- H1: the total score's top quintile beats the bottom over 20d across 2015–2026
  (the basic gate; failing it on history → index fund).
- H2: the 60–72 band beats 72+ (an over-extension penalty or score cap).
- H3: a C+I-weighted score beats the current weights out of sample.
- H4: L/N/S work only in trend regimes (signal × regime interaction).
