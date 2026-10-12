# Compact portfolios (≤ 14 positions) — pre-registration

Written 2026-10-12 ~7:30 PM CT, **before any number from these designs was computed**.

## Why

Owner, Oct-12: "I'm not going to have 500 stocks in a portfolio — at best I'm going to have 14
positions max if we start including index funds. Go back to the drawing board, think of
something new, do some testing, and rework this to be something that will actually be
applicable." This reopens stock-selection research (closed by the Oct-8 decision memo) under a
hard constraint: **at most 14 positions in total, index funds count as positions, trades a person
can copy (monthly).**

What earlier tests established (docs/score-v3-plan.md):

1. **The mega-cap gap.** In 2019–26 a random 30-stock book drawn from the 500 largest trailed SPY
   by ~5.4 pp/yr before any signal — SPY's return came from a few mega-caps at index weight. Any
   stand-alone pick list starts ~5 pp/yr behind.
2. **The stock signal is real but small.** CANSLIM 2.0 (walk-forward v5b: beat streak, surprise,
   ROE, buybacks, low days-to-cover, analyst coverage) ranks stocks with IC ≈ 0.02–0.03; the
   500-stock tilt made +0.67%/yr vs SPY, below a best-of-10 luck bar. It carried over to
   $250M–$1B companies (t 2.06).
3. **Exposure rules (A1/A5) pass on 1928–93 and 1994–2026** but are US-only and trail in steady
   bull markets. A5 is one position.

So a ≤14-position design has to (a) keep the mega-caps through an index core, (b) spend its few
stock slots where the signal is strongest, or (c) express the signal through funds.

## Common rules (all designs)

- **Data:** the point-in-time panel used since v3 (`meta/v3_table.csv.gz`), walk-forward CANSLIM 2.0
  scores `v5b` from `meta/v3/scores_v8_real.csv.gz` (each year's signal set chosen on earlier data
  only), test period **2019-01 → 2026-09** (OOS years 2019–2026).
- **Cadence:** panel dates every 10 sessions; trade every 2nd date (every 20 sessions ≈ monthly),
  exactly as v5. Returns = 20-session forward total returns (dividends in), SPY = SPY total return.
- **Costs:** 19 bps per unit of one-way turnover across **all** positions (SPY and funds included);
  the A5 core uses its own engine costs (`a_exposure.run`: financing, fees, 5 bps per switch, 2-day
  delay). Sector funds: 0.09%/yr fee.
- **Known limitation (unchanged from earlier tests):** a stock with no 20-session return
  (delisted mid-window) counts as 0%.
- **Sectors:** the panel's `sector` field (FMP's 11 sectors).

## Designs (stock-selection trials 12–14; K3 is a composite, not a new selection)

**K1 — Index core + 13 best-scored large caps (trial 12).** 14 positions: SPY 60%; 13 stocks at
40%/13 (3.08%) each. Universe each date: the 100 largest companies with a v5b score. Rank by v5b;
existing holdings are kept while they stay in the 100 largest and rank ≤ 26; empty slots fill from
the top of the ranking, max 3 per sector. All weights reset to target every 20 sessions.

**K3 — A5 core + K1's 13 stocks (composite).** As K1, but the 60% core follows the A5 rule
(2× S&P when the S&P is above its 200-day average AND its 12-month return beats T-bills; 1× when
one holds; T-bills when neither), simulated by the A5 research engine. The 13 stocks are exactly
K1's. Not a new stock-selection trial; it passes only if K1 passes gate 3 and K3 itself passes
gates 1, 2 and 4.

**K4 — 11 sector funds tilted by CANSLIM 2.0 (trial 13).** One position per sector (stand-in for
the sector SPDR funds): the sector's cap-weighted 20-session return within the 500 largest. Sector
score = cap-weighted mean v5b percentile of its members. Weight = sector's cap share × m,
m = 1 + (2u − 1), u = the sector score's percentile rank among the sectors (0×..2×, as v5b);
normalized. (m = 1 everywhere ≈ SPY.)

**K5 — Index core + fresh earnings-beat streaks (trial 14).** SPY 60% plus up to 13 slots of
3.08%. On each trade date, eligible = 500 largest whose `beat_streak` is ≥ 4 AND higher than on
the previous trade date (a new beat reported within the last 20 sessions). Rank eligible by
`surprise_pct` (largest first), max 3 per sector, fill empty slots. Each position is held exactly
3 periods (60 sessions), then sold; unused slots sit in SPY. Weights reset each period.

## Gates (each of K1, K4, K5; K3 as above)

1. CAGR > SPY total return (2019-01 → 2026-09).
2. Beats SPY in ≥ 5 of 8 calendar years (2026 partial counts, as before).
3. **Active return > luck bar**: 95th percentile of the best of **14** (trial ledger after this
   batch) random versions of the **same design** (300 random runs each): K1/K3 — random scores
   inside the same 100-stock universe and rules; K4 — random sector scores; K5 — same slots and
   60-session holds filled with random stocks from the 500 largest (max 3 per sector).
4. Max drawdown ≤ SPY's + 10 pp.

**PASS** = all four. **Reporting only (no verdict):** % of rolling 12-month windows beating SPY;
random median for each design (what the structure alone does); K1 with the 13 largest companies
instead of the 13 best-scored (does the score beat plain size?); K1 sleeve at 30% / 100% instead
of 40%.

## What happens next

- **Any PASS:** that design becomes a Lab strategy (simulated, monthly, ≤ 14 positions) as forward
  confirmation, and the app gets a "copy this portfolio" view with the exact positions and weights.
- **All FAIL:** the applicable answer for a ≤ 14-position account is an index core (with the A5 rule
  as the optional exposure layer, still under forward test), and stock selection on free data is
  closed for good.

**Honest expectation (written before running):** with 13 stock slots the signal's edge is diluted
by single-stock noise; the random luck band for 13-name sleeves will be wide. The most likely
outcome is "about SPY, not distinguishable from luck." K3 is the most likely to beat SPY on
gates 1–2, through A5, not through stock picks.

## Results (2026-10-12 ~8:30 PM CT, single run; placebo first): ALL FAIL

Placebo (stock returns shuffled within each date, 20 luck runs): every design at the 35th–60th
percentile of its random versions — pipeline clean. Real run, 2019-01 → 2026-07 (95 periods,
SPY TR 17.23%/yr), 300 random versions per design, bar = 95th pct of the best of 14:

| Design | CAGR | Active | Years beat | Rolling 12m beat | Max DD | Random median / 95th | Bar | Pct of random | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| K1 SPY 60% + 13 best-scored of 100 largest | 16.11% | −1.12% | 3/8 | 20% | 31.4% | −0.81% / +1.55% | +4.22% | 42nd | **FAIL** (1,2,3) |
| K3 A5 core + K1's stocks | 17.08% | −0.15% | 5/8 | 33% | 30.6% | +0.11% / +2.47% | (K1 gate 3) | 44th | **FAIL** (1,3) |
| K4 11 sector funds tilted | 18.31% | +1.08% | 6/8 | 62% | 30.2% | −0.14% / +2.50% | +3.05% | 69th | **FAIL** (3) |
| K5 SPY 60% + fresh beat streaks | 16.31% | −0.93% | 4/8 | 36% | 34.0% | −1.60% / +0.66% | +1.90% | 68th | **FAIL** (1,2,3) |

Reporting only: K1 with the **13 largest** companies instead of the best-scored: 18.73% (+1.50%,
5/8 yrs, DD 28.5%) — owning mega-caps beat SPY in this era; the score picked worse than size.
K1 sleeve 30%: −0.83%; sleeve 100% (13 stocks, no index): −2.99%. A5 core alone: 18.22% (+0.98%).

**What it means.**
1. **Inside a ≤14-position portfolio the CANSLIM 2.0 score did no better than random picks**
   (K1 42nd, K5 68th percentile). With 13 names, single-stock luck is ±2–4 pp/yr over 7.5 years —
   far larger than the score's ~0.7%/yr edge, which only shows up across hundreds of names.
2. **K4 is the near miss:** +1.08%/yr and 6 of 8 years with 11 funds, but a *random* fixed sector
   lean did as well 31% of the time — indistinguishable from luck on 7.5 years.
3. **The more stock slots replace the index, the worse it got** (30% → −0.83%, 40% → −1.12%,
   100% → −2.99%): the mega-cap gap again.

**Decision (as pre-registered):** no Lab strategy from this batch. For a ≤14-position account the
applicable answer is an index core, with the A5 rule as the optional exposure layer (forward test
running). Trial ledger: 14. Stock selection on free 2016–26 data is closed.
