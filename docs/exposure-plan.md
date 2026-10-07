# Exposure strategies (A-family) — pre-registration (2026-10-07, before any number)

**Owner goal:** a data-driven way to beat SPY; "pivot into indexes as the market
churns, then lean in as it trends." Stock selection gives at best ~+1 pp/yr (Score
v5b, a tilt). This track asks the other question: **how much** market exposure to
hold, and when.

**Data (free):** S&P 500 index daily closes from 1927-12-30 (FMP `^GSPC`);
dividends from Shiller's monthly series (trailing 12-month D / P, accrued daily)
until SPY exists, then SPY dividend-adjusted closes (1993-02 →). Cash rate: NBER
short-term Treasury yields (FRED M1329AUSM193NNBR) 1928–33, 3-month T-bill
monthly (TB3MS) 1934–53, daily (DTB3) 1954 →.

**Periods:** **P1 = 1928-10-01 → 1993-12-31 (never examined in this program)**;
P2 = 1994-01-03 → 2026-10-05 (the 50-day unleveraged gate was tested here and
failed — T1). Benchmark: S&P total return, buy and hold.

**Execution and costs (all rules):** signal from the close of day t−1, position
taken at the close of day t, earning day t+1's return (one full day of delay).
Leverage above 1× pays the cash rate + 0.5%/yr on the borrowed amount plus a
0.9%/yr fund fee on the whole leveraged position (SSO-like); 1× pays 0.09%/yr.
Each change in exposure costs 5 bps of the amount traded.

**Rules (parameters from the published literature, not tuned here):**
- **A1 — 200-day trend, 2× (Gayed & Bilello 2016 "Leverage for the Long Run"):**
  2× S&P when the index is above its 200-day simple average, else cash.
- **A1r — same, 1× (reference only):** the unleveraged version.
- **A2 — volatility targeting (Moreira & Muir 2017 style):** exposure =
  min(2, 16% / realized volatility of the last 21 sessions, annualized), rest in
  cash; 16% ≈ the S&P's long-run volatility.
- **A3 — trend + volatility:** A2's exposure when above the 200-day average,
  cash otherwise.

**Pass rules (each of A1, A2, A3):**
1. CAGR > S&P TR in **P1 and in P2**.
2. Beats S&P TR in **≥ 60% of rolling 10-year windows** (monthly steps, full span).
3. Max drawdown ≤ S&P TR's max drawdown **in each period**.

**PASS → candidate overlay** for the stock model (v5b) and forward paper. Reported:
Sharpe, % time invested, switches/yr, worst year, the 2-day-delay and 0-day-delay
sensitivities, and the same rules with 1.5× and 3× (reporting only, not chosen).

**Prior (stated before running):** published work finds leveraged 200-day trend
beat buy-and-hold over 1928–2015 mainly by sidestepping 1929–32 and 2008; fast
V-shaped crashes (1987, 2020) and choppy decades hurt. P1 likely favourable, P2
uncertain; volatility targeting improves Sharpe but its CAGR edge depends on
leverage costs.

## Results (2026-10-07 ~10:45 AM CT, single run) — `research/pit/a_exposure.py`

Data check first (no strategy numbers): S&P TR 1928–93 9.85%/yr, max DD 83.6%,
worst year −43% (1931); 1994–2026 10.9%/yr, DD 55% — matching the record.

| Rule | P1 1928–93 CAGR (vs 9.61%) | P1 max DD (vs 83.6%) | P2 1994–26 CAGR (vs 10.90%) | P2 max DD (vs 55.2%) | Sharpe P1 / P2 (vs 0.38 / 0.51) | 10-yr windows beating | Verdict |
|---|---|---|---|---|---|---|---|
| **A1 200d 2×** | **14.55%** | 79.9% | **12.33%** | 45.7% | 0.52 / 0.51 | **86%** | **PASS** |
| A1r 200d 1× (ref) | 10.67% | 52.0% | 8.93% | 22.6% | 0.57 / 0.57 | 41% | ref |
| A2 vol target | 9.58% | 60.3% | 9.95% | 48.3% | 0.40 / 0.49 | 43% | FAIL |
| A3 trend + vol | 11.25% | 39.3% | 8.78% | 29.0% | 0.58 / 0.47 | 49% | FAIL |

A1: invested 64% / 75% of days, 5.6 / 6.9 switches a year. Sensitivity
(reporting): 1-day delay 14.93% / 11.11%, 3-day delay 13.31% / 11.07% — **P2
edge shrinks to +0.2 pp/yr with slower execution**; 1.5× 12.52% / 10.48%, 3×
17.30% / 14.87% (DD 93% / 65%).

**A1 PASSES** — the published "leverage for the long run" result replicates on
65 untouched years and on 1994–2026. **Caveats, in plain terms:**
1. The 1994–26 gain is leverage, not better risk-adjusted performance (Sharpe
   0.51 = 0.51); P1 is where the trend filter adds real Sharpe (0.52 vs 0.38).
2. The P2 margin (+1.4 pp/yr) is thin and sensitive to execution speed.
3. It still loses ~46–80% in the worst crashes; worst year −35.6% (P2).
4. ~6 switches a year → short-term gains in a taxable account (fine in an IRA).
5. Implementation = a 2× S&P fund (SSO) or margin, daily-rebalanced; the
   simulation models daily leverage costs and decay.

**Next:** combine A1 (when to lever) with v5b (what to hold) and run both as
forward paper arms.

## C1 pre-registration — A1 exposure × v5b holdings (2026-10-07 ~12:20 PM CT, before the corrected-data v5b numbers exist)

**Question:** does holding the v5b tilted book instead of the index improve A1?

**Rule C1:** each session, exposure from A1 (2× when the S&P closed above its
200-day average on the prior session, cash otherwise, same 1-day delay); the
invested sleeve holds **the v5b tilted portfolio** (re-weighted every 20
sessions exactly as v5b) instead of the S&P. Leverage, financing (cash rate +
0.5%), 0.9%/yr leverage fee and 5 bps per exposure change as A1; v5b's 19 bps
turnover cost on rebalances.

**Data / period:** v5b's out-of-sample years only, **2019-01 → 2026-09**, on the
**corrected ("v4") data** — the v5b re-run in `run_v4data.sh`. Daily returns
of the v5b book: each rebalance's weights × daily stock total returns until the
next rebalance (weights drift).

**Pass (all):** (1) C1 CAGR > A1 CAGR over the same dates; (2) C1 CAGR > SPY
TR; (3) C1 beats A1 in ≥ 5 of 8 calendar years; (4) C1 max DD ≤ A1 max DD +
5 pp. Reported: Sharpe, tracking vs A1, and C1 with 1× (unlevered timing) for
reference. **If v5b fails on the corrected data, C1 is not run** (no holdings
edge to combine). PASS → third Lab strategy (needs a 4th Alpaca paper account).

## Exposure family 2 pre-registration (2026-10-07 ~2:25 PM CT, before any number)

Owner: "What combination will create a model that competes and can potentially do
better than the SPY long term?" A1 passed, but its 1994–2026 edge fell to +0.2 pp/yr
with a 1–3 day execution delay (≈ 7 switches a year). Two published refinements
target exactly that; parameters from the papers, not tuned:

- **A4 — Faber (2007) monthly rule, 2×:** at each month-end, 2× S&P if the index
  closes above its 10-month simple average of month-end closes, else cash; held
  for the whole next month (traded at the first close of the month).
- **A5 — graded dual momentum (Antonacci 2014 absolute momentum + 200-day trend):**
  each session, signal 1 = index above its 200-day average; signal 2 = index
  12-month total return > the cash rate's 12-month return. Exposure 2× if both,
  1× if exactly one, cash if neither.

Same data, costs, delay, periods and gates as A1–A3 (CAGR > S&P TR in P1 and P2;
≥ 60% of rolling 10-year windows; max DD ≤ S&P's in each period). Exposure ledger:
A1–A3 tried → A4, A5 are trials 4–5 on the same 1928–2026 data, so a pass is
reported with the count. Sensitivity (reporting): 1- and 3-day delays.

## Exposure family 2 + combination results (2026-10-07 ~2:35 PM CT)

| Rule | P1 1928–93 (S&P 9.61%) | P2 1994–26 (S&P 10.90%) | P2 max DD (S&P 55.2%) | Sharpe P1 / P2 | Switches/yr | 10-yr windows | P2 at 1 / 3 / 5-day delay | Verdict |
|---|---|---|---|---|---|---|---|---|
| A4 Faber monthly 2× | 11.99% | **14.04%** | 48.6% | 0.42 / 0.55 | 1.4 | 73% | 14.69 / 14.79 / 14.11% | **PASS** |
| **A5 graded dual momentum** | **13.30%** | **13.67%** | **37.3%** | 0.49 / 0.55 | 4.1 | 80% | 12.10 / 13.21 / 14.40% | **PASS** |
| A1 (ref) | 14.55% | 12.33% | 45.7% | 0.52 / 0.51 | 6.9 | 86% | 11.11 / 11.07 / 12.85% | PASS |

Exposure ledger: 5 rules tried on 1928–2026, **3 pass** — the trend/absolute-momentum
+ leverage family works broadly, not one lucky rule. A5 has the best risk profile
(DD 37% vs 55%, worst year −30.5% vs −36.8%) and is delay-robust.

**Combinations (exploratory — C1's pre-registration said skip if v5b failed; run at
the owner's request), 2019-01 → 2026-09:** SPY TR 17.25%; A1 17.53% / on v5b book
17.56%; A4 13.89% / 13.31%; **A5 18.23% / 18.22%**. Holding the v5b book instead of
the S&P adds nothing under leverage: its weak years (2021 −6 pp, 2022 −3 pp vs SPY)
are doubled. All C-variants FAIL their gates. In this bull-dominated window only A5 is
clearly ahead of SPY (+1.0 pp/yr; A4's monthly checks were too slow for 2020).

**Candidate for the Lab: A5** (SSO when both signals, SPY when one, SGOV when none).
