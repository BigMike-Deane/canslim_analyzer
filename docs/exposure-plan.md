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
