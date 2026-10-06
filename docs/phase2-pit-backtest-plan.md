# Phase 2: Point-in-Time Backtest — Scoping Plan (2026-10-05)

**Goal (owner, 2026-10-05):** trust the app enough to copy its trades with real
money, because it beats SPY consistently on *tested* evidence. Otherwise, index
funds. Phase 1 (`docs/signal-check-2026-10-05.md`) could not settle whether the
score picks winners: 13 weeks covers about 2 market regimes. This phase builds
10+ years of history with no lookahead, so H1–H4 can be tested properly.

## Why the existing backtester can't do this
- `create_backtest_static_snapshot` (`backend/backtester.py:149`) copies
  **today's** surprise, revision, ROE, inst% and earnings arrays into every past
  date. Switching to fresh compute took a 4-year backtest from **+136.8% to
  −15.0%** (docstring `backtester.py:2205`). Every pre-Aug backtest number is
  suspect for this reason.
- The fresh-compute path (`_calculate_scores`, layer 4) calls the real scorer
  for C only. A/N/S/L/I/M are inline copies that have **drifted** from
  `canslim_scorer.py` (`backtester.py:2370-2558`). Its earnings "PIT" filter
  just drops `(days_ago+45)//91` entries from today's array
  (`historical_data.py:855`), with no real report dates, and it cannot reach
  back to 2015.

## What history can and cannot be rebuilt (probed 2026-10-05)

| Live input | Feeds | Historical source | Status |
|---|---|---|---|
| Daily OHLCV, 52w high, volume | N, S, L | FMP `historical-price-eod/full` (delisted tickers work: SIVB, FRC, TWTR, ATVI, SGEN) | ✅ |
| SPY/QQQ/DIA MAs | M | FMP EOD | ✅ |
| Quarterly + annual EPS | C, A | **SEC XBRL companyfacts**: every filing's values with a `filed` date, i.e. as first reported (no restatement bias), ~2009→, delisted filers kept | ✅ free |
| Earnings surprise / beat streak | C bonus | FMP `earnings` (actual and estimate per report, back to 2002) | ✅ |
| Market cap (liquidity filter) | universe | FMP `historical-market-capitalization` | ✅ |
| Sector | C/A thresholds | FMP profile (current only) | ⚠ current sector, rarely changes; accepted |
| Institutional % | I | live = yfinance snapshot; FMP 13F history = **402 on our plan**; SEC 13F datasets (free, 2013→, heavy) | ❌ **owner decision** |
| `eps_estimate_revision_pct` (forward consensus growth) | C (+8/−4, excellence) | no archive of past consensus anywhere we can reach | ❌ dropped (neutral) |
| Yahoo adjusted EPS (overwrites FMP GAAP, `async_data_fetcher.py:1259`) | C | no history | ⚠ use GAAP diluted EPS (SEC) |
| Delisted universe | survivorship | FMP list is **page 0 only** (last ~100) on our plan; SEC keeps every filer by CIK | ⚠ **owner decision** / build CIK→ticker map |

**Consequence:** the PIT score will be *close to* the live score, not identical
to it (it uses GAAP EPS, has no forward-consensus term, and may have no I).
How close is measured, not assumed (M2 gate). If the PIT score passes the
history test, the live app moves onto the same PIT-clean scorer, so the
backtest and live trading run the same code from then on.

**Identity rule:** companies are keyed by SEC **CIK**, never by ticker. BBBY
has 2026 prices: the ticker was reused by a different company.

## Milestones and stop gates

| # | Deliverable | Est. | Gate (stop or re-plan if it fails) |
|---|---|---|---|
| **M0** Data audit | CIK universe per quarter 2011→2026 (SEC 10-Q/10-K filers); CIK→ticker-at-the-time map (FMP symbol/CIK, Alpaca inactive assets, XBRL `dei:TradingSymbol`); measured price coverage | 2–3 days | ≥ 90% of liquid filers (cap ≥ $300M) resolve to a price series in every year. Otherwise survivorship bias is too large → paid data |
| **M1** PIT data lake | Parquet store **outside prod**: prices (split-adjusted), as-first-reported EPS with `filed` dates, earnings surprises, index closes, market caps. One-time pull at ≤ 100 FMP calls/min **after hours** (live scanner shares the 300/min budget); SEC ≤ 10 req/s | ~1 week | Spot-check 20 names against the live DB/FMP; zero lookahead in automated "available-as-of" tests |
| **M2** PIT scorer adapter | Feeds PIT data into the **real** `CANSLIMScorer.score_stock` (no copies); snapshot-only inputs neutral; L aligned by date | ~1 week | On the 2026-07-07→10-05 overlap, PIT vs live `stock_scores.total_score` Spearman ≥ 0.80 and top-quintile overlap ≥ 60%. Below that → explain each gap before going on |
| **M3** Signal tests | H1–H4 from phase 1 on 2011–2026: tune on **2011–2020**, test once on **2021–2026** | 2–3 days | **H1 fails on history → recommend index funds** (owner's rule) |
| **M4** Portfolio sim | Live rules (8 positions, stops, trailing, SPY gate) on PIT scores, 19 bps/trade costs, vs SPY and IWM; risk-adjusted and drawdown | ~1 week | Out-of-sample excess return vs SPY not > 0 risk-adjusted after costs → index funds. Pass → phase 4 forward paper as *confirmation* |

Total ≈ 4 weeks of sessions. Nothing here touches the live trading path until
a gate passes and the owner approves a change.

## Side findings (verified, not fixed, not urgent)
- `scoring.canslim.*` keys in `config/default.yaml` are dead: the scorer uses
  hard-coded constants (`canslim_scorer.py:51-80`).
- L compares stock vs SPY by position (`iloc[-252]`), not by date. It's wrong
  when a stock has missing sessions. The PIT adapter aligns by date.
- The industry-group bonus is dormant in live scoring: `async_scanner.py:293`
  doesn't pass `industry_group_rank`.

## Owner decisions needed before M0
1. **Institutional (I):** the best phase-1 signal, and it has no history on our
   plan. (a) build from free SEC 13F datasets (2013→, ~+1 week); (b) upgrade
   FMP for 13F history (also unlocks the full delisted list); (c) test without I
   first and add it later.
2. **Where it runs:** local WSL (recommended: keeps prod's memory and FMP
   budget clear; the VPS has a 2.5 GB container limit) or the VPS.

## Amendment 2026-10-05 — history window 2016–2026 (owner-approved)
**Why.** The M0 survivorship gate failed for the early years. Among companies
with revenue ≥ $100M, price coverage was 66% (2011) rising to 94% (2021+), and
only 31% for companies that later disappeared. Cause: FMP carries few old
delisted stocks (2011 gap of 732: 297 recovered tickers with no FMP prices,
218 where FMP's series starts late or the ticker was reused, 204 with no
ticker, including never-public filers like Publix). Alpaca SIP daily bars (free with
the existing paper account) **do** keep delisted stocks (AET, MON, CELG
verified), but only from 2016. No free source covers 2010–2015.
**Rule.** The backtest window is **2016-01 → 2026**. M3: tune on **2016–2021**,
test once on **2022–2026**. M0 gate years: 2016–2025. FMP prices are filled
from Alpaca where FMP is missing or starts late. Decided before any M3 result
existed. (The 13F/I data from 2013 is unaffected.)

## M3 pre-registration (2026-10-05, written before any M3 number exists)
**Panel.** Every 10th NYSE session from 2016-01-04 to the latest date with 60
sessions of forward prices. Universe on each date: price > $3, market cap ≥
$300M (close × SEC shares outstanding as known that day), and a fresh price.
Scores come from `m2_adapter.score_asof` (the real scorer; passed M2). Forward
returns are 10/20/60 sessions, close to close, from stitched prices. A stock
that stops trading inside the window is marked at its last price and flagged
(delisting returns are unknown: optimistic for bankruptcies, conservative for
cash buyouts). "Excess" = stock return − equal-weight universe return that date.

**H1 — does the score sort winners? (THE GATE)**
Top-quintile minus bottom-quintile 20-session excess, on non-overlapping
dates (every 2nd panel date). PASS needs **mean > 0 with t ≥ 2.0 over
2016–2026 AND mean > 0 on 2022–2026 alone.** FAIL → recommend index funds
(owner's rule). Reported beside it, not gating: the 72+ band vs the universe,
10/60-session horizons, and the trend/chop split.

**H2 — over-extension.** Mean 20-session excess of the 60–72 band minus the 72+
band. Supported if > 0 with t ≥ 2 on 2016–2021 AND > 0 on 2022–2026.

**H3 — reweighting.** On 2016–2021 only, grid-search integer weights 0–3 on
C, A, N, S, L, I (M is the same for every stock on a date, so it can't rank).
Objective: the H1 spread. Then **one** evaluation on 2022–2026 vs the current
weights (all 1). Supported only if it beats current weights on 2022–2026.

**H4 — momentum is regime-dependent.** Quintile spreads of L, N and S, split by
SPY above/below its 50-day MA on the score date. Supported if the
above-minus-below difference is > 0 with t ≥ 2 on 2016–2021 AND same sign on
2022–2026.

Every test also reports the 2016–2021 and 2022–2026 halves separately, and the
residual survivorship caveat (later-delisted coverage 71–79%).

## M3 results (2026-10-06, single run, rules as pre-registered above)

Panel: 647,304 stock-dates, 265 dates (every 10th session 2016-01-04 → 2026-07-07),
4,839 companies, median 2,419 stocks/date, 0 scoring errors; 0.39% of 20d windows
cut short by delisting. Script `research/pit/m3_signal_tests.py`, data
`~/canslim_pit_data/meta/m3_panel.csv.gz` (not in git), log `meta/m3_results.log`.

| Test | Result | Numbers (20-session excess vs equal-weight universe, non-overlapping dates) |
|---|---|---|
| **H1 (gate)** total-score top − bottom quintile | **FAIL** | full +15 bps (t +0.40, n 133); 2016–21 −15 (t −0.29); 2022–26 +55 (t +0.97) |
| H2 60–72 band − 72+ band | not supported | full −24 bps (t −0.89); 2016–21 −25; 2022–26 −22 (phase-1 "over-extension" did not replicate) |
| H3 reweight C/A/N/S/L/I on 2016–21 | not supported | best train weights A3 N1 (train +30 bps) → 2022–26 +51 bps (t +0.96) vs current +55 (t +0.97) |
| H4 L/N/S spread × SPY>50MA | not supported (all 3) | 2016–21 above−below L +227 (t 1.77), N +123 (t 1.08), S +121 (t 1.91); all flip negative 2022–26 |

Reporting only: 10-session spread +7 bps (t 0.26); 60-session +91 bps (t 1.38).
Components (full, 20d): C +16, A −5, N +10, S −30 (t −1.43), L +6, I −10 — none
reach t 2. 72+ buy band vs universe +12 bps (t 0.38); universe vs SPY −26 bps
(t −1.12) → the buy band trailed SPY by ~14 bps per 20 sessions.

Caveats (none plausibly flips the verdict): C component fidelity is weakest
(M2 ρ 0.38–0.68; GAAP not adjusted EPS, no analyst-revision term); residual
survivorship (later-delisted coverage 71–79%) — missing delisted names are more
likely low scorers, which biases the spread *up*, so the true spread is if
anything lower. The test is cross-sectional: the M (market-timing) component is
constant within a date and is not evaluated here.

**Verdict per the pre-registered rule: H1 FAIL → recommend index funds.**
M4 (portfolio sim) is not run as a gate; the selection signal it would
compound has no detectable edge over 10 years.

## Market-timing test pre-registration (2026-10-06, written before any number exists)

Owner chose (Oct-6, after M3 H1 FAIL): run the one CANSLIM piece M3 could not
see — market timing — then a Phase 3 signal lab. This test asks only whether the
timing *signal* beats holding SPY; it is not a simulation of the live book.

**Rules (SPY ↔ cash, daily):**
- **T1 (primary) = the live champion's gate:** in SPY when SPY close > its 50-day
  simple average, else cash (`nostate_optimized` has market_state off → legacy
  `market_regime_gate`, `ai_trader.py` "spy_px < spy_50").
- **T2 (secondary) = the scorer's M:** in SPY when the composite M
  (`data_fetcher.calculate_index_m_score` × `MARKET_INDEX_WEIGHTS` over
  SPY/QQQ/DIA, renormalized over indexes with ≥ 200 sessions) ≥ 7.5 of 15.
- Signals use split-adjusted closes (as live); returns use dividend-adjusted
  closes (total return). Signal at close of day t → position held from close of
  t+1 (one full day of lag; same-close execution reported as sensitivity only).
- Cash earns the 3-month T-bill (FRED DTB3, annual % / 252 per session).
- Cost: 5 bps per switch, each direction.

**Periods:** A = 1994-01-03 → 2015-12-31 (never examined in this program),
B1 = 2016–2021, B2 = 2022-01-03 → 2026-10-05. Data: FMP `historical-price-eod`
dividend-adjusted + full, fetched Oct-6, in `~/canslim_pit_data/timing/`.

**Pass rules (applied to T1; T2 reported by the same rule):**
- **PASS (beats SPY):** timed CAGR > buy-and-hold SPY CAGR, after costs, in
  **all three** periods A, B1, B2.
- **RISK TOOL (does not meet the beat-SPY goal):** not PASS, but timed Sharpe >
  buy-and-hold Sharpe AND timed max drawdown shallower, in all three periods.
- **FAIL:** anything else.

Reporting only: CAGR, Sharpe (excess of T-bill), max drawdown, % time invested,
switches per year, worst whipsaw month, Memmel-corrected Jobson–Korkie test of
the Sharpe difference over the full 1994–2026 span.

Prior, stated before running: published work on 50/200-day trend rules in US
large caps mostly finds lower drawdowns and lower-or-similar returns in
bull-dominated decades, so RISK TOOL or FAIL is more likely than PASS.

## Market-timing results (2026-10-06, single run, rules as pre-registered above)

Script `research/pit/t1_timing.py`, log `~/canslim_pit_data/meta/t1_timing.log`.

| Rule | Period | Timed CAGR | SPY CAGR | Sharpe (timed / SPY) | Max DD (timed / SPY) | Invested | Switches/yr |
|---|---|---|---|---|---|---|---|
| **T1** SPY>50MA | A 1994–2015 | +3.95% | +8.93% | 0.17 / 0.41 | −35.3% / −55.2% | 65% | 19.2 |
| | B1 2016–21 | +10.14% | +17.28% | 0.87 / 0.92 | −15.2% / −33.7% | 76% | 14.7 |
| | B2 2022–26 | +5.86% | +12.39% | 0.21 / 0.53 | −21.1% / −24.5% | 68% | 17.7 |
| **T2** composite M≥7.5 | A 1994–2015 | +6.00% | +8.93% | 0.33 / 0.41 | −29.6% / −55.2% | 71% | 14.2 |
| | B1 2016–21 | +11.30% | +17.28% | 0.90 / 0.92 | −12.7% / −33.7% | 85% | 10.3 |
| | B2 2022–26 | +9.98% | +12.39% | 0.56 / 0.53 | −15.6% / −24.5% | 75% | 9.3 |

Sharpe difference 1994–2026 (Memmel z): T1 −1.35, T2 −0.36. Same-close
execution changes CAGR by < 1.5 pp and flips no verdict.

**Verdict: T1 FAIL, T2 FAIL.** Both cut drawdowns but lose 2.4–7 pp/yr of CAGR
in every period, and neither has a higher Sharpe in all three. Trend timing on
SPY does not beat holding SPY. Caveat for the live book: its gate only blocks
*new buys* (positions exit by their own stops), so the live drag is smaller
than T1's full-cash drag, but this offers no evidence the gate adds value.

## Phase 3 signal-lab pre-registration (2026-10-06, written before any number exists)

Owner chose (Oct-6) to look for an edge outside CANSLIM after M3 H1 FAIL and
timing FAIL. Five hypotheses, fixed definitions, **no tuning**, one run.

**Panel:** the M3 panel exactly (same 265 dates, same universe: actual price >
$3, market cap ≥ $300M as known; same forward returns and exclusions).

**Signals (all known strictly before the panel date D):**
- **S1 PEAD (earnings-surprise drift):** (epsActual − epsEstimated) / actual
  close on D, from the most recent FMP earnings report dated *strictly before* D
  and within the last 63 sessions; else missing. Higher = better.
- **S2 Gross profitability:** latest 10-K fiscal year (filed ≤ D): gross profit
  ÷ total assets at that fiscal year end. Gross profit = GrossProfit, else
  revenue − (CostOfRevenue | CostOfGoodsAndServicesSold | CostOfGoodsSold).
  Assets = Assets at the same period end, same or earlier filing ≤ D. Missing if
  either side missing or assets ≤ 0. Higher = better.
- **S3 Net share issuance:** −log(shares_now / (shares_then × splits between)),
  shares = dei EntityCommonStockSharesOutstanding as known on D vs as known on
  D − 365 days, split ratios from FMP /splits (`common.split_factor`). Higher
  (= buybacks) = better.
- **S4 Momentum 12−1:** close(D − 21 sessions) / close(D − 252 sessions) − 1 on
  the company's stitched price series; missing if fewer than 252 sessions.
- **S5 Composite (fixed, untuned):** mean of the per-date percentile ranks of
  S1–S4, requiring ≥ 3 of 4 present.

**Tests (per signal, 20-session horizon, non-overlapping dates as in M3):**
- **(a) Signal is real:** top − bottom quintile excess spread: full 2016–2026
  **t ≥ 3.0** (raised from 2.0 because five hypotheses are tested and the
  2022–26 holdout has been looked at once), AND mean > 0 in 2016–21 AND in
  2022–26.
- **(b) Usable long-only to beat SPY:** top-quintile equal-weight 20-session
  return − SPY 20-session return − cost > 0 in 2016–21 AND in 2022–26.
  Cost = top-quintile turnover between consecutive non-overlapping dates × 19
  bps round trip (the program's cost assumption).
- **PASS = (a) AND (b)** → the signal advances to M4 portfolio simulation
  (8 positions, live-style exits, costs, vs SPY and IWM). **No signal passes →
  recommend index funds and close the lab.**

Reporting only: 10/60-session spreads, top-quintile vs IWM, coverage per
signal, rank correlation of each signal with log market cap.

Known caveats, stated before running: FMP epsEstimated is FMP's historical
consensus (not verified PIT); FMP earnings are keyed by current symbol; residual
survivorship as in M3 (later-delisted coverage 71–79%), which flatters
long-only results.

**Amendment before any result (2026-10-06):** FMP earnings EPS are
*split-adjusted and rounded to cents* (NVDA 2016 epsActual 0.01–0.02; one 2016
estimate 0.29 left unadjusted). S1's denominator is therefore the
**split-adjusted** close on D (same basis as the EPS), not the actual close.
Rounding makes S1 coarse for heavily split stocks and stray unadjusted
estimates create outliers; quintile ranks limit the damage. Stated as a caveat.

## Data-integrity fix found during phase 3 sample checks (2026-10-06, before any phase 3 result)

Sample outlier review (S3) exposed an **identity bug that also touched M3**:
a ticker reused by two SEC filers whose filing windows overlap (FI = Frank's
International 2013–21, later Fiserv) handed *both* CIKs *both* CUSIPs, so
Frank's/Expro was priced with Fiserv's series. Before the fix 243 CUSIPs were
claimed by > 1 CIK (246 panel CIKs, ~6% of M3 rows), most of them sibling
filers (subsidiaries, LPs, private funds such as two BlackRock credit funds on
BLK) duplicating a listed parent.

Fix (`m0_identity.py`): (1) when several issuers held the symbol inside the
filing window, keep those whose FTD description shares a name token with the
CIK's SEC names; (2) each still-shared CUSIP goes to the claimant whose SEC
name is best explained by the description (≥ 0.6 of tokens, strict winner;
ties → the CIK FMP's profile names); (3) a CIK left with no CUSIP of its own is
`ambiguous` and excluded from the panel (60 panel CIKs, ~1.4% of rows; 12
tickers lose all history, e.g. AGN, HDS). Shared CUSIPs after fix: 0.

Consequences: symbol segments, the M3 panel and the M3 tests are **re-run on
the corrected identity** with unchanged rules; both runs are reported (v1 =
`m3_panel_v1_oct05.csv.gz`, results above). Noise of this kind biases spreads
toward zero, so it could hide a weak signal but not create one.

S3 definition hardening (same review, before results): split ratios are taken
across every ticker the CIK used (a reverse split filed under the later ticker
was missed); both share counts must be filed within 150 days of the date they
stand for (stale 2016 count used for a 2018 "then"); S3 is missing if a
bankruptcy ticker (4 letters + Q) trades inside the window (cancelled old
equity looked like a buyback, VAL 2021).

## M3 re-run on corrected identity (2026-10-06)

Panel v2: 636,775 rows, 4,774 companies, 0 errors. H1 **FAIL** unchanged: total
spread +16 bps/20d (t +0.41); 2016–21 −14, 2022–26 +56 (t +0.98). H2/H3/H4
not supported (numbers within ±0.1 t of v1). 72+ band vs universe +11 bps.
Log `meta/m3_results_v2.log`.

## Phase 3 signal-lab results (2026-10-06, single run, rules as pre-registered)

Script `research/pit/p3_tests.py`, log `meta/p3_results.log`. Coverage S1 85%,
S2 54%, S3 91%, S4 97%, S5 87%.

| Signal | (a) top−bottom 20d: full (t) / 2016–21 / 2022–26 | (b) top quintile − SPY net: 2016–21 / 2022–26 | Verdict |
|---|---|---|---|
| S1 PEAD | +42 bps (t 2.69) / +38 / +46 | +12 / −23 | FAIL (t < 3; (b) no) |
| S2 gross profitability | +11 (t 0.44) / +49 / −39 | +18 / −68 | FAIL |
| S3 net issuance | +47 (t 1.47) / +59 / +33 | +14 / −24 | FAIL |
| S4 momentum 12−1 | +37 (t 0.81) / −19 / +111 | −8 / +3 | FAIL |
| S5 composite | +57 (t 1.81) / +50 / +66 | +10 / −14 | FAIL |

**Verdict: LAB FAIL — no signal passes → recommend index funds** (the
pre-registered rule).

Reporting-only observations (not passes; may motivate a new, separately
pre-registered hypothesis, never a re-scored one): the signals *do* sort stocks
at longer horizons — 60-session top−bottom spreads S1 +141 bps (t 5.06), S5
+154 (t 3.46), S3 +137 (t 2.53), positive in both halves for S1/S3/S5. The
binding failure is (b): the equal-weight universe trailed SPY by ~26 bps/20d
(≈ 3.4 %/yr) over 2016–2026, so even the best quintile only roughly matches SPY
(S5 top − SPY ≈ 0; top − IWM +17 bps). Much of the spread is avoiding losers
(bottom quintile), which a long-only portfolio cannot fully monetize.

## Batch 2 / Score v2 pre-registration (2026-10-06, written before any number exists)

Owner (Oct-6): step back and build our own score from what the tests showed;
chose "batch 2 on free data now + research a clean 2000–2015 dataset". Value,
low volatility and quality have **never been examined** in this program, so
they are fresh tests on 2016–2026. Score v2 also contains S1/S3 (seen in phase
3), so its result is labelled **partly in-sample** and cannot by itself justify
real money; a fully clean test needs 2000–2015 data (vendor research pending).

**Universe:** the corrected M3 panel (v2), restricted per date to the **1,000
largest by market cap as known that day** (actual close × SEC shares).

**Signals (known strictly before D unless stated; no tuning):**
- **V value** = mean of per-date percentile ranks of (i) earnings yield = sum of
  the latest 4 quarterly diluted EPS as known on D (distinct period ends, newest
  within 15 months; derived Q4 as in M2) ÷ actual close on D, and (ii)
  book-to-market = latest StockholdersEquity (instant, filed ≤ D) ÷ market cap.
  Either alone if the other is missing.
- **LV low volatility** = − standard deviation of daily returns over the last
  252 sessions to D on the stitched series (≥ 200 returns required).
- **Q quality** = mean of percentile ranks of (i) ROE = latest 10-K FY net
  income ÷ StockholdersEquity at that FY end (equity > 0), and (ii) −accruals =
  −(net income − operating cash flow) ÷ total assets, same 10-K FY. Either alone
  if the other is missing.
- **SCORE v2** = mean of percentile ranks of V, LV, Q, S1 (PEAD), S3 (net
  issuance), requiring ≥ 4 of 5 present. Equal weights, fixed.

**Horizon:** primary **60 sessions**, using every panel date (overlapping
windows) with **Newey–West t-statistics, 6 lags**. 20 sessions reported only.

**Pass rules (each of V, LV, Q, SCORE v2):**
- **(a)** top − bottom quintile 60-session excess (vs equal-weight universe
  mean): full 2016–2026 **NW t ≥ 3.0**, AND mean > 0 in 2016–21 AND 2022–26.
- **(b)** a long-only version beats SPY after costs in **both** halves:
  either (b1) top quintile equal-weight 60-session return − SPY − turnover × 19
  bps, or (b2) "index minus losers": cap-weighted universe excluding the bottom
  quintile − SPY − turnover × 19 bps. Turnover measured between panel dates 6
  apart (non-overlapping 60-session rebalances).
- **PASS = (a) AND (b).** A passing V/LV/Q is fresh evidence. A passing SCORE
  v2 advances to (1) the clean 2000–2015 test if data is obtained, and (2) a
  portfolio simulation + forward paper trading. **Nothing passes → index funds,
  research closes.**

Reporting only: S1–S4 re-run on this large-cap universe at 60 sessions (seen
before, not evidence), coverage, rank correlation with log market cap.

**Batch 2 amendment before any result (2026-10-06):** sample review found SEC
unit errors (NPO E/P ±95,000; REAL accruals −102× assets). Glitch bounds:
E/P set missing when |E/P| > 5; accruals set missing when |accruals| > 2×
assets. Real extremes are kept (MTD ROE 35 from buyback-shrunk equity).

## Batch 2 / Score v2 results (2026-10-06, single run, rules as pre-registered)

Universe top 1,000 by market cap/date: 265,000 rows, 2,134 companies. Signals
0 errors. Script `research/pit/p4_tests.py`, log `meta/p4_results.log`.

| Hypothesis | (a) top−bottom 60d: full (NW t) / 2016–21 / 2022–26 | (b1) top EW − SPY net halves | (b2) cap-wt ex-bottom − SPY net halves | Verdict |
|---|---|---|---|---|
| V value (E/P + B/M) | +3 bps (t 0.03) / −47 / +71 | −81 / −96 | +47 / −14 | FAIL |
| LV low volatility | −151 (t −1.16) / −126 / −185 | −143 / −212 | −161 / −101 | FAIL |
| Q quality (ROE + low accruals) | +82 (t 1.95) / +135 / +11 | +7 / −139 | +3 / −40 | FAIL |
| SCORE v2 (V, LV, Q, PEAD, issuance) | −4 (t −0.05) / −4 / −5 | −88 / −95 | +41 / −36 | FAIL |

**Verdict: BATCH 2 FAIL → per the pre-registered rule, recommend index funds;
research closes.** Low volatility was strongly *negative* (low-vol large caps
trailed a high-beta, mega-cap-led decade) and cancelled the other inputs in
Score v2. Reporting only (seen before): in large caps at 60d, S1 PEAD +108 bps
(t 2.76) and S3 net issuance +139 (t 2.16) stay positive in both halves — the
only consistent effects found in the program, neither at the t ≥ 3 bar and
neither beats SPY long-only after costs.

### Program summary (2026-10-05 → 10-06)
Five pre-registered gates, all on point-in-time data with delisted names:
CANSLIM score (M3 H1, t 0.41) FAIL; SPY market timing (T1 live gate, T2
composite M, 1994–2026) FAIL; phase 3 lab (PEAD, gross profitability, net
issuance, momentum, composite) FAIL; batch 2 (value, low vol, quality, Score
v2, large caps) FAIL. No tested approach beats SPY long-only after costs over
2016–2026. Clean 2000–2015 data is available (Sharadar, ~$39–69 for one month)
but there is no candidate strong enough to justify the spend.

## "Test it the way it trades" pre-registration (2026-10-06, written before any number exists)

Owner (Oct-6), not ready to stop: the live book's +20.7% vs SPY ≈ +18% since
April rests on one trade (DELL +$3,877 ≈ 15.5 pp; all other realized trades
≈ −$190). DELL entered through the **pre-breakout setup path** (score 69 < 72;
"cup, 2% below pivot", "Est↑ +27%"). Prior tests measured *average* ranking;
CANSLIM claims to find *outliers* and monetize them with exits. Fresh questions,
never examined: big-winner odds, the setup signal, and the live rules as a
strategy. Analyst-revision history is a data gap (vendor research running).

**Panel:** corrected M3 panel v2 (same dates/universe). New fields per row, all
known on D: live `TechnicalAnalyzer.detect_base_pattern` on the last 26 weekly
bars (W-FRI resample of the stitched daily series ending at D — the live app
reads 6 months of Yahoo weekly bars), `TechnicalAnalyzer.is_breaking_out` on the
PIT StockData (`m2_adapter.stock_data_asof`), and `trading_engine.
calculate_entry_signals` → `entry_type` ∈ {pre-breakout, breakout, standard}.
Forward 120-session return r120 (same exclusions as M3: windows spanning a
ticker change dropped; delisting = last price, flagged).

**H5 big-winner odds (groups: G72 = total ≥ 72; GSET = entry_type
pre-breakout; GSET65 = pre-breakout AND total ≥ 65):**
- Per date: big-winner rate (r120 ≥ +50%) in the group minus the universe rate.
  Every panel date, Newey–West t (12 lags).
- **PASS** a group: mean difference > 0, **NW t ≥ 3**, > 0 in 2016–21 AND
  2022–26, AND the group's blowup rate (r120 ≤ −30%) is not higher than the
  universe's by more than its big-winner excess (net tail edge ≥ 0 in both halves).
- Reporting only: +100% rate, mean r120 excess, group size per date.

**H6 setup signal (GSET, GSET65; plus breakout):** per-date mean 60-session
excess vs universe; **PASS** = NW t (6 lags) ≥ 3, > 0 in both halves, AND group
equal-weight 60d return − SPY − 19 bps × turnover > 0 in both halves.

A pass in H5/H6 is necessary but not sufficient: the decision gate is **H7**,
the live rules simulated as a strategy vs SPY, pre-registered separately before
it runs. **Nothing passes H5/H6 → H7 is still run once** (exits could monetize
tails the averages miss), and its result is final for this program.

## H5 / H6 results (2026-10-06, single run, rules as pre-registered)

Setup replay: 636,775 rows, 0 errors; pre-breakout 59% of stock-dates (live
scanner today: ~50% of $3+/$300M+ stocks sit 0–15% below a detected pivot —
replay faithful; "pre-breakout" is common by construction). Universe per-date
rates over 120 sessions: big winner (≥ +50%) 6.47%, +100% 1.53%, blowup
(≤ −30%) 10.05%. Log `meta/p5_results.log`.

| Test | Group | Result |
|---|---|---|
| H5 big-winner odds | G72 score ≥ 72 (51/date) | −0.11 pp vs universe (NW t −0.18); blowups −2.7 pp; mean r120 excess −1.3% → **FAIL** |
| | GSET pre-breakout (1,475/date) | **−1.92 pp (NW t −6.60)**; blowups −2.8 pp; r120 excess −2.4% → **FAIL** |
| | GSET65 pre-breakout & score ≥ 65 (86/date) | **−1.71 pp (NW t −4.03)**; blowups −3.4 pp; r120 excess −2.7% → **FAIL** |
| H6 setup 60d excess | GSET | −24 bps (t −0.79); EW − SPY net −113 → **FAIL** |
| | GSET65 | +5 bps (t 0.11); EW − SPY net −85 → **FAIL** |
| | BRK breaking out (23/date) | −77 bps (t −1.43); EW − SPY net −192 (t −3.13) → **FAIL** |

Reading: CANSLIM setups and high scores select **quieter** stocks — fewer
blowups AND significantly *fewer* big winners than the average stock (opposite
of the outlier premise), with slightly negative mean returns. Breakouts lag.
Per the pre-registration, **H7 (live rules as a strategy vs SPY) still runs
once and is final for this program.**

## H7 pre-registration — the live rules as a strategy (2026-10-06, before any build output)

**Engine:** `backend/backtester.py` (the live trader's mirror) **unmodified in
its trading logic**, subclassed only for data: a point-in-time data provider
(stitched per-CIK prices incl. delisted, universe as of each date) and
`_calculate_scores` served through the existing frozen-score path
(`_build_score_from_frozen`) from a **daily PIT score table** (live
`CANSLIMScorer` via `m2_adapter.score_asof`, every session 2016-01-04 →
2026-10-05, every company eligible that day: fresh price, actual close > $3,
market cap ≥ $300M as known). `projected_growth` uses the backtester's own
formula (EPS growth × 0.30 + annual CAGR × 0.25 + RS momentum × 0.45) from PIT
inputs. Earnings dates for avoidance / coiled-spring checks come from FMP
earnings history (dated, PIT). No start-date survivorship filter (IPOs enter
when eligible; delisted names exit at their last price).

**Profile:** `nostate_cs_bear` (the owner's live config), $25,000 start.
**Costs:** 0.095% per side added to every fill (19 bps round trip); the engine
has none of its own.

**Runs:** five launch vintages (start offsets 0/10/20/30/40 sessions from
2016-01-04) to absorb path noise (twin live books diverge ~7 pp).

**PASS (all required):** the **median vintage's** CAGR > SPY total-return CAGR
(dividends included) over 2016–21, over 2022–26, and over the full period.
Reported: CAGR, Sharpe, max drawdown vs SPY, trades, win rate, and the share of
total profit from the single best trade (DELL-dependence check).

**Known fidelity gaps (stated before running):** C uses GAAP EPS with no
analyst-revision term (M2 C ρ 0.38–0.68); institutional % from 13F; live ML
veto and growth-projection model not modeled; universe approximates the live
index-plus-screener list. **FAIL → recommend index funds for real money; the
program's strategy research closes** (paper trading may continue as a hobby).

## Data-integrity fixes found while building H7 (2026-10-06, before any H7 output)

Setting up H7 showed DELL (the owner's best live trade) absent from every
panel. Root causes, all affecting **every test run today**:
1. **No shares count for multi-class filers.** companyfacts omits per-class dei
   `EntityCommonStockSharesOutstanding`; ~510 CIKs (META, DELL, Alphabet's
   CIK…) had none, failed the market-cap filter, and never entered a panel.
   Fix (`m2_adapter._shares_table`): fall back to weighted-average diluted
   shares (all classes), dated by filing — recovers 495.
2. **Wrong security picked.** FMP search-cik lists every security a CIK issues;
   the first plain candidate was sometimes a note/ETN (JPM → AMJ MLP ETN,
   Comcast → CCZ exchangeable notes, Prudential → PFH). Fix (`m0_ticker_fix.py`):
   if the pick never traded under a CUSIP with issue number "1x" (common-stock
   convention) and another plain candidate did (and traded ≥ half as often;
   5-letter special-security codes excluded), switch — 99 CIKs (JPM, CMCSA,
   PRU, HIG, DTE…).
3. **Universe gap.** Filers reporting EPS only per class (Visa) have no
   undimensioned EPS frame. Fix (`m0_universe.py`): also admit CIKs with
   NetIncomeLoss frames (+3,431 CIKs, mostly non-traded; Visa, Constellation
   Brands among them).

**All gates and tests are re-run on the corrected data ("v3")** — M0 coverage,
M3, phase 3, batch 2, H5/H6 — with unchanged rules, and both versions are
reported. v2 artifacts kept as `*_v2_oct06.*`. H7 runs on v3 only. Alpaca
delisted-price fill is not re-run for the new CIKs (they are mostly current
companies); coverage is re-checked.

**v3 coverage gate (2026-10-06 ~6 PM CT):** the script's blended figure fell
(2016: 84.8%) because the NetIncomeLoss additions bring ~150–450 non-traded
revenue ≥ $100M filers per year (finance subsidiaries, privately held issuers
of public debt) into the denominator — 55–73% of them priced. **Like-for-like
on the original EPS-reporting universe the gate PASSES: 90.6 / 91.5 / 93.1 /
93.9 / 90.2 / 94.3 / 95.3 / 94.8 / 94.5 / 95.9% (2016–2025)**, ≥ v2. The traded
additions (Visa, Constellation Brands, …) enter the panel; the untradeable ones
cannot, by construction.

## v3 re-runs on fully corrected data (2026-10-06 evening, rules unchanged)

Panel v3: 701,326 rows, 5,454 companies (v2: 4,774), median 2,639 stocks/date,
0 errors; META, DELL, JPM, V, CMCSA, PRU now present and correctly priced.
Logs `meta/*_v3.log`. **Every verdict is unchanged:**

| Test | v2 | v3 |
|---|---|---|
| M3 H1 total-score spread 20d | +16 bps, t 0.41 FAIL | **+17 bps, t 0.45 FAIL** (2016–21 −5, 2022–26 +47) |
| M3 72+ band vs universe | +11 bps | +6 bps; H2/H3/H4 not supported |
| S1 PEAD | t 2.69 FAIL | t 2.75 FAIL (both halves +42 bps) |
| S2 / S3 / S4 / S5 | t 0.44 / 1.47 / 0.81 / 1.81 | t 0.35 / 1.51 / 0.89 / 1.93 — all FAIL |
| V value 60d | t 0.03 FAIL | t −0.12 FAIL ((b2) passed, (a) did not → noise) |
| LV low volatility | −151 bps, t −1.16 | −152 bps, t −1.16 FAIL |
| Q quality | t 1.95 | t 2.19 FAIL (2022–26 +32 only) |
| SCORE v2 | t −0.05 | t −0.01 FAIL |
| H5 G72 big-winner odds | −0.11 pp | −0.01 pp FAIL |
| H5 GSET / GSET65 | −1.92 / −1.71 pp (NW t −6.6 / −4.0) | −1.94 / −1.59 pp (NW t −6.7 / −5.1) FAIL |
| H6 GSET / GSET65 / BRK | all FAIL | all FAIL (BRK − SPY net −183 bps, t −2.93) |

The data fixes moved estimates by hundredths. H7 runs on v3.
