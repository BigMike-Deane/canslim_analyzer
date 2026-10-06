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
