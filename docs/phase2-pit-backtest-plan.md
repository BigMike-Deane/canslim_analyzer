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
