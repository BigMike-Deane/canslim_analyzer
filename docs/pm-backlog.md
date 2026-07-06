# PM Work-Loop Backlog

Owner directive (2026-07-06): standing improvement loop — process, bugs, UI,
big or small — to build the best possible app for stock-market returns.
One item shipped per iteration, tests green, deploy on green.

Constraints (from auto-memory, do not relitigate):
- Don't reopen killed strategy levers (score floor, extension guard, winner
  trailing bands, rotation feature, cadence/D2 tuning).
- Don't re-enable ML without a LIVE A/B; don't trust ML backtests.
- Don't reformulate exits — divergence was parity debt, now fixed.
- Backtests cannot validate scorer changes (frozen-score replay).

## Done
- [x] 2026-07-06 `0c3857c` — Freshness gate on buy candidates.
      evaluate_buys excluded 134 scanner-abandoned rows (some frozen since
      March, several scoring 70-80) from the live buy pool.
      Config: `ai_trader.buy_candidates.max_staleness_hours: 48`.
- [x] 2026-07-06 `bd760e7` — FMP company-screener universe supplement.
      Small-cap sources had silently degraded (IWM rate-limited + its >500
      gate can never pass; Finviz 403) → scanner blind to ~1,367 active
      names (ONON, MNDY, FROG, IOT). Universe 2,081 → 3,474, live-verified.
      Config: `scanner.universe.fmp_screener`. Root-cause findings: most
      "missing large caps" were LEGIT index exits (BK→BNY rename, TEAM off
      N100, HOLX/MASI off S&P500) — iter-1 freshness gate handles their rows.
      ▶ WATCH: scan-cycle freshness for a few days (stocks_stale_1h) — if it
      degrades, tighten screener filters via config (no redeploy).

- [x] 2026-07-06 (iter 3, verification-only) — First expanded scan cycle
      VERDICT: 3471/3471 in 78.7 min (cold cache), 0 FMP 429s over ~9.5k
      calls, API responsive throughout, no interval change needed (90-min
      holds with ~11 min worst-case headroom; warm cycle expected faster).
      Payoff: 44 new names >= 72, 79 >= 67, 113 >= 64 (ELMD 87.6, ITIC 87.2,
      OPY 86.3, TSM 82.6). Leak found: HQH/HQL closed-end funds passed FMP
      isFund=false misclassification, scored ~78. Contained: manual
      DelistedTicker block + stocks rows score-zeroed; nothing was bought.

## Next up (ranked by expected returns impact)
1. **Systemic security-type guard (fund/ETF/CEF leakage).** FMP's
   isEtf/isFund flags miss closed-end funds (HQH/HQL proof). Add a
   structural check in the scan path — Yahoo quoteType != EQUITY →
   auto-blocklist (source='security_type') — so misclassified funds can
   never reach the buy pool. Also sweep existing new names below the buy
   zone for fund-like rows (harmless but waste scan calls).
2. **Dead-code cleanup: Yahoo IWM top_holdings path** in
   get_russell2000_tickers — can never pass its >500 gate (top_holdings is
   ~10 rows), wastes a rate-limited call + warning every cache refresh.
   4 tests mock it with unrealistic data; update them. Low returns impact,
   pure hygiene. Consider whether the whole Russell chain is redundant now
   that the screener supplement covers it.
2. **Stale rows in user-facing lists.** /api/stocks (dashboard ranking),
   screener, breaking-out lists — check whether scanner-abandoned rows
   appear with frozen scores; exclude or visibly flag. FLXS-80 at the top
   of a dashboard misleads.
3. **UI: position risk visibility.** Surface per-position trailing-stop
   distance (peak, tier %, exit price) on AIPortfolio cards — the data
   exists (exit_plan.py); confirm it's rendered and accurate; MU sitting
   -19% off peak within its 29% tier looks alarming without that context.
4. **Sector concentration check.** Live portfolio is heavy Financial
   Services (7 of 24 position rows). max_sector_allocation=0.50 /
   max_stocks_per_sector=4 are per-user; verify enforcement is working as
   intended on live data.

## Monitoring clocks (no action until due)
- ~mid-Aug: H-fix strategy-ab-eval re-check (cutoff 2026-07-02).
- ~late Jul: ML-demotion entry-rate re-check (baseline 0.367/day,
  cutoff 2026-07-04; expect entry_rate to RISE if veto was filtering).
- ~Sept: exit-reconciliation poller fires at ≥10 post-fix exits.
