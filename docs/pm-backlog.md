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

## Next up (ranked by expected returns impact)
1. **Scan-universe coverage loss (root cause of the stale rows).**
   Real, liquid large-caps are absent from the scan universe: TEAM stale
   since Apr-21, BK since May-22, HOLX/MASI/ONON/MNDY/NTR/IOT/FROG since
   Jul-01 (~110 rows all last touched 2026-07-01 15:04 UTC — universe
   shrank that day). Missing names = missed buy signals, not just stale
   data. Investigate sp500_tickers.py universe build + FMP screener
   criteria; decide rescan vs. prune per ticker. Zombies (SBNY, ABMD,
   FLT→CPAY, IIVI→COHR, sub-$1 husks) should be marked delisted/pruned.
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
