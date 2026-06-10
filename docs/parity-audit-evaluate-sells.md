# Parity Audit: `evaluate_sells()` ↔ backtester `_evaluate_sells()`

**Date:** 2026-06-10 (read-only audit during June-18 freeze — no code changed)
**Scope:** the automated live sell path (`ai_trader.evaluate_sells` + its executor in
`run_ai_trading_cycle`) vs the backtester sell path (`backtester._evaluate_sells`),
including shared helpers and data sources.
**Purpose:** input for the June-19 post-freeze exit-logic session
(routine `trig_01U9dF3HiUnTJB927uMD4A7F`). The June-9 conclusion was that the exit
problem is **parity, not formulation** — this is the systematic divergence inventory.

Every divergence is tagged with its bias direction:
- **backtest-optimistic** — backtests overstate what live will achieve
- **live-pessimistic** — live exits earlier/tighter than the backtest models
- **behavioral** — decisions differ, net P&L direction unclear

---

## High severity

### D1 — `new_position_guard` missing in `evaluate_sells` (KNOWN, fix staged)

- **Live:** `ai_trader.py` ~1623→1626 goes straight from `calculate_atr_stop()` to the
  stop check. No guard clamp.
- **Present in:** `_check_and_execute_stop_losses_impl` (ai_trader.py ~1304-1314) and
  backtester (~2551-2565).
- **Aggravator:** the guarded function is only reachable via the manual endpoint
  (`main.py` ~3790). The scheduler (`scheduler.py` ~1586) only ever calls
  `run_ai_trading_cycle` → `evaluate_sells`, so **the guarded path never runs
  automatically in production.**
- **Effect:** fast-falling new buys ride the ATR-widened stop (up to the 20% cap)
  instead of being clamped to 8% in their first 21 days. Confirmed live: FSLR −18%,
  AMD −13.7% (June-4 buys, stops unfired).
- **Bias:** backtest-optimistic.
- **Fix (June 19):** port the ~10-line guard block from
  `_check_and_execute_stop_losses_impl` into `evaluate_sells` right after the
  `calculate_atr_stop()` call. Mirror of backtester lines 2551-2565. The
  `stop_guard_monitor.py` sentinel test trips when this lands — delete the monitor
  in the same commit.

### D2 — Score-crash clock: *scans* live vs *trading days* in backtest

Both sides call the same shared `evaluate_score_crash()` (trading_engine.py:579) with
`consecutive_required=3` and the "2 of last 3" gate — but they feed it different clocks:

- **Live:** `check_score_stability` (ai_trader.py:509) pulls the last N `StockScore`
  rows. Scans run ~every 90 minutes, so "3 consecutive low scans" ≈ **4.5 hours** and
  "2 of last 3" can confirm in ≈ 3 hours — all within a single bad afternoon.
- **Backtest:** `_update_score_history` (backtester.py:2465) appends **once per
  trading day**; 3 consecutive low = **3 trading days**, and the 5-score window is a
  full trading week.
- **Effect:** live sells score crashes far faster than any backtest models, and live's
  blip protection window is hours, not days. A one-day data wobble that the backtest
  would shrug off can fully confirm a "crash" live.
- **Bias:** behavioral (live trigger-happy). Same bug *family* as the Jun-05 score
  replay finding (raw scan rows vs per-day dedup).
- **Fix (June 19):** dedup the live stability query to one score per calendar day
  (e.g. last-per-date or max-per-date via SQL, same pattern as the `536ce92` score
  history fix) before feeding `check_score_stability_from_history`. Decide
  deliberately whether "1 per day" should be last-scan or worst-scan — backtest uses
  the day's single score, so last-scan-of-day is the closer mirror.

---

## Medium severity

### D3 — ATR stop: data source, failure mode, determinism

The math is identical (14-period simple TR average, ×2.5 multiplier, capped at
`max_stop_pct` 20, floored at base stop). The plumbing is not:

- **Live:** `calculate_atr_stop` (ai_trader.py:1134) does a **synchronous Yahoo chart
  HTTP fetch per position per cycle** (5s timeout) and on *any* failure silently
  returns the base stop (champion: 7%).
- **Backtest:** `HistoricalDataProvider.get_atr` (historical_data.py:945) from local
  history — always available, deterministic.
- **Effect:** when Yahoo throttles or 404s, a position's stop silently snaps from
  (say) 18% to 7% for that cycle, then back. Live stop levels are non-deterministic
  across cycles; the backtest never models a tight-stop cycle.
- **Bias:** live-pessimistic (random premature stop-outs), plus noise.
- **Fix candidates (June 19):** cache ATR per ticker per day; on fetch failure fall
  back to last-known ATR instead of base; or compute from the same
  `HistoricalDataProvider` the VIX proxy already instantiates in the same function.

### D4 — Peak price: intraday high-watermark vs daily-close watermark

- **Live:** `update_position_prices` (ai_trader.py ~1090-1106) raises `peak_price` on
  every cycle from live intraday prices, and peak *initialization* even backfills
  from historical **daily highs**.
- **Backtest:** `_update_positions` (backtester.py:1981) tracks peak from **daily
  close** only.
- **Effect:** live peak ≥ backtest peak systematically, so `drop_from_peak` is larger
  live and trailing stops (and the giveback-floor lever, if ever enabled) fire
  earlier live than in backtest.
- **Bias:** behavioral / live exits earlier. Same lesson class as the Edge Scorecard
  intraday-max-vs-close preview error (jun-01).
- **Fix:** decide which is *correct* (intraday peak is arguably truer to O'Neil) and
  make the backtester model it (e.g. track peak from daily highs), rather than
  degrading live. This is a backtester change → freeze-safe if wanted early.

### D5 — Evaluation cadence: intraday multiple-times-daily vs once at close

- Live evaluates sells every scan during market hours on live prices; the backtest
  evaluates once per day on closes. A stock that dips through its stop intraday and
  recovers by close gets sold live but is invisible to the backtest.
- This is an inherent fidelity gap, not a bug. **Quantify it** instead: a backtest
  variant that checks stops against the daily **low** (pessimistic bound) vs close
  (current, optimistic bound) brackets the live behavior. Freeze-safe to build now.

---

## Minor / verified-equivalent

| # | Item | Verdict |
|---|------|---------|
| M1 | `take_profit_pct` fallback: live `profile.get(..., config.take_profit_pct)` vs backtest `profile.get(..., 75.0)` | No-op while champion profile defines it; latent divergence if a profile omits it |
| M2 | `sell_score_threshold`: live = user's `AIPortfolioConfig`, backtest = `Backtest` record | Config parity, not code parity — worth checking prod user configs match champion params |
| M3 | Live `purchase_score` is growth-aware (`get_effective_score(use_current=False)`); backtest uses raw `purchase_score` | Differs only for growth-mode positions |
| M4 | Backtest skips `cost_basis <= 0` positions; live relies on stored `gain_loss_pct` | Defensive-only |
| M5 | Partial-trailing peak reset: backtest at signal time (2678), live at execution (3209) | Equivalent — live executor always executes the list |
| M6 | `aging_loser_guard` / `profit_giveback_floor` exist only in backtester | Intentional, default-OFF research levers (c0ab651 verdicts) |
| M7 | `get_partial_profit_action`: live passes `yaml_config` explicitly, backtest uses default | Same global config object |
| M8 | Shared helpers (`select_effective_stop_loss_pct`, `get_trailing_stop_pct`, `apply_pyramid_widening`, `should_take_partial_on_trailing_stop`, `evaluate_score_crash`, `get_tightened_trailing_stop`, partial-profit tiers) | In sync — single source in `trading_engine.py` |
| M9 | Pre-earnings tighten block, score-missing (`score==0`) gate placement, PROTECT GAINS / TAKE PROFIT / WEAK POSITION structure, executor partial math + `reset_peak` | Verified line-by-line equivalent |

---

## Net read

The divergences do **not** all point the same way: D1 makes backtests optimistic,
while D2/D3/D4/D5 generally make live exit *earlier* than backtests model. So live
underperformance vs backtest is not explained by a single missing clamp — it's the
sum of a missing guard (D1), a 30×-faster score-crash clock (D2), random stop
tightening (D3), and a higher peak watermark (D4). This supports the June-9
conclusion: fix parity first, then re-measure, before designing any new exit
formulation.

## June-19 execution order (proposed)

1. **D1** guard port into `evaluate_sells` (+ regression test, delete sentinel monitor).
2. **D2** per-day dedup in `check_score_stability`'s query (+ test pinning scans-vs-days).
3. **D3** ATR caching / last-known fallback.
4. **Parity harness:** shared-fixture test feeding identical positions/prices/scores
   to both `evaluate_sells` and `_evaluate_sells`, asserting identical decisions —
   makes the next drift a test failure, not a live loss. May motivate extracting the
   sell loop into `trading_engine.py`.
5. **D4/D5** backtester-side fidelity work (peak-from-highs, stop-on-low bracket) —
   can start pre-June-18 since it doesn't touch `ai_trader.py`.
