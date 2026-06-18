# Project: Risk & Resilience Layer (AI Portfolio)

**Owner decision 2026-06-18:** next major initiative after the exit-parity work
+ A/B evaluations closed. Chosen over: edge-validation/observability, real
brokerage execution, multi-user productization.

## Thesis
The AI Portfolio was built, tuned, and validated almost entirely inside a bull
market. It has a genuine edge in trending regimes (live +28.95% vs SPY +12.4%,
alpha +13.84% over 52d) but **lags badly when the regime turns** — backtest-
proven: W1 2018-06→20-06 **−9.7pp vs SPY**, W2 2020-06→22-06 **−40pp vs SPY**.
Beta 1.2–1.5, concentrated (8 names, all pyramided). It has never met a real
drawdown with live money. Goal: **reduce full-cycle drawdown and improve risk-
adjusted return in adverse regimes WITHOUT gutting the trend-following edge.**

## Key finding from the kickoff audit (2026-06-18)
Most of a risk layer already exists but is **passive or dormant**:
- **Dynamic cash reserves** (`ai_trader.allocation.cash_reserve_bear` 40% /
  `_strong_bear` 60%) DO engage in the champion (driven by `weighted_signal`
  from market breadth, not the disabled market-state machine) — **but they only
  gate BUYING** (`backtester.py:1530`). They never TRIM existing positions to
  reach the reserve. Bear de-risking is a brake on buying, not active exposure
  reduction.
- **Drawdown circuit-breaker** (`ai_trader.drawdown_protection`): halt buys at
  −15%, liquidate all at −25%, resume <−10%. Active but coarse/late.
- **Market-state machine** (graduated exposure): DISABLED in champion — it HURT
  full-cycle returns (the "NoState binary SPY gate beats it 8.6x" finding). Do
  NOT simply re-enable it.
- **SPY cash-sweep**, **bear position-sizing** (`bearish_max_position_pct`,
  `bearish_stop_loss_pct` 7%), **portfolio heat**, **correction-zone**: present,
  mixed enabled state.

**Implication:** this project is mostly *re-calibrating / actively wiring*
existing machinery to the simple signals the champion already trusts — NOT
building a new complex regime engine (which history killed).

## Hard rules (same gate as all prior strategy work)
1. Every component is a **default-off backtester lever** first (the
   `profile_overrides` pattern, e.g. `aging_loser_guard`, `extension_guard`).
2. **Ship only on a robust multi-window win**: must materially improve the WEAK
   windows (W1/W2 excess-return and/or max drawdown) WITHOUT materially hurting
   (>~3pp return) the STRONG windows (W3/W4). Sign-mixed/one-window = killed.
3. On ship, **mirror to `ai_trader.py`** (trader↔backtester sync rule).
4. Beware the market-state-machine failure mode: **whipsaw** (de-risk into a dip,
   miss the recovery). The backtest must prove net benefit across full cycles.

## Standard sweep windows
W1 2018-06-01→2020-06-01 (correction), W2 2020-06-01→2022-06-01 (recovery bull →
2022 top), W3 2022-06-01→2024-06-01 (bear+recovery), W4 2024-01-01→2026-02-01
(recent). nostate_optimized, universe "all", $25k. Control = champion (no
overrides). ~2 min/run via `POST /api/backtests`.

## Phases
- **Phase 1 — Active bear-exposure throttle (IN PROGRESS).** When the regime is
  bear/strong-bear (`weighted_signal < 0`), actively TRIM stock exposure down to
  the existing `cash_reserve_bear`/`_strong_bear` target (sell weakest-scored
  first), instead of only gating buys. Default-off lever
  `bear_exposure_throttle`. Sweep trim-target variants vs champion. Hypothesis:
  helps W1/W2; risk: whipsaw drag on W2/W3 recovery legs.
- **Phase 2 — Circuit-breaker re-calibration.** The −15%/−25% breaker is late;
  test earlier/graduated thresholds + a cleaner re-entry rule.
- **Phase 3 — Concentration / correlation caps.** Tighter per-name/sector or
  correlation-aware limits to cut single-name and clustered risk.
- **Phase 4 — Conviction + volatility position sizing.** Size down high-vol
  names; integrate with the bear sizing already present.

Each phase: design → default-off lever → multi-window sweep → verdict (ship or
kill) → mirror if shipped. Record verdicts in `docs/road-log-2026-06.md`.
