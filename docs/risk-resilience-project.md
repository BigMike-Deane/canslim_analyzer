# Project: Participation Quality (AI Portfolio) — formerly "Risk & Resilience"

**Owner decision 2026-06-18:** next major initiative after the exit-parity work
+ A/B evaluations closed. Chosen over: edge-validation/observability, real
brokerage execution, multi-user productization.

## ⚠️ PIVOT 2026-06-18 — the kickoff diagnostic refuted the original thesis
The project began as a downside-protection ("survive the bear") effort. The
kickoff backtest audit **disproved that premise** and re-aimed the project.

**What the data showed (runs 846-865, trade-level diag of 846/849):**
- **Max drawdown is 5–13% in EVERY window** (W1 5.3%, W2 8.3%, W3 8.4–8.8%, W4
  7–11%) and live −5.7%. The book never blows up — **downside is already
  well-managed** by existing stops + gates + the −15/−25% circuit-breaker.
- The W1/W2 lag is **NOT** riding losers down — it's **under-participation**:
  - W2 2020-06→22-06: equity dead FLAT (~100%, peak 102%) for two years while
    SPY ran +38%. Idle cash earned nothing; small positions chopped out (10
    stop-loss + 7 trailing).
  - W1 2018-06→20-06: equity 100%→105% over 2yr vs SPY +16%. 24-day avg hold,
    **19 "weak position" shake-outs** — churned out of trends.
- **Phase 1 throttle (bear de-risk) FAILED** and confirmed the pivot: it never
  even fired in W1/W2 (no excess exposure to trim), and where it fired it cut
  −15.6pp in the W4 bull (whipsaw). Adding *more* de-risking worsens the real
  problem. Throttle kept as a default-off DD-reducer only; not shipped.

## Thesis (revised)
The book's downside is solved; its leak is **participation quality** — whipsaw
shake-outs (short holds, weak-position exits), re-entry latency after stops, and
**cash-drag** (idle cash earning nothing while the market trends). Goal:
**improve absolute + benchmark-relative participation in choppy / early-trend /
sideways regimes WITHOUT loosening the downside discipline that already works.**

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
   windows (W1/W2 absolute return / participation) WITHOUT materially hurting
   (>~3pp return) the STRONG windows (W3/W4) **or raising max drawdown beyond the
   current 5–13% envelope.** Sign-mixed/one-window = killed.
3. On ship, **mirror to `ai_trader.py`** (trader↔backtester sync rule).
4. Watch the failure modes: **whipsaw** (already killed the throttle in W4) and
   **closet-indexing** (a cash-sweep that just buys SPY isn't alpha — judge it on
   total return + Sharpe, not excess-vs-SPY, since holding SPY shrinks "excess").

## Standard sweep windows
W1 2018-06-01→2020-06-01 (correction), W2 2020-06-01→2022-06-01 (recovery bull →
2022 top), W3 2022-06-01→2024-06-01 (bear+recovery), W4 2024-01-01→2026-02-01
(recent). nostate_optimized, universe "all", $25k. Control = champion (no
overrides). ~2 min/run via `POST /api/backtests`.

## Phases (revised after the pivot)
- **Phase 0 — Bear-exposure throttle: DONE, KILLED.** (runs 846-865) No edge;
  whipsawed W4 −15.6pp, never fired in W1/W2. Lever kept default-off as a DD
  research tool. See road-log 2026-06-18.
- **Phase 1 — Cash-drag / SPY cash-sweep (IN PROGRESS, runs 866-873).** The
  champion has `spy_sweep` OFF, so idle cash earns nothing while the market
  trends (W2: flat vs SPY +38%). Test enabling the sweep (park idle cash in SPY
  when SPY>50MA, liquidate when below — risk-controlled). Judge on total
  return + Sharpe, NOT excess-vs-SPY (closet-indexing caveat).
- **Phase 2 — Whipsaw / shake-out reduction.** The W1 churn (24-day holds, 19
  "weak position" exits) suggests over-eager exits. Test loosening the
  weak-position / score-crash exit criteria and/or a minimum-hold grace period
  for fresh buys. (The shipped trailing-cadence fix was the first step here.)
- **Phase 3 — Re-entry latency.** After a shake-out in an ongoing uptrend, how
  fast does it re-buy? Test faster re-seeding / re-entry when SPY>50MA and cash
  is idle.
- **Phase 4 — Soft-buy / threshold participation.** Whether the min_score=72
  gate + soft-threshold band leaves valid trend entries on the table in choppy
  regimes.

Each phase: design → default-off lever → multi-window sweep → verdict (ship or
kill) → mirror if shipped. Record verdicts in `docs/road-log-2026-06.md`.
