# Real-Money Deployment Gates

**Status: APPROVED by owner 2026-07-22 — these thresholds are binding.**
Written 2026-07-22 at the owner's direction ("not ready for real money
until we prove we beat SPY").
These criteria are **pre-registered**: they are written *before* the
evidence arrives, so the decision can never be retro-fitted to a hopeful
reading of the data. Editing a gate after its evidence starts accumulating
requires an explicit owner sign-off recorded in this file.

## Why staged, not binary

The unconditional "beats SPY at 95% confidence" test needs ~9 years at the
current blended effect size, because the strategy's edge is regime-dependent
(live attribution, 73 trading days: trend days +33 bps/day over SPY, chop
days −30). Waiting for that single number wastes the edge if it's real;
ignoring statistics risks trading noise. The middle path: **small,
risk-capped real allocations that scale only as pre-defined evidence
milestones pass.** Maximum regret is bounded at every stage.

## Evidence inputs (all already instrumented)

| Signal | Source | Cadence |
|---|---|---|
| Trend-day conditional edge (t-stat, p) | Edge card `regime_edge.trend` | live, daily |
| Exit behavior matches design | exit-reconciliation poller (fires at ≥10 post-fix exits) | ~days away |
| Chop-damper shadow verdict | ABEval, shadow source, `shadow_chop_damper` vs champion | weeks–months |
| Win-rate Wilson CI lower bound | Edge card | per closed trade |
| Unconditional alpha significance | Edge card verdict clock | slow burn |

## Gates

### Gate 0 — today (no real money)
Paper only. Push notifications, radar, screener for **manual** trading
ideas the owner independently evaluates. *Current state.*

### Gate 1 — starter allocation (≤5% of intended capital)
ALL of the following:
- [ ] Exit-reconciliation poller verdict: post-fix exits healthy
      (hold/WR/return shape consistent with the modeled exits).
- [ ] Trend-day edge: t ≥ 1.5 with ≥ 70 trend days observed (direction
      stable as sample grows).
- [ ] No open system-integrity alarms; scans healthy 2+ consecutive weeks.

Rules at Gate 1: mirror BUY signals only on trend-regime days (SPY > 1.5%
above 50MA); always honor the model's exits; hard monthly loss cap = 2% of
total intended capital → drop back to Gate 0 for 1 month if breached.

### Gate 2 — half allocation (≤25%)
ALL of the following, in addition to Gate 1 held for ≥ 6 weeks:
- [ ] Trend-day edge statistically significant (p < 0.05).
- [ ] Chop handling resolved: EITHER chop-damper shadow beats champion on
      mixed-regime weeks (then adopt it) OR chop bleed shrinks to
      > −10 bps/day live without it.
- [ ] Win-rate Wilson CI lower bound ≥ 45% with ≥ 40 closed trades.

### Gate 3 — full allocation
- [ ] Unconditional alpha significant at 95% (the original verdict clock)
      OR 12 consecutive months of positive excess return including ≥ 2
      distinct chop stretches ≥ 3 weeks each.
- [ ] Max live drawdown to date no worse than 1.5× the backtest envelope
      (13% × 1.5 ≈ 20%).

## Standing constraints (do not relax without rewriting this file)
- The champion strategy's config is not tuned in response to real-money
  P&L — evaluation changes go through shadow A/B first, always.
- A gate, once passed, can be UN-passed: if its evidence deteriorates below
  threshold for 4+ weeks, step down one gate.
- This file is the source of truth; the work loop checks these boxes as
  evidence arrives and surfaces gate transitions to the owner — the OWNER
  makes every allocation change manually. Nothing here trades real money
  automatically.

## Why criterion 1 (blended edge) is not a waiting game — measured 2026-09-12

`blended_daily_excess_bps` is **not** a single-window number, and reading it as
one wasted most of a session. It is

    share(trailing 60 days) x trend_mean(full history)
  + (1 - share)            x chop_mean(full history)

while the `chop_share_pct` printed beside it is the **full history's** day mix.
`regime_conditional_edge` takes no window argument at all; `regime_mix_summary`
takes `window_days=60`. So the weight moves with the recent tape while the means
are multi-month averages that barely budge — which is why the figure fell from
−1.7 to −4.2 bps/day in three days and was briefly read as the strategy
decaying. It was the market's regime mix. Both numbers, and the parts they are
built from, are now returned by the criterion (`trend_mean_bps`,
`chop_mean_bps`, `breakeven_trend_share_pct`, `recent_trend_share_pct`,
`mix_window_days`, `chop_share_basis`) so nobody reconstructs them by hand
again. Guarded by `TestCriterionOneReportsItsParts`.

**Measured on user 1, 108 live days (from 2026-04-08, leading flat days dropped):**

| quantity | value | days |
|---|---|---|
| trend-day mean excess | **+22.2 bps/day** | 74 |
| chop-day mean excess | **−20.0 bps/day** | 34 |
| breakeven trend share | **47.4%** | — |
| market's trailing-60 trend share | **48.3%** | 60 |

**The blend therefore sits on zero**, and its sign is decided by a day or two
entering or leaving the 60-day window. Two reasonable reconstructions landed 4.6
bps apart on it — which is the point, not a discrepancy.

⚑ **The consequence: criterion 1 cannot be passed by accumulating more days.**
The strategy's breakeven regime mix is almost exactly the market's own long-run
mix, so the blend cancels however long you wait. Only a shallower chop bleed
moves it. Each 1 bps off `chop_mean` buys ~1.1pp of breakeven share:

| chop_mean | breakeven trend share | vs market's ~48% |
|---|---|---|
| −20.0 (today) | 47.4% | coin-flip |
| −15.0 | 40.3% | passes on a normal mix |
| −10.0 | 31.1% | robustly positive |

This is the measured case for the chop arms (`shadow_chop_spy`,
`shadow_chop_damper`, `shadow_chop_entry_bar`, `shadow_chop_trim`) being the
program that matters, rather than an intuition about chop. Their constraint is
accrual: 7 of 15 chop days as of 2026-09-12.

⚑ **Trap when recomputing this by hand:** the pre-first-trade cash days classify
as *chop* and carry a spurious **positive** excess — the portfolio sits flat
while SPY moves, and SPY fell over that stretch in March 2026. Including them
lifted the chop mean from −20.0 to −11.4 and made the blend look positive. The
gate drops them via `leading_flat_start_index`; any hand reconstruction must too.
