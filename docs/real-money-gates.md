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
