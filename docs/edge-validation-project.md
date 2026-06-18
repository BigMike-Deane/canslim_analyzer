# Project: Edge Validation & Observability (AI Portfolio)

**Owner decision 2026-06-18:** next big project after the participation project
concluded (strategy proven at its efficient frontier — stop tuning). Chosen
sequel: prove whether the measured edge is REAL and durable before betting bigger.

## Thesis
The live edge looks strong (alpha +13.8%, Sharpe 4.28, +29% vs SPY +12% over 52d)
but the alpha point estimate has swung +4% → +7% → +13.8% across three weeks on
the *same* window — classic small-sample noise (≈30 closed trades, 52 trading
days, in a favorable regime). The scorecard reports precise-looking point
estimates with **no uncertainty**, which is actively misleading. Goal: replace
false precision with **honest statistical confidence** — so every decision
(scale capital, go to real execution, add users) rests on a verdict, not a noisy
number.

## Phases
- **Phase 1 — Statistical significance on the live edge (IN PROGRESS).** Extend
  `backend/edge_metrics.py` (pure-stdlib, no trading-path risk) with:
  - **Alpha significance**: OLS of daily portfolio returns on SPY returns →
    intercept (alpha), its standard error, t-stat, df, two-sided p-value, and a
    95% CI. Answers "is alpha distinguishable from zero?"
  - **Win-rate CI**: Wilson score interval on the win proportion.
  - **edge_verdict**: `significant` / `promising_insufficient_sample` /
    `inconclusive` / `no_edge`, from the t-stat + sample size. Surfaced in the
    `/api/ai-portfolio/edge` response (frontend display = follow-up).
- **Phase 2 — Live-vs-backtest reconciliation.** Does live trading track what the
  backtester predicts for the same period? (D1 was a divergence; exit-parity
  fixed several.) A periodic reconciliation report.
- **Phase 3 — Per-rule attribution / decision observability.** Decompose realized
  P&L by exit reason / entry signal so the edge is monitorable, and every future
  change is cleanly measurable.
- **Phase 4 — Power / "how many trades until significant".** Given the observed
  effect size + variance, project the sample size needed to confirm the edge —
  sets expectations for when the live verdict will firm up.

## Conventions
- All stats pure-stdlib in `edge_metrics.py` (unit-testable, zero trading-path
  risk). ai_trader.py / canslim_scorer.py never touched.
- Be honest about small samples: with ~30 trades the expected Phase 1 result is
  "directionally positive, NOT yet statistically significant — need more data."
  That is a valid, valuable answer, not a failure.
