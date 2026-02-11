# Ralph Loop: Automated Parameter Optimization

You are an optimization agent. Your job is to iteratively adjust the `lab` strategy profile in `config/default.yaml` to maximize backtest returns while maintaining acceptable risk metrics.

## Current Best: +41.1% (Backtest #133, balanced profile)

## Rules

1. **Only modify the `lab` profile** in `config/default.yaml`. NEVER touch `balanced` or `growth`.
2. **Change 2-3 parameters per iteration** maximum. Small, measurable changes.
3. **Run backtest** after each change: `bash scripts/run_lab_backtest.sh`
4. **Revert** if results worsen by more than 5pp from previous best.
5. **Stop** after beating +41.1% OR after 10 iterations without improvement.

## Parameter Priority (try in this order)

### Tier 1: Position Management (highest impact)
- `max_seed_investment_pct` (60): Try 50-70%
- `seed_count` (5): Try 4-6
- `max_positions` (8): Try 6-10
- `stop_loss_pct` (8.0): Try 7-10%

### Tier 2: Entry Timing
- `min_score` (72): Try 68-76
- `quality_filters.min_c_score` (10): Try 8-12
- `quality_filters.min_l_score` (8): Try 6-10

### Tier 3: Exit Timing
- `take_profit_pct` (75.0): Try 60-100%
- `trailing_stops.gain_50_plus` (20): Try 15-25%
- `trailing_stops.gain_30_to_50` (15): Try 12-18%
- `score_crash_drop_required` (25): Try 20-30

### Tier 4: Scoring Weights
- `scoring_weights.momentum` (0.25): Try 0.20-0.30
- `scoring_weights.growth_projection` (0.20): Try 0.15-0.25
- Weights must sum to 1.0

### Tier 5: Risk Controls
- `max_single_position_pct` (25): Try 20-30%
- `max_sector_pct` (50): Try 40-60%
- `bearish_stop_loss_pct` (7.0): Try 6-8%

## Iteration Template

```
## Iteration N

**Hypothesis**: [Why this change should improve returns]

**Changes**:
- param1: old_value → new_value
- param2: old_value → new_value

**Result**: Backtest #X
- Return: +X.X%
- vs SPY: +X.X%
- Sharpe: X.XX
- Max DD: X.X%
- Win Rate: X.X%

**Decision**: KEEP / REVERT
**Reasoning**: [Why this worked or didn't]
```

## Key Lessons from Previous Optimization

1. **Seed cap (60%) was crucial** — prevented over-commitment on day 1
2. **Position guard (8% stop for 21 days)** — cut unproven losers fast
3. **No conviction sizing on seeds** — equal-weight initial entries performed better
4. **Wider soft zone (8pt)** — more gradual entry instead of hard threshold
5. **Blocking features HURT returns** — no global cooldown, no FTD gate, no volume gate
6. **Market regime gate is smart** — skip buys when SPY < 50MA

## How to Read Results

- **Return > 41.1%**: Improvement found!
- **Sharpe > 1.95**: Risk-adjusted improvement
- **Max DD < 14.4%**: Better risk control
- **Win Rate > 55.6%**: Better stock selection

Ideally improve ALL metrics, but prioritize total return, then Sharpe.

## Workflow

1. Read current lab profile: `cat config/default.yaml | grep -A 40 "  lab:"`
2. Read previous results: `cat /tmp/lab_results.json`
3. Decide changes based on priority and previous results
4. Edit `config/default.yaml` (lab section only)
5. Run: `bash scripts/run_lab_backtest.sh`
6. Record results in this format
7. Decide KEEP or REVERT
8. Repeat from step 2
