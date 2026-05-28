// ── Position Sizing ────────────────────────────────────────────────
// Translates a BUY signal into a concrete trade ticket: shares, limit
// price, stop, dollar risk, and risk-% of portfolio. Pure function so
// it's easy to test and consume from any page (StockDetail, CommandCenter
// candidates, Screener, etc).
//
// Champion config (nostate_optimized): max_positions=8, stop_loss=7%,
// max_position_pct ~= 1/max_positions = 12.5% of total_value.
//
// Sizing logic:
//   1. limit_price
//      - Pre-breakout (current < pivot):  pivot + $0.10 buffer (ride the breakout)
//      - At/above pivot but <= 5% over:  current + 0.25% slippage cushion
//      - Extended (> 5% above pivot):    suppressed — entry is chasing
//      - No pivot detected:              current + 0.25% slippage cushion
//   2. stop_price = limit_price × (1 − stop_loss_pct/100), 2-dp rounded.
//   3. position_budget = min(cash, total_value / max_positions). Floor
//      protects against trying to spend cash we don't have when several
//      positions are already filled near their cap.
//   4. shares = floor(position_budget / limit_price). Whole shares only —
//      Fidelity supports fractionals but the owner uses limit orders on
//      whole-share quantities.
//   5. cost_basis = shares × limit_price.
//   6. risk_dollars = shares × (limit_price − stop_price).
//   7. risk_pct = risk_dollars / total_value × 100.
//
// Returns null when sizing isn't actionable (extended chase, missing
// inputs, no cash, score below threshold). Callers should hide the card
// rather than render zeros.

const DEFAULTS = {
  stop_loss_pct: 7,        // Champion strategy default
  max_positions: 8,        // Champion strategy default
  min_score: 72,           // Champion buy gate
  extended_pct: 5,         // > pivot+5% is "chasing"
  pivot_buffer: 0.10,      // ride the breakout by 10 cents
  slippage_pct: 0.25,      // limit slightly above market for fills
}

function roundTo(value, dp = 2) {
  const m = 10 ** dp
  return Math.round(value * m) / m
}

export function computePositionSizing({
  // Stock-side inputs
  ticker,
  currentPrice,
  pivotPrice,
  score,                   // 0-100
  // Portfolio-side inputs
  cash,
  totalValue,
  positionsCount = 0,
  maxPositions = DEFAULTS.max_positions,
  stopLossPct = DEFAULTS.stop_loss_pct,
  minScore = DEFAULTS.min_score,
  // Optional overrides
  options = {},
}) {
  const cfg = { ...DEFAULTS, ...options }

  // Hard guards — return null so the UI hides the card cleanly.
  if (currentPrice == null || currentPrice <= 0) return null
  if (totalValue == null || totalValue <= 0) return null
  if (cash == null || cash <= 0) {
    return { kind: 'no_cash', ticker, reason: 'No cash available — portfolio fully invested' }
  }
  if (score != null && score < minScore) {
    return { kind: 'below_threshold', ticker, score, minScore, reason: `Score ${score?.toFixed?.(0) ?? score} below buy gate of ${minScore}` }
  }
  if (positionsCount >= maxPositions) {
    return { kind: 'at_max', ticker, positionsCount, maxPositions, reason: `At max-positions limit (${positionsCount}/${maxPositions}) — would need to close one first` }
  }

  // 1. Limit price decision tree.
  let limitPrice
  let entryNote
  if (pivotPrice != null && pivotPrice > 0) {
    const pctAbovePivot = ((currentPrice - pivotPrice) / pivotPrice) * 100
    if (currentPrice < pivotPrice) {
      limitPrice = roundTo(pivotPrice + cfg.pivot_buffer)
      entryNote = `Pre-breakout entry — limit ${cfg.pivot_buffer.toFixed(2)} above pivot $${pivotPrice.toFixed(2)}`
    } else if (pctAbovePivot <= cfg.extended_pct) {
      limitPrice = roundTo(currentPrice * (1 + cfg.slippage_pct / 100))
      entryNote = `Active breakout — limit ${cfg.slippage_pct}% above current for fill`
    } else {
      return {
        kind: 'extended',
        ticker,
        pctAbovePivot,
        reason: `Extended ${pctAbovePivot.toFixed(1)}% above pivot $${pivotPrice.toFixed(2)} — wait for pullback`,
      }
    }
  } else {
    limitPrice = roundTo(currentPrice * (1 + cfg.slippage_pct / 100))
    entryNote = `No base detected — limit ${cfg.slippage_pct}% above current`
  }

  // 2. Stop.
  const stopPrice = roundTo(limitPrice * (1 - stopLossPct / 100))

  // 3-4. Budget + shares.
  const perSlotBudget = totalValue / maxPositions
  const positionBudget = Math.min(cash, perSlotBudget)
  const shares = Math.floor(positionBudget / limitPrice)
  if (shares <= 0) {
    return {
      kind: 'too_small',
      ticker,
      positionBudget,
      limitPrice,
      reason: `Position budget $${positionBudget.toFixed(0)} below one share at $${limitPrice.toFixed(2)}`,
    }
  }

  // 5-7. Costs and risk.
  const costBasis = roundTo(shares * limitPrice)
  const riskDollars = roundTo(shares * (limitPrice - stopPrice))
  const riskPct = roundTo((riskDollars / totalValue) * 100, 2)

  return {
    kind: 'actionable',
    ticker,
    shares,
    limitPrice,
    stopPrice,
    stopLossPct,
    costBasis,
    riskDollars,
    riskPct,
    perSlotBudget: roundTo(perSlotBudget),
    positionBudget: roundTo(positionBudget),
    pivotPrice: pivotPrice || null,
    entryNote,
  }
}

// Fidelity-compatible trade ticket string. Pasted into the order entry
// form lets the user re-create the trade without retyping numbers from
// the chart.
export function formatTradeTicket(sizing) {
  if (!sizing || sizing.kind !== 'actionable') return ''
  const { ticker, shares, limitPrice, stopPrice } = sizing
  return `BUY ${shares} ${ticker} LIMIT ${limitPrice.toFixed(2)} GTC; STOP ${stopPrice.toFixed(2)}`
}
