// ── Position Health ─────────────────────────────────────────────────
// Roll-up of "is this open position OK or do I need to look at it?"
// Single chip per position so the user doesn't have to cross-reference
// stop distance + score drop + trailing-stop status + earnings on
// 4 different pages.
//
// Three tiers:
//   🔴 act    — stop is about to trigger, fundamentals broke, or
//               position already past hard stop. Demands attention now.
//   🟡 watch  — within striking distance of an exit rule; worth a
//               glance but no immediate action.
//   🟢 healthy — none of the flags fire.
//
// computePositionHealth(p, opts) returns { tier, reasons: [string],
// summary: string } — reasons populate the chip's hover title, summary
// is the most-severe single-line label for the row itself.
//
// Schema-tolerant: positions from /api/ai-portfolio (current_score,
// gain_loss_pct, trailing_stop.near_stop) and /api/command-center
// (score, gain_pct, stop_distance, trail_from_peak) both work. Missing
// fields skip their rules rather than crash.

// Rule thresholds. Override via opts when called from a page that has
// access to config (e.g. AIPortfolio knows the user's sell_score_threshold).
export const DEFAULT_RULES = {
  // Score gates
  sellScoreThreshold: 60,    // < this = act (broken setup, time to exit)
  watchScoreThreshold: 72,   // < this AND >= sell = watch (no longer a buy)
  // Stop / drawdown
  hardStopGainPct: -7,       // <= this = past the 7% O'Neil stop = act
  preStopGainPct: -5,        // (-7 < gain <= -5) = watch zone
  stopDistanceAct: 2,        // stop_distance <= 2% = act
  stopDistanceWatch: 5,      // (2 < stop_distance <= 5) = watch
}

function normalize(p) {
  if (!p) return null
  return {
    ticker: p.ticker,
    score:           p.score          ?? p.current_score          ?? null,
    gainPct:         p.gain_pct       ?? p.gain_loss_pct          ?? null,
    stopDistance:    p.stop_distance  ?? null,                     // command-center only
    trailFromPeak:   p.trail_from_peak ?? p.trailing_stop?.drop_from_peak_pct ?? null,
    nearTrailing:    Boolean(p.trailing_stop?.near_stop),           // ai-portfolio only
  }
}

export function computePositionHealth(position, opts = {}) {
  const cfg = { ...DEFAULT_RULES, ...opts }
  const p = normalize(position)
  if (!p) return { tier: 'healthy', reasons: [], summary: '' }

  const reasons = []
  let tier = 'healthy'

  // ── ACT rules (any one wins, no further evaluation) ────────────
  if (p.gainPct != null && p.gainPct <= cfg.hardStopGainPct) {
    tier = 'act'
    reasons.push(`Down ${p.gainPct.toFixed(1)}% — past hard stop, exit`)
  }
  if (p.score != null && p.score < cfg.sellScoreThreshold) {
    tier = 'act'
    reasons.push(`Score ${p.score.toFixed(0)} below sell threshold ${cfg.sellScoreThreshold} — broken setup`)
  }
  if (p.stopDistance != null && p.stopDistance > 0 && p.stopDistance <= cfg.stopDistanceAct) {
    tier = 'act'
    reasons.push(`Stop ${p.stopDistance.toFixed(1)}% away — about to trigger`)
  }

  if (tier === 'act') {
    return { tier, reasons, summary: reasons[0] }
  }

  // ── WATCH rules (additive — all reasons collected) ─────────────
  if (
    p.gainPct != null &&
    p.gainPct > cfg.hardStopGainPct &&
    p.gainPct <= cfg.preStopGainPct
  ) {
    tier = 'watch'
    reasons.push(`Down ${p.gainPct.toFixed(1)}% — approaching stop`)
  }
  if (
    p.stopDistance != null &&
    p.stopDistance > cfg.stopDistanceAct &&
    p.stopDistance <= cfg.stopDistanceWatch
  ) {
    tier = 'watch'
    reasons.push(`Stop ${p.stopDistance.toFixed(1)}% away`)
  }
  if (
    p.score != null &&
    p.score < cfg.watchScoreThreshold &&
    p.score >= cfg.sellScoreThreshold
  ) {
    tier = 'watch'
    reasons.push(`Score ${p.score.toFixed(0)} below buy threshold ${cfg.watchScoreThreshold}`)
  }
  if (p.nearTrailing) {
    tier = 'watch'
    const peakDrop = p.trailFromPeak
    reasons.push(
      peakDrop != null
        ? `${peakDrop.toFixed(0)}% off peak — trailing stop approaching`
        : 'Trailing stop approaching'
    )
  }

  return {
    tier,
    reasons: reasons.length ? reasons : ['No flags — position healthy'],
    summary: reasons[0] || 'Healthy',
  }
}

