// Canonical chart theming. Recharts components are rendered via React
// inline-style props, so they don't pick up Tailwind tokens automatically —
// keep these in sync with the warm-dark palette in tailwind.config.js.
//
// Hex sources:
//   #1f1416 → dark-800 (warm-tinted card surface)
//   #8c7479 → dark-400 (muted axis labels)
export const tooltipStyle = {
  background: '#1f1416',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: '10px',
  fontFamily: 'JetBrains Mono',
  fontSize: 13,
  maxWidth: '280px',
}

export const tooltipLabelStyle = {
  color: '#8c7479',
  fontSize: 11,
}

// Reusable axis tick / grid colors for any chart that wants them. Most charts
// still inline these literals today (Backtest.jsx, Analytics.jsx) — exporting
// them here as the source-of-truth so future cleanups have a target.
export const chartAxis = {
  tick: '#8c7479',         // dark-400
  axisLine: '#2f2225',     // dark-700
  grid: '#2f2225',         // dark-700
  reference: '#473538',    // dark-600
}

// Brand + P&L semantic colors for chart series. Keep these aligned with the
// Tailwind `brand` and `pnl` tokens.
export const chartColors = {
  brand: '#dc2626',        // primary-400 — brand red for accent series
  brandSoft: '#ef4444',    // primary-300 — softer brand chart line
  accent: '#f59e0b',       // accent-500 — gold highlight series
  pnlUp: '#10b981',        // pnl.up — gain
  pnlUpSoft: '#34d399',    // pnl.up-soft
  pnlDown: '#ef4444',      // pnl.down — loss
  pnlDownSoft: '#f87171',  // pnl.down-soft
  spy: '#f59e0b',          // SPY benchmark series — gold
  muted: '#8c7479',        // dark-400 — neutral series
}
