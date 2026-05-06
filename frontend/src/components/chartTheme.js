// Canonical chart theming. Recharts components are rendered via React
// inline-style props, so they don't pick up Tailwind tokens automatically —
// keep these in sync with the warm-dark palette in tailwind.config.js.
//
// Hex sources:
//   #1a1815 → dark-800 (warm Bloomberg-style card surface)
//   #847a64 → dark-400 (muted axis labels)
export const tooltipStyle = {
  background: '#1a1815',
  border: '1px solid rgba(245,158,11,0.12)',
  borderRadius: '10px',
  fontFamily: 'JetBrains Mono',
  fontSize: 13,
  maxWidth: '280px',
}

export const tooltipLabelStyle = {
  color: '#847a64',
  fontSize: 11,
}

// Reusable axis tick / grid colors for any chart that wants them. Most charts
// still inline these literals today (Backtest.jsx, Analytics.jsx) — exporting
// them here as the source-of-truth so future cleanups have a target.
export const chartAxis = {
  tick: '#847a64',         // dark-400
  axisLine: '#2c2820',     // dark-700
  grid: '#2c2820',         // dark-700
  reference: '#3d3628',    // dark-600
}

// Brand + P&L semantic colors for chart series. Brand is canonical Bloomberg
// amber. SPY benchmark is intentionally cool teal so it doesn't compete with
// brand amber on equity-curve overlays.
export const chartColors = {
  brand: '#f59e0b',        // primary-500 — Bloomberg amber for portfolio series
  brandSoft: '#fbbf24',    // primary-400 — softer brand chart line
  accent: '#fde68a',       // pale gold — secondary highlight, distinct from brand
  pnlUp: '#10b981',        // pnl.up — gain
  pnlUpSoft: '#34d399',    // pnl.up-soft
  pnlDown: '#ef4444',      // pnl.down — loss
  pnlDownSoft: '#f87171',  // pnl.down-soft
  spy: '#5eead4',          // SPY benchmark — cool teal, hue-separated from amber
  muted: '#847a64',        // dark-400 — neutral series
}
