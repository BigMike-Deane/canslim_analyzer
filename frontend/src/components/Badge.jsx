import { getScoreClass } from '../api'
import ScorePopover from './ScorePopover'

// When `ticker` is supplied the badge becomes an interactive trigger that opens
// the CANSLIM component breakdown (see ScorePopover). Without it, it's the same
// plain, presentational badge it has always been — every existing call site is
// unaffected until it opts in by passing a ticker.
export function ScoreBadge({ score, size = 'sm', className = '', ticker = null, details = null }) {
  if (ticker) {
    return <ScorePopover score={score} ticker={ticker} size={size} className={className} details={details} />
  }
  const cls = getScoreClass(score)
  const sizes = {
    xs: 'text-[11px] px-1.5 py-0.5',
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1',
  }

  return (
    <span className={`font-data font-bold rounded-md ${cls} ${sizes[size] || sizes.sm} ${className}`}>
      {score != null ? Math.round(score) : '-'}
    </span>
  )
}

const outcomeCfg = {
  big_win: { label: 'BIG WIN', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30 font-semibold' },
  win: { label: 'WIN', cls: 'bg-green-500/15 text-green-400 border-green-500/20' },
  flat: { label: 'FLAT', cls: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20' },
  loss: { label: 'LOSS', cls: 'bg-red-500/15 text-red-400 border-red-500/20' },
  pending: { label: 'PENDING', cls: 'bg-dark-600/50 text-dark-400 border-dark-500/30' },
}

export function OutcomeBadge({ outcome, className = '' }) {
  const { label, cls } = outcomeCfg[outcome] || outcomeCfg.pending
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded border ${cls} ${className}`}>
      {label}
    </span>
  )
}

const statusCfg = {
  running: { cls: 'bg-primary-500/15 text-primary-400 border-primary-500/20' },
  completed: { cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' },
  failed: { cls: 'bg-red-500/15 text-red-400 border-red-500/20' },
  pending: { cls: 'bg-dark-600/50 text-dark-400 border-dark-500/30' },
  cancelled: { cls: 'bg-orange-500/15 text-orange-400 border-orange-500/20' },
  idle: { cls: 'bg-dark-600/50 text-dark-400 border-dark-500/30' },
  scanning: { cls: 'bg-primary-500/15 text-primary-400 border-primary-500/20' },
}

export function StatusBadge({ status, label, className = '' }) {
  const { cls } = statusCfg[status] || statusCfg.pending
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded border ${cls} ${className}`}>
      {label || status?.toUpperCase() || 'UNKNOWN'}
    </span>
  )
}

const actionCfg = {
  BUY: { cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' },
  SELL: { cls: 'bg-red-500/15 text-red-400 border-red-500/20' },
  HOLD: { cls: 'bg-blue-500/15 text-blue-400 border-blue-500/20' },
  TRIM: { cls: 'bg-orange-500/15 text-orange-400 border-orange-500/20' },
  ADD: { cls: 'bg-green-500/15 text-green-400 border-green-500/20' },
  WATCH: { cls: 'bg-purple-500/15 text-purple-400 border-purple-500/20' },
  SEED: { cls: 'bg-primary-500/15 text-primary-400 border-primary-500/20' },
  // Pyramid additions (live + backtest both write action="PYRAMID" as of 2026-05-05)
  PYRAMID: { cls: 'bg-blue-500/15 text-blue-400 border-blue-500/20' },
  VETOED: { cls: 'bg-amber-500/15 text-amber-400 border-amber-500/20' },
}

export function ActionBadge({ action, className = '' }) {
  const { cls } = actionCfg[action] || actionCfg.HOLD
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${cls} ${className}`}>
      {action}
    </span>
  )
}

// Forwards ...rest (title, aria-*, data-*) onto the span — several call
// sites pass `title` tooltips; without the spread they were silently dropped.
export function TagBadge({ children, color = 'default', className = '', ...rest }) {
  const colors = {
    default: 'bg-dark-700 text-dark-300 border-dark-600',
    cyan: 'bg-primary-500/10 text-primary-400 border-primary-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    teal: 'bg-teal-500/10 text-teal-400 border-teal-500/20',
    green: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  }

  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded border ${colors[color] || colors.default} ${className}`}
      {...rest}
    >
      {children}
    </span>
  )
}

// Canonical display labels for detector base_type enums. The detector emits
// snake_case ('flat'|'cup'|'cup_with_handle'|'double_bottom'); every badge
// renders through this map so no page shows a raw enum string.
const BASE_TYPE_LABELS = {
  flat: 'flat base',
  cup: 'cup',
  cup_with_handle: 'cup+handle',
  double_bottom: 'double bottom',
}

export function formatBaseType(baseType) {
  if (!baseType) return baseType
  return BASE_TYPE_LABELS[baseType] || String(baseType).replace(/_/g, ' ')
}

// Base-pattern tag with actionability status. A detected pattern means "this
// geometry exists in the ~6-month lookback", NOT "this is a live setup":
// price >5% above the pivot = the base already broke out (extended, mirrors
// the backend's extended-entry threshold); price >20% below = the base
// failed (broken). Both render muted so the Screener/cards don't imply a
// tradeable setup. When pivot or price is missing the plain tag renders —
// fail-safe, never hides the pattern.
export function BaseTag({ baseType, weeksInBase, pivotPrice, currentPrice }) {
  if (!baseType || baseType === 'none') return null
  const label = formatBaseType(baseType)
  const prefix = weeksInBase ? `${weeksInBase}w ` : ''
  let status = null
  if (pivotPrice > 0 && currentPrice > 0) {
    const rel = ((currentPrice - pivotPrice) / pivotPrice) * 100
    if (rel > 5) status = 'extended'
    else if (rel < -20) status = 'broken'
  }
  if (!status) {
    return <TagBadge color="cyan">{prefix}{label}</TagBadge>
  }
  const tip = status === 'extended'
    ? 'Price is >5% above this base’s pivot — the breakout already happened; not a fresh entry.'
    : 'Price is >20% below this base’s pivot — the base failed; pattern is historical, not a setup.'
  return (
    <TagBadge color="default" className="opacity-75" title={tip}>
      {prefix}{label} · {status}
    </TagBadge>
  )
}

export function MLConfidenceBadge({ confidence, size = 'xs', className = '' }) {
  if (confidence == null) return null
  const pct = (confidence * 100).toFixed(0)
  const cls = confidence >= 0.65
    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
    : confidence >= 0.50
      ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
      : 'bg-red-500/20 text-red-300 border-red-500/30'
  const sizes = {
    xs: 'text-[10px] px-1.5 py-0.5',
    sm: 'text-xs px-2 py-0.5',
  }
  return (
    <span className={`font-data font-medium rounded border ${cls} ${sizes[size] || sizes.xs} ${className}`} title={`ML Confidence: ${pct}%`}>
      ML {pct}%
    </span>
  )
}

const csConfCfg = {
  high: { label: 'HIGH', cls: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' },
  vhigh: { label: 'V.HIGH', cls: 'bg-emerald-500/25 text-emerald-200 border-emerald-400/40 font-semibold' },
  mod: { label: 'MOD', cls: 'bg-amber-500/20 text-amber-300 border-amber-500/30' },
  low: { label: 'LOW', cls: 'bg-red-500/20 text-red-300 border-red-500/30' },
}

export function CSConfidenceBadge({ confidence, className = '' }) {
  if (confidence == null) return <span className="text-dark-500 text-[10px]">-</span>
  const cfg = confidence >= 80 ? csConfCfg.vhigh
    : confidence >= 60 ? csConfCfg.high
    : confidence >= 30 ? csConfCfg.mod
    : csConfCfg.low
  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded border font-data ${cfg.cls} ${className}`}
      title={`CS Confidence: ${confidence}/100`}
    >
      {cfg.label}
    </span>
  )
}

export function PnlText({ value, className = '', prefix = null, decimals = 1 }) {
  if (value == null) return <span className={`text-dark-500 ${className}`}>-</span>
  const isPositive = value >= 0
  const color = isPositive ? 'text-emerald-400' : 'text-red-400'
  // When the caller passes a prefix (incl. ''), they're taking control of the
  // sign. Otherwise auto-prepend '+' for positives — negatives get their sign
  // from toFixed itself. Default to 1 decimal so we don't leak the raw float
  // (e.g. 0.10269232486492808% in the AI portfolio header); callers that want
  // tighter precision (e.g. the Command Center Portfolio card) pass decimals.
  const sign = prefix !== null ? prefix : (isPositive ? '+' : '')
  const display = typeof value === 'number' ? value.toFixed(decimals) : value
  return (
    <span className={`font-data ${color} ${className}`}>
      {sign}{display}
    </span>
  )
}
