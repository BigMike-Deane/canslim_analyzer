import { getScoreClass } from '../api'

export function ScoreBadge({ score, size = 'sm', className = '' }) {
  const cls = getScoreClass(score)
  const sizes = {
    xs: 'text-[10px] px-1.5 py-0.5',
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1',
  }

  return (
    <span className={`font-data font-medium rounded-md ${cls} ${sizes[size] || sizes.sm} ${className}`}>
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
}

export function ActionBadge({ action, className = '' }) {
  const { cls } = actionCfg[action] || actionCfg.HOLD
  return (
    <span className={`text-[10px] font-semibold px-2 py-0.5 rounded border ${cls} ${className}`}>
      {action}
    </span>
  )
}

export function TagBadge({ children, color = 'default', className = '' }) {
  const colors = {
    default: 'bg-dark-700 text-dark-300 border-dark-600',
    cyan: 'bg-primary-500/10 text-primary-400 border-primary-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    green: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  }

  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded border ${colors[color] || colors.default} ${className}`}>
      {children}
    </span>
  )
}

export function PnlText({ value, className = '', prefix = '' }) {
  if (value == null) return <span className={`text-dark-500 ${className}`}>-</span>
  const isPositive = value >= 0
  const color = isPositive ? 'text-emerald-400' : 'text-red-400'
  const sign = isPositive ? '+' : ''
  return (
    <span className={`font-data ${color} ${className}`}>
      {prefix}{sign}{typeof value === 'number' && !prefix ? value.toFixed(1) : value}
    </span>
  )
}
