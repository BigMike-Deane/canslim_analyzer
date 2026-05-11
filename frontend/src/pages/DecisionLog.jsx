import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatDateTime } from '../api'
import Card from '../components/Card'
import PageHeader from '../components/PageHeader'
import { ActionBadge, TagBadge, MLConfidenceBadge } from '../components/Badge'

// Derive a coarse kind from the free-text reason. Mirrors the prefixes
// ai_trader.py writes (STOP LOSS, TRAILING STOP, PARTIAL TRAILING STOP,
// CIRCUIT BREAKER, PYRAMID, SCORE CRASH); falls back to action.
function classifyTrade(t) {
  const r = (t.reason || '').toUpperCase()
  if (r.startsWith('STOP LOSS')) return 'stop'
  if (r.startsWith('TRAILING STOP') || r.startsWith('PARTIAL TRAILING')) return 'trail'
  if (r.startsWith('CIRCUIT BREAKER')) return 'circuit'
  if (r.startsWith('PYRAMID')) return 'pyramid'
  if (r.includes('SCORE CRASH') || r.includes('SCORE DROP')) return 'score_drop'
  if (t.action === 'BUY') return 'buy'
  if (t.action === 'SELL') return 'sell'
  return 'other'
}

const KIND_META = {
  buy:        { label: 'Buy',           color: 'emerald', match: 'buy' },
  sell:       { label: 'Sell',          color: 'red',     match: 'sell' },
  stop:       { label: 'Stop loss',     color: 'red',     match: 'stop' },
  trail:      { label: 'Trailing stop', color: 'amber',   match: 'trail' },
  pyramid:    { label: 'Pyramid',       color: 'teal',    match: 'pyramid' },
  score_drop: { label: 'Score crash',   color: 'amber',   match: 'score_drop' },
  circuit:    { label: 'Circuit',       color: 'red',     match: 'circuit' },
  other:      { label: 'Other',         color: 'default', match: 'other' },
}

function KindChip({ kind }) {
  const meta = KIND_META[kind] || KIND_META.other
  return <TagBadge color={meta.color}>{meta.label}</TagBadge>
}

// Group trades into [Today, Yesterday, YYYY-MM-DD, ...] sections.
function groupByDay(trades) {
  const groups = new Map()
  const today = new Date().toISOString().slice(0, 10)
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10)
  for (const t of trades) {
    if (!t.executed_at) continue
    const day = t.executed_at.slice(0, 10)
    if (!groups.has(day)) groups.set(day, [])
    groups.get(day).push(t)
  }
  // Map → array sorted desc by day
  const sorted = [...groups.entries()].sort((a, b) => b[0].localeCompare(a[0]))
  return sorted.map(([day, items]) => ({
    day,
    label: day === today ? 'Today' : day === yesterday ? 'Yesterday' : day,
    items,
  }))
}

function TradeRow({ trade }) {
  const kind = classifyTrade(trade)
  const gainPositive = (trade.realized_gain ?? 0) >= 0
  return (
    <div className="flex items-start justify-between py-2.5 px-2 -mx-2 rounded hover:bg-dark-800/50 transition-colors border-b border-dark-700/30 last:border-0">
      <div className="flex items-start gap-2 min-w-0 flex-1">
        <ActionBadge action={trade.action} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Link to={`/stock/${trade.ticker}`} className="font-semibold text-primary-400 hover:underline">
              {trade.ticker}
            </Link>
            <KindChip kind={kind} />
            {trade.is_growth_stock && <TagBadge color="purple">G</TagBadge>}
            {trade.signal_factors?.ml_confidence != null && (
              <MLConfidenceBadge confidence={trade.signal_factors.ml_confidence} size="xs" />
            )}
            <span className="text-[10px] text-dark-500 font-data">
              {new Date(trade.executed_at).toLocaleTimeString('en-US', {
                hour: 'numeric', minute: '2-digit', timeZone: 'America/Chicago',
              })}
            </span>
          </div>
          {trade.reason && (
            <div className="text-[11px] text-dark-400 mt-1 break-words">
              {trade.reason}
            </div>
          )}
        </div>
      </div>
      <div className="text-right shrink-0 ml-3 flex flex-col items-end">
        <div className="font-data text-xs text-dark-200">
          {trade.shares.toFixed(2)} @ {formatCurrency(trade.price)}
        </div>
        {trade.realized_gain != null && (
          <div className={`text-[11px] font-data ${gainPositive ? 'text-emerald-400' : 'text-red-400'}`}>
            {gainPositive ? '+' : ''}{formatCurrency(trade.realized_gain)}
          </div>
        )}
      </div>
    </div>
  )
}

export default function DecisionLog() {
  const [trades, setTrades] = useState(null)
  const [activeKinds, setActiveKinds] = useState(new Set())  // empty = all
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.getAIPortfolioTrades(200)
      .then(t => setTrades(Array.isArray(t) ? t : []))
      .catch(() => setTrades([]))
      .finally(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    if (!trades) return []
    if (activeKinds.size === 0) return trades
    return trades.filter(t => activeKinds.has(classifyTrade(t)))
  }, [trades, activeKinds])

  const grouped = useMemo(() => groupByDay(filtered), [filtered])

  const counts = useMemo(() => {
    const c = {}
    for (const t of trades || []) {
      const k = classifyTrade(t)
      c[k] = (c[k] || 0) + 1
    }
    return c
  }, [trades])

  function toggleKind(kind) {
    setActiveKinds(prev => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind); else next.add(kind)
      return next
    })
  }

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto">
      <PageHeader
        title="Decision Log"
        subtitle="Every trade the AI made — grouped by day, filterable by kind."
        backTo="/ai-portfolio"
        backLabel="AI Portfolio"
      />

      {/* Filter chips */}
      <Card variant="glass" className="mb-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wider text-dark-500 mr-1">Filter</span>
          {Object.entries(KIND_META).map(([kind, meta]) => {
            const count = counts[kind] || 0
            if (count === 0) return null
            const active = activeKinds.has(kind)
            return (
              <button
                key={kind}
                onClick={() => toggleKind(kind)}
                className={`text-[10px] font-data px-2 py-1 rounded border transition-colors ${
                  active
                    ? 'bg-primary-500/15 border-primary-500/40 text-primary-300'
                    : 'bg-dark-800 border-dark-700 text-dark-400 hover:border-dark-600'
                }`}
              >
                {meta.label} · {count}
              </button>
            )
          })}
          {activeKinds.size > 0 && (
            <button
              onClick={() => setActiveKinds(new Set())}
              className="text-[10px] text-dark-500 hover:text-dark-300 ml-1 underline"
            >
              Clear
            </button>
          )}
        </div>
      </Card>

      {loading && (
        <Card variant="glass" className="text-center py-10 text-dark-400">Loading…</Card>
      )}

      {!loading && grouped.length === 0 && (
        <Card variant="glass" className="text-center py-10 text-dark-400">
          {trades && trades.length > 0
            ? 'No trades match the active filter.'
            : 'No trades yet. The AI will populate this log as it executes.'}
        </Card>
      )}

      {!loading && grouped.map(({ day, label, items }) => (
        <Card variant="glass" key={day} className="mb-4" padding="">
          <div className="px-4 pt-3 pb-2 flex justify-between items-baseline border-b border-dark-700/30">
            <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">{label}</span>
            <span className="text-[10px] font-data text-dark-500">{items.length} event{items.length === 1 ? '' : 's'}</span>
          </div>
          <div className="px-4 py-1">
            {items.map(t => <TradeRow key={t.id} trade={t} />)}
          </div>
        </Card>
      ))}
    </div>
  )
}
