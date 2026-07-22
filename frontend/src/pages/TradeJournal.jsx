import { useState, useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, formatCurrency, formatDate, formatPercent } from '../api'
import Card, { CardHeader } from '../components/Card'
import { ScoreBadge, TagBadge, PnlText } from '../components/Badge'
import PageHeader from '../components/PageHeader'

// ── Action badge config ──────────────────────────────────────────────
const actionColors = {
  BUY:     { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/20', dot: 'bg-emerald-400' },
  SELL:    { bg: 'bg-red-500/15',     text: 'text-red-400',     border: 'border-red-500/20',     dot: 'bg-red-400' },
  PYRAMID: { bg: 'bg-blue-500/15',   text: 'text-blue-400',    border: 'border-blue-500/20',    dot: 'bg-blue-400' },
  VETOED:  { bg: 'bg-amber-500/10',   text: 'text-amber-400',   border: 'border-amber-500/20',   dot: 'bg-amber-400' },
}

function ActionTag({ action }) {
  const cfg = actionColors[action] || actionColors.BUY
  return (
    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
      {action}
    </span>
  )
}

// ── Timeline dot ─────────────────────────────────────────────────────
function TimelineDot({ action }) {
  const cfg = actionColors[action] || actionColors.BUY
  return (
    <div className="flex flex-col items-center shrink-0">
      <div className={`w-2.5 h-2.5 rounded-full ${cfg.dot} ring-2 ring-dark-900`} />
      <div className="w-px flex-1 bg-dark-700/50 mt-1" />
    </div>
  )
}

// ── Expandable signal details ────────────────────────────────────────
function SignalDetail({ label, value, color }) {
  if (value == null || value === '' || value === false) return null
  return (
    <div className="flex justify-between items-center py-1 border-b border-dark-700/20 last:border-0">
      <span className="text-[10px] text-dark-400">{label}</span>
      <span className={`text-[10px] font-data ${color || 'text-dark-200'}`}>
        {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(2)) : String(value)}
      </span>
    </div>
  )
}

// ── Single journal entry card ────────────────────────────────────────
function JournalEntry({ entry }) {
  const [expanded, setExpanded] = useState(false)

  const isSell = entry.action === 'SELL'
  const isVetoed = entry.action === 'VETOED'
  const totalValue = entry.price && entry.shares ? entry.price * entry.shares : null

  const dotCfg = actionColors[entry.action] || actionColors.BUY

  return (
    <div className="flex gap-3">
      {/* Timeline connector (desktop) — hidden on mobile to save horizontal space */}
      <div className="hidden sm:flex">
        <TimelineDot action={entry.action} />
      </div>

      {/* Card */}
      <Card
        variant="glass"
        className="flex-1 mb-3 cursor-pointer transition-all hover:border-white/[0.1]"
        onClick={() => setExpanded(e => !e)}
      >
        {/* Mobile timeline marker — vertical stack fallback so mobile users
            still get a chronological action cue (desktop dot is hidden <640px) */}
        <div className="flex sm:hidden items-center gap-2 mb-2 pb-2 border-b border-dark-700/30">
          <span className={`w-2 h-2 rounded-full ${dotCfg.dot} ring-2 ring-dark-900 shrink-0`} />
          <span className="text-[10px] font-semibold uppercase tracking-wider text-dark-400">
            {entry.action}
          </span>
          <span className="text-[10px] text-dark-500 font-data ml-auto">
            {formatDate(entry.date)}
          </span>
        </div>

        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap min-w-0">
            <ActionTag action={entry.action} />
            <Link
              to={`/stock/${entry.ticker}`}
              className="text-sm font-semibold text-primary-400 hover:text-primary-300"
              onClick={e => e.stopPropagation()}
            >
              {entry.ticker}
            </Link>
            {entry.canslim_score != null && (
              <ScoreBadge score={entry.canslim_score} ticker={entry.ticker} size="xs" />
            )}
          </div>
          <span className="text-[10px] text-dark-400 font-data shrink-0">
            {formatDate(entry.date)}
          </span>
        </div>

        {/* Core metrics row */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
          {entry.price != null && (
            <span className="text-xs text-dark-300">
              <span className="text-dark-500">Price</span>{' '}
              <span className="font-data text-dark-100">{formatCurrency(entry.price)}</span>
            </span>
          )}
          {entry.shares != null && (
            <span className="text-xs text-dark-300">
              <span className="text-dark-500">Shares</span>{' '}
              <span className="font-data text-dark-100">{entry.shares}</span>
            </span>
          )}
          {totalValue != null && (
            <span className="text-xs text-dark-300">
              <span className="text-dark-500">Value</span>{' '}
              <span className="font-data text-dark-100">{formatCurrency(totalValue)}</span>
            </span>
          )}
          {entry.entry_type && (
            <span className="text-xs">
              <TagBadge color="cyan">{entry.entry_type}</TagBadge>
            </span>
          )}
        </div>

        {/* Vetoed-specific summary: ML confidence is the headline */}
        {isVetoed && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            <span className="text-xs text-dark-300">
              <span className="text-dark-500">ML conf</span>{' '}
              <span className={`font-data ${entry.ml_confidence < 0.30 ? 'text-amber-400' : 'text-dark-100'}`}>
                {entry.ml_confidence != null ? entry.ml_confidence.toFixed(3) : '-'}
              </span>
            </span>
            {entry.composite_score != null && (
              <span className="text-xs text-dark-300">
                <span className="text-dark-500">Composite</span>{' '}
                <span className="font-data text-dark-100">{entry.composite_score.toFixed(1)}</span>
              </span>
            )}
            {entry.entry_type != null && (
              <span className="text-xs">
                <TagBadge color="amber">no buy</TagBadge>
              </span>
            )}
          </div>
        )}

        {/* ML confidence chip on regular trades. Bonus moved to the expanded
            detail view (it's ~noise per the 2026-06-12 ML investigation). */}
        {!isVetoed && entry.ml_confidence != null && (
          <div className="flex items-center gap-1 mt-1">
            <span className="text-[10px] text-dark-500">ML</span>
            <span className={`text-[10px] font-data ${entry.ml_confidence >= 0.30 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {entry.ml_confidence.toFixed(3)}
            </span>
          </div>
        )}

        {/* Sell-specific summary */}
        {isSell && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2">
            {entry.reason && (
              <span className="text-xs">
                <TagBadge color="red">{entry.reason}</TagBadge>
              </span>
            )}
            {entry.gain_loss_pct != null && (
              <span className="text-xs text-dark-300">
                <span className="text-dark-500">P&L</span>{' '}
                <PnlText value={entry.gain_loss_pct} className="text-xs" />
                <span className="text-dark-500">%</span>
              </span>
            )}
            {entry.holding_days != null && (
              <span className="text-xs text-dark-300">
                <span className="text-dark-500">Held</span>{' '}
                <span className="font-data text-dark-100">{entry.holding_days}d</span>
              </span>
            )}
            {entry.cost_basis != null && (
              <span className="text-xs text-dark-300">
                <span className="text-dark-500">Basis</span>{' '}
                <span className="font-data text-dark-100">{formatCurrency(entry.cost_basis)}</span>
              </span>
            )}
          </div>
        )}

        {/* Expand / collapse indicator */}
        <div className="flex justify-end mt-2">
          <svg
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round"
            className={`text-dark-500 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>

        {/* Expanded signal factors */}
        {expanded && (
          <div className="mt-3 pt-3 border-t border-dark-700/40">
            <p className="text-[10px] font-semibold tracking-widest uppercase text-dark-400 mb-2">
              {isVetoed ? 'ML Features' : 'Signal Factors'}
            </p>
            {isVetoed ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
                <div>
                  <SignalDetail label="ML Confidence" value={entry.ml_confidence} color={entry.ml_confidence < 0.30 ? 'text-amber-400' : 'text-emerald-400'} />
                  <SignalDetail label="Total Score" value={entry.features?.total_score} />
                  <SignalDetail label="Composite Score" value={entry.features?.composite_score} />
                  <SignalDetail label="C / A / N" value={`${entry.features?.c_score ?? '-'} / ${entry.features?.a_score ?? '-'} / ${entry.features?.n_score ?? '-'}`} />
                  <SignalDetail label="S / L / I" value={`${entry.features?.s_score ?? '-'} / ${entry.features?.l_score ?? '-'} / ${entry.features?.i_score ?? '-'}`} />
                </div>
                <div>
                  <SignalDetail label="% from 21MA" value={entry.features?.pct_from_21ma != null ? `${entry.features.pct_from_21ma.toFixed(1)}%` : null} />
                  <SignalDetail label="% from 50MA" value={entry.features?.pct_from_50ma != null ? `${entry.features.pct_from_50ma.toFixed(1)}%` : null} />
                  <SignalDetail label="Relative Volume" value={entry.features?.relative_volume} />
                  <SignalDetail label="Sector RS Rank" value={entry.features?.sector_rs_rank} />
                  <SignalDetail label="Industry Rank" value={entry.features?.industry_group_rank} />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
                <div>
                  <SignalDetail label="Entry Type" value={entry.entry_type} />
                  <SignalDetail label="Market Regime" value={entry.market_regime} />
                  <SignalDetail label="Composite Score" value={entry.composite_score} />
                  <SignalDetail label="CANSLIM Score" value={entry.canslim_score} />
                  <SignalDetail label="ML Confidence" value={entry.ml_confidence} color={entry.ml_confidence != null && entry.ml_confidence < 0.30 ? 'text-amber-400' : 'text-emerald-400'} />
                  <SignalDetail label="Position %" value={entry.position_pct != null ? `${entry.position_pct.toFixed(1)}%` : null} />
                </div>
                <div>
                  <SignalDetail label="Base Quality Bonus" value={entry.base_quality_bonus} color={entry.base_quality_bonus > 0 ? 'text-emerald-400' : undefined} />
                  <SignalDetail label="Breakout Bonus" value={entry.breakout_bonus} color={entry.breakout_bonus > 0 ? 'text-emerald-400' : undefined} />
                  <SignalDetail label="Pre-Breakout Bonus" value={entry.pre_breakout_bonus} color={entry.pre_breakout_bonus > 0 ? 'text-emerald-400' : undefined} />
                  <SignalDetail label="Momentum" value={entry.momentum} />
                  <SignalDetail label="Coiled Spring Bonus" value={entry.coiled_spring_bonus} color={entry.coiled_spring_bonus > 0 ? 'text-purple-400' : undefined} />
                  <SignalDetail label="ML Bonus" value={entry.ml_bonus} color={entry.ml_bonus > 0 ? 'text-emerald-400' : entry.ml_bonus < 0 ? 'text-red-400' : undefined} />
                </div>
              </div>
            )}

            {/* Sell details in expanded view */}
            {isSell && (
              <>
                <p className="text-[10px] font-semibold tracking-widest uppercase text-dark-400 mb-2 mt-3">
                  Exit Details
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6">
                  <div>
                    <SignalDetail label="Sell Reason" value={entry.reason} color="text-red-400" />
                    <SignalDetail label="Gain/Loss %" value={entry.gain_loss_pct != null ? `${entry.gain_loss_pct >= 0 ? '+' : ''}${entry.gain_loss_pct.toFixed(2)}%` : null} color={entry.gain_loss_pct >= 0 ? 'text-emerald-400' : 'text-red-400'} />
                  </div>
                  <div>
                    <SignalDetail label="Holding Days" value={entry.holding_days} />
                    <SignalDetail label="Cost Basis" value={entry.cost_basis != null ? formatCurrency(entry.cost_basis) : null} />
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}

// ── Filter bar ───────────────────────────────────────────────────────
const DAYS_OPTIONS = [
  { value: 30, label: '30 days' },
  { value: 60, label: '60 days' },
  { value: 90, label: '90 days' },
  { value: 180, label: '180 days' },
  { value: 365, label: '1 year' },
]

const ACTION_OPTIONS = [
  { value: '', label: 'All Actions' },
  { value: 'BUY', label: 'Buys' },
  { value: 'SELL', label: 'Sells' },
  { value: 'PYRAMID', label: 'Pyramids' },
  { value: 'VETOED', label: 'Vetoed' },
]

function FilterBar({ days, setDays, ticker, setTicker, action, setAction, includeVetoed, setIncludeVetoed }) {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {/* Days dropdown */}
      <select
        value={days}
        onChange={e => setDays(Number(e.target.value))}
        className="bg-dark-800 border border-dark-700 rounded-lg px-3 py-1.5 text-xs text-dark-100 font-data focus:outline-none focus:border-primary-500/50"
      >
        {DAYS_OPTIONS.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>

      {/* Ticker search */}
      <input
        type="text"
        value={ticker}
        onChange={e => setTicker(e.target.value.toUpperCase())}
        placeholder="Filter ticker..."
        className="bg-dark-800 border border-dark-700 rounded-lg px-3 py-1.5 text-xs text-dark-100 font-data placeholder-dark-500 focus:outline-none focus:border-primary-500/50 w-32"
      />

      {/* Action filter */}
      <div className="flex rounded-lg border border-dark-700 overflow-hidden">
        {ACTION_OPTIONS.map(o => (
          <button
            key={o.value}
            onClick={() => setAction(o.value)}
            className={`px-3 py-1.5 text-[10px] font-semibold transition-colors ${
              action === o.value
                ? 'bg-primary-600 text-white'
                : 'bg-dark-800 text-dark-400 hover:text-dark-200'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>

      {/* Show vetoed toggle */}
      <button
        onClick={() => setIncludeVetoed(v => !v)}
        title="Surface ML-evaluated candidates that the gate vetoed (no buy)"
        className={`px-3 py-1.5 text-[10px] font-semibold rounded-lg border transition-colors ${
          includeVetoed
            ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
            : 'bg-dark-800 text-dark-400 border-dark-700 hover:text-dark-200'
        }`}
      >
        {includeVetoed ? '✓ Vetoes' : '+ Vetoes'}
      </button>
    </div>
  )
}

// ── ML Gate Scoreboard ───────────────────────────────────────────────
// Surfaces MLPrediction.actual_outcome + actual_gain_pct: did the model's
// veto turn out to be right? "Correct" = vetoed AND would have lost.
// "Wrong" = vetoed AND would have won (model cost us upside). Backfilled
// by backend/ml_backfill.py once each veto's holding window closes; we
// render Pending until then.
function MLGateScoreboard({ entries }) {
  const vetoes = entries.filter(e => e.action === 'VETOED')
  if (vetoes.length === 0) return null

  const withOutcome = vetoes.filter(v => v.actual_outcome != null)
  const pending = vetoes.length - withOutcome.length
  const correct = withOutcome.filter(v => v.actual_outcome === 0).length
  const wrong = withOutcome.length - correct
  const correctRate = withOutcome.length > 0
    ? (correct / withOutcome.length) * 100
    : null

  // Avg gain we avoided on correct vetoes — negative (we sidestepped losses).
  const correctWithGain = withOutcome.filter(
    v => v.actual_outcome === 0 && v.actual_gain_pct != null
  )
  const avgAvoidedLoss = correctWithGain.length > 0
    ? correctWithGain.reduce((s, v) => s + v.actual_gain_pct, 0) / correctWithGain.length
    : null

  // Avg gain we missed on wrong vetoes — positive (model cost us upside).
  const wrongWithGain = withOutcome.filter(
    v => v.actual_outcome === 1 && v.actual_gain_pct != null
  )
  const avgMissedGain = wrongWithGain.length > 0
    ? wrongWithGain.reduce((s, v) => s + v.actual_gain_pct, 0) / wrongWithGain.length
    : null

  const rateColor = correctRate == null ? 'text-dark-400'
    : correctRate >= 60 ? 'text-emerald-400'
    : correctRate >= 50 ? 'text-amber-400'
    : 'text-red-400'

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader
        title="ML Gate Scoreboard"
        subtitle="Is the ML veto paying for itself?"
      />
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-2">
        <div>
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">Total Vetoes</p>
          <p className="text-lg font-bold font-data text-amber-400">{vetoes.length}</p>
        </div>
        <div title="Vetoes whose 30-day outcome window has closed and been backfilled">
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">Resolved</p>
          <p className="text-lg font-bold font-data text-dark-100">
            {withOutcome.length}
            {pending > 0 && (
              <span className="text-[10px] text-dark-500 font-normal ml-1">
                · {pending} pending
              </span>
            )}
          </p>
        </div>
        <div title="Correctness rate: vetoes where the predicted-loss actually materialized">
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">Correct rate</p>
          <p className={`text-lg font-bold font-data ${rateColor}`}>
            {correctRate != null ? `${correctRate.toFixed(0)}%` : '–'}
          </p>
          {withOutcome.length > 0 && (
            <p className="text-[10px] text-dark-500">
              {correct}✓ / {wrong}✗
            </p>
          )}
        </div>
        <div title="Average gain on correct vetoes — negative means we sidestepped losses">
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">Avg avoided</p>
          <p className={`text-lg font-bold font-data ${avgAvoidedLoss != null && avgAvoidedLoss < 0 ? 'text-emerald-400' : 'text-dark-100'}`}>
            {avgAvoidedLoss != null
              ? `${avgAvoidedLoss >= 0 ? '+' : ''}${avgAvoidedLoss.toFixed(1)}%`
              : '–'}
          </p>
        </div>
        <div title="Average gain on wrong vetoes — positive means the model cost us upside">
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">Avg missed</p>
          <p className={`text-lg font-bold font-data ${avgMissedGain != null && avgMissedGain > 0 ? 'text-red-400' : 'text-dark-100'}`}>
            {avgMissedGain != null
              ? `+${avgMissedGain.toFixed(1)}%`
              : '–'}
          </p>
        </div>
      </div>
      {withOutcome.length === 0 && (
        <p className="text-[10px] text-dark-500 mt-2 italic">
          No outcomes resolved yet — vetoes are backfilled by ml_backfill.py
          once each 30-day window closes.
        </p>
      )}
    </Card>
  )
}

// ── Summary stats ────────────────────────────────────────────────────
function JournalStats({ entries }) {
  if (!entries || entries.length === 0) return null

  const buys = entries.filter(e => e.action === 'BUY').length
  const sells = entries.filter(e => e.action === 'SELL').length
  const pyramids = entries.filter(e => e.action === 'PYRAMID').length
  const vetoes = entries.filter(e => e.action === 'VETOED').length

  const sellEntries = entries.filter(e => e.action === 'SELL' && e.gain_loss_pct != null)
  const wins = sellEntries.filter(e => e.gain_loss_pct >= 0).length
  const winRate = sellEntries.length > 0 ? ((wins / sellEntries.length) * 100).toFixed(0) : '-'
  const avgGain = sellEntries.length > 0
    ? (sellEntries.reduce((sum, e) => sum + e.gain_loss_pct, 0) / sellEntries.length).toFixed(1)
    : '-'

  const stats = [
    { label: 'Total', value: entries.length, sub: 'decisions' },
    { label: 'Buys', value: buys, color: 'text-emerald-400' },
    { label: 'Sells', value: sells, color: 'text-red-400' },
    { label: 'Pyramids', value: pyramids, color: 'text-blue-400' },
    { label: 'Win Rate', value: winRate !== '-' ? `${winRate}%` : '-', color: Number(winRate) >= 50 ? 'text-emerald-400' : 'text-red-400' },
    { label: 'Avg P&L', value: avgGain !== '-' ? `${avgGain}%` : '-', color: Number(avgGain) >= 0 ? 'text-emerald-400' : 'text-red-400' },
  ]
  if (vetoes > 0) {
    // Replace Avg P&L with Vetoes when vetoes are visible — keeps the row at 6 tiles.
    stats.splice(5, 0, { label: 'Vetoes', value: vetoes, color: 'text-amber-400' })
    stats.length = 6
  }

  return (
    <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-4">
      {stats.map(s => (
        <Card key={s.label} variant="stat" padding="px-3 py-2">
          <p className="text-[10px] text-dark-500 uppercase tracking-wider">{s.label}</p>
          <p className={`text-lg font-bold font-data ${s.color || 'text-dark-100'}`}>{s.value}</p>
          {s.sub && <p className="text-[10px] text-dark-500">{s.sub}</p>}
        </Card>
      ))}
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────
export default function TradeJournal({ embedded = false }) {
  const [searchParams, setSearchParams] = useSearchParams()

  const [entries, setEntries] = useState([])
  const [total, setTotal] = useState(0)
  const [vetoedCount, setVetoedCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Filters — hydrate initial state from the URL on mount, falling back to the
  // component's defaults (days=90, ticker='', action='', includeVetoed=false)
  // when a param is absent. includeVetoed is persisted as "1"/"0".
  const [days, setDays] = useState(() => {
    const raw = parseInt(searchParams.get('days'), 10)
    return Number.isFinite(raw) ? raw : 90
  })
  const [ticker, setTicker] = useState(() => searchParams.get('ticker') || '')
  const [action, setAction] = useState(() => searchParams.get('action') || '')
  const [includeVetoed, setIncludeVetoed] = useState(() => searchParams.get('vetoes') === '1')

  // Persist filters to the URL (replace, so we don't pollute history). Mirrors
  // the canonical pattern in Backtest.jsx: copy current params, set/delete the
  // changed keys, then setSearchParams(next, { replace: true }). Non-default
  // values are written; defaults are removed to keep the URL clean.
  useEffect(() => {
    const next = new URLSearchParams(searchParams)

    if (days !== 90) next.set('days', String(days))
    else next.delete('days')

    if (ticker) next.set('ticker', ticker)
    else next.delete('ticker')

    if (action) next.set('action', action)
    else next.delete('action')

    if (includeVetoed) next.set('vetoes', '1')
    else next.delete('vetoes')

    setSearchParams(next, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [days, ticker, action, includeVetoed])

  // Debounced ticker search
  const [debouncedTicker, setDebouncedTicker] = useState('')
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedTicker(ticker), 400)
    return () => clearTimeout(timer)
  }, [ticker])

  // Filtering by VETOED implies pulling vetoed entries even if the toggle is off.
  const effectiveIncludeVetoed = includeVetoed || action === 'VETOED'

  // Fetch data
  useEffect(() => {
    setLoading(true)
    setError(null)

    api.getTradeJournal(days, debouncedTicker, action, effectiveIncludeVetoed)
      .then(data => {
        setEntries(data.entries || [])
        setTotal(data.total || 0)
        setVetoedCount(data.vetoed_count || 0)
        setError(null)
      })
      .catch(e => setError(e.message || 'Failed to load trade journal'))
      .finally(() => setLoading(false))
  }, [days, debouncedTicker, action, effectiveIncludeVetoed])

  return (
    <div className={embedded ? '' : 'max-w-4xl mx-auto px-4 py-6'}>
      {!embedded && (
        <PageHeader
          title="Trade Journal"
          subtitle={`${total} trade decision${total !== 1 ? 's' : ''} in the last ${days} days`}
          backTo="/"
          backLabel="Command Center"
        />
      )}

      {/* Filters */}
      <FilterBar
        days={days} setDays={setDays}
        ticker={ticker} setTicker={setTicker}
        action={action} setAction={setAction}
        includeVetoed={includeVetoed} setIncludeVetoed={setIncludeVetoed}
      />

      {/* Summary stats */}
      <JournalStats entries={entries} />

      {/* ML gate scoreboard — only renders when there are vetoes to score */}
      <MLGateScoreboard entries={entries} />

      {/* Error state */}
      {error && (
        <Card variant="glass" className="mb-4 text-center py-8">
          <p className="text-red-400 text-sm">{error}</p>
          <button
            onClick={() => { setError(null); setLoading(true) }}
            className="mt-2 text-xs text-primary-400 hover:text-primary-300"
          >
            Retry
          </button>
        </Card>
      )}

      {/* Loading state */}
      {loading && (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map(i => (
            <Card key={i} variant="glass" className="h-24 animate-pulse">
              <div className="h-3 bg-dark-700 rounded w-1/3 mb-3" />
              <div className="h-2 bg-dark-700 rounded w-2/3" />
            </Card>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && entries.length === 0 && (
        <Card variant="glass" className="text-center py-12">
          <p className="text-dark-400 text-sm">No trade decisions found</p>
          <p className="text-dark-500 text-xs mt-1">
            {debouncedTicker || action
              ? 'Try adjusting your filters'
              : 'Trade decisions will appear here once the AI trader runs'}
          </p>
        </Card>
      )}

      {/* Timeline of entries */}
      {!loading && !error && entries.length > 0 && (
        <div className="relative">
          {entries.map((entry, idx) => (
            <JournalEntry key={entry.id || `${entry.date}-${entry.ticker}-${idx}`} entry={entry} />
          ))}
        </div>
      )}
    </div>
  )
}
