import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  Tooltip, Legend, ReferenceLine,
} from 'recharts'
import { api, cache } from '../api'
import { useAuth } from '../auth'
import useApi from '../hooks/useApi'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { tooltipStyle, tooltipLabelStyle, chartAxis, chartColors } from '../components/chartTheme'
import { useToast } from '../components/Toast'

const REFRESH_OPTIONS = [
  { label: 'Off', seconds: 0 },
  { label: '5m', seconds: 5 * 60 },
  { label: '1h', seconds: 60 * 60 },
  { label: '6h', seconds: 6 * 60 * 60 },
]

// The first dropdown option always points at the live Approach 2 strategy —
// that's what the weekly email job targets and what's actively trading. Shadow
// stacks (Step 5+) append below this baseline.
const LIVE_OPTION = {
  source: 'live',
  strategy: 'nostate_cs_bear',
  label: 'Live: nostate_cs_bear (Approach 2)',
}

const SELECTED_STORAGE_KEY = 'abeval.selected'

function loadStoredSelection() {
  try {
    const raw = localStorage.getItem(SELECTED_STORAGE_KEY)
    if (!raw) return LIVE_OPTION
    const parsed = JSON.parse(raw)
    // Defensive: tolerate a stale shape from earlier code by requiring
    // the two fields the API calls actually consume.
    if (parsed && typeof parsed.strategy === 'string' && (parsed.source === 'live' || parsed.source === 'shadow')) {
      return parsed
    }
  } catch {}
  return LIVE_OPTION
}

const DECISION_STYLE = {
  keep: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/40', text: 'text-emerald-300', label: 'KEEP' },
  revert: { bg: 'bg-red-500/10', border: 'border-red-500/40', text: 'text-red-300', label: 'REVERT' },
  marginal: { bg: 'bg-amber-500/10', border: 'border-amber-500/40', text: 'text-amber-300', label: 'MARGINAL' },
  insufficient_data: { bg: 'bg-dark-700/30', border: 'border-dark-600', text: 'text-dark-300', label: 'INSUFFICIENT DATA' },
}

function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return '–'
  const sign = v > 0 ? '+' : ''
  return `${sign}${Number(v).toFixed(digits)}%`
}

function fmtNum(v, digits = 4) {
  if (v == null || Number.isNaN(v)) return '–'
  return Number(v).toFixed(digits)
}

function fmtCount(v) {
  return v == null ? '–' : v.toLocaleString()
}

function deltaClass(v, goodIsPositive = true) {
  if (v == null || Number.isNaN(v)) return 'text-dark-400'
  if (v === 0) return 'text-dark-300'
  const isGood = goodIsPositive ? v > 0 : v < 0
  return isGood ? 'text-emerald-400' : 'text-red-400'
}

function ThresholdRow({ label, value, threshold, passes, formatter = fmtNum, goodIsPositive = true }) {
  const cls = passes == null ? 'text-dark-400' : passes ? 'text-emerald-400' : 'text-red-400'
  const icon = passes == null ? '·' : passes ? '✓' : '✗'
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b border-dark-700/40 last:border-b-0">
      <div className="text-xs text-dark-300">{label}</div>
      <div className="flex items-baseline gap-3">
        <span className={`font-data text-sm ${deltaClass(value, goodIsPositive)}`}>{formatter(value)}</span>
        <span className="text-[10px] text-dark-500">vs {formatter(threshold)}</span>
        <span className={`font-bold text-xs w-3 text-center ${cls}`}>{icon}</span>
      </div>
    </div>
  )
}

function WindowSummary({ title, subtitle, summary, accent }) {
  if (!summary) {
    return (
      <Card variant="accent" accent={accent}>
        <CardHeader title={title} subtitle={subtitle} />
        <div className="text-xs text-dark-500 italic">No data.</div>
      </Card>
    )
  }
  const sells = summary.realized_sell_pct
  return (
    <Card variant="accent" accent={accent}>
      <CardHeader title={title} subtitle={subtitle} />
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <div>
          <dt className="text-dark-500">Total trades</dt>
          <dd className="text-dark-100 font-data">{fmtCount(summary.total_trades)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">SELLs (closed)</dt>
          <dd className="text-dark-100 font-data">{fmtCount(sells?.n ?? 0)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Win rate</dt>
          <dd className="text-dark-100 font-data">
            {sells?.win_rate != null ? `${(sells.win_rate * 100).toFixed(1)}%` : '–'}
          </dd>
        </div>
        <div>
          <dt className="text-dark-500">Avg sell return</dt>
          <dd className={`font-data ${deltaClass(sells?.mean)}`}>{fmtPct(sells?.mean)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Total return</dt>
          <dd className={`font-data ${deltaClass(summary.total_return_pct)}`}>{fmtPct(summary.total_return_pct)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Capital eff.</dt>
          <dd className={`font-data ${deltaClass(summary.capital_efficiency_pct)}`}>{fmtPct(summary.capital_efficiency_pct)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Sharpe / trade</dt>
          <dd className={`font-data ${deltaClass(summary.sharpe_per_trade)}`}>{fmtNum(summary.sharpe_per_trade)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Realized DD</dt>
          <dd className={`font-data ${deltaClass(summary.realized_max_drawdown_pct, false)}`}>{fmtPct(summary.realized_max_drawdown_pct)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Entries / day</dt>
          <dd className="text-dark-200 font-data">{fmtNum(summary.entry_rate_per_day, 2)}</dd>
        </div>
        <div>
          <dt className="text-dark-500">Exits / day</dt>
          <dd className="text-dark-200 font-data">{fmtNum(summary.exit_rate_per_day, 2)}</dd>
        </div>
      </dl>
    </Card>
  )
}

// Decision color drives both the post-window summary card accent AND the
// post line on the cumulative-return chart, so the visual identity of
// "post window" stays consistent across the page.
function decisionAccent(decision) {
  if (decision === 'revert') return chartColors.pnlDown
  if (decision === 'keep') return chartColors.pnlUp
  return chartColors.brand // marginal / insufficient_data → amber
}

// Merge pre + post curves into a single data array for Recharts. Each
// row carries either pre_pct or post_pct (or both, when timestamps
// align), letting one chart render both lines on a shared X axis. The
// cutoff date is the only point both lines share — pre ends at 0% there,
// post begins at 0% there, producing the visual "fork" at the experiment
// boundary.
function mergeCurves(curves) {
  if (!curves) return []
  const map = new Map()
  for (const p of curves.pre || []) {
    if (!p) continue
    const row = map.get(p.date) || { date: p.date }
    row.pre_pct = p.cum_pct
    map.set(p.date, row)
  }
  for (const p of curves.post || []) {
    if (!p) continue
    const row = map.get(p.date) || { date: p.date }
    row.post_pct = p.cum_pct
    map.set(p.date, row)
  }
  return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date))
}

function CumulativeReturnChart({ trades, cutoffDate, decision }) {
  const merged = useMemo(() => mergeCurves(trades?.cumulative_returns), [trades])
  const postSells = (trades?.post_trades || []).filter(
    (t) => t.action === 'SELL' && t.realized_gain != null
  ).length
  if (!merged.length) return null

  const postColor = decisionAccent(decision)
  return (
    <Card>
      <CardHeader
        title="Cumulative realized return"
        subtitle="Each line anchored at 0% on its window start. Post-cutoff line forks at the experiment boundary."
      />
      {postSells === 0 && (
        <div className="mb-3 text-[11px] text-amber-300/90 bg-amber-500/5 border border-amber-500/20 rounded px-2 py-1.5">
          No post-cutoff SELLs yet — chart anchors at 0%. Re-check as trades close.
        </div>
      )}
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={merged} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <XAxis
            dataKey="date"
            tick={{ fill: chartAxis.tick, fontSize: 10 }}
            axisLine={{ stroke: chartAxis.axisLine }}
            tickLine={{ stroke: chartAxis.axisLine }}
            minTickGap={24}
          />
          <YAxis
            tick={{ fill: chartAxis.tick, fontSize: 10 }}
            axisLine={{ stroke: chartAxis.axisLine }}
            tickLine={{ stroke: chartAxis.axisLine }}
            tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
            width={50}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            labelStyle={tooltipLabelStyle}
            formatter={(v, name) => [`${Number(v).toFixed(2)}%`, name]}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: chartAxis.tick }}
            iconType="line"
            // Hide on small screens — chart legend is non-essential there
            // and the axis already implies the colors.
            content={({ payload }) => (
              <div className="hidden sm:flex justify-center gap-4 text-[11px] mt-1">
                {payload?.map((entry) => (
                  <span key={entry.value} style={{ color: entry.color }}>
                    {entry.value}
                  </span>
                ))}
              </div>
            )}
          />
          <ReferenceLine
            x={cutoffDate}
            stroke={chartAxis.reference}
            strokeDasharray="3 3"
            label={{ value: 'cutoff', fill: chartAxis.tick, fontSize: 10, position: 'top' }}
          />
          <Line
            type="monotone"
            dataKey="pre_pct"
            name="Pre-cutoff"
            stroke={chartColors.spy}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="post_pct"
            name="Post-cutoff"
            stroke={postColor}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  )
}

const TABLE_COLUMNS = [
  { key: 'executed_at', label: 'Date' },
  { key: 'ticker', label: 'Ticker' },
  { key: 'action', label: 'Action' },
  { key: 'price', label: 'Entry/Exit' },
  { key: 'cost_basis', label: 'Cost basis' },
  { key: 'realized_pct', label: 'Return %' },
  { key: 'hold_days', label: 'Hold (d)' },
  { key: 'reason', label: 'Reason' },
]

function compareValues(a, b, dir) {
  // Nulls always sort last regardless of direction so missing data
  // doesn't crowd the top of the table.
  if (a == null && b == null) return 0
  if (a == null) return 1
  if (b == null) return -1
  if (typeof a === 'number' && typeof b === 'number') {
    return dir === 'asc' ? a - b : b - a
  }
  const av = String(a), bv = String(b)
  return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
}

function PostTradesTable({ rows, open, onToggle, sortKey, sortDir, onSort }) {
  const sorted = useMemo(() => {
    const arr = [...(rows || [])]
    arr.sort((a, b) => compareValues(a[sortKey], b[sortKey], sortDir))
    return arr
  }, [rows, sortKey, sortDir])

  const fmt = (col, val) => {
    if (val == null) return '–'
    if (col === 'executed_at') return val.slice(0, 10)
    if (col === 'realized_pct') return `${val > 0 ? '+' : ''}${Number(val).toFixed(2)}%`
    if (col === 'price' || col === 'cost_basis') return `$${Number(val).toFixed(2)}`
    return String(val)
  }

  return (
    <Card>
      <button
        onClick={onToggle}
        className="w-full flex items-baseline justify-between text-left"
        type="button"
      >
        <CardHeader
          title={`Post-cutoff trades (${rows?.length ?? 0})`}
          subtitle="Per-trade attribution. Sort by clicking column headers."
        />
        <span className="text-xs text-dark-400 ml-3 whitespace-nowrap">
          {open ? 'Hide ▲' : 'Show ▼'}
        </span>
      </button>
      {open && (
        <>
          {/* Desktop table */}
          <div className="hidden sm:block overflow-x-auto mt-2">
            <table className="w-full text-xs font-data">
              <thead>
                <tr className="text-dark-400 border-b border-dark-700/60">
                  {TABLE_COLUMNS.map((col) => (
                    <th
                      key={col.key}
                      onClick={() => onSort(col.key)}
                      className="text-left py-1.5 px-2 cursor-pointer hover:text-dark-200 select-none"
                    >
                      {col.label}
                      {sortKey === col.key && (
                        <span className="ml-1 text-primary-400">{sortDir === 'asc' ? '▲' : '▼'}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.length === 0 && (
                  <tr>
                    <td colSpan={TABLE_COLUMNS.length} className="py-3 text-center text-dark-500 italic">
                      No post-cutoff trades in this window.
                    </td>
                  </tr>
                )}
                {sorted.map((row) => (
                  <tr key={row.id} className="border-b border-dark-800/60 hover:bg-dark-800/40">
                    {TABLE_COLUMNS.map((col) => {
                      const isReturn = col.key === 'realized_pct'
                      const cls = isReturn ? deltaClass(row.realized_pct) : 'text-dark-200'
                      return (
                        <td key={col.key} className={`py-1.5 px-2 ${cls}`}>
                          {fmt(col.key, row[col.key])}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Mobile condensed list */}
          <div className="sm:hidden mt-2 space-y-2">
            {sorted.length === 0 && (
              <div className="text-xs text-dark-500 italic">No post-cutoff trades in this window.</div>
            )}
            {sorted.map((row) => (
              <div key={row.id} className="border border-dark-700/60 rounded p-2">
                <div className="flex items-baseline justify-between text-xs">
                  <span className="font-bold text-dark-100">{row.ticker}</span>
                  <span className={`font-data ${deltaClass(row.realized_pct)}`}>
                    {fmt('realized_pct', row.realized_pct)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between text-[11px] text-dark-400 mt-1">
                  <span>{row.action} · {fmt('executed_at', row.executed_at)}</span>
                  <span>{row.hold_days != null ? `${row.hold_days}d` : ''}</span>
                </div>
                {row.reason && (
                  <div className="text-[11px] text-dark-500 mt-1 truncate">{row.reason}</div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </Card>
  )
}

function DecisionBanner({ summary, post }) {
  const decision = summary?.decision || 'insufficient_data'
  const style = DECISION_STYLE[decision] || DECISION_STYLE.insufficient_data
  const postSells = post?.realized_sell_pct?.n ?? 0
  const criteria = summary?.decision_criteria || {}
  const minPostSells = criteria.min_post_sells ?? 5
  const minReturnDelta = criteria.min_return_delta_pp ?? -5.0
  const minSharpeDelta = criteria.min_sharpe_delta ?? 0.0
  return (
    <div className={`border ${style.border} ${style.bg} rounded-xl p-5`}>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <div className="text-[10px] tracking-widest text-dark-400 mb-1">DECISION</div>
          <div className={`text-2xl md:text-3xl font-bold ${style.text}`}>{style.label}</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] tracking-widest text-dark-400 mb-1">POST-CUTOFF SELLS</div>
          <div className="text-2xl md:text-3xl font-data text-dark-100">
            {postSells} <span className="text-sm text-dark-500">/ {minPostSells} req.</span>
          </div>
        </div>
      </div>
      {summary?.decision_reason && (
        <div className="mt-3 text-xs text-dark-200 leading-relaxed">{summary.decision_reason}</div>
      )}
      {decision === 'insufficient_data' && (
        <div className="mt-3 pt-3 border-t border-dark-700/40 flex flex-wrap items-center gap-2 text-[11px]">
          <span className="text-dark-400 uppercase tracking-wider">To pass:</span>
          <span className="font-data px-2 py-0.5 rounded border border-dark-700 bg-dark-800/60 text-dark-200">
            sells ≥ {minPostSells}
          </span>
          <span className="font-data px-2 py-0.5 rounded border border-dark-700 bg-dark-800/60 text-dark-200">
            return Δ ≥ {minReturnDelta > 0 ? '+' : ''}{minReturnDelta.toFixed(2)}pp
          </span>
          <span className="font-data px-2 py-0.5 rounded border border-dark-700 bg-dark-800/60 text-dark-200">
            Sharpe Δ ≥ {minSharpeDelta > 0 ? '+' : ''}{minSharpeDelta.toFixed(2)}
          </span>
        </div>
      )}
    </div>
  )
}

// ── Cap-Delta Diagnostics card ────────────────────────────────────────────
//
// Visible only when the selected shadow stack has
// `scorer_overrides.disable_excellence_cap === true`. Renders a population-
// level read of c_score vs c_score_uncapped over a rolling window: headline
// percentage, four-bucket histogram, and the top divergent tickers. The
// goal is to answer the verdict-day question "is the no-cap arm even
// receiving divergent inputs, or is it a silent no-op?" without an SQL
// spelunking session under decision pressure.
const CAP_DELTA_DAYS_KEY = 'abeval.capDelta.days'
const CAP_DELTA_DAY_CHIPS = [1, 3, 7, 14]

function loadStoredCapDeltaDays() {
  try {
    const raw = localStorage.getItem(CAP_DELTA_DAYS_KEY)
    const n = raw == null ? null : Number(raw)
    if (CAP_DELTA_DAY_CHIPS.includes(n)) return n
  } catch {}
  return 1
}

function CapDeltaBucketBar({ buckets }) {
  // Stacked horizontal bar across the four delta_buckets. Width is
  // proportional to bucket pct; tooltips on hover for exact counts.
  const total = buckets.reduce((acc, b) => acc + (b.n || 0), 0)
  if (!total) return <div className="text-xs text-dark-500 italic">No rows in window.</div>
  const palette = {
    '0':       'bg-dark-700',
    '0.5-1.5': 'bg-amber-500/70',
    '1.5-3.0': 'bg-orange-500/80',
    '3.0+':    'bg-red-500/80',
  }
  return (
    <div>
      <div className="flex h-6 rounded overflow-hidden border border-dark-700/60">
        {buckets.map(b => (
          <div
            key={b.bucket}
            className={`${palette[b.bucket] || 'bg-dark-600'} flex items-center justify-center text-[10px] text-dark-100`}
            style={{ width: `${b.pct}%` }}
            title={`Δ ${b.bucket}: ${b.n.toLocaleString()} rows (${b.pct}%)`}
          >
            {b.pct >= 6 ? `${b.pct}%` : ''}
          </div>
        ))}
      </div>
      <div className="flex justify-between mt-1 text-[10px] text-dark-500">
        {buckets.map(b => (
          <span key={b.bucket}>Δ {b.bucket} <span className="text-dark-300 font-data">({b.n.toLocaleString()})</span></span>
        ))}
      </div>
    </div>
  )
}

function CapDeltaCard({ selected, refreshSeconds }) {
  const enabled = selected?.scorerOverrides?.disable_excellence_cap === true
  const [days, setDays] = useState(loadStoredCapDeltaDays)
  const [body, setBody] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(true)
  const timer = useRef(null)

  useEffect(() => {
    try { localStorage.setItem(CAP_DELTA_DAYS_KEY, String(days)) } catch {}
  }, [days])

  const fetchOnce = useCallback(async () => {
    if (!enabled) return
    setLoading(true)
    setError('')
    try {
      const res = await api.getCapDeltaDiagnostics({
        days,
        strategy: selected.parentStrategy || undefined,
      })
      setBody(res)
    } catch (e) {
      setError(e?.message || 'Fetch failed')
    } finally {
      setLoading(false)
    }
  }, [enabled, days, selected])

  useEffect(() => { fetchOnce() }, [fetchOnce])

  // Mirror the parent ABEval auto-refresh cadence — same interval, same
  // bypass-cache semantics. Off when refreshSeconds === 0.
  useEffect(() => {
    if (timer.current) clearTimeout(timer.current)
    if (!enabled) return undefined
    if (refreshSeconds > 0) {
      timer.current = setTimeout(fetchOnce, refreshSeconds * 1000)
    }
    return () => { if (timer.current) clearTimeout(timer.current) }
  }, [refreshSeconds, fetchOnce, enabled])

  if (!enabled) return null

  const pop = body?.population
  const xings = body?.threshold_crossings
  const tickers = body?.top_divergent_tickers || []

  return (
    <Card>
      <div className="flex items-baseline justify-between gap-3">
        <CardHeader
          title="Cap-Delta Diagnostics"
          subtitle="Population view — what would scoring look like without the excellence cap?"
        />
        <button
          onClick={() => setOpen(o => !o)}
          className="text-xs text-dark-400 hover:text-dark-200 px-2 py-1 rounded border border-dark-700"
        >
          {open ? 'Hide' : 'Show'}
        </button>
      </div>
      {open && (
        <>
          <div className="flex flex-wrap items-center gap-2 mt-1 mb-3">
            <span className="text-[11px] text-dark-500">Window:</span>
            {CAP_DELTA_DAY_CHIPS.map(n => (
              <button
                key={n}
                onClick={() => setDays(n)}
                className={`text-[11px] px-2 py-0.5 rounded border ${
                  days === n
                    ? 'border-primary-500/60 text-primary-200 bg-primary-500/10'
                    : 'border-dark-700 text-dark-400 hover:border-dark-600 hover:text-dark-200'
                }`}
              >
                {n}d
              </button>
            ))}
            {loading && <span className="text-[11px] text-dark-500">Loading…</span>}
          </div>
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-300 text-xs px-3 py-2 rounded mb-3">
              {error}
            </div>
          )}
          {pop && (
            <>
              <div className="text-sm text-dark-200 mb-3">
                <span className="font-data text-emerald-300">{pop.rows_with_delta_pct}%</span> of scored
                stocks would score differently without the excellence cap
                {' '}
                (<span className="font-data">{pop.rows_with_delta.toLocaleString()}</span> of{' '}
                <span className="font-data">{pop.rows_total.toLocaleString()}</span> rows in the last {days} day{days === 1 ? '' : 's'}).
                {pop.rows_with_delta > 0 && (
                  <span className="text-dark-400"> Avg Δ when present: <span className="font-data text-dark-200">{pop.avg_delta_when_present}</span>; max Δ: <span className="font-data text-dark-200">{pop.max_delta}</span>.</span>
                )}
              </div>
              <CapDeltaBucketBar buckets={pop.delta_buckets} />
              {xings && (
                <div className="mt-3 text-xs text-dark-400">
                  Threshold crossings (buy_threshold = <span className="font-data text-dark-200">{xings.buy_threshold}</span>):{' '}
                  <span className="font-data text-emerald-300">{xings.rows_baseline_below_threshold_uncapped_above}</span>
                  {' '}rows would qualify without the cap.
                </div>
              )}
            </>
          )}
          {tickers.length > 0 && (
            <div className="mt-4">
              <SectionLabel>Top divergent tickers</SectionLabel>
              <table className="w-full text-xs text-dark-200 mt-2">
                <thead className="text-dark-500 text-[10px] uppercase tracking-wide">
                  <tr>
                    <th className="text-left py-1.5">Ticker</th>
                    <th className="text-right">Max Δ</th>
                    <th className="text-right">Capped</th>
                    <th className="text-right">Uncapped</th>
                    <th className="text-right">Rows</th>
                    <th className="text-right">Crossed?</th>
                  </tr>
                </thead>
                <tbody>
                  {tickers.slice(0, 10).map(t => (
                    <tr key={t.ticker} className="border-t border-dark-700/40">
                      <td className="py-1.5 font-data">
                        <a href={`/stock/${t.ticker}`} className="text-primary-300 hover:underline">{t.ticker}</a>
                      </td>
                      <td className="text-right font-data">{t.max_delta}</td>
                      <td className="text-right font-data text-dark-300">{t.max_capped_score ?? '–'}</td>
                      <td className="text-right font-data">{t.max_uncapped_score ?? '–'}</td>
                      <td className="text-right font-data text-dark-400">{t.n_rows}</td>
                      <td className="text-right">
                        {t.crossed_buy_threshold_at_least_once
                          ? <span className="text-emerald-400">✓</span>
                          : <span className="text-dark-500">·</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </Card>
  )
}


function GateMeter({ metric }) {
  const { label, n, target, kind } = metric
  const met = target != null && n >= target
  const frac = target ? Math.min(n / target, 1) : 0
  // Sufficiency = the weekly A/B rule has enough closed sells to run. It is
  // not the arm's promotion gate, so it never earns the green "met" styling.
  const sufficiency = kind === 'sufficiency'
  const shownLabel = sufficiency ? 'data sufficiency (weekly A/B)' : label
  const hint = sufficiency
    ? 'Closed sells needed before the weekly shadow-vs-baseline rule can evaluate this arm. Not a promotion gate.'
    : label
  return (
    <div className={`flex items-center gap-2 text-xs ${sufficiency ? 'italic' : ''}`}>
      <span className="text-dark-400 flex-1 truncate" title={hint}>{shownLabel}</span>
      <span className={`tabular-nums ${met && !sufficiency ? 'text-emerald-400' : 'text-dark-200'}`}>
        {n}{target != null && ` / ${target}`}{met && !sufficiency && ' ✓'}
      </span>
      {target != null && (
        <div className="w-16 h-1 rounded bg-dark-700 overflow-hidden shrink-0">
          <div
            className={`h-full rounded ${met && !sufficiency ? 'bg-emerald-500' : 'bg-dark-400'}`}
            style={{ width: `${frac * 100}%` }}
          />
        </div>
      )}
    </div>
  )
}

// Accrual toward each arm's pre-registered promotion gate. Read-only
// progress — verdicts always come from the weekly A/B email decision rule.
function GateProgressCard() {
  const { data, error, loading } = useApi(() => api.getExperimentGates(), [])

  if (loading && !data) {
    return <Card><div className="skeleton h-24 rounded" /></Card>
  }
  // Silent degrade: the rest of the page works without the gates readout.
  if (error || !data) return null

  const stops = data.program_clocks?.stop_loss_recheck
  return (
    <Card>
      <CardHeader
        title="Gate Progress"
        subtitle="Accrual toward each arm's pre-registered promotion gate — verdicts come from the weekly email, not this card"
      />
      {stops && (
        <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs border-b border-dark-700/40 pb-3">
          <span className="text-dark-400">{stops.label}:</span>
          <span className={`tabular-nums ${stops.n >= stops.target ? 'text-emerald-400' : 'text-dark-200'}`}>
            {stops.n} / {stops.target} stops
          </span>
          {stops.avg_loss_pct != null && (
            <span className="text-dark-400">
              avg {fmtPct(stops.avg_loss_pct)} vs {fmtPct(stops.bar_pct, 1)} bar
            </span>
          )}
          {/* Mechanical verdict (fires server-side once n >= target): the one
              clock whose pre-registered rule is pure arithmetic. */}
          {stops.verdict && (
            <span className={`text-[10px] px-2 py-0.5 rounded border font-semibold ${
              stops.verdict === 'PASS'
                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20'
                : 'bg-red-500/15 text-red-400 border-red-500/20'
            }`}>
              {stops.verdict}
            </span>
          )}
        </div>
      )}
      {/* Date-based calendar clocks: the pre-registered re-check schedule
          self-reports here instead of living in session notes. */}
      {(data.program_clocks?.calendar || []).length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2 text-[10px] border-b border-dark-700/40 pb-3">
          {data.program_clocks.calendar.map(c => (
            <span
              key={c.label}
              title={`Due ${c.due_date}`}
              className={`px-2 py-0.5 rounded border ${
                c.due
                  ? 'bg-red-500/15 text-red-400 border-red-500/20 font-semibold'
                  : 'bg-dark-850 text-dark-400 border-dark-700'
              }`}
            >
              {c.label} · {c.due ? 'DUE' : `${c.days_until}d`}
            </span>
          ))}
        </div>
      )}
      {/* Vintage spread: the same champion started on staggered dates. The
          spread of alpha-vs-SPY across starts is live launch-luck — the
          confound behind three earlier cohort reads. Benchmarks, not arms. */}
      {(data.program_clocks?.vintage_spread?.stacks || []).length > 0 && (
        <div className="mb-3 text-[10px] border-b border-dark-700/40 pb-3">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-dark-400" title={data.program_clocks.vintage_spread.note}>
              Vintage spread (alpha vs SPY since own start):
            </span>
            {data.program_clocks.vintage_spread.stacks.map(v => (
              <span
                key={v.name}
                className="px-2 py-0.5 rounded border bg-dark-850 text-dark-300 border-dark-700 tabular-nums"
                title={`started ${v.activated_at} · ${v.days}d · return ${v.return_pct ?? '–'}% · SPY ${v.spy_pct ?? '–'}% · ${v.n_positions} positions`}
              >
                {v.label} {v.alpha_pp != null ? `${v.alpha_pp > 0 ? '+' : ''}${v.alpha_pp.toFixed(1)}pp` : 'pending'}
              </span>
            ))}
            {data.program_clocks.vintage_spread.spread_pp != null && (
              <span className="text-dark-200 tabular-nums">
                spread {data.program_clocks.vintage_spread.spread_pp.toFixed(1)}pp · σ {data.program_clocks.vintage_spread.stdev_pp.toFixed(1)}pp
                <span className="text-dark-500"> (n={data.program_clocks.vintage_spread.n})</span>
              </span>
            )}
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4">
        {(data.arms || []).map(arm => (
          <div key={arm.id} className="space-y-1.5 min-w-0">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-xs font-medium text-dark-100 truncate" title={arm.description || arm.name}>
                {arm.name.replace(/^shadow_/, '')}
              </span>
              <span className="text-[10px] text-dark-500 shrink-0">
                {arm.days_accrued != null && `${arm.days_accrued}d`} · {arm.buys}B/{arm.sells}S
                {arm.pyramids > 0 && `/${arm.pyramids}P`}
              </span>
            </div>
            {arm.gate_metrics.map(m => <GateMeter key={m.label} metric={m} />)}
          </div>
        ))}
      </div>
    </Card>
  )
}

// Buy funnel: why each candidate was or wasn't bought, per cycle per
// strategy — the audit trail behind every "why didn't arm X buy Y?" read.
// Rows come from backend/buy_funnel.py (written by the live cycle and every
// shadow arm); stage = the FIRST gate in evaluate_buys that stopped a name.
const FUNNEL_STAGE_LABELS = {
  dead_data: 'dead data', cz_prefilter: 'CZ pre-filter', score_floor: 'score floor',
  bear_exception_pool: 'bear-exception pool', soft_zone_det: 'soft zone (weak det.)',
  no_score: 'no score', quality_c: 'C filter', quality_l: 'L filter', quality_growth: 'growth C/A',
  volume_gate: 'volume gate', earnings_window: 'earnings window', cz_cs_only: 'CZ CS-only',
  sector_cap: 'sector cap', min_position_value: 'min position', bad_price: 'bad price',
  ml_veto: 'ML veto', chop_entry_bar: 'chop entry bar', duplicate_class: 'dup. share class',
  ranked: 'ranked (not taken)', exec_skipped: 'exec skipped', bought: 'bought',
  market_gate: 'market gate', portfolio_full: 'book full', cash_reserve: 'cash reserve',
  circuit_breaker: 'circuit breaker', exec_stopped: 'execution stopped',
}
const funnelStageClass = (stage) => (
  stage === 'bought' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20'
    : stage === 'ranked' || stage === 'exec_skipped' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20'
      : 'bg-dark-850 text-dark-400 border-dark-700'
)
const funnelTime = (iso) => (iso ? new Date(iso).toLocaleString(undefined, {
  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
}) : '')
const FUNNEL_PREVIEW = 25

function BuyFunnelCard() {
  const [key, setKey] = useState('')
  const [tickerInput, setTickerInput] = useState('')
  const [ticker, setTicker] = useState('')
  const [showAll, setShowAll] = useState(false)
  const { data, error, loading } = useApi(
    () => api.getBuyFunnel({ key: key || undefined, ticker: ticker || undefined }),
    [key, ticker],
  )

  if (loading && !data) {
    return <Card><div className="skeleton h-24 rounded" /></Card>
  }
  if (error || !data) return null

  const strategies = data.strategies || []
  const strategyLabel = (s) => (
    s.shadow_strategy_id != null
      ? `${s.strategy.replace(/^nostate_/, '')} · shadow ${s.shadow_strategy_id}`
      : `${s.strategy.replace(/^nostate_/, '')} · user ${s.user_id}`
  )
  const cycle = data.cycle
  const stageOrder = data.stage_order || []
  const rows = ticker ? (data.rows || []) : (cycle?.rows || [])
  const shown = showAll ? rows : rows.slice(0, FUNNEL_PREVIEW)

  return (
    <Card>
      <CardHeader
        title="Buy Funnel"
        subtitle="Why each candidate was or wasn't bought — the first gate that stopped it, per cycle per strategy (21-day retention)"
      />
      <div className="flex flex-wrap items-center gap-2 mb-3 text-xs">
        <select
          value={key}
          onChange={e => { setKey(e.target.value); setShowAll(false) }}
          className="bg-dark-850 border border-dark-700 rounded px-2 py-1 text-dark-200"
          aria-label="Strategy"
        >
          <option value="">latest cycle (any strategy)</option>
          {strategies.map(s => (
            <option key={s.key} value={s.key}>{strategyLabel(s)} · {funnelTime(s.last_cycle_at)}</option>
          ))}
        </select>
        <form
          onSubmit={e => { e.preventDefault(); setTicker(tickerInput.trim().toUpperCase()); setShowAll(false) }}
          className="flex items-center gap-1"
        >
          <input
            value={tickerInput}
            onChange={e => setTickerInput(e.target.value)}
            placeholder="why not… TICKER"
            className="bg-dark-850 border border-dark-700 rounded px-2 py-1 w-36 text-dark-200 placeholder:text-dark-500"
            aria-label="Ticker lookup"
          />
          <button type="submit" className="px-2 py-1 rounded border border-dark-700 text-dark-300 hover:text-dark-100">
            look up
          </button>
          {ticker && (
            <button
              type="button"
              onClick={() => { setTicker(''); setTickerInput('') }}
              className="px-2 py-1 rounded border border-dark-700 text-dark-400 hover:text-dark-100"
            >
              clear
            </button>
          )}
        </form>
        {!ticker && cycle?.cycle_at && (
          <span className="text-dark-500">
            {cycle.strategy?.replace(/^nostate_/, '')} · cycle {funnelTime(cycle.cycle_at)}
          </span>
        )}
      </div>

      {!ticker && cycle && (
        <div className="flex flex-wrap gap-1.5 mb-3 text-[10px]">
          {(cycle.notes || []).map(n => (
            <span key={n.id} className="px-2 py-0.5 rounded border bg-red-500/10 text-red-300 border-red-500/20" title={n.detail || ''}>
              {FUNNEL_STAGE_LABELS[n.stage] || n.stage}{n.detail ? ` · ${n.detail}` : ''}
            </span>
          ))}
          {stageOrder.filter(st => cycle.stage_counts?.[st]).map(st => (
            <span key={st} className={`px-2 py-0.5 rounded border ${funnelStageClass(st)}`}>
              {FUNNEL_STAGE_LABELS[st] || st} · {cycle.stage_counts[st]}
            </span>
          ))}
          {rows.length === 0 && (cycle.notes || []).length === 0 && (
            <span className="text-dark-500">No funnel rows yet — written at the next trading cycle.</span>
          )}
        </div>
      )}

      {rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-[10px] uppercase tracking-wide text-dark-500">
              <tr>
                {ticker && <th className="text-left py-1 pr-3 font-medium">cycle</th>}
                {ticker && <th className="text-left py-1 pr-3 font-medium">strategy</th>}
                {!ticker && <th className="text-left py-1 pr-3 font-medium">ticker</th>}
                <th className="text-left py-1 pr-3 font-medium">stage</th>
                <th className="text-left py-1 pr-3 font-medium">detail</th>
                <th className="text-right py-1 pr-3 font-medium">score</th>
                <th className="text-right py-1 pr-3 font-medium">composite</th>
                <th className="text-right py-1 font-medium">rank</th>
              </tr>
            </thead>
            <tbody>
              {shown.map(r => (
                <tr key={r.id} className="border-t border-dark-800/60">
                  {ticker && <td className="py-1 pr-3 text-dark-400 whitespace-nowrap">{funnelTime(r.cycle_at)}</td>}
                  {ticker && (
                    <td className="py-1 pr-3 text-dark-300 whitespace-nowrap">
                      {r.strategy.replace(/^nostate_/, '')}{r.shadow_strategy_id != null ? ` · s${r.shadow_strategy_id}` : ` · u${r.user_id}`}
                    </td>
                  )}
                  {!ticker && <td className="py-1 pr-3 font-medium text-dark-100">{r.ticker}</td>}
                  <td className="py-1 pr-3">
                    <span className={`px-1.5 py-0.5 rounded border text-[10px] ${funnelStageClass(r.stage)}`}>
                      {FUNNEL_STAGE_LABELS[r.stage] || r.stage}
                    </span>
                  </td>
                  <td className="py-1 pr-3 text-dark-400 max-w-[28rem] truncate" title={r.detail || ''}>{r.detail || ''}</td>
                  <td className="py-1 pr-3 text-right tabular-nums text-dark-300">{r.score != null ? r.score.toFixed(0) : ''}</td>
                  <td className="py-1 pr-3 text-right tabular-nums text-dark-300">{r.composite != null ? r.composite.toFixed(0) : ''}</td>
                  <td className="py-1 text-right tabular-nums text-dark-300">{r.rank ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > FUNNEL_PREVIEW && (
            <button
              type="button"
              onClick={() => setShowAll(v => !v)}
              className="mt-2 text-[11px] text-dark-400 hover:text-dark-100"
            >
              {showAll ? 'show fewer' : `show all ${rows.length}`}
            </button>
          )}
        </div>
      )}
      {ticker && rows.length === 0 && (
        <div className="text-xs text-dark-500">No funnel rows for {ticker} in the last 7 days — it never reached a candidate pool (score below the query floor, stale, or already held).</div>
      )}
    </Card>
  )
}

// Event-sourced program history: what happened and when — the complement to
// Gate Progress (which answers "where are we now?"). Rows come from the
// server-side gate-diff writer, the seeded history, and owner entries here.
const LEDGER_CATEGORIES = ['experiment', 'verdict', 'gate', 'decision', 'fix', 'infra', 'research']
const LEDGER_PREVIEW_COUNT = 10

function ledgerDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleDateString(undefined, {
    timeZone: 'UTC', year: 'numeric', month: 'short', day: 'numeric',
  })
}

function ProgramLedgerCard() {
  const { data, error, loading, refetch, setData } = useApi(() => api.getProgramMilestones(), [])
  const [filter, setFilter] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const [formOpen, setFormOpen] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [newDetail, setNewDetail] = useState('')
  const [newCategory, setNewCategory] = useState('decision')
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  if (loading && !data) {
    return <Card><div className="skeleton h-24 rounded" /></Card>
  }
  // Silent degrade like GateProgressCard — the page works without it.
  if (error || !data) return null

  const rows = filter ? data.filter(r => r.category === filter) : data
  const visible = showAll ? rows : rows.slice(0, LEDGER_PREVIEW_COUNT)

  const handleAdd = async (e) => {
    e.preventDefault()
    const title = newTitle.trim()
    if (!title || saving) return
    setSaving(true)
    try {
      const row = await api.addProgramMilestone({
        title,
        detail: newDetail.trim() || null,
        category: newCategory,
      })
      setData(prev => [row, ...(prev || [])])
      setNewTitle('')
      setNewDetail('')
      setFormOpen(false)
    } catch (err) {
      toast.error(`Failed to add milestone: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id) => {
    setData(prev => (prev || []).filter(r => r.id !== id))
    try {
      await api.deleteProgramMilestone(id)
    } catch (err) {
      toast.error(`Delete failed: ${err.message}`)
      refetch()
    }
  }

  return (
    <Card>
      <div className="flex items-start justify-between gap-2">
        <CardHeader
          title="Program Ledger"
          subtitle="What happened and when — gate crossings and verdicts record themselves; decisions get written down"
        />
        <button
          onClick={() => setFormOpen(o => !o)}
          className="shrink-0 text-xs px-2 py-1 rounded border border-dark-700 text-dark-300 hover:border-dark-500 hover:text-dark-100"
        >
          {formOpen ? 'Cancel' : '+ Milestone'}
        </button>
      </div>

      {formOpen && (
        <form onSubmit={handleAdd} className="mb-3 grid grid-cols-1 sm:grid-cols-[1fr_auto_auto] gap-2 text-xs">
          <input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="What happened?"
            maxLength={200}
            className="bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100"
            autoFocus
          />
          <select
            value={newCategory}
            onChange={(e) => setNewCategory(e.target.value)}
            className="bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100"
          >
            {LEDGER_CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button
            type="submit"
            disabled={!newTitle.trim() || saving}
            className="px-3 py-1.5 rounded border border-primary-500/40 text-primary-300 hover:border-primary-500/70 disabled:opacity-40"
          >
            {saving ? 'Saving…' : 'Add'}
          </button>
          <input
            value={newDetail}
            onChange={(e) => setNewDetail(e.target.value)}
            placeholder="Detail (optional)"
            maxLength={500}
            className="sm:col-span-3 bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100"
          />
        </form>
      )}

      <div className="mb-3 flex flex-wrap gap-1.5 text-[10px]">
        <button
          onClick={() => setFilter(null)}
          className={`px-2 py-0.5 rounded border ${!filter
            ? 'bg-dark-800 text-dark-100 border-dark-500'
            : 'bg-dark-850 text-dark-400 border-dark-700 hover:text-dark-200'}`}
        >
          all · {data.length}
        </button>
        {LEDGER_CATEGORIES.filter(c => data.some(r => r.category === c)).map(c => (
          <button
            key={c}
            onClick={() => setFilter(f => (f === c ? null : c))}
            className={`px-2 py-0.5 rounded border ${filter === c
              ? 'bg-dark-800 text-dark-100 border-dark-500'
              : 'bg-dark-850 text-dark-400 border-dark-700 hover:text-dark-200'}`}
          >
            {c} · {data.filter(r => r.category === c).length}
          </button>
        ))}
      </div>

      <div className="space-y-2.5">
        {visible.map(r => (
          <div key={r.id} className="flex items-start gap-3 min-w-0 group">
            <span className="shrink-0 w-24 pt-0.5 text-[10px] tabular-nums text-dark-400">
              {ledgerDate(r.occurred_at)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 flex-wrap">
                <span className="text-xs text-dark-100">{r.title}</span>
                <span className="text-[9px] px-1.5 py-px rounded border bg-dark-850 text-dark-400 border-dark-700">
                  {r.category}
                </span>
                {r.source === 'auto' && (
                  <span className="text-[9px] text-dark-500" title="Recorded mechanically by the gate-diff writer">
                    auto
                  </span>
                )}
              </div>
              {r.detail && (
                <p className="mt-0.5 text-[11px] leading-snug text-dark-400">{r.detail}</p>
              )}
            </div>
            {r.source === 'owner' && (
              <button
                onClick={() => handleDelete(r.id)}
                title="Delete this entry"
                className="shrink-0 opacity-0 group-hover:opacity-100 text-dark-500 hover:text-red-400 text-xs px-1"
              >
                ×
              </button>
            )}
          </div>
        ))}
        {visible.length === 0 && (
          <p className="text-xs text-dark-400 italic">No milestones in this category yet.</p>
        )}
      </div>

      {rows.length > LEDGER_PREVIEW_COUNT && (
        <button
          onClick={() => setShowAll(s => !s)}
          className="mt-3 text-[11px] text-dark-400 hover:text-dark-200"
        >
          {showAll ? 'Show recent only' : `Show all ${rows.length}`}
        </button>
      )}
    </Card>
  )
}

export default function ABEval() {
  const { user } = useAuth()
  const [selected, setSelected] = useState(loadStoredSelection)
  const [shadows, setShadows] = useState([])
  const [cutoffDate, setCutoffDate] = useState('2026-05-07')
  const [preWindowDays, setPreWindowDays] = useState(30)
  const [postWindowDays, setPostWindowDays] = useState('')  // empty → endpoint default
  const [excludePyramids, setExcludePyramids] = useState(true)
  const [refreshSeconds, setRefreshSeconds] = useState(6 * 60 * 60)
  const [data, setData] = useState(null)
  const [trades, setTrades] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastFetched, setLastFetched] = useState(null)
  const [tableOpen, setTableOpen] = useState(false)
  const [sortKey, setSortKey] = useState('executed_at')
  const [sortDir, setSortDir] = useState('desc')
  const [emailSending, setEmailSending] = useState(false)
  const refreshTimer = useRef(null)
  const toast = useToast()

  const fetchEval = useCallback(async (opts = {}) => {
    setLoading(true)
    setError('')
    try {
      if (opts.bypassCache) {
        cache.invalidate('/api/admin/strategy-ab-eval')
        cache.invalidate('/api/admin/strategy-ab-eval-trades')
      }
      const params = {
        strategy: selected.strategy,
        source: selected.source,
        cutoffDate,
        preWindowDays,
        postWindowDays: postWindowDays === '' ? null : Number(postWindowDays),
        excludePyramids,
      }
      // Fetch both endpoints in parallel — same window math, complementary data.
      const [aggregate, perTrade] = await Promise.all([
        api.getStrategyABEval(params),
        api.getStrategyABEvalTrades(params),
      ])
      setData(aggregate)
      setTrades(perTrade)
      setLastFetched(new Date())
    } catch (e) {
      setError(e.message || 'Fetch failed')
    } finally {
      setLoading(false)
    }
  }, [selected, cutoffDate, preWindowDays, postWindowDays, excludePyramids])

  useEffect(() => { fetchEval() }, [fetchEval])

  // Load shadow strategy registry for the source selector. Failure is silent —
  // the dropdown falls back to a single live option below.
  useEffect(() => {
    api.getShadowStrategies({ archived: false }).then(setShadows).catch(() => {})
  }, [])

  // Persist selection so a reload doesn't bounce shadow watchers back to live.
  useEffect(() => {
    try { localStorage.setItem(SELECTED_STORAGE_KEY, JSON.stringify(selected)) } catch {}
  }, [selected])

  // Trigger an immediate snapshot email — same body as the weekly Mon 9 AM
  // UTC cron. Disabled until at least one fetch has succeeded so we don't
  // ask the backend to render a snapshot for a window that hasn't validated.
  // Step 6: shadow source is now plumbed through, so the same button delivers
  // a [Shadow]-prefixed snapshot when a shadow stack is selected.
  const handleSendTestEmail = useCallback(async () => {
    setEmailSending(true)
    try {
      const res = await api.sendABEvalSnapshotEmail({
        strategy: selected.strategy,
        cutoffDate,
        preWindowDays,
        postWindowDays: postWindowDays === '' ? null : Number(postWindowDays),
        excludePyramids,
        source: selected.source,
      })
      if (res?.sent) {
        toast.success(`Snapshot emailed to ${res.recipient} — verdict: ${res.decision}`)
      } else {
        toast.error('Email send failed — check server logs')
      }
    } catch (e) {
      toast.error(e?.message || 'Email send failed')
    } finally {
      setEmailSending(false)
    }
  }, [selected, cutoffDate, preWindowDays, postWindowDays, excludePyramids, toast])

  const isShadow = selected.source === 'shadow'

  // Build the unified options list: one live + N shadow rows. Each option's
  // value is `${source}:${strategy}` so the same strategy name across sources
  // (unlikely but possible) doesn't collide.
  const shadowOptions = useMemo(() => shadows.map(s => ({
    source: 'shadow',
    strategy: s.name,
    label: `Shadow: ${s.name}${s.description ? ` — ${s.description}` : ''}`,
    parentStrategy: s.parent_strategy,
    description: s.description,
    scorerOverrides: s.scorer_overrides || null,
  })), [shadows])

  const allOptions = useMemo(() => [LIVE_OPTION, ...shadowOptions], [shadowOptions])
  const selectedKey = `${selected.source}:${selected.strategy}`

  const handleSelect = useCallback((nextKey) => {
    const opt = allOptions.find(o => `${o.source}:${o.strategy}` === nextKey)
    if (opt) setSelected(opt)
  }, [allOptions])

  // Auto-refresh: schedule next bypass-cache fetch when interval > 0
  useEffect(() => {
    if (refreshTimer.current) clearTimeout(refreshTimer.current)
    if (refreshSeconds > 0) {
      refreshTimer.current = setTimeout(() => fetchEval({ bypassCache: true }), refreshSeconds * 1000)
    }
    return () => { if (refreshTimer.current) clearTimeout(refreshTimer.current) }
  }, [refreshSeconds, lastFetched, fetchEval])

  if (!user?.is_admin) {
    return (
      <div className="p-6">
        <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-4 rounded-lg text-sm">
          Admin access required.
        </div>
      </div>
    )
  }

  const summary = data?.summary
  const pre = data?.pre
  const post = data?.post
  const delta = data?.delta
  const experiment = data?.experiment
  const warnings = data?.warnings || []
  const criteria = summary?.decision_criteria || {}

  const returnDelta = delta?.total_return_pct_delta
  const sharpeDelta = delta?.sharpe_per_trade_delta
  const minReturnDelta = criteria.min_return_delta_pp ?? -5.0
  const minSharpeDelta = criteria.min_sharpe_delta ?? 0.0

  const returnPasses = returnDelta == null ? null : returnDelta >= minReturnDelta
  const sharpePasses = sharpeDelta == null ? null : sharpeDelta >= minSharpeDelta

  return (
    <div className="p-4 md:p-6 space-y-5">
      {/* Title + last-fetched */}
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold text-dark-100">Live A/B Evaluation</h1>
          <p className="text-xs text-dark-400 mt-0.5">
            Pre vs post-cutoff trade summary for live scoring-rule experiments.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-dark-400">
          {lastFetched && <span>Updated {lastFetched.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}</span>}
          <button
            onClick={() => fetchEval({ bypassCache: true })}
            disabled={loading}
            className="px-3 py-1 rounded border border-dark-700 hover:border-dark-600 hover:text-dark-200 disabled:opacity-50"
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
          <button
            onClick={handleSendTestEmail}
            disabled={emailSending || !lastFetched}
            title={!lastFetched ? 'Run an evaluation first' : 'Send the same body the weekly Mon 9 AM UTC cron will deliver'}
            className="px-3 py-1 rounded border border-primary-500/40 text-primary-300 hover:border-primary-500/70 hover:text-primary-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {emailSending ? 'Sending…' : isShadow ? 'Send shadow test email' : 'Send test email'}
          </button>
        </div>
      </div>

      {/* Gate progress — the whole program's accrual at a glance */}
      <GateProgressCard />
      <BuyFunnelCard />

      {/* Program ledger — the program's event history (auto + owner rows) */}
      <ProgramLedgerCard />

      {/* Controls */}
      <Card>
        <SectionLabel>Experiment</SectionLabel>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          <label className="block">
            <span className="text-dark-400">Source</span>
            <select
              value={selectedKey}
              onChange={(e) => handleSelect(e.target.value)}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100 font-data"
            >
              {allOptions.map(opt => (
                <option key={`${opt.source}:${opt.strategy}`} value={`${opt.source}:${opt.strategy}`}>
                  {opt.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="text-dark-400">Cutoff date</span>
            <input
              type="date"
              value={cutoffDate}
              onChange={(e) => setCutoffDate(e.target.value)}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100 font-data"
            />
          </label>
          <label className="block">
            <span className="text-dark-400">Pre window (days)</span>
            <input
              type="number"
              min="1"
              max="365"
              value={preWindowDays}
              onChange={(e) => setPreWindowDays(Number(e.target.value))}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100 font-data"
            />
          </label>
          <label className="block">
            <span className="text-dark-400">Post window (days)</span>
            <input
              type="number"
              min="1"
              placeholder="auto"
              value={postWindowDays}
              onChange={(e) => setPostWindowDays(e.target.value)}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100 font-data"
            />
          </label>
          <label className="block">
            <span className="text-dark-400">Auto-refresh</span>
            <select
              value={refreshSeconds}
              onChange={(e) => setRefreshSeconds(Number(e.target.value))}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100"
            >
              {REFRESH_OPTIONS.map((opt) => (
                <option key={opt.seconds} value={opt.seconds}>{opt.label}</option>
              ))}
            </select>
          </label>
        </div>
        <label className="mt-3 inline-flex items-center gap-2 text-xs text-dark-300 cursor-pointer">
          <input
            type="checkbox"
            checked={excludePyramids}
            onChange={(e) => setExcludePyramids(e.target.checked)}
            className="accent-primary-500"
          />
          Exclude PYRAMID rows
        </label>
        {isShadow && (
          // Shadow stacks share the parent's scoring code until Step 7 wires
          // up scorer_overrides. Surface that explicitly so a parity readout
          // isn't misread as "the candidate diverged".
          <div className="mt-3 text-[11px] text-dark-400 bg-dark-800/40 border border-dark-700/60 rounded px-2 py-1.5">
            scorer_overrides not yet applied — shadow stack scores identically
            to {selected.parentStrategy || 'parent_strategy'} until Step 7.
          </div>
        )}
      </Card>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-300 p-3 rounded text-sm">
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="flex items-center justify-center min-h-[40vh]">
          <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
        </div>
      )}

      {data && (
        <>
          <DecisionBanner summary={summary} post={post} />

          {/* Side-by-side summary */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <WindowSummary
              title="Pre-cutoff (baseline)"
              subtitle={`${experiment?.pre_window?.start} → ${experiment?.pre_window?.end} (${pre?.days}d)`}
              summary={pre}
              accent="cyan"
            />
            <WindowSummary
              title="Post-cutoff (experiment)"
              subtitle={`${experiment?.post_window?.start} → ${experiment?.post_window?.end} (${post?.days}d)`}
              summary={post}
              accent={summary?.decision === 'revert' ? 'red' : summary?.decision === 'keep' ? 'green' : 'amber'}
            />
          </div>

          {/* Cumulative-return chart */}
          <CumulativeReturnChart
            trades={trades}
            cutoffDate={experiment?.post_window?.start}
            decision={summary?.decision}
          />

          {/* Delta panel with thresholds */}
          <Card>
            <CardHeader
              title="Decision criteria"
              subtitle="Each delta is post − pre. ✓ means the criterion is met; ✗ means it's not."
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
              <ThresholdRow
                label="Total return Δ (pp)"
                value={returnDelta}
                threshold={minReturnDelta}
                passes={returnPasses}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}pp`}
              />
              <ThresholdRow
                label="Sharpe / trade Δ"
                value={sharpeDelta}
                threshold={minSharpeDelta}
                passes={sharpePasses}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(4)}`}
              />
              <ThresholdRow
                label="Capital eff. Δ (pp)"
                value={delta?.capital_efficiency_pct_delta}
                threshold={null}
                passes={null}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}pp`}
              />
              <ThresholdRow
                label="Realized DD Δ (pp)"
                value={delta?.realized_max_drawdown_pct_delta}
                threshold={null}
                passes={null}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}pp`}
                goodIsPositive={false}
              />
              <ThresholdRow
                label="Entries / day Δ"
                value={delta?.entry_rate_per_day_delta}
                threshold={null}
                passes={null}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(3)}`}
              />
              <ThresholdRow
                label="Exits / day Δ"
                value={delta?.exit_rate_per_day_delta}
                threshold={null}
                passes={null}
                formatter={(v) => v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(3)}`}
              />
            </div>
          </Card>

          {/* Warnings */}
          {warnings.length > 0 && (
            <Card variant="accent" accent="amber">
              <CardHeader title="Warnings" subtitle="Conditions that complicate interpretation." />
              <ul className="space-y-1.5 text-xs text-amber-200/90">
                {warnings.map((w, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="text-amber-400">•</span>
                    <span>{w}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Cap-Delta Diagnostics — only renders when the selected shadow stack
              has scorer_overrides.disable_excellence_cap === true. */}
          <CapDeltaCard selected={selected} refreshSeconds={refreshSeconds} />

          {/* Post-cutoff trade table */}
          <PostTradesTable
            rows={trades?.post_trades}
            open={tableOpen}
            onToggle={() => setTableOpen((v) => !v)}
            sortKey={sortKey}
            sortDir={sortDir}
            onSort={(k) => {
              if (k === sortKey) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
              else { setSortKey(k); setSortDir('desc') }
            }}
          />

          {/* Experiment metadata */}
          <Card variant="flat" padding="p-3">
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-dark-500 font-data">
              <span>Source: <span className="text-dark-300">{experiment?.source || 'live'}</span></span>
              <span>Strategy: <span className="text-dark-300">{experiment?.strategy}</span></span>
              <span>Users: <span className="text-dark-300">[{experiment?.user_ids?.join(', ')}]</span></span>
              <span>Reference capital: <span className="text-dark-300">${experiment?.starting_value_reference?.toLocaleString()}</span></span>
              <span>Pyramids: <span className="text-dark-300">{experiment?.exclude_pyramids ? 'excluded' : 'included'}</span></span>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
