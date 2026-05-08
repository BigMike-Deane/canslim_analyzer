import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, ResponsiveContainer,
  Tooltip, Legend, ReferenceLine,
} from 'recharts'
import { api, cache } from '../api'
import { useAuth } from '../auth'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { tooltipStyle, tooltipLabelStyle, chartAxis, chartColors } from '../components/chartTheme'
import { useToast } from '../components/Toast'

const REFRESH_OPTIONS = [
  { label: 'Off', seconds: 0 },
  { label: '5m', seconds: 5 * 60 },
  { label: '1h', seconds: 60 * 60 },
  { label: '6h', seconds: 6 * 60 * 60 },
]

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
  const minPostSells = summary?.decision_criteria?.min_post_sells ?? 5
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
    </div>
  )
}

export default function ABEval() {
  const { user } = useAuth()
  const [strategy, setStrategy] = useState('nostate_optimized')
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
        strategy,
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
  }, [strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids])

  useEffect(() => { fetchEval() }, [fetchEval])

  // Trigger an immediate snapshot email — same body as the weekly Mon 9 AM
  // UTC cron. Disabled until at least one fetch has succeeded so we don't
  // ask the backend to render a snapshot for a window that hasn't validated.
  const handleSendTestEmail = useCallback(async () => {
    setEmailSending(true)
    try {
      const res = await api.sendABEvalSnapshotEmail({
        strategy,
        cutoffDate,
        preWindowDays,
        postWindowDays: postWindowDays === '' ? null : Number(postWindowDays),
        excludePyramids,
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
  }, [strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids, toast])

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
            {emailSending ? 'Sending…' : 'Send test email'}
          </button>
        </div>
      </div>

      {/* Controls */}
      <Card>
        <SectionLabel>Experiment</SectionLabel>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          <label className="block">
            <span className="text-dark-400">Strategy</span>
            <input
              type="text"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="mt-1 w-full bg-dark-900 border border-dark-700 rounded px-2 py-1.5 text-dark-100 font-data"
            />
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
