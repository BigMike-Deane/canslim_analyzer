import { useState, useEffect, useRef, useCallback } from 'react'
import { api, cache } from '../api'
import { useAuth } from '../auth'
import Card, { CardHeader, SectionLabel } from '../components/Card'

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
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastFetched, setLastFetched] = useState(null)
  const refreshTimer = useRef(null)

  const fetchEval = useCallback(async (opts = {}) => {
    setLoading(true)
    setError('')
    try {
      if (opts.bypassCache) cache.invalidate('/api/admin/strategy-ab-eval')
      const result = await api.getStrategyABEval({
        strategy,
        cutoffDate,
        preWindowDays,
        postWindowDays: postWindowDays === '' ? null : Number(postWindowDays),
        excludePyramids,
      })
      setData(result)
      setLastFetched(new Date())
    } catch (e) {
      setError(e.message || 'Fetch failed')
    } finally {
      setLoading(false)
    }
  }, [strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids])

  useEffect(() => { fetchEval() }, [fetchEval])

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
