import { useState, useEffect, useMemo, useRef } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, formatDateTime, formatTime } from '../api'
import { XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine, PieChart, Pie, Cell, Area, AreaChart } from 'recharts'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, ActionBadge, TagBadge, MLConfidenceBadge } from '../components/Badge'
import StatGrid, { StatRow } from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import CollapsibleSection from '../components/CollapsibleSection'
import Sparkline from '../components/Sparkline'
import { tooltipStyle, tooltipLabelStyle, chartAxis, chartColors } from '../components/chartTheme'
import { useToast } from '../components/Toast'
import { buildCsv, downloadCsv } from '../csv'
import Modal from '../components/Modal'
import PortfolioDetailView from '../components/PortfolioDetailView'
import EmptyState from '../components/EmptyState'
import PositionHealthChip from '../components/PositionHealthChip'
import DataTable from '../components/DataTable'

// ── Performance Chart ───────────────────────────────────────────────
// `timeRange` is now controlled by the page-level WindowReturnsBar so the
// chart, header summary, and per-position returns all reflect the same window.
function PerformanceChart({ history, startingCash, timeRange }) {
  if (!history || history.length < 2) {
    return (
      <Card variant="glass" className="mb-4 h-48 flex items-center justify-center text-dark-400">
        Not enough data for chart yet
      </Card>
    )
  }

  // Filter history based on selected time range
  // For multi-day views (7d, 30d, all), keep latest snapshot per day for clean chart
  // For 1d view, show all intraday snapshots for granularity
  const filterHistory = (data, range) => {
    let filtered = data.filter(d => d.timestamp || d.date)
    filtered.sort((a, b) => new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date))

    if (range !== 'all') {
      const now = new Date()
      const cutoff = new Date()
      if (range === '1d') cutoff.setHours(now.getHours() - 24)
      else if (range === '7d') cutoff.setDate(now.getDate() - 7)
      else if (range === '30d') cutoff.setDate(now.getDate() - 30)
      filtered = filtered.filter(d => new Date(d.timestamp || d.date) >= cutoff)
    }

    // Trim leading "flat startingCash" snapshots — the portfolio sits at
    // exactly the starting cash on every snapshot between Initialize and the
    // first BUY. Those days are visually dead and push the actual equity
    // curve into a sliver on the right edge of the chart. Drop everything
    // before the first snapshot whose total_value departs from startingCash
    // by more than 50¢ (epsilon covers rounding when a partial buy lands).
    // Server-side WindowReturns are unaffected — those compute against the
    // window's first snapshot directly, so the summary card and per-position
    // returns still anchor at the user-selected window start.
    let trimmed = false
    if (startingCash != null && filtered.length > 1) {
      const FLAT_EPSILON = 0.5
      const firstActivityIdx = filtered.findIndex(d =>
        d.total_value != null && Math.abs(d.total_value - startingCash) > FLAT_EPSILON
      )
      if (firstActivityIdx > 0) {
        filtered = filtered.slice(firstActivityIdx)
        trimmed = true
      }
    }

    // SPY benchmark is server-normalized against snapshots[0].total_value
    // (the pre-inception $25k baseline). After we trim leading flat days,
    // SPY still carries the Day-0 anchor — so if SPY drifted up during the
    // pre-inception period it would head-start the portfolio on the chart
    // for reasons unrelated to strategy. Rebase SPY client-side so its
    // first visible point lands at startingCash, matching the "Start"
    // reference line. Each downstream point is scaled by the same ratio so
    // SPY's *relative* moves from the new anchor are preserved.
    if (trimmed && filtered.length > 0) {
      const anchorSpy = filtered[0].spy_value
      if (anchorSpy != null && anchorSpy > 0 && startingCash != null) {
        const spyMultiplier = startingCash / anchorSpy
        filtered = filtered.map(d => ({
          ...d,
          spy_value: d.spy_value != null ? d.spy_value * spyMultiplier : null,
        }))
      }
    }

    // For longer views with many data points, dedupe to latest per day
    // But only if there are enough unique days to make a readable chart (7+)
    if (range !== '1d' && filtered.length > 60) {
      const uniqueDays = new Set(filtered.map(d => new Date(d.timestamp || d.date).toISOString().slice(0, 10)))
      if (uniqueDays.size >= 7) {
        const byDay = {}
        for (const d of filtered) {
          const dayKey = new Date(d.timestamp || d.date).toISOString().slice(0, 10)
          if (!byDay[dayKey] || new Date(d.timestamp || d.date) > new Date(byDay[dayKey].timestamp || byDay[dayKey].date)) {
            byDay[dayKey] = d
          }
        }
        filtered = Object.values(byDay).sort((a, b) =>
          new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date)
        )
      }
    }

    return filtered
  }

  const filteredHistory = filterHistory(history, timeRange)
  const latestValue = filteredHistory[filteredHistory.length - 1]?.total_value || startingCash
  const firstValue = filteredHistory[0]?.total_value || startingCash
  const isPositive = latestValue >= firstValue

  // Format timestamp for tooltip - uses centralized CST formatter
  const formatTimestamp = (ts) => ts ? formatDateTime(ts) : ''

  const lineColor = isPositive ? '#10b981' : '#ef4444'
  const gradientId = isPositive ? 'perfGradientGreen' : 'perfGradientRed'

  // Compact-currency tick formatter for the Y axis. Portfolio values live
  // in the $20k–$100k range so "$25k" / "$32.5k" reads cleaner than the
  // full "$25,000.00" — and short labels keep the small h-44 chart from
  // sacrificing plot area to axis text.
  const formatYTick = (v) => {
    if (v == null) return ''
    if (Math.abs(v) >= 1000) {
      // 1 decimal under $10k for resolution, 0 decimals above so the ladder
      // shows clean $25k / $30k / $35k steps when the line is up at altitude.
      return `$${(v / 1000).toFixed(v < 10000 ? 1 : 0)}k`
    }
    return `$${v.toFixed(0)}`
  }

  // X-axis tick formatter is range-aware: intraday view shows time only;
  // multi-day views show month+day. Same CST timezone as the tooltip so the
  // axis ticks and the hover label agree.
  const formatXTick = (ts) => {
    if (!ts) return ''
    const d = new Date(ts)
    if (isNaN(d)) return ''
    if (timeRange === '1d') {
      return d.toLocaleTimeString('en-US', {
        timeZone: 'America/Chicago', hour: 'numeric', minute: '2-digit', hour12: true,
      })
    }
    return d.toLocaleDateString('en-US', {
      timeZone: 'America/Chicago', month: 'short', day: 'numeric',
    })
  }

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Performance</span>
        <span className="text-dark-500 text-[10px] font-data">{filteredHistory.length} pts</span>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filteredHistory} margin={{ top: 8, right: 4, bottom: 0, left: 4 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity={0.25} />
                <stop offset="100%" stopColor={lineColor} stopOpacity={0.0} />
              </linearGradient>
            </defs>
            {/* SPY benchmark overlay — normalized to starting value of the
                window so the two lines compare apples-to-apples. Sourced
                from the backend's MarketSnapshot table, not historical_data. */}
            <Area
              type="monotone"
              dataKey="spy_value"
              stroke={chartColors.spy}
              strokeWidth={1.25}
              strokeDasharray="4 3"
              fill="none"
              dot={false}
              activeDot={{ r: 3, fill: chartColors.spy }}
              isAnimationActive={false}
              connectNulls
            />
            <Area
              type="monotone"
              dataKey="total_value"
              stroke={lineColor}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={filteredHistory.length <= 50}
              activeDot={{ r: 4, fill: lineColor }}
            />
            <ReferenceLine
              y={startingCash}
              stroke={chartAxis.reference}
              strokeDasharray="3 3"
              label={{
                value: `Start ${formatYTick(startingCash)}`,
                position: 'insideTopLeft',
                fill: chartAxis.tick,
                fontSize: 10,
              }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
              formatter={(value, name) => {
                if (value == null) return [null, null]
                const label = name === 'spy_value' ? 'SPY' : 'Portfolio'
                return [formatCurrency(value), label]
              }}
              labelFormatter={(_, payload) => {
                if (payload && payload[0]) {
                  return formatTimestamp(payload[0].payload.timestamp || payload[0].payload.date)
                }
                return ''
              }}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXTick}
              tick={{ fill: chartAxis.tick, fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              minTickGap={36}
            />
            <YAxis
              orientation="right"
              domain={['dataMin - 500', 'dataMax + 500']}
              tickFormatter={formatYTick}
              tick={{ fill: chartAxis.tick, fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={42}
              tickCount={4}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-1 text-[10px] text-dark-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ backgroundColor: lineColor }} />
          Portfolio
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 border-t border-dashed border-dark-400" />
          SPY (normalized)
        </span>
      </div>
    </Card>
  )
}

// ── Summary Card ────────────────────────────────────────────────────
// Window-aware: when `windowReturns` is supplied the return number reflects
// the selected window (1D/7D/30D/All). Falls back to lifetime gain when the
// new endpoint hasn't loaded yet so existing UX is preserved on first paint.
const WINDOW_LABELS = {
  '1d': '1D',
  '7d': '7D',
  '30d': '30D',
  'all': 'Since Inception',
}

function SummaryCard({ summary, config, windowReturns, timeRange, setTimeRange, loading }) {
  if (!summary) return null

  const winRet = windowReturns?.portfolio
  // Prefer windowed number when available; fall back to lifetime fields.
  const returnDollar = winRet?.return ?? summary.total_return
  const returnPct = winRet?.return_pct ?? summary.total_return_pct
  const isPositive = (returnPct ?? 0) >= 0
  const windowLabel = WINDOW_LABELS[timeRange] || WINDOW_LABELS.all

  return (
    <Card variant="glass" className="mb-4">
      {/* Compact pill row sits inside the card header so the controls and
          the windowed return number are visually adjacent — no chance of
          scrolling past it. */}
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">
          AI Portfolio Value
        </span>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="text-[10px] text-dark-500 font-data" aria-live="polite">…</span>
          )}
          <div className="flex bg-dark-850 rounded-lg p-0.5">
            {WINDOW_PILLS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setTimeRange?.(value)}
                className={`px-2 py-0.5 text-[11px] rounded transition-colors ${
                  timeRange === value
                    ? 'bg-primary-500 text-white'
                    : 'text-dark-400 hover:text-white'
                }`}
                aria-pressed={timeRange === value}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="text-3xl font-bold font-data mt-1 mb-1">
        {formatCurrency(summary.total_value)}
      </div>
      <div className={`text-sm flex items-center gap-1.5 font-data ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
        <span>{isPositive ? '+' : ''}{formatCurrency(Math.abs(returnDollar ?? 0))}</span>
        <span className="text-dark-500">({formatPercent(returnPct, true)})</span>
        <span className="text-[10px] uppercase tracking-wider text-dark-500 ml-1">{windowLabel}</span>
      </div>

      <div className="border-t border-dark-700/50 mt-4 pt-3">
        <StatGrid
          columns={3}
          stats={[
            { label: 'Cash', value: formatCurrency(summary.cash) },
            { label: 'Invested', value: formatCurrency(summary.positions_value) },
            { label: 'Positions', value: `${summary.positions_count} / ${config?.max_positions || 15}` },
          ]}
        />
      </div>
    </Card>
  )
}

// ── Edge Scorecard ──────────────────────────────────────────────────
// The one card that answers "is the AI actually generating alpha, or just
// riding beta?" — return vs SPY, beta-adjusted Jensen alpha, Sharpe, max
// drawdown, win rate. All derived on-read server-side from the equity curve +
// MarketSnapshot SPY series (see backend/edge_metrics.py).
// `dim` marks a metric as statistically provisional (small sample). It is
// rendered at reduced opacity with an explanatory tooltip rather than hidden —
// the number still exists, we just signal "don't trust this yet."
function EdgeMetric({ label, value, sub, tone = 'neutral', title, dim = false }) {
  const toneClass =
    tone === 'pos' ? 'text-emerald-400' : tone === 'neg' ? 'text-red-400' : 'text-white'
  return (
    <div
      className={`px-1 ${dim ? 'opacity-40' : ''}`}
      title={dim ? 'Provisional — needs ~20 trading days of snapshots to be reliable' : title}
    >
      <div className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">{label}</div>
      <div className={`text-lg font-bold font-data mt-0.5 ${toneClass}`}>{value}</div>
      {sub && <div className="text-[10px] text-dark-500 mt-0.5">{dim ? 'building…' : sub}</div>}
    </div>
  )
}

// Edge Scorecard window selector. `all` uses a large day count so the
// scorecard covers full history even past a year; the page's initial fetch
// uses the same EDGE_ALL_DAYS so the default `all` view needs no refetch.
const EDGE_ALL_DAYS = 3650
const EDGE_WINDOWS = [
  { value: '7d', label: '7D', days: 7 },
  { value: '30d', label: '30D', days: 30 },
  { value: '90d', label: '90D', days: 90 },
  { value: 'all', label: 'All', days: EDGE_ALL_DAYS },
]
const EDGE_WINDOW_PHRASE = {
  '7d': 'over the last 7 days',
  '30d': 'over the last 30 days',
  '90d': 'over the last 90 days',
  all: 'since inception',
}

// ── Statistical-significance framing for the edge ───────────────────
// Maps backend `edge_verdict` strings (backend/edge_metrics.py:_edge_verdict)
// to a label, a tone color, and a plain-English caption for a non-statistician
// owner. The point of this block is to stop the headline return-vs-SPY number
// from being read as "proven" when it isn't yet statistically distinguishable
// from luck.
const EDGE_VERDICT_CFG = {
  significant_edge: {
    label: 'Significant edge',
    cls: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25',
    caption: 'Alpha is statistically confirmed (p < 0.05). The edge vs SPY is real at the 95% level.',
  },
  promising_insufficient_sample: {
    label: 'Promising · insufficient sample',
    cls: 'text-amber-300 bg-amber-500/10 border-amber-500/25',
    caption: "Positive alpha, but not yet statistically significant — could still be luck. Don't size up on it yet; it needs more closed trades to confirm.",
  },
  no_measurable_edge: {
    label: 'No measurable edge',
    cls: 'text-dark-300 bg-dark-700/40 border-dark-600/40',
    caption: 'No positive alpha vs SPY detected over this sample — returns are explained by market exposure.',
  },
  significant_negative: {
    label: 'Significant underperformance',
    cls: 'text-red-300 bg-red-500/10 border-red-500/25',
    caption: 'Statistically significant negative alpha (p < 0.05) — the strategy is trailing a passive SPY position after adjusting for risk.',
  },
  inconclusive_small_sample: {
    label: 'Inconclusive · small sample',
    cls: 'text-dark-300 bg-dark-700/40 border-dark-600/40',
    caption: 'Too few trading days / closed trades to assess statistical significance yet. Keep accumulating a track record.',
  },
}

// Renders the significance story (verdict chip + annualized-alpha CI + p-value +
// Wilson win-rate band) beneath the metrics grid. Every field is guarded — the
// whole block returns null if the backend didn't emit a verdict, and the
// alpha/win-rate rows each render only when their data is present.
function EdgeSignificance({ data }) {
  const verdict = data?.edge_verdict
  if (!verdict) return null
  const cfg = EDGE_VERDICT_CFG[verdict] || EDGE_VERDICT_CFG.inconclusive_small_sample

  const sig = data.alpha_significance // may be null until ~20 daily obs
  const ci = data.win_rate_ci_95 // [low_pct, high_pct] or null

  const fmtPct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`)
  // Alpha CI straddling zero is the visual tell that the edge isn't proven.
  const spansZero =
    sig && sig.alpha_annualized_ci_low_pct != null && sig.alpha_annualized_ci_high_pct != null &&
    sig.alpha_annualized_ci_low_pct < 0 && sig.alpha_annualized_ci_high_pct > 0

  return (
    <div className="border-t border-dark-700/50 mt-4 pt-3">
      <div className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-md border px-2 py-1 ${cfg.cls}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" aria-hidden="true" />
        {cfg.label}
      </div>
      <p className="text-[11px] leading-snug text-dark-400 mt-2">{cfg.caption}</p>

      {sig && (
        <div className="mt-2.5 grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
          <div className="text-dark-400">
            Alpha (annualized){' '}
            <span className={sig.alpha_annualized_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}>
              {fmtPct(sig.alpha_annualized_pct)}
            </span>
          </div>
          <div className="text-dark-400 text-right">
            p = {sig.p_value == null ? '—' : sig.p_value.toFixed(3)}{' '}
            <span className={sig.significant_95 ? 'text-emerald-400' : 'text-dark-500'}>
              {sig.significant_95 ? '(significant)' : '(not yet significant)'}
            </span>
          </div>
          <div className="col-span-2 text-dark-500">
            95% CI {fmtPct(sig.alpha_annualized_ci_low_pct)} … {fmtPct(sig.alpha_annualized_ci_high_pct)}
            {spansZero && <span className="text-amber-300/90"> · spans 0% — not yet proven</span>}
          </div>
        </div>
      )}

      {Array.isArray(ci) && ci.length === 2 && data.win_rate_pct != null && (
        <div className="mt-1.5 text-[11px] text-dark-500">
          Win rate {data.win_rate_pct}% · 95% likely between {ci[0]?.toFixed(0)}%–{ci[1]?.toFixed(0)}%
        </div>
      )}

      <EdgePower power={data.power} significant={sig?.significant_95} />
    </div>
  )
}

// Phase 4 power analysis: when the alpha isn't yet statistically proven, how
// much MORE data would it take to confirm it (if the current effect size
// holds)? Reads `data.power` (backend/edge_metrics.py:_power_analysis) — a
// z-based one-sample sample-size estimate on the daily alpha residuals.
// Renders nothing once the alpha is already significant (the question is moot)
// or when the effect is undefined / too small to estimate.
function EdgePower({ power, significant }) {
  if (!power || significant) return null
  if (power.already_sufficient) {
    return (
      <div className="mt-2 text-[11px] text-emerald-400/90">
        Sample is now large enough to detect this effect at 95% confidence (80% power) — accumulate the alpha trend to confirm direction.
      </div>
    )
  }
  if (power.additional_days_needed == null) return null
  return (
    <div className="mt-2 text-[11px] text-dark-500">
      <span className="text-dark-400">Power:</span> at the current effect size,
      {' '}≈{power.additional_days_needed} more trading day{power.additional_days_needed === 1 ? '' : 's'}
      {power.est_additional_months != null && ` (~${power.est_additional_months} mo)`}
      {' '}of data would be needed to prove this alpha at 95% confidence (80% power).
      <span className="block text-dark-600 mt-0.5">
        {power.current_days} of ≈{power.required_days} days collected · assumes the edge persists.
      </span>
    </div>
  )
}

function EdgeScorecard({ edge }) {
  // `edge` is the all-window scorecard fetched by the page. Narrower windows
  // are fetched on demand; keep `data` separate so `all` reuses the prop
  // (no refetch) while other windows hit the endpoint with their day count.
  const [range, setRange] = useState('all')
  const [data, setData] = useState(edge)
  const [loading, setLoading] = useState(false)
  // Monotonic request id: rapid window clicks fire overlapping fetches, and
  // a slower earlier response must not clobber the window selected last.
  const fetchSeq = useRef(0)

  // Re-sync to the parent's all-window edge when it loads/changes, but only
  // while viewing `all` — don't clobber a narrower window the user selected.
  useEffect(() => {
    if (range === 'all') setData(edge)
  }, [edge, range])

  async function selectRange(next) {
    if (next === range) return
    setRange(next)
    const seq = ++fetchSeq.current
    if (next === 'all') { setData(edge); return }
    setLoading(true)
    try {
      const win = EDGE_WINDOWS.find(w => w.value === next)
      const result = await api.getAIPortfolioEdge(win.days)
      if (seq === fetchSeq.current) setData(result)
    } catch {
      // Keep the prior window's numbers rather than blanking the card.
    } finally {
      if (seq === fetchSeq.current) setLoading(false)
    }
  }

  if (!data) return null

  const pct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`)
  const num = (v, d = 2) => (v == null ? '—' : v.toFixed(d))
  const tone = (v) => (v == null ? 'neutral' : v >= 0 ? 'pos' : 'neg')

  const insufficient = data.status !== 'ok'
  const excess = data.excess_return_pct
  const phrase = EDGE_WINDOW_PHRASE[range] || 'over the window'
  // `low_sample` trips for any series under ~20 trading days — which includes an
  // established account viewing the 7D/30D tab. Only treat it as a *young
  // account* cold-start on the All window, where a thin series means the
  // portfolio itself is new (not just the selected window).
  const youngAccount = range === 'all' && data.low_sample

  let verdict = null
  if (!insufficient && excess != null) {
    verdict =
      excess >= 0
        ? `Beating SPY by ${excess.toFixed(1)}% ${phrase}`
        : `Trailing SPY by ${Math.abs(excess).toFixed(1)}% ${phrase}`
  }

  const rangeSelector = (
    <div className="flex bg-dark-850 rounded-lg p-0.5" role="group" aria-label="Edge window">
      {EDGE_WINDOWS.map(({ value, label }) => (
        <button
          key={value}
          onClick={() => selectRange(value)}
          className={`px-2 py-0.5 text-[11px] rounded transition-colors ${
            range === value ? 'bg-primary-500 text-white' : 'text-dark-400 hover:text-white'
          }`}
          aria-pressed={range === value}
        >
          {label}
        </button>
      ))}
    </div>
  )

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Edge Scorecard" subtitle="Risk-adjusted performance vs SPY" action={rangeSelector} />
      {insufficient ? (
        <EmptyState
          bare
          compact
          message={(data.trading_days || 0) <= 1 ? 'Building your track record' : 'Not enough history yet'}
          hint={`${data.trading_days || 0} trading day${data.trading_days === 1 ? '' : 's'} so far — edge metrics vs SPY unlock after 2.`}
        />
      ) : (
        <>
          {verdict && (
            <div
              className={`text-sm font-semibold mb-1 ${
                excess >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {verdict}
            </div>
          )}
          {/* Verdict clock — promoted to headline (owner ask, Jul-22): the
              single question is "do we beat SPY, and when will we KNOW?".
              Yellow = counting; green = statistically confirmed. */}
          {range === 'all' && data.alpha_significance?.significant_95 && (
            <div className="text-[11px] font-medium text-emerald-300 bg-emerald-500/10 border border-emerald-500/25 rounded-md px-2.5 py-1.5 mb-3">
              ✓ Alpha statistically confirmed at 95% confidence.
            </div>
          )}
          {range === 'all' && !data.alpha_significance?.significant_95 && data.power?.additional_days_needed != null && (
            <div
              className="text-[11px] font-medium text-amber-300 bg-amber-500/10 border border-amber-500/25 rounded-md px-2.5 py-1.5 mb-1.5"
              title="At the current alpha and volatility, how many more trading days of live data are needed before the edge over SPY is statistically provable (95% confidence, 80% power). Assumes the edge persists — this is a clock, not a promise. This is the UNCONDITIONAL question — see the trend-day clock below for the regime-aware one."
            >
              ⏳ Verdict clock (all days): ≈{data.power.additional_days_needed} trading days
              {data.power.est_additional_months != null && ` (~${data.power.est_additional_months} mo)`} until the
              edge vs SPY is statistically provable. {data.power.current_days}/{data.power.required_days} collected.
            </div>
          )}
          {/* Regime-conditional edge: the strategy is regime-dependent (all live
              outperformance came from trend days), so the trend-day clock is
              the decision-relevant one — it runs ~10x faster than the blended
              clock above because conditioning un-dilutes the effect. */}
          {range === 'all' && data.regime_edge?.trend && (
            <div
              className={`text-[11px] font-medium rounded-md px-2.5 py-1.5 border ${
                data.regime_edge.chop ? 'mb-1.5' : 'mb-3'
              } ${
                data.regime_edge.trend.significant_95
                  ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25'
                  : 'text-sky-300 bg-sky-500/10 border-sky-500/25'
              }`}
              title={`Excess return vs SPY computed only on days SPY sat more than ${data.regime_edge.threshold_pct}% above its 50MA (trend regime). Same persistence caveat as the clock above.`}
            >
              {data.regime_edge.trend.significant_95 ? '✓' : '📈'} Trend-day edge: {data.regime_edge.trend.mean_daily_excess_bps > 0 ? '+' : ''}
              {data.regime_edge.trend.mean_daily_excess_bps} bps/day over SPY
              {' '}(p={data.regime_edge.trend.p_value}, {data.regime_edge.trend.n_days} days)
              {!data.regime_edge.trend.significant_95 && data.regime_edge.trend.additional_days_needed != null &&
                ` — provable in ≈${data.regime_edge.trend.additional_days_needed} more trend days`}
              {data.regime_edge.trend.significant_95 && ' — statistically confirmed'}
            </div>
          )}
          {/* Chop-day edge — first-class line (owner ask, Jul-23): the mirror
              bucket of the trend line above. All live underperformance
              concentrates on chop days, so it renders in the down/red tone
              when negative instead of hiding in the trend line's tooltip. */}
          {range === 'all' && data.regime_edge?.chop && (
            <div
              className={`text-[11px] font-medium rounded-md px-2.5 py-1.5 mb-3 border ${
                data.regime_edge.chop.mean_daily_excess_bps < 0
                  ? 'text-red-300 bg-red-500/10 border-red-500/25'
                  : 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25'
              }`}
              title={`Excess return vs SPY computed only on days SPY sat within ${data.regime_edge.threshold_pct}% of its 50MA or below it (chop regime). Same persistence caveat as the trend line above.`}
            >
              {data.regime_edge.chop.mean_daily_excess_bps < 0 ? '⚠' : '✓'} Chop-day edge: {data.regime_edge.chop.mean_daily_excess_bps > 0 ? '+' : ''}
              {data.regime_edge.chop.mean_daily_excess_bps} bps/day over SPY
              {' '}(p={data.regime_edge.chop.p_value}, {data.regime_edge.chop.n_days} days)
            </div>
          )}
          {/* Small-sample caveat: return-vs-SPY is valid on any window, but the
              regression/annualized stats (β, Sharpe, Alpha) are noise until ~20
              daily observations. Flag them rather than hide them. */}
          {youngAccount && (
            <div className="text-[11px] text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded-md px-2.5 py-1.5 mb-3">
              Early data — only {data.trading_days} trading day{data.trading_days === 1 ? '' : 's'}. Beta, Sharpe &amp; Alpha
              are provisional until ~20 days; your return vs SPY above is already meaningful.
            </div>
          )}
          <div className={`grid grid-cols-3 gap-y-4 gap-x-2 transition-opacity ${loading ? 'opacity-50' : ''}`} aria-busy={loading}>
            <EdgeMetric
              label="You"
              value={pct(data.total_return_pct)}
              tone={tone(data.total_return_pct)}
              sub="total return"
              title="Portfolio total return over the selected window (window-scale, not annualized)"
            />
            <EdgeMetric
              label="SPY"
              value={pct(data.spy_return_pct)}
              tone={tone(data.spy_return_pct)}
              sub="same window"
              title="SPY benchmark return over the same window"
            />
            <EdgeMetric
              label="Alpha"
              value={pct(data.alpha_pct)}
              tone={tone(data.alpha_pct)}
              sub="β-adjusted"
              dim={youngAccount}
              title="Jensen's alpha over the window: return minus beta×SPY return. Positive = skill beyond market exposure."
            />
            <EdgeMetric
              label="Beta"
              value={num(data.beta)}
              sub="vs SPY"
              dim={youngAccount}
              title="Sensitivity to SPY moves. >1 = swings more than the market."
            />
            <EdgeMetric
              label="Sharpe"
              value={num(data.sharpe)}
              tone={tone(data.sharpe)}
              sub="annualized"
              dim={youngAccount}
              title="Return per unit of risk (annualized, risk-free = 0). Higher is better; >1 is good."
            />
            <EdgeMetric
              label="Max DD"
              value={pct(data.max_drawdown_pct)}
              sub={
                data.spy_max_drawdown_pct != null
                  ? `SPY ${data.spy_max_drawdown_pct.toFixed(1)}%`
                  : 'peak → trough'
              }
              title="Largest peak-to-trough decline. SPY's shown for comparison — smaller (less negative) is better."
            />
          </div>
          <EdgeSignificance data={data} />
          <div className="border-t border-dark-700/50 mt-4 pt-2 flex items-center justify-between text-[10px] text-dark-500">
            <span>
              {data.win_rate_pct != null
                ? `Win rate ${data.win_rate_pct}% (${data.closed_trades} closed)`
                : `${data.closed_trades} closed trades`}
            </span>
            <span>
              {data.trading_days} trading days{data.low_sample ? ' · small sample' : ''}
            </span>
          </div>
        </>
      )}
    </Card>
  )
}

// ── Phase 3: per-position attribution ───────────────────────────────
// Decomposes the realized edge across closed positions to answer the
// question the headline return can't: is the gain SKILL (beat SPY over each
// hold window) or just MARKET exposure? Reads the lifetime `edge.attribution`
// block (backend/edge_metrics.py:attribute_returns) — NOT window-scoped, so it
// uses the page-level `edge` prop, never the windowed scorecard `data`.
function EdgeAttribution({ edge }) {
  const a = edge?.attribution
  if (!a || a.status !== 'ok' || !a.closed_positions) return null

  const usd = (v) => (v == null ? '—' : formatCurrency(v))
  const pct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`)
  const signCls = (v) => (v == null ? 'text-dark-400' : v >= 0 ? 'text-emerald-400' : 'text-red-400')

  // The headline insight: realized P&L vs the slice of it that actually beat
  // SPY. Stated as the live proportion rather than a thresholded "mostly
  // market" verdict — a fixed cutoff let one big exit flip the banner's
  // absolutist copy from misleading to absent with no state in between.
  const realized = a.total_realized_gain
  const excess = a.total_excess_gain
  const hasSplit = realized != null && excess != null && realized > 0
  const skillPct = hasSplit ? Math.round((excess / realized) * 100) : null

  const conc = a.concentration || {}

  const PositionRows = ({ rows }) => (
    <div className="space-y-1">
      {rows.map((r) => (
        <div key={r.ticker + r.holding_days} className="grid grid-cols-12 gap-1 items-center text-[11px] py-0.5">
          <span className="col-span-3 font-semibold text-dark-200 truncate">{r.ticker}</span>
          <span className={`col-span-2 text-right ${signCls(r.position_return_pct)}`}>{pct(r.position_return_pct)}</span>
          <span className="col-span-3 text-right text-dark-500">SPY {pct(r.spy_return_pct)}</span>
          <span className={`col-span-2 text-right ${signCls(r.excess_gain)}`} title="Dollars earned beyond what the same capital would have made in SPY over this hold window">{usd(r.excess_gain)}</span>
          <span className="col-span-2 text-right text-dark-400">{r.holding_days != null ? `${r.holding_days}d` : '—'}</span>
        </div>
      ))}
    </div>
  )

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Edge Attribution" subtitle="Which closed positions drive the edge — skill or market?" />

      <div className="grid grid-cols-2 gap-2 mb-3">
        <EdgeMetric
          label="Realized P&L"
          value={usd(realized)}
          tone={realized == null ? 'neutral' : realized >= 0 ? 'pos' : 'neg'}
          sub={`${a.closed_positions} closed`}
          title="Total realized dollar gain/loss across all closed positions."
        />
        <EdgeMetric
          label="Excess vs SPY"
          value={usd(excess)}
          tone={excess == null ? 'neutral' : excess >= 0 ? 'pos' : 'neg'}
          sub="beyond market"
          title="Realized P&L minus what the same capital would have earned sitting in SPY over each position's own hold window. This is the part attributable to selection/timing skill, not market exposure."
        />
      </div>

      {hasSplit && (
        <div
          className={`text-[11px] rounded-md px-2.5 py-1.5 mb-3 border ${
            excess < 0
              ? 'text-amber-300/90 bg-amber-500/10 border-amber-500/20'
              : 'text-dark-400 bg-dark-850/60 border-dark-700/50'
          }`}
        >
          {excess < 0
            ? `The realized gain is all market exposure — the same capital in SPY over the same hold windows would have earned ${usd(Math.abs(excess))} more.`
            : excess >= realized
              ? `All ${usd(realized)} of realized P&L beat SPY — SPY was flat-to-down over these hold windows, so none of the gain is market exposure.`
              : `${usd(excess)} of the ${usd(realized)} realized (${skillPct}%) beat SPY over the same hold windows; the remaining ${usd(realized - excess)} is market exposure.`}
        </div>
      )}

      <div className="flex items-center justify-between text-[11px] text-dark-400 mb-2">
        <span>{a.winners}W / {a.losers}L · winners {usd(a.gross_winners_gain)}, losers {usd(a.gross_losers_loss)}</span>
      </div>

      {(conc.top1_share_of_gains_pct != null || conc.names_for_half_of_gains != null) && (
        <div className="text-[11px] text-dark-500 border-t border-dark-700/50 pt-2 mb-1">
          Concentration:{' '}
          {conc.top1_share_of_gains_pct != null && <>top name = {conc.top1_share_of_gains_pct}% of gains</>}
          {conc.top3_share_of_gains_pct != null && <> · top 3 = {conc.top3_share_of_gains_pct}%</>}
          {conc.names_for_half_of_gains != null && <> · {conc.names_for_half_of_gains} name{conc.names_for_half_of_gains === 1 ? '' : 's'} = half the gains</>}
        </div>
      )}

      <div className="border-t border-dark-700/50 mt-2 pt-2">
        <CollapsibleSection title="Top contributors & detractors" defaultOpen={false}>
          <div className="grid grid-cols-12 gap-1 text-[10px] uppercase tracking-wide text-dark-600 mb-1">
            <span className="col-span-3">Ticker</span>
            <span className="col-span-2 text-right">Return</span>
            <span className="col-span-3 text-right">vs SPY</span>
            <span className="col-span-2 text-right">Excess $</span>
            <span className="col-span-2 text-right">Held</span>
          </div>
          {a.top_contributors?.length > 0 && <PositionRows rows={a.top_contributors} />}
          {a.top_detractors?.length > 0 && (
            <>
              <div className="text-[10px] uppercase tracking-wide text-dark-600 mt-2 mb-1">Detractors</div>
              <PositionRows rows={a.top_detractors} />
            </>
          )}
        </CollapsibleSection>
      </div>
    </Card>
  )
}

// ── Phase 2: live-vs-backtest exit reconciliation ───────────────────
// Surfaces trader↔backtester drift the headline numbers hide: for each exit
// reason, does live trading behave like the reference backtest? A flagged
// reason means the engine takes (or skips) an exit the backtest doesn't, or
// holds/realizes it very differently — the mechanism behind backtests
// overstating live performance. Reads /api/ai-portfolio/edge/reconciliation
// (backend/edge_reconciliation.py). Read-only; ai_trader.py untouched.
const RECON_FLAG_LABEL = {
  hold_days: 'hold days',
  win_rate: 'win rate',
  realized_pct: 'avg return',
  presence: 'one side only',
}

// Live exits before this date carry the pre-parity-fix pathology (short
// pathological holds, missing stop clamp — fixed by the Jun-18 exit-parity
// queue, 6a93c7c). The card defaults to post-fix exits so old debt doesn't
// read as current drift; "All time" opts back in.
const PARITY_FIX_DATE = '2026-06-18'

function ExitReconciliation({ reconciliation: rec, allTime, onWindowChange }) {
  if (!rec) return null

  if (rec.status === 'no_reference_backtest') {
    return (
      <Card variant="glass" className="mb-4">
        <CardHeader title="Live vs Backtest Exits" subtitle="Trader↔backtester exit-behavior drift" />
        <EmptyState
          bare
          compact
          message="No reference backtest yet"
          hint="Run a nostate_optimized backtest to reconcile live exits against the model."
        />
      </Card>
    )
  }
  if (rec.status !== 'ok' || !rec.comparison) return null

  const cmp = rec.comparison
  const aligned = cmp.verdict === 'aligned'
  // Show reasons with enough data to compare, flagged ones first.
  const rows = (cmp.reasons || [])
    .filter((r) => (r.live?.count || 0) + (r.backtest?.count || 0) > 0)
    .sort((a, b) => (b.divergence_flags?.length || 0) - (a.divergence_flags?.length || 0))

  const pct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`)
  const side = (s) =>
    s
      ? `${s.count}× · ${s.avg_hold_days != null ? `${s.avg_hold_days}d` : '—'} · WR ${s.win_rate != null ? `${s.win_rate}%` : '—'} · ${pct(s.avg_realized_pct)}`
      : '—'

  const ref = rec.reference_backtest || {}

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Live vs Backtest Exits" subtitle="Do live exits behave like the model? Divergence = drift." />

      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span
          className={`inline-flex items-center gap-1.5 text-[11px] font-semibold rounded-md border px-2 py-1 ${
            aligned
              ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25'
              : 'text-amber-300 bg-amber-500/10 border-amber-500/25'
          }`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80" aria-hidden="true" />
          {aligned ? 'Exits aligned' : `${cmp.diverged_reasons?.length || 0} reason${cmp.diverged_reasons?.length === 1 ? '' : 's'} diverged`}
        </span>
        <span className="text-[10px] text-dark-500">
          {rec.live_exit_count} live vs {rec.backtest_exit_count} backtest exits
        </span>
        {onWindowChange && (
          <div className="flex gap-1 ml-auto" role="group" aria-label="Live exit window">
            {[
              { v: false, label: 'Post-fix' },
              { v: true, label: 'All time' },
            ].map((o) => (
              <button
                key={o.label}
                onClick={() => onWindowChange(o.v)}
                aria-pressed={allTime === o.v}
                className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                  allTime === o.v
                    ? 'text-primary-300 bg-primary-500/10 border-primary-500/30'
                    : 'text-dark-400 border-dark-700/50 hover:text-dark-200'
                }`}
              >
                {o.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="space-y-2">
        {rows.map((r) => {
          const flagged = r.divergence_flags?.length > 0
          return (
            <div
              key={r.reason}
              className={`rounded-md border px-2.5 py-1.5 ${
                flagged ? 'border-amber-500/25 bg-amber-500/[0.04]' : 'border-dark-700/50'
              }`}
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-[11px] font-semibold text-dark-200">{r.reason}</span>
                <div className="flex flex-wrap gap-1 justify-end">
                  {r.divergence_flags?.map((f) => (
                    <span key={f} className="text-[9px] uppercase tracking-wide text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded px-1 py-0.5">
                      {RECON_FLAG_LABEL[f] || f}
                    </span>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-x-3 text-[10px]">
                <span className="text-dark-500"><span className="text-dark-400">Live</span> {side(r.live)}</span>
                <span className="text-dark-500 text-right"><span className="text-dark-400">BT</span> {side(r.backtest)}</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="border-t border-dark-700/50 mt-3 pt-2 text-[10px] text-dark-500">
        <div>
          Live exits {allTime ? 'since inception (includes pre-parity-fix pathology before Jun 18)' : `since ${PARITY_FIX_DATE} (post parity-fix)`}
          {' · '}vs {ref.name || `backtest #${ref.id}`}
          {ref.strategy && ` (${ref.strategy})`}
          {ref.start_date && ref.end_date && ` · ${ref.start_date} → ${ref.end_date}`}
        </div>
        <div className="mt-0.5">
          fingerprint = count · avg hold · win rate · avg realized
          {' · '}young portfolio vs multi-year backtest: hold-day and one-side-only gaps are partly structural (long-hold exits like TAKE PROFIT can't have occurred live yet)
        </div>
      </div>
    </Card>
  )
}

// ── New-account framing ─────────────────────────────────────────────
// Shown only while the portfolio is too young for edge metrics
// (edge.status !== 'ok', i.e. < 2 trading days of snapshots). Frames the
// otherwise-sparse overview as expected rather than broken, so a brand-new
// user understands the empty Edge card and flat chart are a cold start.
function TrackRecordBanner({ edge }) {
  if (!edge || edge.status === 'ok') return null
  const days = edge.trading_days || 0
  return (
    <Card variant="glass" className="mb-4 border border-primary-500/30">
      <div className="flex items-start gap-2.5">
        <span className="text-lg leading-none mt-0.5" aria-hidden="true">🌱</span>
        <div>
          <div className="text-sm font-semibold text-white">Building your track record</div>
          <div className="text-xs text-dark-400 mt-1 leading-relaxed">
            Your AI portfolio is {days <= 1 ? 'just getting started' : `${days} trading days in`}. Risk-adjusted
            metrics (alpha, beta, Sharpe) and the longer return windows fill in as daily snapshots accumulate —
            check back over the next couple of weeks to see your edge vs SPY take shape.
          </div>
        </div>
      </div>
    </Card>
  )
}

// Pill values shared by SummaryCard's inline selector.
// Drives SummaryCard return number + PositionsList per-row return +
// PerformanceChart filter window.
const WINDOW_PILLS = [
  { value: '1d', label: '1D' },
  { value: '7d', label: '7D' },
  { value: '30d', label: '30D' },
  { value: 'all', label: 'All' },
]

// Sort options for the PositionsList header dropdown. Each key maps to a
// null-safe accessor; direction baked in (e.g. value/gain%/score/days-held
// read high→low, ticker/sector read A→Z). Default is value descending.
const POSITION_SORTS = [
  { value: 'value',  label: 'Value (high → low)' },
  { value: 'gain',   label: 'Gain % (high → low)' },
  { value: 'score',  label: 'Score (high → low)' },
  { value: 'days',   label: 'Days held (most → least)' },
  { value: 'ticker', label: 'Ticker (A → Z)' },
  { value: 'sector', label: 'Sector (A → Z)' },
]

// ── Positions List ──────────────────────────────────────────────────
// Window-aware: when `windowReturns` is supplied, the per-position return %
// reflects the selected window. Otherwise falls back to lifetime gain_loss_pct.
// Mirrors SummaryCard's inline pill row so the user can flip windows while
// scanning the list without scrolling back to the summary at the top.
function PositionsList({ positions, windowReturns, timeRange, setTimeRange, loading }) {
  const [selectedPosition, setSelectedPosition] = useState(null)
  const [sortKey, setSortKey] = useState('value')

  // Effective per-position score depends on stock type (mirrors the render
  // below: growth stocks show current_growth_score, others current_score).
  const scoreOf = (p) => (p.is_growth_stock ? p.current_growth_score : p.current_score) ?? 0
  // Days held — same Date.now()/86_400_000 math the PositionDetailModal uses.
  const daysHeldOf = (p) => p.purchase_date
    ? Math.max(0, Math.floor((Date.now() - new Date(p.purchase_date).getTime()) / 86400000))
    : 0

  // Per-position window returns indexed by ticker (O(1) lookup), shared by the
  // sort comparator and the row render so both agree on the number shown.
  const windowByTicker = useMemo(() => {
    const idx = {}
    for (const p of windowReturns?.positions || []) idx[p.ticker] = p
    return idx
  }, [windowReturns])

  // Effective gain % — MUST mirror the row render below (winRow.return_pct,
  // falling back to lifetime gain_loss_pct). Sorting on lifetime gain while the
  // row shows a windowed number is the bug this fixes (e.g. Gain% sort on 1D
  // ordered by all-time return, not the 1D return on screen).
  const gainOf = (p) => windowByTicker[p.ticker]?.return_pct ?? p.gain_loss_pct ?? 0

  // Derived (never mutate the prop array, never store in state). Spread before
  // sort so the source array is untouched. All comparators are null-safe.
  const sortedPositions = useMemo(() => {
    const arr = [...(positions || [])]
    switch (sortKey) {
      case 'gain':
        return arr.sort((a, b) => gainOf(b) - gainOf(a))
      case 'score':
        return arr.sort((a, b) => scoreOf(b) - scoreOf(a))
      case 'days':
        return arr.sort((a, b) => daysHeldOf(b) - daysHeldOf(a))
      case 'ticker':
        return arr.sort((a, b) => (a.ticker ?? '').localeCompare(b.ticker ?? ''))
      case 'sector':
        return arr.sort((a, b) => (a.sector ?? '').localeCompare(b.sector ?? ''))
      case 'value':
      default:
        return arr.sort((a, b) => (b.current_value ?? 0) - (a.current_value ?? 0))
    }
    // windowByTicker in deps so flipping the time-range window re-sorts the
    // 'gain' order to match the freshly-displayed per-window returns.
  }, [positions, sortKey, windowByTicker])

  if (!positions || positions.length === 0) {
    return (
      <EmptyState
        className="mb-4"
        message="No positions yet"
        hint="Initialize the portfolio to start trading."
      />
    )
  }

  const windowLabel = WINDOW_LABELS[timeRange] || WINDOW_LABELS.all

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <span className="text-sm font-semibold text-dark-100">
          Positions <span className="text-dark-400 font-normal">({positions.length})</span>
        </span>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="text-[10px] text-dark-500 font-data" aria-live="polite">…</span>
          )}
          {/* Sort control — mirrors the strategy <select> styling in ConfigPanel */}
          <div className="relative">
            <select
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value)}
              aria-label="Sort positions"
              className="appearance-none bg-dark-700 border border-dark-600 text-dark-200 text-[11px] rounded-lg pl-2.5 pr-6 py-1 cursor-pointer hover:border-dark-500 focus:border-primary-500 focus:outline-none transition-colors"
            >
              {POSITION_SORTS.map(({ value, label }) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
            <svg className="absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none text-dark-400" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
          <div className="flex bg-dark-850 rounded-lg p-0.5">
            {WINDOW_PILLS.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setTimeRange?.(value)}
                className={`px-2 py-0.5 text-[11px] rounded transition-colors ${
                  timeRange === value
                    ? 'bg-primary-500 text-white'
                    : 'text-dark-400 hover:text-white'
                }`}
                aria-pressed={timeRange === value}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="text-[10px] text-dark-500 mb-2 -mt-1 font-data">
        Return shown: <span className="text-dark-300">{windowLabel}</span>
      </div>
      <div className="space-y-1">
        {sortedPositions.map(position => {
          const winRow = windowByTicker[position.ticker]
          // Prefer windowed return; fall back to lifetime gain_loss_pct if the
          // endpoint hasn't responded yet or for unknown tickers.
          const pct = winRow?.return_pct ?? position.gain_loss_pct
          const isPos = (pct ?? 0) >= 0
          const isMidWindow = winRow?.notes?.includes('opened mid-window')
          const glanceScore = position.is_growth_stock ? position.current_growth_score : position.current_score
          // Trailing-stop risk fields (may be absent: null below the +5%
          // activation tier, or on older payloads) — degrade to nothing.
          const ts = position.trailing_stop
          const offPeak = ts?.drop_from_peak_pct
          const offPeakTone = ts?.near_stop
            ? 'text-red-400'
            : (offPeak ?? 0) > 5 ? 'text-amber-400' : 'text-dark-500'
          return (
            <button
              key={position.id}
              type="button"
              onClick={() => setSelectedPosition(position)}
              className="w-full text-left flex justify-between items-center gap-3 py-2.5 border-b border-dark-700/30 last:border-0 hover:bg-dark-750/50 -mx-2 px-2 rounded transition-colors"
            >
              {/* At-a-glance identity only. Secondary signals (sector full name,
                  insider sentiment, short interest, growth tag, alt score) live in
                  the detail modal that opens on tap — see PositionDetailModal. */}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 min-w-0">
                  <PositionHealthChip position={position} compact />
                  <span className="font-medium text-dark-100 shrink-0">{position.ticker}</span>
                  {ts?.near_stop && (
                    <span
                      className="inline-flex items-center gap-1 text-[9px] font-semibold uppercase tracking-wide text-red-400 bg-red-500/10 border border-red-500/30 rounded px-1 py-px shrink-0"
                      title={`Within 70% of the trailing-stop threshold${ts.threshold_pct != null ? ` (${ts.drop_from_peak_pct}% off peak vs ${ts.threshold_pct}% stop)` : ''}`}
                      aria-label="Near trailing stop"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
                      stop
                    </span>
                  )}
                  {position.sector && (
                    <span className="text-[10px] text-dark-500 truncate" title={position.sector}>
                      {position.sector.split(' ')[0]}
                    </span>
                  )}
                </div>
                <div className="text-dark-400 text-[10px] font-data mt-0.5">
                  {position.shares.toFixed(2)} shares @ {formatCurrency(position.cost_basis)}
                  {offPeak != null && offPeak > 0 && (
                    <span
                      className={`${offPeakTone}`}
                      title={`Peak ${ts.peak_price != null ? formatCurrency(ts.peak_price) : '—'}${ts.peak_date ? ` on ${new Date(ts.peak_date).toLocaleDateString()}` : ''}${ts.threshold_pct != null ? ` · trailing stop at −${ts.threshold_pct}%` : ''}`}
                    >
                      {' '}· ↓{offPeak.toFixed(1)}% off peak
                    </span>
                  )}
                  <ExitPlanChip plan={position.exit_plan} />
                </div>
              </div>
              <div className="text-right shrink-0">
                <div className="font-semibold font-data text-dark-100">{formatCurrency(position.current_value)}</div>
                <span
                  className={`text-[10px] font-data ${isPos ? 'text-emerald-400' : 'text-red-400'}`}
                  title={isMidWindow ? 'Position opened during this window — return shown since purchase' : undefined}
                >
                  {isPos ? '+' : ''}{pct != null ? pct.toFixed(2) : '—'}%
                  {isMidWindow && <span className="text-dark-500 ml-1">·new</span>}
                </span>
              </div>
              <ScoreBadge score={glanceScore} ticker={position.ticker} size="sm" className="shrink-0" />
            </button>
          )
        })}
      </div>
      <PositionDetailModal position={selectedPosition} onClose={() => setSelectedPosition(null)} />
    </Card>
  )
}

// ── Buy Signal Factors (BUY-trade transparency) ─────────────────────
// Backend writes ~15-25 structured keys per BUY into trade.signal_factors;
// this surfaces them grouped so the user can answer "why did the bot buy?"
// without leaving the modal. Conditional bonuses only render when present.

const ENTRY_TYPE_LABEL = {
  'breakout':     { label: 'Breakout',     color: 'emerald' },
  'pre-breakout': { label: 'Pre-breakout', color: 'amber' },
  'standard':     { label: 'Standard',     color: 'default' },
}
const REGIME_LABEL = {
  'bullish':    { label: 'Bullish regime',    color: 'emerald' },
  'neutral':    { label: 'Neutral regime',    color: 'default' },
  'bearish':    { label: 'Bearish regime',    color: 'red' },
  'correction': { label: 'Correction',        color: 'red' },
}

function Chip({ color = 'default', children, title }) {
  // Mirror TagBadge color palette but with tighter sizing for chip grids.
  const palette = {
    default: 'bg-dark-700/40 text-dark-300 border-dark-600',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    amber:   'bg-amber-500/10 text-amber-400 border-amber-500/30',
    red:     'bg-red-500/10 text-red-400 border-red-500/30',
    teal:    'bg-teal-500/10 text-teal-400 border-teal-500/30',
    blue:    'bg-blue-500/10 text-blue-400 border-blue-500/30',
    purple:  'bg-purple-500/10 text-purple-400 border-purple-500/30',
  }
  return (
    <span title={title} className={`text-[10px] font-data px-1.5 py-0.5 rounded border ${palette[color] || palette.default}`}>
      {children}
    </span>
  )
}

function BuySignalFactors({ factors }) {
  if (!factors) return null

  const entry = ENTRY_TYPE_LABEL[factors.entry_type]
  const regime = REGIME_LABEL[factors.market_regime]
  const composite = factors.composite_score

  // CANSLIM components — only render if at least one is present.
  const canslimKeys = ['c_score', 'a_score', 'n_score', 's_score', 'l_score', 'i_score']
  const hasCanslim = canslimKeys.some(k => factors[k] != null)

  // Price-action features
  const paKeys = [
    ['relative_volume', 'RelVol', (v) => `${v.toFixed(1)}x`],
    ['pct_from_21ma',   '21MA',   (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`],
    ['pct_from_50ma',   '50MA',   (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`],
    ['atr_pct',         'ATR',    (v) => `${v.toFixed(1)}%`],
    ['sector_rs_rank',  'Sec RS', (v) => `${v.toFixed(0)}`],
  ]
  const hasPa = paKeys.some(([k]) => factors[k] != null)

  // Bonus signals — only render the ones that fired
  const bonusKeys = [
    ['rs_line_bonus',          'RS line',            'teal'],
    ['earnings_drift_bonus',   'Earnings drift',     'emerald'],
    ['estimate_revision_bonus','Est revision',       'emerald'],
    ['bear_base_bonus',        'Bear base',          'amber'],
  ]
  const bonusChips = bonusKeys
    .filter(([k]) => factors[k] != null && factors[k] !== 0)
    .map(([k, label, color]) => ({ key: k, label, color, value: factors[k] }))

  // Flag-style signals
  const flagChips = []
  if (factors.volume_dry_up) flagChips.push({ key: 'vdu', label: 'Volume dry-up', color: 'teal' })
  if (factors.coiled_spring) flagChips.push({ key: 'cs', label: 'Coiled Spring', color: 'teal' })
  if (factors.correction_zone_entry) flagChips.push({ key: 'cz', label: 'Correction zone', color: 'amber' })
  if (factors.soft_zone) flagChips.push({ key: 'sz', label: 'Soft zone', color: 'amber' })
  if (factors.deterministic_boost) flagChips.push({ key: 'db', label: `Det boost +${factors.deterministic_boost}`, color: 'purple' })

  // Coiled-spring detail (only if coiled_spring fired AND cs_* keys present)
  const csDetail = factors.coiled_spring && factors.cs_bonus != null ? {
    bonus: factors.cs_bonus,
    weeks: factors.cs_weeks_in_base,
    streak: factors.cs_beat_streak,
    days: factors.cs_days_to_earnings,
    inst: factors.cs_institutional_pct,
    rank: factors.cs_quality_rank,
    conf: factors.cs_confidence,
  } : null

  return (
    <div className="pt-3 space-y-3">
      <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Buy Signal</span>

      {/* Header chips: entry_type + regime + composite */}
      <div className="flex flex-wrap gap-1.5 items-center">
        {entry && <Chip color={entry.color} title={`Entry type: ${factors.entry_type}`}>{entry.label}</Chip>}
        {regime && <Chip color={regime.color} title={`Market regime when bought: ${factors.market_regime}`}>{regime.label}</Chip>}
        {composite != null && <Chip color="blue" title="Composite score (CANSLIM + bonuses + market context)">Composite {composite.toFixed(1)}</Chip>}
      </div>

      {/* CANSLIM letter scores */}
      {hasCanslim && (
        <div>
          <div className="text-[9px] uppercase tracking-wider text-dark-500 mb-1">CANSLIM components</div>
          <div className="flex flex-wrap gap-1">
            {canslimKeys.map(k => {
              if (factors[k] == null) return null
              const letter = k[0].toUpperCase()
              return <Chip key={k} title={`${letter} score`}>{letter}: {factors[k].toFixed(1)}</Chip>
            })}
          </div>
        </div>
      )}

      {/* Price action */}
      {hasPa && (
        <div>
          <div className="text-[9px] uppercase tracking-wider text-dark-500 mb-1">Price action</div>
          <div className="flex flex-wrap gap-1">
            {paKeys.map(([k, label, fmt]) => {
              if (factors[k] == null) return null
              return <Chip key={k} title={label}>{label}: {fmt(factors[k])}</Chip>
            })}
          </div>
        </div>
      )}

      {/* Bonuses + flags */}
      {(bonusChips.length > 0 || flagChips.length > 0) && (
        <div>
          <div className="text-[9px] uppercase tracking-wider text-dark-500 mb-1">Bonus signals</div>
          <div className="flex flex-wrap gap-1">
            {bonusChips.map(b => (
              <Chip key={b.key} color={b.color} title={`${b.label} bonus`}>
                {b.label} +{typeof b.value === 'number' ? b.value.toFixed(1) : b.value}
              </Chip>
            ))}
            {flagChips.map(f => <Chip key={f.key} color={f.color}>{f.label}</Chip>)}
          </div>
        </div>
      )}

      {/* Coiled Spring detail */}
      {csDetail && (
        <div className="bg-teal-500/[0.04] border border-teal-500/20 rounded-lg p-2.5">
          <div className="text-[10px] text-teal-300 font-semibold mb-1.5">Coiled Spring detail</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-dark-300 font-data">
            {csDetail.bonus != null && <div>Bonus: <span className="text-teal-400">+{csDetail.bonus.toFixed(1)}</span></div>}
            {csDetail.weeks != null && <div>Base: {csDetail.weeks}w</div>}
            {csDetail.streak != null && <div>Beat streak: {csDetail.streak}x</div>}
            {csDetail.days != null && <div>Earnings: {csDetail.days}d</div>}
            {csDetail.inst != null && <div>Inst: {csDetail.inst.toFixed(1)}%</div>}
            {csDetail.rank != null && <div>Quality rank: {csDetail.rank}</div>}
            {csDetail.conf != null && <div>Confidence: {Math.round(csDetail.conf)}/100</div>}
          </div>
        </div>
      )}
    </div>
  )
}

// Renders the structured SELL signal_factors that ai_trader.py writes for
// every exit: which rule triggered (STOP LOSS / TRAILING STOP / PARTIAL
// TRAILING / SCORE CRASH / TAKE PROFIT / CIRCUIT BREAKER / ...), the gain
// at exit, and the rule-specific context (stop_pct, drop_from_peak,
// sell_pct, drawdown_pct). Mirrors the BUY-side BuySignalFactors layout
// so the modal looks consistent across action types.
const SELL_REASON_COLOR = {
  'STOP LOSS':         'red',
  'CIRCUIT BREAKER':   'red',
  'SCORE CRASH':       'red',
  'WEAK POSITION':     'amber',
  'PROTECT GAINS':     'amber',
  'PRE-EARNINGS':      'amber',
  'TRAILING STOP':     'amber',
  'PARTIAL TRAILING':  'amber',
  'TAKE PROFIT':       'emerald',
  'PARTIAL PROFIT':    'emerald',
}

function SellSignalFactors({ factors }) {
  if (!factors) return null
  const reason = factors.sell_reason
  if (!reason) return null

  const color = SELL_REASON_COLOR[reason] || 'default'
  // Order matters: context chips read left-to-right as the trade story.
  const contextChips = []
  if (factors.gain_pct != null) {
    const v = factors.gain_pct
    const c = v >= 0 ? 'emerald' : 'red'
    contextChips.push({ key: 'gain', label: `Gain ${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, color: c, title: 'Realized gain at exit' })
  }
  if (factors.stop_pct != null) {
    contextChips.push({ key: 'stop', label: `Stop ${factors.stop_pct.toFixed(1)}%`, color: 'default', title: 'Position stop-loss level' })
  }
  if (factors.drop_from_peak != null) {
    contextChips.push({ key: 'dfp', label: `−${factors.drop_from_peak.toFixed(1)}% from peak`, color: 'default', title: 'How far the price fell from its peak before triggering' })
  }
  if (factors.sell_pct != null) {
    contextChips.push({ key: 'pct', label: `Sold ${factors.sell_pct}%`, color: 'default', title: 'Fraction of position sold' })
  }
  if (factors.drawdown_pct != null) {
    contextChips.push({ key: 'dd', label: `DD ${factors.drawdown_pct.toFixed(1)}%`, color: 'red', title: 'Portfolio drawdown that triggered the circuit breaker' })
  }

  return (
    <div className="pt-3 space-y-2">
      <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Sell Signal</span>
      <div className="flex flex-wrap gap-1.5 items-center">
        <Chip color={color} title="Rule that triggered the exit">{reason}</Chip>
        {contextChips.map(c => (
          <Chip key={c.key} color={c.color} title={c.title}>{c.label}</Chip>
        ))}
      </div>
    </div>
  )
}

// ── Trade Detail Modal ──────────────────────────────────────────────
function TradeDetailModal({ trade, onClose }) {
  if (!trade) return null

  // Use centralized formatter (imported from api.js)

  const gainPct = trade.action === 'SELL' && trade.cost_basis
    ? ((trade.price - trade.cost_basis) / trade.cost_basis * 100)
    : null

  return (
    <Modal
      open={!!trade}
      onClose={onClose}
      title={
        <div className="flex items-center gap-2">
          <ActionBadge action={trade.action} />
          <Link
            to={`/stock/${trade.ticker}`}
            className="text-sm font-bold text-primary-400 hover:underline"
            onClick={onClose}
          >
            {trade.ticker}
          </Link>
          {trade.is_growth_stock && <TagBadge color="purple">Growth</TagBadge>}
        </div>
      }
    >
      <div className="space-y-0">
        <StatRow label="Date & Time" value={formatDateTime(trade.executed_at)} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Shares" value={<span className="font-data">{trade.shares.toFixed(4)}</span>} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Price" value={<span className="font-data">{formatCurrency(trade.price)}</span>} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Total Value" value={<span className="font-data">{formatCurrency(trade.total_value)}</span>} />

        {trade.action === 'SELL' && trade.cost_basis && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow label="Cost Basis" value={<span className="font-data">{formatCurrency(trade.cost_basis)}/share</span>} />
          </>
        )}

        {trade.realized_gain != null && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow
              label="Realized Gain/Loss"
              value={
                <span className={`font-data font-medium ${trade.realized_gain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
                  {gainPct != null && ` (${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(1)}%)`}
                </span>
              }
            />
          </>
        )}

        <div className="border-b border-dark-700/30" />
        <StatRow label="CANSLIM Score" value={<span className="font-data">{trade.canslim_score?.toFixed(1) || 'N/A'}</span>} />

        {trade.is_growth_stock && trade.growth_mode_score && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow label="Growth Mode Score" value={<span className="font-data">{trade.growth_mode_score.toFixed(1)}</span>} />
          </>
        )}

        {trade.signal_factors?.ml_confidence != null && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow label="ML Confidence" value={
              <MLConfidenceBadge confidence={trade.signal_factors.ml_confidence} size="sm" />
            } />
            {trade.signal_factors?.ml_bonus != null && trade.signal_factors.ml_bonus !== 0 && (
              <StatRow label="ML Bonus" value={
                <span className={`font-data ${trade.signal_factors.ml_bonus >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {trade.signal_factors.ml_bonus >= 0 ? '+' : ''}{trade.signal_factors.ml_bonus.toFixed(1)}
                </span>
              } />
            )}
          </>
        )}

        {/* Buy Signal Factors — only for BUY trades with structured factors */}
        {trade.action === 'BUY' && trade.signal_factors && (
          <BuySignalFactors factors={trade.signal_factors} />
        )}

        {/* Sell Signal Factors — only for SELL trades with structured factors */}
        {trade.action === 'SELL' && trade.signal_factors?.sell_reason && (
          <SellSignalFactors factors={trade.signal_factors} />
        )}

        {/* Reason Section */}
        <div className="pt-3">
          <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Reason</span>
          <div className="bg-dark-850 rounded-lg p-3 text-sm text-dark-200 mt-1.5">
            {trade.reason || 'No reason recorded'}
          </div>
        </div>
      </div>
    </Modal>
  )
}

// ── Exit Plan ───────────────────────────────────────────────────────
// Read-only view of the live sell triggers for a held position. The math
// (stop / trailing / take-profit price levels, distance-to-trigger) is computed
// server-side in backend/exit_plan.py — which reuses the trader's own threshold
// helpers — so what's shown here is what would actually fire. The component is
// pure presentation; it never re-derives a threshold client-side.
function ExitPlanSection({ plan }) {
  if (!plan || !plan.triggers?.length) return null

  const distLabel = (t) => {
    if (t.reached) return 'target passed'
    if (t.distance_pct == null) return null
    return t.direction === 'up' ? `${t.distance_pct}% to go` : `${t.distance_pct}% away`
  }
  // Protective stops tighten in tone as price nears them; the upside target is
  // always green (nearing it is good).
  const distTone = (t) => {
    if (t.direction === 'up') return 'text-emerald-400'
    if (t.distance_pct == null) return 'text-dark-400'
    if (t.distance_pct <= 5) return 'text-red-400'
    if (t.distance_pct <= 12) return 'text-amber-400'
    return 'text-dark-300'
  }

  return (
    <div className="bg-dark-850 rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Exit Plan</span>
        <span className="text-[10px] text-dark-500">what would trigger a sell</span>
      </div>
      <div className="space-y-1.5">
        {plan.triggers.map((t) => {
          const nearest = t.kind === plan.nearest_kind
          return (
            <div
              key={t.kind}
              className={`flex items-center justify-between rounded-md px-2 py-1.5 ${
                nearest ? 'bg-primary-500/10 ring-1 ring-primary-500/30' : ''
              }`}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-semibold text-dark-100">{t.label}</span>
                  {nearest && <span className="text-[9px] uppercase tracking-wider text-primary-400">nearest</span>}
                  {t.reached && <span className="text-[9px] uppercase tracking-wider text-emerald-400">reached</span>}
                </div>
                <div className="text-[10px] text-dark-500 truncate">{t.note}</div>
              </div>
              <div className="text-right shrink-0 ml-2">
                <div className="text-xs font-data text-dark-100">
                  {t.price != null ? formatCurrency(t.price) : `< ${t.threshold}`}
                </div>
                {distLabel(t) && <div className={`text-[10px] font-data ${distTone(t)}`}>{distLabel(t)}</div>}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// Compact one-line readout of the position's nearest exit trigger, for the
// positions list rows. Same server-computed data as ExitPlanSection (the card
// in the detail modal) — presentation-only, never re-derives a threshold.
// Renders nothing when the plan has no price trigger (e.g. data still loading).
function ExitPlanChip({ plan }) {
  if (!plan?.triggers?.length || !plan.nearest_kind) return null
  const t = plan.triggers.find((x) => x.kind === plan.nearest_kind)
  if (!t || t.price == null) return null
  // Same tone thresholds as ExitPlanSection: protective stops tighten in tone
  // as price nears them; the upside target is always green.
  const tone = t.direction === 'up'
    ? 'text-emerald-400'
    : t.distance_pct == null ? 'text-dark-400'
      : t.distance_pct <= 5 ? 'text-red-400'
        : t.distance_pct <= 12 ? 'text-amber-400'
          : 'text-dark-400'
  const dist = t.distance_pct != null
    ? (t.direction === 'up' ? `${t.distance_pct}% to go` : `${t.distance_pct}% away`)
    : null
  return (
    <span
      className={`font-data ${tone}`}
      title={`Nearest exit trigger: ${t.label}${t.note ? ` — ${t.note}` : ''}`}
    >
      {' '}· {t.label} {formatCurrency(t.price)}{dist ? ` (${dist})` : ''}
    </span>
  )
}

// ── Position Detail Modal ───────────────────────────────────────────
// Column spec for the per-ticker trade-history DataTable inside the modal.
// Mirrors the old hand-rolled <table> (Date/Action/Shares/Price/Total/Reason,
// Reason hidden on mobile) but every column is now click-to-sort.
const POSITION_TRADE_COLUMNS = [
  {
    key: 'executed_at', label: 'Date', sortable: true,
    sortValue: (t) => t.executed_at ? new Date(t.executed_at).getTime() : null,
    render: (v) => <span className="whitespace-nowrap">{formatDateTime(v)}</span>,
    className: 'text-xs text-dark-300',
  },
  {
    key: 'action', label: 'Action', sortable: true,
    render: (v) => <ActionBadge action={v} />,
  },
  {
    key: 'shares', label: 'Shares', align: 'right', sortable: true, mono: true,
    render: (v) => v?.toFixed(4),
    className: 'text-xs text-dark-200',
  },
  {
    key: 'price', label: 'Price', align: 'right', sortable: true, mono: true,
    render: (v) => formatCurrency(v),
    className: 'text-xs text-dark-200',
  },
  {
    key: 'total_value', label: 'Total', align: 'right', sortable: true, mono: true,
    render: (v) => formatCurrency(v),
    className: 'text-xs text-dark-200',
  },
  {
    key: 'reason', label: 'Reason', mobileHide: true,
    render: (v) => v || '-',
    className: 'text-xs text-dark-400',
  },
]

function PositionDetailModal({ position, onClose }) {
  const [trades, setTrades] = useState(null)
  const [scoreHistory, setScoreHistory] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!position) { setTrades(null); setScoreHistory(null); return }
    // Stale-effect guard: tapping position A then quickly position B (or
    // closing the modal) leaves A's fetch in flight — its late resolve must
    // not render A's trades/score history under B's header.
    let stale = false
    setLoading(true)
    // Fetch trades + stock (for score_history) in parallel — both keyed on ticker.
    Promise.all([
      api.getAIPortfolioTrades(200, position.ticker)
        .then(t => Array.isArray(t) ? t : [])
        .catch(() => []),
      api.getStock(position.ticker)
        .then(s => s?.score_history || null)
        .catch(() => null),
    ])
      .then(([t, sh]) => { if (!stale) { setTrades(t); setScoreHistory(sh) } })
      .finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [position?.id, position?.ticker])

  if (!position) return null

  const totalCost = position.shares * position.cost_basis
  const daysHeld = position.purchase_date
    ? Math.max(0, Math.floor((Date.now() - new Date(position.purchase_date).getTime()) / 86400000))
    : null
  const peakDistPct = position.peak_price && position.current_price
    ? ((position.current_price - position.peak_price) / position.peak_price) * 100
    : null
  // Prefer counting actual BUY trades; fall back to pyramid_count + 1 before trades load
  const buyTrades = trades ? trades.filter(t => t.action === 'BUY').length : null
  const buys = buyTrades != null ? buyTrades : (position.pyramid_count ?? 0) + 1
  const pyramidsShown = buyTrades != null ? Math.max(0, buyTrades - 1) : (position.pyramid_count ?? 0)
  const purchaseScore = position.is_growth_stock ? position.purchase_growth_score : position.purchase_score
  const currentScore = position.is_growth_stock ? position.current_growth_score : position.current_score
  const scoreDelta = (purchaseScore != null && currentScore != null) ? (currentScore - purchaseScore) : null
  const gainPositive = (position.gain_loss ?? 0) >= 0

  // Ordering is owned by the sortable DataTable below (default: newest
  // first; every column header click re-sorts client-side).
  const orderedTrades = trades || []

  return (
    <Modal
      open={!!position}
      onClose={onClose}
      size="lg"
      title={
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-base font-bold text-dark-100">{position.ticker}</span>
          {position.sector && <TagBadge color="blue">{position.sector}</TagBadge>}
          {position.is_growth_stock && <TagBadge color="purple">Growth</TagBadge>}
          {position.insider_sentiment === 'bullish' && (
            <TagBadge color="emerald" title="Aggregate insider sentiment is bullish (net buys exceed sells)">
              Insider: bullish
            </TagBadge>
          )}
          {position.insider_sentiment === 'bearish' && (
            <TagBadge color="amber" title="Aggregate insider sentiment is bearish (net sells exceed buys)">
              Insider: bearish
            </TagBadge>
          )}
          {(position.short_interest_pct ?? 0) >= 20 && (
            <TagBadge color="red" title="Short interest above 20% — crowded short, squeeze risk in either direction">
              Short: {position.short_interest_pct.toFixed(0)}%
            </TagBadge>
          )}
          {(position.short_interest_pct ?? 0) >= 10 && (position.short_interest_pct ?? 0) < 20 && (
            <TagBadge color="amber" title="Short interest in the 10-20% range — elevated but not extreme">
              Short: {position.short_interest_pct.toFixed(0)}%
            </TagBadge>
          )}
        </div>
      }
    >
      <div className="space-y-4">
        {/* Top-line P&L banner */}
        <div className="bg-dark-850 rounded-lg p-3 flex items-baseline justify-between">
          <div>
            <div className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Current Value</div>
            <div className="text-xl font-bold font-data text-dark-100">{formatCurrency(position.current_value)}</div>
          </div>
          <div className="text-right">
            <div className={`text-base font-data font-medium ${gainPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {gainPositive ? '+' : ''}{formatCurrency(position.gain_loss)}
            </div>
            <div className={`text-xs font-data ${gainPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {gainPositive ? '+' : ''}{position.gain_loss_pct?.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Stats grid (2 cols) */}
        <div className="grid grid-cols-2 gap-x-6 gap-y-0">
          <StatRow label="Avg Cost" value={<span className="font-data">{formatCurrency(position.cost_basis)}</span>} />
          <StatRow label="Current Price" value={<span className="font-data">{formatCurrency(position.current_price)}</span>} />
          <StatRow label="Shares" value={<span className="font-data">{position.shares?.toFixed(4)}</span>} />
          <StatRow label="Total Cost" value={<span className="font-data">{formatCurrency(totalCost)}</span>} />
          <StatRow label="# of Buys" value={<span className="font-data">{buys}{pyramidsShown > 0 ? ` (${pyramidsShown} pyramid${pyramidsShown > 1 ? 's' : ''})` : ''}</span>} />
          <StatRow label="Days Held" value={<span className="font-data">{daysHeld != null ? daysHeld : '—'}</span>} />
          <StatRow label="Peak Price" value={<span className="font-data">{formatCurrency(position.peak_price)}</span>} />
          <StatRow
            label="From Peak"
            value={
              peakDistPct != null
                ? <span className={`font-data ${peakDistPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {peakDistPct >= 0 ? '+' : ''}{peakDistPct.toFixed(2)}%
                  </span>
                : <span className="font-data text-dark-400">—</span>
            }
          />
          <StatRow label="Score at Buy" value={<span className="font-data">{purchaseScore?.toFixed(1) ?? '—'}</span>} />
          <StatRow
            label="Current Score"
            value={
              <span className="font-data">
                {currentScore?.toFixed(1) ?? '—'}
                {scoreDelta != null && (
                  <span className={`ml-1 text-[10px] ${scoreDelta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    ({scoreDelta >= 0 ? '+' : ''}{scoreDelta.toFixed(1)})
                  </span>
                )}
              </span>
            }
          />
          {/* Alt score — moved off the holdings row; shown here so both scores
              stay available for growth/CANSLIM stocks. */}
          {position.is_growth_stock && position.current_score > 0 && (
            <StatRow label="CANSLIM Score" value={<span className="font-data">{position.current_score.toFixed(1)}</span>} />
          )}
          {!position.is_growth_stock && position.current_growth_score > 0 && (
            <StatRow label="Growth Score" value={<span className="font-data">{position.current_growth_score.toFixed(1)}</span>} />
          )}
          {position.partial_profit_taken > 0 && (
            <StatRow label="Partial Profit" value={<span className="font-data">{position.partial_profit_taken.toFixed(0)}% sold</span>} />
          )}
        </div>

        {/* Exit plan — live sell triggers (server-computed, mirrors evaluate_sells) */}
        <ExitPlanSection plan={position.exit_plan} />

        {/* Score history sparkline */}
        {scoreHistory && scoreHistory.length >= 2 && (
          <div className="bg-dark-850 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Score Trajectory</span>
              <span className="text-[10px] font-data text-dark-500">{scoreHistory.length} pts</span>
            </div>
            <Sparkline
              data={scoreHistory.map(h => h.total_score).filter(v => v != null)}
              width={320}
              height={48}
              strokeWidth={1.75}
              gradient
              className="w-full"
            />
            <div className="flex justify-between mt-1 text-[10px] text-dark-500 font-data">
              <span>{scoreHistory[0]?.total_score?.toFixed(1) ?? '—'}</span>
              <span>{scoreHistory[scoreHistory.length - 1]?.total_score?.toFixed(1) ?? '—'}</span>
            </div>
          </div>
        )}

        {/* Trade history */}
        <div>
          <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Trade History</span>
          {loading && <div className="text-dark-400 text-xs mt-2 py-3 text-center">Loading…</div>}
          {!loading && orderedTrades.length === 0 && (
            <div className="text-dark-400 text-xs mt-2 py-3 text-center">No trades found for this ticker.</div>
          )}
          {!loading && orderedTrades.length > 0 && (
            <div className="bg-dark-850 rounded-lg mt-1.5 overflow-hidden px-2">
              <DataTable
                columns={POSITION_TRADE_COLUMNS}
                data={orderedTrades}
                keyField="id"
                compact
                defaultSort="executed_at"
                defaultSortDir="desc"
                emptyMessage="No trades found for this ticker."
              />
            </div>
          )}
        </div>

        {/* Footer link */}
        <div className="flex justify-end pt-1">
          <Link
            to={`/stock/${position.ticker}`}
            className="text-xs text-primary-400 hover:underline"
            onClick={onClose}
          >
            View Stock Page →
          </Link>
        </div>
      </div>
    </Modal>
  )
}

// ── Trade History ───────────────────────────────────────────────────
// Column order for the AI Portfolio trade CSV. `entry_type`/`sell_reason`
// are flattened out of each trade's `signal_factors` blob at export time.
const AI_TRADE_CSV_COLUMNS = [
  'executed_at', 'ticker', 'action', 'shares', 'price', 'total_value',
  'cost_basis', 'realized_gain', 'holding_days', 'canslim_score',
  'growth_mode_score', 'is_growth_stock', 'entry_type', 'sell_reason', 'reason',
]

// Column spec for the Recent Trades DataTable. Preserves the old list's
// content (action badge + G tag, ticker, shares @ price, colored realized
// P&L, reason) but sortable; Date/Reason drop off on mobile to keep the
// row width close to the old compact list.
const RECENT_TRADE_COLUMNS = [
  {
    key: 'executed_at', label: 'Date', sortable: true, mobileHide: true,
    sortValue: (t) => t.executed_at ? new Date(t.executed_at).getTime() : null,
    render: (v) => <span className="whitespace-nowrap">{formatDateTime(v)}</span>,
    className: 'text-xs text-dark-300',
  },
  {
    key: 'action', label: 'Action', sortable: true,
    render: (v, t) => (
      <span className="inline-flex items-center gap-1.5">
        <ActionBadge action={v} />
        {t.is_growth_stock && <TagBadge color="purple">G</TagBadge>}
      </span>
    ),
  },
  {
    key: 'ticker', label: 'Ticker', sortable: true,
    render: (v) => <span className="font-medium text-dark-100">{v}</span>,
  },
  {
    key: 'shares', label: 'Shares @ Price', align: 'right', mono: true,
    render: (v, t) => `${v?.toFixed(2)} @ ${formatCurrency(t.price)}`,
    className: 'text-xs text-dark-200 whitespace-nowrap',
  },
  {
    key: 'realized_gain', label: 'P&L', align: 'right', sortable: true, mono: true,
    render: (v) => v == null ? <span className="text-dark-600">-</span> : (
      <span className={v >= 0 ? 'text-emerald-400' : 'text-red-400'}>
        {v >= 0 ? '+' : ''}{formatCurrency(v)}
      </span>
    ),
    className: 'text-xs',
  },
  {
    key: 'reason', label: 'Reason', mobileHide: true,
    render: (v) => (
      <span className="block truncate max-w-[180px]" title={v}>{v || '-'}</span>
    ),
    className: 'text-xs text-dark-400',
  },
]

function TradeHistory({ trades }) {
  const [selectedTrade, setSelectedTrade] = useState(null)
  const [exporting, setExporting] = useState(false)
  const toast = useToast()

  if (!trades || trades.length === 0) {
    return null
  }

  // Export ALL trades, not just the ~20 shown here: re-fetch at the endpoint's
  // max (200, comfortably above the live trade count) so the download reflects
  // the full history regardless of what's rendered.
  async function handleExport() {
    setExporting(true)
    try {
      const all = await api.getAIPortfolioTrades(200)
      const rows = all.map(t => ({
        ...t,
        is_growth_stock: t.is_growth_stock ? 'yes' : 'no',
        entry_type: t.signal_factors?.entry_type ?? '',
        sell_reason: t.signal_factors?.sell_reason ?? '',
      }))
      const stamp = new Date().toISOString().slice(0, 10)
      downloadCsv(`ai-portfolio-trades-${stamp}.csv`, buildCsv(AI_TRADE_CSV_COLUMNS, rows))
    } catch (err) {
      toast.error(err?.message || 'Failed to export trades')
    } finally {
      setExporting(false)
    }
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader
        title="Recent Trades"
        action={
          <button
            onClick={handleExport}
            disabled={exporting}
            className="text-xs text-dark-300 hover:text-dark-100 px-3 py-1 rounded border border-dark-700 hover:border-dark-600 transition-colors disabled:opacity-50"
            title="Export all AI Portfolio trades as CSV"
          >
            {exporting ? 'Exporting…' : 'Export CSV'}
          </button>
        }
      />
      {/* No inner max-height: let the table grow to ~20 rows; outer page
          scroll handles overflow. Removes scroll-in-scroll on mobile.
          Row click still opens the TradeDetailModal below. */}
      <DataTable
        columns={RECENT_TRADE_COLUMNS}
        data={trades.slice(0, 20)}
        keyField="id"
        compact
        defaultSort="executed_at"
        defaultSortDir="desc"
        onRowClick={setSelectedTrade}
        emptyMessage="No trades yet"
      />

      {/* Trade Detail Modal */}
      <TradeDetailModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
    </Card>
  )
}

// ── Sector Allocation Chart ─────────────────────────────────────────
// Warm-leaning 10-color qualitative palette for pie slices. Anchors on the
// brand amber + deep copper, fills out with emerald/teal/rose/etc. to stay
// readable against the warm-dark surface. Avoids cool blues/violets that
// clashed with the rebrand.
const SECTOR_COLORS = [chartColors.brand, '#b45309', '#10b981', '#fb923c', '#f43f5e',
  '#fbbf24', '#22d3ee', '#a855f7', '#22c55e', '#fde68a']

function SectorAllocationChart({ riskData, cashPct }) {
  if (!riskData?.sector_concentration || riskData.sector_concentration.length === 0) return null

  const chartData = [
    ...riskData.sector_concentration.map(s => ({ name: s.sector || 'Unknown', value: s.pct })),
    ...(cashPct > 1 ? [{ name: 'Cash', value: Math.round(cashPct * 10) / 10 }] : [])
  ]

  const renderLabel = ({ name, value, cx, cy, midAngle, innerRadius, outerRadius }) => {
    if (value < 5) return null
    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)
    return (
      <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={10} fontFamily="JetBrains Mono">
        {Math.round(value)}%
      </text>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Sector Allocation" />
      <div className="flex items-center">
        <div className="w-1/2 h-40">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={60}
                dataKey="value"
                label={renderLabel}
                labelLine={false}
              >
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.name === 'Cash' ? chartColors.muted : SECTOR_COLORS[i % SECTOR_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v) => `${v.toFixed(1)}%`}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="w-1/2 space-y-1 text-xs">
          {chartData.map((entry, i) => (
            <div key={entry.name} className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: entry.name === 'Cash' ? chartColors.muted : SECTOR_COLORS[i % SECTOR_COLORS.length] }} />
              <span className="text-dark-300 truncate">{entry.name}</span>
              <span className="text-dark-400 ml-auto font-data">{entry.value.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

// ── Config Panel ────────────────────────────────────────────────────
function ConfigPanel({ config, onUpdate, onInitialize, onRunCycle, onRefresh, waitingForTrades }) {
  const [isActive, setIsActive] = useState(config?.is_active || false)
  const [updating, setUpdating] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [strategies, setStrategies] = useState([])
  const [changingStrategy, setChangingStrategy] = useState(false)
  const [startCash, setStartCash] = useState(config?.starting_cash || 25000)

  useEffect(() => {
    setIsActive(config?.is_active || false)
    setStartCash(config?.starting_cash || 25000)
  }, [config])

  // Load available strategies
  useEffect(() => {
    api.getStrategies().then(setStrategies).catch(() => {})
  }, [])

  const handleToggle = async () => {
    setUpdating(true)
    try {
      await onUpdate({ is_active: !isActive })
      setIsActive(!isActive)
    } finally {
      setUpdating(false)
    }
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await onRefresh()
      // Keep spinner for 12 seconds while background task runs
      setTimeout(() => setRefreshing(false), 12000)
    } catch {
      setRefreshing(false)
    }
  }

  const handleRunCycle = async () => {
    try {
      await onRunCycle()
    } catch (err) {
      console.error('Failed to run cycle:', err)
    }
  }

  // Clamp to the endpoint's accepted range ($1k–$1M) so a stray input can't
  // 400 the request; round to whole dollars.
  const initCash = Math.min(1000000, Math.max(1000, Math.round(Number(startCash) || 0)))

  const handleInitialize = async () => {
    if (!confirm(`This will reset the AI Portfolio to $${initCash.toLocaleString()} and clear all history. Continue?`)) {
      return
    }
    setInitializing(true)
    try {
      await onInitialize(initCash)
    } finally {
      setInitializing(false)
    }
  }

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <CardHeader title="AI Trading" className="mb-0" />
          {config?.paper_mode && (
            <TagBadge color="amber" title="Paper mode — trades are simulated, no real positions are affected">
              Paper Mode
            </TagBadge>
          )}
        </div>
        <button
          onClick={handleToggle}
          disabled={updating}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            isActive ? 'bg-emerald-500' : 'bg-dark-600'
          }`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            isActive ? 'translate-x-7' : 'translate-x-1'
          }`} />
        </button>
      </div>

      {isActive && (
        <div className="mb-3 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-emerald-400 text-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>AI Trading Active - Trades execute after each scan</span>
          </div>
        </div>
      )}

      <div className="border-t border-dark-700/30 pt-3 mb-3">
        {/* Strategy Selector */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Strategy</span>
          <div className="relative">
            <select
              value={config?.strategy || 'balanced'}
              onChange={async (e) => {
                setChangingStrategy(true)
                try {
                  await onUpdate({ strategy: e.target.value })
                } finally {
                  setChangingStrategy(false)
                }
              }}
              disabled={changingStrategy}
              className="appearance-none bg-dark-700 border border-dark-600 text-dark-200 text-xs rounded-lg px-3 py-1.5 pr-7 cursor-pointer hover:border-dark-500 focus:border-primary-500 focus:outline-none transition-colors disabled:opacity-50"
            >
              {strategies.length > 0 ? strategies.map(s => (
                <option key={s.name} value={s.name}>{s.label}</option>
              )) : (
                <option value={config?.strategy || 'balanced'}>
                  {(config?.strategy || 'balanced').replace(/_/g, ' ')}
                </option>
              )}
            </select>
            <svg className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-dark-400" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
        </div>

        <StatGrid
          columns={3}
          stats={[
            { label: 'Min Score', value: config?.min_score_to_buy || '72' },
            { label: 'Take Profit', value: `+${config?.take_profit_pct || 75}%`, color: 'text-emerald-400' },
            { label: 'Stop Loss', value: `-${config?.stop_loss_pct || 7}%`, color: 'text-red-400' },
          ]}
          className="text-sm"
        />
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex-1 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          <svg
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            className={refreshing ? 'animate-spin' : ''}
          >
            <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
          </svg>
          <span>{refreshing ? 'Refreshing...' : 'Refresh Prices'}</span>
        </button>
        <button
          onClick={handleRunCycle}
          disabled={waitingForTrades}
          className="flex-1 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {waitingForTrades && (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
          )}
          <span>{waitingForTrades ? 'Running...' : 'Run Trading Cycle'}</span>
        </button>
      </div>

      <div className="flex items-center gap-2 mb-2">
        <label htmlFor="ai-start-cash" className="text-xs text-dark-400 whitespace-nowrap">Starting cash</label>
        <div className="relative flex-1">
          <span className="absolute left-2 top-1/2 -translate-y-1/2 text-dark-500 text-sm pointer-events-none">$</span>
          <input
            id="ai-start-cash"
            type="number"
            min={1000}
            max={1000000}
            step={1000}
            value={startCash}
            onChange={(e) => setStartCash(e.target.value)}
            className="w-full pl-5 pr-2 py-2 bg-dark-800 border border-dark-700 rounded-lg text-sm font-data text-dark-100 focus:border-primary-500 outline-none"
          />
        </div>
      </div>
      <button
        onClick={handleInitialize}
        disabled={initializing}
        className="w-full py-2 mb-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-dark-300"
      >
        {initializing ? 'Resetting...' : `Reset Portfolio ($${initCash.toLocaleString()})`}
      </button>

      <Link
        to="/backtest"
        className="block w-full py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-center text-primary-400"
      >
        Run Historical Backtest
      </Link>
    </Card>
  )
}

// ── Coiled Spring Alerts Section ────────────────────────────────────
function CoiledSpringSection({ csAlerts, csExpanded, setCsExpanded }) {
  if (!csAlerts || csAlerts.length === 0) return null

  return (
    <Card variant="accent" accent="teal" className="mb-4 bg-teal-500/[0.03]">
      <CollapsibleSection
        title="Coiled Spring Alerts"
        badge={<TagBadge color="teal">{csAlerts.length} candidates</TagBadge>}
        defaultOpen={csExpanded}
        onOpenChange={setCsExpanded}
      >
        <div className="text-[10px] text-dark-400 mb-2">
          High-conviction pre-earnings plays: long bases + beat streaks + approaching earnings
        </div>
        <div className="space-y-1">
          {csAlerts.map((stock) => {
            // Color-code entry_status: PRE-BREAKOUT = ideal setup,
            // AT_PIVOT = actionable, EXTENDED = chase risk. Backend writes
            // these labels at backend/main.py:2852.
            const entryColor = stock.entry_status === 'PRE_BREAKOUT' ? 'emerald'
              : stock.entry_status === 'AT_PIVOT' ? 'amber'
              : stock.entry_status === 'EXTENDED' ? 'red'
              : 'cyan'
            const entryLabel = stock.entry_status
              ? stock.entry_status.replace('_', ' ').toLowerCase()
              : null
            // confidence is 0-100 (CSConfidenceBadge titles it `${confidence}/100`;
            // backend thresholds live on the same scale, e.g. min_confidence: 30)
            const confColor = (stock.confidence ?? 0) >= 70 ? 'text-emerald-400'
              : (stock.confidence ?? 0) >= 50 ? 'text-amber-400'
              : 'text-dark-500'
            return (
              <Link
                key={stock.ticker}
                to={`/stock/${stock.ticker}`}
                className="flex justify-between items-center py-2 px-2 -mx-2 rounded hover:bg-dark-750/50 transition-colors border-b border-dark-700/30 last:border-0"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-dark-100">{stock.ticker}</span>
                    {stock.base_type && stock.base_type !== 'none' && (
                      <TagBadge color="cyan">{stock.weeks_in_base}w {stock.base_type}</TagBadge>
                    )}
                    {entryLabel && (
                      <TagBadge color={entryColor}>{entryLabel}</TagBadge>
                    )}
                    {stock.is_breaking_out && stock.entry_status !== 'AT_PIVOT' && (
                      <TagBadge color="amber">Breakout</TagBadge>
                    )}
                  </div>
                  {/* At-a-glance catalyst trio: entry status (badge above),
                      days-to-earnings, confidence. Component scores (C/L), beat
                      count and institutional % live on the stock page this links to. */}
                  <div className="text-[10px] text-dark-400 flex items-center gap-1.5 mt-0.5 font-data">
                    <span className="text-amber-400 whitespace-nowrap">{stock.days_to_earnings}d to earnings</span>
                    {stock.confidence != null && (
                      <>
                        <span className="text-dark-600">{'\u00B7'}</span>
                        <span className={confColor} title="Coiled Spring confidence (0-100) \u2014 backend signal overlap score">
                          conf {Math.round(stock.confidence)}%
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <ScoreBadge score={stock.canslim_score} ticker={stock.ticker} size="sm" />
                  <div className="text-[10px] text-teal-400 font-data mt-0.5">+{stock.cs_bonus} bonus</div>
                </div>
              </Link>
            )
          })}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ── Risk Monitor Section ────────────────────────────────────────────
function RiskMonitorSection({ riskData, riskExpanded, setRiskExpanded }) {
  if (!riskData) return null

  // Concentration limits come from the backend (`limits`) — the trader's
  // OWN caps (check_sector_limit / position sizing), so the tones here flag
  // exactly what would constrain the next buy. Fallbacks match the YAML
  // defaults for older cached payloads.
  const limits = riskData.limits || {}
  const sectorCapPct = limits.max_sector_allocation_pct ?? 50
  const sectorCapCount = limits.max_stocks_per_sector ?? 4
  const posCapPct = limits.max_position_pct ?? 25

  const sectorTone = (s) => {
    if (s.pct >= sectorCapPct || s.count >= sectorCapCount) return 'red'
    if (s.pct >= sectorCapPct * 0.8 || s.count >= sectorCapCount - 1) return 'amber'
    return 'default'
  }
  const weightTone = (w) =>
    w.pct >= posCapPct * 0.95 ? 'bg-red-500'
      : w.pct >= posCapPct * 0.8 ? 'bg-amber-500'
        : 'bg-emerald-500'

  // Smart-hide: when portfolio is genuinely "all clear" (no alerts, low heat,
  // no sector at/near its real cap, no position near its size cap) the
  // section is just dead vertical space. Section reappears the moment any
  // signal flips. Matches the audit's "less scroll when nothing to look at"
  // theme.
  const hasAlerts = (riskData.position_alerts?.length ?? 0) > 0
  const overConcentrated = (riskData.sector_concentration || []).some(s => sectorTone(s) !== 'default')
  const heavyPosition = (riskData.position_weights || []).some(w => w.pct >= posCapPct * 0.8)
  const heatLow = (riskData.portfolio_heat ?? 0) < 10
  if (!hasAlerts && !overConcentrated && !heavyPosition && heatLow) return null

  const heatColor = riskData.heat_status === 'danger' ? 'red'
    : riskData.heat_status === 'warning' ? 'amber' : 'green'

  return (
    <Card variant="glass" className="mb-4">
      <CollapsibleSection
        title="Risk Monitor"
        badge={
          <div className="flex items-center gap-1.5">
            <TagBadge color={heatColor}>Heat: {formatPercent(riskData.portfolio_heat)}</TagBadge>
            {riskData.position_alerts?.length > 0 && (
              <TagBadge color="red">{riskData.position_alerts.length} alerts</TagBadge>
            )}
          </div>
        }
        defaultOpen={riskExpanded}
        onOpenChange={setRiskExpanded}
      >
        <div className="space-y-3 mt-1">
          {/* Heat bar */}
          <div>
            <div className="text-[10px] text-dark-400 mb-1">Portfolio Heat</div>
            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  riskData.portfolio_heat < 10 ? 'bg-emerald-500' :
                  riskData.portfolio_heat < 15 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(riskData.portfolio_heat / 20 * 100, 100)}%` }}
              />
            </div>
          </div>
          {/* Sector concentration — toned against the trader's real caps */}
          {riskData.sector_concentration?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-400 mb-1">
                Sector Concentration
                <span className="text-dark-500"> · caps: {formatPercent(sectorCapPct)} value / {sectorCapCount} stocks</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {riskData.sector_concentration.map(s => (
                  <TagBadge
                    key={s.sector}
                    color={sectorTone(s)}
                    title={
                      sectorTone(s) === 'red'
                        ? `At a sector cap — the trader will block or trim new ${s.sector} buys`
                        : sectorTone(s) === 'amber'
                          ? `Approaching a sector cap (${formatPercent(sectorCapPct)} value or ${sectorCapCount} stocks)`
                          : undefined
                    }
                  >
                    {s.sector}: {s.count} ({formatPercent(s.pct)})
                  </TagBadge>
                ))}
              </div>
            </div>
          )}
          {/* Position weights vs the per-position size cap */}
          {riskData.position_weights?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-400 mb-1">
                Position Weights
                <span className="text-dark-500"> · cap {formatPercent(posCapPct)} each</span>
              </div>
              <div className="space-y-1">
                {riskData.position_weights.map(w => (
                  <div key={w.ticker} className="flex items-center gap-2 text-xs">
                    <span className="font-medium text-dark-200 w-14 shrink-0">{w.ticker}</span>
                    <div className="flex-1 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${weightTone(w)}`}
                        style={{ width: `${Math.min((w.pct / posCapPct) * 100, 100)}%` }}
                      />
                    </div>
                    <span className="font-data text-dark-300 w-12 text-right shrink-0">
                      {formatPercent(w.pct)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* Stop distances */}
          {riskData.stop_distances?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-400 mb-1">Distance to Stop</div>
              {riskData.stop_distances.slice(0, 5).map(s => (
                <div key={s.ticker} className="flex justify-between text-xs py-0.5">
                  <span className="font-medium text-dark-200">{s.ticker}</span>
                  <span className={`font-data ${s.distance_pct < 5 ? 'text-red-400' : 'text-dark-300'}`}>
                    {formatPercent(s.distance_pct)} ({formatPercent(s.gain_pct, true)})
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ── Earnings Calendar Section ───────────────────────────────────────
function EarningsCalendarSection({ earningsCalendar, earningsExpanded, setEarningsExpanded }) {
  if (!earningsCalendar || !earningsCalendar.positions?.length) return null

  return (
    <Card variant="glass" className="mb-4">
      <CollapsibleSection
        title="Earnings Calendar"
        badge={
          <div className="flex items-center gap-1.5">
            {earningsCalendar.upcoming_count?.high > 0 && (
              <TagBadge color="red">{earningsCalendar.upcoming_count.high} this week</TagBadge>
            )}
            {earningsCalendar.upcoming_count?.medium > 0 && (
              <TagBadge color="amber">{earningsCalendar.upcoming_count.medium} next week</TagBadge>
            )}
          </div>
        }
        defaultOpen={earningsExpanded}
        onOpenChange={setEarningsExpanded}
      >
        <div className="space-y-1 mt-1">
          {earningsCalendar.positions.map(p => (
            <div key={p.ticker} className={`flex justify-between items-center py-1.5 px-2 -mx-2 rounded ${
              p.risk_level === 'high' ? 'bg-red-500/5' : ''
            }`}>
              <div>
                <span className="font-medium text-sm text-dark-100">{p.ticker}</span>
                <span className="text-dark-400 text-xs ml-2 font-data">
                  {p.next_earnings_date || `${p.days_to_earnings}d`}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-dark-400 font-data">{p.beat_streak} beats</span>
                <TagBadge color={
                  p.risk_level === 'high' ? 'red' :
                  p.risk_level === 'medium' ? 'amber' : 'green'
                }>
                  {p.days_to_earnings}d
                </TagBadge>
                <span className={`text-xs font-data ${p.gain_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatPercent(p.gain_pct, true)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ══════════════════════════════════════════════════════════════════════
// ── Main Page Component ─────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════
const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'detail', label: 'Detail' },
]

export default function AIPortfolio() {
  const toast = useToast()
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [portfolio, setPortfolio] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const [trades, setTrades] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [waitingForTrades, setWaitingForTrades] = useState(false)
  const [waitingCash, setWaitingCash] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(() => {
    // Persist auto-refresh preference in localStorage
    const saved = localStorage.getItem('aiPortfolioAutoRefresh')
    return saved === 'true'
  })
  const [lastPriceRefresh, setLastPriceRefresh] = useState(null)
  const [isRefreshingPrices, setIsRefreshingPrices] = useState(false)
  const [csAlerts, setCsAlerts] = useState([])
  // Three section expand-states persist to localStorage so the user's
  // last collapse/expand pick survives a refresh. Each key follows the
  // `aiPortfolio<Name>Expanded` convention.
  const [csExpanded, setCsExpanded] = useState(() => {
    const saved = localStorage.getItem('aiPortfolioCsExpanded')
    return saved == null ? true : saved === 'true'
  })
  const [earningsCalendar, setEarningsCalendar] = useState(null)
  const [earningsExpanded, setEarningsExpanded] = useState(() => {
    return localStorage.getItem('aiPortfolioEarningsExpanded') === 'true'
  })
  const [riskData, setRiskData] = useState(null)
  const [riskExpanded, setRiskExpanded] = useState(() => {
    return localStorage.getItem('aiPortfolioRiskExpanded') === 'true'
  })
  // Global time-range selector — drives summary card, positions list,
  // and performance chart. Persisted in localStorage so the user's pick
  // survives page refreshes during a trading session.
  const [timeRange, setTimeRange] = useState(() => {
    const saved = localStorage.getItem('aiPortfolioTimeRange')
    return ['1d', '7d', '30d', 'all'].includes(saved) ? saved : 'all'
  })
  const [windowReturns, setWindowReturns] = useState(null)
  const [windowReturnsLoading, setWindowReturnsLoading] = useState(false)
  const [edge, setEdge] = useState(null)
  const [reconciliation, setReconciliation] = useState(null)
  // Window for the reconciliation card's LIVE exit set. Default post-fix
  // (since PARITY_FIX_DATE). Ref mirrors state so the polling fetchData
  // closure reads the CURRENT choice, not the one captured at effect setup.
  const [reconAllTime, setReconAllTime] = useState(false)
  const reconAllTimeRef = useRef(false)

  const reconSinceParam = (allTime) => (allTime ? null : PARITY_FIX_DATE)

  const handleReconWindowChange = async (allTime) => {
    setReconAllTime(allTime)
    reconAllTimeRef.current = allTime
    const data = await api
      .getAIPortfolioEdgeReconciliation(null, reconSinceParam(allTime))
      .catch(() => null)
    // Only apply if the user hasn't toggled again while this was in flight.
    if (reconAllTimeRef.current === allTime) setReconciliation(data)
  }

  const fetchData = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true)
      // A background refresh (showLoading=false, i.e. the 15s/5s pollers)
      // bypasses the 120s TTL for the price/cash-sensitive portfolio fetch —
      // otherwise the "waiting for trades" cash-change detector can't observe
      // a trade that executes inside the cache window.
      const portfolioOpts = showLoading ? undefined : { noCache: true }
      const [portfolioData, historyData, tradesData, csData, earningsData, riskInfo, edgeData, reconData] = await Promise.all([
        api.getAIPortfolio(portfolioOpts),
        // Since-inception fetch: 90 silently broke the chart's "All" range
        // once the portfolio outlived the window (start slid to ~Apr-15 at
        // $24.8k instead of the true $25k Mar-09 inception). resolution=auto
        // keeps the payload small: intraday for the trailing 7d (the 1d
        // view), last-snapshot-per-day beyond.
        api.getAIPortfolioHistory(3650, 'auto'),
        api.getAIPortfolioTrades(50),
        api.getCoiledSpringCandidates().catch(() => ({ candidates: [] })),
        api.getEarningsCalendar().catch(() => null),
        api.getPortfolioRisk().catch(() => null),
        api.getAIPortfolioEdge(EDGE_ALL_DAYS).catch(() => null),
        api.getAIPortfolioEdgeReconciliation(null, reconSinceParam(reconAllTimeRef.current)).catch(() => null),
      ])
      setPortfolio(portfolioData)
      setHistory(historyData)
      setTrades(tradesData)
      setCsAlerts(csData?.candidates || [])
      setEarningsCalendar(earningsData)
      setRiskData(riskInfo)
      setEdge(edgeData)
      setReconciliation(reconData)
      setLastUpdated(new Date())
      setError(null)

      // Check if data changed while waiting for trades
      if (waitingForTrades && waitingCash !== null) {
        const newCash = portfolioData?.summary?.cash
        if (Math.abs(newCash - waitingCash) > 100) {
          // Cash changed significantly - trades executed
          setWaitingForTrades(false)
          setWaitingCash(null)
        }
      }

      return portfolioData
    } catch (err) {
      console.error('Failed to fetch AI Portfolio:', err)
      setError(err?.message || 'Failed to load portfolio data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()

    // Auto-refresh every 15 seconds to catch background updates
    const interval = setInterval(() => {
      fetchData(false)
    }, 15000)

    return () => clearInterval(interval)
  }, [])

  // Keep polling while waiting for trades
  useEffect(() => {
    if (!waitingForTrades) return

    const pollInterval = setInterval(() => {
      fetchData(false)
    }, 5000) // Poll every 5 seconds while waiting

    // Stop waiting after 2 minutes max
    const timeout = setTimeout(() => {
      setWaitingForTrades(false)
      setWaitingCash(null)
    }, 120000)

    return () => {
      clearInterval(pollInterval)
      clearTimeout(timeout)
    }
  }, [waitingForTrades])

  // Auto-refresh prices every 5 minutes when enabled
  useEffect(() => {
    if (!autoRefresh) return

    const AUTO_REFRESH_INTERVAL = 5 * 60 * 1000 // 5 minutes
    const pendingTimeouts = []

    const refreshPrices = async () => {
      // Check if market is likely open (rough check - M-F 8:30am-4pm CST)
      const now = new Date()
      const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Chicago', hour: 'numeric', weekday: 'short', hour12: false }).formatToParts(now)
      const cstHour = parseInt(parts.find(p => p.type === 'hour')?.value || '0')
      const weekday = parts.find(p => p.type === 'weekday')?.value || 'Sun'
      const isWeekday = !['Sat', 'Sun'].includes(weekday)
      const isMarketHours = cstHour >= 8 && cstHour < 16

      if (!isWeekday || !isMarketHours) {
        // Market closed, skip auto-refresh
        return
      }

      setIsRefreshingPrices(true)
      try {
        await api.refreshAIPortfolio()
        setLastPriceRefresh(new Date())
        // Fetch updated data after refresh completes
        pendingTimeouts.push(setTimeout(() => fetchData(false), 10000))
        pendingTimeouts.push(setTimeout(() => setIsRefreshingPrices(false), 12000))
      } catch (err) {
        console.error('Auto-refresh failed:', err)
        setIsRefreshingPrices(false)
      }
    }

    // Refresh immediately on enable, then every 5 minutes
    refreshPrices()
    const interval = setInterval(refreshPrices, AUTO_REFRESH_INTERVAL)

    return () => {
      clearInterval(interval)
      pendingTimeouts.forEach(t => clearTimeout(t))
    }
  }, [autoRefresh])

  // Persist auto-refresh preference
  useEffect(() => {
    localStorage.setItem('aiPortfolioAutoRefresh', autoRefresh.toString())
  }, [autoRefresh])

  // Fetch window returns whenever the user picks a different window OR
  // the underlying portfolio refreshes (so per-position returns stay fresh
  // as current_price updates from background scans).
  useEffect(() => {
    let cancelled = false
    setWindowReturnsLoading(true)
    api.getAIPortfolioWindowReturns(timeRange)
      .then(data => { if (!cancelled) setWindowReturns(data) })
      .catch(err => {
        if (!cancelled) {
          console.error('Failed to fetch window returns:', err)
          setWindowReturns(null)  // Fall back to lifetime numbers on error
        }
      })
      .finally(() => { if (!cancelled) setWindowReturnsLoading(false) })
    return () => { cancelled = true }
    // `lastUpdated` triggers refetch when fetchData() completes — keeps
    // windowed positions in sync with the auto-refresh cycle.
  }, [timeRange, lastUpdated])

  // Persist time-range pick
  useEffect(() => {
    localStorage.setItem('aiPortfolioTimeRange', timeRange)
  }, [timeRange])

  // Persist collapse/expand state for the three top-of-page sections.
  // Keeping each as its own useEffect lets us re-use the same key shape
  // and stay obvious when greping.
  useEffect(() => {
    localStorage.setItem('aiPortfolioCsExpanded', String(csExpanded))
  }, [csExpanded])
  useEffect(() => {
    localStorage.setItem('aiPortfolioEarningsExpanded', String(earningsExpanded))
  }, [earningsExpanded])
  useEffect(() => {
    localStorage.setItem('aiPortfolioRiskExpanded', String(riskExpanded))
  }, [riskExpanded])

  const handleUpdateConfig = async (config) => {
    try {
      await api.updateAIPortfolioConfig(config)
      fetchData()
    } catch (err) {
      console.error('Failed to update config:', err)
      toast.error(err?.message || 'Failed to update config')
    }
  }

  const handleInitialize = async (startingCash = 25000) => {
    try {
      await api.initializeAIPortfolio(startingCash, portfolio?.config?.strategy)
      fetchData()
    } catch (err) {
      console.error('Failed to initialize:', err)
      toast.error(err?.message || 'Failed to initialize AI Portfolio')
    }
  }

  const handleRefresh = async () => {
    setIsRefreshingPrices(true)
    try {
      const result = await api.refreshAIPortfolio()
      setLastPriceRefresh(new Date())
      // Poll more frequently after triggering a refresh
      if (result.status === 'started') {
        setTimeout(() => fetchData(false), 4000)
        setTimeout(() => fetchData(false), 8000)
        setTimeout(() => fetchData(false), 12000)
        setTimeout(() => setIsRefreshingPrices(false), 12000)
      } else {
        fetchData()
        setIsRefreshingPrices(false)
      }
    } catch (err) {
      console.error('Failed to refresh:', err)
      setIsRefreshingPrices(false)
      toast.error(err?.message || 'Failed to refresh prices')
    }
  }

  const handleRunCycle = async () => {
    try {
      // Store current cash to detect when trades complete
      const currentCash = portfolio?.summary?.cash || 0
      setWaitingCash(currentCash)
      setWaitingForTrades(true)

      const result = await api.runAITradingCycle()
      if (result.status === 'market_closed' || result.status === 'busy') {
        toast.warning(result.message)
        setWaitingForTrades(false)
        fetchData()
      } else if (result.status !== 'started') {
        setWaitingForTrades(false)
        fetchData()
      }
    } catch (err) {
      console.error('Failed to run cycle:', err)
      setWaitingForTrades(false)
      toast.error(err?.message || 'Failed to run trading cycle')
    }
  }

  // ── Loading skeleton ────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-48 mb-5 rounded-lg" />
        <div className="skeleton h-48 rounded-xl mb-4" />
        <div className="skeleton h-32 rounded-xl mb-4" />
        <div className="skeleton h-48 rounded-xl" />
      </div>
    )
  }

  // ── Error state (only when we have no data at all to show — a failed
  // background poll on a populated page keeps rendering the stale data,
  // covered by the Data/Prices timestamps in the header) ──────────────
  if (error && !portfolio) {
    return (
      <div className="p-4 md:p-6">
        <Card variant="glass" className="text-center py-8">
          <div className="text-4xl mb-3">!</div>
          <div className="font-semibold text-dark-50 mb-2">Failed to Load Portfolio</div>
          <p className="text-dark-400 text-sm mb-4">{error}</p>
          <button onClick={() => fetchData()} className="btn-primary">Retry</button>
        </Card>
      </div>
    )
  }

  // ── Page render ─────────────────────────────────────────────────────
  return (
    <div className="p-4 md:p-6">
      {/* Page Header */}
      <PageHeader
        title="AI Portfolio"
        subtitle={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
            <span>Started: <span className="font-data">{formatCurrency(portfolio?.config?.starting_cash || 25000)}</span></span>
            {lastUpdated && (
              <span>Data: <span className="font-data">{formatTime(lastUpdated.toISOString())}</span></span>
            )}
            {lastPriceRefresh && (
              <span>Prices: <span className="font-data">{formatTime(lastPriceRefresh.toISOString())}</span></span>
            )}
          </span>
        }
        badge={
          portfolio?.config?.strategy && portfolio.config.strategy !== 'balanced'
            ? <TagBadge color={portfolio.config.strategy === 'growth' ? 'purple' : 'blue'}>
                {portfolio.config.strategy.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </TagBadge>
            : null
        }
      />

      {/* Auto-refresh toggle */}
      <Card variant="glass" className="mb-4" padding="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              className={`text-primary-400 ${isRefreshingPrices ? 'animate-spin' : ''}`}
            >
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
            <div>
              <div className="text-sm font-medium text-dark-100">Auto-Refresh Prices</div>
              <div className="text-[10px] text-dark-400">
                {autoRefresh ? 'Every 5 min during market hours' : 'Disabled'}
              </div>
            </div>
          </div>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              autoRefresh ? 'bg-emerald-500' : 'bg-dark-600'
            }`}
          >
            <span
              className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform duration-200 ${
                autoRefresh ? 'translate-x-6' : ''
              }`}
            />
          </button>
        </div>
      </Card>

      {/* Waiting for trades banner */}
      {waitingForTrades && (
        <Card variant="glass" className="mb-4 border-primary-500/30 bg-primary-500/5" padding="p-3">
          <div className="flex items-center gap-2 text-primary-400">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
            <span className="font-medium text-sm">Executing trades... This may take up to 2 minutes.</span>
          </div>
          <div className="text-dark-400 text-[10px] mt-1">Page will auto-update when complete.</div>
        </Card>
      )}

      {/* Coiled Spring Alerts */}
      <CoiledSpringSection
        csAlerts={csAlerts}
        csExpanded={csExpanded}
        setCsExpanded={setCsExpanded}
      />

      {/* Paper Mode is now indicated as a chip next to the AI Trading
          header inside ConfigPanel below — saves a full-width banner. */}

      {/* Risk Monitor */}
      <RiskMonitorSection
        riskData={riskData}
        riskExpanded={riskExpanded}
        setRiskExpanded={setRiskExpanded}
      />

      {/* Earnings Calendar */}
      <EarningsCalendarSection
        earningsCalendar={earningsCalendar}
        earningsExpanded={earningsExpanded}
        setEarningsExpanded={setEarningsExpanded}
      />

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-dark-700/50 overflow-x-auto">
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-2 text-xs font-semibold tracking-wider uppercase transition-colors whitespace-nowrap border-b-2 ${
              activeTab === tab.key
                ? 'text-primary-400 border-primary-500'
                : 'text-dark-400 border-transparent hover:text-dark-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <>
          {/* Cold-start framing for brand-new accounts (renders nothing once
              the portfolio has ≥2 trading days of history). */}
          <TrackRecordBanner edge={edge} />

          {/* SummaryCard hoisted to top so the time-range pills (embedded
              in its header) are visible immediately — no scrolling. */}
          <SummaryCard
            summary={portfolio?.summary}
            config={portfolio?.config}
            windowReturns={windowReturns}
            timeRange={timeRange}
            setTimeRange={setTimeRange}
            loading={windowReturnsLoading}
          />

          <PerformanceChart
            history={history}
            startingCash={portfolio?.config?.starting_cash || 25000}
            timeRange={timeRange}
          />

          <EdgeScorecard edge={edge} />

          <EdgeAttribution edge={edge} />

          <ExitReconciliation
            reconciliation={reconciliation}
            allTime={reconAllTime}
            onWindowChange={handleReconWindowChange}
          />

          <SectorAllocationChart
            riskData={riskData}
            cashPct={portfolio?.summary?.total_value > 0
              ? (portfolio.summary.cash / portfolio.summary.total_value) * 100
              : 0}
          />

          <ConfigPanel
            config={portfolio?.config}
            onUpdate={handleUpdateConfig}
            onInitialize={handleInitialize}
            onRefresh={handleRefresh}
            onRunCycle={handleRunCycle}
            waitingForTrades={waitingForTrades}
          />

          <PositionsList
            positions={portfolio?.positions}
            windowReturns={windowReturns}
            timeRange={timeRange}
            setTimeRange={setTimeRange}
            loading={windowReturnsLoading}
          />

          <TradeHistory trades={trades} />

          {/* Links */}
          <SectionLabel>More</SectionLabel>
          <div className="flex gap-4 mb-4">
            <Link to="/analytics" className="text-xs text-primary-400 hover:text-primary-300 transition-colors">
              Trade Analytics
            </Link>
            <Link to="/backtest" className="text-xs text-primary-400 hover:text-primary-300 transition-colors">
              Run Backtest
            </Link>
          </div>
        </>
      )}

      {activeTab === 'detail' && <PortfolioDetailView />}

      <div className="h-4" />
    </div>
  )
}
