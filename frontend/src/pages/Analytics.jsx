import { useState, useEffect } from 'react'
import { api, formatCurrency, formatPercent, formatRelativeTime } from '../api'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, AreaChart, Area, ReferenceLine } from 'recharts'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import CollapsedDrawer from '../components/CollapsedDrawer'
import { tooltipStyle as TOOLTIP_STYLE, chartAxis, chartColors } from '../components/chartTheme'
import { buildCsv, downloadCsv } from '../csv'

const CSV_COLUMNS = [
  'executed_at', 'ticker', 'action', 'sector', 'shares', 'price',
  'total_value', 'cost_basis', 'realized_gain', 'holding_days',
  'canslim_score', 'entry_type', 'sell_reason', 'reason',
]

function downloadTradesCsv(trades) {
  const stamp = new Date().toISOString().slice(0, 10)
  downloadCsv(`trade-analytics-${stamp}.csv`, buildCsv(CSV_COLUMNS, trades))
}

const SELL_REASON_COLORS = {
  'STOP LOSS': chartColors.pnlDown,
  'TRAILING STOP': chartColors.brand,
  'PARTIAL TRAILING': '#eab308',
  'TAKE PROFIT': chartColors.pnlUp,
  'PARTIAL PROFIT': '#22c55e',
  'SCORE CRASH': '#f43f5e',
  'PROTECT GAINS': '#22d3ee',
  'WEAK POSITION': '#6b7280',
  'CIRCUIT BREAKER': '#a855f7',
}

function BreakdownRow({ label, trades, pnl, winRate, indicator }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-dark-700/30 last:border-0">
      <div className="flex items-center gap-2 min-w-0">
        {indicator}
        <span className="text-sm text-dark-100 font-medium truncate">{label}</span>
        <span className="text-[10px] text-dark-500 font-data">{trades} trades</span>
      </div>
      <div className="flex items-center gap-3 shrink-0">
        <PnlText value={pnl} prefix="$" className="text-sm font-medium" />
        <span className="text-[10px] text-dark-400 font-data w-12 text-right">{winRate.toFixed(0)}% win</span>
      </div>
    </div>
  )
}

function CumulativePnLChart({ data }) {
  if (!data || data.length < 2) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Cumulative Realized P&L" />
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="cumGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={chartColors.pnlUp} stopOpacity={0.3} />
                <stop offset="100%" stopColor={chartColors.pnlUp} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: chartAxis.tick, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(d) => {
                const parts = d.split('-')
                return parts.length >= 2 ? `${parts[1]}/${parts[2]}` : d
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: chartAxis.tick, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <ReferenceLine y={0} stroke={chartAxis.reference} strokeWidth={1} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value, name) => {
                if (name === 'cumulative') return [formatCurrency(value), 'Cumulative']
                return [value, name]
              }}
              labelFormatter={(label, payload) => {
                if (payload?.[0]?.payload) {
                  const p = payload[0].payload
                  return `${p.date} | ${p.ticker} ${formatCurrency(p.trade_pnl)}`
                }
                return label
              }}
            />
            <Area
              type="monotone"
              dataKey="cumulative"
              stroke={chartColors.pnlUp}
              strokeWidth={2}
              fill="url(#cumGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function MonthlyPnLChart({ data }) {
  if (!data || data.length === 0) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Monthly P&L" />
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis
              dataKey="month"
              tick={{ fontSize: 10, fill: chartAxis.tick, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(m) => {
                const parts = m.split('-')
                return parts.length >= 2 ? `${parts[1]}/${parts[0].slice(2)}` : m
              }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: chartAxis.tick, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => [formatCurrency(value), 'P&L']}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.pnl >= 0 ? chartColors.pnlUp : chartColors.pnlDown} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function RealizedVsUnrealized({ data }) {
  if (!data || !data.realized && !data.unrealized) return null

  const combined = data.combined || 0
  const realizedPct = data.combined ? ((data.realized / Math.abs(data.combined)) * 100) : 0

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Realized vs Unrealized P&L" />
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider mb-1">Realized</div>
          <PnlText value={data.realized} prefix="$" className="text-lg font-bold" />
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider mb-1">Unrealized</div>
          <PnlText value={data.unrealized} prefix="$" className="text-lg font-bold" />
          <div className="text-[10px] text-dark-500 mt-0.5">{data.open_positions} positions</div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider mb-1">Combined</div>
          <PnlText value={combined} prefix="$" className="text-lg font-bold" />
        </div>
      </div>

      {/* Stacked bar showing realized vs unrealized proportion */}
      {data.combined !== 0 && (
        <div className="h-3 rounded-full overflow-hidden flex bg-dark-800">
          <div
            className="h-full transition-all"
            style={{
              width: `${Math.abs(realizedPct).toFixed(0)}%`,
              backgroundColor: data.realized >= 0 ? '#10b981' : '#ef4444',
            }}
          />
          <div
            className="h-full transition-all"
            style={{
              width: `${(100 - Math.abs(realizedPct)).toFixed(0)}%`,
              backgroundColor: data.unrealized >= 0 ? '#10b98166' : '#ef444466',
            }}
          />
        </div>
      )}

      {/* Open positions breakdown */}
      {data.positions && data.positions.length > 0 && (
        <div className="mt-4 pt-3 border-t border-dark-700/30">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider mb-2">Open Positions</div>
          <div className="space-y-0">
            {data.positions.map((p) => (
              <div key={p.ticker} className="flex justify-between items-center py-1.5 border-b border-dark-700/20 last:border-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-dark-100 font-medium">{p.ticker}</span>
                  <span className="text-[10px] text-dark-500">{p.sector}</span>
                </div>
                <div className="flex items-center gap-3">
                  <PnlText value={p.pnl} prefix="$" className="text-sm" />
                  <span className={`text-[10px] font-data w-14 text-right ${p.return_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {p.return_pct >= 0 ? '+' : ''}{p.return_pct.toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-dark-500 w-10 text-right">{p.days_held}d</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

function TradeCard({ trade, rank }) {
  const isWin = trade.pnl >= 0
  return (
    <div className="flex justify-between items-center py-2 border-b border-dark-700/20 last:border-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[10px] text-dark-600 font-data w-4">{rank}</span>
        <span className="text-sm text-dark-100 font-medium">{trade.ticker}</span>
        <span className="text-[10px] text-dark-500">{trade.sector}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <PnlText value={trade.pnl} prefix="$" className="text-sm font-medium" />
        <span className={`text-[10px] font-data w-14 text-right ${isWin ? 'text-emerald-400' : 'text-red-400'}`}>
          {formatPercent(trade.return_pct, true)}
        </span>
      </div>
    </div>
  )
}

function BestWorstTrades({ best, worst }) {
  if ((!best || best.length === 0) && (!worst || worst.length === 0)) return null

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
      {best && best.length > 0 && (
        <Card variant="glass">
          <CardHeader title="Best Trades" />
          <div className="space-y-0">
            {best.map((t, i) => (
              <TradeCard key={`${t.ticker}-${t.date}`} trade={t} rank={i + 1} />
            ))}
          </div>
        </Card>
      )}
      {worst && worst.length > 0 && (
        <Card variant="glass">
          <CardHeader title="Worst Trades" />
          <div className="space-y-0">
            {worst.map((t, i) => (
              <TradeCard key={`${t.ticker}-${t.date}`} trade={t} rank={i + 1} />
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

// Drawer header badge: "<n> <noun> · <P&L leader>" — the glanceable signal
// the UI grammar requires on demoted sections.
function BreakdownBadge({ data, noun, labelOf }) {
  const leader = data.reduce((a, b) => (b.pnl > a.pnl ? b : a))
  return (
    <span className="text-[10px] font-data text-dark-400">
      {data.length} {noun}{data.length !== 1 ? 's' : ''} ·{' '}
      <span className={leader.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
        {labelOf(leader)} {leader.pnl >= 0 ? '+' : '−'}{formatCurrency(Math.abs(leader.pnl))}
      </span>
    </span>
  )
}

function SectorBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <CollapsedDrawer
      title="Performance by Sector"
      cardProps={{ className: 'mb-4' }}
      badge={<BreakdownBadge data={data} noun="sector" labelOf={s => s.sector} />}
    >
      <div className="space-y-0">
        {data.map((s) => (
          <BreakdownRow
            key={s.sector}
            label={s.sector}
            trades={s.trades}
            pnl={s.pnl}
            winRate={s.win_rate}
          />
        ))}
      </div>
    </CollapsedDrawer>
  )
}

function EntryTypeBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <CollapsedDrawer
      title="Performance by Entry Type"
      cardProps={{ className: 'mb-4' }}
      badge={<BreakdownBadge data={data} noun="type" labelOf={et => et.entry_type.replace(/_/g, ' ')} />}
    >
      <div className="space-y-0">
        {data.map((et) => (
          <BreakdownRow
            key={et.entry_type}
            label={et.entry_type.replace(/_/g, ' ')}
            trades={et.trades}
            pnl={et.pnl}
            winRate={et.win_rate}
          />
        ))}
      </div>
    </CollapsedDrawer>
  )
}

function SellReasonBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <CollapsedDrawer
      title="Performance by Sell Reason"
      cardProps={{ className: 'mb-4' }}
      badge={<BreakdownBadge data={data} noun="reason" labelOf={sr => sr.sell_reason} />}
    >
      <div className="space-y-0">
        {data.map((sr) => (
          <BreakdownRow
            key={sr.sell_reason}
            label={sr.sell_reason}
            trades={sr.trades}
            pnl={sr.pnl}
            winRate={sr.win_rate}
            indicator={
              <div
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: SELL_REASON_COLORS[sr.sell_reason] || '#6b7280' }}
              />
            }
          />
        ))}
      </div>
    </CollapsedDrawer>
  )
}

function HoldDurationBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <CollapsedDrawer
      title="Performance by Hold Duration"
      cardProps={{ className: 'mb-4' }}
      badge={<BreakdownBadge data={data} noun="bucket" labelOf={hd => hd.duration} />}
    >
      <div className="space-y-0">
        {data.map((hd) => (
          <BreakdownRow
            key={hd.duration}
            label={hd.duration}
            trades={hd.trades}
            pnl={hd.pnl}
            winRate={hd.win_rate}
          />
        ))}
      </div>
    </CollapsedDrawer>
  )
}

function AnalyticsOverview() {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await api.getTradeAnalytics()
        setAnalytics(data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch analytics:', err)
        setError(err.message || 'Failed to load analytics')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="p-4 md:p-6 max-w-4xl mx-auto">
        <PageHeader title="Trade Analytics" />
        <div className="space-y-3">
          <div className="skeleton h-24 rounded-2xl" />
          <div className="skeleton h-48 rounded-2xl" />
          <div className="skeleton h-40 rounded-2xl" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 md:p-6 max-w-4xl mx-auto">
        <PageHeader title="Trade Analytics" />
        <Card variant="glass" className="text-center py-8">
          <div className="text-red-400 text-sm mb-2">Failed to load analytics</div>
          <div className="text-dark-500 text-xs">{error}</div>
        </Card>
      </div>
    )
  }

  if (!analytics || !analytics.summary || !analytics.summary.total_trades) {
    return (
      <div className="p-4 md:p-6 max-w-4xl mx-auto">
        <PageHeader title="Trade Analytics" />
        <Card variant="glass" className="text-center py-8">
          <div className="text-dark-400 text-sm">
            No trade data yet. Analytics will appear after the AI trader executes trades.
          </div>
        </Card>
      </div>
    )
  }

  const {
    summary, by_sector, monthly_pnl, by_entry_type, by_sell_reason,
    by_hold_duration, cumulative_pnl, best_trades, worst_trades,
    realized_vs_unrealized, as_of, trades,
  } = analytics

  const exportable = Array.isArray(trades) && trades.length > 0

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto">
      <PageHeader
        title="Trade Analytics"
        subtitle={as_of ? `Updated ${formatRelativeTime(as_of)}` : undefined}
        actions={exportable ? (
          <button
            onClick={() => downloadTradesCsv(trades)}
            className="text-xs text-dark-300 hover:text-dark-100 px-3 py-1 rounded border border-dark-700 hover:border-dark-600 transition-colors"
            title={`Export ${trades.length} trades as CSV`}
          >
            Export CSV
          </button>
        ) : null}
      />

      {/* Summary Stats */}
      <SectionLabel>Overview</SectionLabel>
      <Card variant="glass" className="mb-4">
        <StatGrid
          columns={4}
          stats={[
            {
              label: 'Total Trades',
              value: summary.total_trades,
              sublabel: `${summary.total_buys} buys / ${summary.total_sells} sells`,
            },
            {
              label: 'Win Rate',
              value: `${summary.win_rate.toFixed(0)}%`,
              color: summary.win_rate >= 50 ? 'text-emerald-400' : 'text-red-400',
            },
            {
              label: 'Profit Factor',
              value: summary.profit_factor == null ? 'N/A' : summary.profit_factor.toFixed(2),
              color: summary.profit_factor == null ? 'text-emerald-400' : summary.profit_factor >= 1.5 ? 'text-emerald-400' : summary.profit_factor >= 1 ? 'text-amber-400' : 'text-red-400',
            },
            {
              label: 'Total Realized',
              value: formatCurrency(summary.total_realized),
              color: summary.total_realized >= 0 ? 'text-emerald-400' : 'text-red-400',
            },
          ]}
        />
      </Card>

      <Card variant="glass" className="mb-4">
        <StatGrid
          columns={2}
          stats={[
            {
              label: 'Avg Winner',
              value: `+${summary.avg_gain_pct.toFixed(1)}%`,
              color: 'text-emerald-400',
            },
            {
              label: 'Avg Loser',
              value: `${summary.avg_loss_pct.toFixed(1)}%`,
              color: 'text-red-400',
            },
          ]}
        />
      </Card>

      {/* Realized vs Unrealized */}
      {realized_vs_unrealized && <RealizedVsUnrealized data={realized_vs_unrealized} />}

      {/* Equity Curve */}
      <SectionLabel>Performance</SectionLabel>
      <CumulativePnLChart data={cumulative_pnl} />
      <MonthlyPnLChart data={monthly_pnl} />

      {/* Best / Worst */}
      <BestWorstTrades best={best_trades} worst={worst_trades} />

      {/* Breakdowns */}
      <SectionLabel>Breakdowns</SectionLabel>
      <SectorBreakdown data={by_sector} />
      <EntryTypeBreakdown data={by_entry_type} />
      <SellReasonBreakdown data={by_sell_reason} />
      <HoldDurationBreakdown data={by_hold_duration} />

      <div className="h-4" />
    </div>
  )
}


// ── Tabbed shell (2026-07-22 UI consolidation) ───────────────────────
// Trade Journal was a separate page duplicating this page's subject
// (post-hoc trade analysis) with its own nav slot. It now lives here as
// a tab; /trade-journal redirects to /analytics. Journal keeps its URL
// filter params (?days=&ticker=...) — same router context, same page.
import TradeJournal from './TradeJournal'

const ANALYTICS_TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'journal', label: 'Journal' },
]

export default function Analytics() {
  const [tab, setTab] = useState('overview')
  return (
    <div>
      <div className="flex gap-1 px-4 md:px-6 pt-4 max-w-4xl mx-auto">
        {ANALYTICS_TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            aria-pressed={tab === t.key}
            className={`px-3 py-1 text-xs rounded-md border transition-colors ${
              tab === t.key
                ? 'bg-primary-500/15 text-primary-300 border-primary-500/30'
                : 'text-dark-400 border-dark-700/50 hover:text-dark-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tab === 'overview' ? <AnalyticsOverview /> : (
        <div className="max-w-4xl mx-auto px-4 py-4">
          <TradeJournal embedded />
        </div>
      )}
    </div>
  )
}
