import { useState, useEffect } from 'react'
import { api, formatCurrency } from '../api'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'

const SELL_REASON_COLORS = {
  'STOP LOSS': '#ef4444',
  'TRAILING STOP': '#f59e0b',
  'PARTIAL TRAILING': '#eab308',
  'TAKE PROFIT': '#10b981',
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
              tick={{ fontSize: 10, fill: '#6b7280', fontFamily: 'JetBrains Mono' }}
              tickFormatter={(m) => {
                const parts = m.split('-')
                return `${parts[1]}/${parts[0].slice(2)}`
              }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280', fontFamily: 'JetBrains Mono' }}
              tickFormatter={(v) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                background: '#14141f',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '10px',
                fontFamily: 'JetBrains Mono',
                fontSize: 12,
              }}
              formatter={(value) => [formatCurrency(value), 'P&L']}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.pnl >= 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function SectorBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Performance by Sector" />
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
    </Card>
  )
}

function EntryTypeBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Performance by Entry Type" />
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
    </Card>
  )
}

function SellReasonBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Performance by Sell Reason" />
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
    </Card>
  )
}

function HoldDurationBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Performance by Hold Duration" />
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
    </Card>
  )
}

export default function Analytics() {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await api.getTradeAnalytics()
        setAnalytics(data)
      } catch (err) {
        console.error('Failed to fetch analytics:', err)
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

  const { summary, by_sector, monthly_pnl, by_entry_type, by_sell_reason, by_hold_duration } = analytics

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto">
      <PageHeader title="Trade Analytics" />

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
              value: summary.profit_factor === Infinity ? 'N/A' : summary.profit_factor.toFixed(2),
              color: summary.profit_factor >= 1.5 ? 'text-emerald-400' : summary.profit_factor >= 1 ? 'text-amber-400' : 'text-red-400',
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

      <SectionLabel>Breakdowns</SectionLabel>
      <MonthlyPnLChart data={monthly_pnl} />
      <SectorBreakdown data={by_sector} />
      <EntryTypeBreakdown data={by_entry_type} />
      <SellReasonBreakdown data={by_sell_reason} />
      <HoldDurationBreakdown data={by_hold_duration} />

      <div className="h-4" />
    </div>
  )
}
