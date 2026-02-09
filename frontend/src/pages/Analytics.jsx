import { useState, useEffect } from 'react'
import { api, formatCurrency } from '../api'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'

function StatCard({ label, value, subtext, color }) {
  return (
    <div className="bg-dark-700 rounded-lg p-3">
      <div className="text-dark-400 text-xs mb-1">{label}</div>
      <div className={`text-lg font-bold ${color || 'text-white'}`}>{value}</div>
      {subtext && <div className="text-dark-500 text-xs">{subtext}</div>}
    </div>
  )
}

function MonthlyPnLChart({ data }) {
  if (!data || data.length === 0) return null

  return (
    <div className="card mb-4">
      <h3 className="text-sm font-bold mb-3">Monthly P&L</h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <XAxis
              dataKey="month"
              tick={{ fontSize: 10, fill: '#8e8e93' }}
              tickFormatter={(m) => {
                const parts = m.split('-')
                return `${parts[1]}/${parts[0].slice(2)}`
              }}
            />
            <YAxis tick={{ fontSize: 10, fill: '#8e8e93' }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
            <Tooltip
              contentStyle={{ background: '#2c2c2e', border: 'none', borderRadius: '8px' }}
              formatter={(value) => [formatCurrency(value), 'P&L']}
            />
            <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.pnl >= 0 ? '#34c759' : '#ff3b30'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function SectorBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <div className="card mb-4">
      <h3 className="text-sm font-bold mb-3">Performance by Sector</h3>
      <div className="space-y-2">
        {data.map((s) => (
          <div key={s.sector} className="flex justify-between items-center py-1 border-b border-dark-700 last:border-0">
            <div>
              <span className="text-sm font-medium">{s.sector}</span>
              <span className="text-dark-400 text-xs ml-2">{s.trades} trades</span>
            </div>
            <div className="text-right">
              <span className={`text-sm font-medium ${s.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(s.pnl)}
              </span>
              <span className="text-dark-400 text-xs ml-2">{s.win_rate.toFixed(0)}% win</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function EntryTypeBreakdown({ data }) {
  if (!data || data.length === 0) return null

  return (
    <div className="card mb-4">
      <h3 className="text-sm font-bold mb-3">Performance by Entry Type</h3>
      <div className="space-y-2">
        {data.map((et) => (
          <div key={et.entry_type} className="flex justify-between items-center py-1 border-b border-dark-700 last:border-0">
            <div>
              <span className="text-sm font-medium capitalize">{et.entry_type.replace(/_/g, ' ')}</span>
              <span className="text-dark-400 text-xs ml-2">{et.trades} trades</span>
            </div>
            <div className="text-right">
              <span className={`text-sm font-medium ${et.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(et.pnl)}
              </span>
              <span className="text-dark-400 text-xs ml-2">{et.win_rate.toFixed(0)}% win</span>
            </div>
          </div>
        ))}
      </div>
    </div>
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
      <div className="p-4 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Trade Analytics</h1>
        <div className="text-dark-400 text-center py-12">Loading analytics...</div>
      </div>
    )
  }

  if (!analytics || !analytics.summary || !analytics.summary.total_trades) {
    return (
      <div className="p-4 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Trade Analytics</h1>
        <div className="card text-center text-dark-400 py-8">
          No trade data yet. Analytics will appear after the AI trader executes trades.
        </div>
      </div>
    )
  }

  const { summary, by_sector, monthly_pnl, by_entry_type } = analytics

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Trade Analytics</h1>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <StatCard
          label="Total Trades"
          value={summary.total_trades}
          subtext={`${summary.total_buys} buys, ${summary.total_sells} sells`}
        />
        <StatCard
          label="Win Rate"
          value={`${summary.win_rate.toFixed(0)}%`}
          color={summary.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="Profit Factor"
          value={summary.profit_factor === Infinity ? 'N/A' : summary.profit_factor.toFixed(2)}
          color={summary.profit_factor >= 1.5 ? 'text-green-400' : summary.profit_factor >= 1 ? 'text-yellow-400' : 'text-red-400'}
        />
        <StatCard
          label="Total Realized"
          value={formatCurrency(summary.total_realized)}
          color={summary.total_realized >= 0 ? 'text-green-400' : 'text-red-400'}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <StatCard
          label="Avg Winner"
          value={`+${summary.avg_gain_pct.toFixed(1)}%`}
          color="text-green-400"
        />
        <StatCard
          label="Avg Loser"
          value={`${summary.avg_loss_pct.toFixed(1)}%`}
          color="text-red-400"
        />
      </div>

      <MonthlyPnLChart data={monthly_pnl} />
      <SectorBreakdown data={by_sector} />
      <EntryTypeBreakdown data={by_entry_type} />

      <div className="h-4" />
    </div>
  )
}
