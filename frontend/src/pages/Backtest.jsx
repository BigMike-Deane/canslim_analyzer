import { useState, useEffect, useRef, useCallback } from 'react'
import { api, formatCurrency, APIError } from '../api'
import { Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, Area, AreaChart, ReferenceLine, ComposedChart } from 'recharts'
import Card, { CardHeader } from '../components/Card'
import { StatusBadge, ActionBadge, TagBadge, PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import DataTable from '../components/DataTable'
import PageHeader from '../components/PageHeader'

function PerformanceChart({ data, startingCash }) {
  if (!data || data.length < 2) {
    return (
      <Card variant="glass" className="mb-4 h-48 flex items-center justify-center text-dark-400">
        No chart data available
      </Card>
    )
  }

  const finalReturn = data[data.length - 1]?.return_pct || 0
  const finalSpy = data[data.length - 1]?.spy_return_pct || 0
  const isPositive = finalReturn >= 0
  const beatsSpy = finalReturn >= finalSpy
  const strokeColor = isPositive ? '#34d399' : '#f87171'
  const spyColor = '#f59e0b' // Amber — high contrast on dark bg

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const port = payload.find(p => p.dataKey === 'return_pct')
    const spy = payload.find(p => p.dataKey === 'spy_return_pct')
    const pv = port?.value ?? 0
    const sv = spy?.value ?? 0
    const spread = pv - sv
    return (
      <div className="bg-dark-900 border border-dark-700/60 rounded-xl px-3 py-2 shadow-lg">
        <div className="text-[10px] text-dark-500 mb-1">{label}</div>
        <div className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full" style={{ background: strokeColor }} />
          <span className="text-dark-300">Portfolio</span>
          <span className={`font-data ml-auto ${pv >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{pv >= 0 ? '+' : ''}{pv.toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full" style={{ background: spyColor }} />
          <span className="text-dark-300">SPY</span>
          <span className="font-data ml-auto text-amber-400">{sv >= 0 ? '+' : ''}{sv.toFixed(1)}%</span>
        </div>
        <div className="border-t border-dark-700/40 mt-1.5 pt-1.5 flex items-center gap-2 text-xs">
          <span className="text-dark-500">vs SPY</span>
          <span className={`font-data font-semibold ml-auto ${spread >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>{spread >= 0 ? '+' : ''}{spread.toFixed(1)}pp</span>
        </div>
      </div>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <span className="text-xs font-semibold text-dark-200">Portfolio vs SPY</span>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 rounded" style={{ background: strokeColor }} /> Portfolio</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 rounded" style={{ background: spyColor }} /> SPY</span>
          <span className={`font-data font-semibold ${beatsSpy ? 'text-emerald-400' : 'text-red-400'}`}>
            {beatsSpy ? '+' : ''}{(finalReturn - finalSpy).toFixed(1)}pp
          </span>
        </div>
      </div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <defs>
              <linearGradient id="portGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={strokeColor} stopOpacity={0.20} />
                <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="spyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={spyColor} stopOpacity={0.12} />
                <stop offset="100%" stopColor={spyColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="spy_return_pct"
              name="SPY"
              stroke={spyColor}
              strokeWidth={2}
              fill="url(#spyGrad)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="return_pct"
              name="Portfolio"
              stroke={strokeColor}
              strokeWidth={2.5}
              fill="url(#portGrad)"
              dot={false}
            />
            <ReferenceLine y={0} stroke="#374151" strokeWidth={1} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#6b7280' }}
              tickFormatter={(d) => {
                const date = new Date(d)
                return `${date.getMonth()+1}/${date.getDate()}`
              }}
              interval="preserveStartEnd"
              axisLine={{ stroke: '#1f2937' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6b7280' }}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              domain={['dataMin - 5', 'dataMax + 5']}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function BacktestForm({ onSubmit, isLoading }) {
  const today = new Date().toISOString().split('T')[0]
  const oneYearAgo = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  const [config, setConfig] = useState({
    start_date: oneYearAgo,
    end_date: today,
    starting_cash: 25000,
    stock_universe: 'sp500',
    strategy: 'balanced'
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(config)
  }

  const inputCls = 'w-full bg-dark-850 border border-dark-700/50 rounded-lg px-3 py-2 text-sm font-data text-dark-100 focus:outline-none focus:border-primary-500/50 transition-colors'
  const labelCls = 'text-[10px] font-semibold tracking-wider uppercase text-dark-400 block mb-1'

  return (
    <Card variant="glass" className="mb-4" padding="p-5">
      <form onSubmit={handleSubmit}>
        <CardHeader title="Run New Backtest" />

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <label className={labelCls}>Start Date</label>
            <input
              type="date"
              value={config.start_date}
              onChange={(e) => setConfig({...config, start_date: e.target.value})}
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>End Date</label>
            <input
              type="date"
              value={config.end_date}
              onChange={(e) => setConfig({...config, end_date: e.target.value})}
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Starting Cash</label>
            <input
              type="number"
              value={config.starting_cash}
              onChange={(e) => setConfig({...config, starting_cash: parseFloat(e.target.value) || 25000})}
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Stock Universe</label>
            <select
              value={config.stock_universe}
              onChange={(e) => setConfig({...config, stock_universe: e.target.value})}
              className={inputCls}
            >
              <option value="sp500">S&P 500</option>
              <option value="all">All Stocks (~2000)</option>
            </select>
          </div>
          <div>
            <label className={labelCls}>Strategy</label>
            <select
              value={config.strategy}
              onChange={(e) => setConfig({...config, strategy: e.target.value})}
              className={inputCls}
            >
              <option value="balanced">Balanced (Default)</option>
              <option value="growth">Growth Mode</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-primary-500 hover:bg-primary-400 disabled:bg-dark-700 disabled:text-dark-500 rounded-lg py-2.5 text-sm font-semibold transition-colors"
        >
          {isLoading ? 'Starting...' : 'Start Backtest'}
        </button>
      </form>
    </Card>
  )
}

function ComparisonView({ comparison, onClose }) {
  if (!comparison) return null

  const { backtests, chart_data, stats_table } = comparison
  const colors = ['#34d399', '#22d3ee', '#fbbf24', '#f87171', '#a78bfa']
  const gradientIds = backtests.map((_, i) => `compGrad${i}`)

  return (
    <div className="space-y-4">
      <PageHeader
        title="Backtest Comparison"
        actions={
          <button onClick={onClose} className="text-xs text-dark-400 hover:text-dark-200 transition-colors">
            Close
          </button>
        }
      />

      {/* Overlaid chart */}
      <Card variant="glass">
        <CardHeader title="Return % Over Time" />
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chart_data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <defs>
                {backtests.map((_, i) => (
                  <linearGradient key={i} id={gradientIds[i]} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={colors[i % colors.length]} stopOpacity={0.10} />
                    <stop offset="100%" stopColor={colors[i % colors.length]} stopOpacity={0} />
                  </linearGradient>
                ))}
                <linearGradient id="compSpyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.10} />
                  <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="spy_return" name="SPY" stroke="#f59e0b" strokeWidth={2} fill="url(#compSpyGrad)" dot={false} connectNulls />
              {backtests.map((bt, i) => (
                <Area
                  key={bt.id}
                  type="monotone"
                  dataKey={`bt_${bt.id}_return`}
                  name={bt.name}
                  stroke={colors[i % colors.length]}
                  strokeWidth={2}
                  fill={`url(#${gradientIds[i]})`}
                  dot={false}
                  connectNulls
                />
              ))}
              <ReferenceLine y={0} stroke="#374151" strokeWidth={1} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(d) => { const p = d.split('-'); return `${p[1]}/${p[2]}` }} interval="preserveStartEnd" axisLine={{ stroke: '#1f2937' }} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} tickFormatter={(v) => `${v?.toFixed(0)}%`} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#14141f', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px' }}
                labelStyle={{ color: '#6b7280', fontSize: 11 }}
                formatter={(v, n) => [`${v?.toFixed(2)}%`, n]}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Stats comparison table */}
      <Card variant="glass">
        <CardHeader title="Stats Comparison" />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-dark-700/50">
                <th className="text-left py-2 px-2 text-[10px] font-semibold tracking-wider uppercase text-dark-400">Metric</th>
                {backtests.map((bt) => <th key={bt.id} className="text-right py-2 px-2 text-[10px] font-semibold tracking-wider uppercase text-dark-400">{bt.name}</th>)}
              </tr>
            </thead>
            <tbody>
              {[
                ['Return', 'total_return_pct', (v) => `${v >= 0 ? '+' : ''}${v?.toFixed(1)}%`, true],
                ['SPY Return', 'spy_return_pct', (v) => `${v >= 0 ? '+' : ''}${v?.toFixed(1)}%`, false],
                ['Max Drawdown', 'max_drawdown_pct', (v) => `-${v?.toFixed(1)}%`, false],
                ['Sharpe', 'sharpe_ratio', (v) => v?.toFixed(2), true],
                ['Win Rate', 'win_rate', (v) => `${v?.toFixed(0)}%`, true],
                ['Trades', 'total_trades', (v) => v, false],
              ].map(([label, key, fmt, highlight]) => {
                const values = stats_table[key] || []
                const best = highlight ? Math.max(...values.filter(v => v != null)) : null
                return (
                  <tr key={key} className="border-b border-dark-700/20 last:border-0 hover:bg-dark-800/40">
                    <td className="py-2 px-2 text-dark-400 text-xs">{label}</td>
                    {values.map((v, i) => (
                      <td key={i} className={`text-right py-2 px-2 font-data font-medium text-sm ${highlight && v === best ? 'text-emerald-400' : 'text-dark-100'}`}>
                        {v != null ? fmt(v) : 'N/A'}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BacktestList({ backtests, onSelect, onDelete, onCancel, onCompare }) {
  const [selected, setSelected] = useState(new Set())

  if (!backtests || backtests.length === 0) {
    return (
      <Card variant="glass" className="text-center text-dark-400 py-8">
        No backtests yet. Run one above!
      </Card>
    )
  }

  const toggleSelect = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const columns = [
    {
      key: 'select',
      label: '',
      render: (_, row) => row.status === 'completed' ? (
        <input
          type="checkbox"
          checked={selected.has(row.id)}
          onChange={(e) => { e.stopPropagation(); toggleSelect(row.id) }}
          onClick={(e) => e.stopPropagation()}
          className="accent-primary-500"
        />
      ) : null
    },
    {
      key: 'name',
      label: 'Name',
      sortable: true,
      render: (_, row) => (
        <div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-dark-100">{row.name}</span>
            <StatusBadge status={row.status} />
            {row.strategy === 'growth' && <TagBadge color="purple">Growth</TagBadge>}
          </div>
          <div className="text-dark-500 text-[10px] font-data mt-0.5">
            {row.start_date} to {row.end_date}
          </div>
          {(row.status === 'running' || row.status === 'pending') && (
            <div className="mt-1.5">
              <div className="h-1 bg-dark-700/50 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 rounded-full transition-all duration-300"
                  style={{ width: `${row.progress_pct || 0}%` }}
                />
              </div>
              <div className="flex items-center justify-between mt-1">
                <span className="text-[10px] text-dark-500 font-data">{(row.progress_pct || 0).toFixed(0)}%</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onCancel(row.id)
                  }}
                  className="text-[10px] text-red-400 hover:text-red-300 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )
    },
    {
      key: 'total_return_pct',
      label: 'Return',
      align: 'right',
      sortable: true,
      mono: true,
      render: (_, row) => {
        if (row.status !== 'completed') return null
        const vsSpy = row.total_return_pct - row.spy_return_pct
        return (
          <div>
            <span className={`font-data font-semibold text-sm ${vsSpy >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {row.total_return_pct >= 0 ? '+' : ''}{row.total_return_pct?.toFixed(1)}%
            </span>
            <div className={`text-[10px] font-data mt-0.5 ${vsSpy >= 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
              vs SPY {vsSpy >= 0 ? '+' : ''}{vsSpy?.toFixed(1)}%
            </div>
          </div>
        )
      }
    },
    {
      key: 'actions',
      label: '',
      align: 'right',
      render: (_, row) => (row.status === 'completed' || row.status === 'failed' || row.status === 'cancelled') ? (
        <button
          onClick={(e) => {
            e.stopPropagation()
            onDelete(row.id)
          }}
          className="text-dark-500 hover:text-red-400 transition-colors text-xs p-1"
          title="Delete backtest"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      ) : null
    }
  ]

  return (
    <Card variant="glass">
      <CardHeader
        title="Previous Backtests"
        action={
          selected.size >= 2 && (
            <button
              onClick={() => onCompare([...selected])}
              className="text-[10px] bg-primary-500/15 text-primary-400 border border-primary-500/20 hover:bg-primary-500/25 px-3 py-1.5 rounded-lg font-semibold transition-colors"
            >
              Compare Selected ({selected.size})
            </button>
          )
        }
      />
      <DataTable
        columns={columns}
        data={backtests}
        sortable={false}
        onRowClick={(row) => row.status === 'completed' && onSelect(row.id)}
        emptyMessage="No backtests yet. Run one above!"
      />
    </Card>
  )
}

function BacktestResults({ backtest, onClose }) {
  const { backtest: bt, performance_chart, trades, statistics } = backtest

  const vsSpy = bt.total_return_pct - bt.spy_return_pct

  const summaryStats = [
    {
      label: 'Total Return',
      value: `${bt.total_return_pct >= 0 ? '+' : ''}${bt.total_return_pct?.toFixed(1)}%`,
      sublabel: formatCurrency(bt.final_value),
      color: vsSpy >= 0 ? 'text-emerald-400' : 'text-red-400',
    },
    {
      label: 'vs SPY',
      value: `${vsSpy >= 0 ? '+' : ''}${vsSpy?.toFixed(1)}%`,
      sublabel: `SPY: ${bt.spy_return_pct?.toFixed(1)}%`,
      color: vsSpy >= 0 ? 'text-emerald-400' : 'text-red-400',
    },
    {
      label: 'Max Drawdown',
      value: `-${bt.max_drawdown_pct?.toFixed(1)}%`,
      color: 'text-red-400',
    },
    {
      label: 'Win Rate',
      value: `${bt.win_rate?.toFixed(0)}%`,
      sublabel: `${bt.total_trades} trades`,
    },
  ]

  const detailStats = [
    { label: 'Sharpe Ratio', value: bt.sharpe_ratio?.toFixed(2) || 'N/A' },
    { label: 'Best Trade', value: `+${statistics.best_trade?.toFixed(1)}%`, color: 'text-emerald-400' },
    { label: 'Worst Trade', value: `${statistics.worst_trade?.toFixed(1)}%`, color: 'text-red-400' },
  ]

  const tradeColumns = [
    {
      key: 'date',
      label: 'Date',
      mono: true,
      className: 'text-dark-400',
    },
    {
      key: 'ticker',
      label: 'Ticker',
      render: (v) => <span className="font-medium text-dark-100">{v}</span>,
    },
    {
      key: 'action',
      label: 'Action',
      render: (v) => <ActionBadge action={v} />,
    },
    {
      key: 'price',
      label: 'Price',
      align: 'right',
      mono: true,
      render: (v) => `$${v?.toFixed(2)}`,
    },
    {
      key: 'gain_pct',
      label: 'P/L',
      align: 'right',
      render: (v, row) => row.action === 'SELL'
        ? <PnlText value={v} className="text-sm" />
        : <span className="text-dark-500">-</span>,
    },
    {
      key: 'reason',
      label: 'Reason',
      className: 'text-dark-400 text-xs max-w-[150px] truncate',
    },
  ]

  return (
    <div className="space-y-4">
      <PageHeader
        title={bt.name}
        actions={
          <button onClick={onClose} className="text-xs text-dark-400 hover:text-dark-200 transition-colors">
            Close
          </button>
        }
      />

      {/* Summary Stats */}
      <Card variant="glass">
        <StatGrid stats={summaryStats} columns={4} />
      </Card>

      {/* Additional stats */}
      <Card variant="glass">
        <StatGrid stats={detailStats} columns={3} />
      </Card>

      {/* Performance Chart */}
      <PerformanceChart data={performance_chart} startingCash={bt.starting_cash} />

      {/* Trade History */}
      <Card variant="glass">
        <CardHeader title={`Trade History (${trades.length})`} />
        <DataTable
          columns={tradeColumns}
          data={trades.slice(0, 50)}
          sortable={false}
          compact
        />
        {trades.length > 50 && (
          <div className="text-center text-dark-500 text-[10px] py-2 mt-2">
            Showing first 50 of {trades.length} trades
          </div>
        )}
      </Card>
    </div>
  )
}

function MultiPeriodPanel({ onLaunch, isLoading }) {
  const [presets, setPresets] = useState([])
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    api.getBacktestPresets().then(setPresets).catch(() => {})
  }, [])

  if (!presets.length) return null

  return (
    <Card variant="accent" accent="purple" className="mb-4">
      <button onClick={() => setExpanded(!expanded)} className="w-full flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm text-dark-100">Multi-Period Backtesting</span>
          <TagBadge color="purple">{presets.length} periods</TagBadge>
        </div>
        <svg
          width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          className={`text-dark-400 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
        >
          <path d="M6 9l6 6 6-6" strokeLinecap="round" />
        </svg>
      </button>
      {expanded && (
        <div className="mt-3">
          <p className="text-dark-400 text-xs mb-3">
            Test the strategy across different market regimes simultaneously.
          </p>
          <div className="space-y-1 mb-3">
            {presets.map((p, i) => (
              <div key={i} className="flex justify-between text-xs py-1">
                <span className="font-medium text-dark-200">{p.name}</span>
                <span className="text-dark-500 font-data">{p.desc}</span>
              </div>
            ))}
          </div>
          <button
            onClick={onLaunch}
            disabled={isLoading}
            className="w-full bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 text-purple-300 disabled:bg-dark-700 disabled:text-dark-500 disabled:border-dark-700 rounded-lg py-2 text-sm font-semibold transition-colors"
          >
            {isLoading ? 'Starting...' : `Launch ${presets.length} Backtests`}
          </button>
        </div>
      )}
    </Card>
  )
}

export default function Backtest() {
  const [backtests, setBacktests] = useState([])
  const [selectedBacktest, setSelectedBacktest] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const pollingRef = useRef(null)

  const fetchBacktests = useCallback(async () => {
    try {
      const data = await api.getBacktests()
      setBacktests(data)
      return data
    } catch (err) {
      console.error('Failed to fetch backtests:', err)
      return []
    }
  }, [])

  // Fetch backtests on load
  useEffect(() => {
    fetchBacktests()
  }, [fetchBacktests])

  // Poll for running backtests
  useEffect(() => {
    const hasRunning = backtests.some(b => b.status === 'running' || b.status === 'pending')

    if (hasRunning && !pollingRef.current) {
      pollingRef.current = setInterval(async () => {
        const data = await fetchBacktests()
        const stillRunning = data.some(b => b.status === 'running' || b.status === 'pending')
        if (!stillRunning && pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }
      }, 2000)
    } else if (!hasRunning && pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
        pollingRef.current = null
      }
    }
  }, [backtests, fetchBacktests])

  const startBacktest = async (config) => {
    setIsLoading(true)
    setError(null)
    try {
      await api.createBacktest(config)
      fetchBacktests()
    } catch (err) {
      setError(err instanceof APIError ? err.message : 'Failed to start backtest')
    } finally {
      setIsLoading(false)
    }
  }

  const selectBacktest = async (id) => {
    try {
      const data = await api.getBacktest(id)
      setSelectedBacktest(data)
    } catch (err) {
      console.error('Failed to fetch backtest:', err)
    }
  }

  const deleteBacktest = async (id) => {
    try {
      await api.deleteBacktest(id)
      setBacktests(backtests.filter(b => b.id !== id))
      if (selectedBacktest?.backtest?.id === id) {
        setSelectedBacktest(null)
      }
    } catch (err) {
      console.error('Failed to delete backtest:', err)
    }
  }

  const cancelBacktest = async (id) => {
    try {
      await api.cancelBacktest(id)
      fetchBacktests()
    } catch (err) {
      console.error('Failed to cancel backtest:', err)
      setError(err.message || 'Failed to cancel backtest')
    }
  }

  const handleCompare = async (ids) => {
    try {
      const data = await api.compareBacktests(ids)
      setComparison(data)
    } catch (err) {
      setError(err.message || 'Failed to compare backtests')
    }
  }

  const handleMultiPeriod = async () => {
    setIsLoading(true)
    setError(null)
    try {
      await api.createMultiBacktest({ starting_cash: 25000, stock_universe: 'all' })
      fetchBacktests()
    } catch (err) {
      setError(err.message || 'Failed to start multi-period backtest')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="p-4 md:p-6 max-w-4xl mx-auto">
      <PageHeader
        title="CANSLIM Backtesting"
        subtitle="Test the AI trading strategy against historical data to see how it would have performed."
      />

      {error && (
        <Card variant="accent" accent="red" className="mb-4" padding="p-3">
          <p className="text-red-400 text-sm">{error}</p>
        </Card>
      )}

      {comparison ? (
        <ComparisonView comparison={comparison} onClose={() => setComparison(null)} />
      ) : selectedBacktest ? (
        <BacktestResults
          backtest={selectedBacktest}
          onClose={() => setSelectedBacktest(null)}
        />
      ) : (
        <>
          <BacktestForm onSubmit={startBacktest} isLoading={isLoading} />
          <MultiPeriodPanel onLaunch={handleMultiPeriod} isLoading={isLoading} />
          <BacktestList
            backtests={backtests}
            onSelect={selectBacktest}
            onDelete={deleteBacktest}
            onCancel={cancelBacktest}
            onCompare={handleCompare}
          />
        </>
      )}
    </div>
  )
}
