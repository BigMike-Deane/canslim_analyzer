import { useState, useEffect } from 'react'
import { api, formatCurrency, formatPercent } from '../api'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts'

function PerformanceChart({ data, startingCash }) {
  if (!data || data.length < 2) {
    return (
      <div className="card mb-4 h-48 flex items-center justify-center text-dark-400">
        No chart data available
      </div>
    )
  }

  const finalValue = data[data.length - 1]?.value || startingCash
  const isPositive = finalValue >= startingCash

  return (
    <div className="card mb-4">
      <div className="text-dark-400 text-xs mb-2">Portfolio vs SPY (Buy & Hold)</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <Line
              type="monotone"
              dataKey="return_pct"
              name="Portfolio"
              stroke={isPositive ? '#34c759' : '#ff3b30'}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="spy_return_pct"
              name="SPY"
              stroke="#8e8e93"
              strokeWidth={1}
              strokeDasharray="5 5"
              dot={false}
            />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: '#8e8e93' }}
              tickFormatter={(d) => {
                const date = new Date(d)
                return `${date.getMonth()+1}/${date.getDate()}`
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#8e8e93' }}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              domain={['dataMin - 5', 'dataMax + 5']}
            />
            <Tooltip
              contentStyle={{ background: '#2c2c2e', border: 'none', borderRadius: '8px' }}
              labelStyle={{ color: '#8e8e93' }}
              formatter={(value, name) => [`${value?.toFixed(2)}%`, name]}
            />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function StatCard({ label, value, subtext, color }) {
  return (
    <div className="bg-dark-700 rounded-lg p-3">
      <div className="text-dark-400 text-xs mb-1">{label}</div>
      <div className={`text-lg font-bold ${color || 'text-white'}`}>{value}</div>
      {subtext && <div className="text-dark-500 text-xs">{subtext}</div>}
    </div>
  )
}

function BacktestForm({ onSubmit, isLoading }) {
  const today = new Date().toISOString().split('T')[0]
  const oneYearAgo = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

  const [config, setConfig] = useState({
    start_date: oneYearAgo,
    end_date: today,
    starting_cash: 25000,
    stock_universe: 'sp500'
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(config)
  }

  return (
    <form onSubmit={handleSubmit} className="card mb-4">
      <h2 className="text-lg font-bold mb-4">Run New Backtest</h2>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div>
          <label className="text-dark-400 text-xs block mb-1">Start Date</label>
          <input
            type="date"
            value={config.start_date}
            onChange={(e) => setConfig({...config, start_date: e.target.value})}
            className="w-full bg-dark-700 border border-dark-600 rounded px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="text-dark-400 text-xs block mb-1">End Date</label>
          <input
            type="date"
            value={config.end_date}
            onChange={(e) => setConfig({...config, end_date: e.target.value})}
            className="w-full bg-dark-700 border border-dark-600 rounded px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="text-dark-400 text-xs block mb-1">Starting Cash</label>
          <input
            type="number"
            value={config.starting_cash}
            onChange={(e) => setConfig({...config, starting_cash: parseFloat(e.target.value) || 25000})}
            className="w-full bg-dark-700 border border-dark-600 rounded px-3 py-2 text-sm"
          />
        </div>
        <div>
          <label className="text-dark-400 text-xs block mb-1">Stock Universe</label>
          <select
            value={config.stock_universe}
            onChange={(e) => setConfig({...config, stock_universe: e.target.value})}
            className="w-full bg-dark-700 border border-dark-600 rounded px-3 py-2 text-sm"
          >
            <option value="sp500">S&P 500</option>
            <option value="all">All Stocks (~2000)</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-primary-600 hover:bg-primary-500 disabled:bg-dark-600 rounded py-2 font-semibold transition-colors"
      >
        {isLoading ? 'Starting...' : 'Start Backtest'}
      </button>
    </form>
  )
}

function BacktestList({ backtests, onSelect, onDelete }) {
  if (!backtests || backtests.length === 0) {
    return (
      <div className="card text-center text-dark-400 py-8">
        No backtests yet. Run one above!
      </div>
    )
  }

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return <span className="text-xs bg-green-900 text-green-400 px-2 py-0.5 rounded">Completed</span>
      case 'running':
        return <span className="text-xs bg-blue-900 text-blue-400 px-2 py-0.5 rounded">Running</span>
      case 'failed':
        return <span className="text-xs bg-red-900 text-red-400 px-2 py-0.5 rounded">Failed</span>
      default:
        return <span className="text-xs bg-dark-600 text-dark-300 px-2 py-0.5 rounded">Pending</span>
    }
  }

  return (
    <div className="card">
      <h3 className="text-lg font-bold mb-3">Previous Backtests</h3>
      <div className="space-y-2">
        {backtests.map(bt => (
          <div
            key={bt.id}
            className="bg-dark-700 rounded-lg p-3 flex justify-between items-center cursor-pointer hover:bg-dark-600 transition-colors"
            onClick={() => bt.status === 'completed' && onSelect(bt.id)}
          >
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium">{bt.name}</span>
                {getStatusBadge(bt.status)}
              </div>
              <div className="text-dark-400 text-xs">
                {bt.start_date} to {bt.end_date}
              </div>
              {bt.status === 'running' && (
                <div className="mt-1">
                  <div className="h-1 bg-dark-600 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary-500 transition-all duration-300"
                      style={{ width: `${bt.progress_pct || 0}%` }}
                    />
                  </div>
                  <div className="text-xs text-dark-400 mt-1">{(bt.progress_pct || 0).toFixed(0)}%</div>
                </div>
              )}
            </div>
            {bt.status === 'completed' && (
              <div className="text-right ml-4">
                <div className={`font-bold ${bt.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {bt.total_return_pct >= 0 ? '+' : ''}{bt.total_return_pct?.toFixed(1)}%
                </div>
                <div className="text-dark-500 text-xs">
                  vs SPY {bt.spy_return_pct >= 0 ? '+' : ''}{bt.spy_return_pct?.toFixed(1)}%
                </div>
              </div>
            )}
            {bt.status === 'completed' && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onDelete(bt.id)
                }}
                className="ml-3 text-dark-400 hover:text-red-400 transition-colors"
              >
                X
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function BacktestResults({ backtest, onClose }) {
  const { backtest: bt, performance_chart, trades, statistics } = backtest

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">{bt.name}</h2>
        <button onClick={onClose} className="text-dark-400 hover:text-white">
          Close
        </button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Total Return"
          value={`${bt.total_return_pct >= 0 ? '+' : ''}${bt.total_return_pct?.toFixed(1)}%`}
          subtext={formatCurrency(bt.final_value)}
          color={bt.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="vs SPY"
          value={`${(bt.total_return_pct - bt.spy_return_pct) >= 0 ? '+' : ''}${(bt.total_return_pct - bt.spy_return_pct)?.toFixed(1)}%`}
          subtext={`SPY: ${bt.spy_return_pct?.toFixed(1)}%`}
          color={(bt.total_return_pct - bt.spy_return_pct) >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="Max Drawdown"
          value={`-${bt.max_drawdown_pct?.toFixed(1)}%`}
          color="text-red-400"
        />
        <StatCard
          label="Win Rate"
          value={`${bt.win_rate?.toFixed(0)}%`}
          subtext={`${bt.total_trades} trades`}
        />
      </div>

      {/* Additional stats */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard
          label="Sharpe Ratio"
          value={bt.sharpe_ratio?.toFixed(2) || 'N/A'}
        />
        <StatCard
          label="Best Trade"
          value={`+${statistics.best_trade?.toFixed(1)}%`}
          color="text-green-400"
        />
        <StatCard
          label="Worst Trade"
          value={`${statistics.worst_trade?.toFixed(1)}%`}
          color="text-red-400"
        />
      </div>

      {/* Performance Chart */}
      <PerformanceChart data={performance_chart} startingCash={bt.starting_cash} />

      {/* Trade History */}
      <div className="card">
        <h3 className="text-lg font-bold mb-3">Trade History ({trades.length})</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-dark-400 text-xs">
              <tr>
                <th className="text-left py-2">Date</th>
                <th className="text-left">Ticker</th>
                <th className="text-left">Action</th>
                <th className="text-right">Price</th>
                <th className="text-right">P/L</th>
                <th className="text-left">Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-700">
              {trades.slice(0, 50).map((t, i) => (
                <tr key={i} className="hover:bg-dark-700">
                  <td className="py-2 text-dark-400">{t.date}</td>
                  <td className="font-medium">{t.ticker}</td>
                  <td>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      t.action === 'BUY' ? 'bg-green-900 text-green-400' :
                      t.action === 'SELL' ? 'bg-red-900 text-red-400' :
                      'bg-blue-900 text-blue-400'
                    }`}>
                      {t.action}
                    </span>
                  </td>
                  <td className="text-right">${t.price?.toFixed(2)}</td>
                  <td className={`text-right ${t.gain_pct > 0 ? 'text-green-400' : t.gain_pct < 0 ? 'text-red-400' : ''}`}>
                    {t.action === 'SELL' ? `${t.gain_pct > 0 ? '+' : ''}${t.gain_pct?.toFixed(1)}%` : '-'}
                  </td>
                  <td className="text-dark-400 text-xs max-w-[150px] truncate">{t.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {trades.length > 50 && (
            <div className="text-center text-dark-400 text-xs py-2">
              Showing first 50 of {trades.length} trades
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Backtest() {
  const [backtests, setBacktests] = useState([])
  const [selectedBacktest, setSelectedBacktest] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [pollingId, setPollingId] = useState(null)

  // Fetch backtests on load
  useEffect(() => {
    fetchBacktests()
  }, [])

  // Poll for running backtests
  useEffect(() => {
    const runningBacktest = backtests.find(b => b.status === 'running' || b.status === 'pending')
    if (runningBacktest && !pollingId) {
      const id = setInterval(() => fetchBacktests(), 3000)
      setPollingId(id)
    } else if (!runningBacktest && pollingId) {
      clearInterval(pollingId)
      setPollingId(null)
    }
    return () => {
      if (pollingId) clearInterval(pollingId)
    }
  }, [backtests, pollingId])

  const fetchBacktests = async () => {
    try {
      const data = await fetch('/api/backtests').then(r => r.json())
      setBacktests(data)
    } catch (err) {
      console.error('Failed to fetch backtests:', err)
    }
  }

  const startBacktest = async (config) => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/backtests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Failed to start backtest')
      }
      fetchBacktests()
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const selectBacktest = async (id) => {
    try {
      const data = await fetch(`/api/backtests/${id}`).then(r => r.json())
      setSelectedBacktest(data)
    } catch (err) {
      console.error('Failed to fetch backtest:', err)
    }
  }

  const deleteBacktest = async (id) => {
    try {
      await fetch(`/api/backtests/${id}`, { method: 'DELETE' })
      setBacktests(backtests.filter(b => b.id !== id))
      if (selectedBacktest?.backtest?.id === id) {
        setSelectedBacktest(null)
      }
    } catch (err) {
      console.error('Failed to delete backtest:', err)
    }
  }

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">CANSLIM Backtesting</h1>
      <p className="text-dark-400 text-sm mb-4">
        Test the AI trading strategy against historical data to see how it would have performed.
      </p>

      {error && (
        <div className="bg-red-900/50 border border-red-700 text-red-400 rounded-lg p-3 mb-4">
          {error}
        </div>
      )}

      {selectedBacktest ? (
        <BacktestResults
          backtest={selectedBacktest}
          onClose={() => setSelectedBacktest(null)}
        />
      ) : (
        <>
          <BacktestForm onSubmit={startBacktest} isLoading={isLoading} />
          <BacktestList
            backtests={backtests}
            onSelect={selectBacktest}
            onDelete={deleteBacktest}
          />
        </>
      )}
    </div>
  )
}
