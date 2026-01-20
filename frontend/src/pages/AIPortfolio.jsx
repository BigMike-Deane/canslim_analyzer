import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, formatScore, getScoreClass } from '../api'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from 'recharts'

function PerformanceChart({ history, startingCash }) {
  if (!history || history.length < 2) {
    return (
      <div className="card mb-4 h-48 flex items-center justify-center text-dark-400">
        Not enough data for chart yet
      </div>
    )
  }

  const latestValue = history[history.length - 1]?.total_value || startingCash
  const isPositive = latestValue >= startingCash

  return (
    <div className="card mb-4">
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <Line
              type="monotone"
              dataKey="total_value"
              stroke={isPositive ? '#34c759' : '#ff3b30'}
              strokeWidth={2}
              dot={false}
            />
            <ReferenceLine
              y={startingCash}
              stroke="#666"
              strokeDasharray="3 3"
              label={{ value: 'Start', position: 'right', fill: '#666', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{ background: '#2c2c2e', border: 'none', borderRadius: '8px' }}
              labelStyle={{ color: '#8e8e93' }}
              formatter={(value) => [formatCurrency(value), 'Value']}
              labelFormatter={(label) => {
                const [y, m, d] = label.split('-').map(Number)
                return new Date(y, m - 1, d).toLocaleDateString()
              }}
            />
            <XAxis dataKey="date" hide />
            <YAxis hide domain={['dataMin - 500', 'dataMax + 500']} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function SummaryCard({ summary, config }) {
  if (!summary) return null

  const isPositive = summary.total_return >= 0

  return (
    <div className="card mb-4">
      <div className="text-dark-400 text-sm mb-1">AI Portfolio Value</div>
      <div className="text-3xl font-bold mb-1">
        {formatCurrency(summary.total_value)}
      </div>
      <div className={`text-sm flex items-center gap-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
        <span>{isPositive ? '↑' : '↓'}</span>
        <span>{formatCurrency(Math.abs(summary.total_return))}</span>
        <span className="text-dark-500">({formatPercent(summary.total_return_pct, true)})</span>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-dark-700">
        <div>
          <div className="text-dark-400 text-xs">Cash</div>
          <div className="font-semibold">{formatCurrency(summary.cash)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Invested</div>
          <div className="font-semibold">{formatCurrency(summary.positions_value)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Positions</div>
          <div className="font-semibold">{summary.positions_count} / {config?.max_positions || 15}</div>
        </div>
      </div>
    </div>
  )
}

function PositionsList({ positions }) {
  if (!positions || positions.length === 0) {
    return (
      <div className="card mb-4 text-center py-8 text-dark-400">
        No positions yet. Initialize the portfolio to start trading.
      </div>
    )
  }

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">Positions ({positions.length})</div>
      <div className="space-y-3">
        {positions.map(position => (
          <Link
            key={position.id}
            to={`/stock/${position.ticker}`}
            className="flex justify-between items-center py-2 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors"
          >
            <div>
              <div className="font-medium">{position.ticker}</div>
              <div className="text-dark-400 text-xs">
                {position.shares.toFixed(2)} shares @ {formatCurrency(position.cost_basis)}
              </div>
            </div>
            <div className="text-right">
              <div className="font-semibold">{formatCurrency(position.current_value)}</div>
              <div className={`text-xs ${position.gain_loss_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {position.gain_loss_pct >= 0 ? '+' : ''}{position.gain_loss_pct?.toFixed(2)}%
              </div>
            </div>
            <div className={`ml-3 px-2 py-1 rounded text-sm font-medium ${getScoreClass(position.current_score)}`}>
              {formatScore(position.current_score)}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function TradeHistory({ trades }) {
  if (!trades || trades.length === 0) {
    return null
  }

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">Recent Trades</div>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {trades.slice(0, 20).map(trade => (
          <div
            key={trade.id}
            className="flex justify-between items-center py-2 border-b border-dark-700 last:border-0 text-sm"
          >
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                trade.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}>
                {trade.action}
              </span>
              <span className="font-medium">{trade.ticker}</span>
            </div>
            <div className="text-right">
              <div>{trade.shares.toFixed(2)} @ {formatCurrency(trade.price)}</div>
              <div className="text-dark-400 text-xs">{trade.reason}</div>
            </div>
            {trade.realized_gain != null && (
              <div className={`ml-2 text-xs ${trade.realized_gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function ConfigPanel({ config, onUpdate, onInitialize, onRunCycle, onRefresh }) {
  const [isActive, setIsActive] = useState(config?.is_active || false)
  const [updating, setUpdating] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => {
    setIsActive(config?.is_active || false)
  }, [config])

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
    } finally {
      setRefreshing(false)
    }
  }

  const handleInitialize = async () => {
    if (!confirm('This will reset the AI Portfolio to $25,000 and clear all history. Continue?')) {
      return
    }
    setInitializing(true)
    try {
      await onInitialize()
    } finally {
      setInitializing(false)
    }
  }

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold">AI Trading</div>
        <button
          onClick={handleToggle}
          disabled={updating}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            isActive ? 'bg-green-500' : 'bg-dark-600'
          }`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            isActive ? 'translate-x-7' : 'translate-x-1'
          }`} />
        </button>
      </div>

      {isActive && (
        <div className="mb-3 p-2 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <span className="animate-pulse">●</span>
            <span>AI Trading Active - Trades execute after each scan</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 text-sm mb-3">
        <div>
          <div className="text-dark-400 text-xs">Min Score to Buy</div>
          <div className="font-medium">{config?.min_score_to_buy || 65}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Sell Below Score</div>
          <div className="font-medium">{config?.sell_score_threshold || 45}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Take Profit</div>
          <div className="font-medium text-green-400">+{config?.take_profit_pct || 40}%</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Stop Loss</div>
          <div className="font-medium text-red-400">-{config?.stop_loss_pct || 10}%</div>
        </div>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex-1 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          <span className={refreshing ? 'animate-spin' : ''}>⟳</span>
          <span>{refreshing ? 'Refreshing...' : 'Refresh Prices'}</span>
        </button>
        <button
          onClick={onRunCycle}
          className="flex-1 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg text-sm font-medium transition-colors"
        >
          Run Trading Cycle
        </button>
      </div>

      <button
        onClick={handleInitialize}
        disabled={initializing}
        className="w-full py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-dark-300"
      >
        {initializing ? 'Resetting...' : 'Reset Portfolio ($25k)'}
      </button>
    </div>
  )
}

export default function AIPortfolio() {
  const [loading, setLoading] = useState(true)
  const [portfolio, setPortfolio] = useState(null)
  const [history, setHistory] = useState([])
  const [trades, setTrades] = useState([])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [portfolioData, historyData, tradesData] = await Promise.all([
        api.getAIPortfolio(),
        api.getAIPortfolioHistory(90),
        api.getAIPortfolioTrades(50)
      ])
      setPortfolio(portfolioData)
      setHistory(historyData)
      setTrades(tradesData)
    } catch (err) {
      console.error('Failed to fetch AI Portfolio:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const handleUpdateConfig = async (config) => {
    try {
      await api.updateAIPortfolioConfig(config)
      fetchData()
    } catch (err) {
      console.error('Failed to update config:', err)
    }
  }

  const handleInitialize = async () => {
    try {
      await api.initializeAIPortfolio(25000)
      fetchData()
    } catch (err) {
      console.error('Failed to initialize:', err)
    }
  }

  const handleRefresh = async () => {
    try {
      await api.refreshAIPortfolio()
      fetchData()
    } catch (err) {
      console.error('Failed to refresh:', err)
    }
  }

  const handleRunCycle = async () => {
    try {
      await api.runAITradingCycle()
      fetchData()
    } catch (err) {
      console.error('Failed to run cycle:', err)
    }
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="skeleton h-48 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-48 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <div>
          <div className="text-dark-400 text-sm">Autonomous</div>
          <h1 className="text-xl font-bold">AI Portfolio</h1>
        </div>
        <div className="text-dark-400 text-xs">
          Started: {formatCurrency(portfolio?.config?.starting_cash || 25000)}
        </div>
      </div>

      <PerformanceChart
        history={history}
        startingCash={portfolio?.config?.starting_cash || 25000}
      />

      <SummaryCard
        summary={portfolio?.summary}
        config={portfolio?.config}
      />

      <ConfigPanel
        config={portfolio?.config}
        onUpdate={handleUpdateConfig}
        onInitialize={handleInitialize}
        onRefresh={handleRefresh}
        onRunCycle={handleRunCycle}
      />

      <PositionsList positions={portfolio?.positions} />

      <TradeHistory trades={trades} />

      <div className="h-4" />
    </div>
  )
}
