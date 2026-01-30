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

  // Format timestamp for tooltip - convert to CST
  const formatTimestamp = (ts) => {
    if (!ts) return ''
    try {
      const date = new Date(ts)
      return date.toLocaleString('en-US', {
        timeZone: 'America/Chicago',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) + ' CST'
    } catch {
      return ts
    }
  }

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-2">
        <div className="text-dark-400 text-xs">Performance</div>
        <div className="text-dark-500 text-xs">{history.length} data points</div>
      </div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <Line
              type="monotone"
              dataKey="total_value"
              stroke={isPositive ? '#34c759' : '#ff3b30'}
              strokeWidth={2}
              dot={history.length <= 50}
              activeDot={{ r: 4, fill: isPositive ? '#34c759' : '#ff3b30' }}
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
              labelFormatter={(_, payload) => {
                if (payload && payload[0]) {
                  return formatTimestamp(payload[0].payload.timestamp || payload[0].payload.date)
                }
                return ''
              }}
            />
            <XAxis dataKey="timestamp" hide />
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
              <div className="flex items-center gap-2">
                <span className="font-medium">{position.ticker}</span>
                {position.is_growth_stock && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-400">
                    Growth
                  </span>
                )}
              </div>
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
            <div className="ml-3 text-right">
              {/* Show appropriate score based on stock type */}
              {position.is_growth_stock ? (
                <div className={`px-2 py-1 rounded text-sm font-medium ${getScoreClass(position.current_growth_score)}`}>
                  {formatScore(position.current_growth_score)}
                </div>
              ) : (
                <div className={`px-2 py-1 rounded text-sm font-medium ${getScoreClass(position.current_score)}`}>
                  {formatScore(position.current_score)}
                </div>
              )}
              {/* Show secondary score if available */}
              {position.is_growth_stock && position.current_score > 0 && (
                <div className="text-[10px] text-dark-400 mt-0.5">
                  CANSLIM: {position.current_score?.toFixed(0)}
                </div>
              )}
              {!position.is_growth_stock && position.current_growth_score > 0 && (
                <div className="text-[10px] text-dark-400 mt-0.5">
                  Growth: {position.current_growth_score?.toFixed(0)}
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function TradeDetailModal({ trade, onClose }) {
  if (!trade) return null

  const formatDateTime = (ts) => {
    if (!ts) return 'N/A'
    try {
      const date = new Date(ts)
      return date.toLocaleString('en-US', {
        timeZone: 'America/Chicago',
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) + ' CST'
    } catch {
      return ts
    }
  }

  const gainPct = trade.action === 'SELL' && trade.cost_basis
    ? ((trade.price - trade.cost_basis) / trade.cost_basis * 100)
    : null

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-dark-800 rounded-lg max-w-md w-full p-5 shadow-xl" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1 rounded font-bold ${
              trade.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {trade.action}
            </span>
            <div>
              <Link
                to={`/stock/${trade.ticker}`}
                className="text-xl font-bold text-primary-400 hover:underline"
                onClick={onClose}
              >
                {trade.ticker}
              </Link>
              {trade.is_growth_stock && (
                <span className="ml-2 px-2 py-0.5 rounded text-xs bg-purple-500/20 text-purple-400">Growth</span>
              )}
            </div>
          </div>
          <button onClick={onClose} className="text-dark-400 hover:text-white text-xl">&times;</button>
        </div>

        {/* Trade Details */}
        <div className="space-y-3 text-sm">
          <div className="flex justify-between py-2 border-b border-dark-700">
            <span className="text-dark-400">Date & Time</span>
            <span className="font-medium">{formatDateTime(trade.executed_at)}</span>
          </div>

          <div className="flex justify-between py-2 border-b border-dark-700">
            <span className="text-dark-400">Shares</span>
            <span className="font-medium">{trade.shares.toFixed(4)}</span>
          </div>

          <div className="flex justify-between py-2 border-b border-dark-700">
            <span className="text-dark-400">Price</span>
            <span className="font-medium">{formatCurrency(trade.price)}</span>
          </div>

          <div className="flex justify-between py-2 border-b border-dark-700">
            <span className="text-dark-400">Total Value</span>
            <span className="font-medium">{formatCurrency(trade.total_value)}</span>
          </div>

          {trade.action === 'SELL' && trade.cost_basis && (
            <div className="flex justify-between py-2 border-b border-dark-700">
              <span className="text-dark-400">Cost Basis</span>
              <span className="font-medium">{formatCurrency(trade.cost_basis)}/share</span>
            </div>
          )}

          {trade.realized_gain != null && (
            <div className="flex justify-between py-2 border-b border-dark-700">
              <span className="text-dark-400">Realized Gain/Loss</span>
              <span className={`font-medium ${trade.realized_gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
                {gainPct != null && ` (${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(1)}%)`}
              </span>
            </div>
          )}

          <div className="flex justify-between py-2 border-b border-dark-700">
            <span className="text-dark-400">CANSLIM Score</span>
            <span className="font-medium">{trade.canslim_score?.toFixed(1) || 'N/A'}</span>
          </div>

          {trade.is_growth_stock && trade.growth_mode_score && (
            <div className="flex justify-between py-2 border-b border-dark-700">
              <span className="text-dark-400">Growth Mode Score</span>
              <span className="font-medium">{trade.growth_mode_score.toFixed(1)}</span>
            </div>
          )}

          {/* Reason Section */}
          <div className="pt-2">
            <div className="text-dark-400 mb-2">Reason</div>
            <div className="bg-dark-700 rounded p-3 text-sm">
              {trade.reason || 'No reason recorded'}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-dark-700 hover:bg-dark-600 rounded text-sm"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function TradeHistory({ trades }) {
  const [selectedTrade, setSelectedTrade] = useState(null)

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
            onClick={() => setSelectedTrade(trade)}
            className="flex justify-between items-center py-2 border-b border-dark-700 last:border-0 text-sm cursor-pointer hover:bg-dark-700/50 rounded px-2 -mx-2 transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                trade.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}>
                {trade.action}
              </span>
              <span className="font-medium">{trade.ticker}</span>
              {trade.is_growth_stock && (
                <span className="px-1 py-0.5 rounded text-[9px] bg-purple-500/20 text-purple-400">G</span>
              )}
            </div>
            <div className="text-right">
              <div>{trade.shares.toFixed(2)} @ {formatCurrency(trade.price)}</div>
              <div className="text-dark-400 text-xs truncate max-w-[150px]">{trade.reason}</div>
            </div>
            {trade.realized_gain != null && (
              <div className={`ml-2 text-xs ${trade.realized_gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Trade Detail Modal */}
      {selectedTrade && (
        <TradeDetailModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
      )}
    </div>
  )
}

function ConfigPanel({ config, onUpdate, onInitialize, onRunCycle, onRefresh, waitingForTrades }) {
  const [isActive, setIsActive] = useState(config?.is_active || false)
  const [updating, setUpdating] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editedConfig, setEditedConfig] = useState({
    min_score_to_buy: config?.min_score_to_buy || 65,
    sell_score_threshold: config?.sell_score_threshold || 45,
    take_profit_pct: config?.take_profit_pct || 40,
    stop_loss_pct: config?.stop_loss_pct || 10
  })

  useEffect(() => {
    setIsActive(config?.is_active || false)
    setEditedConfig({
      min_score_to_buy: config?.min_score_to_buy || 65,
      sell_score_threshold: config?.sell_score_threshold || 45,
      take_profit_pct: config?.take_profit_pct || 40,
      stop_loss_pct: config?.stop_loss_pct || 10
    })
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

  const handleSaveConfig = async () => {
    setUpdating(true)
    try {
      await onUpdate(editedConfig)
      setIsEditing(false)
    } finally {
      setUpdating(false)
    }
  }

  const handleCancelEdit = () => {
    setEditedConfig({
      min_score_to_buy: config?.min_score_to_buy || 65,
      sell_score_threshold: config?.sell_score_threshold || 45,
      take_profit_pct: config?.take_profit_pct || 40,
      stop_loss_pct: config?.stop_loss_pct || 10
    })
    setIsEditing(false)
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
          {isEditing ? (
            <input
              type="number"
              min="50"
              max="100"
              value={editedConfig.min_score_to_buy}
              onChange={(e) => setEditedConfig({ ...editedConfig, min_score_to_buy: parseInt(e.target.value) || 65 })}
              className="w-full mt-1 px-2 py-1 bg-dark-700 border border-dark-500 rounded text-sm focus:outline-none focus:border-primary-500"
            />
          ) : (
            <div className="font-medium">{config?.min_score_to_buy || 65}</div>
          )}
        </div>
        <div>
          <div className="text-dark-400 text-xs">Sell Below Score</div>
          {isEditing ? (
            <input
              type="number"
              min="20"
              max="80"
              value={editedConfig.sell_score_threshold}
              onChange={(e) => setEditedConfig({ ...editedConfig, sell_score_threshold: parseInt(e.target.value) || 45 })}
              className="w-full mt-1 px-2 py-1 bg-dark-700 border border-dark-500 rounded text-sm focus:outline-none focus:border-primary-500"
            />
          ) : (
            <div className="font-medium">{config?.sell_score_threshold || 45}</div>
          )}
        </div>
        <div>
          <div className="text-dark-400 text-xs">Take Profit %</div>
          {isEditing ? (
            <input
              type="number"
              min="10"
              max="100"
              value={editedConfig.take_profit_pct}
              onChange={(e) => setEditedConfig({ ...editedConfig, take_profit_pct: parseFloat(e.target.value) || 40 })}
              className="w-full mt-1 px-2 py-1 bg-dark-700 border border-dark-500 rounded text-sm focus:outline-none focus:border-primary-500"
            />
          ) : (
            <div className="font-medium text-green-400">+{config?.take_profit_pct || 40}%</div>
          )}
        </div>
        <div>
          <div className="text-dark-400 text-xs">Stop Loss %</div>
          {isEditing ? (
            <input
              type="number"
              min="5"
              max="50"
              value={editedConfig.stop_loss_pct}
              onChange={(e) => setEditedConfig({ ...editedConfig, stop_loss_pct: parseFloat(e.target.value) || 10 })}
              className="w-full mt-1 px-2 py-1 bg-dark-700 border border-dark-500 rounded text-sm focus:outline-none focus:border-primary-500"
            />
          ) : (
            <div className="font-medium text-red-400">-{config?.stop_loss_pct || 10}%</div>
          )}
        </div>
      </div>

      {isEditing ? (
        <div className="flex gap-2 mb-3">
          <button
            onClick={handleSaveConfig}
            disabled={updating}
            className="flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {updating ? 'Saving...' : 'Save Settings'}
          </button>
          <button
            onClick={handleCancelEdit}
            className="flex-1 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm font-medium transition-colors"
          >
            Cancel
          </button>
        </div>
      ) : (
        <button
          onClick={() => setIsEditing(true)}
          className="w-full py-2 mb-3 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          <span>✎</span>
          <span>Edit Settings</span>
        </button>
      )}

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
          onClick={handleRunCycle}
          disabled={waitingForTrades}
          className="flex-1 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {waitingForTrades && <span className="animate-spin">⟳</span>}
          <span>{waitingForTrades ? 'Running...' : 'Run Trading Cycle'}</span>
        </button>
      </div>

      <button
        onClick={handleInitialize}
        disabled={initializing}
        className="w-full py-2 mb-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-dark-300"
      >
        {initializing ? 'Resetting...' : 'Reset Portfolio ($25k)'}
      </button>

      <Link
        to="/backtest"
        className="block w-full py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-center text-primary-400"
      >
        Run Historical Backtest
      </Link>
    </div>
  )
}

export default function AIPortfolio() {
  const [loading, setLoading] = useState(true)
  const [portfolio, setPortfolio] = useState(null)
  const [history, setHistory] = useState([])
  const [trades, setTrades] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [waitingForTrades, setWaitingForTrades] = useState(false)
  const [waitingCash, setWaitingCash] = useState(null)

  const fetchData = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true)
      const [portfolioData, historyData, tradesData] = await Promise.all([
        api.getAIPortfolio(),
        api.getAIPortfolioHistory(90),
        api.getAIPortfolioTrades(50)
      ])
      setPortfolio(portfolioData)
      setHistory(historyData)
      setTrades(tradesData)
      setLastUpdated(new Date())

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
      const result = await api.refreshAIPortfolio()
      // Poll more frequently after triggering a refresh
      if (result.status === 'started') {
        setTimeout(() => fetchData(false), 4000)
        setTimeout(() => fetchData(false), 8000)
        setTimeout(() => fetchData(false), 12000)
      } else {
        fetchData()
      }
    } catch (err) {
      console.error('Failed to refresh:', err)
    }
  }

  const handleRunCycle = async () => {
    try {
      // Store current cash to detect when trades complete
      const currentCash = portfolio?.summary?.cash || 0
      setWaitingCash(currentCash)
      setWaitingForTrades(true)

      const result = await api.runAITradingCycle()
      if (result.status !== 'started') {
        setWaitingForTrades(false)
        fetchData()
      }
    } catch (err) {
      console.error('Failed to run cycle:', err)
      setWaitingForTrades(false)
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
        <div className="text-right">
          <div className="text-dark-400 text-xs">
            Started: {formatCurrency(portfolio?.config?.starting_cash || 25000)}
          </div>
          {lastUpdated && (
            <div className="text-dark-500 text-xs">
              Updated: {lastUpdated.toLocaleTimeString('en-US', { timeZone: 'America/Chicago' })} CST
            </div>
          )}
        </div>
      </div>

      {waitingForTrades && (
        <div className="card mb-4 p-3 bg-primary-500/10 border border-primary-500/30">
          <div className="flex items-center gap-2 text-primary-400">
            <span className="animate-spin">⟳</span>
            <span className="font-medium">Executing trades... This may take up to 2 minutes.</span>
          </div>
          <div className="text-dark-400 text-xs mt-1">Page will auto-update when complete.</div>
        </div>
      )}

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
        waitingForTrades={waitingForTrades}
      />

      <PositionsList positions={portfolio?.positions} />

      <TradeHistory trades={trades} />

      <div className="h-4" />
    </div>
  )
}
