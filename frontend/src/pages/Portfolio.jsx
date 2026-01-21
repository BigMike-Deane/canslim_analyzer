import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent } from '../api'

// 7-day trend indicator component
function WeeklyTrend({ trend, change }) {
  if (!trend) return null

  const config = {
    improving: { icon: '↗', color: 'text-green-400', bg: 'bg-green-500/10' },
    stable: { icon: '→', color: 'text-dark-400', bg: 'bg-dark-600' },
    deteriorating: { icon: '↘', color: 'text-red-400', bg: 'bg-red-500/10' }
  }

  const { icon, color, bg } = config[trend] || config.stable

  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded ${bg} ${color}`}
      title={`7-day trend: ${change != null ? (change > 0 ? '+' : '') + change.toFixed(1) : ''} pts`}
    >
      {icon} {trend === 'improving' ? 'Up' : trend === 'deteriorating' ? 'Down' : ''}
    </span>
  )
}

function PortfolioSummary({ positions }) {
  if (!positions || positions.length === 0) return null

  const totalValue = positions.reduce((sum, p) => sum + (p.current_value || 0), 0)
  const totalCost = positions.reduce((sum, p) => sum + (p.cost_basis * p.shares || 0), 0)
  const totalGain = totalValue - totalCost
  const totalGainPct = totalCost > 0 ? (totalGain / totalCost * 100) : 0

  return (
    <div className="card mb-4">
      <div className="text-dark-400 text-sm mb-1">Portfolio Value</div>
      <div className="text-3xl font-bold mb-1">{formatCurrency(totalValue)}</div>
      <div className={`text-sm flex items-center gap-1 ${totalGain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        <span>{totalGain >= 0 ? '▲' : '▼'}</span>
        <span>{formatCurrency(Math.abs(totalGain))}</span>
        <span>({formatPercent(Math.abs(totalGainPct))})</span>
        <span className="text-dark-500 ml-1">total</span>
      </div>

      <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-dark-700">
        <div>
          <div className="text-dark-400 text-xs">Positions</div>
          <div className="font-semibold">{positions.length}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Cost Basis</div>
          <div className="font-semibold">{formatCurrency(totalCost)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Avg Score</div>
          <div className="font-semibold">
            {formatScore(positions.reduce((sum, p) => sum + (p.canslim_score || 0), 0) / positions.length)}
          </div>
        </div>
      </div>
    </div>
  )
}

function PositionRow({ position, onDelete }) {
  const gainLoss = position.current_value - (position.cost_basis * position.shares)
  const gainLossPct = position.cost_basis > 0
    ? ((position.current_price / position.cost_basis - 1) * 100)
    : 0
  const isPositive = gainLoss >= 0

  const getRecommendationStyle = (rec) => {
    switch (rec) {
      case 'buy': return 'bg-green-500/20 text-green-400'
      case 'add': return 'bg-blue-500/20 text-blue-400'
      case 'hold': return 'bg-yellow-500/20 text-yellow-400'
      case 'trim': return 'bg-orange-500/20 text-orange-400'
      case 'sell': return 'bg-red-500/20 text-red-400'
      default: return 'bg-dark-600 text-dark-300'
    }
  }

  return (
    <div className="border-b border-dark-700 last:border-0 py-3">
      <div className="flex justify-between items-start">
        <Link to={`/stock/${position.ticker}`} className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold">{position.ticker}</span>
            {position.recommendation && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${getRecommendationStyle(position.recommendation)}`}>
                {position.recommendation.toUpperCase()}
              </span>
            )}
          </div>
          <div className="text-dark-400 text-sm">
            {position.shares} shares @ {formatCurrency(position.cost_basis)}
          </div>
        </Link>

        <div className="text-right">
          <div className="font-semibold">{formatCurrency(position.current_value)}</div>
          <div className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {isPositive ? '+' : ''}{formatCurrency(gainLoss)}
            <span className="text-xs ml-1">({formatPercent(gainLossPct, true)})</span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center gap-3 text-sm">
          <div>
            <span className="text-dark-400">Score: </span>
            <span className={`font-medium ${getScoreClass(position.canslim_score)}`}>
              {formatScore(position.canslim_score)}
            </span>
          </div>
          {position.score_change != null && position.score_change !== 0 && (
            <div
              className={position.score_change > 0 ? 'text-green-400' : 'text-red-400'}
              title={`Score changed by ${position.score_change > 0 ? '+' : ''}${position.score_change.toFixed(1)} pts since last scan`}
            >
              {position.score_change > 0 ? '↑' : '↓'} {Math.abs(position.score_change).toFixed(1)}
            </div>
          )}
          <WeeklyTrend trend={position.score_trend} change={position.trend_change} />
          {position.data_quality === 'low' && (
            <span
              className="text-yellow-500 text-xs cursor-help"
              title="Limited analyst data - projections may be less reliable"
            >
              ⚠
            </span>
          )}
        </div>
        <button
          onClick={() => onDelete(position.id)}
          className="text-red-400 text-sm hover:text-red-300"
        >
          Remove
        </button>
      </div>

      {position.notes && (
        <div className="text-dark-400 text-xs mt-2 italic">
          {position.notes}
        </div>
      )}
    </div>
  )
}

function GameplanCard({ action }) {
  const actionStyles = {
    SELL: { bg: 'bg-red-500/10', border: 'border-red-500/30', icon: '🔴', label: 'SELL', textColor: 'text-red-400' },
    TRIM: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', icon: '🟠', label: 'TAKE PROFITS', textColor: 'text-orange-400' },
    BUY: { bg: 'bg-green-500/10', border: 'border-green-500/30', icon: '🟢', label: 'BUY', textColor: 'text-green-400' },
    ADD: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', icon: '🔵', label: 'ADD MORE', textColor: 'text-blue-400' },
    WATCH: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', icon: '👁️', label: 'WATCH', textColor: 'text-purple-400' },
  }

  const style = actionStyles[action.action] || actionStyles.WATCH

  return (
    <div className={`card mb-3 ${style.bg} border ${style.border}`}>
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{style.icon}</span>
          <span className={`font-bold ${style.textColor}`}>{style.label}</span>
          <span className="font-semibold text-lg">{action.ticker}</span>
        </div>
        {action.estimated_value > 0 && (
          <div className="text-right">
            <div className="text-dark-400 text-xs">Est. Value</div>
            <div className="font-semibold">{formatCurrency(action.estimated_value)}</div>
          </div>
        )}
      </div>

      <div className="mb-3">
        <div className="font-medium text-sm">{action.reason}</div>
      </div>

      {action.shares_action > 0 && (
        <div className="flex items-center gap-4 mb-3 p-2 bg-dark-700/50 rounded-lg">
          <div>
            <div className="text-dark-400 text-xs">Shares to {action.action === 'SELL' || action.action === 'TRIM' ? 'Sell' : 'Buy'}</div>
            <div className="font-bold text-lg">{action.shares_action}</div>
          </div>
          {action.shares_current > 0 && (
            <div>
              <div className="text-dark-400 text-xs">Current Position</div>
              <div className="font-semibold">{action.shares_current} shares</div>
            </div>
          )}
          <div>
            <div className="text-dark-400 text-xs">Price</div>
            <div className="font-semibold">{formatCurrency(action.current_price)}</div>
          </div>
        </div>
      )}

      <div className="space-y-1">
        {action.details?.filter(d => d).map((detail, i) => (
          <div key={i} className="text-dark-300 text-sm flex items-start gap-2">
            <span className="text-dark-500">•</span>
            <span>{detail}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function Gameplan({ gameplan, loading }) {
  if (loading) {
    return (
      <div className="mb-4">
        <div className="skeleton h-6 w-32 mb-3" />
        <div className="skeleton h-32 rounded-2xl mb-3" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  if (!gameplan || gameplan.length === 0) {
    return (
      <div className="card mb-4 text-center py-6">
        <div className="text-3xl mb-2">✅</div>
        <div className="font-semibold">No Actions Needed</div>
        <div className="text-dark-400 text-sm">Your portfolio looks good! Check back after running a scan.</div>
      </div>
    )
  }

  const sellActions = gameplan.filter(a => a.action === 'SELL')
  const trimActions = gameplan.filter(a => a.action === 'TRIM')
  const buyActions = gameplan.filter(a => a.action === 'BUY')
  const addActions = gameplan.filter(a => a.action === 'ADD')
  const watchActions = gameplan.filter(a => a.action === 'WATCH')

  return (
    <div className="mb-4">
      <h2 className="text-lg font-bold mb-3">Gameplan</h2>

      {sellActions.length > 0 && (
        <div className="mb-4">
          <div className="text-red-400 font-semibold text-sm mb-2">SELL POSITIONS ({sellActions.length})</div>
          {sellActions.map((action, i) => (
            <GameplanCard key={`sell-${i}`} action={action} />
          ))}
        </div>
      )}

      {trimActions.length > 0 && (
        <div className="mb-4">
          <div className="text-orange-400 font-semibold text-sm mb-2">TAKE PROFITS ({trimActions.length})</div>
          {trimActions.map((action, i) => (
            <GameplanCard key={`trim-${i}`} action={action} />
          ))}
        </div>
      )}

      {buyActions.length > 0 && (
        <div className="mb-4">
          <div className="text-green-400 font-semibold text-sm mb-2">NEW BUYS ({buyActions.length})</div>
          {buyActions.map((action, i) => (
            <GameplanCard key={`buy-${i}`} action={action} />
          ))}
        </div>
      )}

      {addActions.length > 0 && (
        <div className="mb-4">
          <div className="text-blue-400 font-semibold text-sm mb-2">ADD TO POSITIONS ({addActions.length})</div>
          {addActions.map((action, i) => (
            <GameplanCard key={`add-${i}`} action={action} />
          ))}
        </div>
      )}

      {watchActions.length > 0 && (
        <div className="mb-4">
          <div className="text-purple-400 font-semibold text-sm mb-2">WATCHLIST ({watchActions.length})</div>
          {watchActions.map((action, i) => (
            <GameplanCard key={`watch-${i}`} action={action} />
          ))}
        </div>
      )}
    </div>
  )
}

function AddPositionModal({ onClose, onAdd }) {
  const [ticker, setTicker] = useState('')
  const [shares, setShares] = useState('')
  const [costBasis, setCostBasis] = useState('')
  const [notes, setNotes] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!ticker || !shares) return

    onAdd({
      ticker: ticker.toUpperCase(),
      shares: parseFloat(shares),
      cost_basis: costBasis ? parseFloat(costBasis) : null,
      notes: notes || null
    })
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-end justify-center z-50">
      <div className="bg-dark-800 w-full max-w-lg rounded-t-2xl p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="font-semibold text-lg">Add Position</h2>
          <button onClick={onClose} className="text-dark-400 text-xl">&times;</button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-dark-400 text-sm">Ticker Symbol</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="w-full mt-1"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-dark-400 text-sm">Shares</label>
              <input
                type="number"
                step="0.001"
                value={shares}
                onChange={(e) => setShares(e.target.value)}
                placeholder="10"
                className="w-full mt-1"
                required
              />
            </div>
            <div>
              <label className="text-dark-400 text-sm">Cost per Share</label>
              <input
                type="number"
                step="0.01"
                value={costBasis}
                onChange={(e) => setCostBasis(e.target.value)}
                placeholder="150.00"
                className="w-full mt-1"
              />
            </div>
          </div>

          <div>
            <label className="text-dark-400 text-sm">Notes (optional)</label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Entry reason, target price, etc."
              className="w-full mt-1"
            />
          </div>

          <button type="submit" className="w-full btn-primary">
            Add Position
          </button>
        </form>
      </div>
    </div>
  )
}

export default function Portfolio() {
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [positions, setPositions] = useState([])
  const [showAddModal, setShowAddModal] = useState(false)
  const [gameplan, setGameplan] = useState([])
  const [gameplanLoading, setGameplanLoading] = useState(true)

  const fetchPortfolio = async () => {
    try {
      setLoading(true)
      const data = await api.getPortfolio()
      setPositions(data.positions || [])
    } catch (err) {
      console.error('Failed to fetch portfolio:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchGameplan = async () => {
    try {
      setGameplanLoading(true)
      const data = await api.getGameplan()
      setGameplan(data.gameplan || [])
    } catch (err) {
      console.error('Failed to fetch gameplan:', err)
    } finally {
      setGameplanLoading(false)
    }
  }

  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await api.refreshPortfolio()
      await fetchPortfolio()
      await fetchGameplan()
    } catch (err) {
      console.error('Failed to refresh portfolio:', err)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchPortfolio()
    fetchGameplan()
  }, [])

  const handleAdd = async (position) => {
    try {
      await api.addPosition(position)
      fetchPortfolio()
    } catch (err) {
      console.error('Failed to add position:', err)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Remove this position?')) return
    try {
      await api.deletePosition(id)
      setPositions(prev => prev.filter(p => p.id !== id))
    } catch (err) {
      console.error('Failed to delete position:', err)
    }
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-40 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-xl font-bold">Portfolio</h1>
        <div className="flex gap-2">
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="btn-secondary text-sm flex items-center gap-1"
          >
            {refreshing ? (
              <>
                <span className="animate-spin">⟳</span>
                <span>Refreshing...</span>
              </>
            ) : (
              <>
                <span>🔄</span>
                <span>Refresh</span>
              </>
            )}
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary text-sm"
          >
            + Add
          </button>
        </div>
      </div>

      {positions.length === 0 ? (
        <div className="card text-center py-8">
          <div className="text-4xl mb-3">💼</div>
          <div className="font-semibold mb-2">No Positions Yet</div>
          <div className="text-dark-400 text-sm mb-4">
            Add stocks to your portfolio to track performance and get CANSLIM recommendations.
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary"
          >
            Add First Position
          </button>
        </div>
      ) : (
        <>
          <PortfolioSummary positions={positions} />

          <div className="card mb-4">
            <div className="flex justify-between items-center mb-3">
              <div className="font-semibold">Positions</div>
              <div className="flex items-center gap-3 text-[10px] text-dark-400">
                <span title="Score change since last scan">↑↓ Scan change</span>
                <span title="7-day score trend">↗↘ 7-day trend</span>
                <span className="text-yellow-500" title="Limited analyst data">⚠ Low data</span>
              </div>
            </div>
            {positions.map(position => (
              <PositionRow
                key={position.id}
                position={position}
                onDelete={handleDelete}
              />
            ))}
          </div>

          <Gameplan gameplan={gameplan} loading={gameplanLoading} />
        </>
      )}

      {showAddModal && (
        <AddPositionModal
          onClose={() => setShowAddModal(false)}
          onAdd={handleAdd}
        />
      )}

      <div className="h-4" />
    </div>
  )
}
