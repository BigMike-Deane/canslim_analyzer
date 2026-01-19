import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent } from '../api'

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
      case 'hold': return 'bg-yellow-500/20 text-yellow-400'
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
            <div className={position.score_change > 0 ? 'text-green-400' : 'text-red-400'}>
              {position.score_change > 0 ? '↑' : '↓'} {Math.abs(position.score_change).toFixed(1)}
            </div>
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

  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await api.refreshPortfolio()
      await fetchPortfolio()
    } catch (err) {
      console.error('Failed to refresh portfolio:', err)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchPortfolio()
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

          <div className="card">
            <div className="font-semibold mb-3">Positions</div>
            {positions.map(position => (
              <PositionRow
                key={position.id}
                position={position}
                onDelete={handleDelete}
              />
            ))}
          </div>
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
