import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent } from '../api'

function WatchlistItem({ item, onRemove }) {
  const [stock, setStock] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStock = async () => {
      try {
        const data = await api.getStock(item.ticker)
        setStock(data)
      } catch (err) {
        console.error(`Failed to fetch ${item.ticker}:`, err)
      } finally {
        setLoading(false)
      }
    }
    fetchStock()
  }, [item.ticker])

  if (loading) {
    return (
      <div className="border-b border-dark-700 last:border-0 py-3">
        <div className="skeleton h-16 rounded" />
      </div>
    )
  }

  const meetsTarget = item.target_price && stock?.current_price >= item.target_price
  const meetsScoreAlert = item.alert_score && stock?.canslim_score >= item.alert_score

  return (
    <div className="border-b border-dark-700 last:border-0 py-3">
      <div className="flex justify-between items-start">
        <Link to={`/stock/${item.ticker}`} className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold">{item.ticker}</span>
            {(meetsTarget || meetsScoreAlert) && (
              <span className="text-yellow-400 text-sm">🔔</span>
            )}
          </div>
          {stock && (
            <div className="text-dark-400 text-sm truncate max-w-[200px]">
              {stock.name}
            </div>
          )}
        </Link>

        <div className="text-right">
          {stock ? (
            <>
              <div className="font-semibold">{formatCurrency(stock.current_price)}</div>
              <div className={`text-sm px-2 py-0.5 rounded inline-block ${getScoreClass(stock.canslim_score)}`}>
                {formatScore(stock.canslim_score)}
              </div>
            </>
          ) : (
            <div className="text-dark-400 text-sm">No data</div>
          )}
        </div>
      </div>

      {/* Alerts & Notes */}
      <div className="flex items-center gap-4 mt-2 text-sm">
        {item.target_price && (
          <div className={meetsTarget ? 'text-green-400' : 'text-dark-400'}>
            Target: {formatCurrency(item.target_price)}
            {meetsTarget && ' ✓'}
          </div>
        )}
        {item.alert_score && (
          <div className={meetsScoreAlert ? 'text-green-400' : 'text-dark-400'}>
            Score Alert: {item.alert_score}+
            {meetsScoreAlert && ' ✓'}
          </div>
        )}
      </div>

      {item.notes && (
        <div className="text-dark-400 text-xs mt-2 italic">{item.notes}</div>
      )}

      <div className="flex justify-between items-center mt-2">
        <div className="text-dark-500 text-xs">
          Added {new Date(item.added_at).toLocaleDateString()}
        </div>
        <button
          onClick={() => onRemove(item.id)}
          className="text-red-400 text-sm hover:text-red-300"
        >
          Remove
        </button>
      </div>
    </div>
  )
}

function AddWatchlistModal({ onClose, onAdd }) {
  const [ticker, setTicker] = useState('')
  const [targetPrice, setTargetPrice] = useState('')
  const [alertScore, setAlertScore] = useState('')
  const [notes, setNotes] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!ticker) return

    onAdd({
      ticker: ticker.toUpperCase(),
      target_price: targetPrice ? parseFloat(targetPrice) : null,
      alert_score: alertScore ? parseFloat(alertScore) : null,
      notes: notes || null
    })
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-end justify-center z-50">
      <div className="bg-dark-800 w-full max-w-lg rounded-t-2xl p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="font-semibold text-lg">Add to Watchlist</h2>
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
              <label className="text-dark-400 text-sm">Target Price (optional)</label>
              <input
                type="number"
                step="0.01"
                value={targetPrice}
                onChange={(e) => setTargetPrice(e.target.value)}
                placeholder="200.00"
                className="w-full mt-1"
              />
            </div>
            <div>
              <label className="text-dark-400 text-sm">Score Alert (optional)</label>
              <input
                type="number"
                step="1"
                min="0"
                max="100"
                value={alertScore}
                onChange={(e) => setAlertScore(e.target.value)}
                placeholder="80"
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
              placeholder="Watching for breakout, earnings soon, etc."
              className="w-full mt-1"
            />
          </div>

          <button type="submit" className="w-full btn-primary">
            Add to Watchlist
          </button>
        </form>
      </div>
    </div>
  )
}

export default function Watchlist() {
  const [loading, setLoading] = useState(true)
  const [items, setItems] = useState([])
  const [showAddModal, setShowAddModal] = useState(false)

  const fetchWatchlist = async () => {
    try {
      setLoading(true)
      const data = await api.getWatchlist()
      setItems(data.items || [])
    } catch (err) {
      console.error('Failed to fetch watchlist:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWatchlist()
  }, [])

  const handleAdd = async (item) => {
    try {
      await api.addToWatchlist(item)
      fetchWatchlist()
    } catch (err) {
      console.error('Failed to add to watchlist:', err)
    }
  }

  const handleRemove = async (id) => {
    if (!confirm('Remove from watchlist?')) return
    try {
      await api.removeFromWatchlist(id)
      setItems(prev => prev.filter(i => i.id !== id))
    } catch (err) {
      console.error('Failed to remove:', err)
    }
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-24 rounded-2xl mb-4" />
        <div className="skeleton h-24 rounded-2xl mb-4" />
        <div className="skeleton h-24 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-xl font-bold">Watchlist</h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="btn-primary text-sm"
        >
          + Add
        </button>
      </div>

      {items.length === 0 ? (
        <div className="card text-center py-8">
          <div className="text-4xl mb-3">👁️</div>
          <div className="font-semibold mb-2">Watchlist Empty</div>
          <div className="text-dark-400 text-sm mb-4">
            Add stocks you're watching to track their CANSLIM scores and set price alerts.
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary"
          >
            Add Stock to Watch
          </button>
        </div>
      ) : (
        <div className="card">
          {items.map(item => (
            <WatchlistItem
              key={item.id}
              item={item}
              onRemove={handleRemove}
            />
          ))}
        </div>
      )}

      {showAddModal && (
        <AddWatchlistModal
          onClose={() => setShowAddModal(false)}
          onAdd={handleAdd}
        />
      )}

      <div className="h-4" />
    </div>
  )
}
