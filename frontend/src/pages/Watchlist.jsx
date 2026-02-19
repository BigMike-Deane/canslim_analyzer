import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatDate } from '../api'
import Card from '../components/Card'
import { ScoreBadge } from '../components/Badge'
import PageHeader from '../components/PageHeader'
import Modal from '../components/Modal'

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
      <div className="border-b border-dark-700/30 last:border-0 py-3 px-4">
        <div className="skeleton h-16 rounded" />
      </div>
    )
  }

  const meetsTarget = item.target_price && stock?.current_price >= item.target_price
  const meetsScoreAlert = item.alert_score && stock?.canslim_score >= item.alert_score

  return (
    <div className="border-b border-dark-700/30 last:border-0 py-3 px-4 hover:bg-dark-800/40 transition-colors">
      <div className="flex justify-between items-start">
        <Link to={`/stock/${item.ticker}`} className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-dark-50">{item.ticker}</span>
            {(meetsTarget || meetsScoreAlert) && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/20">
                ALERT
              </span>
            )}
          </div>
          {stock && (
            <div className="text-dark-400 text-sm truncate max-w-[200px]">
              {stock.name}
            </div>
          )}
        </Link>

        <div className="text-right flex flex-col items-end gap-1">
          {stock ? (
            <>
              <span className="font-semibold font-data text-dark-50">{formatCurrency(stock.current_price)}</span>
              <ScoreBadge score={stock.canslim_score} size="xs" />
            </>
          ) : (
            <div className="text-dark-500 text-sm">No data</div>
          )}
        </div>
      </div>

      {/* Alerts & Notes */}
      {(item.target_price || item.alert_score) && (
        <div className="flex items-center gap-4 mt-2 text-sm">
          {item.target_price && (
            <div className={meetsTarget ? 'text-emerald-400' : 'text-dark-400'}>
              <span className="text-[10px] uppercase tracking-wider text-dark-500 mr-1">Target:</span>
              <span className="font-data">{formatCurrency(item.target_price)}</span>
              {meetsTarget && <span className="text-emerald-400 ml-1">&#10003;</span>}
            </div>
          )}
          {item.alert_score && (
            <div className={meetsScoreAlert ? 'text-emerald-400' : 'text-dark-400'}>
              <span className="text-[10px] uppercase tracking-wider text-dark-500 mr-1">Score:</span>
              <span className="font-data">{item.alert_score}+</span>
              {meetsScoreAlert && <span className="text-emerald-400 ml-1">&#10003;</span>}
            </div>
          )}
        </div>
      )}

      {item.notes && (
        <div className="text-dark-500 text-xs mt-2 italic">{item.notes}</div>
      )}

      <div className="flex justify-between items-center mt-2">
        <div className="text-dark-500 text-[10px] font-data">
          Added {formatDate(item.added_at)}
        </div>
        <button
          onClick={() => onRemove(item.id)}
          className="text-xs text-red-400/70 hover:text-red-400 transition-colors"
        >
          Remove
        </button>
      </div>
    </div>
  )
}

function AddWatchlistForm({ onClose, onAdd }) {
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
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">Ticker Symbol</label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="AAPL"
          className="w-full mt-1"
          required
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        <div>
          <label className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">Target Price</label>
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
          <label className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">Score Alert</label>
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
        <label className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">Notes</label>
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
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-24 rounded-2xl mb-4" />
        <div className="skeleton h-24 rounded-2xl mb-4" />
        <div className="skeleton h-24 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Watchlist"
        backTo="/"
        backLabel="Command Center"
        actions={
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary text-sm"
          >
            + Add
          </button>
        }
      />

      {items.length === 0 ? (
        <Card variant="glass" className="text-center py-10">
          <div className="text-3xl mb-3 text-dark-500">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mx-auto text-dark-500">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <div className="font-semibold text-dark-100 mb-2">Watchlist Empty</div>
          <div className="text-dark-400 text-sm mb-4 max-w-xs mx-auto">
            Add stocks you're watching to track their CANSLIM scores and set price alerts.
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary"
          >
            Add Stock to Watch
          </button>
        </Card>
      ) : (
        <Card variant="glass" padding="">
          {items.map(item => (
            <WatchlistItem
              key={item.id}
              item={item}
              onRemove={handleRemove}
            />
          ))}
        </Card>
      )}

      <Modal
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        title="Add to Watchlist"
      >
        <AddWatchlistForm
          onClose={() => setShowAddModal(false)}
          onAdd={handleAdd}
        />
      </Modal>

      <div className="h-4" />
    </div>
  )
}
