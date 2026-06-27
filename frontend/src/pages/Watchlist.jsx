import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatDate, formatRelativeTime } from '../api'
import { saveStockListContext } from '../stockListContext'
import Card, { SectionLabel } from '../components/Card'
import { ScoreBadge } from '../components/Badge'
import PageHeader from '../components/PageHeader'
import Modal from '../components/Modal'
import { useToast } from '../components/Toast'

function hasActiveAlert(item) {
  const meetsTarget = item.target_price && item.current_price != null
    && item.current_price >= item.target_price
  const meetsScore = item.alert_score && item.canslim_score != null
    && item.canslim_score >= item.alert_score
  return meetsTarget || meetsScore
}

function WatchlistItem({ item, onRemove }) {
  // Stock data is now bundled in the /api/watchlist response — no per-row fetch.
  const meetsTarget = item.target_price && item.current_price != null && item.current_price >= item.target_price
  const meetsScoreAlert = item.alert_score && item.canslim_score != null && item.canslim_score >= item.alert_score
  const hasData = item.current_price != null || item.canslim_score != null

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
          {item.name && (
            <div className="text-dark-400 text-sm truncate max-w-[160px] sm:max-w-[240px]">
              {item.name}
            </div>
          )}
        </Link>

        <div className="text-right flex flex-col items-end gap-1">
          {hasData ? (
            <>
              <span className="font-semibold font-data text-dark-50">{formatCurrency(item.current_price)}</span>
              <ScoreBadge score={item.canslim_score} ticker={item.ticker} size="xs" />
            </>
          ) : (
            <div className="text-dark-500 text-sm">No data</div>
          )}
        </div>
      </div>

      {/* Alerts & Notes */}
      {(item.target_price || item.alert_score) && (
        <div className="flex items-center gap-2 sm:gap-4 mt-2 text-sm">
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

function AddWatchlistForm({ onClose, onAdd, onBulkAdd }) {
  const [mode, setMode] = useState('single')   // 'single' | 'bulk'
  const [ticker, setTicker] = useState('')
  const [bulkText, setBulkText] = useState('')
  const [targetPrice, setTargetPrice] = useState('')
  const [alertScore, setAlertScore] = useState('')
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (submitting) return

    if (mode === 'bulk') {
      if (!bulkText.trim()) return
      setSubmitting(true)
      try {
        await onBulkAdd(bulkText)
        onClose()
      } finally {
        setSubmitting(false)
      }
      return
    }

    if (!ticker) return
    setSubmitting(true)
    try {
      await onAdd({
        ticker: ticker.toUpperCase(),
        target_price: targetPrice ? parseFloat(targetPrice) : null,
        alert_score: alertScore ? parseFloat(alertScore) : null,
        notes: notes || null,
      })
      onClose()
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Mode tabs */}
      <div className="flex gap-1 border-b border-dark-700/40">
        {[
          { id: 'single', label: 'Single' },
          { id: 'bulk', label: 'Bulk paste' },
        ].map(t => (
          <button
            key={t.id}
            type="button"
            onClick={() => setMode(t.id)}
            className={`text-xs font-medium px-3 py-1.5 border-b-2 transition-colors ${
              mode === t.id
                ? 'text-primary-400 border-primary-500'
                : 'text-dark-400 border-transparent hover:text-dark-200'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {mode === 'single' ? (
        <>
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
        </>
      ) : (
        <div>
          <label className="text-[10px] uppercase tracking-wider text-dark-400 font-semibold">Paste tickers</label>
          <textarea
            value={bulkText}
            onChange={(e) => setBulkText(e.target.value)}
            placeholder="AAPL, MSFT, NVDA&#10;TSLA GOOGL&#10;META AMZN"
            className="w-full mt-1 h-32 font-data text-sm"
            required
          />
          <p className="text-[11px] text-dark-500 mt-1">
            Separate by commas, spaces, or newlines. Existing tickers are skipped.
            Target price, score alert, and notes can be added per-ticker later.
          </p>
        </div>
      )}

      <button type="submit" disabled={submitting} className="w-full btn-primary disabled:opacity-50">
        {submitting
          ? 'Adding...'
          : mode === 'bulk' ? 'Bulk Import' : 'Add to Watchlist'}
      </button>
    </form>
  )
}

export default function Watchlist() {
  const [loading, setLoading] = useState(true)
  const [items, setItems] = useState([])
  const [asOf, setAsOf] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const toast = useToast()

  const fetchWatchlist = async () => {
    try {
      setLoading(true)
      const data = await api.getWatchlist()
      setItems(data.items || [])
      setAsOf(data.as_of || null)
    } catch (err) {
      console.error('Failed to fetch watchlist:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWatchlist()
  }, [])

  // Split into active-alerts (target/score hit) and the rest. useMemo so the
  // partition is stable across renders — alertItems referenced by the prev/next
  // saveStockListContext effect below.
  const { alertItems, restItems } = useMemo(() => {
    const alertItems = []
    const restItems = []
    for (const item of items) {
      if (hasActiveAlert(item)) alertItems.push(item)
      else restItems.push(item)
    }
    return { alertItems, restItems }
  }, [items])

  // Surface the same alert-first order to StockDetail so its prev/next nav
  // walks the list the way the user sees it.
  useEffect(() => {
    if (items.length > 0) {
      const ordered = [...alertItems, ...restItems].map(i => i.ticker)
      saveStockListContext('Watchlist', ordered)
    }
  }, [alertItems, restItems, items.length])

  const handleAdd = async (item) => {
    try {
      await api.addToWatchlist(item)
      toast.success(`${item.ticker} added to watchlist`)
      fetchWatchlist()
    } catch (err) {
      console.error('Failed to add to watchlist:', err)
      toast.error(err.message || 'Failed to add to watchlist')
    }
  }

  const handleBulkAdd = async (tickersText) => {
    try {
      const result = await api.bulkAddToWatchlist(tickersText)
      const parts = []
      if (result.added?.length) parts.push(`${result.added.length} added`)
      if (result.skipped?.length) parts.push(`${result.skipped.length} already there`)
      if (result.invalid?.length) parts.push(`${result.invalid.length} invalid`)
      const summary = parts.length ? parts.join(' · ') : 'No tickers processed'
      if (result.added?.length) {
        toast.success(`Watchlist import: ${summary}`)
      } else {
        toast.info(`Watchlist import: ${summary}`)
      }
      fetchWatchlist()
    } catch (err) {
      console.error('Failed to bulk import:', err)
      toast.error(err.message || 'Bulk import failed')
    }
  }

  const handleRemove = async (id) => {
    if (!confirm('Remove from watchlist?')) return
    try {
      await api.removeFromWatchlist(id)
      setItems(prev => prev.filter(i => i.id !== id))
      toast.success('Removed from watchlist')
    } catch (err) {
      console.error('Failed to remove:', err)
      toast.error(err.message || 'Failed to remove')
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
        subtitle={asOf ? `Scores updated ${formatRelativeTime(asOf)}` : undefined}
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
        <>
          {alertItems.length > 0 && (
            <>
              <SectionLabel>
                <span className="text-amber-400">
                  Active Alerts &middot; {alertItems.length}
                </span>
              </SectionLabel>
              <Card variant="glass" padding="" className="mb-4 !border-amber-500/20 bg-amber-500/[0.03]">
                {alertItems.map(item => (
                  <WatchlistItem
                    key={`alert-${item.id}`}
                    item={item}
                    onRemove={handleRemove}
                  />
                ))}
              </Card>
              {restItems.length > 0 && <SectionLabel>Watching &middot; {restItems.length}</SectionLabel>}
            </>
          )}
          {restItems.length > 0 && (
            <Card variant="glass" padding="">
              {restItems.map(item => (
                <WatchlistItem
                  key={item.id}
                  item={item}
                  onRemove={handleRemove}
                />
              ))}
            </Card>
          )}
        </>
      )}

      <Modal
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        title="Add to Watchlist"
      >
        <AddWatchlistForm
          onClose={() => setShowAddModal(false)}
          onAdd={handleAdd}
          onBulkAdd={handleBulkAdd}
        />
      </Modal>

      <div className="h-4" />
    </div>
  )
}
