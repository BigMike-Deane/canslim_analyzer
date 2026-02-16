import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, formatCurrency, formatPercent } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, ActionBadge, TagBadge } from '../components/Badge'
import StatGrid, { StatRow } from '../components/StatGrid'
import Modal from '../components/Modal'
import PageHeader from '../components/PageHeader'

function PortfolioSummary({ positions }) {
  if (!positions || positions.length === 0) return null

  const totalValue = positions.reduce((sum, p) => sum + (p.current_value || 0), 0)
  const totalCost = positions.reduce((sum, p) => sum + (p.cost_basis * p.shares || 0), 0)
  const totalGain = totalValue - totalCost
  const totalGainPct = totalCost > 0 ? (totalGain / totalCost * 100) : 0

  const summaryStats = [
    { label: 'Positions', value: positions.length },
    { label: 'Cost Basis', value: formatCurrency(totalCost) },
    {
      label: 'Avg Score',
      value: formatScore(positions.reduce((sum, p) => sum + (p.canslim_score || 0), 0) / positions.length),
    },
  ]

  return (
    <Card variant="glass" className="mb-4">
      <div className="text-dark-400 text-[10px] font-semibold tracking-wider uppercase mb-1">Portfolio Value</div>
      <div className="text-3xl font-bold font-data text-dark-50 mb-1">{formatCurrency(totalValue)}</div>
      <div className={`text-sm flex items-center gap-1 ${totalGain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        <span className="font-data">{totalGain >= 0 ? '+' : ''}{formatCurrency(Math.abs(totalGain))}</span>
        <span className="font-data">({formatPercent(Math.abs(totalGainPct))})</span>
        <span className="text-dark-500 ml-1 text-xs">total</span>
      </div>

      <div className="mt-4 pt-4 border-t border-dark-700/50">
        <StatGrid stats={summaryStats} columns={3} />
      </div>
    </Card>
  )
}

function PositionRow({ position, onDelete, onEdit }) {
  const gainLoss = position.current_value - (position.cost_basis * position.shares)
  const gainLossPct = position.cost_basis > 0
    ? ((position.current_price / position.cost_basis - 1) * 100)
    : 0
  const isPositive = gainLoss >= 0

  return (
    <div className="border-b border-dark-700/30 last:border-0 py-3">
      <div className="flex justify-between items-start">
        <Link to={`/stock/${position.ticker}`} className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-dark-100">{position.ticker}</span>
            {position.is_growth_stock && (
              <TagBadge color="purple">Growth</TagBadge>
            )}
            {position.recommendation && (
              <ActionBadge action={position.recommendation.toUpperCase()} />
            )}
          </div>
          <div className="text-dark-400 text-xs font-data mt-0.5">
            {position.shares} shares @ {formatCurrency(position.cost_basis)}
          </div>
        </Link>

        <div className="text-right">
          <div className="font-semibold font-data text-dark-100">{formatCurrency(position.current_value)}</div>
          <div className={`text-sm font-data ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
            {isPositive ? '+' : ''}{formatCurrency(gainLoss)}
            <span className="text-xs ml-1 opacity-70">({formatPercent(gainLossPct, true)})</span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-2">
        <div className="flex items-center gap-3 text-sm">
          {/* Show primary score based on stock type */}
          {position.is_growth_stock ? (
            <div className="flex items-center gap-1.5">
              <span className="text-purple-400 text-xs">Growth:</span>
              <ScoreBadge score={position.growth_mode_score} size="xs" />
              {position.canslim_score > 0 && (
                <span className="text-dark-500 text-[10px] font-data ml-1">
                  (CANSLIM: {position.canslim_score?.toFixed(0)})
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <span className="text-dark-400 text-xs">CANSLIM:</span>
              <ScoreBadge score={position.canslim_score} size="xs" />
              {position.growth_mode_score > 0 && (
                <span className="text-purple-400 text-[10px] font-data ml-1">
                  (Growth: {position.growth_mode_score?.toFixed(0)})
                </span>
              )}
            </div>
          )}
          {position.data_quality === 'low' && (
            <span
              className="text-amber-500 text-[10px] cursor-help"
              title="Limited analyst data - projections may be less reliable"
            >
              Low Data
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onEdit(position)}
            className="text-primary-400 text-xs hover:text-primary-300 transition-colors"
          >
            Edit
          </button>
          <button
            onClick={() => onDelete(position.id)}
            className="text-red-400 text-xs hover:text-red-300 transition-colors"
          >
            Remove
          </button>
        </div>
      </div>

      {position.notes && (
        <div className="text-dark-500 text-[10px] mt-2 italic">
          {position.notes}
        </div>
      )}
    </div>
  )
}

const gameplanAccentColors = {
  SELL: 'red',
  TRIM: 'amber',
  BUY: 'green',
  ADD: 'green',
  WATCH: 'purple',
}

function GameplanCard({ action, onAddToWatchlist }) {
  const [adding, setAdding] = useState(false)
  const [added, setAdded] = useState(false)

  const accentColor = gameplanAccentColors[action.action] || 'purple'

  const handleAddToWatchlist = async () => {
    if (adding || added) return
    setAdding(true)
    try {
      await onAddToWatchlist(action.ticker, action.reason)
      setAdded(true)
    } catch (err) {
      console.error('Failed to add to watchlist:', err)
    } finally {
      setAdding(false)
    }
  }

  return (
    <Card variant="accent" accent={accentColor} className="mb-3">
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          <ActionBadge action={action.action} />
          <Link to={`/stock/${action.ticker}`} className="font-semibold text-lg text-dark-100 hover:text-primary-400 transition-colors">
            {action.ticker}
          </Link>
        </div>
        {action.estimated_value > 0 && (
          <div className="text-right">
            <div className="text-dark-500 text-[10px]">Est. Value</div>
            <div className="font-semibold font-data text-dark-100">{formatCurrency(action.estimated_value)}</div>
          </div>
        )}
      </div>

      <div className="mb-3">
        <div className="font-medium text-sm text-dark-200">{action.reason}</div>
      </div>

      {action.shares_action > 0 && (
        <div className="flex items-center gap-4 mb-3 p-2.5 bg-dark-850/50 rounded-lg border border-dark-700/30">
          <div>
            <div className="text-dark-500 text-[10px]">Shares to {action.action === 'SELL' || action.action === 'TRIM' ? 'Sell' : 'Buy'}</div>
            <div className="font-bold text-lg font-data text-dark-50">{action.shares_action}</div>
          </div>
          {action.shares_current > 0 && (
            <div>
              <div className="text-dark-500 text-[10px]">Current Position</div>
              <div className="font-semibold font-data text-dark-200">{action.shares_current} shares</div>
            </div>
          )}
          <div>
            <div className="text-dark-500 text-[10px]">Price</div>
            <div className="font-semibold font-data text-dark-200">{formatCurrency(action.current_price)}</div>
          </div>
        </div>
      )}

      <div className="space-y-1">
        {action.details?.filter(d => d).map((detail, i) => (
          <div key={i} className="text-dark-300 text-xs flex items-start gap-2">
            <span className="text-dark-600 mt-0.5">&#8226;</span>
            <span>{detail}</span>
          </div>
        ))}
      </div>

      {action.action === 'WATCH' && onAddToWatchlist && (
        <button
          onClick={handleAddToWatchlist}
          disabled={adding || added}
          className={`mt-3 w-full py-2 rounded-lg text-xs font-semibold transition-colors border ${
            added
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 cursor-default'
              : 'bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 border-purple-500/20'
          }`}
        >
          {added ? 'Added to Watchlist' : adding ? 'Adding...' : '+ Add to Watchlist'}
        </button>
      )}
    </Card>
  )
}

function Gameplan({ gameplan, loading, onAddToWatchlist }) {
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
      <Card variant="glass" className="mb-4 text-center py-6">
        <div className="font-semibold text-dark-200 mb-1">No Actions Needed</div>
        <div className="text-dark-400 text-xs">Your portfolio looks good! Check back after running a scan.</div>
      </Card>
    )
  }

  const sellActions = gameplan.filter(a => a.action === 'SELL')
  const trimActions = gameplan.filter(a => a.action === 'TRIM')
  const buyActions = gameplan.filter(a => a.action === 'BUY')
  const addActions = gameplan.filter(a => a.action === 'ADD')
  const watchActions = gameplan.filter(a => a.action === 'WATCH')

  const sections = [
    { key: 'sell', actions: sellActions, type: 'SELL' },
    { key: 'trim', actions: trimActions, type: 'TRIM' },
    { key: 'buy', actions: buyActions, type: 'BUY' },
    { key: 'add', actions: addActions, type: 'ADD' },
    { key: 'watch', actions: watchActions, type: 'WATCH' },
  ]

  return (
    <div className="mb-4">
      <SectionLabel>Gameplan</SectionLabel>

      {sections
        .filter(s => s.actions.length > 0)
        .map(({ key, actions, type }) => (
          <div key={key} className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <ActionBadge action={type} />
              <span className="text-[10px] text-dark-500 font-data">{actions.length}</span>
            </div>
            {actions.map((action, i) => (
              <GameplanCard
                key={`${key}-${i}`}
                action={action}
                onAddToWatchlist={type === 'WATCH' ? onAddToWatchlist : undefined}
              />
            ))}
          </div>
        ))}
    </div>
  )
}

function AddPositionModal({ open, onClose, onAdd }) {
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

  const inputCls = 'w-full bg-dark-850 border border-dark-700/50 rounded-lg px-3 py-2 text-sm font-data text-dark-100 focus:outline-none focus:border-primary-500/50 transition-colors'
  const labelCls = 'text-[10px] font-semibold tracking-wider uppercase text-dark-400 block mb-1'

  return (
    <Modal open={open} onClose={onClose} title="Add Position">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className={labelCls}>Ticker Symbol</label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="AAPL"
            className={inputCls}
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Shares</label>
            <input
              type="number"
              step="0.001"
              value={shares}
              onChange={(e) => setShares(e.target.value)}
              placeholder="10"
              className={inputCls}
              required
            />
          </div>
          <div>
            <label className={labelCls}>Cost per Share</label>
            <input
              type="number"
              step="0.01"
              value={costBasis}
              onChange={(e) => setCostBasis(e.target.value)}
              placeholder="150.00"
              className={inputCls}
            />
          </div>
        </div>

        <div>
          <label className={labelCls}>Notes (optional)</label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Entry reason, target price, etc."
            className={inputCls}
          />
        </div>

        <button
          type="submit"
          className="w-full bg-primary-500 hover:bg-primary-400 rounded-lg py-2.5 text-sm font-semibold transition-colors"
        >
          Add Position
        </button>
      </form>
    </Modal>
  )
}

function EditPositionModal({ open, position, onClose, onSave }) {
  const [shares, setShares] = useState(position?.shares?.toString() || '')
  const [costBasis, setCostBasis] = useState(position?.cost_basis?.toString() || '')
  const [notes, setNotes] = useState(position?.notes || '')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (position) {
      setShares(position.shares?.toString() || '')
      setCostBasis(position.cost_basis?.toString() || '')
      setNotes(position.notes || '')
    }
  }, [position])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!shares || parseFloat(shares) <= 0) return

    setSaving(true)
    try {
      await onSave(position.id, {
        shares: parseFloat(shares),
        cost_basis: costBasis ? parseFloat(costBasis) : null,
        notes: notes || null
      })
      onClose()
    } catch (err) {
      console.error('Failed to update position:', err)
    } finally {
      setSaving(false)
    }
  }

  const inputCls = 'w-full bg-dark-850 border border-dark-700/50 rounded-lg px-3 py-2 text-sm font-data text-dark-100 focus:outline-none focus:border-primary-500/50 transition-colors'
  const labelCls = 'text-[10px] font-semibold tracking-wider uppercase text-dark-400 block mb-1'

  return (
    <Modal open={open} onClose={onClose} title={`Edit ${position?.ticker || ''}`}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelCls}>Shares</label>
            <input
              type="number"
              step="0.001"
              min="0.001"
              value={shares}
              onChange={(e) => setShares(e.target.value)}
              placeholder="10"
              className={inputCls}
              required
            />
          </div>
          <div>
            <label className={labelCls}>Cost per Share</label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={costBasis}
              onChange={(e) => setCostBasis(e.target.value)}
              placeholder="150.00"
              className={inputCls}
            />
          </div>
        </div>

        <div>
          <label className={labelCls}>Notes (optional)</label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Entry reason, target price, etc."
            className={inputCls}
          />
        </div>

        {position?.current_price > 0 && shares && costBasis && (
          <Card variant="stat" padding="p-3">
            <div className="text-dark-500 text-[10px] font-semibold tracking-wider uppercase mb-2">Preview</div>
            <div className="grid grid-cols-2 gap-2">
              <StatRow label="Value" value={formatCurrency(parseFloat(shares) * position.current_price)} />
              <StatRow label="Cost" value={formatCurrency(parseFloat(shares) * parseFloat(costBasis))} />
              <div className="col-span-2">
                {(() => {
                  const gain = (parseFloat(shares) * position.current_price) - (parseFloat(shares) * parseFloat(costBasis))
                  const gainPct = ((position.current_price / parseFloat(costBasis)) - 1) * 100
                  return (
                    <StatRow
                      label="Gain/Loss"
                      value={
                        <span className={`font-data text-sm ${gain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {gain >= 0 ? '+' : ''}{formatCurrency(gain)}
                        </span>
                      }
                      sublabel={`${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(2)}%`}
                    />
                  )
                })()}
              </div>
            </div>
          </Card>
        )}

        <div className="flex gap-2">
          <button
            type="button"
            onClick={onClose}
            className="flex-1 bg-dark-700 hover:bg-dark-600 text-dark-200 rounded-lg py-2.5 text-sm font-semibold transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={saving}
            className="flex-1 bg-primary-500 hover:bg-primary-400 disabled:bg-dark-700 disabled:text-dark-500 rounded-lg py-2.5 text-sm font-semibold transition-colors"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </form>
    </Modal>
  )
}

export default function Portfolio() {
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [positions, setPositions] = useState([])
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingPosition, setEditingPosition] = useState(null)
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

  const handleUpdate = async (id, data) => {
    await api.updatePosition(id, data)
    await fetchPortfolio()
  }

  const handleAddToWatchlist = async (ticker, reason) => {
    await api.addToWatchlist({
      ticker,
      notes: reason || `Added from gameplan`
    })
  }

  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-40 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Portfolio"
        actions={
          <>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="text-xs bg-dark-700 hover:bg-dark-600 text-dark-200 px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center gap-1.5"
            >
              {refreshing ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 12a8 8 0 018-8" strokeLinecap="round" />
                  </svg>
                  <span>Refreshing...</span>
                </>
              ) : (
                <>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M1 4v6h6M23 20v-6h-6" />
                    <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
                  </svg>
                  <span>Refresh</span>
                </>
              )}
            </button>
            <button
              onClick={() => setShowAddModal(true)}
              className="text-xs bg-primary-500 hover:bg-primary-400 text-dark-950 px-3 py-1.5 rounded-lg font-semibold transition-colors"
            >
              + Add
            </button>
          </>
        }
      />

      {positions.length === 0 ? (
        <Card variant="glass" className="text-center py-8">
          <div className="font-semibold text-dark-200 mb-2">No Positions Yet</div>
          <div className="text-dark-400 text-xs mb-4">
            Add stocks to your portfolio to track performance and get CANSLIM recommendations.
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="bg-primary-500 hover:bg-primary-400 text-dark-950 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
          >
            Add First Position
          </button>
        </Card>
      ) : (
        <>
          <PortfolioSummary positions={positions} />

          <Card variant="glass" className="mb-4">
            <CardHeader
              title="Positions"
              subtitle={
                <span className="text-amber-500 text-[10px]" title="Limited analyst data">Low data = limited projections</span>
              }
            />
            {[...positions].sort((a, b) => a.ticker.localeCompare(b.ticker)).map(position => (
              <PositionRow
                key={position.id}
                position={position}
                onDelete={handleDelete}
                onEdit={setEditingPosition}
              />
            ))}
          </Card>

          <Gameplan gameplan={gameplan} loading={gameplanLoading} onAddToWatchlist={handleAddToWatchlist} />
        </>
      )}

      <AddPositionModal
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAdd={handleAdd}
      />

      <EditPositionModal
        open={!!editingPosition}
        position={editingPosition}
        onClose={() => setEditingPosition(null)}
        onSave={handleUpdate}
      />

      <div className="h-4" />
    </div>
  )
}
