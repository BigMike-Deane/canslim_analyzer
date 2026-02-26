import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, formatDate, formatRelativeTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ActionBadge, TagBadge, PnlText, ScoreBadge } from '../components/Badge'
import PageHeader from '../components/PageHeader'
import { useToast } from '../components/Toast'

function UploadZone({ label, hint, accept, onUpload, uploading }) {
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) onUpload(file)
  }, [onUpload])

  const handleFile = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) onUpload(file)
    e.target.value = ''
  }, [onUpload])

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-xl p-4 text-center transition-all cursor-pointer
        ${dragOver
          ? 'border-primary-500 bg-primary-500/5'
          : 'border-dark-600 hover:border-dark-500 bg-dark-800/30'
        } ${uploading ? 'opacity-50 pointer-events-none' : ''}`}
      onClick={() => document.getElementById(`file-${label}`)?.click()}
    >
      <input
        id={`file-${label}`}
        type="file"
        accept={accept}
        onChange={handleFile}
        className="hidden"
      />
      <div className="flex items-center gap-3">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-dark-400 shrink-0">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
          <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="text-left">
          <span className="text-sm font-medium text-dark-200">{label}</span>
          <p className="text-[11px] text-dark-500">{hint}</p>
        </div>
        {uploading && (
          <div className="text-xs text-primary-400 animate-pulse ml-auto">Uploading...</div>
        )}
      </div>
    </div>
  )
}

function PositionsTable({ positions, cashBalance }) {
  if (!positions?.length) return null

  const totalValue = positions.reduce((s, p) => s + (p.current_value || 0), 0) + (cashBalance || 0)

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-dark-500 text-[10px] uppercase tracking-wider border-b border-dark-700/40">
            <th className="text-left py-2 px-2 font-semibold">Symbol</th>
            <th className="text-right py-2 px-2 font-semibold">Score</th>
            <th className="text-right py-2 px-2 font-semibold">Qty</th>
            <th className="text-right py-2 px-2 font-semibold">Price</th>
            <th className="text-right py-2 px-2 font-semibold">Value</th>
            <th className="text-right py-2 px-2 font-semibold">P&L</th>
            <th className="text-right py-2 px-2 font-semibold hidden sm:table-cell">% Acct</th>
          </tr>
        </thead>
        <tbody>
          {cashBalance > 0 && (
            <tr className="border-b border-dark-700/20 text-dark-400">
              <td className="py-2 px-2 font-medium">CASH</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2 font-data text-dark-200">{formatCurrency(cashBalance)}</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2 font-data hidden sm:table-cell">
                {totalValue > 0 ? `${(cashBalance / totalValue * 100).toFixed(1)}%` : '-'}
              </td>
            </tr>
          )}
          {positions.map((p) => {
            const effectiveScore = p.is_growth_stock && p.growth_mode_score
              ? p.growth_mode_score
              : p.canslim_score

            return (
              <tr key={p.symbol} className="border-b border-dark-700/20 hover:bg-dark-800/30 transition-colors">
                <td className="py-2 px-2">
                  <div className="flex items-center gap-1.5">
                    <Link to={`/stock/${p.symbol}`} className="font-medium text-dark-100 hover:text-primary-400 transition-colors">
                      {p.symbol}
                    </Link>
                    {p.is_growth_stock && (
                      <TagBadge color="purple" className="text-[8px] px-1 py-0">G</TagBadge>
                    )}
                  </div>
                  {p.sector && (
                    <div className="text-[10px] text-dark-600 truncate max-w-[100px]">{p.sector}</div>
                  )}
                </td>
                <td className="text-right py-2 px-2">
                  {effectiveScore != null ? (
                    <ScoreBadge score={effectiveScore} size="xs" />
                  ) : (
                    <span className="text-dark-600 text-[10px]">-</span>
                  )}
                </td>
                <td className="text-right py-2 px-2 font-data text-dark-300">{p.quantity}</td>
                <td className="text-right py-2 px-2 font-data text-dark-300">{formatCurrency(p.last_price)}</td>
                <td className="text-right py-2 px-2 font-data text-dark-200">{formatCurrency(p.current_value)}</td>
                <td className="text-right py-2 px-2">
                  <PnlText value={p.total_gain_loss_pct} className="text-xs" />
                  {p.total_gain_loss_pct != null && <span className="text-dark-600">%</span>}
                </td>
                <td className="text-right py-2 px-2 font-data text-dark-400 hidden sm:table-cell">
                  {p.percent_of_account != null ? `${p.percent_of_account.toFixed(1)}%` : '-'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function TradesTable({ trades }) {
  if (!trades?.length) return (
    <p className="text-xs text-dark-500 text-center py-6">No trades uploaded yet</p>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-dark-500 text-[10px] uppercase tracking-wider border-b border-dark-700/40">
            <th className="text-left py-2 px-2 font-semibold">Date</th>
            <th className="text-left py-2 px-2 font-semibold">Action</th>
            <th className="text-left py-2 px-2 font-semibold">Symbol</th>
            <th className="text-right py-2 px-2 font-semibold">Qty</th>
            <th className="text-right py-2 px-2 font-semibold">Price</th>
            <th className="text-right py-2 px-2 font-semibold">Amount</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t, i) => (
            <tr key={i} className="border-b border-dark-700/20 hover:bg-dark-800/30 transition-colors">
              <td className="py-2 px-2 font-data text-dark-400">{t.run_date}</td>
              <td className="py-2 px-2"><ActionBadge action={t.action} /></td>
              <td className="py-2 px-2">
                <Link to={`/stock/${t.symbol}`} className="font-medium text-dark-100 hover:text-primary-400 transition-colors">
                  {t.symbol}
                </Link>
              </td>
              <td className="text-right py-2 px-2 font-data text-dark-300">{t.quantity}</td>
              <td className="text-right py-2 px-2 font-data text-dark-300">{formatCurrency(t.price)}</td>
              <td className="text-right py-2 px-2 font-data text-dark-200">{formatCurrency(t.amount ? Math.abs(t.amount) : null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
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
          {action.is_growth_stock && (
            <TagBadge color="purple">Growth</TagBadge>
          )}
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
        <div className="flex flex-wrap items-center gap-3 sm:gap-4 mb-3 p-2.5 bg-dark-850/50 rounded-lg border border-dark-700/30">
          <div>
            <div className="text-dark-500 text-[10px]">Shares to {action.action === 'SELL' || action.action === 'TRIM' ? 'Sell' : 'Buy'}</div>
            <div className="font-bold text-lg font-data text-dark-50">{action.shares_action}</div>
          </div>
          {action.shares_current > 0 && (
            <div>
              <div className="text-dark-500 text-[10px]">Current</div>
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

function GameplanSection({ gameplan, loading, onAddToWatchlist }) {
  if (loading) {
    return (
      <div>
        <div className="skeleton h-6 w-32 mb-3" />
        <div className="skeleton h-32 rounded-2xl mb-3" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  if (!gameplan || gameplan.length === 0) {
    return (
      <Card variant="glass" className="text-center py-6">
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
    <div>
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

function ReconciliationView({ data }) {
  if (!data) return null

  const { matches, fidelity_only, ai_only, discrepancies, summary, snapshot_date } = data

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-dark-800/50 rounded-lg p-3 border border-dark-700/30">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider">Fidelity Value</div>
          <div className="text-lg font-bold font-data text-dark-100 mt-1">{formatCurrency(summary.fidelity_total)}</div>
        </div>
        <div className="bg-dark-800/50 rounded-lg p-3 border border-dark-700/30">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider">AI Portfolio Value</div>
          <div className="text-lg font-bold font-data text-dark-100 mt-1">{formatCurrency(summary.ai_total)}</div>
        </div>
        <div className="bg-dark-800/50 rounded-lg p-3 border border-dark-700/30">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider">Overlap</div>
          <div className="text-lg font-bold font-data text-primary-400 mt-1">
            {summary.overlap_count}/{summary.total_unique_symbols}
            <span className="text-xs text-dark-500 ml-1">({summary.overlap_pct}%)</span>
          </div>
        </div>
        <div className="bg-dark-800/50 rounded-lg p-3 border border-dark-700/30">
          <div className="text-[10px] text-dark-500 uppercase tracking-wider">Snapshot</div>
          <div className="text-sm font-medium text-dark-200 mt-1">{snapshot_date || '-'}</div>
        </div>
      </div>

      {matches.length > 0 && (
        <Card variant="glass">
          <CardHeader title={`Matching Positions (${matches.length})`} />
          <div className="space-y-1">
            {matches.map(m => (
              <div key={m.symbol} className="flex items-center justify-between py-1.5 border-b border-dark-700/20 last:border-0">
                <div className="flex items-center gap-2">
                  <TagBadge color="green">MATCH</TagBadge>
                  <Link to={`/stock/${m.symbol}`} className="text-sm font-medium text-dark-100 hover:text-primary-400">
                    {m.symbol}
                  </Link>
                  <span className="text-xs text-dark-500 font-data">{m.fidelity_shares} shares</span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-data text-dark-300">{formatCurrency(m.fidelity_value)}</span>
                  <PnlText value={m.fidelity_gain_pct} className="text-[11px] ml-2" />
                  {m.fidelity_gain_pct != null && <span className="text-dark-600 text-[11px]">%</span>}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {discrepancies.length > 0 && (
        <Card variant="accent" accent="amber">
          <CardHeader title={`Quantity Mismatches (${discrepancies.length})`} />
          <div className="space-y-1">
            {discrepancies.map(d => (
              <div key={d.symbol} className="flex items-center justify-between py-1.5 border-b border-dark-700/20 last:border-0">
                <div className="flex items-center gap-2">
                  <TagBadge color="amber">DIFF</TagBadge>
                  <Link to={`/stock/${d.symbol}`} className="text-sm font-medium text-dark-100 hover:text-primary-400">
                    {d.symbol}
                  </Link>
                </div>
                <div className="text-right text-xs font-data">
                  <span className="text-dark-400">Fidelity: {d.fidelity_shares}</span>
                  <span className="text-dark-600 mx-1">|</span>
                  <span className="text-dark-400">AI: {d.ai_shares}</span>
                  <span className="text-amber-400 ml-2">({d.share_diff > 0 ? '+' : ''}{d.share_diff})</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {fidelity_only.length > 0 && (
        <Card variant="accent" accent="cyan">
          <CardHeader title={`Fidelity Only (${fidelity_only.length})`} subtitle="Positions in Fidelity but not in AI portfolio" />
          <div className="space-y-1">
            {fidelity_only.map(f => (
              <div key={f.symbol} className="flex items-center justify-between py-1.5 border-b border-dark-700/20 last:border-0">
                <div className="flex items-center gap-2">
                  <TagBadge color="cyan">FID</TagBadge>
                  <Link to={`/stock/${f.symbol}`} className="text-sm font-medium text-dark-100 hover:text-primary-400">
                    {f.symbol}
                  </Link>
                  <span className="text-xs text-dark-500 font-data">{f.shares} shares</span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-data text-dark-300">{formatCurrency(f.current_value)}</span>
                  <PnlText value={f.gain_pct} className="text-[11px] ml-2" />
                  {f.gain_pct != null && <span className="text-dark-600 text-[11px]">%</span>}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {ai_only.length > 0 && (
        <Card variant="accent" accent="purple">
          <CardHeader title={`AI Only (${ai_only.length})`} subtitle="AI recommended positions not in Fidelity" />
          <div className="space-y-1">
            {ai_only.map(a => (
              <div key={a.symbol} className="flex items-center justify-between py-1.5 border-b border-dark-700/20 last:border-0">
                <div className="flex items-center gap-2">
                  <TagBadge color="purple">AI</TagBadge>
                  <Link to={`/stock/${a.symbol}`} className="text-sm font-medium text-dark-100 hover:text-primary-400">
                    {a.symbol}
                  </Link>
                  <span className="text-xs text-dark-500 font-data">{a.shares} shares</span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-data text-dark-300">{formatCurrency(a.current_value)}</span>
                  {a.score != null && (
                    <span className="text-xs text-dark-400 ml-2">Score: {Math.round(a.score)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

export default function FidelitySync() {
  const { addToast } = useToast()
  const [tab, setTab] = useState('positions')
  const [uploading, setUploading] = useState(null)
  const [latestSnapshot, setLatestSnapshot] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [reconciliation, setReconciliation] = useState(null)
  const [gameplan, setGameplan] = useState([])
  const [gameplanLoading, setGameplanLoading] = useState(true)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)
  const [showUpload, setShowUpload] = useState(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [latestRes, tradesRes] = await Promise.all([
        api.getFidelityLatest(),
        api.getFidelityTrades(50),
      ])
      setLatestSnapshot(latestRes.snapshot)
      setPositions(latestRes.positions || [])
      setTrades(tradesRes.trades || [])

      // Load reconciliation and gameplan if we have data
      if (latestRes.snapshot) {
        const promises = [api.getFidelityGameplan()]
        try {
          promises.push(api.getFidelityReconciliation())
        } catch { /* no AI portfolio */ }

        const results = await Promise.allSettled(promises)
        if (results[0].status === 'fulfilled') {
          setGameplan(results[0].value.gameplan || [])
        }
        if (results[1]?.status === 'fulfilled') {
          setReconciliation(results[1].value)
        }
        setGameplanLoading(false)
      }
    } catch (err) {
      console.error('Failed to load Fidelity data:', err)
    } finally {
      setLoading(false)
      setGameplanLoading(false)
    }
  }, [])

  useEffect(() => { loadData() }, [loadData])

  const handleUploadPositions = async (file) => {
    setUploading('positions')
    try {
      const result = await api.uploadFidelityPositions(file)
      addToast(`Uploaded ${result.positions_count} positions (${formatCurrency(result.total_value)})`, 'success')
      loadData()
    } catch (err) {
      addToast(err.message || 'Upload failed', 'error')
    } finally {
      setUploading(null)
    }
  }

  const handleUploadActivity = async (file) => {
    setUploading('activity')
    try {
      const result = await api.uploadFidelityActivity(file)
      addToast(`Imported ${result.new_trades} new trades (${result.skipped_duplicates} duplicates skipped)`, 'success')
      loadData()
    } catch (err) {
      addToast(err.message || 'Upload failed', 'error')
    } finally {
      setUploading(null)
    }
  }

  const handleSyncToPortfolio = async () => {
    if (!latestSnapshot) return
    setSyncing(true)
    try {
      const result = await api.syncFidelityToPortfolio()
      addToast(`Portfolio synced: ${result.added} added, ${result.updated} updated, ${result.removed} removed`, 'success')
      loadData()
    } catch (err) {
      addToast(err.message || 'Sync failed', 'error')
    } finally {
      setSyncing(false)
    }
  }

  const handleAddToWatchlist = async (ticker, reason) => {
    await api.addToWatchlist({
      ticker,
      notes: reason || 'Added from Fidelity gameplan'
    })
  }

  // Calculate portfolio-level P&L
  const totalGainLoss = positions.reduce((s, p) => s + (p.total_gain_loss || 0), 0)
  const totalCostBasis = positions.reduce((s, p) => s + (p.cost_basis_total || 0), 0)
  const totalGainPct = totalCostBasis > 0 ? (totalGainLoss / totalCostBasis * 100) : 0

  const tabs = [
    { id: 'gameplan', label: 'Gameplan', badge: gameplan.length || null },
    { id: 'positions', label: 'Positions', badge: positions.length || null },
    { id: 'trades', label: 'Trades', badge: trades.length || null },
    { id: 'reconciliation', label: 'Recon' },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="My Portfolio"
        subtitle="Fidelity holdings with AI-powered trade recommendations"
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowUpload(!showUpload)}
              className="text-xs px-3 py-1.5 rounded-lg bg-dark-700/50 text-dark-300 border border-dark-600/30 hover:bg-dark-600/50 transition-colors"
            >
              {showUpload ? 'Hide Upload' : 'Upload CSV'}
            </button>
            {latestSnapshot && (
              <button
                onClick={handleSyncToPortfolio}
                disabled={syncing}
                className="text-xs px-3 py-1.5 rounded-lg bg-primary-600/20 text-primary-400 border border-primary-500/30 hover:bg-primary-600/30 transition-colors disabled:opacity-50"
              >
                {syncing ? 'Syncing...' : 'Sync to AI Portfolio'}
              </button>
            )}
          </div>
        }
      />

      {/* Collapsible Upload Zones */}
      {showUpload && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <UploadZone
            label="Positions CSV"
            hint="Portfolio_Positions_*.csv"
            accept=".csv"
            onUpload={handleUploadPositions}
            uploading={uploading === 'positions'}
          />
          <UploadZone
            label="Activity CSV"
            hint="Accounts_History*.csv"
            accept=".csv"
            onUpload={handleUploadActivity}
            uploading={uploading === 'activity'}
          />
        </div>
      )}

      {/* Portfolio Summary */}
      {latestSnapshot && (
        <Card variant="glass" className="mb-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-dark-500 text-[10px] font-semibold tracking-wider uppercase">Portfolio Value</div>
              <div className="text-2xl font-bold font-data text-dark-50 mt-1">{formatCurrency(latestSnapshot.total_value)}</div>
              <div className="flex items-center gap-3 mt-1">
                <div className={`text-sm font-data ${totalGainLoss >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {totalGainLoss >= 0 ? '+' : ''}{formatCurrency(Math.abs(totalGainLoss))}
                  <span className="text-xs opacity-70 ml-1">({totalGainPct >= 0 ? '+' : ''}{totalGainPct.toFixed(1)}%)</span>
                </div>
                <span className="text-dark-600">|</span>
                <span className="text-xs text-dark-400">
                  {latestSnapshot.positions_count} positions + {formatCurrency(latestSnapshot.cash_balance)} cash
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-dark-500">{latestSnapshot.snapshot_date}</div>
              <div className="text-[11px] text-dark-600 mt-0.5">
                {formatRelativeTime(latestSnapshot.uploaded_at)}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Tab Bar */}
      <div className="flex gap-1 mb-4 border-b border-dark-700/40 overflow-x-auto">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-xs font-medium px-4 py-2 border-b-2 transition-colors whitespace-nowrap ${
              tab === t.id
                ? 'text-primary-400 border-primary-500'
                : 'text-dark-400 border-transparent hover:text-dark-200'
            }`}
          >
            {t.label}
            {t.badge != null && (
              <span className="ml-1.5 text-[10px] text-dark-500">{t.badge}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {loading ? (
        <div className="text-center py-12 text-dark-500 text-sm animate-pulse">Loading...</div>
      ) : !latestSnapshot ? (
        <Card variant="default" className="text-center py-12">
          <p className="text-dark-500 text-sm">No positions uploaded yet</p>
          <p className="text-dark-600 text-xs mt-1">Upload a Fidelity Positions CSV to get started</p>
          {!showUpload && (
            <button
              onClick={() => setShowUpload(true)}
              className="mt-4 bg-primary-500 hover:bg-primary-400 text-dark-950 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
            >
              Upload CSV
            </button>
          )}
        </Card>
      ) : (
        <>
          {tab === 'gameplan' && (
            <GameplanSection
              gameplan={gameplan}
              loading={gameplanLoading}
              onAddToWatchlist={handleAddToWatchlist}
            />
          )}

          {tab === 'positions' && (
            positions.length > 0 ? (
              <Card variant="default">
                <PositionsTable positions={positions} cashBalance={latestSnapshot?.cash_balance} />
              </Card>
            ) : (
              <Card variant="default" className="text-center py-12">
                <p className="text-dark-500 text-sm">No positions in latest snapshot</p>
              </Card>
            )
          )}

          {tab === 'trades' && (
            <Card variant="default">
              <TradesTable trades={trades} />
            </Card>
          )}

          {tab === 'reconciliation' && (
            reconciliation ? (
              <ReconciliationView data={reconciliation} />
            ) : (
              <Card variant="default" className="text-center py-12">
                <p className="text-dark-500 text-sm">Upload positions to see reconciliation</p>
                <p className="text-dark-600 text-xs mt-1">Compares your Fidelity holdings against AI portfolio recommendations</p>
              </Card>
            )
          )}
        </>
      )}
    </div>
  )
}
