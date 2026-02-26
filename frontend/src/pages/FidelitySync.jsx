import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, formatDate, formatRelativeTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ActionBadge, TagBadge, PnlText } from '../components/Badge'
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
    e.target.value = '' // Reset so same file can be re-uploaded
  }, [onUpload])

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer
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
      <div className="flex flex-col items-center gap-2">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-dark-400">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
          <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div>
          <span className="text-sm font-medium text-dark-200">{label}</span>
          <p className="text-[11px] text-dark-500 mt-1">{hint}</p>
        </div>
        {uploading && (
          <div className="text-xs text-primary-400 animate-pulse">Uploading...</div>
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
            <th className="text-right py-2 px-2 font-semibold">Qty</th>
            <th className="text-right py-2 px-2 font-semibold">Price</th>
            <th className="text-right py-2 px-2 font-semibold">Value</th>
            <th className="text-right py-2 px-2 font-semibold">Cost Basis</th>
            <th className="text-right py-2 px-2 font-semibold">P&L</th>
            <th className="text-right py-2 px-2 font-semibold">% of Acct</th>
          </tr>
        </thead>
        <tbody>
          {cashBalance > 0 && (
            <tr className="border-b border-dark-700/20 text-dark-400">
              <td className="py-2 px-2 font-medium">CASH</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2 font-data text-dark-200">{formatCurrency(cashBalance)}</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2">-</td>
              <td className="text-right py-2 px-2 font-data">
                {totalValue > 0 ? `${(cashBalance / totalValue * 100).toFixed(1)}%` : '-'}
              </td>
            </tr>
          )}
          {positions.map((p) => (
            <tr key={p.symbol} className="border-b border-dark-700/20 hover:bg-dark-800/30 transition-colors">
              <td className="py-2 px-2">
                <Link to={`/stock/${p.symbol}`} className="font-medium text-dark-100 hover:text-primary-400 transition-colors">
                  {p.symbol}
                </Link>
              </td>
              <td className="text-right py-2 px-2 font-data text-dark-300">{p.quantity}</td>
              <td className="text-right py-2 px-2 font-data text-dark-300">{formatCurrency(p.last_price)}</td>
              <td className="text-right py-2 px-2 font-data text-dark-200">{formatCurrency(p.current_value)}</td>
              <td className="text-right py-2 px-2 font-data text-dark-400">{formatCurrency(p.average_cost_basis)}</td>
              <td className="text-right py-2 px-2">
                <PnlText value={p.total_gain_loss_pct} className="text-xs" />
                {p.total_gain_loss_pct != null && <span className="text-dark-600">%</span>}
              </td>
              <td className="text-right py-2 px-2 font-data text-dark-400">
                {p.percent_of_account != null ? `${p.percent_of_account.toFixed(1)}%` : '-'}
              </td>
            </tr>
          ))}
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

function ReconciliationView({ data }) {
  if (!data) return null

  const { matches, fidelity_only, ai_only, discrepancies, summary, snapshot_date } = data

  return (
    <div className="space-y-4">
      {/* Summary stats */}
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

      {/* Matches */}
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

      {/* Discrepancies */}
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

      {/* Fidelity Only */}
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

      {/* AI Only */}
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
  const [tab, setTab] = useState('positions') // positions, trades, reconciliation
  const [uploading, setUploading] = useState(null) // 'positions' or 'activity'
  const [latestSnapshot, setLatestSnapshot] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [reconciliation, setReconciliation] = useState(null)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)

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

      // Only load reconciliation if we have a snapshot
      if (latestRes.snapshot) {
        try {
          const reconRes = await api.getFidelityReconciliation()
          setReconciliation(reconRes)
        } catch {
          // No AI portfolio might exist yet
        }
      }
    } catch (err) {
      console.error('Failed to load Fidelity data:', err)
    } finally {
      setLoading(false)
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

  const tabs = [
    { id: 'positions', label: 'Positions' },
    { id: 'trades', label: 'Trades' },
    { id: 'reconciliation', label: 'Reconciliation' },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Fidelity Sync"
        subtitle="Import positions and trades from Fidelity CSV exports"
        actions={
          latestSnapshot && (
            <button
              onClick={handleSyncToPortfolio}
              disabled={syncing}
              className="text-xs px-3 py-1.5 rounded-lg bg-primary-600/20 text-primary-400 border border-primary-500/30 hover:bg-primary-600/30 transition-colors disabled:opacity-50"
            >
              {syncing ? 'Syncing...' : 'Sync to Portfolio'}
            </button>
          )
        }
      />

      {/* Upload Zones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <UploadZone
          label="Positions CSV"
          hint="Portfolio_Positions_*.csv from Fidelity"
          accept=".csv"
          onUpload={handleUploadPositions}
          uploading={uploading === 'positions'}
        />
        <UploadZone
          label="Activity CSV"
          hint="Accounts_History*.csv from Fidelity"
          accept=".csv"
          onUpload={handleUploadActivity}
          uploading={uploading === 'activity'}
        />
      </div>

      {/* Snapshot Summary */}
      {latestSnapshot && (
        <Card variant="glass" className="mb-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-dark-500 text-[10px] font-semibold tracking-wider uppercase">Latest Snapshot</div>
              <div className="text-2xl font-bold font-data text-dark-50 mt-1">{formatCurrency(latestSnapshot.total_value)}</div>
              <div className="text-xs text-dark-400 mt-0.5">
                {latestSnapshot.positions_count} positions + {formatCurrency(latestSnapshot.cash_balance)} cash
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-dark-500">{latestSnapshot.snapshot_date}</div>
              <div className="text-[11px] text-dark-600 mt-0.5">
                Uploaded {formatRelativeTime(latestSnapshot.uploaded_at)}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Tab Bar */}
      <div className="flex gap-1 mb-4 border-b border-dark-700/40">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-xs font-medium px-4 py-2 border-b-2 transition-colors ${
              tab === t.id
                ? 'text-primary-400 border-primary-500'
                : 'text-dark-400 border-transparent hover:text-dark-200'
            }`}
          >
            {t.label}
            {t.id === 'trades' && trades.length > 0 && (
              <span className="ml-1.5 text-[10px] text-dark-500">{trades.length}</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {loading ? (
        <div className="text-center py-12 text-dark-500 text-sm animate-pulse">Loading...</div>
      ) : (
        <>
          {tab === 'positions' && (
            positions.length > 0 ? (
              <Card variant="default">
                <PositionsTable positions={positions} cashBalance={latestSnapshot?.cash_balance} />
              </Card>
            ) : (
              <Card variant="default" className="text-center py-12">
                <p className="text-dark-500 text-sm">No positions uploaded yet</p>
                <p className="text-dark-600 text-xs mt-1">Upload a Fidelity Positions CSV to get started</p>
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
