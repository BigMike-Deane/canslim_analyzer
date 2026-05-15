import { useState, useEffect } from 'react'
import { api, formatDate, formatDateTime, formatRelativeTime } from '../api'

// "4m 22s" / "1h 04m" / "23s" — duration between two ISO timestamps.
function _formatScanDuration(startIso, endIso) {
  if (!startIso || !endIso) return null
  const ms = new Date(endIso) - new Date(startIso)
  if (!isFinite(ms) || ms < 0) return null
  const totalSec = Math.round(ms / 1000)
  const hrs = Math.floor(totalSec / 3600)
  const mins = Math.floor((totalSec % 3600) / 60)
  const secs = totalSec % 60
  if (hrs > 0) return `${hrs}h ${mins.toString().padStart(2, '0')}m`
  if (mins > 0) return `${mins}m ${secs.toString().padStart(2, '0')}s`
  return `${secs}s`
}
import { useAuth } from '../auth'

export default function Admin() {
  const { user } = useAuth()
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  // Create form state
  const [newEmail, setNewEmail] = useState('')
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  // ML state
  const [mlStatus, setMlStatus] = useState(null)
  const [mlTraining, setMlTraining] = useState(false)
  const [mlValidation, setMlValidation] = useState(null)
  const [mlFeatures, setMlFeatures] = useState(null)
  const [mlMode, setMlMode] = useState('regression')

  // Scanner control state
  const [scanner, setScanner] = useState(null)
  const [scanSource, setScanSource] = useState('all')
  const [scanInterval, setScanInterval] = useState(35)
  const [scannerBusy, setScannerBusy] = useState(false)
  const [scannerMessage, setScannerMessage] = useState(null)

  useEffect(() => {
    loadUsers()
    loadMLStatus()
    loadScanner()
  }, [])

  // Poll scanner status; fast when actively scanning, slow otherwise.
  useEffect(() => {
    const pollMs = scanner?.is_scanning ? 3000 : 15000
    const t = setInterval(loadScanner, pollMs)
    return () => clearInterval(t)
  }, [scanner?.is_scanning])

  async function loadScanner() {
    try {
      const s = await api.getScannerStatus()
      setScanner(s)
      // Only sync form controls from server when user hasn't been editing.
      if (!scannerBusy) {
        if (s.source) setScanSource(s.source)
        if (s.interval_minutes) setScanInterval(s.interval_minutes)
      }
    } catch {}
  }

  async function handleScannerSave() {
    setScannerBusy(true)
    setScannerMessage(null)
    try {
      // PATCH updates config + reschedules without forcing an immediate scan.
      await api.updateScannerConfig(scanSource, scanInterval)
      setScannerMessage({ kind: 'success', text: `Scanner: ${scanSource} every ${scanInterval} min` })
      await loadScanner()
    } catch (err) {
      setScannerMessage({ kind: 'error', text: err.message || 'Update failed' })
    } finally {
      setScannerBusy(false)
    }
  }

  async function handleScannerRunNow() {
    setScannerBusy(true)
    setScannerMessage(null)
    try {
      await api.startScanner(scanSource, scanInterval)
      setScannerMessage({ kind: 'success', text: 'Scan started' })
      await loadScanner()
    } catch (err) {
      setScannerMessage({ kind: 'error', text: err.message || 'Start failed' })
    } finally {
      setScannerBusy(false)
    }
  }

  async function handleScannerStop() {
    if (!confirm('Stop the continuous scanner? Scheduled scans will halt until restarted.')) return
    setScannerBusy(true)
    setScannerMessage(null)
    try {
      await api.stopScanner()
      setScannerMessage({ kind: 'success', text: 'Scanner stopped' })
      await loadScanner()
    } catch (err) {
      setScannerMessage({ kind: 'error', text: err.message || 'Stop failed' })
    } finally {
      setScannerBusy(false)
    }
  }

  async function loadMLStatus() {
    try {
      const [status, validation, features] = await Promise.all([
        api.getMLStatus(),
        api.getMLValidation().catch(() => null),
        // 404 when no active model — fall through to null and render nothing.
        api.getMLFeatures().catch(() => null),
      ])
      setMlStatus(status)
      setMlValidation(validation)
      setMlFeatures(features)
    } catch {}
  }

  async function handleRetrain() {
    setMlTraining(true)
    setError('')
    try {
      const result = await api.triggerMLTraining('nostate_optimized', '', mlMode)
      setError('')
      // Poll for completion
      const poll = setInterval(async () => {
        const s = await api.getMLStatus()
        setMlStatus(s)
        if (s.latest_training?.status !== 'training') {
          clearInterval(poll)
          setMlTraining(false)
          loadMLStatus()
        }
      }, 3000)
    } catch (err) {
      setError(err.message)
      setMlTraining(false)
    }
  }

  async function loadUsers() {
    try {
      setLoading(true)
      const data = await api.getUsers()
      setUsers(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleCreate(e) {
    e.preventDefault()
    setCreating(true)
    setError('')
    try {
      await api.createUser({
        email: newEmail,
        display_name: newName || null,
      })
      setShowCreate(false)
      setNewEmail('')
      setNewName('')
      loadUsers()
    } catch (err) {
      setError(err.message)
    } finally {
      setCreating(false)
    }
  }

  async function toggleActive(u) {
    try {
      await api.updateUser(u.id, { is_active: !u.is_active })
      loadUsers()
    } catch (err) {
      setError(err.message)
    }
  }

  async function toggleAdmin(u) {
    try {
      await api.updateUser(u.id, { is_admin: !u.is_admin })
      loadUsers()
    } catch (err) {
      setError(err.message)
    }
  }

  if (!user?.is_admin) {
    return (
      <div className="p-6">
        <div className="card p-8 text-center">
          <p className="text-dark-400 text-sm">Admin access required</p>
        </div>
      </div>
    )
  }

  // Compact at-a-glance ML metrics so ops doesn't have to scroll past the
  // Scanner + Users sections to know whether the active model is healthy.
  // Full details (per-regime AUC, CV folds, retrain controls) live below in
  // the ML Signal Layer section.
  const mlRibbon = (() => {
    const m = mlStatus?.active_model
    if (!m) return null
    const isRegression = m.model_type === 'regression'
    const primary = isRegression
      ? { label: 'Spearman', value: m.spearman, fmt: v => v?.toFixed(4) ?? '-',
          ok: v => (v ?? 0) >= 0.30, warn: v => (v ?? 0) >= 0.15 }
      : { label: 'ROC AUC', value: m.roc_auc, fmt: v => v?.toFixed(4) ?? '-',
          ok: v => (v ?? 0) >= 0.60, warn: v => (v ?? 0) >= 0.55 }
    const color = primary.ok(primary.value)
      ? 'text-emerald-400' : primary.warn(primary.value)
      ? 'text-amber-400' : 'text-red-400'
    // Eval-backtest fields — the graduation gate. Null on legacy rows.
    const hasEval = m.eval_return_pct != null || m.eval_sharpe != null
    const evalReturnColor = (m.eval_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
    const evalSharpeColor = (m.eval_sharpe ?? 0) >= 1.0
      ? 'text-emerald-400' : (m.eval_sharpe ?? 0) >= 0.5
      ? 'text-amber-400' : 'text-red-400'
    return (
      <a
        href="#ml-signal-layer"
        className="card flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 hover:border-primary-500/30 transition-colors"
      >
        <span className="text-[10px] uppercase tracking-wider text-dark-500">ML Active</span>
        <span className="text-sm font-data text-dark-100">v{m.version}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded-md ${isRegression ? 'bg-accent-500/15 text-accent-400' : 'bg-primary-500/15 text-primary-400'}`}>
          {isRegression ? 'regression' : 'classifier'}
        </span>
        <span className="text-[10px] text-dark-500" title={isRegression ? 'CV Spearman' : 'CV ROC AUC'}>
          {primary.label}
        </span>
        <span className={`text-sm font-data ${color}`}>{primary.fmt(primary.value)}</span>
        {hasEval && (
          <>
            <span className="text-[10px] text-dark-600">·</span>
            <span className="text-[10px] text-dark-500" title="Eval-backtest total return — the graduation gate">eval ret</span>
            <span className={`text-sm font-data ${evalReturnColor}`}>
              {m.eval_return_pct != null ? `${m.eval_return_pct >= 0 ? '+' : ''}${m.eval_return_pct.toFixed(1)}%` : '-'}
            </span>
            <span className="text-[10px] text-dark-500" title="Eval-backtest Sharpe ratio">Sh</span>
            <span className={`text-sm font-data ${evalSharpeColor}`}>
              {m.eval_sharpe != null ? m.eval_sharpe.toFixed(2) : '-'}
            </span>
            {m.eval_max_drawdown_pct != null && (
              <span className="text-[10px] text-dark-500 font-data" title="Eval-backtest max drawdown">
                DD {m.eval_max_drawdown_pct.toFixed(1)}%
              </span>
            )}
            {m.eval_decile_wr != null && (
              <span className="text-[10px] text-dark-500 font-data" title="Top-decile win rate vs baseline">
                D10 {(m.eval_decile_wr * 100).toFixed(1)}%
              </span>
            )}
          </>
        )}
        {m.training_samples != null && (
          <span className="text-[10px] text-dark-500">
            · {m.training_samples} samples
          </span>
        )}
        {m.activated_at && (
          <span className="text-[10px] text-dark-500 ml-auto">
            Activated {formatRelativeTime(m.activated_at)}
          </span>
        )}
      </a>
    )
  })()

  return (
    <div className="p-4 md:p-6 space-y-4">
      {mlRibbon}
      {/* Scanner Control */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-dark-100">Scanner</h2>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
            scanner?.is_scanning
              ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
              : scanner?.enabled
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                : 'bg-dark-700 text-dark-400 border border-dark-600'
          }`}>
            {scanner?.is_scanning ? 'SCANNING' : scanner?.enabled ? 'IDLE · ARMED' : 'STOPPED'}
          </span>
        </div>

        <div className="card space-y-4">
          <p className="text-[11px] text-dark-500 leading-relaxed">
            Global scan cadence — affects every user. Changes take effect immediately and persist until the next restart.
            <br />
            <span className="text-dark-400">Per-user notification preferences live on the </span>
            <a href="/settings" className="text-primary-400 hover:underline">Settings</a>
            <span className="text-dark-400"> page.</span>
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="text-[10px] text-dark-500 uppercase tracking-wider">Source</label>
              <select
                value={scanSource}
                onChange={e => setScanSource(e.target.value)}
                disabled={scannerBusy}
                className="mt-1 w-full px-2 py-1.5 bg-dark-800 border border-dark-600 rounded-lg text-sm text-dark-100 focus:outline-none focus:border-primary-500/50"
              >
                <option value="sp500">S&amp;P 500</option>
                <option value="top50">Top 50</option>
                <option value="russell">Russell</option>
                <option value="all">All Stocks (~2000)</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-dark-500 uppercase tracking-wider">
                Interval (min) <span className="text-dark-600">· 5–120</span>
              </label>
              <div className="mt-1 flex items-center gap-2">
                <input
                  type="range"
                  min="5"
                  max="120"
                  step="5"
                  value={scanInterval}
                  onChange={e => setScanInterval(parseInt(e.target.value))}
                  disabled={scannerBusy}
                  className="flex-1 accent-primary-500"
                />
                <input
                  type="number"
                  min="5"
                  max="120"
                  value={scanInterval}
                  onChange={e => {
                    const v = parseInt(e.target.value) || 5
                    setScanInterval(Math.max(5, Math.min(120, v)))
                  }}
                  disabled={scannerBusy}
                  className="w-16 px-2 py-1 bg-dark-800 border border-dark-600 rounded text-sm text-dark-100 font-data text-right focus:outline-none focus:border-primary-500/50"
                />
              </div>
            </div>
            <div>
              <label className="text-[10px] text-dark-500 uppercase tracking-wider">Next run</label>
              <div className="mt-1 px-2 py-1.5 bg-dark-800/50 border border-dark-700 rounded-lg text-sm text-dark-200 font-data">
                {scanner?.next_run
                  ? new Date(scanner.next_run).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  : '—'}
              </div>
            </div>
          </div>

          {scanner?.is_scanning && (
            <div className="rounded-lg bg-dark-800/40 border border-dark-700/60 px-3 py-2 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-dark-300">
                  {scanner.phase_label || scanner.current_phase || scanner.phase || 'scanning'}
                </span>
                <span className="text-dark-400 font-data">
                  {scanner.phase_total > 0
                    ? `${scanner.phase_current || 0}/${scanner.phase_total}`
                    : scanner.total_stocks > 0
                      ? `${scanner.stocks_scanned || 0}/${scanner.total_stocks}`
                      : '...'}
                </span>
              </div>
              <div className="h-1 bg-dark-700/50 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500/80 transition-all"
                  style={{
                    width: `${
                      scanner.phase_total > 0
                        ? Math.min(100, ((scanner.phase_current || 0) / scanner.phase_total) * 100)
                        : scanner.total_stocks > 0
                          ? Math.min(100, ((scanner.stocks_scanned || 0) / scanner.total_stocks) * 100)
                          : 0
                    }%`,
                  }}
                />
              </div>
              {scanner.phase_detail && (
                <div className="text-[10px] text-dark-500">{scanner.phase_detail}</div>
              )}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleScannerSave}
              disabled={scannerBusy}
              className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:bg-primary-600/40 text-white text-xs font-medium rounded-lg transition-colors"
            >
              Save cadence
            </button>
            <button
              onClick={handleScannerRunNow}
              disabled={scannerBusy || scanner?.is_scanning}
              className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:bg-amber-600/30 disabled:text-amber-300/60 text-white text-xs font-medium rounded-lg transition-colors"
            >
              Run scan now
            </button>
            <button
              onClick={handleScannerStop}
              disabled={scannerBusy || !scanner?.enabled}
              className="px-3 py-1.5 bg-dark-700 hover:bg-red-600/80 disabled:opacity-40 text-dark-100 text-xs font-medium rounded-lg transition-colors"
            >
              Stop scanner
            </button>
            {scanner?.last_scan_end && !scanner?.is_scanning && (
              <span className="ml-auto text-[10px] text-dark-500">
                Last scan: {formatRelativeTime(scanner.last_scan_end)}
                {(() => {
                  const dur = _formatScanDuration(scanner.last_scan_start, scanner.last_scan_end)
                  return dur ? ` · took ${dur}` : ''
                })()}
              </span>
            )}
          </div>

          {scannerMessage && (
            <div className={`text-xs ${scannerMessage.kind === 'success' ? 'text-emerald-400' : 'text-red-400'}`}>
              {scannerMessage.text}
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-dark-100">User Management</h1>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-medium rounded-lg transition-colors"
        >
          {showCreate ? 'Cancel' : 'Invite User'}
        </button>
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
          {error}
          <button onClick={() => setError('')} className="ml-2 text-red-300 hover:text-red-200">&times;</button>
        </div>
      )}

      {/* Invite User Form */}
      {showCreate && (
        <form onSubmit={handleCreate} className="card space-y-3">
          <h3 className="text-sm font-medium text-dark-200">Invite New User</h3>
          <p className="text-xs text-dark-500">They'll sign in with their Google account matching this email.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <input
              type="email"
              value={newEmail}
              onChange={e => setNewEmail(e.target.value)}
              required
              placeholder="user@gmail.com"
              className="px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-sm text-dark-100 placeholder-dark-600 focus:outline-none focus:border-primary-500/50"
            />
            <input
              type="text"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="Display name (optional)"
              className="px-3 py-2 bg-dark-900 border border-dark-700 rounded-lg text-sm text-dark-100 placeholder-dark-600 focus:outline-none focus:border-primary-500/50"
            />
          </div>
          <button
            type="submit"
            disabled={creating}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-green-600/50 text-white text-xs font-medium rounded-lg transition-colors"
          >
            {creating ? 'Creating...' : 'Invite User'}
          </button>
        </form>
      )}

      {/* Users Table */}
      {loading ? (
        <div className="card p-8 text-center">
          <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin mx-auto" />
        </div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-dark-500 border-b border-dark-700/50">
                <th className="text-left py-2 px-3">ID</th>
                <th className="text-left py-2 px-3">Email</th>
                <th className="text-left py-2 px-3">Name</th>
                <th className="text-center py-2 px-3">Active</th>
                <th className="text-center py-2 px-3">Admin</th>
                <th className="text-left py-2 px-3">Created</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.id} className="border-b border-dark-700/30 hover:bg-dark-800/50">
                  <td className="py-2.5 px-3 text-dark-400 font-data">{u.id}</td>
                  <td className="py-2.5 px-3 text-dark-200">{u.email}</td>
                  <td className="py-2.5 px-3 text-dark-300">{u.display_name || '-'}</td>
                  <td className="py-2.5 px-3 text-center">
                    <button
                      onClick={() => toggleActive(u)}
                      disabled={u.id === user.id}
                      className={`w-8 h-4 rounded-full transition-colors relative ${
                        u.is_active ? 'bg-green-600' : 'bg-dark-600'
                      } ${u.id === user.id ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                        u.is_active ? 'left-4' : 'left-0.5'
                      }`} />
                    </button>
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <button
                      onClick={() => toggleAdmin(u)}
                      disabled={u.id === user.id}
                      className={`w-8 h-4 rounded-full transition-colors relative ${
                        u.is_admin ? 'bg-amber-600' : 'bg-dark-600'
                      } ${u.id === user.id ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                    >
                      <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                        u.is_admin ? 'left-4' : 'left-0.5'
                      }`} />
                    </button>
                  </td>
                  <td className="py-2.5 px-3 text-dark-500 text-xs">
                    {u.created_at ? formatDate(u.created_at) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ML Signal Layer */}
      <div className="mt-8" id="ml-signal-layer">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-dark-100">ML Signal Layer</h2>
          <div className="flex items-center gap-2">
            <select
              value={mlMode}
              onChange={e => setMlMode(e.target.value)}
              disabled={mlTraining}
              className="px-2 py-1.5 bg-dark-800 border border-dark-600 rounded-lg text-xs text-dark-200 focus:outline-none focus:border-primary-500/50"
            >
              <option value="regression">Regression (v2)</option>
              <option value="classifier">Classifier (v1)</option>
              <option value="both">Both (best wins)</option>
            </select>
            <button
              onClick={handleRetrain}
              disabled={mlTraining}
              className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:bg-primary-600/50 text-white text-xs font-medium rounded-lg transition-colors"
            >
              {mlTraining ? 'Training...' : 'Retrain Model'}
            </button>
          </div>
        </div>

        <div className="card space-y-4">
          {/* Current Model */}
          <div>
            <h3 className="text-xs font-semibold text-dark-400 uppercase tracking-wider mb-2">Active Model</h3>
            {mlStatus?.active_model ? (() => {
              const m = mlStatus.active_model
              const isRegression = m.model_type === 'regression'
              return (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <div className="text-[10px] text-dark-500">Version</div>
                    <div className="text-sm font-data text-dark-100">v{m.version}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-dark-500">Type</div>
                    <div className={`text-sm font-data ${isRegression ? 'text-accent-500' : 'text-primary-400'}`}>
                      {isRegression ? 'Regression' : 'Classifier'}
                    </div>
                  </div>
                  {isRegression ? (
                    <>
                      <div>
                        <div className="text-[10px] text-dark-500">Spearman</div>
                        <div className={`text-sm font-data ${(m.spearman || 0) >= 0.3 ? 'text-emerald-400' : (m.spearman || 0) >= 0.15 ? 'text-amber-400' : 'text-red-400'}`}>
                          {m.spearman?.toFixed(4) || '-'}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-dark-500">R²</div>
                        <div className="text-sm font-data text-dark-200">{m.r2_score?.toFixed(4) || '-'}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-dark-500">MAE</div>
                        <div className="text-sm font-data text-dark-200">{m.mae?.toFixed(2) || '-'}%</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-dark-500">Direction Acc</div>
                        <div className="text-sm font-data text-dark-200">{m.direction_accuracy ? (m.direction_accuracy * 100).toFixed(1) + '%' : '-'}</div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div>
                        <div className="text-[10px] text-dark-500">ROC AUC</div>
                        <div className={`text-sm font-data ${(m.roc_auc || 0) >= 0.6 ? 'text-emerald-400' : (m.roc_auc || 0) >= 0.55 ? 'text-amber-400' : 'text-red-400'}`}>
                          {m.roc_auc?.toFixed(4) || '-'}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-dark-500">Accuracy</div>
                        <div className="text-sm font-data text-dark-200">{((m.accuracy || 0) * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-dark-500">F1 Score</div>
                        <div className="text-sm font-data text-dark-200">{((m.f1 || 0) * 100).toFixed(1)}%</div>
                      </div>
                    </>
                  )}
                  <div>
                    <div className="text-[10px] text-dark-500">Training Samples</div>
                    <div className="text-sm font-data text-dark-200">{m.training_samples}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-dark-500">Features</div>
                    <div className="text-sm font-data text-dark-200">{m.feature_count}</div>
                  </div>
                  <div className="md:col-span-2">
                    <div className="text-[10px] text-dark-500">Activated</div>
                    <div className="text-sm font-data text-dark-300">
                      {m.activated_at ? formatDateTime(m.activated_at) : '-'}
                    </div>
                  </div>
                </div>
              )
            })() : (
              <p className="text-xs text-dark-500">No active model. Click "Retrain Model" to train.</p>
            )}
          </div>

          {/* Walk-Forward CV Results */}
          {mlValidation?.cv_results?.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-dark-400 uppercase tracking-wider mb-2">Walk-Forward CV</h3>
              <div className="overflow-x-auto">
                {(mlValidation.model_type || 'classifier') === 'regression' ? (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-dark-500 border-b border-dark-700/50">
                        <th className="text-left py-1.5 px-2">Fold</th>
                        <th className="text-right py-1.5 px-2">Spearman</th>
                        <th className="text-right py-1.5 px-2">R²</th>
                        <th className="text-right py-1.5 px-2">MAE</th>
                        <th className="text-right py-1.5 px-2">Dir Acc</th>
                        <th className="text-right py-1.5 px-2">Train</th>
                        <th className="text-right py-1.5 px-2">Test</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mlValidation.cv_results.map((fold, i) => (
                        <tr key={i} className="border-b border-dark-700/30 hover:bg-dark-800/50">
                          <td className="py-1.5 px-2 font-data text-dark-300">Fold {fold.fold}</td>
                          <td className={`py-1.5 px-2 font-data text-right ${(fold.spearman || 0) >= 0.3 ? 'text-emerald-400' : (fold.spearman || 0) >= 0.15 ? 'text-amber-400' : 'text-red-400'}`}>
                            {fold.spearman?.toFixed(4) || '-'}
                          </td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{fold.r2?.toFixed(4) || '-'}</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{fold.mae?.toFixed(2) || '-'}</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{fold.direction_accuracy ? (fold.direction_accuracy * 100).toFixed(1) + '%' : '-'}</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-400">{fold.train_size}</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-400">{fold.test_size}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-dark-500 border-b border-dark-700/50">
                        <th className="text-left py-1.5 px-2">Fold</th>
                        <th className="text-right py-1.5 px-2">AUC</th>
                        <th className="text-right py-1.5 px-2">Acc</th>
                        <th className="text-right py-1.5 px-2">Prec</th>
                        <th className="text-right py-1.5 px-2">Recall</th>
                        <th className="text-right py-1.5 px-2">F1</th>
                        <th className="text-right py-1.5 px-2">Train</th>
                        <th className="text-right py-1.5 px-2">Test</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mlValidation.cv_results.map((fold, i) => (
                        <tr key={i} className="border-b border-dark-700/30 hover:bg-dark-800/50">
                          <td className="py-1.5 px-2 font-data text-dark-300">Fold {fold.fold}</td>
                          <td className={`py-1.5 px-2 font-data text-right ${(fold.roc_auc || 0) >= 0.6 ? 'text-emerald-400' : (fold.roc_auc || 0) >= 0.55 ? 'text-amber-400' : 'text-red-400'}`}>
                            {fold.roc_auc?.toFixed(4) || '-'}
                          </td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{((fold.accuracy || 0) * 100).toFixed(1)}%</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{((fold.precision || 0) * 100).toFixed(1)}%</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{((fold.recall || 0) * 100).toFixed(1)}%</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-300">{((fold.f1 || 0) * 100).toFixed(1)}%</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-400">{fold.train_size}</td>
                          <td className="py-1.5 px-2 font-data text-right text-dark-400">{fold.test_size}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {/* Per-regime AUC */}
          {mlValidation?.cv_results?.length > 0 && mlValidation.cv_results.some(f => f.per_regime_auc) && (() => {
            const regimes = ['bullish', 'neutral', 'bearish']
            const folds = mlValidation.cv_results
            const colorFor = (auc) => {
              if (auc == null) return 'text-dark-500'
              if (auc >= 0.55) return 'text-emerald-400'
              if (auc >= 0.50) return 'text-amber-400'
              return 'text-red-400'
            }
            const fmt = (auc) => (auc == null ? '—' : auc.toFixed(4))
            const meanFor = (regime) => {
              const vals = folds
                .map(f => f.per_regime_auc?.[regime])
                .filter(v => v != null && !Number.isNaN(v))
              if (vals.length === 0) return null
              return vals.reduce((a, b) => a + b, 0) / vals.length
            }
            return (
              <div>
                <h3 className="text-xs font-semibold text-dark-400 uppercase tracking-wider mb-2">Per-regime AUC</h3>
                <p className="text-[10px] text-dark-500 mb-2">Reveals whether the model adds signal beyond the upstream regime gate. "—" = too few samples in that fold/regime.</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-dark-500 border-b border-dark-700/50">
                        <th className="text-left py-1.5 px-2">Regime</th>
                        {folds.map(f => (
                          <th key={f.fold} className="text-right py-1.5 px-2">Fold {f.fold}</th>
                        ))}
                        <th className="text-right py-1.5 px-2">Mean</th>
                      </tr>
                    </thead>
                    <tbody>
                      {regimes.map(regime => {
                        const mean = meanFor(regime)
                        return (
                          <tr key={regime} className="border-b border-dark-700/30 hover:bg-dark-800/50">
                            <td className="py-1.5 px-2 font-data text-dark-300 capitalize">{regime}</td>
                            {folds.map(f => {
                              const auc = f.per_regime_auc?.[regime]
                              return (
                                <td key={f.fold} className={`py-1.5 px-2 font-data text-right ${colorFor(auc)}`}>
                                  {fmt(auc)}
                                </td>
                              )
                            })}
                            <td className={`py-1.5 px-2 font-data text-right ${colorFor(mean)}`}>{fmt(mean)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })()}

          {/* Feature importance bar chart */}
          {mlFeatures?.feature_importance && Object.keys(mlFeatures.feature_importance).length > 0 && (() => {
            const entries = Object.entries(mlFeatures.feature_importance)
              .map(([name, value]) => ({ name, value: Number(value) || 0 }))
              .sort((a, b) => b.value - a.value)
            const top = entries.slice(0, 15)
            const maxValue = top[0]?.value || 1
            const totalShown = top.reduce((s, e) => s + e.value, 0)
            return (
              <div>
                <h3 className="text-xs font-semibold text-dark-400 uppercase tracking-wider mb-2">Feature Importance</h3>
                <p className="text-[10px] text-dark-500 mb-2">
                  XGBoost gain for the top {top.length} features in v{mlFeatures.version}.
                  These {top.length} account for{' '}
                  <span className="font-data">{(totalShown * 100).toFixed(0)}%</span> of total importance.
                </p>
                <div className="space-y-1">
                  {top.map(({ name, value }) => {
                    const pct = maxValue > 0 ? (value / maxValue) * 100 : 0
                    return (
                      <div key={name} className="flex items-center gap-2 text-[11px]">
                        <span className="font-data text-dark-300 w-32 truncate" title={name}>{name}</span>
                        <div className="flex-1 h-2 bg-dark-700/40 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary-500/70"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-data text-dark-400 w-12 text-right">{(value * 100).toFixed(1)}%</span>
                      </div>
                    )
                  })}
                </div>
                {entries.length > top.length && (
                  <p className="text-[10px] text-dark-600 mt-2">
                    {entries.length - top.length} additional feature{entries.length - top.length === 1 ? '' : 's'} not shown
                    {' '}({((1 - totalShown) * 100).toFixed(0)}% of total importance).
                  </p>
                )}
              </div>
            )
          })()}

          {/* Latest Training Status */}
          {mlStatus?.latest_training && mlStatus.latest_training.status !== 'active' && (
            <div>
              <h3 className="text-xs font-semibold text-dark-400 uppercase tracking-wider mb-1">Latest Training</h3>
              <div className="flex items-center gap-2 text-xs">
                <span className={`px-2 py-0.5 rounded border text-[10px] ${
                  mlStatus.latest_training.status === 'completed' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20' :
                  mlStatus.latest_training.status === 'training' ? 'bg-primary-500/15 text-primary-400 border-primary-500/20' :
                  mlStatus.latest_training.status === 'failed' ? 'bg-red-500/15 text-red-400 border-red-500/20' :
                  'bg-dark-600/50 text-dark-400 border-dark-500/30'
                }`}>
                  {mlStatus.latest_training.status?.toUpperCase()}
                </span>
                {mlStatus.latest_training.error_message && (
                  <span className="text-red-400 truncate max-w-[300px]">{mlStatus.latest_training.error_message}</span>
                )}
              </div>
            </div>
          )}

          {/* Config */}
          <div className="text-[10px] text-dark-600">
            ML signal: {mlStatus?.config?.enabled ? 'ENABLED' : 'disabled'}
            {mlStatus?.config?.log_only && ' (log-only)'}
            {mlStatus?.config?.weight && ` | weight: ${mlStatus.config.weight}`}
          </div>
        </div>
      </div>
    </div>
  )
}
