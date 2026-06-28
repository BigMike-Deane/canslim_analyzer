import { useState, useEffect, useRef } from 'react'
import { api, formatRelativeTime } from '../api'
import Card, { CardHeader } from '../components/Card'
import PageHeader from '../components/PageHeader'
import Spinner from '../components/Spinner'
import DataTable from '../components/DataTable'

const AUTO_REFRESH_MS = 30_000

// Health-status pill. Palette aligned to the app's design tokens (emerald /
// amber at /15 + border) — the shared Badge.StatusBadge doesn't know these
// ops statuses (healthy/degraded/scanning/...), so this stays local but must
// not drift to off-brand green/yellow.
function StatusBadge({ status }) {
  const color = status === 'healthy' ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20'
    : status === 'scanning' ? 'bg-primary-500/15 text-primary-400 border-primary-500/20'
    : status === 'degraded' ? 'bg-amber-500/15 text-amber-400 border-amber-500/20'
    : status === 'not configured' || status === 'inactive' || status === 'stopped'
      ? 'bg-dark-600/50 text-dark-400 border-dark-500/30'
    : 'bg-red-500/15 text-red-400 border-red-500/20'
  return <span className={`px-2 py-0.5 rounded border text-xs font-medium ${color}`}>{status}</span>
}

function HealthCard({ title, children, status }) {
  return (
    <Card>
      <CardHeader
        title={<span className="uppercase tracking-wide text-dark-200">{title}</span>}
        action={status && <StatusBadge status={status} />}
      />
      {children}
    </Card>
  )
}

function Stat({ label, value, sub }) {
  return (
    <div className="py-1">
      <div className="text-xs text-dark-400">{label}</div>
      <div className="text-sm text-dark-100 font-medium">{value || '-'}</div>
      {sub && <div className="text-xs text-dark-500">{sub}</div>}
    </div>
  )
}

function TimeAgo({ iso }) {
  if (!iso) return <span className="text-dark-500">never</span>
  const d = new Date(iso)
  const mins = Math.round((Date.now() - d.getTime()) / 60000)
  if (mins < 1) return <span className="text-emerald-400">just now</span>
  if (mins < 60) return <span className="text-dark-200">{mins}m ago</span>
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return <span className="text-dark-300">{hrs}h ago</span>
  return <span className="text-dark-400">{Math.round(hrs / 24)}d ago</span>
}

export default function SystemHealth() {
  const [data, setData] = useState(null)
  const [health, setHealth] = useState(null)  // /health: running build/version stamp
  const [mlStatus, setMlStatus] = useState(null)  // ML model telemetry (moved off Command Center)
  const [backups, setBackups] = useState([])
  const [loading, setLoading] = useState(true)
  const [backingUp, setBackingUp] = useState(false)
  const [error, setError] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastRefreshed, setLastRefreshed] = useState(null)
  // Tick once a second so the "Updated X ago" chip stays current without
  // refetching — re-renders are cheap, network calls aren't.
  const [, setTick] = useState(0)
  const intervalRef = useRef(null)

  useEffect(() => { load() }, [])

  // Auto-refresh data on AUTO_REFRESH_MS cadence while enabled.
  useEffect(() => {
    if (!autoRefresh) return
    intervalRef.current = setInterval(() => { load() }, AUTO_REFRESH_MS)
    return () => clearInterval(intervalRef.current)
  }, [autoRefresh])

  // Tick the relative-time clock once a second (separate from the data refetch).
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 1000)
    return () => clearInterval(t)
  }, [])

  async function load() {
    try {
      const [sysHealth, buildHealth, ml, backupList] = await Promise.all([
        api.getSystemHealth(),
        api.getHealth().catch(() => null),
        api.getMLStatus().catch(() => null),
        api.getBackups().catch(() => ({ backups: [] })),
      ])
      setData(sysHealth)
      setHealth(buildHealth)
      setMlStatus(ml)
      setBackups(backupList.backups || [])
      setLastRefreshed(new Date().toISOString())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleBackup() {
    setBackingUp(true)
    try {
      await api.triggerBackup()
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setBackingUp(false)
    }
  }

  if (loading) return (
    <div className="p-6 flex items-center justify-center min-h-[60vh]">
      <Spinner size="md" />
    </div>
  )

  if (error && !data) return (
    <div className="p-6"><div className="bg-red-500/10 text-red-400 p-4 rounded-lg">{error}</div></div>
  )

  const { database, redis, scanner, scheduler, fmp_api, backups: backupStatus, ai_portfolio } = data

  return (
    <div className="p-4 md:p-6 space-y-6">
      <PageHeader
        title="System Health"
        subtitle={lastRefreshed
          ? `Updated ${formatRelativeTime(lastRefreshed)}${autoRefresh ? ' · auto every 30s' : ''}`
          : undefined}
        actions={
          <>
            <button
              onClick={() => setAutoRefresh(v => !v)}
              className={`text-xs px-3 py-1 rounded border transition-colors ${
                autoRefresh
                  ? 'bg-primary-500/10 text-primary-400 border-primary-500/30'
                  : 'text-dark-400 border-dark-700 hover:border-dark-600'
              }`}
              title={autoRefresh ? 'Pause auto-refresh' : 'Resume auto-refresh'}
            >
              {autoRefresh ? 'Auto' : 'Paused'}
            </button>
            <button
              onClick={load}
              className="text-xs text-dark-400 hover:text-dark-200 px-3 py-1 rounded border border-dark-700 hover:border-dark-600"
            >
              Refresh
            </button>
          </>
        }
      />

      {/* Running build — confirms which deploy is live (see backend/build_info.py) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <HealthCard title="App Build" status={health?.status}>
          <div className="flex flex-wrap gap-x-10 gap-y-1">
            <Stat label="Build" value={<span className="font-data">{health?.build || 'unknown'}</span>} sub="deploy stamp" />
            <Stat label="Version" value={health?.version} />
          </div>
        </HealthCard>

        {/* ML model telemetry — moved here from the Command Center (ops, not at-a-glance). */}
        <HealthCard
          title="ML Model"
          status={mlStatus?.config?.enabled ? (mlStatus?.active_model ? 'healthy' : 'not configured') : 'not configured'}
        >
          {mlStatus?.active_model ? (
            <div className="flex flex-wrap gap-x-8 gap-y-1">
              <Stat label="Active" value={<span className="font-data">v{mlStatus.active_model.version} · {mlStatus.active_model.model_type}</span>}
                    sub={mlStatus?.config?.enabled ? `veto ${mlStatus.config.min_confidence}` : 'log-only'} />
              <Stat
                label={mlStatus.active_model.model_type === 'regression' ? 'Spearman' : 'ROC AUC'}
                value={<span className="font-data">{mlStatus.active_model.model_type === 'regression'
                  ? (mlStatus.active_model.spearman || 0).toFixed(3)
                  : (mlStatus.active_model.roc_auc || 0).toFixed(3)}</span>}
                sub={`${mlStatus.active_model.training_samples} trades`}
              />
            </div>
          ) : (
            <Stat label="Active model" value="none" />
          )}
        </HealthCard>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <HealthCard title="Database" status={database?.status}>
          <Stat label="Size" value={database?.size} />
          <Stat label="Stocks" value={`${database?.scored_count?.toLocaleString()} / ${database?.stock_count?.toLocaleString()}`} sub="scored / total" />
        </HealthCard>

        <HealthCard title="Redis" status={redis?.status}>
          <Stat label="Memory" value={redis?.used_memory_human} />
          <Stat label="Clients" value={redis?.connected_clients} />
        </HealthCard>

        <HealthCard title="Scanner" status={scanner?.is_scanning ? 'scanning' : scanner?.enabled ? 'healthy' : 'stopped'}>
          <Stat label="Phase" value={scanner?.phase_label || scanner?.current_phase || scanner?.phase || 'idle'} />
          <Stat label="Stocks" value={scanner?.total_stocks ? `${scanner.stocks_scanned}/${scanner.total_stocks}` : '-'} sub="phase 1" />
          <Stat
            label="Phase Progress"
            value={scanner?.is_scanning && scanner?.phase_total ? `${scanner.phase_current || 0}/${scanner.phase_total}` : '-'}
          />
          <Stat label="Last Scan" value={<TimeAgo iso={scanner?.last_scan_end} />} />
        </HealthCard>

        <HealthCard title="AI Portfolio" status={ai_portfolio?.active ? 'healthy' : 'inactive'}>
          <Stat label="Strategy" value={ai_portfolio?.strategy} />
          <Stat label="Positions" value={ai_portfolio?.positions} />
        </HealthCard>
      </div>

      {/* Scheduler Health */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <HealthCard title="Scheduler Tasks">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Last Successful Scan</span>
              <TimeAgo iso={scheduler?.last_successful_scan} />
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Last Trade Cycle</span>
              <TimeAgo iso={scheduler?.last_successful_trade_cycle} />
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Scan Failures</span>
              <span className={scheduler?.consecutive_scan_failures > 0 ? 'text-red-400 font-medium' : 'text-emerald-400'}>
                {scheduler?.consecutive_scan_failures || 0}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Trade Failures</span>
              <span className={scheduler?.consecutive_trade_failures > 0 ? 'text-red-400 font-medium' : 'text-emerald-400'}>
                {scheduler?.consecutive_trade_failures || 0}
              </span>
            </div>
          </div>

          {scheduler?.last_scan_error && (
            <div className="mt-3 p-2 bg-red-500/10 rounded text-xs text-red-400">
              <div className="font-medium">Last Scan Error</div>
              <div className="mt-1 text-red-400/70">{scheduler.last_scan_error.error}</div>
              <div className="mt-0.5 text-dark-500"><TimeAgo iso={scheduler.last_scan_error.timestamp} /></div>
            </div>
          )}

          {scheduler?.last_trade_cycle_error && (
            <div className="mt-3 p-2 bg-red-500/10 rounded text-xs text-red-400">
              <div className="font-medium">Last Trade Error</div>
              <div className="mt-1 text-red-400/70">{scheduler.last_trade_cycle_error.error}</div>
              <div className="mt-0.5 text-dark-500"><TimeAgo iso={scheduler.last_trade_cycle_error.timestamp} /></div>
            </div>
          )}
        </HealthCard>

        <HealthCard title="FMP API">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Calls This Minute</span>
              <span className="text-dark-100">{fmp_api?.calls_this_minute || 0} / {fmp_api?.limit_per_minute || 300}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Circuit Breaker</span>
              <span className={fmp_api?.circuit_open ? 'text-red-400 font-medium' : 'text-emerald-400'}>
                {fmp_api?.circuit_open ? 'OPEN' : 'Closed'}
              </span>
            </div>
          </div>
          {/* Usage bar */}
          <div className="mt-3">
            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${(fmp_api?.calls_this_minute || 0) / (fmp_api?.limit_per_minute || 300) > 0.8 ? 'bg-red-500' : 'bg-primary-500'}`}
                style={{ width: `${Math.min(100, ((fmp_api?.calls_this_minute || 0) / (fmp_api?.limit_per_minute || 300)) * 100)}%` }}
              />
            </div>
          </div>
        </HealthCard>
      </div>

      {/* Backups */}
      <HealthCard title="Database Backups">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm text-dark-300">
            {backupStatus?.total_backups || 0} backups ({backupStatus?.daily_count || 0} daily, {backupStatus?.weekly_count || 0} weekly)
          </div>
          <button
            onClick={handleBackup}
            disabled={backingUp}
            className="px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 text-white text-xs rounded font-medium"
          >
            {backingUp ? 'Backing up...' : 'Backup Now'}
          </button>
        </div>

        <DataTable
          compact
          sortable={false}
          keyField="filename"
          data={backups}
          emptyMessage='No backups yet. Click "Backup Now" to create one.'
          columns={[
            { key: 'filename', label: 'Filename', className: 'text-dark-200 font-mono text-xs' },
            { key: 'size_mb', label: 'Size', align: 'right', mobileHide: true, render: v => <span className="text-dark-300">{v} MB</span> },
            { key: 'created', label: 'Created', align: 'right', render: v => <TimeAgo iso={v} /> },
            { key: 'is_weekly', label: 'Type', align: 'right', render: v => (
              <span className={`text-xs px-1.5 py-0.5 rounded ${v ? 'bg-purple-500/20 text-purple-400' : 'bg-dark-700 text-dark-400'}`}>
                {v ? 'weekly' : 'daily'}
              </span>
            ) },
          ]}
        />
      </HealthCard>

      {/* Recent Errors */}
      {scheduler?.errors_today?.length > 0 && (
        <HealthCard title="Recent Errors">
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {[...scheduler.errors_today].reverse().map((e, i) => (
              <div key={i} className="flex items-start gap-2 text-xs p-2 bg-dark-900 rounded">
                <span className="text-red-400 font-medium whitespace-nowrap">{e.task}</span>
                <span className="text-dark-400 flex-1 break-all">{e.error}</span>
                <span className="text-dark-500 whitespace-nowrap"><TimeAgo iso={e.timestamp} /></span>
              </div>
            ))}
          </div>
        </HealthCard>
      )}
    </div>
  )
}
