import { useState, useEffect } from 'react'
import { api } from '../api'

function StatusBadge({ status }) {
  const color = status === 'healthy' ? 'bg-green-500/20 text-green-400'
    : status === 'degraded' ? 'bg-yellow-500/20 text-yellow-400'
    : status === 'not configured' ? 'bg-dark-600 text-dark-400'
    : 'bg-red-500/20 text-red-400'
  return <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>{status}</span>
}

function Card({ title, children, status }) {
  return (
    <div className="bg-dark-800 rounded-lg border border-dark-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-dark-200 uppercase tracking-wide">{title}</h3>
        {status && <StatusBadge status={status} />}
      </div>
      {children}
    </div>
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
  if (mins < 1) return <span className="text-green-400">just now</span>
  if (mins < 60) return <span className="text-dark-200">{mins}m ago</span>
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return <span className="text-dark-300">{hrs}h ago</span>
  return <span className="text-dark-400">{Math.round(hrs / 24)}d ago</span>
}

export default function SystemHealth() {
  const [data, setData] = useState(null)
  const [backups, setBackups] = useState([])
  const [loading, setLoading] = useState(true)
  const [backingUp, setBackingUp] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => { load() }, [])

  async function load() {
    try {
      const [health, backupList] = await Promise.all([
        api.getSystemHealth(),
        api.getBackups().catch(() => ({ backups: [] })),
      ])
      setData(health)
      setBackups(backupList.backups || [])
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
      <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )

  if (error && !data) return (
    <div className="p-6"><div className="bg-red-500/10 text-red-400 p-4 rounded-lg">{error}</div></div>
  )

  const { database, redis, scanner, scheduler, fmp_api, backups: backupStatus, ai_portfolio } = data

  return (
    <div className="p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-dark-100">System Health</h1>
        <button
          onClick={load}
          className="text-xs text-dark-400 hover:text-dark-200 px-3 py-1 rounded border border-dark-700 hover:border-dark-600"
        >
          Refresh
        </button>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card title="Database" status={database?.status}>
          <Stat label="Size" value={database?.size} />
          <Stat label="Stocks" value={`${database?.scored_count?.toLocaleString()} / ${database?.stock_count?.toLocaleString()}`} sub="scored / total" />
        </Card>

        <Card title="Redis" status={redis?.status}>
          <Stat label="Memory" value={redis?.used_memory_human} />
          <Stat label="Clients" value={redis?.connected_clients} />
        </Card>

        <Card title="Scanner" status={scanner?.is_scanning ? 'scanning' : scanner?.enabled ? 'healthy' : 'stopped'}>
          <Stat label="Phase" value={scanner?.phase || 'idle'} />
          <Stat label="Progress" value={scanner?.total_stocks ? `${scanner.stocks_scanned}/${scanner.total_stocks}` : '-'} />
          <Stat label="Last Scan" value={<TimeAgo iso={scanner?.last_scan_end} />} />
        </Card>

        <Card title="AI Portfolio" status={ai_portfolio?.active ? 'healthy' : 'inactive'}>
          <Stat label="Strategy" value={ai_portfolio?.strategy} />
          <Stat label="Positions" value={ai_portfolio?.positions} />
        </Card>
      </div>

      {/* Scheduler Health */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Scheduler Tasks">
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
              <span className={scheduler?.consecutive_scan_failures > 0 ? 'text-red-400 font-medium' : 'text-green-400'}>
                {scheduler?.consecutive_scan_failures || 0}
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Trade Failures</span>
              <span className={scheduler?.consecutive_trade_failures > 0 ? 'text-red-400 font-medium' : 'text-green-400'}>
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
        </Card>

        <Card title="FMP API">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Calls This Minute</span>
              <span className="text-dark-100">{fmp_api?.calls_this_minute || 0} / {fmp_api?.limit_per_minute || 300}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-dark-300">Circuit Breaker</span>
              <span className={fmp_api?.circuit_open ? 'text-red-400 font-medium' : 'text-green-400'}>
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
        </Card>
      </div>

      {/* Backups */}
      <Card title="Database Backups">
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

        {backups.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-dark-400 text-xs border-b border-dark-700">
                  <th className="text-left py-2 font-medium">Filename</th>
                  <th className="text-right py-2 font-medium">Size</th>
                  <th className="text-right py-2 font-medium">Created</th>
                  <th className="text-right py-2 font-medium">Type</th>
                </tr>
              </thead>
              <tbody>
                {backups.map(b => (
                  <tr key={b.filename} className="border-b border-dark-700/50">
                    <td className="py-1.5 text-dark-200 font-mono text-xs">{b.filename}</td>
                    <td className="py-1.5 text-right text-dark-300">{b.size_mb} MB</td>
                    <td className="py-1.5 text-right"><TimeAgo iso={b.created} /></td>
                    <td className="py-1.5 text-right">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${b.is_weekly ? 'bg-purple-500/20 text-purple-400' : 'bg-dark-700 text-dark-400'}`}>
                        {b.is_weekly ? 'weekly' : 'daily'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-4 text-dark-500 text-sm">No backups yet. Click "Backup Now" to create one.</div>
        )}
      </Card>

      {/* Recent Errors */}
      {scheduler?.errors_today?.length > 0 && (
        <Card title="Recent Errors">
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {[...scheduler.errors_today].reverse().map((e, i) => (
              <div key={i} className="flex items-start gap-2 text-xs p-2 bg-dark-900 rounded">
                <span className="text-red-400 font-medium whitespace-nowrap">{e.task}</span>
                <span className="text-dark-400 flex-1 break-all">{e.error}</span>
                <span className="text-dark-500 whitespace-nowrap"><TimeAgo iso={e.timestamp} /></span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
