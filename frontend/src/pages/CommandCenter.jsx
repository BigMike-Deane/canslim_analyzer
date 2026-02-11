import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, getScoreClass, formatScore } from '../api'

// Auto-refresh during market hours (M-F 8:30am-4pm CST)
function useMarketRefresh(callback, intervalMs = 60000) {
  useEffect(() => {
    const check = () => {
      const now = new Date()
      const cst = new Date(now.toLocaleString('en-US', { timeZone: 'America/Chicago' }))
      const day = cst.getDay()
      const hours = cst.getHours()
      const mins = cst.getMinutes()
      const totalMins = hours * 60 + mins
      return day >= 1 && day <= 5 && totalMins >= 510 && totalMins <= 960
    }

    if (check()) {
      const id = setInterval(callback, intervalMs)
      return () => clearInterval(id)
    }
  }, [callback, intervalMs])
}

function RegimeBadge({ regime }) {
  const config = {
    bullish: { label: 'BULL', color: 'bg-green-500/20 text-green-400' },
    neutral: { label: 'NEUT', color: 'bg-yellow-500/20 text-yellow-400' },
    bearish: { label: 'BEAR', color: 'bg-red-500/20 text-red-400' },
  }
  const { label, color } = config[regime] || { label: '---', color: 'bg-dark-600 text-dark-400' }
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded ${color}`}>
      {label}
    </span>
  )
}

function IndexRow({ label, data }) {
  if (!data || !data.price) return null
  const aboveMa = data.ma50 ? data.price > data.ma50 : null
  return (
    <div className="flex items-center justify-between text-xs py-1">
      <span className="text-dark-400 w-8 font-medium">{label}</span>
      <span className="font-mono">{data.price?.toFixed(2)}</span>
      {data.ma50 && (
        <span className={`text-[10px] px-1.5 py-0.5 rounded ${aboveMa ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
          {aboveMa ? '> 50MA' : '< 50MA'}
        </span>
      )}
    </div>
  )
}

function Sparkline({ data, width = 200, height = 60 }) {
  if (!data || data.length < 2) return <div style={{ height }} />

  const values = data.map(d => d.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1

  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4)
    return `${x},${y}`
  }).join(' ')

  const startVal = values[0]
  const endVal = values[values.length - 1]
  const isUp = endVal >= startVal

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={points}
        fill="none"
        stroke={isUp ? '#34c759' : '#ff3b30'}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function PnlText({ value, pct }) {
  if (value == null) return <span className="text-dark-400">-</span>
  const isUp = value >= 0
  const sign = isUp ? '+' : ''
  return (
    <span className={`text-xs font-mono ${isUp ? 'text-green-400' : 'text-red-400'}`}>
      {sign}{formatCurrency(value)} ({sign}{pct?.toFixed(1)}%)
    </span>
  )
}

export default function CommandCenter() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const result = await api.getCommandCenter()
      setData(result)
      setLastUpdate(new Date())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])
  useMarketRefresh(fetchData)

  if (loading && !data) {
    return (
      <div className="p-4">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-16 rounded-2xl mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <div className="skeleton h-48 rounded-2xl" />
          <div className="skeleton h-48 rounded-2xl" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="skeleton h-32 rounded-2xl" />
          <div className="skeleton h-32 rounded-2xl" />
          <div className="skeleton h-32 rounded-2xl" />
        </div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="p-4">
        <div className="card text-center py-8 text-red-400">
          Failed to load Command Center: {error}
        </div>
      </div>
    )
  }

  const { market, portfolio, sparkline, positions, candidates, risk, earnings, trades, scanner } = data || {}

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-dark-400 text-sm">CANSLIM</div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold">Command Center</h1>
            {portfolio?.paper_mode && (
              <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded font-medium">PAPER</span>
            )}
          </div>
        </div>
        <div className="text-dark-500 text-xs">
          {lastUpdate ? lastUpdate.toLocaleTimeString('en-US', { timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' CST' : ''}
        </div>
      </div>

      {/* Row 1: Market + Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className="card md:col-span-2">
          <div className="flex justify-between items-center mb-3">
            <div className="font-semibold text-sm">Market Regime</div>
            <div className="flex items-center gap-3">
              <RegimeBadge regime={market?.regime} />
              <span className="text-dark-400 text-xs font-mono">
                {market?.weighted_signal?.toFixed(2) || '-'}
              </span>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <IndexRow label="SPY" data={market?.spy} />
            <IndexRow label="QQQ" data={market?.qqq} />
            <IndexRow label="DIA" data={market?.dia} />
          </div>
        </div>

        <div className="card">
          <div className="font-semibold text-sm mb-2">Portfolio</div>
          <div className="text-2xl font-bold mb-1 font-mono">
            {formatCurrency(portfolio?.total_value)}
          </div>
          <PnlText value={portfolio?.total_return} pct={portfolio?.total_return_pct} />
          <div className="mt-3 pt-3 border-t border-dark-700 grid grid-cols-2 gap-2 text-xs">
            <div>
              <div className="text-dark-400">Cash</div>
              <div className="font-semibold font-mono">{formatCurrency(portfolio?.cash)}</div>
            </div>
            <div>
              <div className="text-dark-400">Positions</div>
              <div className="font-semibold font-mono">{portfolio?.positions_count}/{portfolio?.max_positions}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Row 2: Sparkline */}
      {sparkline && sparkline.length > 1 && (
        <div className="card mb-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-dark-400 text-xs">30-Day Performance</span>
            <Link to="/ai-portfolio" className="text-primary-500 text-xs">Details</Link>
          </div>
          <div className="flex justify-center">
            <Sparkline data={sparkline} width={typeof window !== 'undefined' ? Math.min(window.innerWidth - 64, 600) : 400} height={60} />
          </div>
        </div>
      )}

      {/* Row 3: Positions + Candidates */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
        <div className="card">
          <div className="font-semibold text-sm mb-2">Positions ({positions?.length || 0})</div>
          <div className="max-h-64 overflow-y-auto">
            {positions?.length === 0 && (
              <div className="text-dark-400 text-xs py-4 text-center">No active positions</div>
            )}
            {positions?.map(p => (
              <Link
                key={p.ticker}
                to={`/stock/${p.ticker}`}
                className="flex items-center justify-between py-2 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-primary-400 w-10">{p.ticker}</span>
                  <span className="text-dark-500 text-xs font-mono">{p.position_pct?.toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${getScoreClass(p.score)}`}>
                    {p.score?.toFixed(0) || '-'}
                  </span>
                  <span className={`font-mono text-xs w-14 text-right ${p.gain_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct?.toFixed(1)}%
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="font-semibold text-sm mb-2">Top Candidates</div>
          <div className="max-h-64 overflow-y-auto">
            {candidates?.length === 0 && (
              <div className="text-dark-400 text-xs py-4 text-center">No candidates above threshold</div>
            )}
            {candidates?.map(c => (
              <Link
                key={c.ticker}
                to={`/stock/${c.ticker}`}
                className="flex items-center justify-between py-2 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium text-primary-400 w-10">{c.ticker}</span>
                  <span className="text-dark-500 text-[10px] truncate max-w-[50px]">
                    {c.sector?.slice(0, 6)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${getScoreClass(c.score)}`}>
                    {c.score?.toFixed(0)}
                  </span>
                  {c.audit_confidence != null && (
                    <span className={`text-[10px] px-1 py-0.5 rounded ${
                      c.audit_confidence >= 70 ? 'bg-green-500/20 text-green-400' :
                      c.audit_confidence >= 50 ? 'bg-yellow-500/20 text-yellow-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      A{c.audit_confidence?.toFixed(0)}
                    </span>
                  )}
                  {c.projected_growth != null && c.projected_growth > 0 && (
                    <span className="text-green-400 text-xs font-mono w-10 text-right">
                      +{c.projected_growth?.toFixed(0)}%
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Row 4: Risk + Earnings + Recent Trades */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className="card">
          <div className="font-semibold text-sm mb-2">Risk</div>
          <div className="flex items-center gap-2 mb-3">
            <span className="text-dark-400 text-xs">Heat</span>
            <span className={`text-sm font-bold font-mono ${
              risk?.heat_status === 'normal' ? 'text-green-400' :
              risk?.heat_status === 'warning' ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {risk?.portfolio_heat?.toFixed(1)}%
            </span>
          </div>
          {risk?.top_sectors?.slice(0, 3).map(s => (
            <div key={s.sector} className="flex justify-between text-xs py-1 border-b border-dark-700 last:border-0">
              <span className="text-dark-400 truncate max-w-[100px]">{s.sector}</span>
              <span className="font-mono">{s.pct}% ({s.count})</span>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="font-semibold text-sm mb-2">Earnings</div>
          {earnings?.length === 0 ? (
            <div className="text-dark-400 text-xs py-4 text-center">No upcoming earnings</div>
          ) : (
            <div>
              {earnings?.slice(0, 5).map(e => (
                <div key={e.ticker} className="flex items-center justify-between text-xs py-1.5 border-b border-dark-700 last:border-0">
                  <span className="font-medium text-primary-400">{e.ticker}</span>
                  <span className={`px-1.5 py-0.5 rounded ${
                    e.days <= 7 ? 'bg-red-500/20 text-red-400' :
                    e.days <= 14 ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-dark-600 text-dark-300'
                  }`}>
                    {e.days}d
                  </span>
                  <span className="text-dark-400 font-mono">
                    {e.beat_streak > 0 ? `${e.beat_streak} beats` : '-'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <div className="font-semibold text-sm mb-2">Recent Trades</div>
          <div className="max-h-40 overflow-y-auto">
            {trades?.slice(0, 5).map((t, i) => (
              <div key={i} className="flex items-center justify-between text-xs py-1.5 border-b border-dark-700 last:border-0">
                <div className="flex items-center gap-1.5">
                  <span className={`px-1.5 py-0.5 rounded font-bold ${
                    t.action === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {t.action === 'BUY' ? 'B' : 'S'}
                  </span>
                  <span className="font-medium text-primary-400">{t.ticker}</span>
                </div>
                <span className="text-dark-400">
                  {t.executed_at ? new Date(t.executed_at).toLocaleDateString('en-US', {
                    month: 'numeric', day: 'numeric', timeZone: 'America/Chicago'
                  }) : '-'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 5: Scanner Status */}
      <div className="card p-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-dark-400 text-xs font-medium">Scanner</span>
          <span className={`text-xs ${scanner?.is_scanning ? 'text-green-400' : 'text-dark-500'}`}>
            {scanner?.is_scanning ? (
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                {scanner?.phase || 'scanning'}
              </span>
            ) : 'IDLE'}
          </span>
        </div>
        {scanner?.is_scanning && scanner?.stocks_scanned != null && (
          <span className="text-dark-400 text-xs font-mono">
            {scanner.stocks_scanned}/{scanner.total_stocks}
          </span>
        )}
        {scanner?.last_scan_end && (
          <span className="text-dark-500 text-xs">
            Last: {new Date(scanner.last_scan_end).toLocaleTimeString('en-US', {
              timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit'
            })}
          </span>
        )}
      </div>

      <div className="h-4" />
    </div>
  )
}
