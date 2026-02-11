import { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent } from '../api'

// Terminal color palette
const C = {
  green: '#00ff88',
  red: '#ff4444',
  yellow: '#ffcc00',
  cyan: '#00ccff',
  dim: '#666',
  bright: '#e0e0e0',
  bg: '#0a0a0a',
  panel: '#111',
  border: '#222',
  accent: '#007aff',
}

function useMono(val, decimals = 1) {
  if (val == null) return '-'
  return typeof val === 'number' ? val.toFixed(decimals) : val
}

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
      // M-F, 8:30 AM to 4:00 PM CST
      return day >= 1 && day <= 5 && totalMins >= 510 && totalMins <= 960
    }

    if (check()) {
      const id = setInterval(callback, intervalMs)
      return () => clearInterval(id)
    }
  }, [callback, intervalMs])
}

function RegimeBadge({ regime }) {
  const colors = {
    bullish: C.green,
    neutral: C.yellow,
    bearish: C.red,
    unknown: C.dim,
  }
  return (
    <span style={{
      color: colors[regime] || C.dim,
      fontWeight: 'bold',
      textTransform: 'uppercase',
      fontSize: '0.75rem',
      letterSpacing: '0.1em',
    }}>
      {regime === 'bullish' ? 'BULL' : regime === 'bearish' ? 'BEAR' : regime === 'neutral' ? 'NEUT' : '---'}
    </span>
  )
}

function IndexRow({ label, data }) {
  if (!data || !data.price) return null
  const aboveMa = data.ma50 ? data.price > data.ma50 : null
  return (
    <div className="flex items-center justify-between text-xs" style={{ fontFamily: 'monospace' }}>
      <span style={{ color: C.dim, width: '32px' }}>{label}</span>
      <span style={{ color: C.bright }}>{data.price?.toFixed(2)}</span>
      {data.ma50 && (
        <span style={{ color: aboveMa ? C.green : C.red, fontSize: '0.65rem' }}>
          {aboveMa ? 'MA' : 'MA'}
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
        stroke={isUp ? C.green : C.red}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function Panel({ title, children, className = '' }) {
  return (
    <div
      className={`rounded border p-3 ${className}`}
      style={{
        background: C.panel,
        borderColor: C.border,
      }}
    >
      <div className="text-xs font-bold mb-2 tracking-wider" style={{ color: C.dim, letterSpacing: '0.15em' }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function PnlText({ value, pct, size = 'text-sm' }) {
  if (value == null) return <span style={{ color: C.dim }}>-</span>
  const isUp = value >= 0
  const color = isUp ? C.green : C.red
  const sign = isUp ? '+' : ''
  return (
    <span className={size} style={{ color, fontFamily: 'monospace' }}>
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
      <div className="min-h-screen flex items-center justify-center" style={{ background: C.bg }}>
        <div style={{ color: C.dim, fontFamily: 'monospace' }}>LOADING COMMAND CENTER...</div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: C.bg }}>
        <div style={{ color: C.red, fontFamily: 'monospace' }}>ERROR: {error}</div>
      </div>
    )
  }

  const { market, portfolio, sparkline, positions, candidates, risk, earnings, trades, scanner } = data || {}

  return (
    <div className="min-h-screen p-2 md:p-4" style={{ background: C.bg, fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace" }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold tracking-widest" style={{ color: C.bright }}>
            CANSLIM COMMAND CENTER
          </h1>
          {portfolio?.paper_mode && (
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: '#333', color: C.yellow }}>PAPER</span>
          )}
        </div>
        <div className="text-xs" style={{ color: C.dim }}>
          {lastUpdate ? lastUpdate.toLocaleTimeString('en-US', { timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''}
        </div>
      </div>

      {/* Row 1: Market + Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-2">
        <Panel title="MARKET REGIME" className="md:col-span-2">
          <div className="flex items-center gap-4 mb-2">
            <RegimeBadge regime={market?.regime} />
            <span className="text-xs" style={{ color: C.dim }}>
              Signal: {market?.weighted_signal?.toFixed(2) || '-'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <IndexRow label="SPY" data={market?.spy} />
            <IndexRow label="QQQ" data={market?.qqq} />
            <IndexRow label="DIA" data={market?.dia} />
          </div>
        </Panel>

        <Panel title="PORTFOLIO">
          <div className="text-lg font-bold mb-1" style={{ color: C.bright, fontFamily: 'monospace' }}>
            {formatCurrency(portfolio?.total_value)}
          </div>
          <PnlText value={portfolio?.total_return} pct={portfolio?.total_return_pct} size="text-xs" />
          <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
            <div>
              <span style={{ color: C.dim }}>Cash </span>
              <span style={{ color: C.bright }}>{formatCurrency(portfolio?.cash)}</span>
            </div>
            <div>
              <span style={{ color: C.dim }}>Pos </span>
              <span style={{ color: C.bright }}>{portfolio?.positions_count}/{portfolio?.max_positions}</span>
            </div>
          </div>
        </Panel>
      </div>

      {/* Row 2: Sparkline */}
      {sparkline && sparkline.length > 1 && (
        <div className="mb-2 rounded border p-2" style={{ background: C.panel, borderColor: C.border }}>
          <div className="flex items-center justify-between">
            <span className="text-xs tracking-wider" style={{ color: C.dim }}>30-DAY PERFORMANCE</span>
            <Link to="/ai-portfolio" className="text-xs" style={{ color: C.accent }}>
              Details
            </Link>
          </div>
          <div className="mt-1 flex justify-center">
            <Sparkline data={sparkline} width={typeof window !== 'undefined' ? Math.min(window.innerWidth - 48, 600) : 400} height={60} />
          </div>
        </div>
      )}

      {/* Row 3: Positions + Candidates */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-2">
        <Panel title={`POSITIONS (${positions?.length || 0})`}>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {positions?.length === 0 && (
              <div className="text-xs" style={{ color: C.dim }}>No active positions</div>
            )}
            {positions?.map(p => (
              <Link
                key={p.ticker}
                to={`/stock/${p.ticker}`}
                className="flex items-center justify-between text-xs py-0.5 hover:opacity-80"
              >
                <div className="flex items-center gap-2">
                  <span style={{ color: C.cyan, fontWeight: 'bold', width: '40px' }}>{p.ticker}</span>
                  <span style={{ color: C.dim }}>{p.position_pct?.toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-3">
                  <span style={{ color: C.bright }}>{p.score?.toFixed(0) || '-'}</span>
                  <span style={{
                    color: p.gain_pct >= 0 ? C.green : C.red,
                    width: '55px',
                    textAlign: 'right',
                  }}>
                    {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct?.toFixed(1)}%
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </Panel>

        <Panel title="TOP CANDIDATES">
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {candidates?.length === 0 && (
              <div className="text-xs" style={{ color: C.dim }}>No candidates above threshold</div>
            )}
            {candidates?.map(c => (
              <Link
                key={c.ticker}
                to={`/stock/${c.ticker}`}
                className="flex items-center justify-between text-xs py-0.5 hover:opacity-80"
              >
                <div className="flex items-center gap-2">
                  <span style={{ color: C.cyan, fontWeight: 'bold', width: '40px' }}>{c.ticker}</span>
                  <span style={{ color: C.dim, width: '50px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {c.sector?.slice(0, 6)}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span style={{ color: C.bright }}>{c.score?.toFixed(0)}</span>
                  {c.audit_confidence != null && (
                    <span style={{
                      color: c.audit_confidence >= 70 ? C.green : c.audit_confidence >= 50 ? C.yellow : C.red,
                      fontSize: '0.6rem',
                    }}>
                      A{c.audit_confidence?.toFixed(0)}
                    </span>
                  )}
                  {c.projected_growth != null && c.projected_growth > 0 && (
                    <span style={{ color: C.green, width: '40px', textAlign: 'right' }}>
                      +{c.projected_growth?.toFixed(0)}%
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>
        </Panel>
      </div>

      {/* Row 4: Risk + Earnings + Recent Trades */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-2">
        <Panel title="RISK">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs" style={{ color: C.dim }}>Heat</span>
            <span className="text-sm font-bold" style={{
              color: risk?.heat_status === 'normal' ? C.green : risk?.heat_status === 'warning' ? C.yellow : C.red
            }}>
              {risk?.portfolio_heat?.toFixed(1)}%
            </span>
          </div>
          <div className="space-y-0.5">
            {risk?.top_sectors?.slice(0, 3).map(s => (
              <div key={s.sector} className="flex justify-between text-xs">
                <span style={{ color: C.dim }}>{s.sector?.slice(0, 12)}</span>
                <span style={{ color: C.bright }}>{s.pct}% ({s.count})</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="EARNINGS">
          {earnings?.length === 0 ? (
            <div className="text-xs" style={{ color: C.dim }}>No upcoming earnings</div>
          ) : (
            <div className="space-y-0.5">
              {earnings?.slice(0, 5).map(e => (
                <div key={e.ticker} className="flex justify-between text-xs">
                  <span style={{ color: C.cyan }}>{e.ticker}</span>
                  <span style={{
                    color: e.days <= 7 ? C.red : e.days <= 14 ? C.yellow : C.bright,
                  }}>
                    {e.days}d
                  </span>
                  <span style={{ color: C.dim }}>
                    {e.beat_streak > 0 ? `${e.beat_streak}` : '-'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <Panel title="RECENT TRADES">
          <div className="space-y-0.5 max-h-40 overflow-y-auto">
            {trades?.slice(0, 5).map((t, i) => (
              <div key={i} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1">
                  <span style={{
                    color: t.action === 'BUY' ? C.green : C.red,
                    fontWeight: 'bold',
                    width: '24px',
                  }}>
                    {t.action === 'BUY' ? 'B' : 'S'}
                  </span>
                  <span style={{ color: C.cyan }}>{t.ticker}</span>
                </div>
                <span style={{ color: C.dim }}>
                  {t.executed_at ? new Date(t.executed_at).toLocaleDateString('en-US', {
                    month: 'numeric', day: 'numeric', timeZone: 'America/Chicago'
                  }) : '-'}
                </span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {/* Row 5: Scanner Status */}
      <div className="rounded border px-3 py-2 flex items-center justify-between" style={{ background: C.panel, borderColor: C.border }}>
        <div className="flex items-center gap-3">
          <span className="text-xs tracking-wider" style={{ color: C.dim }}>SCANNER</span>
          <span className="text-xs" style={{
            color: scanner?.is_scanning ? C.green : C.dim,
          }}>
            {scanner?.is_scanning ? (
              <>
                <span className="inline-block w-1.5 h-1.5 rounded-full mr-1" style={{ background: C.green }} />
                {scanner?.phase || 'scanning'}
              </>
            ) : 'IDLE'}
          </span>
        </div>
        {scanner?.is_scanning && scanner?.stocks_scanned != null && (
          <span className="text-xs" style={{ color: C.dim }}>
            {scanner.stocks_scanned}/{scanner.total_stocks}
          </span>
        )}
        {scanner?.last_scan_end && (
          <span className="text-xs" style={{ color: C.dim }}>
            Last: {new Date(scanner.last_scan_end).toLocaleTimeString('en-US', {
              timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit'
            })}
          </span>
        )}
      </div>

      {/* Quick nav links */}
      <div className="mt-3 flex flex-wrap gap-2">
        {[
          { to: '/dashboard', label: 'Dashboard' },
          { to: '/ai-portfolio', label: 'AI Portfolio' },
          { to: '/screener', label: 'Screener' },
          { to: '/backtest', label: 'Backtests' },
          { to: '/analytics', label: 'Analytics' },
        ].map(link => (
          <Link
            key={link.to}
            to={link.to}
            className="text-xs px-3 py-1.5 rounded border hover:opacity-80"
            style={{ borderColor: C.border, color: C.accent }}
          >
            {link.label}
          </Link>
        ))}
      </div>
    </div>
  )
}
