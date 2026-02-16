import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, getScoreClass } from '../api'
import Card, { SectionLabel } from '../components/Card'
import { ScoreBadge, OutcomeBadge, ActionBadge, TagBadge, PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import Sparkline from '../components/Sparkline'

// Auto-refresh during market hours (M-F 8:30am-4pm CST)
function useMarketRefresh(callback, intervalMs = 60000) {
  useEffect(() => {
    const check = () => {
      const now = new Date()
      const cst = new Date(now.toLocaleString('en-US', { timeZone: 'America/Chicago' }))
      const day = cst.getDay()
      const totalMins = cst.getHours() * 60 + cst.getMinutes()
      return day >= 1 && day <= 5 && totalMins >= 510 && totalMins <= 960
    }

    if (check()) {
      const id = setInterval(callback, intervalMs)
      return () => clearInterval(id)
    }
  }, [callback, intervalMs])
}

const MARKET_STATE_CFG = {
  TRENDING:   { label: 'TRENDING',   color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20', dot: 'bg-emerald-400' },
  PRESSURE:   { label: 'PRESSURE',   color: 'text-amber-400',   bg: 'bg-amber-500/10 border-amber-500/20',   dot: 'bg-amber-400' },
  CORRECTION: { label: 'CORRECTION', color: 'text-red-400',     bg: 'bg-red-500/10 border-red-500/20',       dot: 'bg-red-400' },
  RECOVERY:   { label: 'RECOVERY',   color: 'text-blue-400',    bg: 'bg-blue-500/10 border-blue-500/20',     dot: 'bg-blue-400' },
  CONFIRMED:  { label: 'CONFIRMED',  color: 'text-teal-400',    bg: 'bg-teal-500/10 border-teal-500/20',     dot: 'bg-teal-400' },
}

function MarketStateBadge({ state }) {
  const cfg = MARKET_STATE_CFG[state] || { label: state || '---', color: 'text-dark-400', bg: 'bg-dark-700 border-dark-600', dot: 'bg-dark-400' }
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold tracking-wider px-2.5 py-1 rounded-md border ${cfg.bg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      <span className={cfg.color}>{cfg.label}</span>
    </span>
  )
}

function IndexRow({ label, data }) {
  if (!data || !data.price) return null
  const aboveMa = data.ma50 ? data.price > data.ma50 : null
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-dark-400 text-xs font-medium w-9">{label}</span>
      <span className="font-data text-xs text-dark-100">{data.price?.toFixed(2)}</span>
      {data.ma50 && (
        <span className={`text-[10px] font-data ${aboveMa ? 'text-emerald-400' : 'text-red-400'}`}>
          {aboveMa ? '>' : '<'} 50MA
        </span>
      )}
    </div>
  )
}

function CollapsibleSection({ title, badge, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center justify-between w-full mb-2 group"
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">{title}</span>
          {badge}
        </div>
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round"
          className={`text-dark-500 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>
      {open && children}
    </div>
  )
}

function CoiledSpringSection({ cs }) {
  if (!cs) return null
  const { candidates, stats, recent_results } = cs
  const hasData = (candidates?.length > 0) || (stats?.total > 0)
  if (!hasData) return null

  return (
    <Card variant="accent" accent="purple" className="mb-3 bg-purple-500/[0.03]">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-purple-300">Coiled Spring</span>
          <TagBadge color="purple">CATALYST</TagBadge>
        </div>
        <Link to="/coiled-spring/history" className="text-purple-400 text-[10px] hover:text-purple-300 transition-colors">
          History &rarr;
        </Link>
      </div>

      {stats?.total > 0 && (
        <StatGrid
          columns={4}
          className="mb-3"
          stats={[
            { label: 'Win Rate', value: `${stats.win_rate}%`, color: 'text-purple-300' },
            { label: 'Wins', value: stats.wins, color: 'text-emerald-400' },
            { label: 'Losses', value: stats.losses, color: 'text-red-400' },
            { label: 'Flat', value: stats.flat, color: 'text-yellow-400' },
          ]}
        />
      )}

      {candidates?.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-dark-400 uppercase tracking-wider mb-1.5">Upcoming</div>
          {candidates.map(c => (
            <Link
              key={c.ticker}
              to={`/stock/${c.ticker}`}
              className="flex items-center justify-between py-1.5 border-b border-dark-700/30 last:border-0 hover:bg-purple-500/5 -mx-1 px-1 rounded transition-colors"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium text-purple-300 text-xs w-10">{c.ticker}</span>
                {c.base_type && <TagBadge>{c.base_type}</TagBadge>}
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-[10px] font-data ${c.days_to_earnings <= 7 ? 'text-red-400' : 'text-amber-400'}`}>
                  {c.days_to_earnings}d
                </span>
                <span className="text-[10px] text-dark-500 font-data">{c.beat_streak}x</span>
                <ScoreBadge score={c.score} size="xs" />
              </div>
            </Link>
          ))}
        </div>
      )}

      {recent_results?.length > 0 && (
        <div>
          <div className="text-[10px] text-dark-400 uppercase tracking-wider mb-1.5">Results</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1.5">
            {recent_results.map((r, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-dark-200">{r.ticker}</span>
                <OutcomeBadge outcome={r.outcome} />
                <PnlText value={r.price_change_pct} className="text-[10px]" />
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

export default function CommandCenter() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [runningAction, setRunningAction] = useState(null)

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

  const handleAction = async (action) => {
    setRunningAction(action)
    try {
      if (action === 'cycle') await api.runAITradingCycle()
      if (action === 'scan') await api.startScanner('all', 90)
      fetchData()
    } catch (e) {
      console.error(e)
    } finally {
      setRunningAction(null)
    }
  }

  if (loading && !data) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-48 mb-5 rounded-lg" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-3">
            <div className="skeleton h-40 rounded-xl" />
            <div className="skeleton h-28 rounded-xl" />
          </div>
          <div className="space-y-3">
            <div className="skeleton h-64 rounded-xl" />
            <div className="skeleton h-64 rounded-xl" />
          </div>
          <div className="space-y-3">
            <div className="skeleton h-48 rounded-xl" />
            <div className="skeleton h-36 rounded-xl" />
            <div className="skeleton h-36 rounded-xl" />
          </div>
        </div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="p-4 md:p-6">
        <Card className="text-center py-12">
          <div className="text-red-400 text-sm mb-2">Failed to load Command Center</div>
          <div className="text-dark-500 text-xs">{error}</div>
          <button onClick={fetchData} className="btn-primary mt-4 text-xs">Retry</button>
        </Card>
      </div>
    )
  }

  const { market, portfolio, sparkline, positions, candidates, risk, earnings, trades, scanner, coiled_spring } = data || {}
  const marketState = market?.market_state?.current_state || market?.regime?.toUpperCase()

  return (
    <div className="p-4 md:p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-dark-50">Command Center</h1>
          {portfolio?.paper_mode && (
            <TagBadge color="amber">PAPER</TagBadge>
          )}
          <MarketStateBadge state={marketState} />
        </div>
        <div className="flex items-center gap-3">
          {/* Quick Actions */}
          <div className="hidden md:flex items-center gap-2">
            <button
              onClick={() => handleAction('cycle')}
              disabled={!!runningAction}
              className="text-[10px] font-medium px-3 py-1.5 rounded-lg bg-primary-600/15 text-primary-400 border border-primary-500/20 hover:bg-primary-600/25 transition-colors disabled:opacity-50"
            >
              {runningAction === 'cycle' ? 'Running...' : 'Run Cycle'}
            </button>
            <button
              onClick={() => handleAction('scan')}
              disabled={!!runningAction || scanner?.is_scanning}
              className="text-[10px] font-medium px-3 py-1.5 rounded-lg bg-dark-700 text-dark-300 border border-dark-600 hover:bg-dark-600 transition-colors disabled:opacity-50"
            >
              {runningAction === 'scan' ? 'Starting...' : 'Start Scan'}
            </button>
          </div>
          <span className="text-[10px] text-dark-500 font-data">
            {lastUpdate ? lastUpdate.toLocaleTimeString('en-US', {
              timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit'
            }) : ''}
          </span>
        </div>
      </div>

      {/* 3-Column Desktop / Stacked Mobile */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
        {/* ═══ LEFT COLUMN ═══ */}
        <div className="md:col-span-3 space-y-3">
          {/* Market Regime */}
          <Card variant="glass" animate stagger={1}>
            <SectionLabel>Market</SectionLabel>
            <div className="space-y-0.5">
              <IndexRow label="SPY" data={market?.spy} />
              <IndexRow label="QQQ" data={market?.qqq} />
              <IndexRow label="DIA" data={market?.dia} />
            </div>
            {market?.weighted_signal != null && (
              <div className="mt-2 pt-2 border-t border-dark-700/30 flex items-center justify-between">
                <span className="text-[10px] text-dark-500">Signal</span>
                <span className={`text-xs font-data font-medium ${
                  market.weighted_signal > 0.5 ? 'text-emerald-400' :
                  market.weighted_signal < -0.5 ? 'text-red-400' : 'text-dark-300'
                }`}>
                  {market.weighted_signal?.toFixed(2)}
                </span>
              </div>
            )}
          </Card>

          {/* Portfolio Summary */}
          <Card variant="glass" animate stagger={2}>
            <SectionLabel>Portfolio</SectionLabel>
            <div className={`text-2xl font-bold font-data mb-1 ${
              portfolio?.total_return >= 0 ? 'text-emerald-400 glow-green' : 'text-red-400 glow-red'
            }`}>
              {formatCurrency(portfolio?.total_value)}
            </div>
            <div className="flex items-center gap-2 mb-3">
              <PnlText
                value={portfolio?.total_return_pct}
                className="text-xs"
                prefix={portfolio?.total_return_pct >= 0 ? '+' : ''}
              />
              <span className="text-dark-600">|</span>
              <span className="text-xs font-data text-dark-400">
                {formatCurrency(portfolio?.total_return)}
              </span>
            </div>

            {/* Sparkline */}
            {sparkline && sparkline.length > 1 && (
              <div className="mb-3">
                <Sparkline data={sparkline} width={200} height={40} gradient className="w-full" />
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-dark-700/30">
              <div>
                <div className="text-[10px] text-dark-500">Cash</div>
                <div className="text-xs font-data font-medium text-dark-200">{formatCurrency(portfolio?.cash)}</div>
              </div>
              <div>
                <div className="text-[10px] text-dark-500">Positions</div>
                <div className="text-xs font-data font-medium text-dark-200">{portfolio?.positions_count}/{portfolio?.max_positions}</div>
              </div>
            </div>
          </Card>

          {/* Risk */}
          <Card variant="glass" animate stagger={3} className="hidden md:block">
            <SectionLabel>Risk</SectionLabel>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-dark-400">Heat</span>
              <span className={`text-sm font-data font-semibold ${
                risk?.heat_status === 'normal' ? 'text-emerald-400' :
                risk?.heat_status === 'warning' ? 'text-amber-400' : 'text-red-400'
              }`}>
                {risk?.portfolio_heat?.toFixed(1)}%
              </span>
            </div>
            {risk?.top_sectors?.slice(0, 3).map(s => (
              <div key={s.sector} className="flex justify-between text-[11px] py-1 border-b border-dark-700/20 last:border-0">
                <span className="text-dark-400 truncate max-w-[100px]">{s.sector}</span>
                <span className="font-data text-dark-300">{s.pct}%</span>
              </div>
            ))}
          </Card>
        </div>

        {/* ═══ CENTER COLUMN ═══ */}
        <div className="md:col-span-5 space-y-3">
          {/* Positions */}
          <Card variant="glass" animate stagger={2}>
            <CollapsibleSection
              title="Positions"
              badge={<span className="text-[10px] font-data text-dark-500">{positions?.length || 0}</span>}
            >
              <div className="max-h-72 overflow-y-auto -mx-1">
                {positions?.length === 0 && (
                  <div className="text-dark-500 text-xs py-6 text-center">No active positions</div>
                )}
                {positions?.map(p => (
                  <Link
                    key={p.ticker}
                    to={`/stock/${p.ticker}`}
                    className="flex items-center justify-between py-2 px-2 rounded-lg hover:bg-dark-750/50 transition-colors group"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs font-semibold text-primary-400 w-11 group-hover:text-primary-300">{p.ticker}</span>
                      <span className="text-[10px] font-data text-dark-500">{p.position_pct?.toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <ScoreBadge score={p.score} size="xs" />
                      <span className={`font-data text-xs w-14 text-right ${p.gain_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct?.toFixed(1)}%
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </CollapsibleSection>
          </Card>

          {/* Top Candidates */}
          <Card variant="glass" animate stagger={3}>
            <CollapsibleSection
              title="Top Candidates"
              badge={<span className="text-[10px] font-data text-dark-500">{candidates?.length || 0}</span>}
            >
              <div className="max-h-72 overflow-y-auto -mx-1">
                {candidates?.length === 0 && (
                  <div className="text-dark-500 text-xs py-6 text-center">No candidates above threshold</div>
                )}
                {candidates?.map(c => (
                  <Link
                    key={c.ticker}
                    to={`/stock/${c.ticker}`}
                    className="flex items-center justify-between py-2 px-2 rounded-lg hover:bg-dark-750/50 transition-colors group"
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs font-semibold text-primary-400 w-11 group-hover:text-primary-300">{c.ticker}</span>
                      <span className="text-[10px] text-dark-500 truncate max-w-[60px]">{c.sector?.split(' ')[0]}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <ScoreBadge score={c.score} size="xs" />
                      {c.audit_confidence != null && (
                        <span className={`text-[9px] font-data px-1 py-0.5 rounded ${
                          c.audit_confidence >= 70 ? 'text-emerald-400 bg-emerald-500/10' :
                          c.audit_confidence >= 50 ? 'text-amber-400 bg-amber-500/10' :
                          'text-red-400 bg-red-500/10'
                        }`}>
                          A{c.audit_confidence?.toFixed(0)}
                        </span>
                      )}
                      {c.projected_growth > 0 && (
                        <span className="text-emerald-400 text-[10px] font-data">
                          +{c.projected_growth?.toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </Link>
                ))}
              </div>
            </CollapsibleSection>
          </Card>
        </div>

        {/* ═══ RIGHT COLUMN ═══ */}
        <div className="md:col-span-4 space-y-3">
          {/* Coiled Spring */}
          <div className="animate-fade-in-up opacity-0 stagger-3">
            <CoiledSpringSection cs={coiled_spring} />
          </div>

          {/* Earnings Countdown */}
          <Card variant="glass" animate stagger={4}>
            <CollapsibleSection title="Earnings">
              {earnings?.length === 0 ? (
                <div className="text-dark-500 text-xs py-4 text-center">No upcoming earnings</div>
              ) : (
                <div className="space-y-0.5">
                  {earnings?.slice(0, 6).map(e => (
                    <div key={e.ticker} className="flex items-center justify-between py-1.5">
                      <Link to={`/stock/${e.ticker}`} className="text-xs font-medium text-primary-400 hover:text-primary-300 transition-colors">
                        {e.ticker}
                      </Link>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-data px-1.5 py-0.5 rounded ${
                          e.days <= 7 ? 'bg-red-500/10 text-red-400' :
                          e.days <= 14 ? 'bg-amber-500/10 text-amber-400' :
                          'bg-dark-700 text-dark-400'
                        }`}>
                          {e.days}d
                        </span>
                        {e.beat_streak > 0 && (
                          <span className="text-[10px] font-data text-dark-500">{e.beat_streak}x</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CollapsibleSection>
          </Card>

          {/* Recent Trades */}
          <Card variant="glass" animate stagger={5}>
            <CollapsibleSection title="Trades">
              <div className="space-y-0.5">
                {trades?.slice(0, 6).map((t, i) => (
                  <div key={i} className="flex items-center justify-between py-1.5">
                    <div className="flex items-center gap-2">
                      <ActionBadge action={t.action} />
                      <Link to={`/stock/${t.ticker}`} className="text-xs font-medium text-primary-400 hover:text-primary-300 transition-colors">
                        {t.ticker}
                      </Link>
                    </div>
                    <span className="text-[10px] font-data text-dark-500">
                      {t.executed_at ? new Date(t.executed_at).toLocaleDateString('en-US', {
                        month: 'numeric', day: 'numeric', timeZone: 'America/Chicago'
                      }) : '-'}
                    </span>
                  </div>
                ))}
              </div>
            </CollapsibleSection>
          </Card>

          {/* Mobile Risk (hidden on desktop, shown above) */}
          <Card variant="glass" className="md:hidden">
            <CollapsibleSection title="Risk" defaultOpen={false}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-dark-400">Heat</span>
                <span className={`text-sm font-data font-semibold ${
                  risk?.heat_status === 'normal' ? 'text-emerald-400' :
                  risk?.heat_status === 'warning' ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {risk?.portfolio_heat?.toFixed(1)}%
                </span>
              </div>
              {risk?.top_sectors?.slice(0, 3).map(s => (
                <div key={s.sector} className="flex justify-between text-[11px] py-1 border-b border-dark-700/20 last:border-0">
                  <span className="text-dark-400 truncate max-w-[100px]">{s.sector}</span>
                  <span className="font-data text-dark-300">{s.pct}%</span>
                </div>
              ))}
            </CollapsibleSection>
          </Card>

          {/* Scanner Status */}
          <Card variant="glass" padding="px-4 py-3" animate stagger={6}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Scanner</span>
                {scanner?.is_scanning ? (
                  <span className="flex items-center gap-1.5 text-[10px] text-primary-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-pulse-dot" />
                    {scanner?.phase || 'scanning'}
                  </span>
                ) : (
                  <span className="text-[10px] text-dark-500">IDLE</span>
                )}
              </div>
              <div className="flex items-center gap-3">
                {scanner?.is_scanning && scanner?.stocks_scanned != null && (
                  <span className="text-[10px] font-data text-dark-400">
                    {scanner.stocks_scanned}/{scanner.total_stocks}
                  </span>
                )}
                {scanner?.last_scan_end && (
                  <span className="text-[10px] font-data text-dark-500">
                    {new Date(scanner.last_scan_end).toLocaleTimeString('en-US', {
                      timeZone: 'America/Chicago', hour: '2-digit', minute: '2-digit'
                    })}
                  </span>
                )}
              </div>
            </div>
          </Card>

          {/* Mobile Quick Actions */}
          <div className="flex gap-2 md:hidden">
            <button
              onClick={() => handleAction('cycle')}
              disabled={!!runningAction}
              className="flex-1 text-xs font-medium py-2.5 rounded-lg bg-primary-600/15 text-primary-400 border border-primary-500/20 hover:bg-primary-600/25 transition-colors disabled:opacity-50"
            >
              {runningAction === 'cycle' ? 'Running...' : 'Run Cycle'}
            </button>
            <button
              onClick={() => handleAction('scan')}
              disabled={!!runningAction || scanner?.is_scanning}
              className="flex-1 text-xs font-medium py-2.5 rounded-lg bg-dark-700 text-dark-300 border border-dark-600 hover:bg-dark-600 transition-colors disabled:opacity-50"
            >
              {runningAction === 'scan' ? 'Starting...' : 'Start Scan'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
