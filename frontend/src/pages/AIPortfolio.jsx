import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent } from '../api'
import { XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine, PieChart, Pie, Cell, Area, AreaChart } from 'recharts'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, ActionBadge, TagBadge } from '../components/Badge'
import StatGrid, { StatRow } from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import Modal from '../components/Modal'

// ── Collapsible Section (local helper) ──────────────────────────────
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

// ── Performance Chart ───────────────────────────────────────────────
function PerformanceChart({ history, startingCash }) {
  const [timeRange, setTimeRange] = useState('all')

  if (!history || history.length < 2) {
    return (
      <Card variant="glass" className="mb-4 h-48 flex items-center justify-center text-dark-400">
        Not enough data for chart yet
      </Card>
    )
  }

  // Filter history based on selected time range
  // For multi-day views (7d, 30d, all), keep latest snapshot per day for clean chart
  // For 24h view, show all intraday snapshots for granularity
  const filterHistory = (data, range) => {
    let filtered = data.filter(d => d.timestamp || d.date)
    filtered.sort((a, b) => new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date))

    if (range !== 'all') {
      const now = new Date()
      const cutoff = new Date()
      if (range === '24h') cutoff.setHours(now.getHours() - 24)
      else if (range === '7d') cutoff.setDate(now.getDate() - 7)
      else if (range === '30d') cutoff.setDate(now.getDate() - 30)
      filtered = filtered.filter(d => new Date(d.timestamp || d.date) >= cutoff)
    }

    // For longer views with many data points, dedupe to latest per day
    // But only if there are enough unique days to make a readable chart (7+)
    if (range !== '24h' && filtered.length > 60) {
      const uniqueDays = new Set(filtered.map(d => new Date(d.timestamp || d.date).toISOString().slice(0, 10)))
      if (uniqueDays.size >= 7) {
        const byDay = {}
        for (const d of filtered) {
          const dayKey = new Date(d.timestamp || d.date).toISOString().slice(0, 10)
          if (!byDay[dayKey] || new Date(d.timestamp || d.date) > new Date(byDay[dayKey].timestamp || byDay[dayKey].date)) {
            byDay[dayKey] = d
          }
        }
        filtered = Object.values(byDay).sort((a, b) =>
          new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date)
        )
      }
    }

    return filtered
  }

  const filteredHistory = filterHistory(history, timeRange)
  const latestValue = filteredHistory[filteredHistory.length - 1]?.total_value || startingCash
  const firstValue = filteredHistory[0]?.total_value || startingCash
  const isPositive = latestValue >= firstValue

  // Format timestamp for tooltip - convert to CST
  const formatTimestamp = (ts) => {
    if (!ts) return ''
    try {
      const date = new Date(ts)
      return date.toLocaleString('en-US', {
        timeZone: 'America/Chicago',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) + ' CST'
    } catch {
      return ts
    }
  }

  const timeRanges = [
    { value: '24h', label: '24H' },
    { value: '7d', label: '7D' },
    { value: '30d', label: '30D' },
    { value: 'all', label: 'All' },
  ]

  const lineColor = isPositive ? '#10b981' : '#ef4444'
  const gradientId = isPositive ? 'perfGradientGreen' : 'perfGradientRed'

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Performance</span>
        <div className="flex items-center gap-2">
          <div className="flex bg-dark-850 rounded-lg p-0.5">
            {timeRanges.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setTimeRange(value)}
                className={`px-2 py-0.5 text-xs rounded transition-colors ${
                  timeRange === value
                    ? 'bg-primary-500 text-white'
                    : 'text-dark-400 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="text-dark-500 text-[10px] font-data">{filteredHistory.length} pts</span>
        </div>
      </div>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={filteredHistory}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={lineColor} stopOpacity={0.25} />
                <stop offset="100%" stopColor={lineColor} stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="total_value"
              stroke={lineColor}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={filteredHistory.length <= 50}
              activeDot={{ r: 4, fill: lineColor }}
            />
            <ReferenceLine
              y={startingCash}
              stroke="#666"
              strokeDasharray="3 3"
              label={{ value: 'Start', position: 'right', fill: '#666', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(20, 20, 31, 0.95)',
                border: '1px solid #222233',
                borderRadius: '10px',
                fontFamily: 'JetBrains Mono',
                fontSize: 12,
              }}
              labelStyle={{ color: '#6e6e82' }}
              formatter={(value) => [formatCurrency(value), 'Value']}
              labelFormatter={(_, payload) => {
                if (payload && payload[0]) {
                  return formatTimestamp(payload[0].payload.timestamp || payload[0].payload.date)
                }
                return ''
              }}
            />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={['dataMin - 500', 'dataMax + 500']} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

// ── Summary Card ────────────────────────────────────────────────────
function SummaryCard({ summary, config }) {
  if (!summary) return null

  const isPositive = summary.total_return >= 0

  return (
    <Card variant="glass" className="mb-4">
      <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">AI Portfolio Value</span>
      <div className="text-3xl font-bold font-data mt-1 mb-1">
        {formatCurrency(summary.total_value)}
      </div>
      <div className={`text-sm flex items-center gap-1.5 font-data ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
        <span>{isPositive ? '+' : ''}{formatCurrency(Math.abs(summary.total_return))}</span>
        <span className="text-dark-500">({formatPercent(summary.total_return_pct, true)})</span>
      </div>

      <div className="border-t border-dark-700/50 mt-4 pt-3">
        <StatGrid
          columns={3}
          stats={[
            { label: 'Cash', value: formatCurrency(summary.cash) },
            { label: 'Invested', value: formatCurrency(summary.positions_value) },
            { label: 'Positions', value: `${summary.positions_count} / ${config?.max_positions || 15}` },
          ]}
        />
      </div>
    </Card>
  )
}

// ── Positions List ──────────────────────────────────────────────────
function PositionsList({ positions }) {
  if (!positions || positions.length === 0) {
    return (
      <Card variant="glass" className="mb-4 text-center py-8 text-dark-400">
        No positions yet. Initialize the portfolio to start trading.
      </Card>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title={`Positions (${positions.length})`} />
      <div className="space-y-1">
        {positions.map(position => (
          <Link
            key={position.id}
            to={`/stock/${position.ticker}`}
            className="flex justify-between items-center py-2.5 border-b border-dark-700/30 last:border-0 hover:bg-dark-750/50 -mx-2 px-2 rounded transition-colors"
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-dark-100">{position.ticker}</span>
                {position.is_growth_stock && (
                  <TagBadge color="purple">Growth</TagBadge>
                )}
              </div>
              <div className="text-dark-400 text-[10px] font-data mt-0.5">
                {position.shares.toFixed(2)} shares @ {formatCurrency(position.cost_basis)}
              </div>
            </div>
            <div className="text-right">
              <div className="font-semibold font-data text-dark-100">{formatCurrency(position.current_value)}</div>
              <span className={`text-[10px] font-data ${position.gain_loss_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {position.gain_loss_pct >= 0 ? '+' : ''}{position.gain_loss_pct?.toFixed(2)}%
              </span>
            </div>
            <div className="ml-3 text-right">
              {/* Show appropriate score based on stock type */}
              {position.is_growth_stock ? (
                <ScoreBadge score={position.current_growth_score} size="sm" />
              ) : (
                <ScoreBadge score={position.current_score} size="sm" />
              )}
              {/* Show secondary score if available */}
              {position.is_growth_stock && position.current_score > 0 && (
                <div className="text-[10px] text-dark-400 font-data mt-0.5">
                  CANSLIM: {position.current_score?.toFixed(0)}
                </div>
              )}
              {!position.is_growth_stock && position.current_growth_score > 0 && (
                <div className="text-[10px] text-dark-400 font-data mt-0.5">
                  Growth: {position.current_growth_score?.toFixed(0)}
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>
    </Card>
  )
}

// ── Trade Detail Modal ──────────────────────────────────────────────
function TradeDetailModal({ trade, onClose }) {
  if (!trade) return null

  const formatDateTime = (ts) => {
    if (!ts) return 'N/A'
    try {
      const date = new Date(ts)
      return date.toLocaleString('en-US', {
        timeZone: 'America/Chicago',
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }) + ' CST'
    } catch {
      return ts
    }
  }

  const gainPct = trade.action === 'SELL' && trade.cost_basis
    ? ((trade.price - trade.cost_basis) / trade.cost_basis * 100)
    : null

  return (
    <Modal
      open={!!trade}
      onClose={onClose}
      title={
        <div className="flex items-center gap-2">
          <ActionBadge action={trade.action} />
          <Link
            to={`/stock/${trade.ticker}`}
            className="text-sm font-bold text-primary-400 hover:underline"
            onClick={onClose}
          >
            {trade.ticker}
          </Link>
          {trade.is_growth_stock && <TagBadge color="purple">Growth</TagBadge>}
        </div>
      }
    >
      <div className="space-y-0">
        <StatRow label="Date & Time" value={formatDateTime(trade.executed_at)} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Shares" value={<span className="font-data">{trade.shares.toFixed(4)}</span>} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Price" value={<span className="font-data">{formatCurrency(trade.price)}</span>} />

        <div className="border-b border-dark-700/30" />
        <StatRow label="Total Value" value={<span className="font-data">{formatCurrency(trade.total_value)}</span>} />

        {trade.action === 'SELL' && trade.cost_basis && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow label="Cost Basis" value={<span className="font-data">{formatCurrency(trade.cost_basis)}/share</span>} />
          </>
        )}

        {trade.realized_gain != null && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow
              label="Realized Gain/Loss"
              value={
                <span className={`font-data font-medium ${trade.realized_gain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
                  {gainPct != null && ` (${gainPct >= 0 ? '+' : ''}${gainPct.toFixed(1)}%)`}
                </span>
              }
            />
          </>
        )}

        <div className="border-b border-dark-700/30" />
        <StatRow label="CANSLIM Score" value={<span className="font-data">{trade.canslim_score?.toFixed(1) || 'N/A'}</span>} />

        {trade.is_growth_stock && trade.growth_mode_score && (
          <>
            <div className="border-b border-dark-700/30" />
            <StatRow label="Growth Mode Score" value={<span className="font-data">{trade.growth_mode_score.toFixed(1)}</span>} />
          </>
        )}

        {/* Reason Section */}
        <div className="pt-3">
          <span className="text-[10px] font-semibold tracking-widest uppercase text-dark-400">Reason</span>
          <div className="bg-dark-850 rounded-lg p-3 text-sm text-dark-200 mt-1.5">
            {trade.reason || 'No reason recorded'}
          </div>
        </div>
      </div>
    </Modal>
  )
}

// ── Trade History ───────────────────────────────────────────────────
function TradeHistory({ trades }) {
  const [selectedTrade, setSelectedTrade] = useState(null)

  if (!trades || trades.length === 0) {
    return null
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Recent Trades" />
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {trades.slice(0, 20).map(trade => (
          <div
            key={trade.id}
            onClick={() => setSelectedTrade(trade)}
            className="flex justify-between items-center py-2 border-b border-dark-700/30 last:border-0 text-sm cursor-pointer hover:bg-dark-750/50 rounded px-2 -mx-2 transition-colors"
          >
            <div className="flex items-center gap-2">
              <ActionBadge action={trade.action} />
              <span className="font-medium text-dark-100">{trade.ticker}</span>
              {trade.is_growth_stock && <TagBadge color="purple">G</TagBadge>}
            </div>
            <div className="text-right">
              <div className="font-data text-dark-200">
                {trade.shares.toFixed(2)} @ {formatCurrency(trade.price)}
              </div>
              <div className="text-dark-400 text-[10px] truncate max-w-[150px]">{trade.reason}</div>
            </div>
            {trade.realized_gain != null && (
              <span className={`ml-2 text-xs font-data ${trade.realized_gain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {trade.realized_gain >= 0 ? '+' : ''}{formatCurrency(trade.realized_gain)}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Trade Detail Modal */}
      <TradeDetailModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
    </Card>
  )
}

// ── Sector Allocation Chart ─────────────────────────────────────────
const SECTOR_COLORS = ['#10b981', '#22d3ee', '#f59e0b', '#ef4444', '#a855f7',
  '#ec4899', '#6366f1', '#3b82f6', '#22c55e', '#eab308']

function SectorAllocationChart({ riskData, cashPct }) {
  if (!riskData?.sector_concentration || riskData.sector_concentration.length === 0) return null

  const chartData = [
    ...riskData.sector_concentration.map(s => ({ name: s.sector || 'Unknown', value: s.pct })),
    ...(cashPct > 1 ? [{ name: 'Cash', value: Math.round(cashPct * 10) / 10 }] : [])
  ]

  const renderLabel = ({ name, value, cx, cy, midAngle, innerRadius, outerRadius }) => {
    if (value < 5) return null
    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)
    return (
      <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={10} fontFamily="JetBrains Mono">
        {Math.round(value)}%
      </text>
    )
  }

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Sector Allocation" />
      <div className="flex items-center">
        <div className="w-1/2 h-40">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={60}
                dataKey="value"
                label={renderLabel}
                labelLine={false}
              >
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.name === 'Cash' ? '#4a4a5e' : SECTOR_COLORS[i % SECTOR_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'rgba(20, 20, 31, 0.95)',
                  border: '1px solid #222233',
                  borderRadius: '10px',
                  fontFamily: 'JetBrains Mono',
                  fontSize: 12,
                }}
                formatter={(v) => `${v.toFixed(1)}%`}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="w-1/2 space-y-1 text-xs">
          {chartData.map((entry, i) => (
            <div key={entry.name} className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: entry.name === 'Cash' ? '#4a4a5e' : SECTOR_COLORS[i % SECTOR_COLORS.length] }} />
              <span className="text-dark-300 truncate">{entry.name}</span>
              <span className="text-dark-400 ml-auto font-data">{entry.value.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  )
}

// ── Config Panel ────────────────────────────────────────────────────
function ConfigPanel({ config, onUpdate, onInitialize, onRunCycle, onRefresh, waitingForTrades }) {
  const [isActive, setIsActive] = useState(config?.is_active || false)
  const [updating, setUpdating] = useState(false)
  const [initializing, setInitializing] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  useEffect(() => {
    setIsActive(config?.is_active || false)
  }, [config])

  const handleToggle = async () => {
    setUpdating(true)
    try {
      await onUpdate({ is_active: !isActive })
      setIsActive(!isActive)
    } finally {
      setUpdating(false)
    }
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await onRefresh()
      // Keep spinner for 12 seconds while background task runs
      setTimeout(() => setRefreshing(false), 12000)
    } catch {
      setRefreshing(false)
    }
  }

  const handleRunCycle = async () => {
    try {
      await onRunCycle()
    } catch (err) {
      console.error('Failed to run cycle:', err)
    }
  }

  const handleInitialize = async () => {
    if (!confirm('This will reset the AI Portfolio to $25,000 and clear all history. Continue?')) {
      return
    }
    setInitializing(true)
    try {
      await onInitialize()
    } finally {
      setInitializing(false)
    }
  }

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-3">
        <CardHeader title="AI Trading" className="mb-0" />
        <button
          onClick={handleToggle}
          disabled={updating}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            isActive ? 'bg-emerald-500' : 'bg-dark-600'
          }`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            isActive ? 'translate-x-7' : 'translate-x-1'
          }`} />
        </button>
      </div>

      {isActive && (
        <div className="mb-3 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-emerald-400 text-sm">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>AI Trading Active - Trades execute after each scan</span>
          </div>
        </div>
      )}

      <div className="border-t border-dark-700/30 pt-3 mb-3">
        <StatGrid
          columns={2}
          stats={[
            { label: 'Min Score to Buy', value: config?.min_score_to_buy || '72' },
            { label: 'Strategy', value: config?.strategy?.replace(/_/g, ' ') || 'balanced' },
            { label: 'Take Profit', value: `+${config?.take_profit_pct || 75}%`, color: 'text-emerald-400' },
            { label: 'Stop Loss', value: `-${config?.stop_loss_pct || 7}%`, color: 'text-red-400' },
          ]}
          className="text-sm"
        />
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex-1 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
        >
          <svg
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            className={refreshing ? 'animate-spin' : ''}
          >
            <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
          </svg>
          <span>{refreshing ? 'Refreshing...' : 'Refresh Prices'}</span>
        </button>
        <button
          onClick={handleRunCycle}
          disabled={waitingForTrades}
          className="flex-1 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {waitingForTrades && (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
          )}
          <span>{waitingForTrades ? 'Running...' : 'Run Trading Cycle'}</span>
        </button>
      </div>

      <button
        onClick={handleInitialize}
        disabled={initializing}
        className="w-full py-2 mb-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-dark-300"
      >
        {initializing ? 'Resetting...' : 'Reset Portfolio ($25k)'}
      </button>

      <Link
        to="/backtest"
        className="block w-full py-2 bg-dark-700 hover:bg-dark-600 rounded-lg text-sm font-medium transition-colors text-center text-primary-400"
      >
        Run Historical Backtest
      </Link>
    </Card>
  )
}

// ── Coiled Spring Alerts Section ────────────────────────────────────
function CoiledSpringSection({ csAlerts, csExpanded, setCsExpanded }) {
  if (!csAlerts || csAlerts.length === 0) return null

  return (
    <Card variant="accent" accent="purple" className="mb-4 bg-purple-500/[0.03]">
      <CollapsibleSection
        title="Coiled Spring Alerts"
        badge={<TagBadge color="purple">{csAlerts.length} candidates</TagBadge>}
        defaultOpen={csExpanded}
      >
        <div className="text-[10px] text-dark-400 mb-2">
          High-conviction pre-earnings plays: long bases + beat streaks + approaching earnings
        </div>
        <div className="space-y-1">
          {csAlerts.map((stock) => (
            <Link
              key={stock.ticker}
              to={`/stock/${stock.ticker}`}
              className="flex justify-between items-center py-2 px-2 -mx-2 rounded hover:bg-dark-750/50 transition-colors border-b border-dark-700/30 last:border-0"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-dark-100">{stock.ticker}</span>
                  {stock.base_type && stock.base_type !== 'none' && (
                    <TagBadge color="cyan">{stock.weeks_in_base}w {stock.base_type}</TagBadge>
                  )}
                  {stock.is_breaking_out && (
                    <TagBadge color="amber">Breakout</TagBadge>
                  )}
                </div>
                <div className="text-[10px] text-dark-400 flex gap-2 mt-0.5 font-data">
                  <span>C:{stock.c_score?.toFixed(0)}</span>
                  <span>L:{stock.l_score?.toFixed(1)}</span>
                  <span className="text-dark-500">|</span>
                  <span>{stock.earnings_beat_streak} beats</span>
                  <span className="text-dark-500">|</span>
                  <span className="text-amber-400">{stock.days_to_earnings}d to earnings</span>
                  <span className="text-dark-500">|</span>
                  <span>{stock.institutional_holders_pct?.toFixed(1)}% inst</span>
                </div>
              </div>
              <div className="text-right">
                <ScoreBadge score={stock.canslim_score} size="sm" />
                <div className="text-[10px] text-purple-400 font-data mt-0.5">+{stock.cs_bonus} bonus</div>
              </div>
            </Link>
          ))}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ── Risk Monitor Section ────────────────────────────────────────────
function RiskMonitorSection({ riskData, riskExpanded, setRiskExpanded }) {
  if (!riskData) return null

  const heatColor = riskData.heat_status === 'danger' ? 'red'
    : riskData.heat_status === 'warning' ? 'amber' : 'green'

  return (
    <Card variant="glass" className="mb-4">
      <CollapsibleSection
        title="Risk Monitor"
        badge={
          <div className="flex items-center gap-1.5">
            <TagBadge color={heatColor}>Heat: {riskData.portfolio_heat}%</TagBadge>
            {riskData.position_alerts?.length > 0 && (
              <TagBadge color="red">{riskData.position_alerts.length} alerts</TagBadge>
            )}
          </div>
        }
        defaultOpen={riskExpanded}
      >
        <div className="space-y-3 mt-1">
          {/* Heat bar */}
          <div>
            <div className="text-[10px] text-dark-400 mb-1">Portfolio Heat</div>
            <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  riskData.portfolio_heat < 10 ? 'bg-emerald-500' :
                  riskData.portfolio_heat < 15 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(riskData.portfolio_heat / 20 * 100, 100)}%` }}
              />
            </div>
          </div>
          {/* Sector concentration */}
          {riskData.sector_concentration?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-400 mb-1">Sector Concentration</div>
              <div className="flex flex-wrap gap-1">
                {riskData.sector_concentration.map(s => (
                  <TagBadge
                    key={s.sector}
                    color={s.count >= 3 ? 'amber' : 'default'}
                  >
                    {s.sector}: {s.count} ({s.pct}%)
                  </TagBadge>
                ))}
              </div>
            </div>
          )}
          {/* Stop distances */}
          {riskData.stop_distances?.length > 0 && (
            <div>
              <div className="text-[10px] text-dark-400 mb-1">Distance to Stop</div>
              {riskData.stop_distances.slice(0, 5).map(s => (
                <div key={s.ticker} className="flex justify-between text-xs py-0.5">
                  <span className="font-medium text-dark-200">{s.ticker}</span>
                  <span className={`font-data ${s.distance_pct < 5 ? 'text-red-400' : 'text-dark-300'}`}>
                    {s.distance_pct}% ({s.gain_pct >= 0 ? '+' : ''}{s.gain_pct}%)
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ── Earnings Calendar Section ───────────────────────────────────────
function EarningsCalendarSection({ earningsCalendar, earningsExpanded, setEarningsExpanded }) {
  if (!earningsCalendar || !earningsCalendar.positions?.length) return null

  return (
    <Card variant="glass" className="mb-4">
      <CollapsibleSection
        title="Earnings Calendar"
        badge={
          <div className="flex items-center gap-1.5">
            {earningsCalendar.upcoming_count?.high > 0 && (
              <TagBadge color="red">{earningsCalendar.upcoming_count.high} this week</TagBadge>
            )}
            {earningsCalendar.upcoming_count?.medium > 0 && (
              <TagBadge color="amber">{earningsCalendar.upcoming_count.medium} next week</TagBadge>
            )}
          </div>
        }
        defaultOpen={earningsExpanded}
      >
        <div className="space-y-1 mt-1">
          {earningsCalendar.positions.map(p => (
            <div key={p.ticker} className={`flex justify-between items-center py-1.5 px-2 -mx-2 rounded ${
              p.risk_level === 'high' ? 'bg-red-500/5' : ''
            }`}>
              <div>
                <span className="font-medium text-sm text-dark-100">{p.ticker}</span>
                <span className="text-dark-400 text-xs ml-2 font-data">
                  {p.next_earnings_date || `${p.days_to_earnings}d`}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-dark-400 font-data">{p.beat_streak} beats</span>
                <TagBadge color={
                  p.risk_level === 'high' ? 'red' :
                  p.risk_level === 'medium' ? 'amber' : 'green'
                }>
                  {p.days_to_earnings}d
                </TagBadge>
                <span className={`text-xs font-data ${p.gain_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </CollapsibleSection>
    </Card>
  )
}

// ══════════════════════════════════════════════════════════════════════
// ── Main Page Component ─────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════
export default function AIPortfolio() {
  const [loading, setLoading] = useState(true)
  const [portfolio, setPortfolio] = useState(null)
  const [history, setHistory] = useState([])
  const [trades, setTrades] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [waitingForTrades, setWaitingForTrades] = useState(false)
  const [waitingCash, setWaitingCash] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(() => {
    // Persist auto-refresh preference in localStorage
    const saved = localStorage.getItem('aiPortfolioAutoRefresh')
    return saved === 'true'
  })
  const [lastPriceRefresh, setLastPriceRefresh] = useState(null)
  const [isRefreshingPrices, setIsRefreshingPrices] = useState(false)
  const [csAlerts, setCsAlerts] = useState([])
  const [csExpanded, setCsExpanded] = useState(true)
  const [earningsCalendar, setEarningsCalendar] = useState(null)
  const [earningsExpanded, setEarningsExpanded] = useState(false)
  const [riskData, setRiskData] = useState(null)
  const [riskExpanded, setRiskExpanded] = useState(false)

  const fetchData = async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true)
      const [portfolioData, historyData, tradesData, csData, earningsData, riskInfo] = await Promise.all([
        api.getAIPortfolio(),
        api.getAIPortfolioHistory(90),
        api.getAIPortfolioTrades(50),
        api.getCoiledSpringCandidates().catch(() => ({ candidates: [] })),
        api.getEarningsCalendar().catch(() => null),
        api.getPortfolioRisk().catch(() => null),
      ])
      setPortfolio(portfolioData)
      setHistory(historyData)
      setTrades(tradesData)
      setCsAlerts(csData?.candidates || [])
      setEarningsCalendar(earningsData)
      setRiskData(riskInfo)
      setLastUpdated(new Date())

      // Check if data changed while waiting for trades
      if (waitingForTrades && waitingCash !== null) {
        const newCash = portfolioData?.summary?.cash
        if (Math.abs(newCash - waitingCash) > 100) {
          // Cash changed significantly - trades executed
          setWaitingForTrades(false)
          setWaitingCash(null)
        }
      }

      return portfolioData
    } catch (err) {
      console.error('Failed to fetch AI Portfolio:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()

    // Auto-refresh every 15 seconds to catch background updates
    const interval = setInterval(() => {
      fetchData(false)
    }, 15000)

    return () => clearInterval(interval)
  }, [])

  // Keep polling while waiting for trades
  useEffect(() => {
    if (!waitingForTrades) return

    const pollInterval = setInterval(() => {
      fetchData(false)
    }, 5000) // Poll every 5 seconds while waiting

    // Stop waiting after 2 minutes max
    const timeout = setTimeout(() => {
      setWaitingForTrades(false)
      setWaitingCash(null)
    }, 120000)

    return () => {
      clearInterval(pollInterval)
      clearTimeout(timeout)
    }
  }, [waitingForTrades])

  // Auto-refresh prices every 5 minutes when enabled
  useEffect(() => {
    if (!autoRefresh) return

    const AUTO_REFRESH_INTERVAL = 5 * 60 * 1000 // 5 minutes

    const refreshPrices = async () => {
      // Check if market is likely open (rough check - M-F 8:30am-4pm CST)
      const now = new Date()
      const cstHour = new Date(now.toLocaleString('en-US', { timeZone: 'America/Chicago' })).getHours()
      const dayOfWeek = now.getDay()
      const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5
      const isMarketHours = cstHour >= 8 && cstHour < 16

      if (!isWeekday || !isMarketHours) {
        // Market closed, skip auto-refresh
        return
      }

      setIsRefreshingPrices(true)
      try {
        await api.refreshAIPortfolio()
        setLastPriceRefresh(new Date())
        // Fetch updated data after refresh completes
        setTimeout(() => fetchData(false), 10000)
        setTimeout(() => setIsRefreshingPrices(false), 12000)
      } catch (err) {
        console.error('Auto-refresh failed:', err)
        setIsRefreshingPrices(false)
      }
    }

    // Refresh immediately on enable, then every 5 minutes
    refreshPrices()
    const interval = setInterval(refreshPrices, AUTO_REFRESH_INTERVAL)

    return () => clearInterval(interval)
  }, [autoRefresh])

  // Persist auto-refresh preference
  useEffect(() => {
    localStorage.setItem('aiPortfolioAutoRefresh', autoRefresh.toString())
  }, [autoRefresh])

  const handleUpdateConfig = async (config) => {
    try {
      await api.updateAIPortfolioConfig(config)
      fetchData()
    } catch (err) {
      console.error('Failed to update config:', err)
    }
  }

  const handleInitialize = async () => {
    try {
      await api.initializeAIPortfolio(25000)
      fetchData()
    } catch (err) {
      console.error('Failed to initialize:', err)
    }
  }

  const handleRefresh = async () => {
    setIsRefreshingPrices(true)
    try {
      const result = await api.refreshAIPortfolio()
      setLastPriceRefresh(new Date())
      // Poll more frequently after triggering a refresh
      if (result.status === 'started') {
        setTimeout(() => fetchData(false), 4000)
        setTimeout(() => fetchData(false), 8000)
        setTimeout(() => fetchData(false), 12000)
        setTimeout(() => setIsRefreshingPrices(false), 12000)
      } else {
        fetchData()
        setIsRefreshingPrices(false)
      }
    } catch (err) {
      console.error('Failed to refresh:', err)
      setIsRefreshingPrices(false)
    }
  }

  const handleRunCycle = async () => {
    try {
      // Store current cash to detect when trades complete
      const currentCash = portfolio?.summary?.cash || 0
      setWaitingCash(currentCash)
      setWaitingForTrades(true)

      const result = await api.runAITradingCycle()
      if (result.status === 'market_closed' || result.status === 'busy') {
        alert(result.message)
        setWaitingForTrades(false)
        fetchData()
      } else if (result.status !== 'started') {
        setWaitingForTrades(false)
        fetchData()
      }
    } catch (err) {
      console.error('Failed to run cycle:', err)
      setWaitingForTrades(false)
    }
  }

  // ── Loading skeleton ────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-48 mb-5 rounded-lg" />
        <div className="skeleton h-48 rounded-xl mb-4" />
        <div className="skeleton h-32 rounded-xl mb-4" />
        <div className="skeleton h-48 rounded-xl" />
      </div>
    )
  }

  // ── Page render ─────────────────────────────────────────────────────
  return (
    <div className="p-4 md:p-6">
      {/* Page Header */}
      <PageHeader
        title="AI Portfolio"
        subtitle={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
            <span>Started: <span className="font-data">{formatCurrency(portfolio?.config?.starting_cash || 25000)}</span></span>
            {lastUpdated && (
              <span>Data: <span className="font-data">{lastUpdated.toLocaleTimeString('en-US', { timeZone: 'America/Chicago' })} CST</span></span>
            )}
            {lastPriceRefresh && (
              <span>Prices: <span className="font-data">{lastPriceRefresh.toLocaleTimeString('en-US', { timeZone: 'America/Chicago' })} CST</span></span>
            )}
          </span>
        }
        badge={
          portfolio?.config?.strategy && portfolio.config.strategy !== 'balanced'
            ? <TagBadge color={portfolio.config.strategy === 'growth' ? 'purple' : 'blue'}>
                {portfolio.config.strategy.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </TagBadge>
            : null
        }
      />

      {/* Auto-refresh toggle */}
      <Card variant="glass" className="mb-4" padding="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              className={`text-primary-400 ${isRefreshingPrices ? 'animate-spin' : ''}`}
            >
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
            <div>
              <div className="text-sm font-medium text-dark-100">Auto-Refresh Prices</div>
              <div className="text-[10px] text-dark-400">
                {autoRefresh ? 'Every 5 min during market hours' : 'Disabled'}
              </div>
            </div>
          </div>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`relative w-12 h-6 rounded-full transition-colors ${
              autoRefresh ? 'bg-emerald-500' : 'bg-dark-600'
            }`}
          >
            <span
              className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform duration-200 ${
                autoRefresh ? 'translate-x-6' : ''
              }`}
            />
          </button>
        </div>
      </Card>

      {/* Waiting for trades banner */}
      {waitingForTrades && (
        <Card variant="glass" className="mb-4 border-primary-500/30 bg-primary-500/5" padding="p-3">
          <div className="flex items-center gap-2 text-primary-400">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="animate-spin">
              <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
            <span className="font-medium text-sm">Executing trades... This may take up to 2 minutes.</span>
          </div>
          <div className="text-dark-400 text-[10px] mt-1">Page will auto-update when complete.</div>
        </Card>
      )}

      {/* Coiled Spring Alerts */}
      <CoiledSpringSection
        csAlerts={csAlerts}
        csExpanded={csExpanded}
        setCsExpanded={setCsExpanded}
      />

      {/* Paper Mode Banner */}
      {portfolio?.config?.paper_mode && (
        <Card variant="glass" className="mb-4 border-amber-500/30 bg-amber-500/5" padding="p-3">
          <div className="flex items-center gap-2 text-amber-400">
            <span className="text-sm font-semibold">PAPER MODE</span>
            <span className="text-[10px] text-dark-400">Trades are simulated - no real positions affected</span>
          </div>
        </Card>
      )}

      {/* Risk Monitor */}
      <RiskMonitorSection
        riskData={riskData}
        riskExpanded={riskExpanded}
        setRiskExpanded={setRiskExpanded}
      />

      {/* Earnings Calendar */}
      <EarningsCalendarSection
        earningsCalendar={earningsCalendar}
        earningsExpanded={earningsExpanded}
        setEarningsExpanded={setEarningsExpanded}
      />

      <PerformanceChart
        history={history}
        startingCash={portfolio?.config?.starting_cash || 25000}
      />

      <SectorAllocationChart
        riskData={riskData}
        cashPct={portfolio?.summary?.total_value > 0
          ? (portfolio.summary.cash / portfolio.summary.total_value) * 100
          : 0}
      />

      <SummaryCard
        summary={portfolio?.summary}
        config={portfolio?.config}
      />

      <ConfigPanel
        config={portfolio?.config}
        onUpdate={handleUpdateConfig}
        onInitialize={handleInitialize}
        onRefresh={handleRefresh}
        onRunCycle={handleRunCycle}
        waitingForTrades={waitingForTrades}
      />

      <PositionsList positions={portfolio?.positions} />

      <TradeHistory trades={trades} />

      {/* Links */}
      <SectionLabel>More</SectionLabel>
      <div className="flex gap-4 mb-4">
        <Link to="/analytics" className="text-xs text-primary-400 hover:text-primary-300 transition-colors">
          Trade Analytics
        </Link>
        <Link to="/backtest" className="text-xs text-primary-400 hover:text-primary-300 transition-colors">
          Run Backtest
        </Link>
      </div>

      <div className="h-4" />
    </div>
  )
}
