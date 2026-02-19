import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, getScoreClass, formatCurrency, formatTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, TagBadge, PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'

// ---------------------------------------------------------------------------
// IndexCard (market direction)
// ---------------------------------------------------------------------------
function IndexCard({ ticker, label, weight, price, ma50, ma200, signal }) {
  const signalConfig = {
    2: { icon: '\u25B2\u25B2', color: 'text-green-400', label: 'Strong' },
    1: { icon: '\u25B2', color: 'text-green-400', label: 'Bullish' },
    0: { icon: '\u25BA', color: 'text-yellow-400', label: 'Neutral' },
    '-1': { icon: '\u25BC', color: 'text-red-400', label: 'Bearish' },
  }

  const config = signalConfig[signal] || signalConfig[0]
  const above50 = price > ma50
  const above200 = price > ma200

  return (
    <Card variant="stat" padding="p-3" rounded="rounded-lg">
      <div className="flex justify-between items-center mb-2">
        <div>
          <span className="font-semibold text-sm">{ticker}</span>
          <span className="text-dark-500 text-xs ml-1 font-data">({weight}%)</span>
        </div>
        <span className={`text-xs font-medium ${config.color}`}>
          {config.icon}
        </span>
      </div>
      <div className="text-lg font-bold font-data mb-2">
        {price ? formatCurrency(price) : '-'}
      </div>
      <div className="space-y-1 text-xs">
        <div className={`flex justify-between items-center px-1.5 py-0.5 rounded ${above50 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
          <span>50MA</span>
          <span className="font-medium font-data">{ma50 ? formatCurrency(ma50) : '-'} {above50 ? '\u25B2' : '\u25BC'}</span>
        </div>
        <div className={`flex justify-between items-center px-1.5 py-0.5 rounded ${above200 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
          <span>200MA</span>
          <span className="font-medium font-data">{ma200 ? formatCurrency(ma200) : '-'} {above200 ? '\u25B2' : '\u25BC'}</span>
        </div>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// MarketStatus
// ---------------------------------------------------------------------------
function MarketStatus({ market, onRefresh }) {
  const [refreshing, setRefreshing] = useState(false)

  if (!market) return null

  const trendColors = {
    bullish: 'text-green-400',
    neutral: 'text-yellow-400',
    bearish: 'text-red-400'
  }

  const trendIcons = {
    bullish: '\u25B2',
    neutral: '\u25BA',
    bearish: '\u25BC'
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await onRefresh()
    } finally {
      setRefreshing(false)
    }
  }

  // Extract index data (new multi-index format)
  const indexes = market.indexes || {}
  const spy = indexes.SPY || {}
  const qqq = indexes.QQQ || {}
  const dia = indexes.DIA || {}

  // Fallback to old format if new format not available
  const spyPrice = spy.price || market.spy_price
  const spyMa50 = spy.ma_50 || market.spy_50_ma
  const spyMa200 = spy.ma_200 || market.spy_200_ma
  const spySignal = spy.signal ?? (spyPrice > spyMa200 ? (spyPrice > spyMa50 ? 2 : 1) : (spyPrice > spyMa50 ? 0 : -1))

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader
        title="Market Direction"
        action={
          <div className="flex items-center gap-3">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="text-dark-400 hover:text-white transition-colors p-1"
              title="Refresh market data"
            >
              <span className={refreshing ? 'animate-spin inline-block' : ''}>{'\u27F3'}</span>
            </button>
            <span className={`px-2 py-1 rounded text-sm font-medium font-data ${getScoreClass((market.score / 15) * 100)}`}>
              M: {market.score != null ? `${market.score.toFixed(1)}/15` : '-'}
            </span>
            <div className={`text-sm font-medium ${trendColors[market.trend] || 'text-dark-400'}`}>
              {trendIcons[market.trend]} {market.trend?.toUpperCase()}
            </div>
          </div>
        }
      />

      {/* Three Index Cards */}
      <div className="grid grid-cols-3 gap-3">
        <IndexCard
          ticker="SPY"
          label="S&P 500"
          weight={50}
          price={spyPrice}
          ma50={spyMa50}
          ma200={spyMa200}
          signal={spySignal}
        />
        <IndexCard
          ticker="QQQ"
          label="NASDAQ"
          weight={30}
          price={qqq.price}
          ma50={qqq.ma_50}
          ma200={qqq.ma_200}
          signal={qqq.signal}
        />
        <IndexCard
          ticker="DIA"
          label="Dow Jones"
          weight={20}
          price={dia.price}
          ma50={dia.ma_50}
          ma200={dia.ma_200}
          signal={dia.signal}
        />
      </div>

      {/* Weighted Signal */}
      {market.weighted_signal != null && (
        <div className="mt-3 pt-3 border-t border-dark-700 text-center">
          <span className="text-dark-400 text-xs">
            Weighted Signal: <span className={`font-medium font-data ${market.weighted_signal >= 1 ? 'text-green-400' : market.weighted_signal <= -0.5 ? 'text-red-400' : 'text-yellow-400'}`}>
              {market.weighted_signal.toFixed(2)}
            </span>
            <span className="text-dark-500 ml-2">(SPY x50% + QQQ x30% + DIA x20%)</span>
          </span>
        </div>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Inline helpers: ScoreTrend, WeeklyTrend
// ---------------------------------------------------------------------------
function ScoreTrend({ change }) {
  if (change == null) return null
  if (change === 0) return <span className="text-xs text-dark-400 font-data">-</span>
  return <PnlText value={change} className="text-xs" />
}

function WeeklyTrend({ trend, change }) {
  if (!trend) return null

  const config = {
    improving: { icon: '\u2197', color: 'green' },
    stable: { icon: '\u2192', color: 'default' },
    deteriorating: { icon: '\u2198', color: 'red' }
  }

  const { icon, color } = config[trend] || config.stable

  return (
    <TagBadge
      color={color}
      className="cursor-default"
    >
      <span title={`7-day trend: ${change != null ? (change > 0 ? '+' : '') + change.toFixed(1) : ''} pts`}>
        {icon} {trend === 'improving' ? 'Up' : trend === 'deteriorating' ? 'Down' : ''}
      </span>
    </TagBadge>
  )
}

// ---------------------------------------------------------------------------
// Ranked stock row (shared by TopStocksList, TopGrowthStocks, BreakingOutStocks, CoiledSpringAlerts)
// ---------------------------------------------------------------------------
function RankedStockRow({ stock, index, rankColor = 'bg-dark-600 text-dark-100', children, subtitle, badges, scoreValue, scoreClass }) {
  return (
    <Link
      to={`/stock/${stock.ticker}`}
      className="flex justify-between items-center py-1.5 border-b border-dark-700/30 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors"
    >
      <div className="flex items-center gap-2">
        <div className={`w-5 h-5 text-[10px] rounded-full flex items-center justify-center font-bold font-data ${rankColor}`}>
          {index + 1}
        </div>
        <div>
          <div className="flex items-center gap-1">
            <span className="font-medium text-sm">{stock.ticker}</span>
            {badges}
          </div>
          {subtitle && <div className="text-dark-500 text-[10px]">{subtitle}</div>}
        </div>
      </div>
      <div className="text-right">
        <ScoreBadge score={scoreValue ?? stock.canslim_score} size="xs" className={scoreClass || ''} />
        {children}
      </div>
    </Link>
  )
}

// ---------------------------------------------------------------------------
// TopStocksList
// ---------------------------------------------------------------------------
function TopStocksList({ stocks, title, compact = false, loading = false }) {
  if (loading) {
    return (
      <Card variant="glass" className={compact ? 'mb-3' : 'mb-4'}>
        <CardHeader title={title} />
        <div className="animate-pulse space-y-2">
          {[1,2,3,4,5].map(i => <div key={i} className="h-6 bg-dark-700 rounded" />)}
        </div>
      </Card>
    )
  }

  if (!stocks || stocks.length === 0) return null

  return (
    <Card variant="glass" className={compact ? 'mb-3' : 'mb-4'}>
      <CardHeader
        title={title}
        action={<Link to="/screener" className="text-primary-500 text-xs">See All</Link>}
      />

      <div className={compact ? 'space-y-1' : 'space-y-3'}>
        {stocks.map((stock, index) =>
          compact ? (
            <RankedStockRow
              key={stock.ticker}
              stock={stock}
              index={index}
              subtitle={null}
            >
              <div className="text-[10px] text-dark-400 font-data">
                ${stock.current_price?.toFixed(0) || '-'}
              </div>
            </RankedStockRow>
          ) : (
            <Link
              key={stock.ticker}
              to={`/stock/${stock.ticker}`}
              className="flex justify-between items-center py-2 border-b border-dark-700/30 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors"
            >
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 text-xs rounded-full bg-dark-600 flex items-center justify-center font-bold font-data">
                  {index + 1}
                </div>
                <div>
                  <div className="flex items-center gap-1">
                    <span className="font-medium">{stock.ticker}</span>
                    <ScoreTrend change={stock.score_change} />
                    <WeeklyTrend trend={stock.score_trend} change={stock.trend_change} />
                    {stock.data_quality === 'low' && (
                      <span className="text-yellow-500 text-[10px]" title="Limited analyst data">{'\u26A0'}</span>
                    )}
                  </div>
                  <div className="text-dark-400 text-xs truncate max-w-[150px]">{stock.name}</div>
                </div>
              </div>
              <div className="text-right">
                <ScoreBadge score={stock.canslim_score} size="sm" />
                <div className="text-xs mt-1">
                  <div className="text-dark-300 font-data">
                    {stock.current_price != null ? `$${stock.current_price.toFixed(2)}` : '-'}
                  </div>
                  <PnlText
                    value={stock.projected_growth}
                    className="text-xs"
                    prefix=""
                  />
                  {stock.projected_growth != null && (
                    <span className="text-dark-500 text-[10px]"> proj</span>
                  )}
                </div>
              </div>
            </Link>
          )
        )}
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// TopGrowthStocks
// ---------------------------------------------------------------------------
function TopGrowthStocks({ stocks, loading }) {
  if (loading) {
    return (
      <Card variant="glass" className="mb-3">
        <CardHeader title={<><span className="text-green-400 mr-1">{'\u2197'}</span> Top Growth Stocks</>} />
        <div className="animate-pulse space-y-2">
          {[1,2,3,4,5].map(i => <div key={i} className="h-6 bg-dark-700 rounded" />)}
        </div>
      </Card>
    )
  }

  if (!stocks || stocks.length === 0) {
    return (
      <Card variant="glass" className="mb-3">
        <CardHeader title={<><span className="text-green-400 mr-1">{'\u2197'}</span> Top Growth Stocks</>} />
        <div className="text-dark-400 text-xs py-4 text-center">
          No growth stocks found yet. Run a scan to analyze stocks.
        </div>
      </Card>
    )
  }

  return (
    <Card variant="glass" className="mb-3">
      <CardHeader
        title={<><span className="text-green-400 mr-1">{'\u2197'}</span> Top Growth Stocks</>}
        action={<TagBadge>Growth Mode</TagBadge>}
      />

      <div className="space-y-1">
        {stocks.map((stock, index) => (
          <RankedStockRow
            key={stock.ticker}
            stock={stock}
            index={index}
            rankColor="bg-green-500/20 text-green-400"
            scoreValue={stock.growth_mode_score}
            scoreClass="bg-green-500/20 text-green-400"
            subtitle={stock.revenue_growth_pct != null ? `Rev +${stock.revenue_growth_pct.toFixed(0)}%` : stock.sector?.slice(0,12) || '-'}
            badges={
              stock.is_breaking_out
                ? <TagBadge color="amber">BO</TagBadge>
                : null
            }
          >
            <div className="text-[10px] text-dark-400 font-data">
              ${stock.current_price?.toFixed(0) || '-'}
            </div>
          </RankedStockRow>
        ))}
      </div>

      <div className="text-[10px] text-dark-500 mt-2 pt-2 border-t border-dark-700">
        Growth Mode: Revenue-focused scoring for high-growth stocks
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// BreakingOutStocks
// ---------------------------------------------------------------------------
function BreakingOutStocks({ stocks, loading }) {
  if (loading) {
    return (
      <Card variant="glass" className="mb-3">
        <CardHeader title={<><span className="text-yellow-400 mr-1">{'\u26A1'}</span> Breaking Out</>} />
        <div className="animate-pulse space-y-2">
          {[1,2,3,4,5].map(i => <div key={i} className="h-6 bg-dark-700 rounded" />)}
        </div>
      </Card>
    )
  }

  if (!stocks || stocks.length === 0) {
    return (
      <Card variant="glass" className="mb-3">
        <CardHeader title={<><span className="text-yellow-400 mr-1">{'\u26A1'}</span> Breaking Out</>} />
        <div className="text-dark-400 text-xs py-4 text-center">
          No breakouts detected. Stocks break out when price clears a base pattern with strong volume.
        </div>
      </Card>
    )
  }

  return (
    <Card variant="glass" className="mb-3">
      <CardHeader
        title={<><span className="text-yellow-400 mr-1">{'\u26A1'}</span> Breaking Out</>}
        action={
          <div className="flex items-center gap-2">
            <TagBadge color="amber">Buy Zone</TagBadge>
            <Link to="/breakouts" className="text-primary-500 text-xs">See All</Link>
          </div>
        }
      />

      <div className="space-y-1">
        {stocks.map((stock, index) => (
          <RankedStockRow
            key={stock.ticker}
            stock={stock}
            index={index}
            rankColor="bg-yellow-500/20 text-yellow-400"
            subtitle={stock.breakout_volume_ratio ? `Vol ${stock.breakout_volume_ratio.toFixed(1)}x` : stock.sector?.slice(0,12) || '-'}
            badges={
              stock.base_type && stock.base_type !== 'none'
                ? <TagBadge color="cyan">{stock.base_type}</TagBadge>
                : null
            }
          >
            <div className="text-[10px] text-dark-400 font-data">
              ${stock.current_price?.toFixed(0) || '-'}
            </div>
          </RankedStockRow>
        ))}
      </div>

      <div className="text-[10px] text-dark-500 mt-2 pt-2 border-t border-dark-700">
        Stocks clearing base patterns with 40%+ above-average volume
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// CoiledSpringAlerts
// ---------------------------------------------------------------------------
function CoiledSpringAlerts({ candidates, loading }) {
  if (loading) {
    return (
      <Card variant="accent" accent="purple" className="mb-3 bg-purple-500/5">
        <CardHeader title={<><span className="text-purple-400 mr-1">{'\uD83C\uDF00'}</span> Coiled Spring Alerts</>} />
        <div className="animate-pulse space-y-2">
          {[1,2,3].map(i => <div key={i} className="h-6 bg-dark-700 rounded" />)}
        </div>
      </Card>
    )
  }

  if (!candidates || candidates.length === 0) {
    return null  // Don't show empty section - only show when there are candidates
  }

  return (
    <Card variant="accent" accent="purple" className="mb-3 bg-purple-500/5">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <span className="text-purple-400">{'\uD83C\uDF00'}</span> Coiled Spring Alerts
            <TagBadge color="purple">{candidates.length} found</TagBadge>
          </span>
        }
        action={<TagBadge>Pre-Earnings Catalyst</TagBadge>}
      />

      <div className="space-y-1">
        {candidates.slice(0, 5).map((stock, index) => (
          <RankedStockRow
            key={stock.ticker}
            stock={stock}
            index={index}
            rankColor="bg-purple-500/20 text-purple-400"
            subtitle={
              <span className="flex gap-2">
                <span>{stock.earnings_beat_streak} beats</span>
                <span>{'\u2022'}</span>
                <span>{stock.days_to_earnings}d to earn</span>
                <span>{'\u2022'}</span>
                <span>{stock.institutional_holders_pct?.toFixed(1)}% inst</span>
              </span>
            }
            badges={
              stock.base_type && stock.base_type !== 'none'
                ? <TagBadge color="cyan">{stock.weeks_in_base}w {stock.base_type}</TagBadge>
                : null
            }
          >
            <div className="text-[10px] text-purple-400 font-medium font-data">
              +{stock.cs_bonus} CS
            </div>
          </RankedStockRow>
        ))}
      </div>

      <div className="text-[10px] text-dark-500 mt-2 pt-2 border-t border-dark-700">
        Stocks with long bases, earnings beat streaks, approaching earnings - high conviction setups
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// CoiledSpringStats
// ---------------------------------------------------------------------------
function CoiledSpringStats({ stats, loading }) {
  if (loading) {
    return (
      <Card variant="accent" accent="purple" className="mb-3 bg-purple-500/5">
        <CardHeader title={<><span className="text-purple-400 mr-1">{'\uD83D\uDCCA'}</span> CS Performance</>} />
        <div className="animate-pulse h-16 bg-dark-700 rounded" />
      </Card>
    )
  }

  if (!stats) return null

  const { cumulative_stats } = stats
  if (!cumulative_stats || cumulative_stats.with_outcome === 0) {
    return null  // Don't show if no outcomes tracked yet
  }

  const winRateColor = cumulative_stats.overall_win_rate >= 60 ? 'text-green-400' :
                       cumulative_stats.overall_win_rate >= 40 ? 'text-yellow-400' : 'text-red-400'

  // Get top performing base type
  const baseTypes = cumulative_stats.by_base_type || {}
  const sortedBases = Object.entries(baseTypes)
    .filter(([_, data]) => data.with_outcome >= 3)
    .sort((a, b) => b[1].win_rate - a[1].win_rate)

  return (
    <Card variant="accent" accent="purple" className="mb-3 bg-purple-500/5">
      <CardHeader
        title={<><span className="text-purple-400 mr-1">{'\uD83D\uDCCA'}</span> CS Performance</>}
        action={<TagBadge>{cumulative_stats.total_alerts_all_time} alerts tracked</TagBadge>}
      />

      <StatGrid
        columns={4}
        stats={[
          { label: 'Win Rate', value: `${cumulative_stats.overall_win_rate}%`, color: winRateColor },
          { label: 'Wins', value: cumulative_stats.wins, color: 'text-green-400' },
          { label: 'Losses', value: cumulative_stats.losses, color: 'text-red-400' },
          { label: 'Big Wins', value: cumulative_stats.big_wins, color: 'text-blue-400' },
        ]}
        className="mb-3"
      />

      {/* Best Pattern */}
      {sortedBases.length > 0 && (
        <div className="text-[10px] text-dark-500 pt-2 border-t border-dark-700">
          <span className="font-medium text-purple-400">Best pattern: </span>
          {sortedBases[0][0].replace('_', ' ')} ({sortedBases[0][1].win_rate}% win rate, {sortedBases[0][1].with_outcome} samples)
        </div>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// QuickStats
// ---------------------------------------------------------------------------
function QuickStats({ stats }) {
  if (!stats) return null

  return (
    <Card variant="glass" className="mb-4">
      <CardHeader title="Quick Stats" />
      <StatGrid
        columns={2}
        stats={[
          { label: 'Stocks Analyzed', value: stats.total_stocks || 0 },
          { label: 'Score 80+', value: stats.high_score_count || 0, color: 'text-green-400' },
          { label: 'Portfolio Positions', value: stats.portfolio_count || 0 },
          { label: 'Watchlist', value: stats.watchlist_count || 0 },
        ]}
      />
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ScanControls
// ---------------------------------------------------------------------------
function ScanControls({ onScan, scanning, scanSource, setScanSource }) {
  const [expanded, setExpanded] = useState(false)

  const sourceOptions = [
    { value: 'all', label: 'All Stocks', count: '~950+' },
    { value: 'sp500', label: 'S&P 500', count: '~500' },
    { value: 'russell', label: 'Russell 2000', count: '~750' },
    { value: 'top50', label: 'Top 50', count: '50' },
  ]

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-dark-500 text-xs hover:text-dark-300 flex items-center gap-1 mb-2"
      >
        <span className="text-[10px]">{expanded ? '\u25BC' : '\u25B6'}</span>
        <span>Manual Scan</span>
      </button>

      {expanded && (
        <div className="flex gap-2">
          <select
            value={scanSource}
            onChange={(e) => setScanSource(e.target.value)}
            disabled={scanning}
            className="flex-1 bg-dark-700 border border-dark-600 rounded-lg px-2 py-1.5 text-xs"
          >
            {sourceOptions.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label} ({opt.count})
              </option>
            ))}
          </select>
          <button
            onClick={onScan}
            disabled={scanning}
            className="btn-primary text-xs px-3 py-1.5"
          >
            {scanning ? '...' : 'Run'}
          </button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ContinuousScanner
// ---------------------------------------------------------------------------
function ContinuousScanner({ scannerStatus, onToggle, onConfigChange }) {
  const [source, setSource] = useState(scannerStatus?.source || 'all')
  const [interval, setInterval] = useState(scannerStatus?.interval_minutes || 90)
  const [updating, setUpdating] = useState(false)

  const sourceOptions = [
    { value: 'all', label: 'All Stocks', interval: 90 },
    { value: 'sp500', label: 'S&P 500', interval: 30 },
    { value: 'russell', label: 'Russell 2000', interval: 45 },
    { value: 'top50', label: 'Top 50', interval: 15 },
  ]

  const intervalOptions = [15, 30, 45, 60, 90, 120, 180]

  const handleToggle = async () => {
    setUpdating(true)
    try {
      await onToggle(!scannerStatus?.enabled, source, interval)
    } finally {
      setUpdating(false)
    }
  }

  const handleSourceChange = (newSource) => {
    setSource(newSource)
    const suggested = sourceOptions.find(o => o.value === newSource)?.interval || 15
    setInterval(suggested)
    if (scannerStatus?.enabled) {
      onConfigChange(newSource, suggested)
    }
  }

  const handleIntervalChange = (newInterval) => {
    setInterval(newInterval)
    if (scannerStatus?.enabled) {
      onConfigChange(source, newInterval)
    }
  }

  const formatLastScan = (isoString) => {
    if (!isoString) return 'Never'
    return formatTime(isoString)
  }

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold">Auto-Scan</div>
        <button
          onClick={handleToggle}
          disabled={updating}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            scannerStatus?.enabled ? 'bg-green-500' : 'bg-dark-600'
          }`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
            scannerStatus?.enabled ? 'translate-x-7' : 'translate-x-1'
          }`} />
        </button>
      </div>

      {scannerStatus?.enabled && (
        <div className="mb-3 p-2 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <span className="animate-pulse">{'\u25CF'}</span>
            <span>
              {scannerStatus?.is_scanning ? (
                scannerStatus?.phase === 'saving' ? 'Saving to database...' :
                scannerStatus?.phase === 'ai_trading' ? 'Running AI trader...' :
                'Scanning...'
              ) : 'Active'}
              {scannerStatus?.next_run && !scannerStatus?.is_scanning && (
                <span className="text-dark-400 ml-2">
                  Next: {formatTime(scannerStatus.next_run)}
                </span>
              )}
            </span>
          </div>

          {/* Progress bar when scanning */}
          {scannerStatus?.is_scanning && scannerStatus?.total_stocks > 0 && (
            <div className="mt-2">
              <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 transition-all"
                  style={{ width: `${(scannerStatus.stocks_scanned / scannerStatus.total_stocks * 100)}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-dark-400 mt-1 font-data">
                <span>{scannerStatus.stocks_scanned} / {scannerStatus.total_stocks} stocks</span>
                <span>
                  {scannerStatus.last_scan_start && (() => {
                    const elapsed = Math.floor((Date.now() - new Date(scannerStatus.last_scan_start).getTime()) / 1000);
                    const mins = Math.floor(elapsed / 60);
                    const secs = elapsed % 60;
                    return `${mins}:${secs.toString().padStart(2, '0')} - `;
                  })()}
                  {Math.round(scannerStatus.stocks_scanned / scannerStatus.total_stocks * 100)}%
                </span>
              </div>
              {/* Phase indicator */}
              {scannerStatus.phase && (
                <div className="mt-2 text-xs text-green-300 flex items-center gap-2">
                  <span className="font-medium">
                    {scannerStatus.phase === 'scanning' && 'Phase 1: Scanning'}
                    {scannerStatus.phase === 'saving' && 'Phase 2: Saving'}
                    {scannerStatus.phase === 'ai_trading' && 'Phase 3: AI Trading'}
                  </span>
                  {scannerStatus.phase_detail && (
                    <span className="text-dark-400">{'\u2014'} {scannerStatus.phase_detail}</span>
                  )}
                </div>
              )}
            </div>
          )}

          {scannerStatus?.last_scan_end && !scannerStatus?.is_scanning && (
            <div className="text-xs text-dark-400 mt-1">
              Last scan: {formatLastScan(scannerStatus.last_scan_end)} ({scannerStatus.stocks_scanned} stocks)
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-dark-400 text-xs mb-1 block">Universe</label>
          <select
            value={source}
            onChange={(e) => handleSourceChange(e.target.value)}
            disabled={updating}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm"
          >
            {sourceOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-dark-400 text-xs mb-1 block">Interval</label>
          <select
            value={interval}
            onChange={(e) => handleIntervalChange(Number(e.target.value))}
            disabled={updating}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm"
          >
            {intervalOptions.map(mins => (
              <option key={mins} value={mins}>{mins} min</option>
            ))}
          </select>
        </div>
      </div>

      <div className="text-xs text-dark-500 mt-2">
        Continuously scans stocks to keep data fresh. Stays within API limits.
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ScanProgress
// ---------------------------------------------------------------------------
function ScanProgress({ scanJob, scanStartTime }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (!scanStartTime) return

    const updateElapsed = () => {
      setElapsed(Math.floor((Date.now() - scanStartTime) / 1000))
    }

    updateElapsed()
    const interval = setInterval(updateElapsed, 1000)
    return () => clearInterval(interval)
  }, [scanStartTime])

  const formatElapsed = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const pct = scanJob.tickers_total > 0
    ? (scanJob.tickers_processed / scanJob.tickers_total * 100)
    : 0

  const rate = elapsed > 0 ? (scanJob.tickers_processed / elapsed).toFixed(1) : 0
  const remaining = rate > 0
    ? Math.ceil((scanJob.tickers_total - scanJob.tickers_processed) / rate)
    : 0

  return (
    <Card variant="glass" className="mb-4">
      <div className="flex justify-between items-center mb-2">
        <div className="text-sm text-dark-400">Scan Progress</div>
        <div className="text-sm font-data">{formatElapsed(elapsed)}</div>
      </div>
      <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-dark-400 mt-1 font-data">
        <span>{scanJob.tickers_processed} / {scanJob.tickers_total} stocks</span>
        <span>{rate}/s {remaining > 0 && `\u00B7 ~${formatElapsed(remaining)} left`}</span>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Dashboard (main export)
// ---------------------------------------------------------------------------
export default function Dashboard() {
  const [loading, setLoading] = useState(true)
  const [data, setData] = useState(null)
  const [scanning, setScanning] = useState(false)
  const [scanJob, setScanJob] = useState(null)
  const [scanSource, setScanSource] = useState('sp500')
  const [scanStartTime, setScanStartTime] = useState(null)
  const [scannerStatus, setScannerStatus] = useState(null)
  const [growthStocks, setGrowthStocks] = useState(null)
  const [growthLoading, setGrowthLoading] = useState(true)
  const [breakoutStocks, setBreakoutStocks] = useState(null)
  const [breakoutLoading, setBreakoutLoading] = useState(true)
  const [csAlerts, setCsAlerts] = useState(null)
  const [csLoading, setCsLoading] = useState(true)
  const [csStats, setCsStats] = useState(null)
  const [csStatsLoading, setCsStatsLoading] = useState(true)

  const fetchData = async () => {
    try {
      setLoading(true)
      setGrowthLoading(true)
      setBreakoutLoading(true)
      setCsLoading(true)
      setCsStatsLoading(true)
      const [dashboard, scanner, growth, breakouts, coiledSpring, coiledSpringStats] = await Promise.all([
        api.getDashboard(),
        api.getScannerStatus().catch(() => null),
        api.getTopGrowthStocks(10).catch(() => ({ stocks: [] })),
        api.getBreakingOutStocks(10).catch(() => ({ stocks: [] })),
        api.getCoiledSpringCandidates().catch(() => ({ candidates: [] })),
        api.getCoiledSpringHistory(1, 100).catch(() => null)
      ])
      setData(dashboard)
      setScannerStatus(scanner)
      setGrowthStocks(growth?.stocks || [])
      setBreakoutStocks(breakouts?.stocks || [])
      setCsAlerts(coiledSpring?.candidates || [])
      setCsStats(coiledSpringStats)
    } catch (err) {
      console.error('Failed to fetch dashboard:', err)
    } finally {
      setLoading(false)
      setGrowthLoading(false)
      setBreakoutLoading(false)
      setCsLoading(false)
      setCsStatsLoading(false)
    }
  }

  // Poll scanner status when enabled (faster when scanning)
  useEffect(() => {
    if (!scannerStatus?.enabled) return

    const pollInterval = scannerStatus?.is_scanning ? 3000 : 10000 // 3s when scanning, 10s otherwise

    const interval = setInterval(async () => {
      try {
        const status = await api.getScannerStatus()
        setScannerStatus(status)
      } catch (err) {
        console.error('Failed to get scanner status:', err)
      }
    }, pollInterval)

    return () => clearInterval(interval)
  }, [scannerStatus?.enabled, scannerStatus?.is_scanning])

  useEffect(() => {
    fetchData()
  }, [])

  // Poll for scan job progress
  useEffect(() => {
    if (!scanJob || scanJob.status === 'completed' || scanJob.status === 'failed') {
      return
    }

    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(scanJob.job_id)
        setScanJob(status)

        if (status.status === 'completed' || status.status === 'failed') {
          setScanning(false)
          fetchData() // Refresh data after scan
        }
      } catch (err) {
        console.error('Failed to get job status:', err)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [scanJob])

  const handleScan = async () => {
    try {
      setScanning(true)
      setScanStartTime(Date.now())
      const result = await api.startScan(null, scanSource)
      setScanJob(result)
    } catch (err) {
      console.error('Failed to start scan:', err)
      setScanning(false)
      setScanStartTime(null)
    }
  }

  const handleScannerToggle = async (enabled, source, interval) => {
    try {
      let status
      if (enabled) {
        status = await api.startScanner(source, interval)
      } else {
        status = await api.stopScanner()
      }
      setScannerStatus(status)
    } catch (err) {
      console.error('Failed to toggle scanner:', err)
    }
  }

  const handleScannerConfigChange = async (source, interval) => {
    try {
      const status = await api.updateScannerConfig(source, interval)
      setScannerStatus(status)
    } catch (err) {
      console.error('Failed to update scanner config:', err)
    }
  }

  const handleMarketRefresh = async () => {
    try {
      const result = await api.refreshMarket()
      if (result && !result.error) {
        setData(prev => ({
          ...prev,
          market: {
            ...prev?.market,
            spy_price: result.indexes?.SPY?.price || result.spy_price,
            spy_50_ma: result.indexes?.SPY?.ma_50 || result.spy_50_ma,
            spy_200_ma: result.indexes?.SPY?.ma_200 || result.spy_200_ma,
            indexes: result.indexes,
            weighted_signal: result.weighted_signal,
            trend: result.market_trend,
            score: result.market_score
          }
        }))
      }
    } catch (err) {
      console.error('Failed to refresh market:', err)
    }
  }

  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-48 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Stock Analyzer"
        subtitle="CANSLIM"
      />

      <MarketStatus market={data?.market} onRefresh={handleMarketRefresh} />

      <SectionLabel>Catalysts</SectionLabel>

      <CoiledSpringAlerts candidates={csAlerts} loading={csLoading} />

      <CoiledSpringStats stats={csStats} loading={csStatsLoading} />

      <SectionLabel>Overview</SectionLabel>

      <QuickStats stats={data?.stats} />

      {/* Stock Lists Grid - 2x2 on larger screens */}
      <SectionLabel>Stock Lists</SectionLabel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        <TopStocksList stocks={data?.top_stocks?.slice(0, 10)} title="Top CANSLIM" compact loading={loading} />
        <TopGrowthStocks stocks={growthStocks} loading={growthLoading} />
        <TopStocksList stocks={data?.top_stocks_under_25?.slice(0, 10)} title="Top Under $25" compact loading={loading} />
        <BreakingOutStocks stocks={breakoutStocks} loading={breakoutLoading} />
      </div>

      {scanJob && scanJob.status === 'running' && (
        <ScanProgress scanJob={scanJob} scanStartTime={scanStartTime} />
      )}

      <SectionLabel>Scanner</SectionLabel>

      <ScanControls
        onScan={handleScan}
        scanning={scanning}
        scanSource={scanSource}
        setScanSource={setScanSource}
      />

      <ContinuousScanner
        scannerStatus={scannerStatus}
        onToggle={handleScannerToggle}
        onConfigChange={handleScannerConfigChange}
      />

      <div className="h-4" />
    </div>
  )
}
