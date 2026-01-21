import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent, formatMarketCap } from '../api'

function MarketStatus({ market, onRefresh }) {
  const [refreshing, setRefreshing] = useState(false)

  if (!market) return null

  const trendColors = {
    bullish: 'text-green-400',
    neutral: 'text-yellow-400',
    bearish: 'text-red-400'
  }

  const trendIcons = {
    bullish: '▲',
    neutral: '►',
    bearish: '▼'
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await onRefresh()
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold">Market Direction</span>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="text-dark-400 hover:text-white transition-colors p-1"
            title="Refresh SPY data"
          >
            <span className={refreshing ? 'animate-spin inline-block' : ''}>⟳</span>
          </button>
        </div>
        <div className={`text-sm font-medium ${trendColors[market.trend] || 'text-dark-400'}`}>
          {trendIcons[market.trend]} {market.trend?.toUpperCase()}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-dark-400 text-xs mb-1">SPY</div>
          <div className="font-semibold">{formatCurrency(market.spy_price)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs mb-1">50 MA</div>
          <div className={`font-semibold ${market.spy_price > market.spy_50_ma ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(market.spy_50_ma)}
          </div>
        </div>
        <div>
          <div className="text-dark-400 text-xs mb-1">200 MA</div>
          <div className={`font-semibold ${market.spy_price > market.spy_200_ma ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(market.spy_200_ma)}
          </div>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-dark-700">
        <div className="flex items-center justify-between">
          <span className="text-dark-400 text-sm">M Score</span>
          <span className={`px-2 py-1 rounded text-sm font-medium ${getScoreClass((market.score / 15) * 100)}`}>
            {market.score != null ? `${market.score.toFixed(1)}/15` : '-'}
          </span>
        </div>
      </div>
    </div>
  )
}

function ScoreTrend({ change }) {
  if (change == null) return null

  if (change === 0) {
    return <span className="text-xs text-dark-400">-</span>
  }

  const isUp = change > 0
  return (
    <span className={`text-xs font-medium ${isUp ? 'text-green-400' : 'text-red-400'}`}>
      {isUp ? '+' : ''}{change.toFixed(1)}
    </span>
  )
}

function WeeklyTrend({ trend, change }) {
  if (!trend) return null

  const config = {
    improving: { icon: '↗', color: 'text-green-400', bg: 'bg-green-500/10' },
    stable: { icon: '→', color: 'text-dark-400', bg: 'bg-dark-600' },
    deteriorating: { icon: '↘', color: 'text-red-400', bg: 'bg-red-500/10' }
  }

  const { icon, color, bg } = config[trend] || config.stable

  return (
    <span
      className={`text-[10px] px-1.5 py-0.5 rounded ${bg} ${color}`}
      title={`7-day trend: ${change != null ? (change > 0 ? '+' : '') + change.toFixed(1) : ''} pts`}
    >
      {icon} {trend === 'improving' ? 'Up' : trend === 'deteriorating' ? 'Down' : ''}
    </span>
  )
}

function TopStocksList({ stocks, title, compact = false }) {
  if (!stocks || stocks.length === 0) return null

  return (
    <div className={compact ? "card mb-3" : "card mb-4"}>
      <div className="flex justify-between items-center mb-2">
        <div className={compact ? "font-semibold text-sm" : "font-semibold"}>{title}</div>
        <Link to="/screener" className="text-primary-500 text-xs">See All</Link>
      </div>

      <div className={compact ? "space-y-1" : "space-y-3"}>
        {stocks.map((stock, index) => (
          <Link
            key={stock.ticker}
            to={`/stock/${stock.ticker}`}
            className={`flex justify-between items-center ${compact ? 'py-1' : 'py-2'} border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors`}
          >
            <div className="flex items-center gap-2">
              <div className={`${compact ? 'w-5 h-5 text-[10px]' : 'w-6 h-6 text-xs'} rounded-full bg-dark-600 flex items-center justify-center font-bold`}>
                {index + 1}
              </div>
              <div>
                <div className="flex items-center gap-1">
                  <span className={compact ? "font-medium text-sm" : "font-medium"}>{stock.ticker}</span>
                  {!compact && <ScoreTrend change={stock.score_change} />}
                  {!compact && <WeeklyTrend trend={stock.score_trend} change={stock.trend_change} />}
                  {stock.data_quality === 'low' && (
                    <span className="text-yellow-500 text-[10px]" title="Limited analyst data">⚠</span>
                  )}
                </div>
                {!compact && <div className="text-dark-400 text-xs truncate max-w-[150px]">{stock.name}</div>}
              </div>
            </div>
            <div className="text-right">
              <div className={`inline-block px-1.5 py-0.5 rounded ${compact ? 'text-xs' : 'text-sm'} font-medium ${getScoreClass(stock.canslim_score)}`}>
                {formatScore(stock.canslim_score)}
              </div>
              {!compact && (
                <div className="text-xs mt-1">
                  <div className="text-dark-300">
                    {stock.current_price != null ? `$${stock.current_price.toFixed(2)}` : '-'}
                  </div>
                  <div className={stock.projected_growth != null && stock.projected_growth >= 0 ? 'text-green-400' : stock.projected_growth != null ? 'text-red-400' : 'text-dark-500'}>
                    {stock.projected_growth != null ? `${stock.projected_growth >= 0 ? '+' : ''}${stock.projected_growth.toFixed(0)}% proj` : '-'}
                  </div>
                </div>
              )}
              {compact && (
                <div className="text-[10px] text-dark-400">
                  ${stock.current_price?.toFixed(0) || '-'}
                </div>
              )}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}

function TopGrowthStocks({ stocks, loading }) {
  if (loading) {
    return (
      <div className="card mb-3">
        <div className="font-semibold text-sm mb-2 flex items-center gap-2">
          <span className="text-green-400">↗</span> Top Growth Stocks
        </div>
        <div className="animate-pulse space-y-2">
          {[1,2,3,4,5].map(i => <div key={i} className="h-6 bg-dark-700 rounded" />)}
        </div>
      </div>
    )
  }

  if (!stocks || stocks.length === 0) {
    return (
      <div className="card mb-3">
        <div className="font-semibold text-sm mb-2 flex items-center gap-2">
          <span className="text-green-400">↗</span> Top Growth Stocks
        </div>
        <div className="text-dark-400 text-xs py-4 text-center">
          No growth stocks found yet. Run a scan to analyze stocks.
        </div>
      </div>
    )
  }

  return (
    <div className="card mb-3">
      <div className="flex justify-between items-center mb-2">
        <div className="font-semibold text-sm flex items-center gap-2">
          <span className="text-green-400">↗</span> Top Growth Stocks
        </div>
        <span className="text-[10px] text-dark-400 bg-dark-700 px-1.5 py-0.5 rounded">Growth Mode</span>
      </div>

      <div className="space-y-1">
        {stocks.map((stock, index) => (
          <Link
            key={stock.ticker}
            to={`/stock/${stock.ticker}`}
            className="flex justify-between items-center py-1.5 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors"
          >
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 text-[10px] rounded-full bg-green-500/20 text-green-400 flex items-center justify-center font-bold">
                {index + 1}
              </div>
              <div>
                <div className="flex items-center gap-1">
                  <span className="font-medium text-sm">{stock.ticker}</span>
                  {stock.is_breaking_out && (
                    <span className="text-[9px] bg-yellow-500/20 text-yellow-400 px-1 rounded" title="Breaking out!">BO</span>
                  )}
                </div>
                <div className="text-dark-500 text-[10px]">
                  {stock.revenue_growth_pct != null ? `Rev +${stock.revenue_growth_pct.toFixed(0)}%` : stock.sector?.slice(0,12) || '-'}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="inline-block px-1.5 py-0.5 rounded text-xs font-medium bg-green-500/20 text-green-400">
                {formatScore(stock.growth_mode_score)}
              </div>
              <div className="text-[10px] text-dark-400">
                ${stock.current_price?.toFixed(0) || '-'}
              </div>
            </div>
          </Link>
        ))}
      </div>

      <div className="text-[10px] text-dark-500 mt-2 pt-2 border-t border-dark-700">
        Growth Mode: Revenue-focused scoring for high-growth stocks
      </div>
    </div>
  )
}

function QuickStats({ stats }) {
  if (!stats) return null

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">Quick Stats</div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-dark-400 text-xs">Stocks Analyzed</div>
          <div className="font-semibold text-lg">{stats.total_stocks || 0}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Score 80+</div>
          <div className="font-semibold text-lg text-green-400">{stats.high_score_count || 0}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Portfolio Positions</div>
          <div className="font-semibold text-lg">{stats.portfolio_count || 0}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Watchlist</div>
          <div className="font-semibold text-lg">{stats.watchlist_count || 0}</div>
        </div>
      </div>
    </div>
  )
}

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
        <span className="text-[10px]">{expanded ? '▼' : '▶'}</span>
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
    // Auto-suggest interval based on source
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
    const date = new Date(isoString)
    return date.toLocaleTimeString('en-US', { timeZone: 'America/Chicago' }) + ' CST'
  }

  return (
    <div className="card mb-4">
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
            <span className="animate-pulse">●</span>
            <span>
              {scannerStatus?.is_scanning ? 'Scanning...' : 'Active'}
              {scannerStatus?.next_run && !scannerStatus?.is_scanning && (
                <span className="text-dark-400 ml-2">
                  Next: {new Date(scannerStatus.next_run).toLocaleTimeString('en-US', { timeZone: 'America/Chicago' })} CST
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
              <div className="flex justify-between text-xs text-dark-400 mt-1">
                <span>{scannerStatus.stocks_scanned} / {scannerStatus.total_stocks} stocks</span>
                <span>{Math.round(scannerStatus.stocks_scanned / scannerStatus.total_stocks * 100)}%</span>
              </div>
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
    </div>
  )
}

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

  const formatTime = (seconds) => {
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
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-2">
        <div className="text-sm text-dark-400">Scan Progress</div>
        <div className="text-sm font-mono">{formatTime(elapsed)}</div>
      </div>
      <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-primary-500 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-dark-400 mt-1">
        <span>{scanJob.tickers_processed} / {scanJob.tickers_total} stocks</span>
        <span>{rate}/s {remaining > 0 && `· ~${formatTime(remaining)} left`}</span>
      </div>
    </div>
  )
}

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

  const fetchData = async () => {
    try {
      setLoading(true)
      setGrowthLoading(true)
      const [dashboard, scanner, growth] = await Promise.all([
        api.getDashboard(),
        api.getScannerStatus().catch(() => null),
        api.getTopGrowthStocks(10).catch(() => ({ stocks: [] }))
      ])
      setData(dashboard)
      setScannerStatus(scanner)
      setGrowthStocks(growth?.stocks || [])
    } catch (err) {
      console.error('Failed to fetch dashboard:', err)
    } finally {
      setLoading(false)
      setGrowthLoading(false)
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
        // Update market data in state
        setData(prev => ({
          ...prev,
          market: {
            ...prev?.market,
            spy_price: result.spy_price,
            spy_50_ma: result.spy_50_ma,
            spy_200_ma: result.spy_200_ma,
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
      <div className="p-4">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-48 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <div>
          <div className="text-dark-400 text-sm">CANSLIM</div>
          <h1 className="text-xl font-bold">Stock Analyzer</h1>
        </div>
      </div>

      <MarketStatus market={data?.market} onRefresh={handleMarketRefresh} />

      <QuickStats stats={data?.stats} />

      {/* Stock Lists Grid - 2 columns on larger screens */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        <TopStocksList stocks={data?.top_stocks?.slice(0, 8)} title="Top CANSLIM" compact />
        <TopGrowthStocks stocks={growthStocks} loading={growthLoading} />
      </div>

      <TopStocksList stocks={data?.top_stocks_under_25?.slice(0, 8)} title="Top Under $25" compact />

      {scanJob && scanJob.status === 'running' && (
        <ScanProgress scanJob={scanJob} scanStartTime={scanStartTime} />
      )}

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
