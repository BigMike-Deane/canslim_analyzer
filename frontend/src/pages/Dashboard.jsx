import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent, formatMarketCap } from '../api'

function MarketStatus({ market }) {
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

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold">Market Direction</div>
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
          <span className={`px-2 py-1 rounded text-sm font-medium ${getScoreClass(market.score)}`}>
            {formatScore(market.score)}
          </span>
        </div>
      </div>
    </div>
  )
}

function TopStocksList({ stocks, title }) {
  if (!stocks || stocks.length === 0) return null

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold">{title}</div>
        <Link to="/screener" className="text-primary-500 text-sm">See All</Link>
      </div>

      <div className="space-y-3">
        {stocks.map((stock, index) => (
          <Link
            key={stock.ticker}
            to={`/stock/${stock.ticker}`}
            className="flex justify-between items-center py-2 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-2 px-2 rounded transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 rounded-full bg-dark-600 flex items-center justify-center text-xs font-bold">
                {index + 1}
              </div>
              <div>
                <div className="font-medium">{stock.ticker}</div>
                <div className="text-dark-400 text-xs truncate max-w-[150px]">{stock.name}</div>
              </div>
            </div>
            <div className="text-right">
              <div className={`px-2 py-1 rounded text-sm font-medium ${getScoreClass(stock.canslim_score)}`}>
                {formatScore(stock.canslim_score)}
              </div>
              {stock.projected_growth != null && (
                <div className="text-xs text-dark-400 mt-1">
                  +{stock.projected_growth.toFixed(0)}% proj
                </div>
              )}
            </div>
          </Link>
        ))}
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
          <div className="text-dark-400 text-xs">Score 70+</div>
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
  const sourceOptions = [
    { value: 'sp500', label: 'S&P 500', count: '~500' },
    { value: 'top50', label: 'Top 50', count: '50' },
    { value: 'russell', label: 'Russell 2000', count: '~750' },
    { value: 'all', label: 'All Stocks', count: '~950+' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <select
          value={scanSource}
          onChange={(e) => setScanSource(e.target.value)}
          disabled={scanning}
          className="flex-1 bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm"
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
          className="flex-1 btn-primary flex items-center justify-center gap-2"
        >
          {scanning ? (
            <>
              <span className="animate-spin">⟳</span>
              <span>Scanning...</span>
            </>
          ) : (
            <>
              <span>Run Scan</span>
            </>
          )}
        </button>
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

  const fetchData = async () => {
    try {
      setLoading(true)
      const dashboard = await api.getDashboard()
      setData(dashboard)
    } catch (err) {
      console.error('Failed to fetch dashboard:', err)
    } finally {
      setLoading(false)
    }
  }

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
      const result = await api.startScan(null, scanSource)
      setScanJob(result)
    } catch (err) {
      console.error('Failed to start scan:', err)
      setScanning(false)
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

      <MarketStatus market={data?.market} />

      <QuickStats stats={data?.stats} />

      <TopStocksList stocks={data?.top_stocks} title="Top Rated Stocks" />

      {scanJob && scanJob.status === 'running' && (
        <div className="card mb-4">
          <div className="text-sm text-dark-400 mb-2">Scan Progress</div>
          <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 transition-all"
              style={{
                width: `${scanJob.tickers_total > 0 ? (scanJob.tickers_processed / scanJob.tickers_total * 100) : 0}%`
              }}
            />
          </div>
          <div className="text-xs text-dark-400 mt-1">
            {scanJob.tickers_processed} / {scanJob.tickers_total} stocks
          </div>
        </div>
      )}

      <ScanControls
        onScan={handleScan}
        scanning={scanning}
        scanSource={scanSource}
        setScanSource={setScanSource}
      />

      <div className="h-4" />
    </div>
  )
}
