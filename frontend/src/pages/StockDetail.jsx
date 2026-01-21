import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { api, formatScore, getScoreClass, getScoreLabel, formatCurrency, formatPercent, formatMarketCap } from '../api'

function ScoreGauge({ score, label }) {
  const radius = 40
  const circumference = 2 * Math.PI * radius
  const progress = (score || 0) / 100
  const strokeDashoffset = circumference * (1 - progress)

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={radius}
            stroke="#3a3a3c"
            strokeWidth="8"
            fill="none"
          />
          <circle
            cx="50"
            cy="50"
            r={radius}
            stroke={score >= 80 ? '#34c759' : score >= 65 ? '#30d158' : score >= 50 ? '#ffcc00' : score >= 35 ? '#ff9500' : '#ff3b30'}
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold">{formatScore(score)}</span>
        </div>
      </div>
      <span className="text-dark-400 text-sm mt-1">{label}</span>
    </div>
  )
}

function CANSLIMDetail({ stock }) {
  const scores = [
    { key: 'C', label: 'Current Earnings', value: stock.c_score, max: 15, desc: 'Quarterly earnings growth' },
    { key: 'A', label: 'Annual Earnings', value: stock.a_score, max: 15, desc: 'Annual earnings growth' },
    { key: 'N', label: 'New Highs', value: stock.n_score, max: 15, desc: 'New products, management, price highs' },
    { key: 'S', label: 'Supply/Demand', value: stock.s_score, max: 15, desc: 'Shares outstanding and volume' },
    { key: 'L', label: 'Leader/Laggard', value: stock.l_score, max: 15, desc: 'Relative strength vs market' },
    { key: 'I', label: 'Institutional', value: stock.i_score, max: 10, desc: 'Institutional sponsorship' },
    { key: 'M', label: 'Market Direction', value: stock.m_score, max: 15, desc: 'Overall market trend' },
  ]

  // Normalize score to 0-100 for color coding
  const normalizeScore = (value, max) => {
    if (value == null || max === 0) return 0
    return (value / max) * 100
  }

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">CANSLIM Breakdown</div>
      <div className="space-y-3">
        {scores.map(s => {
          const normalized = normalizeScore(s.value, s.max)
          return (
            <div key={s.key} className="flex items-center gap-3">
              <div className={`w-8 h-8 rounded-lg font-bold flex items-center justify-center ${getScoreClass(normalized)}`}>
                {s.key}
              </div>
              <div className="flex-1">
                <div className="flex justify-between">
                  <span className="font-medium text-sm">{s.label}</span>
                  <span className={`text-sm font-semibold ${getScoreClass(normalized)}`}>
                    {s.value != null ? `${s.value.toFixed(1)}/${s.max}` : '-'}
                  </span>
                </div>
                <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden mt-1">
                  <div
                    className={`h-full rounded-full transition-all ${
                      normalized >= 80 ? 'bg-green-500' :
                      normalized >= 65 ? 'bg-emerald-500' :
                      normalized >= 50 ? 'bg-yellow-500' :
                      normalized >= 35 ? 'bg-orange-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${normalized}%` }}
                  />
                </div>
                <div className="text-dark-400 text-xs mt-0.5">{s.desc}</div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function PriceInfo({ stock }) {
  const fromHigh = stock.week_52_high
    ? ((stock.current_price / stock.week_52_high - 1) * 100)
    : null

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">Price Information</div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-dark-400 text-xs">Current Price</div>
          <div className="text-xl font-bold">{formatCurrency(stock.current_price)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Market Cap</div>
          <div className="text-lg font-semibold">{formatMarketCap(stock.market_cap)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">52 Week High</div>
          <div className="font-semibold">{formatCurrency(stock.week_52_high)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">52 Week Low</div>
          <div className="font-semibold">{formatCurrency(stock.week_52_low)}</div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">From 52W High</div>
          <div className={`font-semibold ${fromHigh < 0 ? 'text-red-400' : 'text-green-400'}`}>
            {fromHigh != null ? formatPercent(fromHigh, true) : '-'}
          </div>
        </div>
        <div>
          <div className="text-dark-400 text-xs">Projected Growth</div>
          <div className="font-semibold text-green-400">
            {stock.projected_growth != null ? `+${stock.projected_growth.toFixed(0)}%` : '-'}
          </div>
        </div>
      </div>
    </div>
  )
}

function ScoreHistory({ history }) {
  if (!history || history.length < 2) return null

  return (
    <div className="card mb-4">
      <div className="font-semibold mb-3">Score History</div>
      <div className="h-40 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <Line
              type="monotone"
              dataKey="total_score"
              stroke="#007aff"
              strokeWidth={2}
              dot={false}
            />
            <Tooltip
              contentStyle={{ background: '#2c2c2e', border: 'none', borderRadius: '8px' }}
              labelStyle={{ color: '#8e8e93' }}
              formatter={(value) => [formatScore(value), 'Score']}
            />
            <XAxis dataKey="date" hide />
            <YAxis hide domain={[0, 100]} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function GrowthModeSection({ stock }) {
  // Only show if stock has growth mode data
  if (!stock.is_growth_stock && !stock.growth_mode_score) return null

  const details = stock.growth_mode_details || {}

  const scores = [
    { key: 'R', label: 'Revenue Growth', value: details.r, color: 'text-green-400' },
    { key: 'F', label: 'Funding Health', value: details.f, color: 'text-blue-400' },
  ]

  return (
    <div className="card mb-4 border border-green-500/30">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold flex items-center gap-2">
          <span className="text-green-400">↗</span> Growth Mode Score
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
            {stock.is_growth_stock ? 'Growth Stock' : 'Hybrid'}
          </span>
          <span className="text-xl font-bold text-green-400">
            {stock.growth_mode_score?.toFixed(1) || '-'}
          </span>
        </div>
      </div>

      <div className="text-dark-400 text-xs mb-3">
        Alternative scoring for pre-revenue and high-growth companies. Uses revenue momentum instead of earnings.
      </div>

      <div className="space-y-2">
        {scores.map(s => (
          <div key={s.key} className="flex items-center justify-between py-1 border-b border-dark-700 last:border-0">
            <div className="flex items-center gap-2">
              <span className={`font-bold ${s.color}`}>{s.key}</span>
              <span className="text-sm">{s.label}</span>
            </div>
            <span className="text-dark-400 text-sm">{s.value || '-'}</span>
          </div>
        ))}
      </div>

      {stock.revenue_growth_pct != null && (
        <div className="mt-3 pt-3 border-t border-dark-700 flex justify-between items-center">
          <span className="text-dark-400 text-sm">Revenue Growth (YoY)</span>
          <span className={`font-semibold ${stock.revenue_growth_pct >= 20 ? 'text-green-400' : stock.revenue_growth_pct >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
            {stock.revenue_growth_pct >= 0 ? '+' : ''}{stock.revenue_growth_pct.toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  )
}

function TechnicalAnalysis({ stock }) {
  const hasData = stock.base_type || stock.volume_ratio || stock.is_breaking_out

  return (
    <div className="card mb-4">
      <div className="flex justify-between items-center mb-3">
        <div className="font-semibold">Technical Analysis</div>
        {stock.is_breaking_out && (
          <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded flex items-center gap-1">
            <span>⚡</span> Breaking Out
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-dark-400 text-xs">Base Pattern</div>
          <div className="font-semibold capitalize">
            {stock.base_type && stock.base_type !== 'none' ? (
              <span className="text-blue-400">{stock.base_type} base</span>
            ) : (
              <span className="text-dark-500">No base</span>
            )}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-xs">Weeks in Base</div>
          <div className="font-semibold">
            {stock.weeks_in_base > 0 ? (
              <span>{stock.weeks_in_base} weeks</span>
            ) : (
              <span className="text-dark-500">-</span>
            )}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-xs">Volume Ratio</div>
          <div className={`font-semibold ${stock.volume_ratio >= 1.5 ? 'text-green-400' : stock.volume_ratio >= 1.0 ? 'text-yellow-400' : 'text-dark-400'}`}>
            {stock.volume_ratio ? `${stock.volume_ratio.toFixed(1)}x avg` : '-'}
          </div>
        </div>

        <div>
          <div className="text-dark-400 text-xs">Breakout Volume</div>
          <div className="font-semibold">
            {stock.breakout_volume_ratio ? (
              <span className="text-yellow-400">{stock.breakout_volume_ratio.toFixed(1)}x</span>
            ) : (
              <span className="text-dark-500">-</span>
            )}
          </div>
        </div>
      </div>

      {stock.eps_acceleration && (
        <div className="mt-3 pt-3 border-t border-dark-700 flex items-center gap-2">
          <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">EPS Accelerating</span>
          {stock.earnings_surprise_pct > 0 && (
            <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
              Beat estimates +{stock.earnings_surprise_pct.toFixed(0)}%
            </span>
          )}
        </div>
      )}

      <div className="text-dark-500 text-xs mt-3">
        {stock.is_breaking_out
          ? 'Stock is breaking out of a consolidation pattern with strong volume - potential buy zone.'
          : stock.base_type && stock.base_type !== 'none'
          ? 'Stock is building a base pattern. Watch for breakout with volume.'
          : 'No clear base pattern detected.'}
      </div>
    </div>
  )
}

export default function StockDetail() {
  const { ticker } = useParams()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [stock, setStock] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  const fetchStock = async () => {
    try {
      setLoading(true)
      const data = await api.getStock(ticker)
      setStock(data)
    } catch (err) {
      console.error('Failed to fetch stock:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStock()
  }, [ticker])

  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await api.refreshStock(ticker)
      await fetchStock()
    } catch (err) {
      console.error('Failed to refresh:', err)
    } finally {
      setRefreshing(false)
    }
  }

  const handleAddToWatchlist = async () => {
    try {
      await api.addToWatchlist({ ticker })
      alert('Added to watchlist!')
    } catch (err) {
      console.error('Failed to add to watchlist:', err)
    }
  }

  const handleAddToPortfolio = async () => {
    const shares = prompt('Number of shares:')
    const costBasis = prompt('Cost per share:')
    if (shares && costBasis) {
      try {
        await api.addPosition({
          ticker,
          shares: parseFloat(shares),
          cost_basis: parseFloat(costBasis)
        })
        alert('Added to portfolio!')
      } catch (err) {
        console.error('Failed to add to portfolio:', err)
      }
    }
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="skeleton h-8 w-32 mb-4" />
        <div className="skeleton h-32 rounded-2xl mb-4" />
        <div className="skeleton h-48 rounded-2xl mb-4" />
        <div className="skeleton h-32 rounded-2xl" />
      </div>
    )
  }

  if (!stock) {
    return (
      <div className="p-4">
        <div className="card text-center py-8">
          <div className="text-4xl mb-3">❓</div>
          <div className="font-semibold mb-2">Stock Not Found</div>
          <div className="text-dark-400 text-sm mb-4">
            {ticker} has not been analyzed yet.
          </div>
          <button onClick={() => navigate(-1)} className="btn-primary">
            Go Back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <button
            onClick={() => navigate(-1)}
            className="text-primary-500 text-sm mb-2 flex items-center gap-1"
          >
            ← Back
          </button>
          <h1 className="text-2xl font-bold">{stock.ticker}</h1>
          <div className="text-dark-400">{stock.name}</div>
          <div className="text-dark-500 text-sm">{stock.sector} • {stock.industry}</div>
        </div>
        <ScoreGauge score={stock.canslim_score} label={getScoreLabel(stock.canslim_score)} />
      </div>

      <PriceInfo stock={stock} />

      <GrowthModeSection stock={stock} />

      <TechnicalAnalysis stock={stock} />

      <CANSLIMDetail stock={stock} />

      <ScoreHistory history={stock.score_history} />

      {/* Actions */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <button onClick={handleAddToWatchlist} className="btn-secondary">
          + Watchlist
        </button>
        <button onClick={handleAddToPortfolio} className="btn-primary">
          + Portfolio
        </button>
      </div>

      <button
        onClick={handleRefresh}
        disabled={refreshing}
        className="w-full btn-secondary flex items-center justify-center gap-2"
      >
        {refreshing ? (
          <>
            <span className="animate-spin">⟳</span>
            <span>Refreshing...</span>
          </>
        ) : (
          <>
            <span>🔄</span>
            <span>Refresh Analysis</span>
          </>
        )}
      </button>

      <div className="text-dark-500 text-xs text-center mt-3">
        Last updated: {stock.last_updated ? new Date(stock.last_updated).toLocaleString('en-US', { timeZone: 'America/Chicago' }) + ' CST' : 'Never'}
      </div>

      <div className="h-4" />
    </div>
  )
}
