import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency, formatPercent, formatMarketCap } from '../api'

function FilterBar({ filters, onFilterChange, sectors }) {
  return (
    <div className="mb-4 space-y-3">
      <div className="flex gap-2">
        <select
          value={filters.sector || ''}
          onChange={(e) => onFilterChange({ ...filters, sector: e.target.value || null })}
          className="flex-1 text-sm"
        >
          <option value="">All Sectors</option>
          {sectors.map(sector => (
            <option key={sector} value={sector}>{sector}</option>
          ))}
        </select>

        <select
          value={filters.sort_by || 'canslim_score'}
          onChange={(e) => onFilterChange({ ...filters, sort_by: e.target.value })}
          className="flex-1 text-sm"
        >
          <option value="canslim_score">Score (High to Low)</option>
          <option value="projected_growth">Growth Potential</option>
          <option value="market_cap">Market Cap</option>
        </select>
      </div>

      <div className="flex gap-2 items-center">
        <span className="text-dark-400 text-sm">Min Score:</span>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={filters.min_score || 0}
          onChange={(e) => onFilterChange({ ...filters, min_score: parseInt(e.target.value) })}
          className="flex-1"
        />
        <span className="text-sm font-medium w-8">{filters.min_score || 0}</span>
      </div>
    </div>
  )
}

function CANSLIMBreakdown({ stock }) {
  const scores = [
    { key: 'C', label: 'Current Earnings', value: stock.c_score },
    { key: 'A', label: 'Annual Earnings', value: stock.a_score },
    { key: 'N', label: 'New Highs', value: stock.n_score },
    { key: 'S', label: 'Supply/Demand', value: stock.s_score },
    { key: 'L', label: 'Leader/Laggard', value: stock.l_score },
    { key: 'I', label: 'Institutional', value: stock.i_score },
    { key: 'M', label: 'Market Direction', value: stock.m_score },
  ]

  return (
    <div className="flex gap-1 mt-2">
      {scores.map(s => (
        <div
          key={s.key}
          title={`${s.label}: ${formatScore(s.value)}`}
          className={`w-6 h-6 rounded text-xs font-bold flex items-center justify-center ${getScoreClass(s.value)}`}
        >
          {s.key}
        </div>
      ))}
    </div>
  )
}

function StockRow({ stock }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border-b border-dark-700 last:border-0">
      <div
        className="flex justify-between items-center py-3 cursor-pointer hover:bg-dark-700/50 -mx-4 px-4 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold">{stock.ticker}</span>
            <span className="text-dark-400 text-xs">{formatMarketCap(stock.market_cap)}</span>
          </div>
          <div className="text-dark-400 text-sm truncate">{stock.name}</div>
          <div className="text-dark-500 text-xs">{stock.sector}</div>
        </div>

        <div className="text-right ml-3">
          <div className={`px-3 py-1 rounded font-semibold ${getScoreClass(stock.canslim_score)}`}>
            {formatScore(stock.canslim_score)}
          </div>
          {stock.projected_growth != null && (
            <div className="text-xs text-green-400 mt-1">
              +{stock.projected_growth.toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      {expanded && (
        <div className="pb-3 px-4 -mx-4 bg-dark-700/30">
          <CANSLIMBreakdown stock={stock} />

          <div className="grid grid-cols-3 gap-2 mt-3 text-sm">
            <div>
              <div className="text-dark-400 text-xs">Price</div>
              <div className="font-medium">{formatCurrency(stock.current_price)}</div>
            </div>
            <div>
              <div className="text-dark-400 text-xs">52W High</div>
              <div className="font-medium">{formatCurrency(stock.week_52_high)}</div>
            </div>
            <div>
              <div className="text-dark-400 text-xs">From High</div>
              <div className={`font-medium ${stock.current_price < stock.week_52_high * 0.9 ? 'text-red-400' : 'text-green-400'}`}>
                {stock.week_52_high ? formatPercent((stock.current_price / stock.week_52_high - 1) * 100) : '-'}
              </div>
            </div>
          </div>

          <div className="flex gap-2 mt-3">
            <Link
              to={`/stock/${stock.ticker}`}
              className="flex-1 btn-primary text-center text-sm"
            >
              View Details
            </Link>
            <button className="btn-secondary text-sm px-3">
              + Watch
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function Screener() {
  const [loading, setLoading] = useState(true)
  const [stocks, setStocks] = useState([])
  const [sectors, setSectors] = useState([])
  const [filters, setFilters] = useState({
    sector: null,
    min_score: 0,
    sort_by: 'canslim_score',
    limit: 50
  })
  const [total, setTotal] = useState(0)

  const fetchStocks = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.getStocks(filters)
      setStocks(data.stocks || [])
      setTotal(data.total || 0)

      // Extract unique sectors
      if (data.stocks?.length > 0) {
        const uniqueSectors = [...new Set(data.stocks.map(s => s.sector).filter(Boolean))]
        setSectors(prev => {
          const combined = [...new Set([...prev, ...uniqueSectors])]
          return combined.sort()
        })
      }
    } catch (err) {
      console.error('Failed to fetch stocks:', err)
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    fetchStocks()
  }, [fetchStocks])

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters)
  }

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">Stock Screener</h1>

      <FilterBar
        filters={filters}
        onFilterChange={handleFilterChange}
        sectors={sectors}
      />

      <div className="text-dark-400 text-sm mb-3">
        {total} stocks found
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="skeleton h-20 rounded-lg" />
          ))}
        </div>
      ) : stocks.length === 0 ? (
        <div className="card text-center py-8">
          <div className="text-4xl mb-3">📊</div>
          <div className="font-semibold mb-2">No Stocks Found</div>
          <div className="text-dark-400 text-sm">
            Try adjusting your filters or run a scan from the Dashboard.
          </div>
        </div>
      ) : (
        <div className="card">
          {stocks.map(stock => (
            <StockRow key={stock.ticker} stock={stock} />
          ))}
        </div>
      )}

      <div className="h-4" />
    </div>
  )
}
