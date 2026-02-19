import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api, getScoreClass, formatCurrency, formatPercent, formatMarketCap } from '../api'
import Card, { SectionLabel } from '../components/Card'
import { ScoreBadge } from '../components/Badge'
import { MiniStat } from '../components/StatGrid'
import PageHeader from '../components/PageHeader'

function FilterBar({ filters, onFilterChange, sectors }) {
  return (
    <Card variant="glass" className="mb-4">
      <SectionLabel>Filters</SectionLabel>
      <div className="space-y-3">
        <div className="flex gap-2">
          <select
            value={filters.sector || ''}
            onChange={(e) => onFilterChange({ ...filters, sector: e.target.value || null })}
            className="flex-1 text-sm bg-dark-800 border border-dark-700/50 rounded-lg px-3 py-2 text-dark-100 focus:border-primary-500/40 focus:outline-none transition-colors"
          >
            <option value="">All Sectors</option>
            {(sectors || []).map(sector => (
              <option key={sector} value={sector}>{sector}</option>
            ))}
          </select>

          <select
            value={filters.sort_by || 'canslim_score'}
            onChange={(e) => onFilterChange({ ...filters, sort_by: e.target.value })}
            className="flex-1 text-sm bg-dark-800 border border-dark-700/50 rounded-lg px-3 py-2 text-dark-100 focus:border-primary-500/40 focus:outline-none transition-colors"
          >
            <option value="canslim_score">Score (High to Low)</option>
            <option value="projected_growth">Growth Potential</option>
            <option value="market_cap">Market Cap</option>
          </select>
        </div>

        <div className="flex gap-2 items-center">
          <span className="text-dark-400 text-[10px] tracking-wider uppercase font-semibold">Min Score</span>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={filters.min_score || 0}
            onChange={(e) => onFilterChange({ ...filters, min_score: parseInt(e.target.value) })}
            className="flex-1 accent-primary-500"
          />
          <span className="text-sm font-data font-medium text-dark-100 w-8 text-right">{filters.min_score || 0}</span>
        </div>
      </div>
    </Card>
  )
}

function CANSLIMBreakdown({ stock }) {
  if (!stock) return null
  const scores = [
    { key: 'C', label: 'Current Earnings', value: stock.c_score, max: 15 },
    { key: 'A', label: 'Annual Earnings', value: stock.a_score, max: 15 },
    { key: 'N', label: 'New Highs', value: stock.n_score, max: 15 },
    { key: 'S', label: 'Supply/Demand', value: stock.s_score, max: 15 },
    { key: 'L', label: 'Leader/Laggard', value: stock.l_score, max: 15 },
    { key: 'I', label: 'Institutional', value: stock.i_score, max: 10 },
    { key: 'M', label: 'Market Direction', value: stock.m_score, max: 15 },
  ]

  const normalizeScore = (value, max) => {
    if (value == null || max === 0) return 0
    return (value / max) * 100
  }

  return (
    <div className="flex gap-1 mt-2">
      {scores.map(s => {
        const normalized = normalizeScore(s.value, s.max)
        return (
          <div
            key={s.key}
            title={`${s.label}: ${s.value?.toFixed(1) || 0}/${s.max}`}
            className={`w-6 h-6 rounded text-[10px] font-bold font-data flex items-center justify-center ${getScoreClass(normalized)}`}
          >
            {s.key}
          </div>
        )
      })}
    </div>
  )
}

function StockRow({ stock }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border-b border-dark-700/30 last:border-0">
      <div
        className="flex justify-between items-center py-3 cursor-pointer hover:bg-dark-750/50 -mx-4 px-4 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-dark-50">{stock.ticker}</span>
            <span className="text-dark-500 text-[10px] font-data">{formatMarketCap(stock.market_cap)}</span>
          </div>
          <div className="text-dark-400 text-sm truncate">{stock.name}</div>
          <div className="text-dark-500 text-[10px] tracking-wide">{stock.sector}</div>
        </div>

        <div className="text-right ml-3 flex flex-col items-end gap-1">
          <ScoreBadge score={stock.canslim_score} size="md" />
          {stock.projected_growth != null && (
            <span className="text-[10px] text-emerald-400 font-data">
              +{stock.projected_growth.toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      {expanded && (
        <div className="pb-3 px-4 -mx-4 bg-dark-800/40 border-t border-dark-700/20">
          <CANSLIMBreakdown stock={stock} />

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3 mt-3">
            <MiniStat label="Price" value={formatCurrency(stock.current_price)} />
            <MiniStat label="52W High" value={formatCurrency(stock.week_52_high)} />
            <MiniStat
              label="From High"
              value={stock.week_52_high ? formatPercent((stock.current_price / stock.week_52_high - 1) * 100) : '-'}
              color={stock.current_price < stock.week_52_high * 0.9 ? 'text-red-400' : 'text-emerald-400'}
            />
          </div>

          <div className="flex gap-2 mt-3">
            <Link
              to={`/stock/${stock.ticker}`}
              className="flex-1 text-center text-sm py-2 rounded-lg bg-primary-500/15 text-primary-400 border border-primary-500/20 hover:bg-primary-500/25 transition-colors font-medium"
            >
              View Details
            </Link>
            <button className="text-sm px-4 py-2 rounded-lg bg-dark-700 text-dark-300 border border-dark-600 hover:bg-dark-600 transition-colors">
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
  const [error, setError] = useState(null)
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
      setError(null)

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
      setError(err.message)
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
    <div className="p-4 md:p-6">
      <PageHeader
        title="Stock Screener"
        subtitle={`${total} stocks found`}
      />

      <FilterBar
        filters={filters}
        onFilterChange={handleFilterChange}
        sectors={sectors}
      />

      {error && !loading && stocks.length === 0 && (
        <Card variant="glass" className="text-center py-8 text-red-400 mb-4">
          Failed to load stocks: {error}
        </Card>
      )}

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="h-20 rounded-xl bg-dark-800/40 animate-pulse" />
          ))}
        </div>
      ) : stocks.length === 0 && !error ? (
        <Card variant="glass" className="text-center py-8">
          <div className="font-semibold text-dark-100 mb-2">No Stocks Found</div>
          <div className="text-dark-400 text-sm">
            Try adjusting your filters or run a scan from the Dashboard.
          </div>
        </Card>
      ) : (
        <Card variant="glass">
          {stocks.map(stock => (
            <StockRow key={stock.ticker} stock={stock} />
          ))}
        </Card>
      )}

      <div className="h-4" />
    </div>
  )
}
