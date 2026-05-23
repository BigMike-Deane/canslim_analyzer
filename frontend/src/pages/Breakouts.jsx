import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatRelativeTime } from '../api'
import { saveStockListContext } from '../stockListContext'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, TagBadge } from '../components/Badge'
import PageHeader from '../components/PageHeader'

const SORT_OPTIONS = [
  { value: 'canslim_score', label: 'Score' },
  { value: 'volume_ratio', label: 'Volume' },
  { value: 'current_price', label: 'Price' },
  { value: 'pct_from_high', label: 'Near 52wk High' },
]

const BASE_TYPE_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'cup_with_handle', label: 'Cup+Handle' },
  { key: 'cup', label: 'Cup' },
  { key: 'flat_base', label: 'Flat Base' },
  { key: 'double_bottom', label: 'Double Bottom' },
]

function BreakoutRow({ stock }) {
  const pctFromHigh = stock.week_52_high && stock.current_price
    ? ((stock.week_52_high - stock.current_price) / stock.week_52_high * 100)
    : null

  return (
    <Link
      to={`/stock/${stock.ticker}`}
      className="flex justify-between items-center py-3 border-b border-dark-700/30 last:border-0 hover:bg-dark-750/50 -mx-4 px-4 transition-colors"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center text-lg">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="13 2 13 9 20 9" />
            <polygon points="13 2 20 9 20 22 4 22 4 2 13 2" />
            <polyline points="7 13 10 16 17 9" />
          </svg>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-dark-50">{stock.ticker}</span>
            {stock.base_type && stock.base_type !== 'none' && (
              <TagBadge color="cyan">
                {stock.base_type === 'cup_with_handle' ? 'cup+handle' : stock.base_type}
              </TagBadge>
            )}
            {stock.is_breaking_out && (
              <TagBadge color="green">Breakout</TagBadge>
            )}
          </div>
          <div className="text-dark-400 text-sm truncate max-w-[180px] sm:max-w-[260px]">
            {stock.name || stock.sector || '-'}
          </div>
        </div>
      </div>

      <div className="text-right">
        <div className="font-data font-semibold text-dark-100">{formatCurrency(stock.current_price)}</div>
        <div className="flex items-center gap-2 justify-end mt-1">
          <ScoreBadge score={stock.canslim_score} size="sm" />
          {stock.volume_ratio != null && (
            <span className={`text-[10px] font-data ${stock.volume_ratio >= 1.5 ? 'text-emerald-400' : 'text-dark-500'}`}>
              {stock.volume_ratio.toFixed(1)}x vol
            </span>
          )}
        </div>
      </div>
    </Link>
  )
}

export default function Breakouts() {
  const [loading, setLoading] = useState(true)
  const [stocks, setStocks] = useState([])
  const [asOf, setAsOf] = useState(null)
  const [error, setError] = useState(null)
  const [sortBy, setSortBy] = useState('canslim_score')
  const [baseType, setBaseType] = useState('all')
  const [sector, setSector] = useState('')

  useEffect(() => {
    const fetchBreakouts = async () => {
      try {
        setLoading(true)
        const data = await api.getBreakingOutStocks(50) // Get up to 50
        setStocks(data.stocks || [])
        setAsOf(data.as_of || null)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch breakouts:', err)
        setError(err.message || 'Failed to load breakouts')
      } finally {
        setLoading(false)
      }
    }
    fetchBreakouts()
  }, [])

  // Distinct sectors present in the fetched set
  const sectors = useMemo(() => {
    const s = new Set()
    stocks.forEach(st => { if (st.sector) s.add(st.sector) })
    return [...s].sort()
  }, [stocks])

  // Apply filter + sort. Reset sector if it disappears from the set.
  const displayStocks = useMemo(() => {
    let arr = stocks.slice()
    if (baseType !== 'all') {
      arr = arr.filter(s => s.base_type === baseType)
    }
    if (sector) {
      arr = arr.filter(s => s.sector === sector)
    }
    arr.sort((a, b) => {
      if (sortBy === 'pct_from_high') {
        // Closer-to-high first (smaller pct gap = first)
        const aPct = a.week_52_high && a.current_price
          ? (a.week_52_high - a.current_price) / a.week_52_high
          : Infinity
        const bPct = b.week_52_high && b.current_price
          ? (b.week_52_high - b.current_price) / b.week_52_high
          : Infinity
        return aPct - bPct
      }
      const av = a[sortBy] ?? -Infinity
      const bv = b[sortBy] ?? -Infinity
      return bv - av
    })
    return arr
  }, [stocks, sortBy, baseType, sector])

  // Surface the (filtered+sorted) breakout order to StockDetail so its
  // prev/next nav walks through the same list the user is viewing.
  useEffect(() => {
    if (displayStocks.length > 0) {
      saveStockListContext('Breakouts', displayStocks.map(s => s.ticker))
    }
  }, [displayStocks])

  if (loading) {
    return (
      <div className="p-4 md:p-6">
        <PageHeader title="Breaking Out" />
        <Card variant="glass">
          <div className="space-y-3">
            {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
              <div key={i} className="h-16 bg-dark-750/50 rounded-lg animate-pulse" />
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const hasFilters = baseType !== 'all' || sector !== ''
  const countLabel = hasFilters && displayStocks.length !== stocks.length
    ? `${displayStocks.length} of ${stocks.length}`
    : `${stocks.length}`

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Breaking Out"
        subtitle={asOf
          ? `Stocks clearing base patterns with strong volume · Updated ${formatRelativeTime(asOf)}`
          : 'Stocks clearing base patterns with strong volume'}
        badge={
          <span className="text-xs font-data text-dark-400">{countLabel} stocks</span>
        }
      />

      {/* Filters */}
      {!error && stocks.length > 0 && (
        <div className="mb-3 space-y-2">
          {/* Base-type pills */}
          <div className="flex gap-1.5 overflow-x-auto pb-1 -mx-1 px-1">
            {BASE_TYPE_FILTERS.map(f => (
              <button
                key={f.key}
                onClick={() => setBaseType(f.key)}
                className={`text-xs px-3.5 py-2 rounded-full whitespace-nowrap transition-colors ${
                  baseType === f.key
                    ? 'bg-primary-500/30 text-primary-300 font-medium'
                    : 'bg-dark-700 text-dark-400 hover:bg-dark-600'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Sort + sector dropdowns */}
          <div className="flex gap-2 flex-wrap">
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              className="text-xs bg-dark-700 text-dark-300 border border-dark-600 rounded-lg px-2.5 py-2 focus:outline-none focus:border-primary-500/40"
              aria-label="Sort by"
            >
              {SORT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>Sort: {o.label}</option>
              ))}
            </select>
            {sectors.length > 1 && (
              <select
                value={sector}
                onChange={e => setSector(e.target.value)}
                className="text-xs bg-dark-700 text-dark-300 border border-dark-600 rounded-lg px-2.5 py-2 focus:outline-none focus:border-primary-500/40"
                aria-label="Filter by sector"
              >
                <option value="">All Sectors</option>
                {sectors.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            )}
            {hasFilters && (
              <button
                onClick={() => { setBaseType('all'); setSector('') }}
                className="text-xs px-3 py-2 rounded-lg text-dark-400 hover:text-dark-200 transition-colors"
              >
                Clear filters
              </button>
            )}
          </div>
        </div>
      )}

      <Card variant="glass" className="mb-4">
        {error ? (
          <div className="text-center py-8">
            <div className="text-red-400 text-sm mb-2">Failed to load breakouts</div>
            <div className="text-dark-500 text-xs">{error}</div>
          </div>
        ) : stocks.length === 0 ? (
          <div className="text-center py-8">
            <div className="font-semibold text-dark-100 mb-2">No Breakouts Detected</div>
            <div className="text-dark-400 text-sm">
              Breakouts occur when stocks clear consolidation patterns with above-average volume.
              Check back after market activity.
            </div>
          </div>
        ) : displayStocks.length === 0 ? (
          <div className="text-center py-8">
            <div className="font-semibold text-dark-100 mb-2">No matches for this filter</div>
            <div className="text-dark-400 text-sm">
              Try a different base type or sector — {stocks.length} breakout{stocks.length !== 1 ? 's' : ''} available with other filters.
            </div>
          </div>
        ) : (
          <div>
            {displayStocks.map(stock => (
              <BreakoutRow key={stock.ticker} stock={stock} />
            ))}
          </div>
        )}
      </Card>

      <Card variant="glass" className="bg-dark-850/30">
        <CardHeader title="What is a Breakout?" />
        <div className="text-dark-400 text-sm space-y-2">
          <p>A breakout occurs when a stock's price moves above a resistance level (pivot point) with increased volume.</p>
          <SectionLabel>Key Signals</SectionLabel>
          <ul className="space-y-1.5 ml-1">
            <li className="flex items-start gap-2">
              <span className="text-primary-500 mt-1 shrink-0">--</span>
              <span><span className="text-dark-200 font-medium">Base pattern:</span> Flat base, cup-with-handle, or double-bottom</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-500 mt-1 shrink-0">--</span>
              <span><span className="text-dark-200 font-medium">Volume surge:</span> 40%+ above average on breakout day</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary-500 mt-1 shrink-0">--</span>
              <span><span className="text-dark-200 font-medium">Price action:</span> Within 5% above the pivot point</span>
            </li>
          </ul>
        </div>
      </Card>

      <div className="h-4" />
    </div>
  )
}
