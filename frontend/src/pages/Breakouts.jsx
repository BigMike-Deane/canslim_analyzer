import { useState, useEffect, useMemo } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, formatCurrency, formatRelativeTime } from '../api'
import { saveStockListContext } from '../stockListContext'
import Card, { SectionLabel } from '../components/Card'
import CollapsedDrawer from '../components/CollapsedDrawer'
import { ScoreBadge, TagBadge, BaseTag } from '../components/Badge'
import PageHeader from '../components/PageHeader'
import useApi from '../hooks/useApi'

const SORT_OPTIONS = [
  { value: 'canslim_score', label: 'Score' },
  { value: 'volume_ratio', label: 'Volume' },
  { value: 'current_price', label: 'Price' },
  { value: 'pct_from_high', label: 'Near 52wk High' },
  { value: 'freshness', label: 'Freshness' },
]

// Canonical per-stock freshness column (mirrors backend `Stock.last_updated`,
// the same field the page-level `as_of` is derived from). Returns a millisecond
// epoch, or -Infinity when the row carries no/invalid timestamp so unknowns
// always sort last under most-recent-first ordering.
function freshnessMillis(stock) {
  const raw = stock?.last_updated
  if (!raw) return -Infinity
  const t = Date.parse(raw)
  return Number.isNaN(t) ? -Infinity : t
}

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
            <BaseTag
              baseType={stock.base_type}
              pivotPrice={stock.pivot_price}
              currentPrice={stock.current_price}
            />
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
          <ScoreBadge score={stock.canslim_score} ticker={stock.ticker} size="sm" />
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
  const [searchParams, setSearchParams] = useSearchParams()
  const { data, error, loading } = useApi(() => api.getBreakingOutStocks(50), [])
  const stocks = data?.stocks || []
  const asOf = data?.as_of || null
  // Seed filter + sort state from the URL so links/back-button restore the view.
  const [sortBy, setSortBy] = useState(() => {
    const s = searchParams.get('sort')
    return SORT_OPTIONS.some(o => o.value === s) ? s : 'canslim_score'
  })
  const [baseType, setBaseType] = useState(() => {
    const b = searchParams.get('base')
    return BASE_TYPE_FILTERS.some(f => f.key === b) ? b : 'all'
  })
  const [sector, setSector] = useState(() => searchParams.get('sector') || '')

  // Mirror the current filter/sort state back into the URL (replace, no history
  // spam). Defaults are dropped so a pristine view has a clean URL.
  useEffect(() => {
    const next = new URLSearchParams(searchParams)
    if (sortBy && sortBy !== 'canslim_score') next.set('sort', sortBy)
    else next.delete('sort')
    if (baseType && baseType !== 'all') next.set('base', baseType)
    else next.delete('base')
    if (sector) next.set('sector', sector)
    else next.delete('sector')
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortBy, baseType, sector])

  // Distinct sectors present in the fetched set
  const sectors = useMemo(() => {
    const s = new Set()
    stocks.forEach(st => { if (st.sector) s.add(st.sector) })
    return [...s].sort()
  }, [stocks])

  // A sector seeded from the URL may not exist in the freshly-fetched set;
  // drop it so the dropdown never shows a value with no matching <option>.
  useEffect(() => {
    if (sector && stocks.length > 0 && !sectors.includes(sector)) {
      setSector('')
    }
  }, [sectors, sector, stocks.length])

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
      if (sortBy === 'freshness') {
        // Most-recently-updated first; missing timestamps sort last.
        return freshnessMillis(b) - freshnessMillis(a)
      }
      const av = a[sortBy] ?? -Infinity
      const bv = b[sortBy] ?? -Infinity
      return bv - av
    })
    return arr
  }, [stocks, sortBy, baseType, sector])

  // Per-pill match counts, computed against the set with all OTHER (non
  // base-type) filters applied — i.e. the sector filter — so each pill shows
  // exactly how many rows selecting it would yield. The "all" pill counts the
  // full (sector-filtered) set.
  const baseTypeCounts = useMemo(() => {
    const base = sector ? stocks.filter(s => s.sector === sector) : stocks
    const counts = { all: base.length }
    for (const f of BASE_TYPE_FILTERS) {
      if (f.key === 'all') continue
      counts[f.key] = base.filter(s => s.base_type === f.key).length
    }
    return counts
  }, [stocks, sector])

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
            {BASE_TYPE_FILTERS.map(f => {
              const count = baseTypeCounts[f.key] ?? 0
              return (
                <button
                  key={f.key}
                  onClick={() => setBaseType(f.key)}
                  className={`text-xs px-3.5 py-2 rounded-full whitespace-nowrap transition-colors ${
                    baseType === f.key
                      ? 'bg-primary-500/30 text-primary-300 font-medium'
                      : 'bg-dark-700 text-dark-400 hover:bg-dark-600'
                  }`}
                >
                  {f.label} <span className="opacity-60">({count})</span>
                </button>
              )
            })}
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
            <div className="text-dark-500 text-xs">{error.message || 'Failed to load breakouts'}</div>
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

      <CollapsedDrawer
        title="What is a Breakout?"
        cardProps={{ className: 'bg-dark-850/30' }}
        badge={<TagBadge color="cyan">GUIDE</TagBadge>}
      >
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
      </CollapsedDrawer>

      <div className="h-4" />
    </div>
  )
}
