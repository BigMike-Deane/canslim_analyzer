import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import { ScoreBadge, TagBadge } from '../components/Badge'
import PageHeader from '../components/PageHeader'

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
          <div className="text-dark-400 text-sm truncate max-w-[140px] sm:max-w-[220px]">
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
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchBreakouts = async () => {
      try {
        setLoading(true)
        const data = await api.getBreakingOutStocks(50) // Get up to 50
        setStocks(data.stocks || [])
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

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Breaking Out"
        subtitle="Stocks clearing base patterns with strong volume"
        badge={
          <span className="text-xs font-data text-dark-400">{stocks.length} stocks</span>
        }
      />

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
        ) : (
          <div>
            {stocks.map(stock => (
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
