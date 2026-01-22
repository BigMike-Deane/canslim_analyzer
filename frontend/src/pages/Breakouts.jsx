import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatScore, getScoreClass, formatCurrency } from '../api'

function BreakoutRow({ stock }) {
  const pctFromHigh = stock.week_52_high && stock.current_price
    ? ((stock.week_52_high - stock.current_price) / stock.week_52_high * 100)
    : null

  return (
    <Link
      to={`/stock/${stock.ticker}`}
      className="flex justify-between items-center py-3 border-b border-dark-700 last:border-0 hover:bg-dark-700/50 -mx-4 px-4 transition-colors"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-yellow-500/20 text-yellow-400 flex items-center justify-center text-lg">
          ⚡
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold">{stock.ticker}</span>
            {stock.base_type && stock.base_type !== 'none' && (
              <span className="text-[10px] bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded">
                {stock.base_type === 'cup_with_handle' ? 'cup+handle' : stock.base_type}
              </span>
            )}
            {stock.is_breaking_out && (
              <span className="text-[10px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                Breakout
              </span>
            )}
          </div>
          <div className="text-dark-400 text-sm truncate max-w-[180px]">
            {stock.name || stock.sector || '-'}
          </div>
        </div>
      </div>

      <div className="text-right">
        <div className="font-semibold">{formatCurrency(stock.current_price)}</div>
        <div className="flex items-center gap-2 justify-end">
          <span className={`text-sm px-2 py-0.5 rounded ${getScoreClass(stock.canslim_score)}`}>
            {formatScore(stock.canslim_score)}
          </span>
          {stock.volume_ratio && (
            <span className={`text-xs ${stock.volume_ratio >= 1.5 ? 'text-green-400' : 'text-dark-400'}`}>
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

  useEffect(() => {
    const fetchBreakouts = async () => {
      try {
        setLoading(true)
        const data = await api.getBreakingOutStocks(50) // Get up to 50
        setStocks(data.stocks || [])
      } catch (err) {
        console.error('Failed to fetch breakouts:', err)
      } finally {
        setLoading(false)
      }
    }
    fetchBreakouts()
  }, [])

  if (loading) {
    return (
      <div className="p-4">
        <h1 className="text-xl font-bold mb-4">Breaking Out</h1>
        <div className="card">
          <div className="animate-pulse space-y-3">
            {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
              <div key={i} className="h-16 bg-dark-700 rounded" />
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <span className="text-yellow-400">⚡</span> Breaking Out
        </h1>
        <span className="text-sm text-dark-400">{stocks.length} stocks</span>
      </div>

      <div className="card mb-4">
        <div className="text-sm text-dark-400 mb-3">
          Stocks clearing base patterns with strong volume - ideal CANSLIM buy points.
        </div>

        {stocks.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-4xl mb-3">📊</div>
            <div className="font-semibold mb-2">No Breakouts Detected</div>
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
      </div>

      <div className="card bg-dark-800/50">
        <div className="font-semibold text-sm mb-2">What is a Breakout?</div>
        <div className="text-dark-400 text-sm space-y-2">
          <p>A breakout occurs when a stock's price moves above a resistance level (pivot point) with increased volume.</p>
          <p><strong>Key signals:</strong></p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>Base pattern:</strong> Flat base, cup-with-handle, or double-bottom</li>
            <li><strong>Volume surge:</strong> 40%+ above average on breakout day</li>
            <li><strong>Price action:</strong> Within 5% above the pivot point</li>
          </ul>
        </div>
      </div>

      <div className="h-4" />
    </div>
  )
}
