import { Link } from 'react-router-dom'
import { api } from '../api'
import Card, { CardHeader } from '../components/Card'
import PageHeader from '../components/PageHeader'
import useApi from '../hooks/useApi'

function getCellColor(value, isDiagonal) {
  if (isDiagonal) return 'bg-dark-600'
  if (value > 0.8) return 'bg-red-600/70'
  if (value > 0.6) return 'bg-orange-500/60'
  if (value > 0.3) return 'bg-amber-500/50'
  return 'bg-emerald-600/40'
}

function getCellTextColor(value, isDiagonal) {
  if (isDiagonal) return 'text-dark-300'
  if (value > 0.8) return 'text-red-200'
  if (value > 0.6) return 'text-orange-200'
  if (value > 0.3) return 'text-amber-200'
  return 'text-emerald-200'
}

function getCorrelationColor(value) {
  if (value > 0.8) return 'text-red-400'
  if (value > 0.6) return 'text-orange-400'
  if (value > 0.3) return 'text-amber-400'
  return 'text-emerald-400'
}

function getCorrelationBg(value) {
  if (value > 0.8) return 'bg-red-500/20 border-red-500/30'
  if (value > 0.6) return 'bg-orange-500/20 border-orange-500/30'
  if (value > 0.3) return 'bg-amber-500/20 border-amber-500/30'
  return 'bg-emerald-500/20 border-emerald-500/30'
}

export default function CorrelationMatrix() {
  const { data, loading, error } = useApi(
    () => api.getCorrelationMatrix().then(r => r.data || r), [])

  if (loading) {
    return (
      <div className="p-4 md:p-6 max-w-5xl mx-auto">
        <PageHeader title="Correlation Matrix" subtitle="Portfolio position correlations" />
        <div className="space-y-3">
          <div className="skeleton h-24 rounded-2xl" />
          <div className="skeleton h-64 rounded-2xl" />
        </div>
      </div>
    )
  }

  if (error || !data || data.error) {
    return (
      <div className="p-4 md:p-6 max-w-5xl mx-auto">
        <PageHeader title="Correlation Matrix" subtitle="Portfolio position correlations" />
        <Card variant="glass" className="text-center py-8">
          <div className="text-red-400 text-sm">{error?.message || data?.error || 'No data available'}</div>
        </Card>
      </div>
    )
  }

  const { matrix, tickers, max_correlation, max_pair, positions_count } = data

  if (!tickers || tickers.length < 2) {
    return (
      <div className="p-4 md:p-6 max-w-5xl mx-auto">
        <PageHeader title="Correlation Matrix" subtitle="Portfolio position correlations" />
        <Card variant="glass" className="py-10 px-4 text-center">
          <div className="w-12 h-12 rounded-full bg-dark-700 flex items-center justify-center mx-auto mb-3 text-primary-400">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="9" cy="9" r="4" />
              <circle cx="17" cy="15" r="4" />
            </svg>
          </div>
          <div className="text-dark-100 font-semibold text-base mb-2">
            Diversification check unavailable
          </div>
          <p className="text-dark-400 text-sm max-w-md mx-auto mb-2">
            The correlation matrix shows how closely your positions move together — high correlations (above 0.7) signal concentration risk that simple sector counts can miss.
          </p>
          <p className="text-dark-500 text-xs mb-4">
            Currently holding {positions_count || 0} position{positions_count !== 1 ? 's' : ''}. Need at least 2 to compute correlations.
          </p>
          <Link
            to="/ai-portfolio"
            className="inline-flex items-center gap-1.5 text-xs px-3.5 py-2 rounded-full bg-primary-500/20 text-primary-300 hover:bg-primary-500/30 transition-colors"
          >
            Open AI Portfolio
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 18l6-6-6-6" />
            </svg>
          </Link>
        </Card>
      </div>
    )
  }

  const hasWarning = max_correlation > 0.70

  return (
    <div className="p-4 md:p-6 max-w-5xl mx-auto">
      <PageHeader
        title="Correlation Matrix"
        subtitle={`${positions_count} positions analyzed`}
      />

      {/* Warning banner */}
      {hasWarning && (
        <div className="mb-4 flex items-center gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-red-400 flex-shrink-0">
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span className="text-red-300 text-sm">
            High correlation detected: {max_pair?.[0]}-{max_pair?.[1]} at {max_correlation?.toFixed(2)} — consider diversifying
          </span>
        </div>
      )}

      {/* Summary card */}
      <Card variant="glass" className="mb-4">
        <CardHeader title="Summary" />
        <div className="flex items-center gap-3">
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${getCorrelationBg(max_correlation)}`}>
            <span className="text-xs text-dark-300">Highest correlation:</span>
            <span className={`text-sm font-data font-semibold ${getCorrelationColor(max_correlation || 0)}`}>
              {max_pair ? `${max_pair[0]}-${max_pair[1]} at ${max_correlation?.toFixed(2)}` : 'N/A'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-4 mt-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-emerald-600/40" />
            <span className="text-[10px] text-dark-500">Low (&lt;0.3)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-amber-500/50" />
            <span className="text-[10px] text-dark-500">Medium (0.3-0.6)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-orange-500/60" />
            <span className="text-[10px] text-dark-500">High (0.6-0.8)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-red-600/70" />
            <span className="text-[10px] text-dark-500">Very High (&gt;0.8)</span>
          </div>
        </div>
      </Card>

      {/* Heat map */}
      <Card variant="glass">
        <CardHeader title="Correlation Heat Map" />
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-xs text-dark-500 font-medium text-left" />
                {tickers.map(ticker => (
                  <th key={ticker} className="p-2 text-xs text-dark-300 font-data font-medium text-center min-w-[60px]">
                    {ticker}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tickers.map((rowTicker, rowIdx) => (
                <tr key={rowTicker}>
                  <td className="p-2 text-xs text-dark-300 font-data font-medium whitespace-nowrap">
                    {rowTicker}
                  </td>
                  {tickers.map((colTicker, colIdx) => {
                    const value = matrix[rowIdx][colIdx]
                    const isDiagonal = rowIdx === colIdx
                    return (
                      <td
                        key={colTicker}
                        className={`p-2 text-center text-xs font-data font-medium rounded ${getCellColor(value, isDiagonal)} ${getCellTextColor(value, isDiagonal)}`}
                        title={`${rowTicker} vs ${colTicker}: ${value.toFixed(2)}`}
                      >
                        {value.toFixed(2)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="h-4" />
    </div>
  )
}
