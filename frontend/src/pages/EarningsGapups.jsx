import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import Card from '../components/Card'
import { ScoreBadge, TagBadge } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import DataTable from '../components/DataTable'

const DAY_OPTIONS = [3, 7, 14, 30]

export default function EarningsGapups() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [days, setDays] = useState(7)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api.getEarningsGapups(days)
      .then(d => { if (!cancelled) { setData(d?.data || d); setError(null) } })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [days])

  const gapups = data?.gapups || []
  const actionable = gapups.filter(g => g.is_actionable)
  const avgGap = gapups.length > 0
    ? (gapups.reduce((s, g) => s + (g.gap_pct || 0), 0) / gapups.length).toFixed(1)
    : '-'

  const columns = [
    {
      key: 'ticker',
      label: 'Ticker',
      sortable: true,
      render: (_, row) => (
        <div>
          <Link to={`/stock/${row.ticker}`} className="text-primary-400 font-medium hover:text-primary-300">
            {row.ticker}
          </Link>
          <div className="text-[10px] text-dark-500 truncate max-w-[120px] sm:max-w-[180px]">
            {row.name || '-'}
          </div>
        </div>
      ),
    },
    {
      key: 'gap_pct',
      label: 'Gap',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => (
        <span className={`font-data text-sm font-semibold ${val >= 10 ? 'text-emerald-300' : 'text-emerald-400'}`}>
          +{val?.toFixed(1)}%
        </span>
      ),
    },
    {
      key: 'is_actionable',
      label: 'Entry',
      align: 'center',
      sortable: true,
      render: (val) => val
        ? <TagBadge color="green">ACTIONABLE</TagBadge>
        : <span className="text-dark-500 text-[10px]">extended</span>,
    },
    {
      key: 'canslim_score',
      label: 'CANSLIM',
      align: 'center',
      sortable: true,
      render: (val, row) => <ScoreBadge score={val} ticker={row.ticker} size="sm" />,
    },
    {
      key: 'volume_ratio',
      label: 'Gap Vol',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => val != null
        ? <span className={`font-data text-xs ${val >= 1.5 ? 'text-emerald-400' : 'text-dark-300'}`}>{val.toFixed(1)}x</span>
        : <span className="text-dark-500 text-xs">-</span>,
    },
    {
      key: 'beat_streak',
      label: 'Beats',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => <span className="font-data text-xs text-dark-200">{val ?? '-'}</span>,
    },
    {
      key: 'earnings_surprise_pct',
      label: 'Surprise',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => val != null
        ? <span className={`font-data text-xs ${val > 0 ? 'text-emerald-400' : 'text-red-400'}`}>{val > 0 ? '+' : ''}{val.toFixed(1)}%</span>
        : <span className="text-dark-500 text-xs">-</span>,
    },
    {
      key: 'base_type',
      label: 'Base',
      mobileHide: true,
      render: (val) => val ? (
        <TagBadge color="cyan">
          {val === 'cup_with_handle' ? 'cup+handle' : val.replace(/_/g, ' ')}
        </TagBadge>
      ) : <span className="text-dark-500 text-[10px]">--</span>,
    },
    {
      key: 'earnings_date',
      label: 'Earnings',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => <span className="font-data text-xs text-dark-400">{val || '-'}</span>,
    },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Earnings Gap-Ups"
        backTo="/"
        backLabel="Command Center"
        subtitle="Stocks that gapped up on earnings with beat streaks — high-probability CANSLIM entries when not extended"
        badge={<TagBadge color="green">POST-EARNINGS</TagBadge>}
      />

      {/* Lookback selector */}
      <div className="flex gap-2 mb-4">
        {DAY_OPTIONS.map(d => (
          <button
            key={d}
            onClick={() => setDays(d)}
            className={`text-xs px-3 py-1.5 rounded-lg border transition-colors ${
              days === d
                ? 'bg-primary-500/20 border-primary-500/40 text-primary-300'
                : 'bg-dark-800 border-dark-700 text-dark-400 hover:text-dark-200'
            }`}
          >
            {d}d
          </button>
        ))}
      </div>

      {loading && !data && (
        <div className="space-y-3">
          <div className="skeleton h-16 rounded-2xl" />
          <div className="skeleton h-64 rounded-2xl" />
        </div>
      )}

      {error && !data && (
        <Card variant="glass" className="text-center py-8 text-red-400">
          Failed to load: {error}
        </Card>
      )}

      {data && (
        <>
          <Card variant="glass" className="mb-4">
            <StatGrid
              columns={3}
              stats={[
                { label: 'Gap-Ups', value: data.total ?? gapups.length, color: 'text-dark-50' },
                { label: 'Actionable', value: actionable.length, color: actionable.length > 0 ? 'text-emerald-400' : 'text-dark-400' },
                { label: 'Avg Gap', value: avgGap === '-' ? '-' : `+${avgGap}%`, color: 'text-emerald-400' },
              ]}
            />
          </Card>

          <Card variant="glass" padding="">
            <DataTable
              columns={columns}
              data={gapups}
              keyField="ticker"
              sortable={true}
              defaultSort="gap_pct"
              compact
              emptyMessage={`No qualifying gap-ups in the last ${days} days`}
            />
          </Card>

          {loading && (
            <div className="text-center text-xs text-dark-500 mt-2">Refreshing…</div>
          )}

          <div className="h-4" />
        </>
      )}
    </div>
  )
}
