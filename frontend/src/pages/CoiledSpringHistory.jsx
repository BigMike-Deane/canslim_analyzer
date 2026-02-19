import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatShortDate } from '../api'
import Card from '../components/Card'
import { OutcomeBadge, TagBadge, PnlText } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import DataTable, { Pagination } from '../components/DataTable'

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'win', label: 'Winners' },
  { key: 'loss', label: 'Losers' },
  { key: 'flat', label: 'Flat' },
  { key: 'pending', label: 'Pending' },
]

export default function CoiledSpringHistory() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [page, setPage] = useState(1)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    setLoading(true)
    api.getCoiledSpringHistory(page, 50)
      .then(d => { setData(d); setError(null) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [page])

  const stats = data?.cumulative_stats
  const pagination = data?.pagination

  // Client-side filter on the current page of alerts
  const filteredAlerts = (data?.alerts || []).filter(a => {
    if (filter === 'all') return true
    if (filter === 'win') return a.outcome === 'win' || a.outcome === 'big_win'
    if (filter === 'loss') return a.outcome === 'loss'
    if (filter === 'flat') return a.outcome === 'flat'
    if (filter === 'pending') return !a.outcome
    return true
  })

  const columns = [
    {
      key: 'ticker',
      label: 'Ticker',
      sortable: true,
      render: (_, row) => (
        <Link to={`/stock/${row.ticker}`} className="text-purple-400 font-medium hover:text-purple-300">
          {row.ticker}
        </Link>
      ),
    },
    {
      key: 'alert_date',
      label: 'Date',
      sortable: true,
      render: (val) => val ? formatShortDate(val + 'T00:00:00') : '-',
    },
    {
      key: 'base_type',
      label: 'Base',
      render: (val) =>
        val ? <TagBadge>{val}</TagBadge> : null,
    },
    {
      key: 'price_at_alert',
      label: 'Entry',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) =>
        val != null ? <span className="font-data text-xs">${val.toFixed(2)}</span> : '-',
    },
    {
      key: 'price_after_earnings',
      label: 'Post-ER',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) =>
        val != null ? <span className="font-data text-xs">${val.toFixed(2)}</span> : '-',
    },
    {
      key: 'price_change_pct',
      label: 'Change',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) =>
        val != null ? <PnlText value={val} className="text-xs" /> : <span className="text-dark-500">-</span>,
    },
    {
      key: 'outcome',
      label: 'Result',
      align: 'center',
      render: (val) => <OutcomeBadge outcome={val} />,
    },
    {
      key: 'beat_streak',
      label: 'Beats',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => <span className="font-data text-xs text-dark-400">{val || '-'}</span>,
    },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Coiled Spring History"
        backTo="/"
        backLabel="Command Center"
        badge={
          <TagBadge color="purple">EARNINGS CATALYST</TagBadge>
        }
        className="[&_h1]:text-purple-300"
      />

      {loading && !data && (
        <div className="space-y-3">
          <div className="skeleton h-24 rounded-2xl" />
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
          {/* Stats card */}
          <Card variant="glass" className="mb-4 !border-purple-500/20 bg-purple-500/5">
            <StatGrid
              columns={5}
              stats={[
                { label: 'Win Rate', value: `${stats?.overall_win_rate || 0}%`, color: 'text-purple-300' },
                { label: 'Wins', value: stats?.wins || 0, color: 'text-green-400' },
                { label: 'Big Wins', value: stats?.big_wins || 0, color: 'text-emerald-300' },
                { label: 'Losses', value: stats?.losses || 0, color: 'text-red-400' },
                { label: 'Flat', value: stats?.flat || 0, color: 'text-yellow-400' },
              ]}
            />
          </Card>

          {/* Filters */}
          <div className="flex gap-2 mb-3 overflow-x-auto">
            {FILTERS.map(f => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className={`text-xs px-3 py-1.5 rounded-full whitespace-nowrap transition-colors ${
                  filter === f.key
                    ? 'bg-purple-500/30 text-purple-300 font-medium'
                    : 'bg-dark-700 text-dark-400 hover:bg-dark-600'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Alerts table */}
          <Card variant="glass" padding="">
            <DataTable
              columns={columns}
              data={filteredAlerts}
              keyField="id"
              sortable={true}
              defaultSort="alert_date"
              compact
              emptyMessage="No alerts match this filter"
            />
          </Card>

          {/* Pagination */}
          <Pagination
            page={pagination?.page || page}
            pages={pagination?.pages || 1}
            total={pagination?.total}
            label="alerts"
            onPageChange={setPage}
          />
        </>
      )}

      <div className="h-4" />
    </div>
  )
}
