import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatCompactValue, formatMarketCap } from '../api'
import Card from '../components/Card'
import { ScoreBadge, TagBadge } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import DataTable, { Pagination } from '../components/DataTable'

const SENTIMENTS = [
  { key: 'bullish', label: 'Bullish' },
  { key: 'bearish', label: 'Bearish' },
  { key: 'all', label: 'All' },
]

const SORT_OPTIONS = [
  { value: 'insider_net_value', label: 'Net $ Value' },
  { value: 'insider_buy_count', label: 'Buy Count' },
  { value: 'canslim_score', label: 'Score' },
  { value: 'insider_buy_value', label: 'Buy Value' },
]

const PAGE_SIZE = 50

export default function InsiderSentiment() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sentiment, setSentiment] = useState('bullish')
  const [sortBy, setSortBy] = useState('insider_net_value')
  const [sector, setSector] = useState('')
  const [page, setPage] = useState(1)

  useEffect(() => {
    setLoading(true)
    api.getInsiderSentiment({
      sentiment,
      sort_by: sortBy,
      sector: sector || undefined,
      limit: PAGE_SIZE,
      offset: (page - 1) * PAGE_SIZE,
    })
      .then(d => { setData(d); setError(null) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [sentiment, sortBy, sector, page])

  // Reset page when filters change
  useEffect(() => { setPage(1) }, [sentiment, sortBy, sector])

  const summary = data?.summary
  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 1

  const columns = [
    {
      key: 'ticker',
      label: 'Ticker',
      sortable: true,
      render: (_, row) => (
        <Link to={`/stock/${row.ticker}`} className="text-emerald-400 font-medium hover:text-emerald-300">
          {row.ticker}
        </Link>
      ),
    },
    {
      key: 'canslim_score',
      label: 'Score',
      align: 'center',
      sortable: true,
      render: (val) => <ScoreBadge score={val} size="xs" />,
    },
    {
      key: 'current_price',
      label: 'Price',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => val != null ? <span className="font-data text-xs">{formatCurrency(val)}</span> : '-',
    },
    {
      key: 'insider_sentiment',
      label: 'Sentiment',
      align: 'center',
      render: (val) => {
        const color = val === 'bullish' ? 'green' : val === 'bearish' ? 'red' : 'default'
        return val ? <TagBadge color={color}>{val}</TagBadge> : '-'
      },
    },
    {
      key: 'insider_buy_count',
      label: 'Buys',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => val ? <span className="font-data text-xs text-green-400">{val}</span> : <span className="text-dark-500">0</span>,
    },
    {
      key: 'insider_sell_count',
      label: 'Sells',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => val ? <span className="font-data text-xs text-red-400">{val}</span> : <span className="text-dark-500">0</span>,
    },
    {
      key: 'insider_net_value',
      label: 'Net Value',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => {
        if (val == null) return <span className="text-dark-500">-</span>
        const color = val > 0 ? 'text-green-400' : val < 0 ? 'text-red-400' : 'text-dark-400'
        return <span className={`font-data text-xs ${color}`}>{formatCompactValue(val)}</span>
      },
    },
    {
      key: 'insider_largest_buy',
      label: 'Largest Buy',
      align: 'right',
      mono: true,
      sortable: true,
      render: (val) => val ? <span className="font-data text-xs text-emerald-300">{formatCompactValue(val)}</span> : <span className="text-dark-500">-</span>,
    },
    {
      key: 'insider_largest_buyer_title',
      label: 'Buyer Title',
      render: (val) => val ? <span className="text-xs text-dark-300 truncate max-w-[120px] inline-block">{val}</span> : <span className="text-dark-500">-</span>,
    },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Insider Sentiment"
        backTo="/"
        backLabel="Command Center"
        badge={<TagBadge color="green">SMART MONEY</TagBadge>}
        className="[&_h1]:text-emerald-300"
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
          {/* Summary Stats */}
          <Card variant="glass" className="mb-4 !border-emerald-500/20 bg-emerald-500/5">
            <StatGrid
              columns={4}
              stats={[
                { label: 'Bullish Stocks', value: summary?.total_bullish || 0, color: 'text-green-400' },
                { label: 'Net $ Flow', value: formatCompactValue(summary?.net_insider_value), color: summary?.net_insider_value >= 0 ? 'text-green-400' : 'text-red-400' },
                { label: 'Avg Score (Bull)', value: summary?.avg_score_bullish || 0, color: 'text-emerald-300' },
                { label: 'Sectors', value: summary?.sectors?.length || 0, color: 'text-dark-300' },
              ]}
            />
          </Card>

          {/* Filters Row */}
          <div className="flex flex-wrap items-center gap-3 mb-3">
            {/* Sentiment pills */}
            <div className="flex gap-1.5">
              {SENTIMENTS.map(s => (
                <button
                  key={s.key}
                  onClick={() => setSentiment(s.key)}
                  className={`text-xs px-3 py-1.5 rounded-full whitespace-nowrap transition-colors ${
                    sentiment === s.key
                      ? 'bg-emerald-500/30 text-emerald-300 font-medium'
                      : 'bg-dark-700 text-dark-400 hover:bg-dark-600'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* Sort dropdown */}
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              className="text-xs bg-dark-700 text-dark-300 border border-dark-600 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-emerald-500/40"
            >
              {SORT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>

            {/* Sector dropdown */}
            {summary?.sectors?.length > 0 && (
              <select
                value={sector}
                onChange={e => setSector(e.target.value)}
                className="text-xs bg-dark-700 text-dark-300 border border-dark-600 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-emerald-500/40"
              >
                <option value="">All Sectors</option>
                {summary.sectors.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            )}

            {loading && <span className="text-[10px] text-dark-500 animate-pulse">Loading...</span>}
          </div>

          {/* Data Table */}
          <Card variant="glass" padding="">
            <DataTable
              columns={columns}
              data={data.stocks || []}
              keyField="ticker"
              sortable={true}
              defaultSort="insider_net_value"
              compact
              emptyMessage="No stocks with insider activity match your filters"
              onRowClick={(row) => window.location.href = `/stock/${row.ticker}`}
            />
          </Card>

          {/* Pagination */}
          {totalPages > 1 && (
            <Pagination
              page={page}
              pages={totalPages}
              total={data.total}
              label="stocks"
              onPageChange={setPage}
            />
          )}

          {/* Info section */}
          <details className="mt-4">
            <summary className="text-xs text-dark-500 cursor-pointer hover:text-dark-400">
              About Insider Sentiment
            </summary>
            <Card variant="glass" className="mt-2 text-xs text-dark-400 space-y-2">
              <p>
                <strong className="text-dark-300">What is insider trading data?</strong> SEC requires corporate insiders (officers, directors, 10%+ owners) to report their trades within 2 business days via Form 4 filings.
              </p>
              <p>
                <strong className="text-dark-300">Why it matters:</strong> Insiders have unique knowledge of their company. Clusters of insider buying often precede positive stock moves. Insider selling is less informative (could be for taxes, diversification, or personal reasons).
              </p>
              <p>
                <strong className="text-dark-300">How we calculate sentiment:</strong> Bullish = buy count &gt; 1.5x sell count over the last 90 days. Bearish = sell count &gt; 1.5x buy count. Data sourced from FMP + yfinance.
              </p>
            </Card>
          </details>
        </>
      )}

      <div className="h-4" />
    </div>
  )
}
