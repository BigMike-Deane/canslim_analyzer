import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatRelativeTime } from '../api'
import Card from '../components/Card'
import { ScoreBadge, TagBadge } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import PageHeader from '../components/PageHeader'
import DataTable from '../components/DataTable'

function ReadinessScore({ score }) {
  if (score == null) return <span className="text-dark-500 text-xs">-</span>
  const cls = score >= 60
    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
    : score >= 40
      ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
      : 'bg-red-500/20 text-red-300 border-red-500/30'
  return (
    <span className={`text-xs font-data font-medium px-2 py-0.5 rounded border ${cls}`}>
      {Math.round(score)}
    </span>
  )
}

function BoolBadge({ value, label }) {
  if (!value) return <span className="text-dark-600 text-[10px]">--</span>
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded border bg-emerald-500/15 text-emerald-400 border-emerald-500/20">
      {label}
    </span>
  )
}

function ReadinessBreakdown({ factors }) {
  if (!factors) return null
  const items = [
    { key: 'fundamentals', label: 'Fundamentals', max: 25 },
    { key: 'relative_strength', label: 'Relative Strength', max: 20 },
    { key: 'base_quality', label: 'Base Quality', max: 20 },
    { key: 'accumulation', label: 'Accumulation', max: 15 },
    { key: 'industry_group', label: 'Industry Group', max: 10 },
    { key: 'pivot_proximity', label: 'Pivot Proximity', max: 10 },
  ]

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 px-4 py-3 bg-dark-850/50 border-t border-dark-700/30">
      {items.map(item => {
        const val = factors[item.key]
        if (val == null) return null
        const pct = (val / item.max) * 100
        const barColor = pct >= 60 ? 'bg-emerald-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
        return (
          <div key={item.key} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-dark-400">{item.label}</span>
              <span className="text-[10px] font-data text-dark-200">{val}/{item.max}</span>
            </div>
            <div className="h-1 bg-dark-700 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function MarketBanner({ isBear, spyPrice, spyMa50 }) {
  if (isBear) {
    return (
      <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-red-500/20 flex items-center justify-center shrink-0">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-red-400">
            <polyline points="22 17 13.5 8.5 8.5 13.5 2 7" />
            <polyline points="16 17 22 17 22 11" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-red-300">BEAR MARKET - SPY below 50MA</div>
          <div className="text-[10px] text-red-400/70 font-data mt-0.5">
            SPY ${spyPrice?.toFixed(2) || '--'} | 50MA ${spyMa50?.toFixed(2) || '--'}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center shrink-0">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-amber-400">
          <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
          <polyline points="16 7 22 7 22 13" />
        </svg>
      </div>
      <div>
        <div className="text-sm font-semibold text-amber-300">Market bullish — this list shows stocks that were building bases</div>
        <div className="text-[10px] text-amber-400/70 font-data mt-0.5">
          SPY ${spyPrice?.toFixed(2) || '--'} | 50MA ${spyMa50?.toFixed(2) || '--'}
        </div>
      </div>
    </div>
  )
}

export default function BearBase() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedTicker, setExpandedTicker] = useState(null)

  useEffect(() => {
    setLoading(true)
    api.getBearBaseCandidates(50)
      .then(d => { setData(d?.data || d); setError(null) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const candidates = data?.candidates || []

  // Compute sector breakdown from candidates
  const sectorCounts = {}
  candidates.forEach(c => {
    const s = c.sector || 'Unknown'
    sectorCounts[s] = (sectorCounts[s] || 0) + 1
  })
  const topSectors = Object.entries(sectorCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)

  const avgReadiness = candidates.length > 0
    ? (candidates.reduce((sum, c) => sum + (c.readiness_score || 0), 0) / candidates.length).toFixed(0)
    : 0

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
      key: 'readiness_score',
      label: 'Readiness',
      align: 'center',
      sortable: true,
      render: (val) => <ReadinessScore score={val} />,
    },
    {
      key: 'canslim_score',
      label: 'CANSLIM',
      align: 'center',
      sortable: true,
      render: (val, row) => <ScoreBadge score={val} ticker={row.ticker} size="sm" />,
    },
    {
      key: 'base_type',
      label: 'Base',
      sortable: true,
      render: (val) => val ? (
        <TagBadge color="cyan">
          {val === 'cup_with_handle' ? 'cup+handle' : val.replace(/_/g, ' ')}
        </TagBadge>
      ) : <span className="text-dark-500 text-[10px]">--</span>,
    },
    {
      key: 'weeks_in_base',
      label: 'Weeks',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => <span className="font-data text-xs text-dark-200">{val ?? '-'}</span>,
    },
    {
      key: 'rs_3m',
      label: 'RS 3m',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => {
        if (val == null) return <span className="text-dark-500 text-xs">-</span>
        const color = val >= 0 ? 'text-emerald-400' : 'text-red-400'
        return <span className={`font-data text-xs ${color}`}>{val > 0 ? '+' : ''}{val.toFixed(1)}%</span>
      },
    },
    {
      key: 'industry_group_rank',
      label: 'Ind. Rank',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => val != null
        ? <span className={`font-data text-xs ${val <= 50 ? 'text-emerald-400' : val <= 100 ? 'text-amber-400' : 'text-dark-400'}`}>#{val}</span>
        : <span className="text-dark-500 text-xs">-</span>,
    },
    {
      key: 'volume_dry_up',
      label: 'Vol Dry-Up',
      align: 'center',
      mobileHide: true,
      render: (val) => <BoolBadge value={val} label="DRY-UP" />,
    },
    {
      key: 'institutional_acc',
      label: 'Inst. Acc.',
      align: 'center',
      mobileHide: true,
      render: (val) => <BoolBadge value={val} label="ACC" />,
    },
    {
      key: 'days_on_list',
      label: 'Days',
      align: 'right',
      mono: true,
      sortable: true,
      mobileHide: true,
      render: (val) => <span className="font-data text-xs text-dark-300">{val ?? '-'}</span>,
    },
  ]

  return (
    <div className="p-4 md:p-6">
      <PageHeader
        title="Bear Base Watchlist"
        backTo="/"
        backLabel="Command Center"
        subtitle={
          data?.as_of
            ? `Stocks building sound bases during market weakness — Updated ${formatRelativeTime(data.as_of)}`
            : 'Stocks building sound bases during market weakness — first to break out on recovery'
        }
        badge={
          <TagBadge color="red">BASE BUILDING</TagBadge>
        }
      />

      {loading && !data && (
        <div className="space-y-3">
          <div className="skeleton h-16 rounded-2xl" />
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
          <MarketBanner
            isBear={data.is_bear_market}
            spyPrice={data.spy_price}
            spyMa50={data.spy_ma50}
          />

          {/* Summary stats */}
          <Card variant="glass" className="mb-4 !border-red-500/20 bg-red-500/5">
            <StatGrid
              columns={4}
              stats={[
                { label: 'Candidates', value: data.total || candidates.length, color: 'text-dark-50' },
                { label: 'Avg Readiness', value: avgReadiness, color: Number(avgReadiness) >= 60 ? 'text-emerald-400' : Number(avgReadiness) >= 40 ? 'text-amber-400' : 'text-red-400' },
                { label: 'Top Sector', value: topSectors[0]?.[0] || '-', color: 'text-primary-400' },
                { label: 'Sector Count', value: topSectors[0]?.[1] || '-', color: 'text-dark-200' },
              ]}
            />
          </Card>

          {/* Sector breakdown pills */}
          {topSectors.length > 1 && (
            <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
              {topSectors.map(([sector, count]) => (
                <span
                  key={sector}
                  className="text-[10px] px-2.5 py-1.5 rounded-full bg-dark-700 text-dark-300 whitespace-nowrap"
                >
                  {sector} ({count})
                </span>
              ))}
            </div>
          )}

          {/* Candidates table */}
          <Card variant="glass" padding="">
            <DataTable
              columns={columns}
              data={candidates}
              keyField="ticker"
              sortable={true}
              defaultSort="readiness_score"
              compact
              emptyMessage="No base-building candidates found"
              onRowClick={(row) =>
                setExpandedTicker(expandedTicker === row.ticker ? null : row.ticker)
              }
            />
            {/* Expanded readiness factors (rendered outside DataTable rows) */}
            {expandedTicker && (() => {
              const row = candidates.find(c => c.ticker === expandedTicker)
              if (!row?.readiness_factors) return null
              return (
                <div className="border-t border-dark-700/30">
                  <div className="px-4 pt-2 pb-1">
                    <span className="text-[10px] font-semibold tracking-wider uppercase text-dark-400">
                      {expandedTicker} Readiness Breakdown
                    </span>
                  </div>
                  <ReadinessBreakdown factors={row.readiness_factors} />
                </div>
              )
            })()}
          </Card>

          <div className="h-4" />
        </>
      )}
    </div>
  )
}
