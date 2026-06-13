import { useState, useEffect, useMemo } from 'react'
import { api, formatRelativeTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import PageHeader from '../components/PageHeader'
import DataTable from '../components/DataTable'

function RankBadge({ rank }) {
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-data font-semibold border ${getRankColor(rank)}`}>
      #{rank}
    </span>
  )
}

function getRankColor(rank) {
  if (rank >= 80) return 'text-emerald-400 bg-emerald-500/20 border-emerald-500/30'
  if (rank >= 60) return 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20'
  if (rank >= 40) return 'text-amber-400 bg-amber-500/15 border-amber-500/25'
  if (rank >= 20) return 'text-orange-400 bg-orange-500/15 border-orange-500/25'
  return 'text-red-400 bg-red-500/15 border-red-500/25'
}

function RotationEntry({ group, type }) {
  const isImproving = type === 'improving'
  const rsDiff = group.avg_rs_3m != null && group.avg_rs_12m != null
    ? (group.avg_rs_3m - group.avg_rs_12m).toFixed(2)
    : null

  return (
    <div className="flex items-center justify-between py-2.5 border-b border-dark-700/30 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm text-dark-100 font-medium truncate">{group.industry}</span>
          <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-data font-semibold border ${getRankColor(group.rank)}`}>
            #{group.rank}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-0.5">
          <span className="text-[10px] text-dark-500">{group.stock_count} stocks</span>
          <span className="text-[10px] text-dark-500 font-data">RS: {group.composite_rs?.toFixed(2) || '-'}</span>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {rsDiff !== null && (
          <span className={`text-sm font-data font-medium flex items-center gap-0.5 ${isImproving ? 'text-emerald-400' : 'text-red-400'}`}>
            {isImproving ? (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="18 15 12 9 6 15" />
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            )}
            {rsDiff}
          </span>
        )}
      </div>
    </div>
  )
}

export default function SectorRotation() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await api.getIndustryGroups()
        setData(result.data || result)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch industry groups:', err)
        setError(err.message || 'Failed to load industry groups')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const allGroups = data?.groups || []

  // Search filters; DataTable owns sorting (rank desc by default).
  const filteredGroups = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return allGroups
    return allGroups.filter(g => (g.industry || '').toLowerCase().includes(q))
  }, [allGroups, search])

  if (loading) {
    return (
      <div className="p-4 md:p-6 max-w-5xl mx-auto">
        <PageHeader title="Sector Rotation" subtitle="Industry group relative strength" />
        <div className="space-y-3">
          <div className="skeleton h-24 rounded-2xl" />
          <div className="skeleton h-64 rounded-2xl" />
        </div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="p-4 md:p-6 max-w-5xl mx-auto">
        <PageHeader title="Sector Rotation" subtitle="Industry group relative strength" />
        <Card variant="glass" className="text-center py-8">
          <div className="text-red-400 text-sm">{error || 'No data available'}</div>
        </Card>
      </div>
    )
  }

  const { total = 0, rotation = {} } = data
  const improving = rotation.improving || []
  const deteriorating = rotation.deteriorating || []

  return (
    <div className="p-4 md:p-6 max-w-5xl mx-auto">
      <PageHeader
        title="Sector Rotation"
        subtitle={data.as_of
          ? `${total} industry groups tracked · Updated ${formatRelativeTime(data.as_of)}`
          : `${total} industry groups tracked`}
      />

      {/* Summary stat */}
      <Card variant="glass" className="mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs text-dark-400">Total Groups Tracked</div>
            <div className="text-2xl font-data font-bold text-dark-100">{total}</div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-xs text-dark-400">Improving</div>
              <div className="text-lg font-data font-semibold text-emerald-400">{improving.length}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-dark-400">Deteriorating</div>
              <div className="text-lg font-data font-semibold text-red-400">{deteriorating.length}</div>
            </div>
          </div>
        </div>
      </Card>

      {/* Two-column rotation layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {/* Improving */}
        <Card variant="glass">
          <CardHeader
            title="Improving Groups"
            subtitle={`${improving.length} groups gaining strength`}
          />
          {improving.length === 0 ? (
            <div className="text-dark-500 text-sm text-center py-4">No improving groups</div>
          ) : (
            <div className="space-y-0">
              {improving.map(g => (
                <RotationEntry key={g.industry} group={g} type="improving" />
              ))}
            </div>
          )}
        </Card>

        {/* Deteriorating */}
        <Card variant="glass">
          <CardHeader
            title="Deteriorating Groups"
            subtitle={`${deteriorating.length} groups losing strength`}
          />
          {deteriorating.length === 0 ? (
            <div className="text-dark-500 text-sm text-center py-4">No deteriorating groups</div>
          ) : (
            <div className="space-y-0">
              {deteriorating.map(g => (
                <RotationEntry key={g.industry} group={g} type="deteriorating" />
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Full groups table */}
      <SectionLabel>All Industry Groups · {allGroups.length}</SectionLabel>
      <div className="mb-3 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div className="relative flex-1 sm:max-w-xs">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 text-dark-500 pointer-events-none"
            width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search industry groups..."
            className="w-full !pl-9"
          />
        </div>
        <div className="text-[10px] text-dark-500 font-data shrink-0">
          Showing {filteredGroups.length} of {allGroups.length}
        </div>
      </div>
      <Card variant="glass" padding="px-2 py-1">
        <DataTable
          compact
          keyField="industry"
          data={filteredGroups}
          defaultSort="rank"
          defaultSortDir="desc"
          emptyMessage={search ? `No industry groups match "${search}"` : 'No industry groups'}
          columns={[
            { key: 'industry', label: 'Industry', sortable: true, className: 'text-dark-100 font-medium' },
            { key: 'rank', label: 'Rank', sortable: true, render: v => <RankBadge rank={v} /> },
            { key: 'composite_rs', label: 'Composite RS', sortable: true, mono: true, className: 'text-dark-200', render: v => v?.toFixed(2) || '-' },
            { key: 'stock_count', label: 'Stocks', sortable: true, mono: true, className: 'text-dark-300' },
            { key: 'avg_rs_12m', label: 'RS 12m', mono: true, mobileHide: true, className: 'text-dark-300', render: v => v?.toFixed(2) || '-' },
            { key: 'avg_rs_3m', label: 'RS 3m', mono: true, mobileHide: true, className: 'text-dark-300', render: v => v?.toFixed(2) || '-' },
          ]}
        />
      </Card>

      <div className="h-4" />
    </div>
  )
}
