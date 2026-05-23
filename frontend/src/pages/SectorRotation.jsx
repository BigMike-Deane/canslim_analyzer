import { useState, useEffect } from 'react'
import { api, formatRelativeTime } from '../api'
import Card, { CardHeader, SectionLabel } from '../components/Card'
import PageHeader from '../components/PageHeader'

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

function SortableHeader({ label, sortKey, currentSort, currentDir, onSort }) {
  const isActive = currentSort === sortKey
  return (
    <th
      className="px-3 py-2 text-xs text-dark-400 font-medium text-left cursor-pointer hover:text-dark-200 transition-colors select-none"
      onClick={() => onSort(sortKey)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {isActive && (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="text-primary-400">
            {currentDir === 'desc'
              ? <polyline points="6 9 12 15 18 9" />
              : <polyline points="18 15 12 9 6 15" />
            }
          </svg>
        )}
      </span>
    </th>
  )
}

export default function SectorRotation() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sortBy, setSortBy] = useState('rank')
  const [sortDir, setSortDir] = useState('desc')

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

  const handleSort = (key) => {
    if (sortBy === key) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(key)
      setSortDir('desc')
    }
  }

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

  const { groups = [], total = 0, rotation = {} } = data
  const improving = rotation.improving || []
  const deteriorating = rotation.deteriorating || []

  const sortedGroups = [...groups].sort((a, b) => {
    const aVal = a[sortBy] ?? 0
    const bVal = b[sortBy] ?? 0
    if (typeof aVal === 'string') {
      return sortDir === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal)
    }
    return sortDir === 'desc' ? bVal - aVal : aVal - bVal
  })

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
      <SectionLabel>All Industry Groups · {sortedGroups.length}</SectionLabel>
      <Card variant="glass" padding="">
        <div className="text-[10px] text-dark-500 px-3 py-2 border-b border-dark-700/30 flex items-center gap-1.5 sm:hidden">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
            <polyline points="9 18 15 12 9 6" transform="translate(8 0)" />
          </svg>
          Swipe to see more columns
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-dark-700/50">
                <SortableHeader label="Industry" sortKey="industry" currentSort={sortBy} currentDir={sortDir} onSort={handleSort} />
                <SortableHeader label="Rank" sortKey="rank" currentSort={sortBy} currentDir={sortDir} onSort={handleSort} />
                <SortableHeader label="Composite RS" sortKey="composite_rs" currentSort={sortBy} currentDir={sortDir} onSort={handleSort} />
                <SortableHeader label="Stocks" sortKey="stock_count" currentSort={sortBy} currentDir={sortDir} onSort={handleSort} />
                <th className="px-3 py-2 text-xs text-dark-400 font-medium text-left">RS 12m</th>
                <th className="px-3 py-2 text-xs text-dark-400 font-medium text-left">RS 3m</th>
              </tr>
            </thead>
            <tbody>
              {sortedGroups.map(g => (
                <tr key={g.industry} className="border-b border-dark-700/20 hover:bg-dark-700/20 transition-colors">
                  <td className="px-3 py-2.5 text-sm text-dark-100 font-medium">{g.industry}</td>
                  <td className="px-3 py-2.5">
                    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-data font-semibold border ${getRankColor(g.rank)}`}>
                      #{g.rank}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-sm font-data text-dark-200">{g.composite_rs?.toFixed(2) || '-'}</td>
                  <td className="px-3 py-2.5 text-sm font-data text-dark-300">{g.stock_count}</td>
                  <td className="px-3 py-2.5 text-sm font-data text-dark-300">{g.avg_rs_12m?.toFixed(2) || '-'}</td>
                  <td className="px-3 py-2.5 text-sm font-data text-dark-300">{g.avg_rs_3m?.toFixed(2) || '-'}</td>
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
