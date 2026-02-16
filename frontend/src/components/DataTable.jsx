import { useState } from 'react'

export default function DataTable({
  columns,
  data,
  keyField = 'id',
  onRowClick,
  emptyMessage = 'No data available',
  sortable = true,
  defaultSort,
  defaultSortDir = 'desc',
  compact = false,
  className = '',
}) {
  const [sortCol, setSortCol] = useState(defaultSort || null)
  const [sortDir, setSortDir] = useState(defaultSortDir)

  const handleSort = (col) => {
    if (!sortable || !col.sortable) return
    const key = col.sortKey || col.key
    if (sortCol === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(key)
      setSortDir('desc')
    }
  }

  const sortedData = [...(data || [])]
  if (sortCol) {
    sortedData.sort((a, b) => {
      const col = columns.find(c => (c.sortKey || c.key) === sortCol)
      const aVal = col?.sortValue ? col.sortValue(a) : a[sortCol]
      const bVal = col?.sortValue ? col.sortValue(b) : b[sortCol]
      if (aVal == null && bVal == null) return 0
      if (aVal == null) return 1
      if (bVal == null) return -1
      if (typeof aVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortDir === 'asc' ? aVal - bVal : bVal - aVal
    })
  }

  const py = compact ? 'py-1.5' : 'py-2.5'

  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-dark-700/50">
            {columns.map(col => (
              <th
                key={col.key}
                className={`${py} px-2 text-[10px] font-semibold tracking-wider uppercase text-dark-400 ${
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                } ${sortable && col.sortable ? 'cursor-pointer select-none hover:text-dark-200 transition-colors' : ''}`}
                onClick={() => handleSort(col)}
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {sortable && col.sortable && sortCol === (col.sortKey || col.key) && (
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" className="text-primary-500">
                      {sortDir === 'asc'
                        ? <path d="M5 2L9 7H1L5 2Z" />
                        : <path d="M5 8L1 3H9L5 8Z" />
                      }
                    </svg>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="text-center py-8 text-dark-500 text-xs">
                {emptyMessage}
              </td>
            </tr>
          )}
          {sortedData.map((row, i) => (
            <tr
              key={row[keyField] ?? i}
              className={`border-b border-dark-700/20 last:border-0 transition-colors ${
                onRowClick ? 'cursor-pointer hover:bg-dark-750' : 'hover:bg-dark-800/40'
              } ${i % 2 === 1 ? 'bg-dark-850/30' : ''}`}
              onClick={() => onRowClick?.(row)}
            >
              {columns.map(col => (
                <td
                  key={col.key}
                  className={`${py} px-2 ${
                    col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                  } ${col.mono ? 'font-data' : ''} ${col.className || ''}`}
                >
                  {col.render ? col.render(row[col.key], row) : (row[col.key] ?? '-')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Pagination({ page, pages, total, onPageChange, label = 'items' }) {
  if (!pages || pages <= 1) return null

  return (
    <div className="flex items-center justify-center gap-3 mt-4">
      <button
        onClick={() => onPageChange(Math.max(1, page - 1))}
        disabled={page <= 1}
        className="text-xs px-3 py-1.5 rounded-lg bg-dark-700 text-dark-300 hover:bg-dark-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        Prev
      </button>
      <span className="text-[10px] text-dark-400 font-data">
        {page} / {pages}
        {total != null && ` (${total} ${label})`}
      </span>
      <button
        onClick={() => onPageChange(Math.min(pages, page + 1))}
        disabled={page >= pages}
        className="text-xs px-3 py-1.5 rounded-lg bg-dark-700 text-dark-300 hover:bg-dark-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        Next
      </button>
    </div>
  )
}
