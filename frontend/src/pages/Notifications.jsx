import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, formatRelativeTime, formatDateTime } from '../api'
import { useToast } from '../components/Toast'
import Spinner from '../components/Spinner'
import useApi from '../hooks/useApi'

const PAGE_SIZE = 50

// Categorical kind colors. Kept emerald/amber/orange/yellow as semantic (gain,
// warning, risk, info). Replaced cool cyan/violet/blue/slate with warm
// equivalents that harmonize with the brand-red theme.
const KIND_META = {
  trade:              { label: 'Trade',        cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
  stop_loss:          { label: 'Stop Loss',    cls: 'bg-red-500/10 text-red-400 border-red-500/30' },
  score_crash:        { label: 'Score Drop',   cls: 'bg-amber-500/10 text-amber-400 border-amber-500/30' },
  breakout:           { label: 'Breakout',     cls: 'bg-primary-500/10 text-primary-400 border-primary-500/30' },
  coiled_spring:      { label: 'Coiled Spring',cls: 'bg-rose-500/10 text-rose-400 border-rose-500/30' },
  risk_alert:         { label: 'Risk',         cls: 'bg-orange-500/10 text-orange-400 border-orange-500/30' },
  spy_gate_change:    { label: 'SPY Gate',     cls: 'bg-accent-500/10 text-accent-500 border-accent-500/30' },
  market_turn:        { label: 'Market Turn',  cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
  bear_base_update:   { label: 'Bear Bases',   cls: 'bg-dark-700/40 text-dark-300 border-dark-600' },
  bear_market_report: { label: 'Bear Report',  cls: 'bg-dark-700/40 text-dark-300 border-dark-600' },
  watchlist:          { label: 'Watchlist',    cls: 'bg-accent-500/10 text-accent-500 border-accent-500/30' },
  system_alarm:       { label: 'System Alarm', cls: 'bg-red-500/10 text-red-400 border-red-500/30' },
}

function KindBadge({ kind }) {
  const meta = KIND_META[kind] || { label: kind, cls: 'bg-dark-700/40 text-dark-300 border-dark-600' }
  return (
    <span className={`text-[9px] font-semibold tracking-wide px-1.5 py-0.5 rounded border ${meta.cls}`}>
      {meta.label.toUpperCase()}
    </span>
  )
}

export default function Notifications() {
  const [unreadOnly, setUnreadOnly] = useState(false)
  const [kindFilter, setKindFilter] = useState('')
  const [offset, setOffset] = useState(0)
  const navigate = useNavigate()
  const toast = useToast()

  // useApi's internal request seq guards the filter/paging races: toggling
  // filters fires overlapping getNotifications calls, and a slower earlier
  // response must not render another filter's items under the one selected
  // last.
  const { data, loading, setData } = useApi(
    () => api.getNotifications({
      unread_only: unreadOnly,
      kind: kindFilter || undefined,
      limit: PAGE_SIZE,
      offset,
    }),
    [unreadOnly, kindFilter, offset]
  )
  const items = data?.items || []
  const total = data?.total || 0
  const unreadCount = data?.unread_count || 0

  // Reset to page 1 when either filter toggles
  useEffect(() => { setOffset(0) }, [unreadOnly, kindFilter])

  // Optimistic updates go through setData's functional form: if the user
  // fires multiple actions quickly, each handler closes over a different
  // snapshot, and `prev =>` reads the latest committed state instead.
  async function handleMarkRead(n) {
    if (n.read_at) return
    try {
      await api.markNotificationRead(n.id)
      const now = new Date().toISOString()
      setData(prev => prev && {
        ...prev,
        items: (prev.items || []).map(i => i.id === n.id ? { ...i, read_at: now } : i),
        unread_count: Math.max(0, (prev.unread_count || 0) - 1),
      })
    } catch (err) {
      toast.error(err?.message || 'Failed to mark as read')
    }
  }

  async function handleDelete(n, e) {
    e.stopPropagation()
    try {
      await api.deleteNotification(n.id)
      setData(prev => prev && {
        ...prev,
        items: (prev.items || []).filter(i => i.id !== n.id),
        total: Math.max(0, (prev.total || 0) - 1),
        unread_count: n.read_at
          ? prev.unread_count
          : Math.max(0, (prev.unread_count || 0) - 1),
      })
    } catch (err) {
      toast.error(err?.message || 'Failed to delete')
    }
  }

  async function handleMarkAllRead() {
    try {
      await api.markAllNotificationsRead()
      const now = new Date().toISOString()
      setData(prev => prev && {
        ...prev,
        unread_count: 0,
        items: (prev.items || []).map(i => i.read_at ? i : { ...i, read_at: now }),
      })
    } catch (err) {
      toast.error(err?.message || 'Failed to mark all as read')
    }
  }

  function handleRowClick(n) {
    handleMarkRead(n)
    const ticker = n?.data?.ticker
    if (ticker) navigate(`/stock/${ticker}`)
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1

  return (
    <div className="px-4 py-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-lg font-semibold text-dark-100">Notifications</h1>
          <p className="text-xs text-dark-500 mt-0.5">
            {unreadCount > 0
              ? `${unreadCount} unread of ${total}`
              : `${total} total`}
          </p>
        </div>
        <div className="flex gap-2 flex-wrap justify-end">
          <select
            value={kindFilter}
            onChange={e => setKindFilter(e.target.value)}
            className="text-[11px] bg-dark-800 text-dark-300 border border-dark-700 rounded-md px-2.5 py-1.5 focus:outline-none focus:border-primary-500/40"
            aria-label="Filter by kind"
          >
            <option value="">All types</option>
            {Object.entries(KIND_META).map(([k, m]) => (
              <option key={k} value={k}>{m.label}</option>
            ))}
          </select>
          <button
            onClick={() => setUnreadOnly(v => !v)}
            className={`text-[11px] px-3 py-1.5 rounded-md border transition-colors ${
              unreadOnly
                ? 'bg-primary-500/10 text-primary-400 border-primary-500/30'
                : 'bg-dark-800 text-dark-300 border-dark-700 hover:border-dark-600'
            }`}
          >
            {unreadOnly ? 'Showing unread' : 'Unread only'}
          </button>
          {unreadCount > 0 && (
            <button
              onClick={handleMarkAllRead}
              className="text-[11px] px-3 py-1.5 rounded-md bg-dark-800 text-dark-300 border border-dark-700 hover:border-dark-600 transition-colors"
            >
              Mark all read
            </button>
          )}
        </div>
      </div>

      <div className="bg-dark-900 border border-dark-700/60 rounded-lg overflow-hidden">
        {loading && (
          <Spinner label="Loading notifications…" />
        )}
        {!loading && items.length === 0 && (
          <div className="px-4 py-12 text-center text-xs text-dark-500">
            {kindFilter
              ? `No ${(KIND_META[kindFilter]?.label || kindFilter).toLowerCase()} notifications${unreadOnly ? ' unread' : ''}.`
              : unreadOnly ? 'Nothing unread.' : 'No notifications yet.'}
          </div>
        )}
        {!loading && items.map(n => (
          <div
            key={n.id}
            onClick={() => handleRowClick(n)}
            className={`px-4 py-3 border-b border-dark-700/40 last:border-b-0 cursor-pointer hover:bg-dark-800/60 transition-colors flex gap-3 ${
              !n.read_at ? 'bg-primary-500/[0.04]' : ''
            }`}
          >
            <div className={`w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0 ${
              !n.read_at
                ? (n.priority === 'urgent' ? 'bg-red-500' : 'bg-primary-500')
                : 'bg-transparent'
            }`} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <KindBadge kind={n.kind} />
                <span className={`text-xs font-semibold ${!n.read_at ? 'text-dark-100' : 'text-dark-300'}`}>
                  {n.title}
                </span>
                <span className="text-[10px] text-dark-500" title={formatDateTime(n.created_at)}>
                  · {formatRelativeTime(n.created_at)}
                </span>
              </div>
              {n.body && (
                <div className="text-xs text-dark-400 mt-1 whitespace-pre-line">
                  {n.body}
                </div>
              )}
            </div>
            <button
              onClick={(e) => handleDelete(n, e)}
              title="Delete"
              className="text-dark-600 hover:text-red-400 transition-colors flex-shrink-0 self-start mt-0.5"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
                <path d="M10 11v6M14 11v6" />
              </svg>
            </button>
          </div>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 mt-4 text-xs text-dark-400">
          <button
            onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            disabled={offset === 0}
            className="px-3 py-1 rounded border border-dark-700 disabled:opacity-30 hover:border-dark-600"
          >
            ← Prev
          </button>
          <span className="text-dark-500">Page {currentPage} of {totalPages}</span>
          <button
            onClick={() => setOffset(offset + PAGE_SIZE)}
            disabled={offset + PAGE_SIZE >= total}
            className="px-3 py-1 rounded border border-dark-700 disabled:opacity-30 hover:border-dark-600"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
