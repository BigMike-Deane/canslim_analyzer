import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, formatRelativeTime, formatDateTime } from '../api'

const PAGE_SIZE = 50

const KIND_META = {
  trade:              { label: 'Trade',        cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
  stop_loss:          { label: 'Stop Loss',    cls: 'bg-red-500/10 text-red-400 border-red-500/30' },
  score_crash:        { label: 'Score Drop',   cls: 'bg-amber-500/10 text-amber-400 border-amber-500/30' },
  breakout:           { label: 'Breakout',     cls: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30' },
  coiled_spring:      { label: 'Coiled Spring',cls: 'bg-violet-500/10 text-violet-400 border-violet-500/30' },
  risk_alert:         { label: 'Risk',         cls: 'bg-orange-500/10 text-orange-400 border-orange-500/30' },
  spy_gate_change:    { label: 'SPY Gate',     cls: 'bg-blue-500/10 text-blue-400 border-blue-500/30' },
  market_turn:        { label: 'Market Turn',  cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' },
  bear_base_update:   { label: 'Bear Bases',   cls: 'bg-slate-500/10 text-slate-300 border-slate-500/30' },
  bear_market_report: { label: 'Bear Report',  cls: 'bg-slate-500/10 text-slate-300 border-slate-500/30' },
  morning_briefing:   { label: 'Briefing',     cls: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30' },
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
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [unreadCount, setUnreadCount] = useState(0)
  const [unreadOnly, setUnreadOnly] = useState(false)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getNotifications({
        unread_only: unreadOnly, limit: PAGE_SIZE, offset,
      })
      setItems(data.items || [])
      setTotal(data.total || 0)
      setUnreadCount(data.unread_count || 0)
    } finally {
      setLoading(false)
    }
  }, [unreadOnly, offset])

  useEffect(() => { load() }, [load])

  // Reset to page 1 when filter toggles
  useEffect(() => { setOffset(0) }, [unreadOnly])

  async function handleMarkRead(n) {
    if (n.read_at) return
    try {
      await api.markNotificationRead(n.id)
      setItems(items.map(i => i.id === n.id ? { ...i, read_at: new Date().toISOString() } : i))
      setUnreadCount(c => Math.max(0, c - 1))
    } catch {}
  }

  async function handleDelete(n, e) {
    e.stopPropagation()
    try {
      await api.deleteNotification(n.id)
      setItems(items.filter(i => i.id !== n.id))
      setTotal(t => Math.max(0, t - 1))
      if (!n.read_at) setUnreadCount(c => Math.max(0, c - 1))
    } catch {}
  }

  async function handleMarkAllRead() {
    try {
      await api.markAllNotificationsRead()
      setUnreadCount(0)
      const now = new Date().toISOString()
      setItems(items.map(i => i.read_at ? i : { ...i, read_at: now }))
    } catch {}
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
        <div className="flex gap-2">
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
          <div className="px-4 py-12 text-center text-xs text-dark-500">Loading…</div>
        )}
        {!loading && items.length === 0 && (
          <div className="px-4 py-12 text-center text-xs text-dark-500">
            {unreadOnly ? 'Nothing unread.' : 'No notifications yet.'}
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
