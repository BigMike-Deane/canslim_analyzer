import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, formatRelativeTime } from '../api'

const POLL_INTERVAL_MS = 30_000

/**
 * Sidebar bell with unread badge + click-to-open dropdown of recent notifications.
 *
 * Polls /unread-count every 30s while mounted. Opens a dropdown that fetches
 * the 10 most recent notifications. Each row links to /stock/<ticker> when
 * the notification carries a ticker; the row is marked read on click.
 */
export default function NotificationBell({ collapsed }) {
  const [unreadCount, setUnreadCount] = useState(0)
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const ref = useRef(null)

  // Poll unread count while mounted (mounts once, runs forever)
  useEffect(() => {
    let cancelled = false
    async function tick() {
      try {
        const data = await api.getUnreadNotificationCount()
        if (!cancelled) setUnreadCount(data.count || 0)
      } catch {
        // silently ignore — bell is non-critical UI
      }
    }
    tick()
    const id = setInterval(tick, POLL_INTERVAL_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Fetch recent items every time the dropdown opens
  useEffect(() => {
    if (!open) return
    let cancelled = false
    setLoading(true)
    api.getNotifications({ limit: 10 })
      .then(data => {
        if (cancelled) return
        setItems(data.items || [])
        setUnreadCount(data.unread_count || 0)
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [open])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  async function handleRowClick(n) {
    setOpen(false)
    if (!n.read_at) {
      try {
        await api.markNotificationRead(n.id)
        setUnreadCount(c => Math.max(0, c - 1))
      } catch {}
    }
    const ticker = n?.data?.ticker
    if (ticker) navigate(`/stock/${ticker}`)
    else navigate('/notifications')
  }

  async function handleMarkAllRead(e) {
    e.stopPropagation()
    try {
      await api.markAllNotificationsRead()
      setUnreadCount(0)
      setItems(items.map(i => i.read_at ? i : { ...i, read_at: new Date().toISOString() }))
    } catch {}
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        title="Notifications"
        className={`flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs font-medium transition-all duration-150 w-full ${
          open
            ? 'text-primary-400 bg-primary-500/[0.08]'
            : 'text-dark-400 hover:text-dark-200 hover:bg-dark-800/50'
        } ${collapsed ? 'justify-center' : ''}`}
      >
        <span className="relative inline-flex">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 8a6 6 0 00-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
            <path d="M13.73 21a2 2 0 01-3.46 0" />
          </svg>
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1.5 min-w-[14px] h-[14px] px-1 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center leading-none">
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </span>
        {!collapsed && <span>Notifications</span>}
      </button>

      {open && (
        <div className={`absolute z-50 ${collapsed ? 'left-full ml-2 top-0' : 'left-0 right-0 mt-1'} w-80 bg-dark-900 border border-dark-700 rounded-lg shadow-2xl overflow-hidden`}>
          <div className="flex items-center justify-between px-3 py-2 border-b border-dark-700/60">
            <span className="text-[11px] font-semibold tracking-wide text-dark-200">
              NOTIFICATIONS {unreadCount > 0 && <span className="text-primary-400">({unreadCount})</span>}
            </span>
            {unreadCount > 0 && (
              <button
                onClick={handleMarkAllRead}
                className="text-[10px] text-dark-400 hover:text-primary-400 transition-colors"
              >
                Mark all read
              </button>
            )}
          </div>

          <div className="max-h-96 overflow-y-auto">
            {loading && (
              <div className="px-3 py-6 text-center text-[11px] text-dark-500">Loading…</div>
            )}
            {!loading && items.length === 0 && (
              <div className="px-3 py-8 text-center text-[11px] text-dark-500">
                No notifications yet
              </div>
            )}
            {!loading && items.map(n => (
              <button
                key={n.id}
                onClick={() => handleRowClick(n)}
                className={`w-full text-left px-3 py-2.5 border-b border-dark-700/40 hover:bg-dark-800/60 transition-colors flex gap-2 ${
                  !n.read_at ? 'bg-primary-500/[0.04]' : ''
                }`}
              >
                <div className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${
                  !n.read_at
                    ? (n.priority === 'urgent' ? 'bg-red-500' : 'bg-primary-500')
                    : 'bg-transparent'
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <span className={`text-[11px] font-semibold truncate ${
                      !n.read_at ? 'text-dark-100' : 'text-dark-300'
                    }`}>
                      {n.title}
                    </span>
                    <span className="text-[9px] text-dark-500 flex-shrink-0 mt-0.5">
                      {formatRelativeTime(n.created_at)}
                    </span>
                  </div>
                  {n.body && (
                    <div className="text-[10px] text-dark-400 line-clamp-2 mt-0.5 whitespace-pre-line">
                      {n.body}
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>

          <div className="border-t border-dark-700/60">
            <button
              onClick={() => { setOpen(false); navigate('/notifications') }}
              className="w-full px-3 py-2 text-[11px] text-dark-300 hover:text-primary-400 hover:bg-dark-800/40 transition-colors"
            >
              View all →
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
