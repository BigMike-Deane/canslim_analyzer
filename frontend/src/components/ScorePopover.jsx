import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api, getScoreClass } from '../api'

/**
 * Interactive CANSLIM score badge: looks identical to <ScoreBadge> but, when a
 * `ticker` is supplied, becomes a button that opens a popover with the 7
 * C-A-N-S-L-I-M component bars. Lets the user read WHY a stock scores what it
 * does from any list/table without navigating to Stock Detail.
 *
 * Data comes from api.getStock(ticker) on first open. That endpoint is a plain
 * DB read cached for 600s in api.js, so the second tap on any ticker (and any
 * page that already loaded its Stock Detail) is instant. Callers that already
 * hold the full row (with c_score..m_score) can pass `details` to skip the
 * fetch entirely — but only when ALL seven fields are present, else the missing
 * bars render empty.
 *
 * Portal + anchor + outside-click pattern mirrors NotificationBell.jsx so the
 * panel escapes overflow-hidden tables/cards and follows the trigger on
 * scroll/resize.
 */

// Single source for the breakdown — letters, source field, and max points.
// Maxes mirror the scorer: C/A/N/S/L/M = 15, I = 10 (totals 100).
const COMPONENTS = [
  { key: 'C', field: 'c_score', max: 15, label: 'Current Earnings' },
  { key: 'A', field: 'a_score', max: 15, label: 'Annual Earnings' },
  { key: 'N', field: 'n_score', max: 15, label: 'New Highs / Products' },
  { key: 'S', field: 's_score', max: 15, label: 'Supply / Demand' },
  { key: 'L', field: 'l_score', max: 15, label: 'Leader / Laggard' },
  { key: 'I', field: 'i_score', max: 10, label: 'Institutional Sponsorship' },
  { key: 'M', field: 'm_score', max: 15, label: 'Market Direction' },
]

const SIZES = {
  xs: 'text-[11px] px-1.5 py-0.5',
  sm: 'text-xs px-2 py-0.5',
  md: 'text-sm px-2.5 py-1',
  lg: 'text-base px-3 py-1',
}

// Tiers + hues mirror getScoreClass / StockDetail's CANSLIMDetail so the bars
// agree with the gauge and badges (and keep mid-scores off brand amber).
function barColorFor(normalized) {
  return normalized >= 80 ? 'bg-emerald-500'
    : normalized >= 65 ? 'bg-green-500'
    : normalized >= 50 ? 'bg-stone-400'
    : normalized >= 35 ? 'bg-rose-500'
    : 'bg-red-500'
}

const PANEL_WIDTH = 240   // px — keep in sync with the w-60 panel class
const GAP = 6

export default function ScorePopover({ score, ticker, size = 'sm', className = '', details = null }) {
  const [open, setOpen] = useState(false)
  const [anchor, setAnchor] = useState(null)   // {top|bottom, left} fixed coords
  const [data, setData] = useState(details)     // object with c_score..m_score
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)
  const triggerRef = useRef(null)
  const panelRef = useRef(null)

  const cls = getScoreClass(score)
  const display = score != null ? Math.round(score) : '-'

  // Lazily fetch component scores the first time the popover opens.
  useEffect(() => {
    if (!open || data) return
    let cancelled = false
    setLoading(true)
    setError(false)
    api.getStock(ticker)
      .then(d => { if (!cancelled) setData(d) })
      .catch(() => { if (!cancelled) setError(true) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [open, data, ticker])

  // Position the portaled panel relative to the trigger; recompute on
  // open/resize/scroll. Prefer below the badge; flip above when near the
  // viewport bottom so the bars stay on-screen.
  useEffect(() => {
    if (!open) { setAnchor(null); return }
    function compute() {
      if (!triggerRef.current) return
      const rect = triggerRef.current.getBoundingClientRect()
      const left = Math.max(GAP, Math.min(rect.left, window.innerWidth - PANEL_WIDTH - GAP))
      const flipUp = window.innerHeight - rect.bottom < 260 && rect.top > 260
      if (flipUp) {
        setAnchor({ bottom: window.innerHeight - rect.top + GAP, left })
      } else {
        setAnchor({ top: rect.bottom + GAP, left })
      }
    }
    compute()
    window.addEventListener('resize', compute)
    window.addEventListener('scroll', compute, true)
    return () => {
      window.removeEventListener('resize', compute)
      window.removeEventListener('scroll', compute, true)
    }
  }, [open])

  // Close on outside click + Escape — check both trigger and portaled panel.
  useEffect(() => {
    if (!open) return
    function onClick(e) {
      const inTrigger = triggerRef.current && triggerRef.current.contains(e.target)
      const inPanel = panelRef.current && panelRef.current.contains(e.target)
      if (!inTrigger && !inPanel) setOpen(false)
    }
    function onKey(e) { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <>
      {/* A <span role="button"> rather than a real <button>: many call sites
          render the badge inside a row-level <Link>/<a>, and a nested <button>
          is invalid HTML the browser would hoist out, breaking layout. A span
          is valid phrasing content inside an anchor. stopPropagation +
          preventDefault keep the enclosing link/row from also firing. */}
      <span
        ref={triggerRef}
        role="button"
        tabIndex={0}
        onClick={(e) => { e.stopPropagation(); e.preventDefault(); setOpen(o => !o) }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.stopPropagation(); e.preventDefault(); setOpen(o => !o)
          }
        }}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Show CANSLIM breakdown"
        className={`inline-block font-data font-bold rounded-md cursor-pointer transition-all hover:ring-1 hover:ring-primary-500/40 ${
          open ? 'ring-1 ring-primary-500/60' : ''
        } ${cls} ${SIZES[size] || SIZES.sm} ${className}`}
      >
        {display}
      </span>

      {open && anchor && createPortal(
        <div
          ref={panelRef}
          role="dialog"
          aria-label={`${ticker} CANSLIM breakdown`}
          style={{ top: anchor.top, bottom: anchor.bottom, left: anchor.left, width: PANEL_WIDTH }}
          className="fixed z-[100] bg-dark-900 border border-dark-700 rounded-lg shadow-2xl p-3"
        >
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[11px] font-semibold tracking-wide text-dark-200">{ticker}</span>
            <span className={`text-xs font-data font-bold rounded px-1.5 py-0.5 ${cls}`}>{display}</span>
          </div>

          {loading && (
            <div className="py-4 text-center text-[11px] text-dark-500">Loading…</div>
          )}
          {error && (
            <div className="py-3 text-center text-[11px] text-red-400">Couldn't load breakdown</div>
          )}
          {!loading && !error && data && (
            <div className="space-y-1.5">
              {COMPONENTS.map(c => {
                const value = data[c.field]
                const normalized = value == null
                  ? 0
                  : Math.max(0, Math.min(100, (value / c.max) * 100))
                return (
                  <div key={c.key} className="flex items-center gap-2" title={c.label}>
                    <span className="w-3 text-[11px] font-data font-bold text-dark-300 shrink-0">{c.key}</span>
                    <div className="flex-1 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${barColorFor(normalized)}`}
                        style={{ width: `${normalized}%` }}
                      />
                    </div>
                    <span className="w-9 text-right text-[10px] font-data text-dark-400 tabular-nums shrink-0">
                      {value != null ? `${Math.round(value)}/${c.max}` : '—'}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>,
        document.body
      )}
    </>
  )
}
