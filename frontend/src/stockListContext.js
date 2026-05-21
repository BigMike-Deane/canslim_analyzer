// Shared "stock list context" — lets StockDetail render prev/next ticker nav
// based on the ordered list the user was browsing when they clicked into the
// detail page. Stored in sessionStorage so the context survives the navigation
// but resets when the user closes the tab.
//
// Producer pages (Screener, Watchlist, Breakouts, etc.) call saveStockListContext
// whenever their visible ticker list changes. StockDetail calls
// getAdjacentTickers(currentTicker) on mount + on ticker change.

const KEY = 'stockListContext.v1'

// Drop stale contexts after an hour — if a user comes back the next day,
// resuming an old browse list would be more confusing than helpful.
const MAX_AGE_MS = 60 * 60 * 1000

export function saveStockListContext(source, tickers) {
  if (!Array.isArray(tickers) || tickers.length === 0) return
  try {
    sessionStorage.setItem(KEY, JSON.stringify({
      source,
      tickers: tickers.filter(Boolean),
      savedAt: Date.now(),
    }))
  } catch {
    // sessionStorage full / unavailable — non-fatal, nav just won't show.
  }
}

export function getStockListContext() {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || !Array.isArray(parsed.tickers)) return null
    if (Date.now() - (parsed.savedAt || 0) > MAX_AGE_MS) return null
    return parsed
  } catch {
    return null
  }
}

export function getAdjacentTickers(currentTicker) {
  const ctx = getStockListContext()
  if (!ctx) return { prev: null, next: null, source: null }
  const idx = ctx.tickers.indexOf(currentTicker)
  if (idx === -1) return { prev: null, next: null, source: ctx.source }
  return {
    prev: idx > 0 ? ctx.tickers[idx - 1] : null,
    next: idx < ctx.tickers.length - 1 ? ctx.tickers[idx + 1] : null,
    source: ctx.source,
    position: { idx: idx + 1, total: ctx.tickers.length },
  }
}
