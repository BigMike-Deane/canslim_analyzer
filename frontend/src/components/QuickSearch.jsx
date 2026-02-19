import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { ScoreBadge } from './Badge'

const PAGES = [
  { label: 'Command Center', path: '/', keywords: 'home cmd' },
  { label: 'AI Portfolio', path: '/ai-portfolio', keywords: 'ai trading positions' },
  { label: 'Dashboard', path: '/dashboard', keywords: 'research market' },
  { label: 'Screener', path: '/screener', keywords: 'filter stocks scan' },
  { label: 'Breakouts', path: '/breakouts', keywords: 'breaking base pattern' },
  { label: 'Backtest', path: '/backtest', keywords: 'test historical' },
  { label: 'Analytics', path: '/analytics', keywords: 'trades performance' },
  { label: 'Watchlist', path: '/watchlist', keywords: 'watch alerts' },
  { label: 'Coiled Spring', path: '/coiled-spring/history', keywords: 'earnings catalyst' },
  { label: 'Insider Sentiment', path: '/insider-sentiment', keywords: 'insiders smart money' },
  { label: 'Portfolio', path: '/portfolio', keywords: 'holdings positions manual' },
  { label: 'Documentation', path: '/docs', keywords: 'help canslim guide' },
]

export default function QuickSearch() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [selected, setSelected] = useState(0)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)
  const navigate = useNavigate()
  const abortRef = useRef(null)

  // Cmd+K / Ctrl+K to open
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery('')
      setResults([])
      setSelected(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  // Search stocks when query changes
  useEffect(() => {
    if (!query || query.length < 1) {
      setResults([])
      setSelected(0)
      return
    }

    // Debounce
    const timer = setTimeout(async () => {
      if (abortRef.current) abortRef.current.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setLoading(true)
      try {
        const data = await api.searchStocks(query)
        if (!controller.signal.aborted) {
          setResults(data || [])
          setSelected(0)
        }
      } catch {
        if (!controller.signal.aborted) setResults([])
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }, 150)

    return () => clearTimeout(timer)
  }, [query])

  // Filter pages by query
  const filteredPages = query
    ? PAGES.filter(p =>
        p.label.toLowerCase().includes(query.toLowerCase()) ||
        p.keywords.includes(query.toLowerCase())
      )
    : PAGES.slice(0, 6)

  // Combined items: stocks first, then pages
  const allItems = [
    ...results.map(r => ({ type: 'stock', ...r })),
    ...filteredPages.map(p => ({ type: 'page', ...p })),
  ]

  const go = useCallback((item) => {
    setOpen(false)
    if (item.type === 'stock') {
      navigate(`/stock/${item.ticker}`)
    } else {
      navigate(item.path)
    }
  }, [navigate])

  // Keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelected(s => Math.min(s + 1, allItems.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelected(s => Math.max(s - 1, 0))
    } else if (e.key === 'Enter' && allItems[selected]) {
      e.preventDefault()
      go(allItems[selected])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center pt-[15vh] sm:pt-[20vh]"
      onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}
    >
      <div className="absolute inset-0 bg-dark-950/70 backdrop-blur-sm" />

      <div className="relative w-full max-w-md mx-3 bg-dark-800 border border-dark-700/50 rounded-xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-dark-700/50">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-dark-400 shrink-0">
            <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search stocks or pages..."
            className="flex-1 bg-transparent text-sm text-dark-100 placeholder-dark-500 outline-none"
          />
          <kbd className="hidden sm:inline-block text-[10px] text-dark-500 bg-dark-700 px-1.5 py-0.5 rounded border border-dark-600">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto py-1">
          {/* Stock results */}
          {results.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] text-dark-500 uppercase tracking-wider font-semibold">
                Stocks
              </div>
              {results.map((r, i) => (
                <button
                  key={r.ticker}
                  onClick={() => go({ type: 'stock', ...r })}
                  onMouseEnter={() => setSelected(i)}
                  className={`w-full flex items-center justify-between px-3 py-2 text-left transition-colors ${
                    selected === i ? 'bg-primary-500/10 text-dark-50' : 'text-dark-300 hover:bg-dark-750'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="font-semibold font-data text-sm">{r.ticker}</span>
                    <span className="text-dark-400 text-xs truncate">{r.name}</span>
                  </div>
                  <ScoreBadge score={r.score} size="xs" />
                </button>
              ))}
            </>
          )}

          {/* Page results */}
          {filteredPages.length > 0 && (
            <>
              <div className="px-3 pt-2 pb-1 text-[10px] text-dark-500 uppercase tracking-wider font-semibold">
                {query ? 'Pages' : 'Quick Navigate'}
              </div>
              {filteredPages.map((p, idx) => {
                const i = results.length + idx
                return (
                  <button
                    key={p.path}
                    onClick={() => go({ type: 'page', ...p })}
                    onMouseEnter={() => setSelected(i)}
                    className={`w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors ${
                      selected === i ? 'bg-primary-500/10 text-dark-50' : 'text-dark-300 hover:bg-dark-750'
                    }`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-dark-500 shrink-0">
                      <polyline points="9 18 15 12 9 6" />
                    </svg>
                    <span className="text-sm">{p.label}</span>
                  </button>
                )
              })}
            </>
          )}

          {/* Empty state */}
          {query && results.length === 0 && filteredPages.length === 0 && !loading && (
            <div className="text-center py-6 text-dark-500 text-xs">
              No results for "{query}"
            </div>
          )}

          {/* Loading */}
          {loading && results.length === 0 && (
            <div className="text-center py-4 text-dark-500 text-xs animate-pulse">
              Searching...
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="border-t border-dark-700/50 px-3 py-2 flex items-center gap-3 text-[10px] text-dark-500">
          <span><kbd className="bg-dark-700 px-1 py-0.5 rounded border border-dark-600">↑↓</kbd> navigate</span>
          <span><kbd className="bg-dark-700 px-1 py-0.5 rounded border border-dark-600">↵</kbd> select</span>
          <span><kbd className="bg-dark-700 px-1 py-0.5 rounded border border-dark-600">esc</kbd> close</span>
        </div>
      </div>
    </div>
  )
}
