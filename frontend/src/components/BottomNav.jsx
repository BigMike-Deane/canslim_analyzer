import { NavLink, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import Icon from './Icon'

const moreItems = [
  { to: '/notifications', label: 'Notifications', icon: 'bell' },
  { to: '/settings', label: 'Settings', icon: 'settings' },
  { to: '/screener', label: 'Screener', icon: 'filter' },
  { to: '/breakouts', label: 'Breakouts', icon: 'trending' },
  { to: '/coiled-spring/history', label: 'Coiled Spring', icon: 'zap' },
  { to: '/insider-sentiment', label: 'Insiders', icon: 'users' },
  { to: '/trade-journal', label: 'Journal', icon: 'book' },
  { to: '/bear-base', label: 'Bear Bases', icon: 'shield' },
  { to: '/analytics', label: 'Analytics', icon: 'chart' },
  { to: '/watchlist', label: 'Watchlist', icon: 'eye' },
  { to: '/fidelity', label: 'My Portfolio', icon: 'sync' },
  { to: '/docs', label: 'Documentation', icon: 'book' },
]

export default function BottomNav() {
  const [moreOpen, setMoreOpen] = useState(false)
  const navigate = useNavigate()

  const mainTabs = [
    { to: '/', icon: 'terminal', label: 'CMD', end: true },
    { to: '/ai-portfolio', icon: 'brain', label: 'AI' },
    { type: 'search', icon: 'search', label: 'Search' },
    { to: '/screener', icon: 'filter', label: 'Research' },
    { to: '/backtest', icon: 'rewind', label: 'Test' },
  ]

  return (
    <>
      {/* More Menu Overlay */}
      {moreOpen && (
        <div className="fixed inset-0 z-40 md:hidden" onClick={() => setMoreOpen(false)}>
          <div className="absolute inset-0 bg-dark-950/80 animate-fade-in" />
          <div className="absolute bottom-16 left-3 right-3 bg-dark-800 border border-dark-700/50 rounded-xl p-2 animate-slide-up shadow-xl">
            {/* Search bar */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                setMoreOpen(false)
                window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true }))
              }}
              className="w-full flex items-center gap-2 px-3 py-2.5 mb-2 rounded-lg bg-dark-700/50 border border-dark-600 text-dark-400 text-xs"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
              </svg>
              <span>Search stocks or pages...</span>
            </button>
            <div className="grid grid-cols-4 gap-1">
              {moreItems.map(item => (
                <button
                  key={item.to}
                  onClick={(e) => {
                    e.stopPropagation()
                    navigate(item.to)
                    setMoreOpen(false)
                  }}
                  className="flex flex-col items-center gap-1.5 py-3 px-1 rounded-lg text-dark-300 hover:text-dark-100 hover:bg-dark-700/50 transition-colors"
                >
                  <Icon name={item.icon} />
                  <span className="text-[10px]">{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Bottom Nav Bar */}
      <nav className="fixed bottom-0 left-0 right-0 bg-dark-900/95 backdrop-blur-md border-t border-dark-700/40 pb-safe md:hidden z-30">
        <div className="flex justify-around py-1.5">
          {mainTabs.map(item => item.type === 'search' ? (
            <button
              key="search"
              onClick={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true }))}
              className="flex-1 flex flex-col items-center gap-0.5 py-1.5 transition-colors text-dark-500 active:text-primary-400"
            >
              <Icon name={item.icon} size={20} />
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          ) : (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `flex-1 flex flex-col items-center gap-0.5 py-1.5 transition-colors relative ${
                  isActive ? 'text-primary-400' : 'text-dark-500'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <div className="absolute -top-1.5 w-6 h-0.5 bg-primary-500 rounded-full" />
                  )}
                  <Icon name={item.icon} size={20} />
                  <span className="text-[10px] font-medium">{item.label}</span>
                </>
              )}
            </NavLink>
          ))}

          {/* More button */}
          <button
            onClick={() => setMoreOpen(!moreOpen)}
            className={`flex-1 flex flex-col items-center gap-0.5 py-1.5 transition-colors ${
              moreOpen ? 'text-primary-400' : 'text-dark-500'
            }`}
          >
            <Icon name="more" size={20} />
            <span className="text-[10px] font-medium">More</span>
          </button>
        </div>
      </nav>
    </>
  )
}
