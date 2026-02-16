import { NavLink, useNavigate } from 'react-router-dom'
import { useState } from 'react'

function TabIcon({ name, size = 20 }) {
  const props = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }

  switch (name) {
    case 'terminal':
      return <svg {...props}><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></svg>
    case 'brain':
      return <svg {...props}><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" /><line x1="9" y1="22" x2="15" y2="22" /></svg>
    case 'grid':
      return <svg {...props}><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /></svg>
    case 'rewind':
      return <svg {...props}><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
    case 'more':
      return <svg {...props}><circle cx="12" cy="12" r="1.5" fill="currentColor" /><circle cx="12" cy="5" r="1.5" fill="currentColor" /><circle cx="12" cy="19" r="1.5" fill="currentColor" /></svg>
    default:
      return <svg {...props}><circle cx="12" cy="12" r="4" /></svg>
  }
}

const moreItems = [
  { to: '/dashboard', icon: 'grid', label: 'Dashboard' },
  { to: '/screener', label: 'Screener', icon: 'filter' },
  { to: '/breakouts', label: 'Breakouts', icon: 'trending' },
  { to: '/coiled-spring/history', label: 'Coiled Spring', icon: 'zap' },
  { to: '/analytics', label: 'Analytics', icon: 'chart' },
  { to: '/watchlist', label: 'Watchlist', icon: 'eye' },
  { to: '/portfolio', label: 'Portfolio', icon: 'briefcase' },
  { to: '/docs', label: 'Documentation', icon: 'book' },
]

function MoreIcon({ name }) {
  const props = { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }

  switch (name) {
    case 'grid':
      return <svg {...props}><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /></svg>
    case 'filter':
      return <svg {...props}><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
    case 'trending':
      return <svg {...props}><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
    case 'zap':
      return <svg {...props}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
    case 'chart':
      return <svg {...props}><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></svg>
    case 'eye':
      return <svg {...props}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
    case 'briefcase':
      return <svg {...props}><rect x="2" y="7" width="20" height="14" rx="2" /><path d="M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2" /></svg>
    case 'book':
      return <svg {...props}><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M4 4.5A2.5 2.5 0 016.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15z" /></svg>
    default:
      return null
  }
}

export default function BottomNav() {
  const [moreOpen, setMoreOpen] = useState(false)
  const navigate = useNavigate()

  const mainTabs = [
    { to: '/', icon: 'terminal', label: 'CMD', end: true },
    { to: '/ai-portfolio', icon: 'brain', label: 'AI' },
    { to: '/dashboard', icon: 'grid', label: 'Research' },
    { to: '/backtest', icon: 'rewind', label: 'Test' },
  ]

  return (
    <>
      {/* More Menu Overlay */}
      {moreOpen && (
        <div className="fixed inset-0 z-40 md:hidden" onClick={() => setMoreOpen(false)}>
          <div className="absolute inset-0 bg-dark-950/70 backdrop-blur-sm animate-fade-in" />
          <div className="absolute bottom-16 left-3 right-3 bg-dark-800 border border-dark-700/50 rounded-xl p-2 animate-slide-up shadow-xl">
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
                  <MoreIcon name={item.icon} />
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
          {mainTabs.map(item => (
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
                  <TabIcon name={item.icon} />
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
            <TabIcon name="more" />
            <span className="text-[10px] font-medium">More</span>
          </button>
        </div>
      </nav>
    </>
  )
}
