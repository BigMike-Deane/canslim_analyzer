import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'

const navGroups = [
  {
    label: 'TRADE',
    items: [
      { to: '/', icon: 'terminal', label: 'Command Center', end: true },
      { to: '/ai-portfolio', icon: 'brain', label: 'AI Portfolio' },
      { to: '/analytics', icon: 'chart', label: 'Analytics' },
    ],
  },
  {
    label: 'RESEARCH',
    items: [
      { to: '/dashboard', icon: 'grid', label: 'Dashboard' },
      { to: '/screener', icon: 'filter', label: 'Screener' },
      { to: '/breakouts', icon: 'trending', label: 'Breakouts' },
      { to: '/coiled-spring/history', icon: 'zap', label: 'Coiled Spring' },
    ],
  },
  {
    label: 'TOOLS',
    items: [
      { to: '/backtest', icon: 'rewind', label: 'Backtest' },
      { to: '/watchlist', icon: 'eye', label: 'Watchlist' },
      { to: '/portfolio', icon: 'briefcase', label: 'Portfolio' },
    ],
  },
  {
    label: 'REFERENCE',
    items: [
      { to: '/docs', icon: 'book', label: 'Documentation' },
    ],
  },
]

function NavIcon({ name, size = 16 }) {
  const s = size
  const props = { width: s, height: s, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }

  switch (name) {
    case 'terminal':
      return <svg {...props}><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></svg>
    case 'brain':
      return <svg {...props}><path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" /><line x1="9" y1="22" x2="15" y2="22" /></svg>
    case 'chart':
      return <svg {...props}><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></svg>
    case 'grid':
      return <svg {...props}><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /></svg>
    case 'filter':
      return <svg {...props}><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" /></svg>
    case 'trending':
      return <svg {...props}><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
    case 'zap':
      return <svg {...props}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
    case 'rewind':
      return <svg {...props}><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
    case 'eye':
      return <svg {...props}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>
    case 'briefcase':
      return <svg {...props}><rect x="2" y="7" width="20" height="14" rx="2" /><path d="M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2" /></svg>
    case 'book':
      return <svg {...props}><path d="M4 19.5A2.5 2.5 0 016.5 17H20" /><path d="M4 4.5A2.5 2.5 0 016.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15z" /></svg>
    default:
      return <svg {...props}><circle cx="12" cy="12" r="4" /></svg>
  }
}

export default function Sidebar({ collapsed, onToggle }) {
  const location = useLocation()

  return (
    <aside className={`hidden md:flex flex-col h-screen sticky top-0 bg-dark-900 border-r border-dark-700/40 transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-52'
    }`}>
      {/* Logo/Brand */}
      <div className={`flex items-center h-14 px-4 border-b border-dark-700/40 ${collapsed ? 'justify-center' : 'gap-2.5'}`}>
        <div className="w-7 h-7 rounded-lg bg-primary-600/20 border border-primary-500/30 flex items-center justify-center flex-shrink-0">
          <span className="text-primary-400 font-bold text-xs font-data">C</span>
        </div>
        {!collapsed && (
          <span className="text-sm font-semibold text-dark-100 whitespace-nowrap">CANSLIM</span>
        )}
      </div>

      {/* Nav Groups */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {navGroups.map(group => (
          <div key={group.label} className="mb-4">
            {!collapsed && (
              <div className="px-2 mb-1.5">
                <span className="text-[9px] font-semibold tracking-[0.15em] text-dark-500">
                  {group.label}
                </span>
              </div>
            )}
            <div className="space-y-0.5">
              {group.items.map(item => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => {
                    const active = isActive
                    return `flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs font-medium transition-all duration-150 group relative ${
                      active
                        ? 'text-primary-400 bg-primary-500/[0.08]'
                        : 'text-dark-400 hover:text-dark-200 hover:bg-dark-800/50'
                    } ${collapsed ? 'justify-center' : ''}`
                  }}
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-primary-500 rounded-r" />
                      )}
                      <NavIcon name={item.icon} size={collapsed ? 18 : 16} />
                      {!collapsed && <span>{item.label}</span>}
                      {collapsed && (
                        <div className="absolute left-full ml-2 px-2 py-1 bg-dark-800 border border-dark-700 rounded text-[10px] text-dark-200 whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-50 transition-opacity">
                          {item.label}
                        </div>
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center h-10 border-t border-dark-700/40 text-dark-400 hover:text-dark-200 transition-colors"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
          className={`transition-transform duration-300 ${collapsed ? 'rotate-180' : ''}`}
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>
    </aside>
  )
}
