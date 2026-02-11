import { Routes, Route, NavLink } from 'react-router-dom'
import CommandCenter from './pages/CommandCenter'
import Dashboard from './pages/Dashboard'
import Screener from './pages/Screener'
import StockDetail from './pages/StockDetail'
import Portfolio from './pages/Portfolio'
import Watchlist from './pages/Watchlist'
import Documentation from './pages/Documentation'
import AIPortfolio from './pages/AIPortfolio'
import Breakouts from './pages/Breakouts'
import Backtest from './pages/Backtest'
import Analytics from './pages/Analytics'

function NavIcon({ icon, label }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-xl">{icon}</span>
      <span className="text-xs">{label}</span>
    </div>
  )
}

function BottomNav() {
  const navItems = [
    { to: '/', icon: '>', label: 'CMD' },
    { to: '/dashboard', icon: '📊', label: 'Home' },
    { to: '/ai-portfolio', icon: '🤖', label: 'AI' },
    { to: '/screener', icon: '🔍', label: 'Screen' },
    { to: '/watchlist', icon: '👁️', label: 'Watch' },
    { to: '/backtest', icon: '📈', label: 'Test' },
  ]

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-dark-800 border-t border-dark-700 pb-safe">
      <div className="flex justify-around py-2">
        {navItems.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex-1 flex justify-center py-2 transition-colors ${
                isActive ? 'text-primary-500' : 'text-dark-400 hover:text-dark-200'
              }`
            }
          >
            <NavIcon icon={item.icon} label={item.label} />
          </NavLink>
        ))}
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <div className="min-h-screen pb-20">
      <Routes>
        <Route path="/" element={<CommandCenter />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/screener" element={<Screener />} />
        <Route path="/stock/:ticker" element={<StockDetail />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/ai-portfolio" element={<AIPortfolio />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/breakouts" element={<Breakouts />} />
        <Route path="/docs" element={<Documentation />} />
        <Route path="/backtest" element={<Backtest />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
      <BottomNav />
    </div>
  )
}
