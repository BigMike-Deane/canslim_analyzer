import { Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import { ToastProvider } from './components/Toast'
import Sidebar from './components/Sidebar'
import BottomNav from './components/BottomNav'
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
import CoiledSpringHistory from './pages/CoiledSpringHistory'

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <ToastProvider>
      <div className="flex min-h-screen">
        {/* Desktop Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(c => !c)}
        />

        {/* Main Content */}
        <main className="flex-1 min-w-0 pb-20 md:pb-0">
          <div className="max-w-[1400px] mx-auto">
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
              <Route path="/coiled-spring/history" element={<CoiledSpringHistory />} />
            </Routes>
          </div>
        </main>

        {/* Mobile Bottom Nav */}
        <BottomNav />
      </div>
    </ToastProvider>
  )
}
