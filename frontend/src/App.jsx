import { Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import { AuthProvider, useAuth } from './auth'
import { ToastProvider } from './components/Toast'
import Sidebar from './components/Sidebar'
import BottomNav from './components/BottomNav'
import QuickSearch from './components/QuickSearch'
import Login from './pages/Login'
import CommandCenter from './pages/CommandCenter'
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
import InsiderSentiment from './pages/InsiderSentiment'
import FidelitySync from './pages/FidelitySync'
import Breadth from './pages/Breadth'
import Admin from './pages/Admin'
import BacktestCompare from './pages/BacktestCompare'
import MLMatrix from './pages/MLMatrix'
import TradeJournal from './pages/TradeJournal'
import BearBase from './pages/BearBase'
import CorrelationMatrix from './pages/CorrelationMatrix'
import SectorRotation from './pages/SectorRotation'
import SystemHealth from './pages/SystemHealth'
import PortfolioSummary from './pages/PortfolioSummary'
import Settings from './pages/Settings'
import Notifications from './pages/Notifications'

function AppContent() {
  const { user, loading } = useAuth()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-dark-950">
        <div className="w-8 h-8 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
      </div>
    )
  }

  if (!user) {
    return <Login />
  }

  return (
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
            <Route path="/screener" element={<Screener />} />
            <Route path="/stock/:ticker" element={<StockDetail />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/ai-portfolio" element={<AIPortfolio />} />
            <Route path="/watchlist" element={<Watchlist />} />
            <Route path="/breakouts" element={<Breakouts />} />
            <Route path="/docs" element={<Documentation />} />
            <Route path="/backtest/compare" element={<BacktestCompare />} />
            <Route path="/backtest/ml-matrix" element={<MLMatrix />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/coiled-spring/history" element={<CoiledSpringHistory />} />
            <Route path="/insider-sentiment" element={<InsiderSentiment />} />
            <Route path="/fidelity" element={<FidelitySync />} />
            <Route path="/breadth" element={<Breadth />} />
            <Route path="/trade-journal" element={<TradeJournal />} />
            <Route path="/bear-base" element={<BearBase />} />
            <Route path="/correlation" element={<CorrelationMatrix />} />
            <Route path="/sector-rotation" element={<SectorRotation />} />
            <Route path="/portfolio-summary" element={<PortfolioSummary />} />
            <Route path="/system-health" element={<SystemHealth />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/notifications" element={<Notifications />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
        </div>
      </main>

      {/* Mobile Bottom Nav */}
      <BottomNav />
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <AppContent />
        <QuickSearch />
      </ToastProvider>
    </AuthProvider>
  )
}
