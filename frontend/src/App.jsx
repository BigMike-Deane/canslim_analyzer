import { Routes, Route, Navigate } from 'react-router-dom'
import { useState, lazy, Suspense } from 'react'
import { AuthProvider, useAuth } from './auth'
import { ToastProvider } from './components/Toast'
import Sidebar from './components/Sidebar'
import BottomNav from './components/BottomNav'
import QuickSearch from './components/QuickSearch'
// Eager: Login is the pre-auth screen, CommandCenter is the default
// landing route — lazy-loading either just adds a spinner to every visit.
import Login from './pages/Login'
import CommandCenter from './pages/CommandCenter'

// Everything else loads on navigation; recharts-heavy pages (Backtest,
// Analytics, AIPortfolio, StockDetail) stop taxing first paint.
const Screener = lazy(() => import('./pages/Screener'))
const StockDetail = lazy(() => import('./pages/StockDetail'))
const Watchlist = lazy(() => import('./pages/Watchlist'))
const Documentation = lazy(() => import('./pages/Documentation'))
const AIPortfolio = lazy(() => import('./pages/AIPortfolio'))
const Breakouts = lazy(() => import('./pages/Breakouts'))
const Backtest = lazy(() => import('./pages/Backtest'))
const Analytics = lazy(() => import('./pages/Analytics'))
const CoiledSpringHistory = lazy(() => import('./pages/CoiledSpringHistory'))
// FidelitySync retired 2026-08-25 (owner call: no brokerage integration until
// the app earns real-money confidence; last CSV upload was 2026-03-03).
// Page component + backend routes + data kept intact for revival — re-add the
// lazy import, the /fidelity route, and the Sidebar entry to bring it back.
const Admin = lazy(() => import('./pages/Admin'))
const BearBase = lazy(() => import('./pages/BearBase'))
const EarningsGapups = lazy(() => import('./pages/EarningsGapups'))
const CorrelationMatrix = lazy(() => import('./pages/CorrelationMatrix'))
const SystemHealth = lazy(() => import('./pages/SystemHealth'))
const ABEval = lazy(() => import('./pages/ABEval'))
const Settings = lazy(() => import('./pages/Settings'))
const Notifications = lazy(() => import('./pages/Notifications'))

function RouteFallback() {
  return (
    <div className="flex items-center justify-center py-24">
      <div className="w-8 h-8 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
    </div>
  )
}

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
          <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<CommandCenter />} />
            <Route path="/screener" element={<Screener />} />
            <Route path="/stock/:ticker" element={<StockDetail />} />
            <Route path="/portfolio" element={<Navigate to="/ai-portfolio" replace />} />
            <Route path="/ai-portfolio" element={<AIPortfolio />} />
            <Route path="/watchlist" element={<Watchlist />} />
            <Route path="/breakouts" element={<Breakouts />} />
            <Route path="/docs" element={<Documentation />} />
            <Route path="/backtest/compare" element={<Backtest />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/coiled-spring/history" element={<CoiledSpringHistory />} />
            <Route path="/trade-journal" element={<Navigate to="/analytics" replace />} />
            <Route path="/bear-base" element={<BearBase />} />
            <Route path="/gapups" element={<EarningsGapups />} />
            <Route path="/correlation" element={<CorrelationMatrix />} />
            <Route path="/portfolio-summary" element={<Navigate to="/ai-portfolio" replace />} />
            <Route path="/system-health" element={<SystemHealth />} />
            <Route path="/admin/ab-eval" element={<ABEval />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/notifications" element={<Notifications />} />
            <Route path="/decision-log" element={<Navigate to="/ai-portfolio" replace />} />
            <Route path="/fidelity" element={<Navigate to="/" replace />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
          </Suspense>
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
