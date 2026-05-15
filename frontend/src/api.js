import { cache } from './cache'
import { ensureValidToken } from './auth'

const API_BASE = ''

// Cache TTL configuration (seconds)
const CACHE_TTL = {
  '/api/command-center': 60,       // 1 min (primary dashboard)
  '/api/dashboard': 180,           // 3 min
  '/api/stocks': 300,              // 5 min (list)
  '/api/stocks/': 600,             // 10 min (individual stock)
  '/api/stocks/breaking-out': 300,
  '/api/top-growth-stocks': 300,
  '/api/portfolio': 300,
  '/api/portfolio/gameplan': 300,
  '/api/watchlist': 600,
  '/api/scanner/status': 30,       // 30 sec (fast polling)
  '/api/market-direction': 300,
  '/api/ai-portfolio': 120,        // 2 min (price sensitive)
  '/api/backtests': 600,
  '/api/earnings-audit': 300,      // 5 min
  '/api/insider-sentiment': 300,   // 5 min
  '/api/fidelity/latest': 120,     // 2 min
  '/api/fidelity/snapshots': 300,  // 5 min
  '/api/fidelity/trades': 300,     // 5 min
  '/api/fidelity/reconciliation': 120, // 2 min
  '/api/ml/status': 30,              // 30 sec (may be training)
  '/api/trade-journal': 120,          // 2 min
  '/api/bear-base': 300,               // 5 min
  '/api/system-health': 30,            // 30 sec (monitoring)
  '/api/admin/strategy-ab-eval': 60,   // 1 min — experiment moves slowly, but operator may tweak window params
  '/api/admin/strategy-ab-eval-trades': 60,  // 1 min — per-trade sibling fetched alongside the aggregate
  '/api/admin/shadow-strategies': 60,  // 1 min — registry only changes on YAML edits + boot
  '/api/notifications': 15,            // 15 sec — list + bell dropdown
  '/api/notifications/unread-count': 15,
}

// Longest-prefix wins. Object.entries() returns insertion order, so without
// sorting "/api/stocks" (300s) would shadow "/api/stocks/" (600s) for
// /api/stocks/MPWR. Sort once at module init.
const CACHE_TTL_ENTRIES = Object.entries(CACHE_TTL).sort((a, b) => b[0].length - a[0].length)

function getCacheTTL(endpoint) {
  // Check exact matches first
  if (CACHE_TTL[endpoint]) return CACHE_TTL[endpoint]
  // Longest matching prefix wins
  for (const [pattern, ttl] of CACHE_TTL_ENTRIES) {
    if (endpoint.startsWith(pattern)) return ttl
  }
  return 300 // Default 5 min
}

class APIError extends Error {
  constructor(message, status, code) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.code = code
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  const method = options.method || 'GET'

  // Only cache GET requests
  if (method === 'GET') {
    const cached = cache.get(endpoint)
    if (cached) {
      console.debug(`[Cache HIT] ${endpoint}`)
      return cached
    }
  }

  // Get fresh JWT token (auto-refreshes if expiring)
  const token = await ensureValidToken()
  const authHeaders = token ? { 'Authorization': `Bearer ${token}` } : {}
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders,
      ...options.headers
    },
    ...options
  }

  let response = await fetch(url, config)

  // On 401, try refreshing the token once
  if (response.status === 401 && token) {
    try {
      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        const refreshResp = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        })
        if (refreshResp.ok) {
          const tokens = await refreshResp.json()
          localStorage.setItem('access_token', tokens.access_token)
          localStorage.setItem('refresh_token', tokens.refresh_token)
          // Retry the original request with the new token
          config.headers['Authorization'] = `Bearer ${tokens.access_token}`
          response = await fetch(url, config)
        }
      }
    } catch {
      // Refresh failed — fall through to error handling
    }
  }

  // If still 401, show brief message then redirect to login
  if (response.status === 401) {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    // Show a visible message before redirecting
    const banner = document.createElement('div')
    banner.textContent = 'Session expired — signing out...'
    // Brand amber (primary-600) banner for session-expired notice — matches the
    // rebrand and stays distinct from P&L red used in price/return cells.
    banner.style.cssText = 'position:fixed;top:0;left:0;right:0;padding:12px;background:#d97706;color:white;text-align:center;z-index:9999;font-size:14px'
    document.body.appendChild(banner)
    setTimeout(() => { window.location.href = '/' }, 1500)
    throw new APIError('Session expired', 401)
  }

  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new APIError(
      data.detail || `Request failed: ${response.statusText}`,
      response.status,
      data.code
    )
  }

  const data = await response.json()

  // Cache GET responses
  if (method === 'GET') {
    cache.set(endpoint, {}, data, getCacheTTL(endpoint))
    console.debug(`[Cache SET] ${endpoint} (TTL: ${getCacheTTL(endpoint)}s)`)
  }

  return data
}

export const api = {
  // Auth
  getMe: () => request('/api/auth/me'),

  updateMyWebhook: (webhook_url) => request('/api/auth/me/webhook', {
    method: 'PATCH',
    body: JSON.stringify({ webhook_url }),
  }),

  testMyWebhook: () => request('/api/auth/me/webhook/test', { method: 'POST' }),

  updateNotificationPrefs: (prefs) => request('/api/auth/me/notification-prefs', {
    method: 'PATCH',
    body: JSON.stringify(prefs),
  }),

  // Web Push (per-device, native browser notifications)
  getVapidPublicKey: () => request('/api/push/vapid-public-key'),
  subscribePush: (subscription) => request('/api/push/subscribe', {
    method: 'POST',
    body: JSON.stringify(subscription),
  }),
  listPushSubscriptions: () => request('/api/push/subscriptions'),
  deletePushSubscription: (id) => request(`/api/push/subscriptions/${id}`, { method: 'DELETE' }),
  testPush: () => request('/api/push/test', { method: 'POST' }),

  // Notifications (in-app, per-user)
  getNotifications: (params = {}) => {
    const sp = new URLSearchParams()
    if (params.unread_only) sp.set('unread_only', 'true')
    if (params.limit) sp.set('limit', params.limit)
    if (params.offset != null) sp.set('offset', params.offset)
    const q = sp.toString()
    return request(`/api/notifications${q ? `?${q}` : ''}`)
  },

  getUnreadNotificationCount: () => request('/api/notifications/unread-count'),

  // Sample of recent per-stock alert scores so Settings can show
  // "would surface X of N alerts" preview without re-querying per slider tick.
  getNotificationThresholdPreview: (lookbackDays = 7) =>
    request(`/api/notifications/threshold-preview?lookback_days=${lookbackDays}`),

  markNotificationRead: async (id) => {
    const result = await request(`/api/notifications/${id}/read`, { method: 'POST' })
    cache.invalidate('/api/notifications')
    cache.invalidate('/api/notifications/unread-count')
    return result
  },

  markAllNotificationsRead: async () => {
    const result = await request('/api/notifications/read-all', { method: 'POST' })
    cache.invalidate('/api/notifications')
    cache.invalidate('/api/notifications/unread-count')
    return result
  },

  deleteNotification: async (id) => {
    const result = await request(`/api/notifications/${id}`, { method: 'DELETE' })
    cache.invalidate('/api/notifications')
    cache.invalidate('/api/notifications/unread-count')
    return result
  },

  // Health
  getHealth: () => request('/health'),

  // Dashboard
  getDashboard: () => request('/api/dashboard'),

  // Stocks / Screener
  getStocks: (params = {}) => {
    const searchParams = new URLSearchParams()
    if (params.sector) searchParams.set('sector', params.sector)
    if (params.min_score != null) searchParams.set('min_score', params.min_score)
    if (params.sort_by) searchParams.set('sort_by', params.sort_by)
    if (params.limit) searchParams.set('limit', params.limit)
    if (params.offset != null && params.offset > 0) searchParams.set('offset', params.offset)
    const query = searchParams.toString()
    return request(`/api/stocks${query ? `?${query}` : ''}`)
  },

  searchStocks: (q) => request(`/api/stocks/search?q=${encodeURIComponent(q)}&limit=8`),

  getStock: (ticker, { resolution } = {}) => {
    const qs = resolution ? `?resolution=${encodeURIComponent(resolution)}` : ''
    return request(`/api/stocks/${ticker}${qs}`)
  },

  refreshStock: async (ticker) => {
    const result = await request(`/api/stocks/${ticker}/refresh`, { method: 'POST' })
    // Invalidate caches after refresh
    cache.invalidate(`/api/stocks/${ticker}`)
    cache.invalidate('/api/stocks')
    cache.invalidate('/api/dashboard')
    return result
  },

  // Analysis jobs
  startScan: (tickers = null, source = 'sp500') => {
    const params = new URLSearchParams()
    if (source) params.set('source', source)
    const query = params.toString()
    return request(`/api/analyze/scan${query ? `?${query}` : ''}`, {
      method: 'POST',
      body: JSON.stringify({ tickers })
    })
  },

  getJobStatus: (jobId) => request(`/api/analyze/jobs/${jobId}`),

  // Portfolio
  getPortfolio: () => request('/api/portfolio'),

  addPosition: async (position) => {
    const result = await request('/api/portfolio', {
      method: 'POST',
      body: JSON.stringify(position)
    })
    // Invalidate portfolio-related caches
    cache.invalidate('/api/portfolio')
    cache.invalidate('/api/dashboard')
    return result
  },

  updatePosition: async (id, data) => {
    const result = await request(`/api/portfolio/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
    cache.invalidate('/api/portfolio')
    cache.invalidate('/api/dashboard')
    return result
  },

  deletePosition: async (id) => {
    const result = await request(`/api/portfolio/${id}`, { method: 'DELETE' })
    cache.invalidate('/api/portfolio')
    cache.invalidate('/api/dashboard')
    return result
  },

  refreshPortfolio: async () => {
    const result = await request('/api/portfolio/refresh', { method: 'POST' })
    cache.invalidate('/api/portfolio')
    cache.invalidate('/api/dashboard')
    return result
  },

  getGameplan: () => request('/api/portfolio/gameplan'),

  // Watchlist
  getWatchlist: () => request('/api/watchlist'),

  addToWatchlist: async (item) => {
    const result = await request('/api/watchlist', {
      method: 'POST',
      body: JSON.stringify(item)
    })
    cache.invalidate('/api/watchlist')
    return result
  },

  removeFromWatchlist: async (id) => {
    const result = await request(`/api/watchlist/${id}`, { method: 'DELETE' })
    cache.invalidate('/api/watchlist')
    return result
  },

  // Continuous Scanner
  getScannerStatus: () => request('/api/scanner/status'),

  startScanner: async (source = 'sp500', interval = 15) => {
    const params = new URLSearchParams()
    params.set('source', source)
    params.set('interval', interval)
    const result = await request(`/api/scanner/start?${params}`, { method: 'POST' })
    cache.invalidate('/api/scanner')
    return result
  },

  stopScanner: async () => {
    const result = await request('/api/scanner/stop', { method: 'POST' })
    cache.invalidate('/api/scanner')
    return result
  },

  updateScannerConfig: async (source, interval) => {
    const params = new URLSearchParams()
    if (source) params.set('source', source)
    if (interval) params.set('interval', interval)
    const result = await request(`/api/scanner/config?${params}`, { method: 'PATCH' })
    cache.invalidate('/api/scanner')
    return result
  },

  // Market Data
  getMarket: () => request('/api/market-direction'),

  refreshMarket: async () => {
    const result = await request('/api/market-direction/refresh', { method: 'POST' })
    cache.invalidate('/api/market-direction')
    cache.invalidate('/api/dashboard')
    return result
  },

  // AI Portfolio
  getAIPortfolio: () => request('/api/ai-portfolio'),

  getAIPortfolioHistory: (days = 30) => request(`/api/ai-portfolio/history?days=${days}`),

  getAIPortfolioWindowReturns: (window = 'all') =>
    request(`/api/ai-portfolio/window-returns?window=${window}`),

  getAIPortfolioTrades: (limit = 50, ticker = null) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (ticker) params.set('ticker', ticker)
    return request(`/api/ai-portfolio/trades?${params}`)
  },

  initializeAIPortfolio: async (startingCash = 25000) => {
    const result = await request(`/api/ai-portfolio/initialize?starting_cash=${startingCash}`, { method: 'POST' })
    cache.invalidate('/api/ai-portfolio')
    return result
  },

  refreshAIPortfolio: async () => {
    const result = await request('/api/ai-portfolio/refresh', { method: 'POST' })
    cache.invalidate('/api/ai-portfolio')
    return result
  },

  runAITradingCycle: async () => {
    const result = await request('/api/ai-portfolio/run-cycle', { method: 'POST' })
    cache.invalidate('/api/ai-portfolio')
    cache.invalidate('/api/dashboard')
    return result
  },

  updateAIPortfolioConfig: async (config) => {
    const params = new URLSearchParams()
    Object.entries(config).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        params.set(key, value)
      }
    })
    const result = await request(`/api/ai-portfolio/config?${params}`, { method: 'PATCH' })
    cache.invalidate('/api/ai-portfolio')
    return result
  },

  // Growth Mode Stocks
  getTopGrowthStocks: (limit = 10) => request(`/api/top-growth-stocks?limit=${limit}`),

  getBreakingOutStocks: (limit = 10) => request(`/api/stocks/breaking-out?limit=${limit}`),

  // Coiled Spring Alerts
  getCoiledSpringAlerts: (days = 7) => request(`/api/coiled-spring/alerts?days=${days}`),
  getCoiledSpringCandidates: () => request('/api/coiled-spring/candidates'),
  getCoiledSpringHistory: (page = 1, pageSize = 50) => request(`/api/coiled-spring/history?page=${page}&page_size=${pageSize}`),

  // Backtesting
  getBacktests: () => request('/api/backtests'),

  createBacktest: async (config) => {
    const result = await request('/api/backtests', {
      method: 'POST',
      body: JSON.stringify(config)
    })
    cache.invalidate('/api/backtests')
    return result
  },

  getBacktest: (id) => request(`/api/backtests/${id}`),

  deleteBacktest: async (id) => {
    const result = await request(`/api/backtests/${id}`, { method: 'DELETE' })
    cache.invalidate('/api/backtests')
    return result
  },

  cancelBacktest: async (id) => {
    const result = await request(`/api/backtests/${id}/cancel`, { method: 'POST' })
    cache.invalidate('/api/backtests')
    return result
  },

  // Backtest Comparison & Multi-Period
  compareBacktests: (ids) => request(`/api/backtests/compare?ids=${ids.join(',')}`),
  getBacktestPresets: () => request('/api/backtests/presets'),
  createMultiBacktest: async (config = {}) => {
    const params = new URLSearchParams()
    if (config.starting_cash) params.set('starting_cash', config.starting_cash)
    if (config.stock_universe) params.set('stock_universe', config.stock_universe)
    if (config.strategy) params.set('strategy', config.strategy)
    const result = await request(`/api/backtests/multi?${params}`, { method: 'POST' })
    cache.invalidate('/api/backtests')
    return result
  },

  createBatchBacktest: async (config) => {
    const result = await request('/api/backtests/batch', {
      method: 'POST',
      body: JSON.stringify(config)
    })
    cache.invalidate('/api/backtests')
    return result
  },
  getBacktestQueue: () => request('/api/backtests/queue'),

  // Trade Analytics
  getTradeAnalytics: () => request('/api/analytics/trades'),

  // Market Breadth
  getMarketBreadth: () => request('/api/market-breadth'),

  // Earnings Calendar
  getEarningsCalendar: () => request('/api/ai-portfolio/earnings-calendar'),

  // Portfolio Risk
  getPortfolioRisk: () => request('/api/ai-portfolio/risk'),

  // Command Center
  getCommandCenter: () => request('/api/command-center'),

  // Earnings Audit
  getEarningsAudits: (limit = 30, minConfidence = 0) =>
    request(`/api/earnings-audit?limit=${limit}&min_confidence=${minConfidence}`),
  getEarningsAudit: (ticker) => request(`/api/earnings-audit/${ticker}`),

  // Strategy Profiles
  getStrategies: ({ include_hidden = false } = {}) =>
    request(`/api/strategies${include_hidden ? '?include_hidden=true' : ''}`),

  // Insider Sentiment
  getInsiderSentiment: (params = {}) => {
    const searchParams = new URLSearchParams()
    if (params.sentiment) searchParams.set('sentiment', params.sentiment)
    if (params.min_score != null) searchParams.set('min_score', params.min_score)
    if (params.sort_by) searchParams.set('sort_by', params.sort_by)
    if (params.sort_dir) searchParams.set('sort_dir', params.sort_dir)
    if (params.sector) searchParams.set('sector', params.sector)
    if (params.limit) searchParams.set('limit', params.limit)
    if (params.offset != null && params.offset > 0) searchParams.set('offset', params.offset)
    const query = searchParams.toString()
    return request(`/api/insider-sentiment${query ? `?${query}` : ''}`)
  },

  // Fidelity Sync
  uploadFidelityPositions: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    const token = await ensureValidToken()
    const headers = token ? { Authorization: `Bearer ${token}` } : {}
    const response = await fetch(`${API_BASE}/api/fidelity/upload-positions`, {
      method: 'POST',
      headers,
      body: formData,
    })
    if (!response.ok) {
      const data = await response.json().catch(() => ({}))
      throw new APIError(data.detail || 'Upload failed', response.status)
    }
    cache.invalidate('/api/fidelity/latest')
    cache.invalidate('/api/fidelity/snapshots')
    cache.invalidate('/api/fidelity/reconciliation')
    cache.invalidate('/api/fidelity/gameplan')
    return response.json()
  },

  uploadFidelityActivity: async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    const token = await ensureValidToken()
    const headers = token ? { Authorization: `Bearer ${token}` } : {}
    const response = await fetch(`${API_BASE}/api/fidelity/upload-activity`, {
      method: 'POST',
      headers,
      body: formData,
    })
    if (!response.ok) {
      const data = await response.json().catch(() => ({}))
      throw new APIError(data.detail || 'Upload failed', response.status)
    }
    cache.invalidate('/api/fidelity/trades')
    return response.json()
  },

  getFidelityLatest: () => request('/api/fidelity/latest'),
  getFidelitySnapshots: () => request('/api/fidelity/snapshots'),
  getFidelityTrades: (limit = 50, symbol = null) => {
    const params = new URLSearchParams()
    if (limit) params.set('limit', limit)
    if (symbol) params.set('symbol', symbol)
    return request(`/api/fidelity/trades?${params}`)
  },
  getFidelityReconciliation: () => request('/api/fidelity/reconciliation'),
  getFidelityGameplan: () => request('/api/fidelity/gameplan'),

  syncFidelityToPortfolio: async () => {
    const result = await request('/api/fidelity/sync-to-portfolio', { method: 'POST' })
    cache.invalidate('/api/portfolio')
    cache.invalidate('/api/fidelity/reconciliation')
    return result
  },

  // ML Signal Layer
  getMLStatus: (strategy = 'nostate_optimized') => request(`/api/ml/status?strategy=${strategy}`),
  getMLFeatures: (strategy = 'nostate_optimized') => request(`/api/ml/features?strategy=${strategy}`),
  getMLValidation: (strategy = 'nostate_optimized') => request(`/api/ml/validation?strategy=${strategy}`),
  getMLMatrices: (limit = 20) => request(`/api/ml/matrices?limit=${limit}`),
  triggerMLTraining: async (strategy = 'nostate_optimized', backtestIds = '', mode = 'regression') => {
    const result = await request(`/api/ml/train?strategy=${strategy}&backtest_ids=${backtestIds}&mode=${mode}`, { method: 'POST' })
    cache.invalidate('/api/ml/status')
    return result
  },

  // Correlation Matrix
  getCorrelationMatrix: () => request('/api/ai-portfolio/correlation'),

  // Industry Groups / Sector Rotation
  getIndustryGroups: () => request('/api/industry-groups'),

  // Bear Base Watchlist
  getBearBaseCandidates: (limit = 50) => request(`/api/bear-base?limit=${limit}`),

  // Trade Journal
  getTradeJournal: (days = 90, ticker = '', action = '', includeVetoed = false) => {
    const params = new URLSearchParams()
    params.set('days', days)
    if (ticker) params.set('ticker', ticker)
    if (action) params.set('action', action)
    if (includeVetoed) params.set('include_vetoed', 'true')
    return request(`/api/trade-journal?${params}`)
  },

  // Admin (user management)
  getUsers: () => request('/api/admin/users'),

  createUser: async (userData) => {
    const result = await request('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify(userData),
    })
    return result
  },

  updateUser: async (userId, data) => {
    const result = await request(`/api/admin/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
    return result
  },

  // Portfolio Summary & Analytics
  getPortfolioSummary: () => request('/api/portfolio-summary'),
  getSignalAttribution: (days = 365) => request(`/api/analytics/signal-attribution?days=${days}`),
  getExitQuality: (days = 365) => request(`/api/analytics/exit-quality?days=${days}`),

  // System Health & Backups
  getSystemHealth: () => request('/api/system-health'),
  triggerBackup: () => request('/api/system/backup', { method: 'POST' }),
  getBackups: () => request('/api/system/backups'),

  // Admin — Live A/B evaluation (Approach 2 keep/revert dashboard)
  // `source` is optional — defaults to 'live' on the backend. Pass 'shadow'
  // (with the shadow stack name in `strategy`) to read ShadowTrade rows.
  getStrategyABEval: ({ strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids, source } = {}) => {
    const params = new URLSearchParams()
    if (strategy) params.set('strategy', strategy)
    if (cutoffDate) params.set('cutoff_date', cutoffDate)
    if (preWindowDays != null) params.set('pre_window_days', preWindowDays)
    if (postWindowDays != null) params.set('post_window_days', postWindowDays)
    if (excludePyramids != null) params.set('exclude_pyramids', String(excludePyramids))
    if (source) params.set('source', source)
    return request(`/api/admin/strategy-ab-eval?${params}`)
  },

  // Per-trade sibling: trade rows + cumulative-return curves for the chart + table.
  getStrategyABEvalTrades: ({ strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids, source } = {}) => {
    const params = new URLSearchParams()
    if (strategy) params.set('strategy', strategy)
    if (cutoffDate) params.set('cutoff_date', cutoffDate)
    if (preWindowDays != null) params.set('pre_window_days', preWindowDays)
    if (postWindowDays != null) params.set('post_window_days', postWindowDays)
    if (excludePyramids != null) params.set('exclude_pyramids', String(excludePyramids))
    if (source) params.set('source', source)
    return request(`/api/admin/strategy-ab-eval-trades?${params}`)
  },

  // Shadow paper-trading registry — drives the ABEval source selector.
  // archived=false (default) returns only active stacks.
  getShadowStrategies: ({ archived = false } = {}) =>
    request(`/api/admin/shadow-strategies${archived ? '?archived=true' : ''}`),

  // Cap-Delta Diagnostics — population view of c_score vs c_score_uncapped over
  // a rolling window. Only meaningful when a shadow stack with
  // scorer_overrides.disable_excellence_cap === true is registered; the ABEval
  // page hides the card otherwise.
  getCapDeltaDiagnostics: ({ days = 1, minDelta = 0.5, strategy, buyThreshold } = {}) => {
    const params = new URLSearchParams()
    params.set('days', String(days))
    params.set('min_delta', String(minDelta))
    if (strategy) params.set('strategy', strategy)
    if (buyThreshold != null) params.set('buy_threshold', String(buyThreshold))
    return request(`/api/admin/cap-delta-diagnostics?${params}`)
  },

  // Trigger an immediate snapshot email — same body as the weekly Mon 9 AM
  // UTC cron, sent on demand. Used by the ABEval "Send test snapshot email"
  // button so the operator can verify wiring without waiting a week.
  sendABEvalSnapshotEmail: ({ strategy, cutoffDate, preWindowDays, postWindowDays, excludePyramids, recipientOverride, source } = {}) => {
    const body = {
      strategy,
      cutoff_date: cutoffDate,
      pre_window_days: preWindowDays,
      post_window_days: postWindowDays,
      exclude_pyramids: excludePyramids,
    }
    if (recipientOverride) body.recipient_override = recipientOverride
    if (source && source !== 'live') body.source = source
    return request('/api/admin/strategy-ab-eval/email-test', {
      method: 'POST',
      body: JSON.stringify(body),
    })
  },
}

// Formatting utilities
export function formatScore(score) {
  if (score == null) return '-'
  return score.toFixed(1)
}

export function getScoreClass(score) {
  if (score == null) return ''
  if (score >= 80) return 'score-excellent'
  if (score >= 65) return 'score-good'
  if (score >= 50) return 'score-average'
  if (score >= 35) return 'score-poor'
  return 'score-bad'
}

export function getScoreLabel(score) {
  if (score == null) return 'N/A'
  if (score >= 80) return 'Excellent'
  if (score >= 65) return 'Good'
  if (score >= 50) return 'Average'
  if (score >= 35) return 'Poor'
  return 'Weak'
}

export function formatCurrency(value) {
  if (value == null) return '-'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value)
}

export function formatPercent(value, includeSign = false) {
  if (value == null) return '-'
  const sign = includeSign && value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}%`
}

export function formatMarketCap(value) {
  if (value == null) return '-'
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
  return formatCurrency(value)
}

export function formatCompactValue(value) {
  if (value == null) return '-'
  const abs = Math.abs(value)
  const sign = value >= 0 ? '+' : '-'
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(0)}K`
  return `${sign}$${abs.toFixed(0)}`
}

// Date/time formatting — always CST for consistency
const CST = 'America/Chicago'

export function formatDateTime(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  if (isNaN(d)) return '-'
  return d.toLocaleString('en-US', {
    timeZone: CST, month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit', hour12: true
  })
}

export function formatDate(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  if (isNaN(d)) return '-'
  return d.toLocaleDateString('en-US', {
    timeZone: CST, month: 'short', day: 'numeric', year: 'numeric'
  })
}

export function formatTime(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  if (isNaN(d)) return '-'
  return d.toLocaleTimeString('en-US', {
    timeZone: CST, hour: 'numeric', minute: '2-digit', hour12: true
  })
}

export function formatShortDate(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  if (isNaN(d)) return '-'
  return d.toLocaleDateString('en-US', {
    timeZone: CST, month: 'numeric', day: 'numeric'
  })
}

export function formatRelativeTime(dateStr) {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  if (isNaN(d)) return '-'
  const now = new Date()
  const diffMs = now - d
  const diffMins = Math.floor(diffMs / 60000)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  const diffHrs = Math.floor(diffMins / 60)
  if (diffHrs < 24) return `${diffHrs}h ago`
  const diffDays = Math.floor(diffHrs / 24)
  return `${diffDays}d ago`
}

export { APIError, cache }
