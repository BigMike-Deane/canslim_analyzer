import { cache } from './cache'

const API_BASE = ''

// Cache TTL configuration (seconds)
const CACHE_TTL = {
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
}

function getCacheTTL(endpoint) {
  // Check exact matches first
  if (CACHE_TTL[endpoint]) return CACHE_TTL[endpoint]
  // Check prefix matches (e.g., /api/stocks/AAPL matches /api/stocks/)
  for (const [pattern, ttl] of Object.entries(CACHE_TTL)) {
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

  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  }

  const response = await fetch(url, config)

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
    if (params.offset) searchParams.set('offset', params.offset)
    const query = searchParams.toString()
    return request(`/api/stocks${query ? `?${query}` : ''}`)
  },

  getStock: (ticker) => request(`/api/stocks/${ticker}`),

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

  getAIPortfolioTrades: (limit = 50) => request(`/api/ai-portfolio/trades?limit=${limit}`),

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
  }
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

export { APIError, cache }
