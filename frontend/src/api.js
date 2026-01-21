const API_BASE = ''

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

  return response.json()
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

  refreshStock: (ticker) => request(`/api/stocks/${ticker}/refresh`, { method: 'POST' }),

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

  addPosition: (position) => request('/api/portfolio', {
    method: 'POST',
    body: JSON.stringify(position)
  }),

  updatePosition: (id, data) => request(`/api/portfolio/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data)
  }),

  deletePosition: (id) => request(`/api/portfolio/${id}`, { method: 'DELETE' }),

  refreshPortfolio: () => request('/api/portfolio/refresh', { method: 'POST' }),

  getGameplan: () => request('/api/portfolio/gameplan'),

  // Watchlist
  getWatchlist: () => request('/api/watchlist'),

  addToWatchlist: (item) => request('/api/watchlist', {
    method: 'POST',
    body: JSON.stringify(item)
  }),

  removeFromWatchlist: (id) => request(`/api/watchlist/${id}`, { method: 'DELETE' }),

  // Continuous Scanner
  getScannerStatus: () => request('/api/scanner/status'),

  startScanner: (source = 'sp500', interval = 15) => {
    const params = new URLSearchParams()
    params.set('source', source)
    params.set('interval', interval)
    return request(`/api/scanner/start?${params}`, { method: 'POST' })
  },

  stopScanner: () => request('/api/scanner/stop', { method: 'POST' }),

  updateScannerConfig: (source, interval) => {
    const params = new URLSearchParams()
    if (source) params.set('source', source)
    if (interval) params.set('interval', interval)
    return request(`/api/scanner/config?${params}`, { method: 'PATCH' })
  },

  // Market Data
  getMarket: () => request('/api/market-direction'),

  refreshMarket: () => request('/api/market-direction/refresh', { method: 'POST' }),

  // AI Portfolio
  getAIPortfolio: () => request('/api/ai-portfolio'),

  getAIPortfolioHistory: (days = 30) => request(`/api/ai-portfolio/history?days=${days}`),

  getAIPortfolioTrades: (limit = 50) => request(`/api/ai-portfolio/trades?limit=${limit}`),

  initializeAIPortfolio: (startingCash = 25000) =>
    request(`/api/ai-portfolio/initialize?starting_cash=${startingCash}`, { method: 'POST' }),

  refreshAIPortfolio: () => request('/api/ai-portfolio/refresh', { method: 'POST' }),

  runAITradingCycle: () => request('/api/ai-portfolio/run-cycle', { method: 'POST' }),

  updateAIPortfolioConfig: (config) => {
    const params = new URLSearchParams()
    Object.entries(config).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        params.set(key, value)
      }
    })
    return request(`/api/ai-portfolio/config?${params}`, { method: 'PATCH' })
  },

  // Growth Mode Stocks
  getTopGrowthStocks: (limit = 10) => request(`/api/top-growth-stocks?limit=${limit}`),

  getBreakingOutStocks: (limit = 10) => request(`/api/stocks/breaking-out?limit=${limit}`)
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

export { APIError }
