// Simple memory cache with TTL for API responses
class ResponseCache {
  constructor() {
    this.cache = new Map()
    this.timers = new Map()
  }

  getCacheKey(endpoint, params = {}) {
    const queryString = Object.entries(params)
      .filter(([_, v]) => v !== undefined && v !== null)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join('&')
    return `${endpoint}${queryString ? '?' + queryString : ''}`
  }

  get(endpoint, params = {}) {
    const key = this.getCacheKey(endpoint, params)
    const entry = this.cache.get(key)
    if (!entry) return null
    if (Date.now() > entry.expiresAt) {
      this.delete(key)
      return null
    }
    return entry.data
  }

  set(endpoint, params, data, ttlSeconds) {
    const key = this.getCacheKey(endpoint, params)
    if (this.timers.has(key)) clearTimeout(this.timers.get(key))

    this.cache.set(key, { data, expiresAt: Date.now() + (ttlSeconds * 1000) })
    this.timers.set(key, setTimeout(() => this.delete(key), ttlSeconds * 1000))
  }

  delete(key) {
    this.cache.delete(key)
    if (this.timers.has(key)) {
      clearTimeout(this.timers.get(key))
      this.timers.delete(key)
    }
  }

  invalidate(endpointPrefix) {
    for (const key of this.cache.keys()) {
      if (key.startsWith(endpointPrefix)) this.delete(key)
    }
  }

  clear() {
    for (const timer of this.timers.values()) clearTimeout(timer)
    this.cache.clear()
    this.timers.clear()
  }
}

export const cache = new ResponseCache()
