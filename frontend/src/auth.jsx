import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [googleClientId, setGoogleClientId] = useState('')

  // Check for existing session on mount
  useEffect(() => {
    async function initAuth() {
      // Fetch auth config (Google Client ID)
      try {
        const configResp = await fetch('/api/auth/config')
        if (configResp.ok) {
          const config = await configResp.json()
          setGoogleClientId(config.google_client_id || '')
        }
      } catch {
        // Server not reachable — will show login anyway
      }

      const token = localStorage.getItem('access_token')
      if (token) {
        try {
          setUser(await fetchMe(token))
          return
        } catch {
          // Token expired/invalid — try refresh
          try {
            await refreshTokens()
            setUser(await fetchMe(localStorage.getItem('access_token')))
            return
          } catch {
            logout()
          }
        }
      }
      // No token — try fetching /me without auth (works when REQUIRE_AUTH=false)
      try {
        const resp = await fetch('/api/auth/me')
        if (resp.ok) {
          setUser(await resp.json())
          return
        }
      } catch {
        // Server requires auth — show login page
      }
    }
    initAuth().finally(() => setLoading(false))
  }, [])

  async function loginWithGoogle(credential) {
    const resp = await fetch('/api/auth/google', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ credential }),
    })
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}))
      throw new Error(data.detail || 'Google login failed')
    }
    const tokens = await resp.json()
    localStorage.setItem('access_token', tokens.access_token)
    localStorage.setItem('refresh_token', tokens.refresh_token)
    const me = await fetchMe(tokens.access_token)
    setUser(me)
    return me
  }

  function logout() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, loading, loginWithGoogle, logout, googleClientId }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}

async function fetchMe(token) {
  const resp = await fetch('/api/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!resp.ok) throw new Error('Not authenticated')
  return resp.json()
}

async function refreshTokens() {
  const refreshToken = localStorage.getItem('refresh_token')
  if (!refreshToken) throw new Error('No refresh token')
  const resp = await fetch('/api/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })
  if (!resp.ok) throw new Error('Refresh failed')
  const tokens = await resp.json()
  localStorage.setItem('access_token', tokens.access_token)
  localStorage.setItem('refresh_token', tokens.refresh_token)
}

// Auto-refresh helper — exported for use by api.js
export async function ensureValidToken() {
  const token = localStorage.getItem('access_token')
  if (!token) return null
  // Try a quick decode to check expiry (JWT is base64)
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    const expiresAt = payload.exp * 1000
    if (Date.now() < expiresAt - 60000) return token // Valid for >1 min
    // Token expiring soon — refresh
    await refreshTokens()
    return localStorage.getItem('access_token')
  } catch {
    return token // Can't decode — let server decide
  }
}
