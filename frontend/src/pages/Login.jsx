import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../auth'

export default function Login() {
  const { loginWithGoogle, googleClientId } = useAuth()
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const googleBtnRef = useRef(null)

  useEffect(() => {
    if (!googleClientId || !window.google?.accounts?.id) return

    window.google.accounts.id.initialize({
      client_id: googleClientId,
      callback: handleGoogleCallback,
      auto_select: true,
    })

    window.google.accounts.id.renderButton(googleBtnRef.current, {
      type: 'standard',
      theme: 'filled_black',
      size: 'large',
      text: 'signin_with',
      shape: 'rectangular',
      width: 320,
    })
  }, [googleClientId])

  async function handleGoogleCallback(response) {
    setError('')
    setLoading(true)
    try {
      await loginWithGoogle(response.credential)
    } catch (err) {
      setError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-dark-950 px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-primary-600/20 border border-primary-500/30 flex items-center justify-center">
            <span className="text-primary-400 font-bold text-lg font-data">C</span>
          </div>
          <span className="text-xl font-semibold text-dark-100">CANSLIM Analyzer</span>
        </div>

        {/* Login Card */}
        <div className="card-glass p-6 space-y-5">
          <div>
            <h2 className="text-lg font-semibold text-dark-100 mb-1">Sign in</h2>
            <p className="text-xs text-dark-500">Use your Google account to continue</p>
          </div>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center py-4">
              <div className="w-6 h-6 border-2 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
            </div>
          )}

          {/* Google Sign-In Button */}
          <div className="flex justify-center">
            {googleClientId ? (
              <div ref={googleBtnRef} />
            ) : (
              <div className="w-full bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3 flex items-start gap-2.5">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-400 shrink-0 mt-0.5">
                  <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                <div>
                  <p className="text-xs font-semibold text-amber-300">Google Sign-In not configured</p>
                  <p className="text-[11px] text-amber-400/70 mt-0.5">Set <code className="bg-amber-500/10 px-1 rounded text-amber-300">GOOGLE_CLIENT_ID</code> in your environment.</p>
                </div>
              </div>
            )}
          </div>

          <p className="text-[10px] text-dark-600 text-center">
            Invite-only access. Contact the admin if you need an account.
          </p>
        </div>
      </div>
    </div>
  )
}
