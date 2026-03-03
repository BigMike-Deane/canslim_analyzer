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
              <p className="text-xs text-dark-500 text-center py-4">
                Google Sign-In not configured.<br />
                Set GOOGLE_CLIENT_ID in your environment.
              </p>
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
