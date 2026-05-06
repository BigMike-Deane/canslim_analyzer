import { useState, useEffect, useCallback } from 'react'
import { api, formatRelativeTime } from '../api'
import { useAuth } from '../auth'

// Convert a Base64URL VAPID public key to the Uint8Array the browser wants.
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4)
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
  const raw = atob(base64)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; ++i) out[i] = raw.charCodeAt(i)
  return out
}

const PUSH_SUPPORTED = typeof window !== 'undefined' &&
  'serviceWorker' in navigator && 'PushManager' in window

export default function Settings() {
  const { user } = useAuth()
  const [webhookUrl, setWebhookUrl] = useState('')
  const [originalUrl, setOriginalUrl] = useState('')
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [message, setMessage] = useState(null)

  // Push state
  const [pushPermission, setPushPermission] = useState(
    PUSH_SUPPORTED && typeof Notification !== 'undefined' ? Notification.permission : 'unsupported'
  )
  const [pushBusy, setPushBusy] = useState(false)
  const [pushSubs, setPushSubs] = useState([])
  const [pushMessage, setPushMessage] = useState(null)

  useEffect(() => {
    api.getMe().then(me => {
      const url = me.webhook_url || ''
      setWebhookUrl(url)
      setOriginalUrl(url)
    }).catch(() => {})
  }, [])

  const refreshPushSubs = useCallback(async () => {
    if (!PUSH_SUPPORTED) return
    try {
      const subs = await api.listPushSubscriptions()
      setPushSubs(subs)
    } catch {}
  }, [])

  useEffect(() => { refreshPushSubs() }, [refreshPushSubs])

  const dirty = webhookUrl !== originalUrl

  async function handleSave() {
    setSaving(true)
    setMessage(null)
    try {
      const updated = await api.updateMyWebhook(webhookUrl.trim())
      setOriginalUrl(updated.webhook_url || '')
      setWebhookUrl(updated.webhook_url || '')
      setMessage({ kind: 'success', text: 'Saved.' })
    } catch (err) {
      setMessage({ kind: 'error', text: err.message || 'Save failed' })
    } finally {
      setSaving(false)
    }
  }

  async function handleTest() {
    setTesting(true)
    setMessage(null)
    try {
      const r = await api.testMyWebhook()
      setMessage(r.sent
        ? { kind: 'success', text: 'Test sent — check your phone.' }
        : { kind: 'error', text: 'Webhook returned a non-2xx response. Check the URL.' })
    } catch (err) {
      setMessage({ kind: 'error', text: err.message || 'Test failed' })
    } finally {
      setTesting(false)
    }
  }

  async function handleEnablePush() {
    setPushBusy(true)
    setPushMessage(null)
    try {
      const permission = await Notification.requestPermission()
      setPushPermission(permission)
      if (permission !== 'granted') {
        setPushMessage({ kind: 'error', text: 'Permission denied. Enable in your browser settings to retry.' })
        return
      }

      const reg = await navigator.serviceWorker.ready
      const { public_key } = await api.getVapidPublicKey()
      const sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(public_key),
      })

      // The browser hands back endpoint + keys as ArrayBuffers — serialize them.
      const subJson = sub.toJSON()
      await api.subscribePush({
        endpoint: subJson.endpoint,
        keys: { p256dh: subJson.keys.p256dh, auth: subJson.keys.auth },
      })
      setPushMessage({ kind: 'success', text: 'Push enabled on this device.' })
      refreshPushSubs()
    } catch (err) {
      setPushMessage({ kind: 'error', text: err.message || 'Failed to enable push' })
    } finally {
      setPushBusy(false)
    }
  }

  async function handleTestPush() {
    setPushBusy(true)
    setPushMessage(null)
    try {
      const r = await api.testPush()
      setPushMessage(r.sent > 0
        ? { kind: 'success', text: `Test push sent to ${r.sent} device(s).` }
        : { kind: 'error', text: 'No devices received the push. Re-enable on this device?' })
    } catch (err) {
      setPushMessage({ kind: 'error', text: err.message || 'Test failed' })
    } finally {
      setPushBusy(false)
    }
  }

  async function handleRevoke(subId) {
    try {
      await api.deletePushSubscription(subId)
      // Also unregister the local PushSubscription if this is the same device
      try {
        const reg = await navigator.serviceWorker.ready
        const local = await reg.pushManager.getSubscription()
        if (local) {
          // We can't tell from id alone whether this is local — only the
          // endpoint matches. Best-effort: if the user revokes any device
          // we don't unsubscribe locally; the backend won't deliver here
          // anyway. The user can flip permission off in browser settings.
        }
      } catch {}
      refreshPushSubs()
    } catch (err) {
      setPushMessage({ kind: 'error', text: err.message || 'Revoke failed' })
    }
  }

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-2xl font-semibold text-white mb-6">Settings</h1>

      {/* Web Push (per-device) */}
      <section className="bg-dark-900 border border-dark-700 rounded-lg p-5 mb-6">
        <h2 className="text-lg font-medium text-white mb-1">Push notifications</h2>
        <p className="text-sm text-dark-400 mb-4">
          Get native phone alerts when trades fire, breakouts trigger, or the SPY gate flips.
          Each device is registered separately. {' '}
          <span className="text-amber-400">On iOS</span>, this only works if you've added CANSLIM to your home screen first.
        </p>

        {pushPermission === 'unsupported' && (
          <div className="text-sm text-amber-400">
            This browser doesn't support Web Push.
          </div>
        )}

        {pushPermission !== 'unsupported' && (
          <div className="flex gap-2 flex-wrap">
            <button
              type="button"
              onClick={handleEnablePush}
              disabled={pushBusy}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 disabled:text-dark-500 text-white text-sm rounded"
            >
              {pushBusy ? 'Working…' : pushPermission === 'granted' ? 'Re-register this device' : 'Enable on this device'}
            </button>
            {pushSubs.length > 0 && (
              <button
                type="button"
                onClick={handleTestPush}
                disabled={pushBusy}
                className="px-4 py-2 bg-dark-700 hover:bg-dark-600 disabled:bg-dark-800 disabled:text-dark-500 text-dark-200 text-sm rounded"
              >
                Send test push
              </button>
            )}
          </div>
        )}

        {pushMessage && (
          <div className={`mt-3 text-sm ${pushMessage.kind === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {pushMessage.text}
          </div>
        )}

        {pushSubs.length > 0 && (
          <div className="mt-4 border-t border-dark-700/60 pt-4">
            <div className="text-xs font-semibold tracking-wide text-dark-400 mb-2">REGISTERED DEVICES ({pushSubs.length})</div>
            <ul className="space-y-1.5">
              {pushSubs.map(s => (
                <li key={s.id} className="flex items-center justify-between gap-3 text-xs">
                  <div className="min-w-0 flex-1">
                    <div className="text-dark-200 truncate" title={s.user_agent || 'Unknown device'}>
                      {s.user_agent || 'Unknown device'}
                    </div>
                    <div className="text-dark-500">
                      added {formatRelativeTime(s.created_at)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRevoke(s.id)}
                    className="text-red-400 hover:text-red-300 transition-colors"
                  >
                    Revoke
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {/* ntfy webhook (legacy / fallback) */}
      <section className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <h2 className="text-lg font-medium text-white mb-1">ntfy webhook (fallback)</h2>
        <p className="text-sm text-dark-400 mb-4">
          Trade buys, sells, and stop-losses for <span className="text-dark-200">{user?.email}</span> also fire
          to the webhook URL below. Per-user — only your trades fire to your URL.
          Leave blank to disable. Web Push above is the primary channel; ntfy is here for redundancy.
        </p>

        <label className="block text-sm text-dark-300 mb-1">Webhook URL</label>
        <input
          type="url"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
          value={webhookUrl}
          onChange={e => setWebhookUrl(e.target.value)}
          placeholder="https://ntfy.sh/your-private-topic"
          className="w-full px-3 py-2 bg-dark-800 border border-dark-600 rounded text-white text-sm font-mono focus:outline-none focus:border-primary-500"
        />
        <p className="text-xs text-dark-500 mt-1">
          ntfy.sh format works out of the box. Pick a long, hard-to-guess topic name (anyone with the URL can read your alerts).
        </p>

        <div className="flex gap-2 mt-4">
          <button
            type="button"
            onClick={handleSave}
            disabled={!dirty || saving}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 disabled:text-dark-500 text-white text-sm rounded"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            onClick={handleTest}
            disabled={!originalUrl || testing || dirty}
            title={dirty ? 'Save first, then test' : !originalUrl ? 'No URL configured' : 'Send a test notification'}
            className="px-4 py-2 bg-dark-700 hover:bg-dark-600 disabled:bg-dark-800 disabled:text-dark-500 text-dark-200 text-sm rounded"
          >
            {testing ? 'Sending…' : 'Send test'}
          </button>
        </div>

        {message && (
          <div className={`mt-3 text-sm ${message.kind === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {message.text}
          </div>
        )}
      </section>
    </div>
  )
}
