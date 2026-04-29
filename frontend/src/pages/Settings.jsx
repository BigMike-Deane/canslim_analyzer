import { useState, useEffect } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

export default function Settings() {
  const { user } = useAuth()
  const [webhookUrl, setWebhookUrl] = useState('')
  const [originalUrl, setOriginalUrl] = useState('')
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [message, setMessage] = useState(null)

  useEffect(() => {
    api.getMe().then(me => {
      const url = me.webhook_url || ''
      setWebhookUrl(url)
      setOriginalUrl(url)
    }).catch(() => {})
  }, [])

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

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-2xl font-semibold text-white mb-6">Settings</h1>

      <section className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <h2 className="text-lg font-medium text-white mb-1">Notifications</h2>
        <p className="text-sm text-gray-400 mb-4">
          Trade buys, sells, and stop-losses for <span className="text-gray-200">{user?.email}</span> route to the
          webhook URL below. Each user has their own URL — only your trades fire to your URL.
          Leave blank to silence all notifications for your account.
        </p>

        <label className="block text-sm text-gray-300 mb-1">Webhook URL</label>
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
        <p className="text-xs text-gray-500 mt-1">
          ntfy.sh format works out of the box. Pick a long, hard-to-guess topic name (anyone with the URL can read your alerts).
        </p>

        <div className="flex gap-2 mt-4">
          <button
            type="button"
            onClick={handleSave}
            disabled={!dirty || saving}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 disabled:text-gray-500 text-white text-sm rounded"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            type="button"
            onClick={handleTest}
            disabled={!originalUrl || testing || dirty}
            title={dirty ? 'Save first, then test' : !originalUrl ? 'No URL configured' : 'Send a test notification'}
            className="px-4 py-2 bg-dark-700 hover:bg-dark-600 disabled:bg-dark-800 disabled:text-gray-500 text-gray-200 text-sm rounded"
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
