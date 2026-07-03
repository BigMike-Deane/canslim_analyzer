import { useState, useEffect, useCallback } from 'react'
import { api, formatRelativeTime } from '../api'
import { useAuth } from '../auth'
import { useOwnerPrefs } from '../hooks/useOwnerPrefs'

// Notification kinds the user can mute. Matches backend _VALID_KINDS in
// routes/auth.py and frontend KIND_META in pages/Notifications.jsx.
// Urgent-priority items (stop losses, circuit breakers) bypass mute by
// design — they're not in this list.
const MUTABLE_KINDS = [
  { key: 'trade',              label: 'Trade executed' },
  { key: 'breakout',           label: 'Breakout alerts' },
  { key: 'coiled_spring',      label: 'Coiled Spring' },
  { key: 'score_crash',        label: 'Score drops' },
  { key: 'risk_alert',         label: 'Risk alerts' },
  { key: 'spy_gate_change',    label: 'SPY gate flips' },
  { key: 'market_turn',        label: 'Market turns' },
  { key: 'bear_base_update',   label: 'Bear bases update' },
  { key: 'bear_market_report', label: 'Bear market report' },
  // NOTE: no 'morning_briefing' — no backend producer exists for that kind,
  // and _VALID_KINDS rejects unknown kinds (the checkbox 400'd the save).
  { key: 'watchlist',          label: 'Watchlist alerts' },
]

// Hour 0-23 in America/Chicago, matching the timezone backend/email_utils.py
// uses for quiet-hours gating. Intl.DateTimeFormat handles DST correctly.
function _currentChicagoHour() {
  try {
    const fmt = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/Chicago',
      hour: '2-digit',
      hour12: false,
    })
    return parseInt(fmt.format(new Date()), 10)
  } catch {
    return new Date().getUTCHours() // fallback if tz lookup fails
  }
}

// Mirrors backend _should_deliver: in_quiet iff qs <= hour < qe (qs < qe)
// OR hour >= qs OR hour < qe (qs > qe, window crosses midnight).
function _quietHoursStatus(hour, qs, qe) {
  if (qs == null || qe == null || qs === qe) return null
  const inQuiet = qs < qe ? (qs <= hour && hour < qe) : (hour >= qs || hour < qe)
  // Whole hours until window edge — coarse but matches backend's hour-level gate
  const hoursUntil = (target) => ((target - hour + 24) % 24)
  return inQuiet
    ? { active: true, hoursUntil: hoursUntil(qe), edge: qe }
    : { active: false, hoursUntil: hoursUntil(qs), edge: qs }
}

function HourSelect({ value, onChange, disabled }) {
  return (
    <select
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value === '' ? null : parseInt(e.target.value))}
      disabled={disabled}
      className="px-2 py-1 bg-dark-800 border border-dark-600 rounded text-white text-sm font-data disabled:opacity-50"
    >
      <option value="">--</option>
      {Array.from({ length: 24 }, (_, h) => (
        <option key={h} value={h}>
          {h.toString().padStart(2, '0')}:00
        </option>
      ))}
    </select>
  )
}

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

  // Current Chicago-local hour (matches backend's _should_deliver tz).
  // Polls every minute so the active-now indicator under quiet hours stays
  // honest without forcing a save.
  const [chicagoHour, setChicagoHour] = useState(() => _currentChicagoHour())
  // Owner preferences — three persistent client-side filters that
  // shape which candidates appear on CommandCenter (and, eventually,
  // other list pages). Stored in localStorage; no server roundtrip.
  const { prefs: ownerPrefs, setPrefs: setOwnerPrefs } = useOwnerPrefs()
  useEffect(() => {
    const tick = () => setChicagoHour(_currentChicagoHour())
    const id = setInterval(tick, 60_000)
    return () => clearInterval(id)
  }, [])

  // Notification preferences (per-kind mute + quiet hours + score threshold)
  const [mutedKinds, setMutedKinds] = useState(new Set())
  const [quietStart, setQuietStart] = useState(null)
  const [quietEnd, setQuietEnd] = useState(null)
  const [scoreThreshold, setScoreThreshold] = useState(null)
  const [originalPrefs, setOriginalPrefs] = useState({ muted: [], qs: null, qe: null, st: null })
  const [prefsBusy, setPrefsBusy] = useState(false)
  const [prefsMessage, setPrefsMessage] = useState(null)

  // Recent per-stock alert scores (7-day window) so we can show
  // "would surface X of N" beside the threshold without re-querying on every
  // slider tick. Filtering happens client-side.
  const [thresholdPreview, setThresholdPreview] = useState(null)
  useEffect(() => {
    api.getNotificationThresholdPreview(7)
      .then(setThresholdPreview)
      .catch(() => setThresholdPreview({ scores: [], total_with_score: 0, lookback_days: 7 }))
  }, [])

  useEffect(() => {
    api.getMe().then(me => {
      const url = me.webhook_url || ''
      setWebhookUrl(url)
      setOriginalUrl(url)
      const muted = me.mute_kinds || []
      setMutedKinds(new Set(muted))
      setQuietStart(me.quiet_hours_start ?? null)
      setQuietEnd(me.quiet_hours_end ?? null)
      setScoreThreshold(me.score_alert_threshold ?? null)
      setOriginalPrefs({
        muted,
        qs: me.quiet_hours_start ?? null,
        qe: me.quiet_hours_end ?? null,
        st: me.score_alert_threshold ?? null,
      })
    }).catch(() => {})
  }, [])

  const refreshPushSubs = useCallback(async () => {
    if (!PUSH_SUPPORTED) return
    try {
      const subs = await api.listPushSubscriptions()
      setPushSubs(subs)
    } catch (err) {
      // Called fire-and-forget after handleRevoke succeeds — if the
      // refresh silently failed the just-revoked row stayed in the UI
      // and looked like revoke hadn't worked. Surface via the existing
      // pushMessage channel (same surface handleTest / handleRevoke use).
      setPushMessage({ kind: 'error', text: err?.message || 'Failed to load push subscriptions' })
    }
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

  const prefsDirty = (() => {
    const curMuted = [...mutedKinds].sort().join(',')
    const origMuted = [...originalPrefs.muted].sort().join(',')
    return curMuted !== origMuted
      || quietStart !== originalPrefs.qs
      || quietEnd !== originalPrefs.qe
      || scoreThreshold !== originalPrefs.st
  })()

  function toggleMute(kind) {
    setMutedKinds(prev => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind); else next.add(kind)
      return next
    })
  }

  async function handleSavePrefs() {
    setPrefsBusy(true)
    setPrefsMessage(null)
    try {
      const body = { mute_kinds: [...mutedKinds] }
      if (quietStart == null && quietEnd == null) {
        body.clear_quiet_hours = true
      } else {
        body.quiet_hours_start = quietStart
        body.quiet_hours_end = quietEnd
      }
      if (scoreThreshold == null) {
        body.clear_score_alert_threshold = true
      } else {
        body.score_alert_threshold = scoreThreshold
      }
      const updated = await api.updateNotificationPrefs(body)
      setOriginalPrefs({
        muted: updated.mute_kinds || [],
        qs: updated.quiet_hours_start ?? null,
        qe: updated.quiet_hours_end ?? null,
        st: updated.score_alert_threshold ?? null,
      })
      setPrefsMessage({ kind: 'success', text: 'Preferences saved.' })
    } catch (err) {
      setPrefsMessage({ kind: 'error', text: err.message || 'Save failed' })
    } finally {
      setPrefsBusy(false)
    }
  }

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-2xl font-semibold text-white mb-6">Settings</h1>

      {/* Notification preferences (per-kind mute + quiet hours) */}
      <section className="bg-dark-900 border border-dark-700 rounded-lg p-5 mb-6">
        <h2 className="text-lg font-medium text-white mb-1">Notification preferences</h2>
        <p className="text-sm text-dark-400 mb-4">
          Mute specific kinds and silence push during quiet hours.{' '}
          <span className="text-amber-400">Urgent alerts</span>{' '}
          (stop losses, circuit breakers) always come through. In-app
          notifications are always recorded — only push delivery is gated.
        </p>

        <div className="mb-4">
          <div className="text-xs font-semibold tracking-wide text-dark-400 mb-2">MUTE KINDS</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {MUTABLE_KINDS.map(({ key, label }) => {
              const muted = mutedKinds.has(key)
              return (
                <label
                  key={key}
                  className={`flex items-center gap-2 px-2.5 py-1.5 rounded border cursor-pointer transition-colors ${
                    muted ? 'bg-dark-800 border-dark-700 text-dark-500'
                          : 'bg-dark-850 border-dark-700/60 text-dark-200 hover:border-dark-600'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={muted}
                    onChange={() => toggleMute(key)}
                    className="accent-primary-500"
                  />
                  <span className="text-sm">{label}</span>
                </label>
              )
            })}
          </div>
        </div>

        <div className="mb-4">
          <div className="text-xs font-semibold tracking-wide text-dark-400 mb-2">QUIET HOURS (CHICAGO TIME)</div>
          <div className="flex items-center gap-2 text-sm text-dark-300">
            <span>From</span>
            <HourSelect value={quietStart} onChange={setQuietStart} />
            <span>to</span>
            <HourSelect value={quietEnd} onChange={setQuietEnd} />
            {(quietStart != null || quietEnd != null) && (
              <button
                type="button"
                onClick={() => { setQuietStart(null); setQuietEnd(null) }}
                className="ml-2 text-xs text-dark-500 hover:text-dark-300 underline"
              >
                Clear
              </button>
            )}
          </div>
          {quietStart != null && quietEnd != null && quietStart !== quietEnd && (
            <>
              <p className="text-[11px] text-dark-500 mt-1.5">
                {quietStart < quietEnd
                  ? `Push silenced ${quietStart.toString().padStart(2, '0')}:00 – ${quietEnd.toString().padStart(2, '0')}:00 (Chicago)`
                  : `Push silenced ${quietStart.toString().padStart(2, '0')}:00 – next day ${quietEnd.toString().padStart(2, '0')}:00 (Chicago)`}
              </p>
              {(() => {
                const status = _quietHoursStatus(chicagoHour, quietStart, quietEnd)
                if (!status) return null
                const edge = status.edge.toString().padStart(2, '0') + ':00'
                return status.active ? (
                  <p className="text-[11px] text-amber-400 mt-1 font-medium">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5 align-middle"></span>
                    Quiet hours ACTIVE · ends at {edge} Chicago ({status.hoursUntil}h)
                  </p>
                ) : (
                  <p className="text-[11px] text-emerald-400 mt-1">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 align-middle"></span>
                    Quiet hours not active · next active at {edge} Chicago ({status.hoursUntil}h)
                  </p>
                )
              })()}
            </>
          )}
        </div>

        <div className="mb-4">
          <div className="text-xs font-semibold tracking-wide text-dark-400 mb-2">MINIMUM SCORE FOR ALERTS</div>
          <p className="text-[11px] text-dark-500 mb-2">
            Per-stock alerts (breakouts, coiled springs, trade signals) with a CANSLIM score below this threshold won't ping you.
            System-wide alerts (SPY gate, market turns) are unaffected.
          </p>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={scoreThreshold ?? 0}
              onChange={(e) => setScoreThreshold(parseInt(e.target.value))}
              disabled={prefsBusy}
              className="flex-1 accent-primary-500 disabled:opacity-50"
            />
            <input
              type="number"
              min="0"
              max="100"
              value={scoreThreshold ?? ''}
              placeholder="—"
              onChange={(e) => {
                const v = e.target.value
                if (v === '') return setScoreThreshold(null)
                const n = parseInt(v)
                setScoreThreshold(Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : null)
              }}
              disabled={prefsBusy}
              className="w-16 px-2 py-1 bg-dark-800 border border-dark-600 rounded text-sm text-dark-100 font-data text-right focus:outline-none focus:border-primary-500/50 disabled:opacity-50"
            />
            {scoreThreshold != null && (
              <button
                type="button"
                onClick={() => setScoreThreshold(null)}
                className="text-xs text-dark-500 hover:text-dark-300 underline"
              >
                Clear
              </button>
            )}
          </div>
          <p className="text-[11px] text-dark-500 mt-1.5">
            {scoreThreshold == null
              ? 'No threshold — every alert passes through.'
              : `Suppressing alerts with score < ${scoreThreshold}.`}
          </p>
          {(() => {
            // Live preview against the user's last 7 days of per-stock alerts.
            // null = still loading; empty array = no alerts to evaluate.
            if (thresholdPreview == null) return null
            const total = thresholdPreview.total_with_score ?? 0
            if (total === 0) {
              return (
                <p className="text-[11px] text-dark-600 mt-1 italic">
                  No per-stock alerts in the last {thresholdPreview.lookback_days || 7} days
                  to preview against yet.
                </p>
              )
            }
            const t = scoreThreshold ?? 0
            const wouldPass = (thresholdPreview.scores || [])
              .filter(s => s >= t).length
            const pct = Math.round((wouldPass / total) * 100)
            return (
              <p className="text-[11px] text-primary-400 mt-1">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary-400 mr-1.5 align-middle"></span>
                Last {thresholdPreview.lookback_days || 7} days: <span className="font-data text-primary-300">{wouldPass} of {total}</span> alerts ({pct}%) would pass this threshold.
              </p>
            )
          })()}
        </div>

        <button
          type="button"
          onClick={handleSavePrefs}
          disabled={!prefsDirty || prefsBusy}
          className="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 disabled:text-dark-500 text-white text-sm rounded"
        >
          {prefsBusy ? 'Saving…' : 'Save preferences'}
        </button>
        {prefsMessage && (
          <div className={`mt-3 text-sm ${prefsMessage.kind === 'success' ? 'text-green-400' : 'text-red-400'}`}>
            {prefsMessage.text}
          </div>
        )}
      </section>

      {/* Owner preferences — client-side filters that shape which
          candidates appear on the home dashboard. Stored in localStorage
          per-device (not synced to other browsers). Mirrors the
          trading-prefs comment block in CLAUDE.md. */}
      <section className="bg-dark-900 border border-dark-700 rounded-lg p-5 mb-6">
        <h2 className="text-lg font-medium text-white mb-1">Owner preferences</h2>
        <p className="text-sm text-dark-400 mb-4">
          Filter candidates to your usual trading style. Applied client-side on
          the Command Center "Top Candidates" list. Other pages (Screener,
          Breakouts) keep their own filters independent for now.
        </p>

        {/* Max share price — owner's CLAUDE.md preference is under $25. */}
        <div className="mb-4">
          <label className="block text-xs font-semibold tracking-wide text-dark-400 mb-1.5">
            MAX SHARE PRICE
          </label>
          <div className="flex items-center gap-2">
            <span className="text-dark-500">$</span>
            <input
              type="number"
              min="0"
              step="1"
              placeholder="No limit"
              value={ownerPrefs.maxSharePrice ?? ''}
              onChange={(e) => {
                const raw = e.target.value
                const next = raw === '' ? null : Math.max(0, Number(raw))
                setOwnerPrefs({ maxSharePrice: Number.isFinite(next) ? next : null })
              }}
              className="w-32 px-3 py-1.5 rounded bg-dark-850 border border-dark-700 text-dark-100 text-sm focus:border-primary-500/60 focus:outline-none"
            />
            {ownerPrefs.maxSharePrice != null && (
              <button
                onClick={() => setOwnerPrefs({ maxSharePrice: null })}
                className="text-xs text-dark-500 hover:text-dark-300"
              >
                Clear
              </button>
            )}
          </div>
          <p className="text-[11px] text-dark-500 mt-1">
            Hide candidates priced above this. Leave blank to show all.
          </p>
        </div>

        {/* Hide extended — chases above pivot+5%. */}
        <label className="flex items-start gap-3 px-3 py-2.5 rounded border border-dark-700/60 bg-dark-850 mb-2 cursor-pointer hover:border-dark-600">
          <input
            type="checkbox"
            checked={ownerPrefs.hideExtended}
            onChange={(e) => setOwnerPrefs({ hideExtended: e.target.checked })}
            className="mt-0.5 accent-primary-500"
          />
          <div className="flex-1">
            <div className="text-sm text-dark-100">Hide extended candidates</div>
            <div className="text-[11px] text-dark-500">
              Drops anything more than 5% above its pivot — chasing the breakout.
            </div>
          </div>
        </label>

        {/* Hide near-earnings — gap-down risk on fresh entries. */}
        <label className="flex items-start gap-3 px-3 py-2.5 rounded border border-dark-700/60 bg-dark-850 mb-2 cursor-pointer hover:border-dark-600">
          <input
            type="checkbox"
            checked={ownerPrefs.hideNearEarnings}
            onChange={(e) => setOwnerPrefs({ hideNearEarnings: e.target.checked })}
            className="mt-0.5 accent-primary-500"
          />
          <div className="flex-1">
            <div className="text-sm text-dark-100">
              Hide candidates within{' '}
              <input
                type="number"
                min="1"
                max="30"
                value={ownerPrefs.nearEarningsDays}
                onChange={(e) => {
                  const n = Math.max(1, Math.min(30, Number(e.target.value) || 7))
                  setOwnerPrefs({ nearEarningsDays: n })
                }}
                onClick={(e) => e.stopPropagation()}
                className="w-12 px-1.5 py-0.5 rounded bg-dark-900 border border-dark-700 text-dark-100 text-xs text-center"
              />
              {' '}days of earnings
            </div>
            <div className="text-[11px] text-dark-500">
              Avoids the gap-down risk on positions taken right before a report.
            </div>
          </div>
        </label>
      </section>

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
