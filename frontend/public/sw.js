// v2: HTML is now network-first so a fresh deploy doesn't get masked by a
// stale cached index.html that points at an obsolete hashed bundle. Hashed
// /assets/* files are still cache-first since their filenames change on each
// build. Bumping the cache name forces v1 to be purged on activate.
const CACHE_NAME = 'canslim-v2'
const STATIC_ASSETS = [
  '/manifest.json',
]

// Install: cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  )
  self.skipWaiting()
})

// Activate: clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  )
  self.clients.claim()
})

// Push: incoming Web Push from backend → show system notification.
// Payload shape (set by backend/email_utils.py): {title, body, data}
self.addEventListener('push', (event) => {
  if (!event.data) return
  let payload
  try { payload = event.data.json() }
  catch { payload = { title: 'CANSLIM', body: event.data.text() } }

  const title = payload.title || 'CANSLIM'
  const opts = {
    body: payload.body || '',
    icon: '/icons/icon-192.svg',
    badge: '/icons/icon-192.svg',
    data: payload.data || {},
    tag: payload.data?.kind || 'canslim',
    renotify: true,
  }
  event.waitUntil(self.registration.showNotification(title, opts))
})

// Notification click: focus an existing tab or open the right path.
self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const targetUrl = event.notification.data?.url || '/notifications'
  event.waitUntil((async () => {
    const clientsArr = await self.clients.matchAll({ type: 'window', includeUncontrolled: true })
    for (const client of clientsArr) {
      const url = new URL(client.url)
      if (url.origin === self.location.origin) {
        await client.focus()
        if ('navigate' in client) {
          try { await client.navigate(targetUrl) } catch {}
        }
        return
      }
    }
    if (self.clients.openWindow) await self.clients.openWindow(targetUrl)
  })())
})

// Fetch strategy:
//   - /api/* and /health: pass through to network (no SW interception)
//   - /assets/* (Vite-hashed JS/CSS): cache-first. Filenames are content-hashed
//     so a new build produces new filenames; stale files are harmless.
//   - everything else (HTML shell, manifest, icons): network-first. Falls back
//     to cache only when offline. This is what guarantees a fresh deploy
//     reaches users on the next normal refresh.
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return
  const url = new URL(event.request.url)

  if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
    return
  }

  const isHashedAsset = url.pathname.startsWith('/assets/')

  if (isHashedAsset) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone()
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone))
          }
          return response
        })
      })
    )
    return
  }

  // Network-first for HTML / navigations / everything else
  event.respondWith(
    fetch(event.request).then((response) => {
      if (response.ok) {
        const clone = response.clone()
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone))
      }
      return response
    }).catch(() =>
      caches.match(event.request).then((cached) => {
        if (cached) return cached
        // Offline fallback for navigation requests
        if (event.request.mode === 'navigate') return caches.match('/manifest.json')
        return new Response('', { status: 504, statusText: 'Offline' })
      })
    )
  )
})
