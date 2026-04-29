const CACHE_NAME = 'canslim-v1'
const STATIC_ASSETS = [
  '/',
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

// Fetch: network-first for API, cache-first for static
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)

  // API calls: always go to network
  if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
    return
  }

  // Static assets: try cache, fall back to network
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached
      return fetch(event.request).then((response) => {
        // Cache successful GET responses
        if (response.ok && event.request.method === 'GET') {
          const clone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone))
        }
        return response
      })
    }).catch(() => {
      // Offline fallback for navigation requests
      if (event.request.mode === 'navigate') {
        return caches.match('/')
      }
    })
  )
})
