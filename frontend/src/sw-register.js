if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    // updateViaCache: 'none' bypasses HTTP cache when checking for sw.js
    // updates so a new deploy is picked up on the very next page load
    // instead of waiting up to 24h for the default refresh window.
    navigator.serviceWorker
      .register('/sw.js', { updateViaCache: 'none' })
      .then((reg) => reg.update())
      .catch(() => {})
  })
}
