import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Manual chunk split keeps the slow-changing vendor deps out of the main
// bundle so the browser cache survives app-code-only deploys. The previous
// single-bundle output triggered Vite's >500kB warning (the all-in-one chunk
// was ~1.07MB / ~281kB gz). React + react-router are tiny but tightly bound;
// recharts is large enough to deserve its own chunk so pages that never
// import it (e.g. CommandCenter, Notifications) don't pay for the parse.
// Point the dev proxy elsewhere (e.g. the VPS over Tailscale) without
// touching this file: VITE_API_TARGET=http://100.104.189.36:8001 npm run dev
const apiTarget = process.env.VITE_API_TARGET || 'http://localhost:8001'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Function form: vite 8's rolldown bundler dropped the object form
        // ("manualChunks is not a function"). Same split as before.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (/node_modules\/(recharts|d3-[^/]+|victory-vendor|react-smooth)\//.test(id)) {
            return 'vendor-recharts'
          }
          if (/node_modules\/(react|react-dom|react-router|react-router-dom|scheduler)\//.test(id)) {
            return 'vendor-react'
          }
          return undefined
        },
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true
      },
      '/health': {
        target: apiTarget,
        changeOrigin: true
      }
    }
  }
})
