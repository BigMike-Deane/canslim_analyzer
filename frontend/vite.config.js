import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Manual chunk split keeps the slow-changing vendor deps out of the main
// bundle so the browser cache survives app-code-only deploys. The previous
// single-bundle output triggered Vite's >500kB warning (the all-in-one chunk
// was ~1.07MB / ~281kB gz). React + react-router are tiny but tightly bound;
// recharts is large enough to deserve its own chunk so pages that never
// import it (e.g. CommandCenter, Notifications) don't pay for the parse.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-recharts': ['recharts'],
        },
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true
      }
    }
  }
})
