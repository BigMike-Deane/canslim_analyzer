import { defineConfig, devices } from '@playwright/test'

/**
 * E2E config.
 *
 * Runs against a locally served, freshly seeded stack -- never production.
 * The backend serves the built frontend itself (backend/main.py mounts
 * frontend/dist), so one uvicorn process provides both the API and the UI
 * and there is no separate vite server to coordinate.
 *
 * `npm run build` must have run first; the e2e npm script chains it.
 * Set E2E_BASE_URL to point the suite at an already-running instance
 * instead (skips the managed server).
 */
const PORT = process.env.E2E_PORT ?? '8011'
const BASE = process.env.E2E_BASE_URL ?? `http://127.0.0.1:${PORT}`

export default defineConfig({
  testDir: './e2e',
  // Live market data means values move between page loads; the specs are
  // written to compare sources captured together rather than across time,
  // but keep retries at 0 so a real disagreement is never masked as flake.
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: 'python3 ../scripts/e2e_server.py',
        url: `${BASE}/health`,
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
        stdout: 'pipe',
        stderr: 'pipe',
      },
})
