import { test, expect, Page } from '@playwright/test'
import { readFileSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

// package.json declares "type": "module", so __dirname does not exist here.
const here = dirname(fileURLToPath(import.meta.url))

/**
 * Cross-page number consistency.
 *
 * These exist because on 2026-09-09 three bugs shipped where a number
 * rendered on one page disagreed with the same number on another: the
 * Command Center's 30d, the AI Portfolio chart, and the Edge hero. All three
 * came from mixing two notions of "now" -- the LIVE book (re-priced by every
 * scan, round the clock) and the SNAPSHOT series (written only during market
 * hours).
 *
 * The fixture (scripts/e2e_server.py) deliberately makes the live book
 * (10500) differ from the newest snapshot (11000) and writes no snapshot for
 * today -- the pre-market condition under which the bugs were visible. The
 * old unit tests seeded those EQUAL, which is exactly why they missed it.
 *
 * The scheduler is disabled and the DB is static, so nothing moves between
 * page loads. Expected values are still derived from the API rather than
 * hardcoded, so the specs stay honest if the fixture changes.
 */

type Fixture = {
  token: string
  live_cash: number
  anchor_value: number
  stale_snapshot: number
}

const fixture: Fixture = JSON.parse(
  readFileSync(join(here, '.auth.json'), 'utf-8'),
)

const auth = { Authorization: `Bearer ${fixture.token}` }

async function signIn(page: Page) {
  await page.addInitScript((token) => {
    localStorage.setItem('access_token', token)
  }, fixture.token)
}

/** "+5.0%", "-1.05%", "30d +5.0%" -> 5.0 / -1.05 / 5.0 */
function parsePct(text: string | null): number {
  expect(text, 'element had no text').toBeTruthy()
  const m = String(text).match(/([+-]?\d+(?:\.\d+)?)\s*%/)
  expect(m, `no percentage found in "${text}"`).not.toBeNull()
  return parseFloat(m![1])
}

test.describe('the fixture itself', () => {
  test('live book differs from the newest snapshot', async () => {
    // Guards the guard: if these ever match, every spec below passes
    // vacuously -- the exact failure mode of the pre-2026-09-09 unit tests.
    expect(fixture.live_cash).not.toBe(fixture.stale_snapshot)
  })
})

test.describe('API surfaces agree', () => {
  test('every surface ends the window on the live book', async ({ request }) => {
    // Captured together, so no drift can straddle the comparison.
    const [wrRes, ccRes, histRes] = await Promise.all([
      request.get('/api/ai-portfolio/window-returns?window=30d', { headers: auth }),
      request.get('/api/command-center', { headers: auth }),
      request.get('/api/ai-portfolio/history?days=30&resolution=auto', { headers: auth }),
    ])
    expect(wrRes.ok()).toBeTruthy()
    expect(ccRes.ok()).toBeTruthy()
    expect(histRes.ok()).toBeTruthy()

    const wr = (await wrRes.json()).portfolio
    const spark = (await ccRes.json()).sparkline
    const hist = (await histRes.json())
      .filter((r: any) => r.total_value != null)
      .sort((a: any, b: any) =>
        String(a.timestamp ?? a.date).localeCompare(String(b.timestamp ?? b.date)))

    const ends = {
      'window-returns': round2(wr.current_value),
      'command-center': round2(spark[spark.length - 1].value),
      history: round2(hist[hist.length - 1].total_value),
    }
    expect(new Set(Object.values(ends)).size, `ends disagree: ${JSON.stringify(ends)}`).toBe(1)
    expect(ends['window-returns']).toBe(round2(fixture.live_cash))

    // ...and they share the anchor, not just the end.
    expect(round2(wr.start_value)).toBe(round2(spark[0].value))
    expect(round2(wr.start_value)).toBe(round2(fixture.anchor_value))
  })

  test('the chart right edge is flagged as the live point', async ({ request }) => {
    const res = await request.get('/api/ai-portfolio/history?days=30&resolution=auto', {
      headers: auth,
    })
    const rows = await res.json()
    const live = rows.filter((r: any) => r.is_live)
    expect(live).toHaveLength(1)
    expect(round2(live[0].total_value)).toBe(round2(fixture.live_cash))
  })
})

test.describe('rendered pages agree', () => {
  test('Command Center 30d matches the AI Portfolio 30D slicer', async ({ page, request }) => {
    const wr = (
      await (
        await request.get('/api/ai-portfolio/window-returns?window=30d', { headers: auth })
      ).json()
    ).portfolio

    await signIn(page)

    await page.goto('/')
    const ccText = await page.getByTestId('cc-30d-return').first().textContent()
    const ccPct = parsePct(ccText)

    await page.goto('/ai-portfolio')
    await page.getByRole('button', { name: '30D', exact: true }).first().click()
    const aiEl = page.getByTestId('ai-window-return').first()
    await expect(aiEl).toHaveAttribute('data-window', '30d')
    const aiPct = parsePct(await aiEl.textContent())

    // The two pages round to different precision (1dp vs 2dp), so compare at
    // the coarser one -- a real divergence was 0.90pp, far outside this.
    expect(Math.abs(ccPct - aiPct)).toBeLessThan(0.1)
    expect(Math.abs(ccPct - wr.return_pct)).toBeLessThan(0.1)
  })
})

function round2(n: number): number {
  return Math.round(n * 100) / 100
}
