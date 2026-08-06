import { useState, useEffect, useCallback, useRef } from 'react'

/**
 * Declarative data fetching with the correctness guards this codebase keeps
 * re-implementing by hand (see CLAUDE.md "Frontend data fetching"):
 *
 * - Out-of-order guard: overlapping fetches (per-keystroke filter changes,
 *   poll ticks racing a manual refetch) commit only the LATEST request's
 *   result — a slower earlier response can never render under newer state.
 *   Same monotonic-seq idiom as ce9d69c, hoisted out of the pages.
 * - Deps change → automatic refetch; in-flight fetches from the previous
 *   deps generation (or an unmounted component) are invalidated.
 * - Poll ticks refresh data in the background without flipping `loading`,
 *   so polling never flashes spinners over live content.
 * - Errors from a superseded request are dropped, not rendered.
 *
 * @param {(ctx: {isPoll: boolean}) => Promise<any>} fetcher
 *        Async function returning the response. Receives {isPoll} so pollers
 *        can pass api options like {noCache: true} on background ticks.
 * @param {Array} deps  Re-fetch when these change (like useEffect deps).
 * @param {Object} [opts]
 * @param {number}  [opts.pollMs=0]      Poll interval; 0 disables polling.
 * @param {boolean} [opts.enabled=true]  False skips fetching entirely (e.g.
 *                                       modal closed); flipping true fetches.
 * @param {any}     [opts.initialData=null]
 * @returns {{data, error, loading, refetch, setData}}
 *        `refetch()` re-runs with the spinner; `refetch({background: true})`
 *        refreshes silently. `setData` accepts the functional form for
 *        optimistic updates after mutations.
 */
export default function useApi(fetcher, deps = [], opts = {}) {
  const { pollMs = 0, enabled = true, initialData = null } = opts
  const [data, setData] = useState(initialData)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(enabled)

  // Monotonic request id shared by effect fetches, poll ticks, and manual
  // refetches: only the request holding the latest seq may commit state.
  const seqRef = useRef(0)

  // Always call the latest fetcher closure, but let the `deps` array — not
  // the fetcher's identity — decide when to re-run. Callers can pass inline
  // arrows without memoizing them.
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const run = useCallback(async ({ isPoll = false } = {}) => {
    const seq = ++seqRef.current
    if (!isPoll) setLoading(true)
    try {
      const result = await fetcherRef.current({ isPoll })
      if (seq !== seqRef.current) return // superseded by a newer request
      setData(result)
      setError(null)
    } catch (err) {
      if (seq !== seqRef.current) return
      console.error('useApi fetch failed:', err)
      setError(err)
    } finally {
      if (seq === seqRef.current && !isPoll) setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      // Disabled (possibly mid-flight): the cleanup below bumps seqRef, so an
      // in-flight request can no longer clear its own spinner in `finally`.
      // Reset loading here so a consumer that flips enabled=false — e.g.
      // EdgeScorecard switching back to 'All' before a window fetch resolves —
      // doesn't stay dimmed (aria-busy) forever.
      setLoading(false)
      return undefined
    }
    run()
    const timer = pollMs > 0 ? setInterval(() => run({ isPoll: true }), pollMs) : null
    return () => {
      if (timer) clearInterval(timer)
      // Invalidate in-flight requests on deps change/unmount so they can
      // neither commit stale data nor clear a newer fetch's spinner.
      seqRef.current++
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pollMs, run, ...deps])

  const refetch = useCallback(
    ({ background = false } = {}) => run({ isPoll: background }),
    [run]
  )

  return { data, error, loading, refetch, setData }
}
