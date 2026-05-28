// ── Owner Preferences ──────────────────────────────────────────────
// Three persistent filters that every list-of-candidates page can
// respect: max share price (owner prefers <$25 stocks per CLAUDE.md),
// hide-extended (>5% above pivot — chasing the breakout), and hide-
// near-earnings (skip stocks reporting within N days, default 7).
//
// Stored under a single localStorage key as JSON so adding a new pref
// later doesn't break the existing shape. Read on mount, written on
// every setPrefs() call. No React context — the hook reads localStorage
// directly so multiple pages stay in sync if the user toggles a pref
// on Settings then navigates back to a list page (the Settings save
// hits localStorage; the list page re-renders on focus or remount).
//
// matchesPrefs(item) returns TRUE when item should be SHOWN. The page
// uses Array.filter(matchesPrefs).

import { useCallback, useState } from 'react'

const STORAGE_KEY = 'ownerPrefs.v1'

export const DEFAULT_PREFS = {
  // null means "no cap" — show all. Owner's CLAUDE.md preference is
  // under $25, but we leave the filter inactive by default so existing
  // behavior is preserved until the user opts in.
  maxSharePrice: null,
  // Drop stocks where (current − pivot) / pivot > 5%. The pivot tier
  // chip on CommandCenter already paints these red as a visual signal;
  // this filter goes further and removes them from view entirely.
  hideExtended: false,
  // Drop stocks reporting earnings within this many days. Owner has
  // been bitten by earnings gap-downs on freshly-entered positions, so
  // pre-filtering during candidate scan is a real risk reduction.
  hideNearEarnings: false,
  nearEarningsDays: 7,
}

function readStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_PREFS
    const parsed = JSON.parse(raw)
    // Forward-compat: merge with defaults so a newer pref added in
    // code defaults sensibly on a client with an older saved blob.
    return { ...DEFAULT_PREFS, ...parsed }
  } catch {
    return DEFAULT_PREFS
  }
}

export function useOwnerPrefs() {
  const [prefs, setPrefsState] = useState(readStored)

  const setPrefs = useCallback((partial) => {
    setPrefsState((prev) => {
      const next = { ...prev, ...partial }
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      } catch {
        // localStorage can throw in Safari private-browsing — fall
        // through silently. In-memory state stays correct.
      }
      return next
    })
  }, [])

  // matchesPrefs is intentionally schema-tolerant: it reads `price`
  // (or `current_price`), `pivot_price`, and `days_to_earnings` if
  // present and skips a filter when the input is missing. Callers
  // don't have to massage list items into a canonical shape.
  const matchesPrefs = useCallback((item) => {
    if (!item) return true
    const price = item.price ?? item.current_price
    const pivot = item.pivot_price
    const dte = item.days_to_earnings

    if (prefs.maxSharePrice != null && prefs.maxSharePrice > 0 && price != null) {
      if (price > prefs.maxSharePrice) return false
    }
    if (prefs.hideExtended && pivot != null && pivot > 0 && price != null) {
      const pctAbove = ((price - pivot) / pivot) * 100
      if (pctAbove > 5) return false
    }
    if (prefs.hideNearEarnings && dte != null && dte >= 0 && dte <= (prefs.nearEarningsDays || 7)) {
      return false
    }
    return true
  }, [prefs])

  return { prefs, setPrefs, matchesPrefs }
}
