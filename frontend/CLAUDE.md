# Frontend Conventions

Patterns for `frontend/src/`. Moved from the root CLAUDE.md so they load only when working on frontend files.

#### Frontend data fetching: use the `useApi` hook
New fetch-and-render code uses `hooks/useApi.js` instead of hand-rolling
`loading`/`error` state + effects. It bakes in the correctness guards this
codebase kept re-implementing per page (out-of-order response guard from
`ce9d69c`, unmount safety, spinner-free background polling):

```javascript
const { data, error, loading, refetch, setData } = useApi(
  () => api.getStocks(filters),   // inline arrow is fine — no useCallback needed
  [filters],                      // deps decide when to re-fetch
  { pollMs: 30000 }               // optional background polling
)
```

Optimistic updates after mutations go through `setData(prev => ...)`.
Migrated exemplars: Breakouts (one-shot), Screener (deps-driven filters),
Notifications (optimistic updates + paging). Complex multi-fetch
orchestration (AIPortfolio, Backtest) still hand-rolls — migrate
opportunistically, not wholesale.

#### UI grammar (ui-revamp, Jul-30 2026)
Every page answers its core question with always-visible content; everything
else is demoted, not deleted. The reusable units:

- `components/AlertChip.jsx` — "something needs attention". Tones: `hot`
  (act now, red stripe), `warm` (watch, amber stripe), `ok` (context,
  neutral). A chip that names a problem MUST carry `onClick` to the
  decision surface (position modal, stock page) — never a dead-end alert.
- `components/CollapsedDrawer.jsx` — demoted sections: slim header +
  LIVE count badge ("5 upcoming", "last 6h ago"), content one tap away,
  collapsed by default. A bare title with a chevron is a missed signal.
- One hero number per screen; severity stripes on rows use the SAME
  thresholds as their matching chips (drawdown < -5% / score fade >= 12 /
  near_stop) so a striped row always has a chip explaining it.
- Amber discipline: brand amber = identity/interactive; warnings get amber
  PLUS an icon or stripe; nothing purely decorative gets amber.
- `text-dark-500` is decorative-only (labels, hints); information text
  floors at `text-dark-400`.
- Loading = skeleton blocks (`.skeleton`), never spinners, for page-level
  loads. Prose cards (PortfolioNarrative) template ALL copy from live
  payloads — never author claims into JSX that data can invalidate.

#### setState: use the functional form when reading prior state
When a handler computes the next state from the current state, use the
functional form `setX(prev => next)` — never `setX(items.map(...))` that
closes over the state variable directly. The closed-over snapshot is
frozen at handler-define time, so concurrent updates (polling refreshes,
overlapping click handlers, async callbacks) eat each other's changes.

```javascript
// Wrong — stale closure. If a poll lands between click and this line,
// the polled data is overwritten by the filter of pre-poll items.
setItems(items.filter(i => i.id !== id))

// Right — reads the latest committed state.
setItems(prev => prev.filter(i => i.id !== id))
```

Real bugs from this class: `20bedb5` (Notifications mark-read/delete),
`3f64176` (Backtest delete vs 2s polling refresh).

