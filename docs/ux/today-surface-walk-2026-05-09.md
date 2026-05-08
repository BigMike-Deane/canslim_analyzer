# "Today" Surface Walk — 2026-05-09

**Scope**: morning-workflow audit of the home screen (`/` → `CommandCenter.jsx`)
and adjacent surfaces. Goal: rank highest-friction items in the daily flow,
ship ONE fix.

**Methodology constraint**: the diagnostic walk was performed via code-level
inspection (component markup, Tailwind classes, API response shapes), not in
a real Chrome 390×844 emulator session. Visual issues that aren't visible from
markup (truncation, real tap-target sizing, scroll jank) were not exercised.
Structural friction (data missing, buried under clicks, wrong information
hierarchy) is what this walk catches.

**Backing files**:
- `frontend/src/pages/CommandCenter.jsx` (default route per `App.jsx:60`)
- `backend/main.py:5284` — `GET /api/command-center` consolidated handler

---

## Flow A — "Are any of my held positions in trouble?"

**Question the owner asks at 8:30 AM**: which positions need attention before
market open — close to a stop, score crashed, earnings imminent, holding too
long?

**Path on current Command Center**:
1. Land on `/` → "Positions" card (mobile order: 3rd card).
2. Each row renders `[ticker] [pos%]   [score badge] [gain %]`.
3. To check stop distance: tap ticker → load StockDetail → scroll → read.
4. To check earnings: scroll right column to "Earnings" card (mobile order:
   ~6th), eyeball ticker overlap with positions list manually.
5. To check days-held: tap ticker → load StockDetail.

**Time-to-information for the underlying question**: ~3-5 clicks per position
× number of positions held. For a 6-position portfolio, ~20+ taps before the
owner has a complete "any trouble?" picture.

**Critical structural finding**: the `/api/command-center` handler at
`main.py:5404-5415` already computes `stop_distance`, `trail_from_peak`, and
`days_held` for every position. **The frontend `PositionRow` component
(`CommandCenter.jsx:142-160`) drops all three fields and renders only the
score/gain pair.** The data is computed, shipped over the wire, and discarded
client-side.

`stop_distance` (computed at `main.py:5395` as `base_stop + g_pct`, i.e. the
% cushion before the stop fires given the current gain) is the single most
load-bearing number for this flow.

---

## Flow B — "What new candidates are actionable today?"

**Question**: which screened stocks are worth a fresh look — under $25 (per
owner preference in CLAUDE.md), pre-breakout (not extended), strong fundamentals?

**Path**:
1. "Top Candidates" card (mobile order: 4th card).
2. Each row renders `[ticker] [sector first word]   [score] [+projected %]`.
3. To check price (the owner's <$25 filter): tap → StockDetail.
4. To check breakout stage: tap → StockDetail → scroll to base/breakout panel.
5. To check fundamental quality: tap → StockDetail → scroll to audit panel.

**Critical structural finding**: handler at `main.py:5442-5450` ships
`price`, `audit_confidence`, and `name` per candidate. **The frontend
`CandidateRow` (`CommandCenter.jsx:162-182`) drops `price` and
`audit_confidence`.** The owner's primary filter (under $25) is invisible from
the home screen despite being computed.

---

## Flow C — "What did the system do overnight?"

**Question**: trades fired, alerts triggered, score crashes detected, stops
hit since yesterday's close.

**Path**:
1. "Trades" card (mobile order: ~7th).
2. Each row renders `[BUY/SELL pill] [ticker]   [realized $] [relative time]`.
3. Only most recent 6 are shown (handler limit at `main.py:5494` is 10, UI
   slices to 6 at `CommandCenter.jsx:551`).
4. Notification feed lives at `/notifications` — separate page, no preview
   on home screen, no unread badge surfaced on Command Center.

**Structural friction**: there's no "since yesterday's close" filter — the 6
recent trades could span days. No alert summary card on home (e.g.,
"3 breakouts triggered, 2 stops fired, 1 score crash since yesterday").
The bell badge in Sidebar/BottomNav exists but doesn't break out trade vs
alert vs system-event types — it's a single number.

---

## Flow D — "Is today a buy day or a defense day?"

**Question**: SPY-gate state — winner strategy is binary (SPY > 50MA = buy
day; SPY < 50MA = defense day, per CLAUDE.md "Strategy System" section).

**Path**:
1. Header `MarketStateBadge` shows `TRENDING / PRESSURE / CORRECTION /
   RECOVERY / CONFIRMED` — but those are 5-state market-state names, NOT the
   binary buy/defense the active strategy actually gates on.
2. Left column "Market" card shows `SPY [price] >50MA` or `<50MA` per
   `IndexRow` (`CommandCenter.jsx:50-64`). The `>` / `<` glyph is the actual
   gate signal but it's two scrolls deep on mobile and small text.

**Structural friction**: the active strategy's gate (binary SPY vs 50MA) is
the only signal that determines whether the trader should be playing offense
or defense today, but the home screen uses 5-state market-state language for
the most prominent badge. The user has to translate from `PRESSURE`/etc. to
"can I buy today?" themselves.

---

## Ranked findings

| # | Finding | Evidence | Impact | Fix shape |
|---|---|---|---|---|
| **1** | **PositionRow drops `stop_distance`, `trail_from_peak`, `days_held`** — already computed by API, never rendered. Owner can't see "in trouble" status without per-ticker drill-down. | `main.py:5404-5415` ships fields; `CommandCenter.jsx:142-160` renders only score+gain | **Hit every morning** — Flow A is the most-asked daily question | Pure JSX add: surface `stop_distance` (color-coded by danger) + cross-reference earnings DTE on each row. No API change. |
| 2 | CandidateRow drops `price` and `audit_confidence` — owner's <$25 filter is invisible from home screen | `main.py:5442-5450` ships price + audit; `CommandCenter.jsx:162-182` renders only score+projected% | Hit every morning during scan-review | Pure JSX add: surface price (with $25 visual emphasis) + audit_confidence chip |
| 3 | "Buy day vs defense day" requires interpretation of 5-state market-state badge — actual gate is binary SPY-vs-50MA | `CommandCenter.jsx:32-48` MARKET_STATE_CFG; SPY-gate is in handler at `main.py:5308` | Hit every morning during gate-check | Add a binary "BUY DAY / DEFENSE DAY" pill derived from `market.spy.price > market.spy.ma50`, prominent in header |
| 4 | No "since yesterday's close" filter on Trades card; activity span ambiguous | `CommandCenter.jsx:543-578`, handler limits at `main.py:5490-5494` | Daily, low magnitude | Either timestamp the trades section ("since 4 PM ET yesterday: 2 sells, 1 buy") or split overnight/today |
| 5 | Mobile card order shows Earnings (right col, ~6th) far below Positions (center col, ~3rd) — Flow A requires correlating them mentally | `CommandCenter.jsx:307-715` mobile-order comments | Hit during earnings-week mornings | If Finding #1 ships, this auto-resolves (earnings DTE inline on PositionRow) |

Findings 1 and 2 share the same root cause: handler ships rich per-row data,
UI cards render impoverished subsets. They could be fixed together, but the
discipline of ONE fix per session keeps the before/after clean and gives a
second-leverage shipment for the next UX iteration.

---

## Step 3 — Pick & ship: Finding #1

**Why this one**:
- Highest morning frequency: Flow A is the very first question the owner
  asks every market-open day.
- Best information-density-per-pixel: `stop_distance` is one number that
  collapses "how worried should I be about this position" into a glance.
- Zero risk: pure frontend rendering of fields the API already ships and
  the existing `TestCommandCenterRoute` already lightly exercises. No
  scoring/ML/trading touched. Eval-window-locked path is fully honored.
- Cross-references existing data: earnings DTE comes from the
  `earnings[]` array already on the page — just needs a ticker lookup.

**Implementation**:
- Build a `useMemo`'d `earningsByTicker` map at the parent
  `CommandCenter` component.
- Pass `earningsDays` prop to `PositionRow`.
- Render `stop_distance` always (color-coded: muted >5%, amber 2-5%, red
  ≤2%) and a compact `E:Nd` pill only when DTE ≤14 (red ≤7, amber 8-14).

**Out**: `trail_from_peak` and `days_held` — also dropped, but the row gets
visually busy with all four. Surface in a follow-up session (see Step 5).

---

## Step 5 — Next 1-2 fixes (pre-staged)

1. **Finding #2 — CandidateRow surfaces `price` + `audit_confidence`.**
   Symmetric fix to Finding #1 on the candidates card. Closes Flow B.
   Recommended next UX session.

2. **Finding #3 — Binary BUY DAY / DEFENSE DAY pill in header.**
   Derives from existing `market.spy.price > market.spy.ma50`. One small
   pill component change in the header, no API touch. Useful follow-up
   when the owner explicitly asks "translate market state to action".

If three UX sessions in a row would crowd out the queued CSO #2 (Fidelity
CSV hardening) and Shadow Step 2, hold Finding #3 — Finding #2 is the
direct symmetric continuation and adds the most value.
