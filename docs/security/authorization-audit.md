# Authorization Audit

Date: 2026-05-08
Branch: main (post-`00d5cf0`)
Scope: Every `/api/*` route across `backend/main.py` and `backend/routes/*.py` (121 routes total).
Methodology: Cross-reference each route's auth dependency (`get_current_active_user` / `get_admin_user` / none) against per-resource ownership filtering. For routes that take a resource id from path/query, confirm the underlying query filters on `current_user.id` for any model that has a `user_id` column.

## Categories

- **OK** — Per-user resource. Handler filters by `current_user.id`.
- **GLOBAL** — System-wide data, no per-user concept needed (e.g. stock universe, market state, ML models).
- **PUBLIC** — Intentionally unauthenticated (e.g. health, OAuth handshake, VAPID public key).
- **GAP** — Per-user resource. Handler accepts a resource id but does NOT filter by `current_user.id`. **IDOR risk.**
- **MISSING_AUTH** — Should require auth but does not.

## Auth Contract Notes

- `get_current_user` (backend/auth.py:100) decodes the JWT and returns the matching `User`. With `REQUIRE_AUTH=false` and no token, it returns `User(id=1)` (the owner). All IDOR tests must therefore present valid bearer tokens for both alice and bob to avoid silent owner-impersonation.
- `get_current_active_user` adds an `is_active` check.
- `get_admin_user` adds an `is_admin` check (403 otherwise).

## Route Table

### `backend/main.py` (top-level `app`)

| Method | Path | Auth | Category | Notes |
|---|---|---|---|---|
| GET | `/health` | none | PUBLIC | Liveness probe |
| GET | `/api/system-health` | active | GLOBAL | System-wide health |
| POST | `/api/system/backup` | active | GLOBAL | Triggers DB backup; admin-style but currently active. *(out of IDOR scope; flagged for follow-up CSO session on admin endpoint hardening.)* |
| GET | `/api/system/backups` | active | GLOBAL | List of backup files |
| GET | `/api/dashboard` | active | GLOBAL | Aggregate market view |
| GET | `/api/market-direction` | active | GLOBAL | |
| POST | `/api/market-direction/refresh` | active | GLOBAL | |
| GET | `/api/market` | active | GLOBAL | |
| POST | `/api/market/refresh` | active | GLOBAL | |
| GET | `/api/rate-limit-stats` | active | GLOBAL | |
| POST | `/api/rate-limit-stats/reset` | active | GLOBAL | |
| GET | `/api/stocks` | active | GLOBAL | |
| GET | `/api/stocks/search` | active | GLOBAL | |
| GET | `/api/stocks/sectors` | active | GLOBAL | |
| GET | `/api/top-growth-stocks` | active | GLOBAL | |
| GET | `/api/stocks/breaking-out` | active | GLOBAL | |
| GET | `/api/insider-sentiment` | active | GLOBAL | |
| GET | `/api/stocks/{ticker}` | active | GLOBAL | Stock is not a per-user resource |
| POST | `/api/stocks/{ticker}/refresh` | active | GLOBAL | |
| POST | `/api/analyze/scan` | admin | GLOBAL | Producer of AnalysisJob rows |
| GET | `/api/analyze/jobs/{job_id}` | **admin** | GLOBAL | **HARDENED** — was `active`. Mirrors `POST /api/analyze/scan`. |
| GET | `/api/scanner/status` | active | GLOBAL | |
| POST | `/api/scanner/start` | admin | GLOBAL | |
| POST | `/api/scanner/stop` | admin | GLOBAL | |
| PATCH | `/api/scanner/config` | admin | GLOBAL | |
| GET | `/api/portfolio` | active | OK | filters `PortfolioPosition.user_id == current_user.id` |
| POST | `/api/portfolio` | active | OK | sets `user_id=current_user.id` on insert |
| DELETE | `/api/portfolio/{position_id}` | active | OK | filters by user_id |
| PUT | `/api/portfolio/{position_id}` | active | OK | filters by user_id |
| POST | `/api/portfolio/refresh` | active | OK | scoped per user |
| GET | `/api/portfolio/gameplan` | active | OK | filters by user_id |
| GET | `/api/coiled-spring/alerts` | active | GLOBAL | system alerts |
| GET | `/api/coiled-spring/candidates` | active | GLOBAL | |
| GET | `/api/coiled-spring/history` | active | GLOBAL | |
| POST | `/api/coiled-spring/record` | active | GLOBAL | |
| POST | `/api/coiled-spring/cleanup-duplicates` | active | GLOBAL | |
| POST | `/api/coiled-spring/update-outcomes` | active | GLOBAL | |
| GET | `/api/ai-portfolio` | active | OK | filters AIPortfolioPosition + config by user_id |
| GET | `/api/ai-portfolio/history` | active | OK | filters AIPortfolioSnapshot by user_id |
| POST | `/api/ai-portfolio/refresh` | active | OK | captures user_id pre-bg-task |
| GET | `/api/ai-portfolio/trades` | active | OK | filters by user_id |
| POST | `/api/ai-portfolio/initialize` | active | OK | scoped via initialize_ai_portfolio(user_id=) |
| POST | `/api/ai-portfolio/run-cycle` | active | OK | captures user_id pre-bg-task |
| GET | `/api/watchlist` | active | OK | filters by user_id |
| POST | `/api/watchlist` | active | OK | sets user_id on insert |
| DELETE | `/api/watchlist/{item_id}` | active | OK | filters by user_id |
| POST | `/api/watchlist/bulk` | active | OK | filters by user_id |
| POST | `/api/backtests` | active | OK | sets user_id on insert |
| GET | `/api/backtests` | active | OK | filters by user_id |
| GET | `/api/backtests/compare` | active | OK | filters by user_id |
| GET | `/api/backtests/presets` | active | GLOBAL | constant table |
| POST | `/api/backtests/multi` | active | OK | scoped per user |
| POST | `/api/backtests/batch` | active | OK | scoped per user |
| GET | `/api/backtests/queue` | active | GLOBAL | shared queue snapshot |
| GET | `/api/backtests/{backtest_id}` | active | OK | filters by user_id |
| GET | `/api/backtests/{backtest_id}/status` | active | OK | filters by user_id |
| DELETE | `/api/backtests/{backtest_id}` | active | OK | filters by user_id |
| POST | `/api/backtests/{backtest_id}/cancel` | active | OK | filters by user_id |
| POST | `/api/backtests/{backtest_id}/rerun` | active | OK | filters by user_id |
| GET | `/api/market-breadth` | active | GLOBAL | |
| GET | `/api/industry-groups` | active | GLOBAL | |
| GET | `/api/ai-portfolio/correlation` | active | OK | filters by user_id |
| GET | `/api/earnings-gapups` | active | GLOBAL | |
| GET | `/api/bear-base` | active | GLOBAL | |
| GET | `/api/trade-journal` | active | OK | filters AIPortfolioTrade by user_id |
| GET | `/api/analytics/trades` | active | OK | filters by user_id |
| GET | `/api/analytics/exit-quality` | active | OK | filters by user_id |
| GET | `/api/analytics/signal-attribution` | active | OK | filters by user_id |
| GET | `/api/portfolio-summary` | active | OK | filters by user_id |
| GET | `/api/ai-portfolio/earnings-calendar` | active | OK | scoped per user |
| GET | `/api/ai-portfolio/risk` | active | OK | scoped per user |
| PATCH | `/api/ai-portfolio/config` | active | OK | uses get_or_create_config(user_id=) |
| GET | `/api/strategies` | active | GLOBAL | YAML-driven |
| GET | `/api/earnings-audit` | active | GLOBAL | |
| GET | `/api/earnings-audit/{ticker}` | active | GLOBAL | |
| GET | `/api/command-center` | active | OK | aggregates per-user data |

### `backend/routes/auth.py` (`/api/auth`)

| Method | Path | Auth | Category | Notes |
|---|---|---|---|---|
| POST | `/google` | none | PUBLIC | OAuth handshake |
| POST | `/refresh` | none | PUBLIC | refresh_token signature is the gate |
| GET | `/me` | active | OK | returns the caller |
| PATCH | `/me/webhook` | active | OK | mutates the caller |
| POST | `/me/webhook/test` | active | OK | uses caller's webhook |
| GET | `/config` | none | PUBLIC | exposes Google client id only |

### `backend/routes/admin.py` (`/api/admin`)

All routes require `get_admin_user`. GLOBAL.

| Method | Path | Auth | Category |
|---|---|---|---|
| GET | `/users` | admin | GLOBAL |
| POST | `/users` | admin | GLOBAL |
| PATCH | `/users/{user_id}` | admin | GLOBAL |
| GET | `/strategy-health` | admin | GLOBAL |
| GET | `/strategy-ab-eval` | admin | GLOBAL |
| GET | `/strategy-ab-eval-trades` | admin | GLOBAL |
| POST | `/strategy-ab-eval/email-test` | admin | GLOBAL |

### `backend/routes/fidelity.py` (`/api/fidelity`)

| Method | Path | Auth | Category | Notes |
|---|---|---|---|---|
| POST | `/upload-positions` | active | OK | sets user_id |
| POST | `/upload-activity` | active | OK | dedupe filters by user_id |
| GET | `/snapshots` | active | OK | filters by user_id |
| GET | `/latest` | active | OK | filters by user_id |
| GET | `/trades` | active | OK | filters by user_id |
| GET | `/reconciliation` | active | OK | filters by user_id |
| GET | `/gameplan` | active | OK | filters by user_id |
| POST | `/sync-to-portfolio` | active | **OK (FIXED)** | Was GAP — see Findings below |

### `backend/routes/ml.py` (`/api/ml`)

All ML routes are GLOBAL — model state is system-wide and not per-user.

| Method | Path | Auth | Category |
|---|---|---|---|
| POST | `/train` | admin | GLOBAL |
| POST | `/evaluate-oos` | admin | GLOBAL |
| POST | `/diagnose` | admin | GLOBAL |
| GET | `/health` | active | GLOBAL |
| POST | `/fix-active` | admin | GLOBAL |
| GET | `/status` | active | GLOBAL |
| GET | `/predict/{ticker}` | active | GLOBAL |
| GET | `/features` | active | GLOBAL |
| GET | `/validation` | active | GLOBAL |
| GET | `/training-data` | admin | GLOBAL |
| POST | `/compare` | admin | GLOBAL |
| POST | `/compare-matrix` | admin | GLOBAL |
| GET | `/cache-stats` | active | GLOBAL |
| GET | `/matrices` | active | GLOBAL |

### `backend/routes/notifications.py` (`/api/notifications`)

| Method | Path | Auth | Category | Notes |
|---|---|---|---|---|
| GET | `` | active | OK | filters Notification by user_id |
| GET | `/unread-count` | active | OK | filters by user_id |
| POST | `/{notification_id}/read` | active | OK | filters by user_id |
| POST | `/read-all` | active | OK | filters by user_id |
| DELETE | `/{notification_id}` | active | OK | filters by user_id |

### `backend/routes/push.py` (`/api/push`)

| Method | Path | Auth | Category | Notes |
|---|---|---|---|---|
| GET | `/vapid-public-key` | none | PUBLIC | Key is meant to be exposed |
| POST | `/subscribe` | active | OK | upsert by endpoint, reassigns to caller |
| GET | `/subscriptions` | active | OK | filters by user_id |
| DELETE | `/subscriptions/{subscription_id}` | active | OK | filters by user_id |
| POST | `/test` | active | OK | sends to caller's subs only |

## Findings

### F-1 (CRITICAL) — `POST /api/fidelity/sync-to-portfolio` cross-tenant write

**File:** `backend/routes/fidelity.py:1065-1133`
**Status:** FIXED in this commit.

Three IDOR bugs in the same handler:

1. Line 1090 (pre-fix): `db.query(PortfolioPosition).filter(PortfolioPosition.ticker == fp.symbol).first()` — no user_id filter. Bob's sync would *overwrite* Alice's manual position rows for any ticker shared between Bob's Fidelity snapshot and Alice's manual portfolio.
2. Line 1102 (pre-fix): `PortfolioPosition(...)` constructor never sets `user_id`. New rows landed with `user_id=NULL`, invisible to every user.
3. Line 1116 (pre-fix): `~PortfolioPosition.ticker.in_(fid_symbols)` with no user_id filter — Bob's call DELETED every PortfolioPosition row across the whole tenant base whose ticker did not match a ticker in Bob's Fidelity snapshot. Mass cross-tenant data destruction by any active user.

**Fix:** Add `PortfolioPosition.user_id == current_user.id` to all three queries; set `user_id=current_user.id` on insert.

**Eval-safety:** `PortfolioPosition` is the *manual* portfolio (Portfolio page, hand-entered positions). It is not used by the AI Trader, scoring, ML, or backtester pipelines. Fix is fully orthogonal to the 2026-06-18 A/B eval.

### F-2 (MEDIUM) — `GET /api/analyze/jobs/{job_id}` privilege asymmetry

**File:** `backend/main.py:1790`
**Status:** FIXED in this commit.

`AnalysisJob` rows have no `user_id` column — they are system-level scan-job records. The producer `POST /api/analyze/scan` (line 1667) is admin-gated, but the reader was active-user-gated. A non-admin authenticated user could enumerate scan jobs and read scan progress/error_message. Not strictly IDOR (no per-user resource), but a privilege-scope mismatch.

**Fix:** Change auth dep to `get_admin_user` to mirror the producer.

**Eval-safety:** Read-only metadata endpoint. No scoring/ML/trading code path touched.

## Deferred (Future CSO Sessions)

Named here so they don't get lost; each warrants its own ~2-3hr slot once IDOR ships.

1. **Secrets / credentials audit.** Scan `.env`, docker-compose, settings for plaintext secrets, weak defaults (`JWT_SECRET_KEY=dev-secret-key-change-in-production` is a real-world risk if `REQUIRE_AUTH=true` ever flips on a misconfigured host), key-rotation story.
2. **Dependency CVE scan.** Run `pip-audit` / `npm audit` against backend + frontend lockfiles; review `yfinance`, `aiohttp`, `python-jose` (the JOSE library has had multiple CVEs).
3. **Fidelity CSV upload hardening.** `routes/fidelity.py` accepts arbitrary CSV bytes and parses them. Audit for: file-size DoS, parser exception leaks, multipart-form-bypass on `endswith('.csv')`, CSV injection (`=cmd|...`) on any downstream re-export path.
4. **Rate-limit audit on admin endpoints.** `POST /api/ml/train`, `POST /api/admin/strategy-ab-eval/email-test`, `POST /api/scanner/start` are all expensive operations gated only by `is_admin`. A compromised admin token (or a foot-gun script) could DoS the box. Add per-route rate limits and idempotency keys.
5. **`POST /api/system/backup` privilege check.** Currently `active`-only; should likely be admin. Out of IDOR scope (system-wide resource).
6. **`POST /api/push/subscribe` endpoint-takeover semantics.** The handler upserts by endpoint and reassigns user_id to the caller. Endpoint URLs are vendor-issued opaque tokens, but document the threat model explicitly.
