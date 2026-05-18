# CANSLIM Analyzer - Project Context

## Overview
A mobile-first web application for CANSLIM stock analysis with React frontend and FastAPI backend, deployed via Docker on a VPS.

**GitHub Repository**: [BigMike-Deane/canslim_analyzer](https://github.com/BigMike-Deane/canslim_analyzer)

## Architecture
- **Frontend**: React + Vite + TailwindCSS (mobile-first design)
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Cache**: 3-tier (Memory → Redis → DB → API fetch)
- **Deployment**: Docker (3 containers + 3 volumes) on VPS at `/opt/canslim_analyzer`
- **Docker command**: Use `docker-compose` (with hyphen, old version on VPS)
- **Container name**: `canslim-analyzer`
- **Port**: 8001

## CRITICAL Rules
- **Multi-container VPS**: NEVER use `docker rm -f $(docker ps -aq)` — kills Finance Tracker too
- **AI Trader <> Backtester MUST stay in sync**: Every change to `ai_trader.py` trading logic MUST be mirrored in `backtester.py`
- **Asyncio primitives**: MUST create inside async context (not at module level)
- **Database sessions**: Always use try/finally
- **Score details**: Use lowercase keys `score_details["c"]`
- **Timezone**: Add `.replace(tzinfo=timezone.utc)` if naive
- **`scripts/` directory**: NOT copied into Docker image
- **FastAPI routes**: Literal routes BEFORE parameterized routes
- **Backtester universe**: MUST be "all" (winning stocks are mid/small-caps)
- **FMP API**: `/stable/` endpoints only (no batch); `/api/v3/` requires legacy subscription
- **Earnings avoidance**: `avoidance_days` != `allow_buy_days` — different purposes, NOT off-by-one
- **Force push**: User prefers normal push; avoid rewriting shared history

## Configuration System (YAML-based)
- `config/default.yaml` - Base configuration for all environments
- `config/development.yaml` - Dev overrides
- `config/production.yaml` - Production settings
- `CANSLIM_ENV=production` selects environment (default: development)

```python
from config_loader import config
workers = config.get('scanner.workers', default=4)
config.reload()  # Hot-reload
```

Key sections: `scanner`, `cache`, `scoring.canslim`, `scoring.growth_mode`, `market.indexes`, `ai_trader`, `technical`, `api`, `strategy_profiles`, `coiled_spring`

## Strategy System
7 profiles in YAML, configurable via API and frontend.
**Winner**: `nostate_optimized` — market state DISABLED, binary SPY gate only (SPY < 50MA = no buys).

Champion config: min_score=72, max_positions=8, stop_loss=7%, take_profit=75%, seed_count=4, trailing stops (50+: 25%, 30-50: 18%, 20-30: 12%, 10-20: 8%)

Key finding: 5-state market state machine HURTS over full cycles. NoState's binary gate is crude but far more effective (8.6x better over 4yr).

## Key Features
- **CANSLIM Scoring** (100pts): C(15) A(15) N(15) S(15) L(15) I(10) M(15)
- **Growth Mode Scoring** (100pts): R(20) F(15) N(15) S(15) L(15) I(10) M(10) for pre-revenue stocks
- **AI Trading**: Trailing stops, partial profits (25%/40% tiers), score crash detection, sector limits
- **Backtesting**: Full day-by-day simulation with SPY benchmark, trade history
- **Coiled Spring**: Earnings catalyst detection — base patterns + beat streaks + approaching earnings
- **Base Patterns**: Flat base, cup, cup-with-handle, double bottom detection
- **Breakout Detection**: Pre-breakout (-3% to 0%), active (0% to +5%), extended (>5% penalized)
- **Market Breadth**: A/D ratio, new highs/lows, sector rotation
- **Fidelity Sync**: CSV upload, position tracking, trade parsing
- **Watchlist Alerts**: Email notifications on price/score targets
- **Data Sources**: FMP (earnings, metrics), Yahoo Finance (prices, fallback), Finviz (institutional)

## Admin Diagnostics
- `GET /api/admin/strategy-health` — pre/post-graduation health audit for a strategy.
- `GET /api/admin/strategy-ab-eval` — live A/B comparison framework: pre vs post-cutoff trade summary + decision (keep/revert/marginal/insufficient_data). Used to evaluate scoring-rule experiments shipped to live trading; backtest replay can't honestly evaluate scoring changes (snapshots freeze today's point-in-time scalars). First consumer: Approach 2 (commit `ec73f83`, deployed 2026-05-07).

## File Structure

### Root Directory
- `canslim_scorer.py` - CANSLIM scoring logic
- `data_fetcher.py` / `async_data_fetcher.py` - Data fetching (sync/async)
- `async_scanner.py` - Async batch scanning
- `growth_projector.py` - Growth projection model
- `sp500_tickers.py` - Stock universe management (~2000+ tickers)
- `config_loader.py` - YAML configuration loader
- `redis_cache.py` - Redis caching layer
- `email_report.py` - Email notifications

### Backend (`/backend/`)
- `main.py` - FastAPI endpoints + shared helpers (~4,750 lines)
- `routes/fidelity.py` - Fidelity sync routes (extracted from main.py)
- `database.py` - SQLAlchemy models + migrations
- `scheduler.py` - Continuous scanning logic
- `ai_trader.py` - AI Portfolio trading logic
- `backtester.py` - Historical backtesting engine
- `historical_data.py` - Historical data provider
- `market_state.py` - Market state machine (disabled in winner strategy)
- `fidelity_sync.py` - Fidelity CSV parsing

### Frontend (`/frontend/src/`)
- `pages/` - Dashboard, AIPortfolio, Backtest, Analytics, Breadth, Screener, etc.
- `components/` - Sidebar, shared UI components
- `api.js` - API client with TTL-based caching
- `App.jsx` - Main application router

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

### Tests (`/tests/`)
- 576 tests passing, 5 skipped (Redis tests when unavailable)
- `conftest.py` - Shared fixtures with in-memory SQLite
- Key files: `test_backtester.py`, `test_canslim_scorer.py`, `test_bug_regressions.py`

#### Test Isolation: FastAPI Dependency Overrides
Use the canonical primitives in `tests/conftest.py` for any test that needs to override `get_current_user`, `get_current_admin_user`, or `get_db`. Module-level `app.dependency_overrides[...] = ...` collides when multiple test files override the same key — pytest collection imports every file before any test runs, so last-loaded wins and tests pass/fail based on collection order.

Canonical IDs (defined in `conftest.py`, in the 99000+ range to avoid collision with hand-rolled fixtures):
- `TEST_ADMIN_ID = 99001` — admin user
- `TEST_USER_A_ID = 99002` — non-admin user A
- `TEST_USER_B_ID = 99003` — non-admin user B
- `TEST_NONADMIN_ID = 99004` — generic non-admin

Scoped override (preferred — restores prior state on exit, even if test raises):
```python
from tests.conftest import override_dependency, TEST_ADMIN_ID

def test_admin_endpoint(client):
    with override_dependency(get_current_admin_user, lambda: User(id=TEST_ADMIN_ID, is_admin=True)):
        resp = client.get("/api/admin/foo")
    # override automatically cleared here
```

Don't reach for module-level `app.dependency_overrides[...] = ...` in new tests — it's the pattern we migrated 9 files off of in `f946080`.

## Deployment Commands
See CLAUDE.local.md for deployment commands with actual VPS addresses.
```bash
# Standard deploy pattern
ssh root@$VPS_IP 'cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build'

# Start backtest (nostate_optimized)
curl -X POST https://canslim.duckdns.org/api/backtests -H "Content-Type: application/json" \
  -d '{"start_date": "2022-01-01", "end_date": "2026-02-19", "starting_cash": 25000, "stock_universe": "all", "strategy": "nostate_optimized"}'
```

## Testing
```bash
export CANSLIM_ENV=development
python3 -m pytest tests/ -v                           # All tests
python3 -m pytest tests/ --cov=. --cov-report=html    # With coverage
python3 -m pytest tests/test_backtester.py -v          # Specific file
```

## Dependencies
- **Backend**: fastapi, uvicorn, sqlalchemy, yfinance>=0.2.40, pandas, numpy, aiohttp, httpx, pyyaml, redis, psycopg2-binary
- **Frontend**: React 18, Vite, TailwindCSS, Recharts

## Owner's Trading Preferences
- Likes stocks under $25 that fit CANSLIM criteria
- Default scan: All Stocks at 90-minute intervals
- Prefers actionable pre-breakout entries over chasing extended stocks
