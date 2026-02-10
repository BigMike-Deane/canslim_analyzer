# CANSLIM Analyzer - Project Context

## Session Summary: Feb 9, 2026

### Strategy Profiles (Growth Mode) - DEPLOYED

**Purpose**: Added a first-class `strategy` parameter that controls thresholds, stops, position sizing, and scoring weights. Currently supports "balanced" (default) and "growth" profiles.

**Implementation**:
1. **YAML Config** (`config/default.yaml`): Added `strategy_profiles` section with `balanced` and `growth` profiles. Each profile controls: min_score, max_positions, position sizing, stop loss, trailing stops, take profit, scoring weights, quality filters, and C/L/N score multipliers.

2. **Database** (`backend/database.py`): Added `strategy` column (String, default "balanced") to both `BacktestRun` and `AIPortfolioConfig`.

3. **Backtester** (`backend/backtester.py`): `get_strategy_profile()` helper loads profile dict. `__init__` reads strategy from BacktestRun record. `_seed_initial_positions`, `_evaluate_buys`, `_evaluate_sells` all use profile overrides.

4. **AI Trader** (`backend/ai_trader.py`): Same profile loading in `evaluate_buys()`, `evaluate_sells()`, and `check_and_execute_stop_losses()`. Profile overrides for stops, trailing, take profit, score crash, and composite scoring weights.

5. **API** (`backend/main.py`): `BacktestCreate` accepts `strategy` field. Multi-period endpoint accepts `strategy` query param. AI portfolio config PATCH accepts strategy updates.

6. **Frontend** (`frontend/src/pages/Backtest.jsx`): Strategy dropdown in new backtest form. Purple "Growth" badge on growth backtests. (`frontend/src/pages/AIPortfolio.jsx`): Growth Mode badge on header.

**Growth Mode Iterations**:
| Version | min_score | stop_loss | take_profit | trailing_50+ | Result |
|---------|-----------|-----------|-------------|---------------|--------|
| v1 | 60 | 10% | 60% | 18% | +3.2% (too loose) |
| v2 | 68 | 8% | 55% | 15% | +16.0% (tighter risk) |
| v3 | 68 | 8% | 75% | 20% | **+23.7%** (ride winners) |

**Key Insight**: The "ride winners" sell logic (wider trailing for big gains, higher take profit, lenient score crash) was universally beneficial — not growth-specific.

### Ride-Winners Sell Logic Merged into Balanced Default - DEPLOYED

Merged the proven sell improvements from growth v3 into the balanced profile:
- **Take profit**: 40% → **75%** (let big winners run)
- **Trailing stop at 50%+ gain**: 15% → **20%** (wider room for volatile winners)
- **Trailing stop at 30-50% gain**: 12% → **15%**
- **Score crash drop required**: 20pt → **25pt** (less reactive to score noise)
- **Score crash ignore if profitable**: 10% → **20%** (hold profitable positions through dips)

**6-month backtest (#91)**: +19.0% vs SPY +9.3%, Sharpe 2.39, DD 3.9%, WR 72.7%

### ALL View Chart Bug - FIXED

**Problem**: Performance chart showed 61 data points in 24H view but only 2 in ALL view.
**Root Cause**: `filterHistory()` in AIPortfolio.jsx deduped to latest-per-day for non-24h views when >60 points. With only 2 unique dates, this collapsed 61 intraday points to 2.
**Fix**: Only dedup when 7+ unique days exist. With fewer days, all intraday points are preserved.

### Removed Editable Config Parameters from AI Portfolio UI

Removed the editable "Min Score to Buy", "Sell Below Score", "Take Profit %", and "Stop Loss %" inputs from the AI Portfolio config panel. These conflicted with the strategy profile system — the profile controls all these values. Replaced with clean read-only displays showing active values + strategy name.

**Files Modified**:
- `config/default.yaml` - `strategy_profiles` section (balanced + growth)
- `backend/database.py` - `strategy` column on BacktestRun + AIPortfolioConfig
- `backend/backtester.py` - `get_strategy_profile()`, profile overrides in buy/sell/seed
- `backend/ai_trader.py` - Profile overrides in evaluate_buys/sells, stop losses
- `backend/main.py` - Strategy in BacktestCreate, multi-period, AI portfolio config
- `frontend/src/pages/Backtest.jsx` - Strategy dropdown + badge
- `frontend/src/pages/AIPortfolio.jsx` - Chart fix, config cleanup, strategy badge
- `frontend/src/api.js` - Strategy param in multi-backtest
- `tests/test_backtester.py` - TestStrategyProfiles (9 tests)

**Tests**: 266 passed, 5 skipped, 0 failures

### Backtest Results (This Session)

| # | Strategy | Return | vs SPY | Sharpe | Max DD | Win Rate | Notes |
|---|----------|--------|--------|--------|--------|----------|-------|
| #87 | balanced | +21.9% | +5.6% | 1.43 | 11.0% | 56.0% | 1yr baseline |
| #88 | growth v1 | +3.2% | -13.1% | 0.17 | 17.7% | 40.9% | Too loose |
| #89 | growth v2 | +16.0% | -0.3% | 0.78 | 10.4% | 46.9% | Tightened risk |
| #90 | growth v3 | **+23.7%** | **+7.4%** | 1.14 | 10.6% | 54.5% | Ride winners |
| #91 | balanced (new) | **+19.0%** | **+9.3%** | **2.39** | **3.9%** | **72.7%** | 6mo, ride-winners merged |

---

## Session Summary: Feb 5, 2026

### Scanner Only Processing 12% of Stocks - FIXED

**Problem**: Scans were only completing 238/1916 stocks (12.4% success rate) instead of all stocks.

**Symptoms**:
- Most stocks showed "CACHE HIT" for earnings data
- But stocks silently failed before returning valid StockData
- No visible errors (failures logged at DEBUG level)

**Root Cause**: Module-level `asyncio.Semaphore` and `asyncio.Lock` in `async_data_fetcher.py` were bound to the event loop that existed when the module loaded. When `asyncio.run()` creates a new event loop for each scan, these objects became invalid:
```
RuntimeError: <asyncio.locks.Semaphore object at 0x...> is bound to a different event loop
```

**Fix** (`ad5b8fb`): Initialize asyncio primitives at the START of each scan:
```python
def _init_async_primitives():
    """Initialize asyncio primitives for the current event loop"""
    global api_semaphore, _rate_lock
    api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    _rate_lock = asyncio.Lock()
    # Reset rate limiter state for fresh scan
    _rate_limiter["calls_this_minute"] = 0
    # ... etc
```

Called at start of `fetch_stocks_batch_async()`.

**Diagnostic Logging Added** (`1eb7522`): Batch exceptions now logged at WARNING level:
```python
if isinstance(result, Exception):
    logger.warning(f"BATCH EXCEPTION for {batch[j]}: {type(result).__name__}: {result}")
```

**Result**: Scans now complete 93-100% of stocks with 0 asyncio exceptions. Remaining failures are legitimately invalid tickers (return 404 from Yahoo).

**Files Modified**:
- `async_data_fetcher.py` - Added `_init_async_primitives()`, diagnostic logging

**Verification**:
- Self-healing mechanism working: 2,176 tickers auto-removed from delisted list when they started working
- 171 delisted tickers are legitimately invalid (all return HTTP 404 from Yahoo)

---

## Session Summary: Feb 4, 2026 (Late Night)

### Watchlist Alerts Feature - DEPLOYED

**Purpose**: Email notifications when watchlist stocks hit target price or CANSLIM score thresholds.

**Commits**: `3b6dc17`, `0f32e6c`, `e0f1636`

**Files Modified**:
- `backend/database.py` - Added Watchlist alert tracking fields (`alert_triggered_at`, `alert_sent`, `last_check_price`)
- `backend/scheduler.py` - Added `check_watchlist_alerts()` function, integrated into scan workflow
- `email_report.py` - Added `send_watchlist_alert_email()` function
- `config/default.yaml` - Added `watchlist.alerts` configuration section

**Configuration** (`config/default.yaml`):
```yaml
watchlist:
  alerts:
    enabled: true
    check_after_scan: true
    cooldown_hours: 24
```

**How It Works**:
1. Add stocks to watchlist with `target_price` and/or `alert_score`
2. After each scan, `check_watchlist_alerts()` runs
3. If price >= target OR score >= threshold, email is sent
4. 24-hour cooldown prevents duplicate alerts

### Coiled Spring Performance Dashboard - DEPLOYED

**Enhanced `/api/coiled-spring/history` endpoint** with cumulative stats:
- `total_alerts_all_time` - Total alerts ever recorded
- `overall_win_rate` - Win rate across all alerts (not just current page)
- `by_base_type` - Breakdown by pattern type with win rates per pattern

**Frontend Component**: `CoiledSpringStats` in Dashboard.jsx shows:
- Overall win rate, wins, losses, big wins
- Best performing base pattern

### Institutional Percentage Bug - FIXED

**Problem**: Some stocks showed 1% institutional when they should show 100%+.

**Root Cause**: Yahoo's `heldPercentInstitutions` returns decimals (0.65 = 65%, 1.00002 = 100.002%). The condition `inst_pct <= 1` failed to convert edge cases like 1.00002.

**Fix**: Changed threshold from `<= 1` to `<= 1.5` in:
- `async_data_fetcher.py` (lines 1032, 1132)
- `data_fetcher.py` (line 1801)

**Data Migration**: Fixed 528 stocks in database with bad institutional_pct values.

### GOOG/GOOGL Deduplication - FIXED

**Problem**: Both GOOG and GOOGL appeared in `/api/stocks` results.

**Fix**: Added `filter_duplicate_stocks()` call to `/api/stocks` endpoint in `backend/main.py`.

**Verified Working** on all endpoints:
- `/api/stocks` - GOOGL only
- `/api/dashboard` - GOOGL only
- `/api/top-growth-stocks` - GOOGL only
- `/api/stocks/breaking-out` - GOOGL only

### Scanner Checkpoint Logging - IMPROVED

**Problem**: Scanner only processed 263/1916 stocks due to stale checkpoint.

**Fix**: Enhanced logging in `async_data_fetcher.py`:
- `load_scan_progress()` now logs checkpoint scan_id, completed count, and age
- `clear_scan_progress()` now logs success/failure instead of silent pass

### Test Coverage - ADDED

**New File**: `tests/test_watchlist_alerts.py` (17 tests)
- Price alert trigger conditions
- Score alert trigger conditions
- Cooldown prevention
- Alert_sent flag handling
- Email content generation
- Institutional percentage extraction from score_details

### Verified Existing Features

**StockScore Cleanup**: Working - 0 records older than 30 days (auto-cleanup after scans)

**Frontend API Caching**: Working - TTL-based cache with invalidation on mutations

---

## Session Summary: Feb 4, 2026 (Evening)

### A Score = 0 Bug - FIXED

**Problem**: All stocks showing A=0 or A=N/A, AI Portfolio scores dropped 10-15 points.

**Root Causes** (4 layers):
1. Cache limit too small (500 vs 2500 needed)
2. Cache key mismatch: `load_cache_from_db()` used "quarterly"/"annual" but code expected "quarterly_eps"/"annual_eps"
3. `main.py analyze_stock()` wasn't including annual_eps in score_details
4. `save_stock_to_db()` missing line to save score_details

**Files Fixed**: `data_fetcher.py`, `async_data_fetcher.py`, `backend/main.py`, `config/default.yaml`

### Individual Scores = 0 Bug - FIXED (`ce2b9cc`)

**Problem**: Frontend showed all C/A/N/S/L/I scores as 0, but total score was correct.

**Root Cause**: Case mismatch - score_result used lowercase keys ("c", "a") but code accessed uppercase ("C", "A").

**Fix**: Changed lines 514-520 in `backend/main.py` to use lowercase keys.

### Coiled Spring Improvements - PRE-BREAKOUT FOCUS

**Problem**: CS was catching stocks AFTER they spiked (like SLAB with 16x volume), not before.

**Solution**: Added pre-breakout preference and relaxed thresholds.

**New Commits**: `ce2b9cc`, `7784782`, `7bbad04`, `a38824a`

**Standard Thresholds** (for BREAKING_OUT stocks):
- `min_weeks_in_base: 15`, `min_beat_streak: 3`, `min_c_score: 10`
- `min_total_score: 55`, `max_institutional_pct: 75`, `min_l_score: 6`

**Pre-Breakout Thresholds** (relaxed - catalyst hasn't happened):
- `min_c_score: 5` (lower C ok - earnings catalyst pending)
- `min_total_score: 48` (will improve after earnings)

**Quality Ranking Bonuses/Penalties**:
- `pre_breakout_bonus: +15` - Ideal entry, hasn't moved yet
- `extended_penalty: -20` - Already spiked (>2x volume at high)
- `low_inst_bonus: +10` - Room to run if inst < 30%

**Entry Status Categories**:
- **PRE-BREAKOUT**: Ideal entry (gets +15 bonus)
- **BREAKING_OUT**: Active breakout with normal volume
- **EXTENDED**: Already spiked (penalized -20)

**Success Tracking Endpoints**:
- `GET /api/coiled-spring/history` - View past alerts with win rates
- `POST /api/coiled-spring/record?ticker=XYZ` - Record alert for tracking
- `POST /api/coiled-spring/update-outcomes` - Update outcomes after earnings

**Current Top Candidates** (Feb 4):
1. CROX (166.4) - PRE-BREAKOUT, 26w cup_w_handle, 22 beats, -29.9% from high
2. MRNA (140.3) - PRE-BREAKOUT, 26w cup_w_handle, 11 beats, -24.0% from high
3. OI (125.5) - PRE-BREAKOUT, 26w cup, 4 beats, -0.9% from high
4. RSI (116.2) - PRE-BREAKOUT, 26w cup, 3 beats, -22.9% from high
5. STNG (115.2) - BREAKING_OUT, 26w cup, 9 beats
6. KO (108.7) - PRE-BREAKOUT, 15w dbl_btm, 7 beats

### Diagnostic Commands

```bash
# Check CS candidates (ranked by quality)
curl -s "http://100.104.189.36:8001/api/coiled-spring/candidates?limit=10" | python3 -m json.tool

# Pre-breakout only (ideal entries)
curl -s "http://100.104.189.36:8001/api/coiled-spring/candidates?pre_breakout_only=true" | python3 -m json.tool

# Check CS history and success rates
curl -s "http://100.104.189.36:8001/api/coiled-spring/history" | python3 -m json.tool

# Record a CS alert for tracking
curl -X POST "http://100.104.189.36:8001/api/coiled-spring/record?ticker=CROX"

# Update outcomes after earnings
curl -X POST "http://100.104.189.36:8001/api/coiled-spring/update-outcomes"
```

---

## Session Summary: Feb 4, 2026 (Earlier)

### Coiled Spring / Earnings Catalyst Feature - DEPLOYED

**Purpose**: Identify stocks with explosive earnings potential BEFORE they move - long bases + beat streaks + approaching earnings.

**Commits**: `6fb1f75`, `b0daf14`, `7f0154c`, `d9d4e54`, `94e6b15`

**Files Modified**:
- `config/default.yaml` - Added `coiled_spring:` configuration section
- `canslim_scorer.py` - Added `calculate_coiled_spring_score()` function
- `backend/database.py` - Added `CoiledSpringAlert` model
- `backend/ai_trader.py` - CS detection, bonus scoring, position sizing, alert recording
- `backend/backtester.py` - Mirrored CS logic for historical testing
- `backend/main.py` - Added `/api/coiled-spring/alerts` and `/api/coiled-spring/candidates` endpoints
- `email_report.py` - Added CS alerts section to email reports
- `frontend/src/pages/Dashboard.jsx` - Added `CoiledSpringAlerts` component (purple card)
- `frontend/src/pages/AIPortfolio.jsx` - Added collapsible CS alerts section
- `frontend/src/api.js` - Added `getCoiledSpringAlerts()` and `getCoiledSpringCandidates()`
- `tests/test_coiled_spring.py` - 23 unit tests

### Earlier Bug Fixes

**1. Scan Session Timeout** - FIXED (`94e6b15`)
- Problem: Scans stopped at ~100-130 stocks instead of completing all 1917
- Root cause: `aiohttp.ClientTimeout(total=60)` timed out the ENTIRE session after 60 seconds
- Fix: Changed to per-request timeouts: `total=None, connect=30, sock_read=60`
- File: `async_data_fetcher.py` line 1471

**2. institutional_holders_pct AttributeError** - FIXED (`d9d4e54`)
- Problem: Growth stocks API returned 500 error
- Root cause: Stock model doesn't have `institutional_holders_pct` column - data is in `score_details` JSON
- Fix: Extract from `(stock.score_details or {}).get('i', {}).get('institutional_pct')`
- File: `backend/main.py`

**3. P1 Fields Missing from API** - FIXED (`b0daf14`)
- Problem: `days_to_earnings`, `earnings_beat_streak`, `institutional_holders_pct` not returned by stock APIs
- Fix: Added fields to `/api/stocks` list and `/api/stocks/{ticker}` detail endpoints
- File: `backend/main.py`

---

## Session Summary: Feb 3, 2026

### Bug Fixes Deployed (commit `17437a6`)

**1. Stuck Backtest Cancel Button** - FIXED
- Problem: Backtest stuck at 30% with non-functional cancel button
- Root cause: Cancel endpoint only set a `cancel_requested` flag, but orphaned processes never checked it
- Fix: Cancel endpoint now detects stuck backtests (running >2 hours) and directly marks them as cancelled
- File: `backend/main.py` lines 2634-2676

**2. Auto-Refresh Toggle UI** - FIXED
- Problem: Toggle appeared in "on" position but showed "Disabled" text
- Root cause: CSS positioning used `translate-x-1`/`translate-x-7` without explicit `left` anchor
- Fix: Added `left-1` base position, changed to `translate-x-6`/`''` for proper knob positioning
- File: `frontend/src/pages/AIPortfolio.jsx` lines 863-875

**3. Progress Counter Showing Wrong Totals** - FIXED
- Problem: Scan showed 263/1917 then jumped to 3/3
- Root cause: `update_progress()` callback was overwriting `total_stocks` for all phases (stocks, insider_short, p1_data)
- Fix: Only update `stocks_scanned` and `total_stocks` when `phase == "stocks"`
- File: `backend/scheduler.py`

**4. Insider/Short Progress Showing 3175 Instead of 1917** - FIXED
- Problem: Progress was counting combined API tasks (insider + short) instead of unique tickers
- Fix: Track unique tickers processed using a set instead of task count
- Files: `async_scanner.py` - `fetch_insider_short_batch_async()` and `fetch_p1_data_batch_async()`

### Verifications Completed

**Stock Count Verified**: 1917 stocks is accurate
- 167 delisted tickers are legitimately acquired/privatized companies
- Examples: JNPR (acquired by HPE), SAGE (acquired by Supernus), SQSP (acquired by Permira)
- Self-healing mechanism working: false delistings auto-corrected

**Scan Health Verified**: Completed scan showed healthy metrics
- 1917/1917 stocks scanned
- 0 rate limits hit
- 383 "Invalid Crumb" errors handled gracefully (Yahoo session refresh working)

### Performance Optimization Plan (Planned, Not Yet Implemented)

A detailed plan exists at `/home/bayer/.claude/plans/curried-imagining-balloon.md` for:
1. **StockScore Cleanup** - Delete records >30 days old to prevent DB bloat (400K+ records)
2. **Stock Detail Background Refresh** - Return stale cache immediately, refresh in background
3. **Frontend API Caching** - Memory cache with TTL for GET requests

### Quick Reference Commands

```bash
# Check scanner status
curl -s http://100.104.189.36:8001/api/scanner/status | python3 -m json.tool

# Cancel a stuck backtest
curl -X POST http://100.104.189.36:8001/api/backtests/{id}/cancel

# Check for running backtests
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, BacktestRun; db = SessionLocal(); [print(f'ID {b.id}: {b.status}') for b in db.query(BacktestRun).filter(BacktestRun.status.in_(['running','pending'])).all()]; db.close()"

# Standard deploy
ssh root@100.104.189.36 "cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build"
```

---

## RESOLVED: P1 Features Fix (Feb 2, 2026)

**Status**: DEPLOYED & VERIFIED

P1 data (earnings beat_streak, days_to_earnings, analyst revisions) now populating correctly using FMP `/stable/earnings` endpoint instead of `/stable/earnings-calendar`.

---

## RESOLVED: Market Cap & 52-Week Low Fix (Feb 1, 2026)

**Status**: FIXED - Data is now being saved correctly. More stocks gaining data with each scan cycle.

**Solution Summary**:
The issue was that the FMP `/api/v3/` batch endpoints require a legacy subscription (before Aug 2025). We switched to fetching individual quotes/profiles via `/stable/` endpoints, which work but are slower. The fix is working - user confirmed "seeing several new stocks with market caps and 52 week lows now that it has scanned a few times throughout the night."

**Code Cleanup Done**:
- Removed duplicate `week_52_low` assignment in `scheduler.py` (was at lines 365 AND 382, now only at 381)

**Data Flow (working correctly)**:
1. `fetch_fmp_single_quote()` → returns `market_cap`, `low_52w`, `high_52w` from FMP `/stable/quote`
2. `fetch_fmp_single_profile()` → fallback with `mktCap` and range parsing for 52w data
3. `get_stock_data_async()` → applies quote/profile data to StockData object with Yahoo fallbacks
4. `analyze_stocks_async()` → builds result dict with `market_cap` (line 174), `week_52_low` (line 223)
5. `save_stock_to_db()` → saves to database at lines 364 (market_cap), 381 (week_52_low)

**Verification Commands** (run on VPS):
```bash
# Check portfolio stocks data
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock, PortfolioPosition; db = SessionLocal(); tickers = [p.ticker for p in db.query(PortfolioPosition.ticker).distinct().all()]; stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all(); [print(f'{s.ticker}: mktcap={int(s.market_cap) if s.market_cap else 0}, 52wLow={s.week_52_low or 0}') for s in stocks]; db.close()"

# Check overall data coverage
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock; db = SessionLocal(); total = db.query(Stock).count(); with_data = db.query(Stock).filter(Stock.market_cap > 0, Stock.week_52_low > 0).count(); print(f'Stocks with BOTH market_cap AND week_52_low: {with_data}/{total} ({with_data/total*100:.1f}%)'); db.close()"
```

**Why Some Stocks Initially Had No Data**:
- Rate limiting on first scan passes caused some API calls to fail
- Portfolio stocks are prioritized and get processed first
- Each subsequent scan fills in more stocks as retry attempts succeed
- Expected: 90%+ coverage after 3-4 complete scan cycles

---

## Overview
A mobile-first web application for CANSLIM stock analysis with React frontend and FastAPI backend, deployed via Docker on a VPS.

**GitHub Repository**: [BigMike-Deane/canslim_analyzer](https://github.com/BigMike-Deane/canslim_analyzer)
- **Languages**: Python (77.9%), JavaScript (21.5%)
- **Commits**: 211+

## Architecture
- **Frontend**: React + Vite + TailwindCSS (mobile-first design)
- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Deployment**: Docker on VPS at `/opt/canslim_analyzer`
- **Docker command**: Use `docker-compose` (with hyphen, old version)
- **Container name**: `canslim-analyzer`
- **Port**: 8001
- **VPS IP**: 147.93.72.73

## Configuration System (YAML-based)

Environment-based configuration with hot-reloading support.

### Config Files
- `config/default.yaml` - Base configuration for all environments
- `config/development.yaml` - Dev overrides (faster scans, shorter cache, 2 workers)
- `config/production.yaml` - Production settings (8 workers, full cache intervals)

### Usage
```python
from config_loader import config

# Get specific values
workers = config.get('scanner.workers', default=4)
cache_ttl = config.get('cache.freshness_intervals.earnings')

# Get entire sections
scanner_config = config.scanner
ai_trader_config = config.ai_trader

# Hot-reload config
config.reload()
```

### Environment Variable
```bash
export CANSLIM_ENV=production  # or development (default)
```

### Key Configuration Sections
| Section | Description |
|---------|-------------|
| `scanner` | Workers, delays, batch size, retries |
| `cache` | Redis settings, freshness intervals |
| `scoring.canslim` | Max scores and weights for C/A/N/S/L/I/M |
| `scoring.growth_mode` | Growth Mode scoring parameters |
| `market.indexes` | SPY/QQQ/DIA weights |
| `ai_trader` | Trailing stops, insider signals, short penalties |
| `technical` | Base pattern thresholds, breakout detection |
| `api` | Rate limits for FMP/Yahoo/Finviz |

## 3-Tier Cache Hierarchy

```
User Request
    ↓
Memory Cache (instant) → HIT? → Return
    ↓ MISS
Redis Cache (milliseconds) → HIT? → Store in Memory → Return
    ↓ MISS
DB Cache (slower) → HIT? → Store in Redis + Memory → Return
    ↓ MISS
API Fetch → Store in all 3 caches → Return
```

### Expected Cache Hit Rates
- Memory cache: 5-10% (hot data)
- Redis cache: 60-70% (warm data)
- DB cache: 10-15% (cold start)
- API fetch: 15-25% (new/stale data)

### Redis Cache Usage
```python
from redis_cache import redis_cache

# Set data (with automatic TTL from config)
redis_cache.set("AAPL", "earnings", earnings_data)

# Get data
earnings = redis_cache.get("AAPL", "earnings")

# Get stats
stats = redis_cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}%")
```

## CRITICAL: Multi-Container VPS Setup
**This VPS runs multiple applications. NEVER use `docker rm -f $(docker ps -aq)` as it kills ALL containers including other apps.**

Other apps on the VPS:
- **Finance Tracker**: `/opt/finance-tracker` (separate container)

### Safe Deployment Command (USE THIS)
```bash
cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build
```

### If ContainerConfig error occurs
```bash
cd /opt/canslim_analyzer && docker-compose down && docker rm -f canslim-analyzer 2>/dev/null; docker-compose up -d --build
```

## Key Features Implemented

### CANSLIM Scoring (100 points total)
- **C** (15 pts): Current quarterly earnings - TTM growth + EPS acceleration bonus
- **A** (15 pts): Annual earnings - 3-year CAGR + ROE quality check (17%+ threshold)
- **N** (15 pts): New highs - Price proximity to 52-week high + volume confirmation
- **S** (15 pts): Supply/Demand - Volume analysis + price trend
- **L** (15 pts): Leader/Laggard - Multi-timeframe RS (60% 12mo + 40% 3mo)
- **I** (10 pts): Institutional ownership - Scraped from Finviz (FMP was empty)
- **M** (15 pts): Market direction - SPY vs 50/200 day MAs

### Growth Projection Model
Weighted factors: Momentum 20%, Earnings 15%, Analyst 25%, Valuation 15%, CANSLIM 15%, Sector 10%
- Includes PEG-style valuation analysis
- `growth_confidence` field: high/medium/low based on data quality

### Data Sources
- **Financial Modeling Prep (FMP)**: Earnings, ROE, key metrics, analyst targets
- **Yahoo Finance**: Price history, volume data (chart API), fallback for analyst data
- **Finviz**: Institutional ownership (web scraping)

### API Rate Limiting & Performance
- FMP limit: 300 calls/minute
- **NEW**: Batch FMP endpoints (500 tickers per call for quotes/profiles)
- Full 2080 stock scan completes in ~10-12 minutes (was 25-35 min)
- Yahoo Finance handles price history (no strict rate limit)
- Extended cache intervals: earnings/balance_sheet 7 days, institutional 14 days
- DB write batching: commits every 50 stocks instead of per-stock

### Async Implementation (5-10x Performance Boost)

The async scanner uses `aiohttp` for concurrent API requests.

| Stocks | Sync Time | Async Time | Speedup |
|--------|-----------|------------|---------|
| 10     | 30-50s    | 3-5s       | 10x     |
| 50     | 3-5 min   | 20-30s     | 10x     |
| 500    | 30-45 min | 3-5 min    | 10x     |
| 2080   | 2-3 hours | 10-15 min  | 10x     |

**Key Files**:
- `async_data_fetcher.py` - Async version using `aiohttp`
- `async_scanner.py` - Batch processing integration

**Safety Features**:
- Semaphore limit: Max 10 concurrent requests
- Batch processing: Default 50 stocks per batch
- Inter-batch delay: 0.5s pause between batches
- Timeout: 30s per request

**Usage**:
```python
from async_scanner import run_async_scan
results = run_async_scan(tickers, batch_size=50)
```

**Dependencies**: `aiohttp>=3.9.0`

### Stock Universe Coverage (~2000+ tickers)
Fetched dynamically from Wikipedia (with fallbacks):
- **S&P 500**: ~503 large-cap stocks
- **S&P MidCap 400**: ~400 mid-cap stocks (important for CANSLIM growth)
- **S&P SmallCap 600**: ~603 quality small-caps
- **Russell 2000**: ~1200 curated small-caps
- **Portfolio tickers**: Always included in every scan (highest priority)

Portfolio tickers are automatically fetched from the database and scanned first, regardless of which universe is selected. This ensures your holdings always have fresh data.

## Recent Improvements (Jan 2025)

### Base Pattern Detection & Breakout Fixes (Jan 31)

**Problem**: Breakouts page showed extended stocks (GOLF +16.5%, TTC +19.9% above pivot) that had already broken out weeks ago. These are NOT actionable buy points.

**Root Causes**:
1. **Flat base detection was wrong**: Checked if individual weeks had tight intraweek ranges (<15%), but a flat base should measure the TOTAL consolidation range across all weeks
2. **Breakout criteria too loose**: Allowed stocks up to 10% above pivot to be marked as "breaking out"
3. **Fallback query pollution**: API had a fallback that added stocks within 10% of 52-week high, ignoring the `is_breaking_out` flag
4. **NULL filter bug**: `Stock.market_cap > 0` filtered out all stocks when `market_cap` was NULL

**Fixes Implemented**:

1. **Fixed Flat Base Detection** (`canslim_scorer.py`):
   - Now calculates `(highest_high - lowest_low) / lowest_low` across the entire base window
   - Finds the longest window (5-15 weeks) where total range < 15%
   - Returns `base_depth` percentage for transparency
   - Test: NVDA flat 11w (14.7%), MSFT flat 10w (12.9%), GOOG flat 9w (14.7%)

2. **Tightened Breakout Criteria** (`canslim_scorer.py:is_breaking_out()`):
   - Pre-breakout: -3% to 0% from pivot (building for breakout)
   - Active breakout: 0% to +5% from pivot (optimal buy zone)
   - **Extended stocks (>5% above pivot) now return `is_breaking_out = False`**
   - This ensures breakout list shows actionable opportunities only

3. **Removed Fallback Query** (`backend/main.py`):
   - Deleted the "near 52-week high" fallback that ignored `is_breaking_out` flag
   - The scanner's `is_breaking_out` flag is now the sole source of truth

4. **Fixed NULL Filter** (`backend/main.py`):
   - Removed `market_cap > 0` and `week_52_high > 0` filters that failed on NULL values
   - API now returns stocks based on `is_breaking_out` flag only

**Files Modified**:
- `canslim_scorer.py` - `_detect_flat_base()`, `is_breaking_out()`
- `backend/main.py` - `/api/stocks/breaking-out` endpoint

**Volume Ratio Explained**:
- `volume_ratio` = current volume ÷ 50-day average
- 1.5x+ = significant volume surge (green on frontend)
- Breakouts require 50+ effective volume score (either 1.0x single day OR multi-day confirmation)
- Stocks with NULL volume_ratio now default to 1.0 in API responses

**Useful Commands**:
```bash
# Check how many stocks are breaking out
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock; db = SessionLocal(); print(f'Breaking out: {db.query(Stock).filter(Stock.is_breaking_out == True).count()}'); db.close()"

# Test the breakouts API
curl -s "http://localhost:8001/api/stocks/breaking-out?limit=10" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'API returned: {len(d.get(\"stocks\",[]))} stocks'); [print(f'  {s[\"ticker\"]}') for s in d.get('stocks',[])]"

# Verify breakout stock details
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock; db = SessionLocal(); stocks = db.query(Stock).filter(Stock.is_breaking_out == True).limit(5).all(); [print(f'{s.ticker}: base={s.base_type}, score={s.canslim_score:.0f}') for s in stocks]; db.close()"
```

### Market Cap & 52-Week Low Fix (Feb 1)

**Problem**: All stocks had `market_cap=0` and `week_52_low=0` in the database, which affected:
- Backtester `market_cap >= 10B` filter (excluded ALL stocks)
- Stock detail page displaying $0 for these fields
- Screener sorting by market cap

**Root Causes (Multiple Issues Found)**:

1. **FMP `/stable/` batch endpoints broken**: The `/stable/quote?symbol=A,B,C` format returns empty `[]` for comma-separated symbols. Each ticker works individually, but batches fail silently.

2. **Profile low_52w not extracted**: `fetch_fmp_batch_profiles` extracted `high_52w` from the range field ("169.21-288.62") but NOT `low_52w`

3. **Profile market_cap unused**: Profile has `mktCap` but `get_stock_data_async` never used it as fallback

4. **Yahoo chart API incomplete**: Only extracted `fiftyTwoWeekHigh`, not `fiftyTwoWeekLow`

5. **Yahoo info market_cap not returned**: `fetch_yahoo_info_comprehensive_async` fetched `marketCap` but didn't include it in the result dict

**Fixes Implemented**:

1. **Fixed FMP Batch Endpoints** (`async_data_fetcher.py`):
   ```python
   # Changed from /stable/ to /api/v3/ for batch support
   url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={FMP_API_KEY}"
   url = f"https://financialmodelingprep.com/api/v3/profile/{symbols}?apikey={FMP_API_KEY}"
   ```

2. **Extract low_52w from Profile Range** (`async_data_fetcher.py`):
   ```python
   # Range format is "low-high" e.g. "169.21-288.62"
   range_parts = profile.get("range", "").split("-")
   if len(range_parts) >= 2:
       low_52w = float(range_parts[0].strip())
       high_52w = float(range_parts[-1].strip())
   ```

3. **Added Profile Fallbacks** (`async_data_fetcher.py`):
   ```python
   if not stock_data.low_52w:
       stock_data.low_52w = profile.get("low_52w", 0) or 0
   if not stock_data.market_cap:
       stock_data.market_cap = profile.get("market_cap", 0) or 0
   ```

4. **Added Yahoo Chart API Fields** (`data_fetcher.py`):
   - Now extracts `fiftyTwoWeekLow` from chart API meta
   - Note: Chart API has `fiftyTwoWeekLow` but NOT `marketCap`

5. **Added Yahoo Info market_cap** (`async_data_fetcher.py`):
   - Added `market_cap` to `_fetch_yahoo_info()` result dict
   - Added fallback in `get_stock_data_async`

**Data Flow (Fallback Chain)**:
- **market_cap**: FMP /stable/ quotes → FMP /stable/ profiles (mktCap) → Yahoo info
- **week_52_low**: FMP /stable/ quotes (yearLow) → FMP /stable/ profiles (from range) → Yahoo chart API

**Files Modified**:
- `async_data_fetcher.py` - Individual /stable/ fetches, profile range extraction, all fallbacks
- `data_fetcher.py` - Chart API returns `low_52w`

**Verification**:
```bash
# Check portfolio stocks have data
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock, PortfolioPosition; db = SessionLocal(); tickers = [p.ticker for p in db.query(PortfolioPosition.ticker).distinct().all()]; stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all(); [print(f'{s.ticker}: mktcap={s.market_cap}, 52wLow={s.week_52_low}') for s in stocks]; db.close()"

# Check overall data coverage
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock; db = SessionLocal(); with_data = db.query(Stock).filter(Stock.market_cap > 0, Stock.week_52_low > 0).count(); total = db.query(Stock).count(); print(f'Stocks with market_cap AND week_52_low: {with_data}/{total} ({with_data/total*100:.1f}%)'); db.close()"
```

**CRITICAL: FMP API Subscription Tiers** (discovered Feb 2026):
- `/api/v3/` endpoints are **LEGACY ONLY** - require subscription before Aug 31, 2025
- `/api/v3/quote/AAPL,MSFT` returns 403 Forbidden for newer accounts
- `/stable/` endpoints work but **DO NOT support comma-separated batch requests**
- `/stable/quote?symbol=AAPL,MSFT` returns empty `[]` - must fetch individually
- Solution: Fetch quotes/profiles individually via `/stable/` in parallel batches of 50

### Progress Tracking + Delisted Ticker System (Jan 30)

**Problem**: Scan progress showed >100% (e.g., 3466/2083 at 166%). AMD showed 531% ROE instead of ~5%.

**Root Causes**:
1. **Double-counting bug**: Progress calculation was `len(results) + len(completed_tickers)` but `completed_tickers` already included new results
2. **ROE storage bug**: Yahoo returns ROE as decimal (0.05 = 5%), code was multiplying by 100 twice
3. **Checkpoint pollution**: Checkpoint file included tickers no longer in scan list (delisted)

**Fixes Implemented**:

1. **Progress Tracking** (`async_data_fetcher.py`):
   - Changed progress to just `len(completed_tickers)` (no double-count)
   - Filter checkpoint tickers to only include those in current scan list
   - Scheduler callback now syncs `total_stocks` if it differs

2. **Delisted Ticker Tracking** (new system):
   - `DelistedTicker` table in `database.py` - tracks invalid tickers
   - `mark_ticker_as_delisted()` in `data_fetcher.py` - marks failing tickers
   - `get_delisted_tickers()` - returns set of tickers to exclude
   - `get_all_tickers()` in `sp500_tickers.py` - filters out delisted tickers
   - Tickers marked when: Yahoo 404, no price data, stale cache >7 days
   - After 3 failures, ticker excluded for 30 days before recheck

3. **ROE Fix** (`async_data_fetcher.py`):
   - Store ROE as decimal from Yahoo (don't multiply by 100)
   - Frontend handles display multiplication

**Files Modified**:
- `async_data_fetcher.py` - Progress fix, delisted marking, ROE storage
- `data_fetcher.py` - `mark_ticker_as_delisted()`, `get_delisted_tickers()`, `clear_delisted_ticker()`
- `backend/database.py` - Added `DelistedTicker` model
- `backend/scheduler.py` - Progress callback syncs total
- `sp500_tickers.py` - Filter delisted in `get_all_tickers()`

**Useful Commands**:
```bash
# Check excluded ticker count
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app'); from data_fetcher import get_delisted_tickers; print(f'Excluded: {len(get_delisted_tickers())}')"

# Manually mark tickers as delisted
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app'); from data_fetcher import mark_ticker_as_delisted; [mark_ticker_as_delisted(t, reason='404_not_found', source='manual') for t in ['TICKER1', 'TICKER2']]"

# Check ROE distribution
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, StockDataCache; db = SessionLocal(); stocks = db.query(StockDataCache).filter(StockDataCache.roe != None, StockDataCache.roe > 0).all(); total = len(stocks); over17 = len([s for s in stocks if s.roe >= 0.17]); print(f'ROE >= 17%: {over17}/{total} ({over17/total*100:.1f}%)'); db.close()"

# Clear scan checkpoint (for fresh scan)
docker exec canslim-analyzer rm -f /app/scan_progress_*.json
```

**ROE Threshold Analysis**:
- 17% ROE threshold is working correctly - ~35% of stocks qualify
- This is intentional: O'Neil's CANSLIM uses 17% as quality filter
- Distribution: 17%+ (35%), 10%+ (63%), 5%+ (87%)
- Tech/growth stocks often have lower ROE due to R&D reinvestment

### Redis Security Fix + Healthcheck (Jan 28)

**CERT-Bund Security Alert**: Received alert that Redis (port 6379) was publicly accessible from the internet without authentication.

**Problem**: `docker-compose.yml` had `ports: "6379:6379"` which exposed Redis to all network interfaces (`0.0.0.0`), allowing anyone on the internet to connect.

**Solution**: Removed the port mapping entirely. The app connects via Docker's internal network using the hostname `redis`, so no external port is needed.

**Also Fixed**: Docker healthcheck was using `curl` which isn't installed in the container, causing perpetual "unhealthy" status. Changed to use Python `urllib` instead.

**Files Modified**:
- `docker-compose.yml` - Removed Redis port exposure, changed healthcheck to use Python

**Verification Commands**:
```bash
# Confirm Redis not exposed publicly (should return nothing)
ss -tlnp | grep 6379

# Confirm both containers healthy
docker ps | grep canslim

# Test Redis connectivity internally
docker exec canslim-analyzer python3 -c "import redis; r = redis.Redis(host='redis', port=6379); print('Redis ping:', r.ping())"
```

**Note**: If you need local Redis access for development, either:
- Run Redis natively on your machine
- Use `ports: ["127.0.0.1:6379:6379"]` (localhost only, not public)

### Async Scanner Insider/Short Data Fix (Jan 25)

**Problem**: Stocks scanned via the async scanner had empty insider sentiment and short interest data. The `async_scanner.py` was using empty placeholder dicts instead of fetching the data.

**Solution**: Added async versions of the insider/short fetchers and integrated them into the async scanner:

1. **New Functions in `async_data_fetcher.py`**:
   - `fetch_fmp_insider_trading_async()` - Async FMP API call for insider buy/sell data
   - `fetch_short_interest_async()` - Wraps yfinance in executor to avoid blocking

2. **New Batch Function in `async_scanner.py`**:
   - `fetch_insider_short_batch_async()` - Fetches insider/short data for all stocks in parallel
   - Respects freshness intervals (insider: 14 days, short: 3 days)
   - Uses caching system to avoid redundant API calls

3. **Integration**:
   - After fetching stock data, creates aiohttp session for insider/short batch fetch
   - Results merged into analysis output for each ticker

**Files Modified**:
- `async_data_fetcher.py` - Added async insider/short fetchers
- `async_scanner.py` - Added batch fetcher, integrated into analysis loop

**Test Results** (10 stocks):
- Short interest: 10/10 stocks with data
- Insider data: 0/10 (expected - large caps like AAPL/MSFT have minimal insider trading in last 90 days)

### Major Performance Optimization v2.0 (Jan 24)

**Scan Time: 25-35 min → 10-12 min (3x faster)**

**Key Optimizations**:
1. **Batch FMP Endpoints**: Fetch 500 tickers per API call for quotes/profiles
   - Before: 2 API calls per ticker (quote + profile) = 4160 calls
   - After: 4-5 batch calls total for quotes + 4-5 for profiles = ~10 calls

2. **Consolidated Income Statement Calls**: Earnings + revenue from one endpoint
   - Before: 4 calls per ticker (quarterly earnings, annual earnings, quarterly revenue, annual revenue)
   - After: 2 calls per ticker (quarterly all, annual all)

3. **Extended Cache Intervals** (reduces re-scan API calls by 70%+):
   | Data Type | Old | New | Reason |
   |-----------|-----|-----|--------|
   | earnings | 1 day | 7 days | Only changes quarterly |
   | balance_sheet | 1 day | 7 days | Only changes quarterly |
   | key_metrics | 1 day | 7 days | Derived from quarterly data |
   | institutional | 7 days | 14 days | 13F filings are quarterly |
   | short_interest | 1 day | 3 days | Bi-weekly updates |

4. **Skip Yahoo When FMP Complete**: Only call Yahoo Finance if FMP missing data
   - Checks: quarterly_earnings (4+), ROE, sector
   - Reduces Yahoo calls by ~60-70%

5. **Database Write Batching**: Commit every 50 stocks instead of per-stock
   - Reduces SQLite I/O overhead significantly

6. **Exponential Backoff for 429s**: Auto-retry with increasing delays

7. **Progress Persistence**: Checkpoint file saves scan progress
   - Interrupted scans can resume from checkpoint
   - Checkpoint expires after 1 hour

**Files Changed**:
- `async_data_fetcher.py` - Complete rewrite with batch endpoints
- `data_fetcher.py` - Extended DATA_FRESHNESS_INTERVALS
- `backend/scheduler.py` - Batched DB commits (every 50 stocks)

**Testing Results** (100 stocks):
- Fetch time: 31.8s (0.32s per stock)
- Extrapolated 2080 stocks: ~11 minutes

### Bug Fixes + UI Improvements (Jan 21 - Late)

**Fixed Scanner AttributeError (`avg_volume`)**:
- **Problem**: Scanner was failing with `'StockData' object has no attribute 'avg_volume'` for every stock
- **Cause**: `backend/scheduler.py:199` referenced `stock_data.avg_volume` but the `StockData` class defines it as `avg_volume_50d`
- **Fix**: Changed `stock_data.avg_volume` → `stock_data.avg_volume_50d` in scheduler.py
- **Lesson**: When deploying, ensure `git pull` actually updated files; use `git reset --hard origin/main` if needed

**Fixed UTC Timestamp Display**:
- **Problem**: "Last updated" times displayed in UTC instead of local timezone
- **Cause**: Backend returned `.isoformat()` without "Z" suffix, so JavaScript didn't know it was UTC
- **Fix**: Added "Z" suffix to all `isoformat()` calls in `backend/main.py` (lines 381, 765, 876, 936, 1122)
- Frontend already had `timeZone: 'America/Chicago'` conversion, now works correctly

**Market Direction Cards - Show MA Values**:
- Added actual dollar values for 50MA and 200MA under each index (SPY, QQQ, DIA)
- Users can now see price targets for moving averages
- File: `frontend/src/pages/Dashboard.jsx` - `IndexCard` component

**Deployment Note**:
If changes don't apply after `git pull && docker-compose up -d --build`, use:
```bash
cd /opt/canslim_analyzer && git fetch origin && git reset --hard origin/main && docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

### Scanner Speed + DB-Backed Caching + ZETA Fix (Jan 21)

**Scanner Speed Improvements (8.5x faster)**:
- Increased workers from 4 to 8
- Reduced delay from 2.5-4.0s to 0.3-0.6s per stock
- Now scans ~65-88 stocks/min (was ~12/min)
- Full 2080 stock scan: ~25-35 min (was ~3 hours)

**DB-Backed Caching (`StockDataCache` table)**:
- Raw API data now persists across container restarts
- On startup, loads cached data from DB into memory
- Background threads save to DB without blocking scans
- Data types cached: earnings, revenue, balance_sheet, analyst, institutional, key_metrics
- Each has separate `*_updated_at` timestamp for freshness checks

**ZETA C Score Fix (GAAP vs Adjusted EPS)**:
- **Problem**: FMP income-statement returns GAAP EPS (includes stock-based compensation = negative for growth companies)
- **Solution**: Always use Yahoo `earnings_history.epsActual` which returns adjusted EPS (what analysts track)
- **Code change**: `data_fetcher.py` line ~1211: `if True:  # Always run to get adjusted EPS`
- **Requirement**: yfinance >= 0.2.40 (has `earnings_history` attribute)
- **Critical**: `backend/requirements.txt` must have `yfinance>=0.2.40` not `0.2.33`

**Multi-Index Market Direction (SPY + QQQ + DIA)**:
- Weighted market signal: SPY (50%), QQQ (30%), DIA (20%)
- Signal values: +2 (strong bullish), +1 (bullish), 0 (neutral), -1 (bearish)
- M score now uses combined weighted signal instead of just SPY
- Dashboard shows all 3 indexes with individual status
- New endpoint: `/api/market-direction` and `/api/market-direction/refresh`

**Database Changes**:
- Added `StockDataCache` table for persistent raw data caching
- Added multi-index fields to `MarketSnapshot`: `spy_signal`, `qqq_*`, `dia_*`, `weighted_signal`

**Files Modified**:
- `backend/database.py` - Added StockDataCache model, MarketSnapshot multi-index fields
- `backend/scheduler.py` - 8 workers, 0.3-0.6s delay
- `data_fetcher.py` - DB-backed caching functions, Yahoo adjusted EPS fix
- `backend/requirements.txt` - yfinance>=0.2.40

### AI Trading Enhancements (Jan 21)

**Trailing Stop Loss**:
- Tracks `peak_price` and `peak_date` for each position
- Dynamic trailing stop thresholds based on peak gain:
  - 50%+ gain: 15% trailing stop
  - 30-50% gain: 12% trailing stop
  - 20-30% gain: 10% trailing stop
  - 10-20% gain: 8% trailing stop
- Protects gains while letting winners run

**Insider Trading Signals**:
- Fetches last 3 months of insider transactions from FMP API (`/v4/insider-trading`)
- Tracks: `insider_buy_count`, `insider_sell_count`, `insider_net_shares`, `insider_sentiment`
- Sentiment: "bullish" (buys > 1.5x sells), "bearish" (sells > 1.5x buys), "neutral"
- Buy evaluation: +5 bonus for bullish insiders, -3 penalty for bearish
- Refreshed weekly (slow-changing data)

**Short Interest Tracking**:
- Fetches from Yahoo Finance: `short_interest_pct` (% of float), `short_ratio` (days to cover)
- Buy evaluation: -5 penalty for >20% short interest, -2 for >10%
- Refreshed daily

**Trade Reason Indicators**:
- `👔 Insiders buying (X)` - shown when bullish insider activity
- `⚠️ Short X%` - shown when elevated short interest (>15%)
- `TRAILING STOP: Peak $X → $Y (-Z%)` - for trailing stop sell triggers

**API Updates**:
- `/api/stocks/{ticker}` - Now includes insider and short interest data
- `/api/ai-portfolio` - Positions include `trailing_stop` object and stock signals

**Database Changes**:
- `ai_portfolio_positions`: Added `peak_price`, `peak_date`
- `stocks`: Added `insider_buy_count`, `insider_sell_count`, `insider_net_shares`, `insider_sentiment`, `insider_updated_at`, `short_interest_pct`, `short_ratio`, `short_updated_at`

**Files Modified**:
- `backend/database.py` - New columns and migrations
- `backend/ai_trader.py` - Trailing stop logic, insider/short in buy evaluation
- `backend/scheduler.py` - Fetch and save insider/short data
- `data_fetcher.py` - `fetch_fmp_insider_trading()`, `fetch_short_interest()`
- `backend/main.py` - API responses include new fields

### Growth Mode Scoring + Dual Portfolio Management (Jan 20)

**Problem Solved**: Pre-revenue growth stocks (like LCTX) scored 0 on CANSLIM because they have no earnings. Now they get properly evaluated.

**Growth Mode Scoring (100 points)**:
| Component | Points | Criteria |
|-----------|--------|----------|
| R (Revenue) | 20 | Revenue growth rate YoY |
| F (Funding) | 15 | Cash runway, institutional backing |
| N (New Highs) | 15 | Price proximity to 52-week high |
| S (Supply) | 15 | Volume surge patterns |
| L (Leader) | 15 | Relative strength vs market |
| I (Institutional) | 10 | Same as CANSLIM |
| M (Market) | 10 | Same as CANSLIM |

**AI Trader Updates**:
- Queries both CANSLIM AND Growth Mode stocks for buy candidates
- Uses `get_effective_score()` helper - Growth stocks use Growth Mode score, traditional use CANSLIM
- Sell/pyramid decisions use appropriate score for each stock type
- Logs include stock type (Growth vs CANSLIM)

**Database Fields Added**:
- `Stock`: `is_growth_stock`, `growth_mode_score`, `growth_mode_details`
- `AIPortfolioPosition`: `is_growth_stock`, `purchase_growth_score`, `current_growth_score`
- `AIPortfolioTrade`: `growth_mode_score`, `is_growth_stock`

**UI Updates**:
- Purple "Growth" badge on growth stocks in portfolios
- Shows primary score (Growth or CANSLIM) with secondary in parentheses
- Trade history shows "G" badge for growth stock trades

### Dashboard Enhancements (Jan 20)

**New Tables on Home Page (2x2 grid)**:
1. **Top CANSLIM** - Top 10 by CANSLIM score
2. **Top Growth Stocks** - Top 10 by Growth Mode score
3. **Top Under $25** - Budget-friendly picks
4. **Breaking Out** - Stocks breaking out of base patterns with volume

**Duplicate Ticker Filtering**:
- GOOG/GOOGL now filtered in all tables (shows highest scorer only)
- Uses `DUPLICATE_TICKERS` constant and `filter_duplicate_stocks()` helper

**Stock Detail Page Upgrades**:
- `GrowthModeSection`: Shows R+F breakdown, revenue growth %
- `TechnicalAnalysis`: Base pattern, weeks in base, volume ratio, EPS acceleration

### Tiered Data Fetching (Jan 20)

**Reduces API calls by ~70% on subsequent scans**:
```python
DATA_FRESHNESS_INTERVALS = {
    "price": 0,              # Always fetch (real-time)
    "earnings": 24 * 3600,   # Once per day
    "revenue": 24 * 3600,
    "balance_sheet": 24 * 3600,
    "institutional": 7 * 24 * 3600,  # Once per week
}
```

**How it works**:
- First scan of the day: full API calls
- Subsequent scans: skip slow-changing data if fresh
- `fetch_with_cache()` wrapper checks freshness before fetching

### Technical Analysis Integration (Jan 20)

**Base Pattern Detection**:
- Flat base: 5+ weeks of tight price action (<15% range)
- Cup pattern detection (simplified)
- `weeks_in_base` counter

**Breakout Detection**:
- Price within 5% of pivot point
- Volume surge (1.5x+ average)
- `is_breaking_out` flag on Stock model

**New Stock Fields**:
- `volume_ratio`: Current vs 50-day average
- `weeks_in_base`: Duration of consolidation
- `base_type`: 'flat', 'cup', 'none'
- `is_breaking_out`: Boolean flag
- `breakout_volume_ratio`: Volume on breakout day

### Scheduler Fixes (Jan 20-21)
- **Fixed `project_growth()` call**: Removed incorrect `ticker` param, pass full `CANSLIMScore` object
- **Fixed `StockData` attribute names**: `company_name` → `name`, `analyst_target` → `analyst_target_price`, `pe_ratio` → `trailing_pe`
- **Fixed `CANSLIMScore` access**: Build `score_details` dict from individual `*_detail` fields (not `.details`)
- **Fixed progress tracking**: Thread-safe counters with lock, updates `stocks_scanned` in real-time
- Scanner now shows accurate progress during scans instead of stuck at 0

### Code Quality Fixes
- Fixed scheduler bug: `calculate_score()` → `score_stock()` (auto-scan was broken)
- Consolidated duplicate `adjust_score_for_market()` function to module level
- Consolidated `DUPLICATE_TICKERS` constant (GOOG/GOOGL handling)
- Added `expand_tickers_with_duplicates()` and `filter_duplicate_stocks()` helpers
- Replaced all `print()` with proper `logger` in data_fetcher.py
- Added input validation on portfolio position creation

### Performance Optimizations
- Added database indexes on `stocks.current_price` and composite `(canslim_score, current_price)`
- Optimized dashboard stats: combined multiple count queries into single query
- Added bounded LRU cache to DataFetcher (max 1000 entries, prevents memory growth)

### Score Trend Analysis
- 7-day trend analysis: improving/stable/deteriorating signals
- `get_score_trend()` and `get_score_trends_batch()` functions in main.py
- Frontend shows "↗ Up" (green) or "↘ Down" (red) badges on dashboard and portfolio
- Threshold: ±3 points over 7 days triggers trend status

### Portfolio Trending Indicators
- **↑↓ Score change**: Points changed since last scan (shown in green/red)
- **↗↘ 7-day trend**: Weekly improving/deteriorating indicator with tooltip
- **⚠ Low data warning**: Yellow flag for stocks with limited analyst data (growth_confidence != high/medium)
- Legend in Portfolio header explains all indicators

### Backtesting Data Preparation
- Changed StockScore from 1 record/day to 1 record/scan (6x more data)
- Added `timestamp` column for precise timing
- Added `week_52_high` column to track breakout proximity over time
- ~84,000 records after 2 weeks of scanning (enough for backtesting)

### Frontend Enhancements
- Collapsible Manual Scan section (cleaner Home tab)
- Default scan settings: "All Stocks" at 90-minute intervals
- WeeklyTrend component showing 7-day score trajectory
- Data quality warning icon for stocks with limited analyst data

### Portfolio Gameplan
Generates actionable recommendations with position sizing:
- **SELL**: Weak fundamentals + losses (score < 35, loss > 10%)
- **TRIM**: Take profits on big winners (100%+ gains or 50%+ with large position)
- **BUY**: Top stocks not in portfolio (score 75+, 15%+ projected)
- **ADD**: Strong stock on pullback with room to grow
- **WATCH**: Stocks approaching breakout (5-15% from 52-week high)

## Database Schema Notes

### StockScore Table (Historical Data)
Stores one record per stock per scan for granular backtesting:
- `timestamp`: When the scan occurred
- `date`: Date for easy daily grouping
- `total_score`, `c/a/n/s/l/i/m_score`: All CANSLIM components
- `projected_growth`, `current_price`, `week_52_high`

### Key Indexes
- `ix_stocks_canslim` on canslim_score
- `ix_stocks_price` on current_price
- `ix_stocks_score_price` composite for filtered queries
- `ix_stocks_breaking_out` on (is_breaking_out, canslim_score)
- `ix_stocks_growth` on (is_growth_stock, growth_mode_score)
- `ix_stock_scores_stock_timestamp` for backtesting queries

## Backtesting System (Jan 2026)

Historical backtesting to validate the CANSLIM AI trading strategy.

### Features
- **1-Year Historical Simulation**: Uses Yahoo Finance price data
- **Full AI Trading Logic**: Reuses buy/sell signals, trailing stops, pyramiding, sector limits
- **Performance Metrics**: Total return, max drawdown, Sharpe ratio, win rate
- **SPY Benchmark**: Compares strategy vs buy-and-hold SPY
- **Trade History**: Full log of all simulated trades with reasons

### Files
- `backend/historical_data.py` - Historical data provider with point-in-time accuracy
- `backend/backtester.py` - Core backtesting engine with day-by-day simulation
- `backend/database.py` - 4 new models: BacktestRun, BacktestSnapshot, BacktestTrade, BacktestPosition
- `frontend/src/pages/Backtest.jsx` - UI for running and viewing backtests

### API Endpoints
- `POST /api/backtests` - Create and start a new backtest
- `GET /api/backtests` - List all backtests
- `GET /api/backtests/{id}` - Get detailed results with chart data
- `GET /api/backtests/{id}/status` - Poll progress during run
- `POST /api/backtests/{id}/cancel` - Cancel a running backtest
- `DELETE /api/backtests/{id}` - Delete a backtest

### Access
- Navigate to `/backtest` in the frontend
- Or click "Run Historical Backtest" link on the AI Portfolio page

### Configuration
```python
starting_cash: float = 25000.0
stock_universe: str = "sp500"  # or "all"
max_positions: int = 20
min_score_to_buy: int = 65
stop_loss_pct: float = 10.0
```

### Tests
```bash
python3 -m pytest tests/test_backtester.py -v  # 13 tests
```

### Stock Picking Model Improvements (Jan 31, 2026)

**Problem**: Backtesting showed +16.9% vs SPY +19.7% - underperforming the benchmark. The model was correctly identifying stocks doing well but buying them AFTER they broke out (at 52-week highs) rather than BEFORE.

**Key Changes**:

1. **SPY Comparison Display Fix** (`Backtest.jsx`):
   - Returns now show RED when underperforming SPY (was showing green for any positive return)
   - Displays the difference vs SPY (e.g., "vs SPY -2.8%")

2. **N Score Refactored** (`backtester.py`, `ai_trader.py`):
   - Now scores based on proximity to PIVOT POINT (from base pattern), not just 52-week high
   - Stocks with proper base patterns scored on pivot: 0-5% below pivot = 15 pts (best entry)
   - Stocks at 52-week high WITHOUT base pattern = reduced score (10 pts max)

3. **Pre-Breakout Bonus** (+20 points):
   - Stocks 5-15% BELOW pivot with valid base pattern get priority
   - This is the optimal entry zone - before the crowd notices
   - Larger position sizing (15% more) for pre-breakout entries

4. **Extended Stock Penalty** (-10 to -20 points):
   - Stocks more than 5% ABOVE their pivot/breakout point are penalized
   - 5-10% above pivot: -10 points (moderate penalty)
   - >10% above pivot: -20 points (heavy penalty - chasing)

5. **Base Pattern Quality Bonus** (up to 15 points):
   - cup_with_handle: +10 (best pattern)
   - cup: +8
   - double_bottom: +7
   - flat: +6
   - Longer consolidation (8+ weeks): +5 extra

6. **Revised Composite Score Weights**:
   ```
   Old: 30% growth, 30% score, 25% momentum, 15% breakout
   New: 25% growth, 25% score, 20% momentum, 20% breakout/pre-breakout, 10% base quality
   ```

**Files Modified**:
- `frontend/src/pages/Backtest.jsx` - SPY comparison color fix
- `backend/backtester.py` - N score refactor, pre-breakout/extended logic
- `backend/ai_trader.py` - Consistent logic for live trading

**Trade Reasons Now Include**:
- `🚀 BREAKOUT (flat) 1.8x vol` - Confirmed breakout with volume
- `📈 PRE-BREAKOUT (cup) 8% below pivot` - Pre-breakout entry
- `⚠️ Extended 12% above pivot` - Warning for extended stocks
- `Base: flat 6w` - Shows base pattern type and duration

### Predictive Performance Improvement Plan (Feb 2, 2026)

**Problem**: Backtester returned +16.9% vs SPY +19.7% (underperforming by 2.8%). The model identified good stocks but bought them too late and sold too early due to score volatility.

**Root Causes Identified**:
1. Pre-breakout entries undervalued - Breakout got +25 bonus, pre-breakout only +20 (should be flipped)
2. Score crash sells too sensitive - Backtester sold on single score drop without stability check
3. EPS acceleration underweighted - Only +3 points for one of the strongest predictors
4. One-size-fits-all thresholds - 25% growth threshold ignored sector differences

**Phase 1: Critical Fixes (Implemented)**

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Pre-breakout bonus | +20 | **+30** | Better entries before crowd |
| At-pivot bonus | +15 | **+25** | Reward optimal timing |
| Breakout bonus | +25 | **+20** | Reduce chasing |
| Position multiplier (pre-breakout) | 1.15x | **1.30x** | Larger positions at best entries |
| Score crash requirement | 1 scan | **2 consecutive** | Avoid whipsaw sells |

**Files Modified**: `backend/ai_trader.py`, `backend/backtester.py`, `config/default.yaml`

**Phase 2: Scoring Enhancements (Implemented)**

1. **EPS Acceleration Bonus**: Increased from 3 to **5 points** (strong predictor of price performance)

2. **Sector-Adjusted Growth Thresholds**:
   | Sector | Excellent | Good |
   |--------|-----------|------|
   | Technology | 30% | 20% |
   | Healthcare | 25% | 15% |
   | Industrials | 20% | 12% |
   | Utilities | 12% | 8% |
   | Default | 25% | 15% |

3. **Partial Profit Taking**:
   - Sell **25%** at +25% gain (if score >= 60)
   - Sell **50%** at +40% gain (if score >= 60)
   - Let remaining position run

**Files Modified**: `canslim_scorer.py`, `backend/ai_trader.py`, `backend/backtester.py`

**Phase 3: Fine-Tuning (Implemented)**

1. **Momentum Confirmation**: Check `rs_3m >= rs_12m * 0.95` before buying
   - If recent momentum is fading, apply 15% penalty to composite score

2. **Expanded Institutional Range**: Changed sweet spot from 20-60% to **25-75%**
   - Many quality growth stocks have 60-75% institutional ownership

**New Database Columns**:
- `stocks.rs_12m` - 12-month relative strength vs S&P 500
- `stocks.rs_3m` - 3-month relative strength vs S&P 500
- `ai_portfolio_positions.partial_profit_taken` - Track partial sales (0, 0.25, 0.50)

**New Test Cases** (`tests/test_backtester.py`):
- `TestPreBreakoutBonuses` - Verify entry bonus calculations
- `TestScoreStability` - Verify blip detection and consecutive scan logic
- `TestPartialProfitTaking` - Verify partial sell mechanics
- `TestMomentumConfirmation` - Verify RS ratio penalty

### Performance Optimizations (Feb 2, 2026)

**Database Indexes Added**:
- `ix_stocks_breaking_out` on `(is_breaking_out, canslim_score)` - Faster breakout queries
- `ix_stocks_growth` on `(is_growth_stock, growth_mode_score)` - Faster growth stock queries

**N+1 Query Fixes**:
- `/api/portfolio/gameplan` - Batch fetch position stocks (was: N queries, now: 1 query)
- `/api/ai-portfolio` - Batch fetch position stocks (was: N queries, now: 1 query)
- Result: 50-80% faster tab-to-tab navigation

**Files Modified**: `backend/database.py`, `backend/main.py`

## Pending Features

### Ready to Build
- **Watchlist Alerts**: `target_price` and `alert_score` fields exist but aren't monitored
- **Transaction Protection**: Wrap portfolio refresh in atomic transaction

### Lower Priority
- **Portfolio Correlation**: Show hidden concentration risk
- **Earnings Calendar**: Avoid buying before earnings surprises

## Common Issues & Fixes

### Docker "ContainerConfig" Error
```bash
cd /opt/canslim_analyzer && docker-compose down && docker rm -f canslim-analyzer 2>/dev/null; docker-compose up -d --build
```

### Frontend changes not appearing
Force full rebuild (clears Docker cache):
```bash
cd /opt/canslim_analyzer && docker-compose down && docker rmi $(docker images -q canslim_analyzer*) 2>/dev/null; docker-compose build --no-cache && docker-compose up -d
```

### Check container logs
```bash
docker logs -f canslim-analyzer
```

### Debug database queries
```bash
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, Stock; db = SessionLocal(); stock = db.query(Stock).first(); print(f'Ticker: {stock.ticker}, Score: {stock.canslim_score}'); db.close()"
```

### Check StockScore data for backtesting
```bash
docker exec canslim-analyzer python3 -c "import sys; sys.path.insert(0, '/app/backend'); from database import SessionLocal, StockScore; db = SessionLocal(); count = db.query(StockScore).count(); print(f'StockScore records: {count}'); db.close()"
```

## File Structure

### Root Directory
- `canslim_scorer.py` - CANSLIM scoring logic (100 points)
- `data_fetcher.py` - Data fetching from APIs (with bounded cache)
- `async_data_fetcher.py` - Async version using aiohttp
- `async_scanner.py` - Async batch scanning
- `growth_projector.py` - Growth projection model
- `sp500_tickers.py` - Stock universe management
- `config_loader.py` - YAML configuration loader
- `redis_cache.py` - Redis caching layer
- `portfolio_analyzer.py` - Portfolio analysis tools
- `portfolio_manager.py` - Portfolio management

### Backend (`/backend/`)
- `main.py` - FastAPI endpoints + shared helpers
- `database.py` - SQLAlchemy models + migrations
- `scheduler.py` - Continuous scanning logic
- `ai_trader.py` - AI Portfolio trading logic
- `backtester.py` - Historical backtesting engine
- `historical_data.py` - Historical data provider
- `config.py` - Backend configuration

### Frontend (`/frontend/src/`)
- `pages/` - React page components (Dashboard, Backtest, etc.)
- `components/` - Reusable UI components
- `api.js` - API client + utility functions
- `App.jsx` - Main application router

### Configuration (`/config/`)
- `default.yaml` - Base configuration
- `development.yaml` - Development overrides
- `production.yaml` - Production settings

### Tests (`/tests/`)
- `conftest.py` - Shared pytest fixtures
- `test_canslim_scorer.py` - Scoring logic tests (19 tests)
- `test_config_loader.py` - Configuration tests (10 tests)
- `test_redis_cache.py` - Redis cache tests (8 tests)
- `test_backtester.py` - Backtesting tests (21 tests) - includes pre-breakout, score stability, partial profits, momentum

## Unit Testing

### Quick Start
```bash
# Set environment
export CANSLIM_ENV=development

# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Run specific test file
python3 -m pytest tests/test_canslim_scorer.py -v
```

### Test Coverage (as of Jan 2026)
| Module | Coverage |
|--------|----------|
| Config Loader | 95.59% |
| CANSLIM Scorer | 47.37% |
| Redis Cache | 32.19% |

### Running with Redis
```bash
# Start Redis for full test coverage
docker run -d -p 6379:6379 --name test-redis redis:7-alpine

# Run tests
python3 -m pytest tests/ -v

# Cleanup
docker stop test-redis && docker rm test-redis
```

### Test Results
```
50 passed, 5 skipped (Redis tests when unavailable)
Test execution: ~6 seconds
```

## Dependencies

### Root `requirements.txt`
```
pyyaml>=6.0.1
redis>=5.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
aiohttp>=3.9.0
```

### Backend `backend/requirements.txt`
```
fastapi>=0.109.0
uvicorn>=0.27.0
sqlalchemy>=2.0.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
beautifulsoup4>=4.12.0
httpx>=0.26.0
pyyaml>=6.0.1
redis>=5.0.0
```

### Frontend `frontend/package.json`
- React 18.x
- Vite (build tool)
- TailwindCSS (styling)
- Recharts (charts)

## Documentation Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | This file - comprehensive project context |
| `CURRENT_STATUS.md` | Latest changes and active issues |
| `ASYNC_IMPLEMENTATION.md` | Async scanner guide and performance metrics |
| `IMPLEMENTATION_SUMMARY.md` | Config system, Redis cache, unit tests summary |
| `README_TESTING.md` | Testing guide and CI/CD setup |

## Owner's Trading Preferences
- Likes stocks under $25 that fit CANSLIM criteria
- Uses the "Top Under $25" section for finding opportunities
- Default scan: All Stocks at 90-minute intervals
