# CANSLIM Analyzer - Project Context

## Overview
A mobile-first web application for CANSLIM stock analysis with React frontend and FastAPI backend, deployed via Docker on a VPS.

## Architecture
- **Frontend**: React + Vite + TailwindCSS (mobile-first design)
- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Deployment**: Docker on VPS at `/opt/canslim_analyzer`
- **Docker command**: Use `docker-compose` (with hyphen, old version)
- **Container name**: `canslim-analyzer`
- **Port**: 8001
- **VPS IP**: 147.93.72.73

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

### API Rate Limiting
- FMP limit: 300 calls/minute
- Current config: 6 workers with 0.5-1.0s delay (~50-70 stocks/min)
- Full 2080 stock scan completes in ~35-45 minutes
- Yahoo Finance handles most data (no strict rate limit)
- DB cache at 100%+ means most FMP calls are skipped on subsequent scans

### Stock Universe Coverage (~2000+ tickers)
Fetched dynamically from Wikipedia (with fallbacks):
- **S&P 500**: ~503 large-cap stocks
- **S&P MidCap 400**: ~400 mid-cap stocks (important for CANSLIM growth)
- **S&P SmallCap 600**: ~603 quality small-caps
- **Russell 2000**: ~1200 curated small-caps
- **Portfolio tickers**: Always included in every scan (highest priority)

Portfolio tickers are automatically fetched from the database and scanned first, regardless of which universe is selected. This ensures your holdings always have fresh data.

## Recent Improvements (Jan 2025)

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
- `ix_stock_scores_stock_timestamp` for backtesting queries

## Pending Features

### Ready to Build
- **Watchlist Alerts**: `target_price` and `alert_score` fields exist but aren't monitored
- **Transaction Protection**: Wrap portfolio refresh in atomic transaction

### Waiting on Data
- **Backtesting UI**: Need 1-2 weeks of scan data (collecting now)

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
- `/backend/main.py` - FastAPI endpoints + shared helpers
- `/backend/database.py` - SQLAlchemy models + migrations
- `/backend/scheduler.py` - Continuous scanning logic
- `/backend/ai_trader.py` - AI Portfolio trading logic
- `/frontend/src/pages/` - React page components
- `/frontend/src/api.js` - API client + utility functions
- `/canslim_scorer.py` - CANSLIM scoring logic
- `/data_fetcher.py` - Data fetching from APIs (with bounded cache)
- `/growth_projector.py` - Growth projection model

## Owner's Trading Preferences
- Likes stocks under $25 that fit CANSLIM criteria
- Uses the "Top Under $25" section for finding opportunities
- Default scan: All Stocks at 90-minute intervals
