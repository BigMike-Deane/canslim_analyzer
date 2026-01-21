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
- Current config: 4 workers with 2.5-4.0s delay (~60-90 stocks/min)
- Avoids 429 errors while maintaining reasonable scan speed

## Recent Improvements (Jan 2025)

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
