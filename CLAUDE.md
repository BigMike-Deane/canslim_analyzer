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

### Data Sources
- **Financial Modeling Prep (FMP)**: Earnings, ROE, key metrics, analyst targets
- **Yahoo Finance**: Price history, volume data (chart API)
- **Finviz**: Institutional ownership (web scraping)

### API Rate Limiting
- FMP limit: 300 calls/minute
- Current config: 6 workers with 1.5-2.5s delay (~90 stocks/min)
- Avoids 429 errors while maintaining reasonable scan speed

## Recent Improvements (Jan 2025)

### Frontend
- Score 80+ threshold for "Top Stocks" display
- Scan progress timer with elapsed time and ETA
- Scan source dropdown (S&P 500, Top 50, Russell 2000, All)
- "Top Under $25" section on Dashboard
- CANSLIM letter colors normalized (scores are out of 15/10, not 100)
- M Score in Market Direction box fixed (was showing red for 15/15)
- Documentation page with full methodology explanation
- Portfolio Gameplan feature with detailed action cards

### Backend
- Parallel scanning with ThreadPoolExecutor (6 workers)
- 52-week high/low fallback calculation from price history
- week_52_high added to /api/stocks endpoint
- Portfolio refresh auto-scans positions without stock data
- Gameplan endpoint generates BUY/SELL/TRIM/ADD/WATCH recommendations

### Portfolio Gameplan
Generates actionable recommendations with position sizing:
- **SELL**: Weak fundamentals + losses
- **TRIM**: Take profits on big winners (100%+ gains)
- **BUY**: Top stocks not in portfolio (score 75+, 15%+ projected)
- **ADD**: Add to strong positions on dips
- **WATCH**: Stocks approaching breakout (5-15% from 52-week high)

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

## File Structure
- `/backend/main.py` - FastAPI endpoints
- `/backend/database.py` - SQLAlchemy models
- `/frontend/src/pages/` - React page components
- `/frontend/src/api.js` - API client + utility functions
- `/canslim_scorer.py` - CANSLIM scoring logic
- `/data_fetcher.py` - Data fetching from APIs
- `/growth_projector.py` - Growth projection model
- `/portfolio_analyzer.py` - BUY/HOLD/SELL recommendation logic

## Owner's Trading Preferences
- Likes stocks under $25 that fit CANSLIM criteria
- Uses the "Top Under $25" section for finding opportunities
