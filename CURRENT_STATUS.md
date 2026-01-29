# Current Status - Jan 29, 2026

## Latest Improvements

### AI Trading Logic Improvements - Score Crash Protection

**Problem**: AI Portfolio sold $B and $BKR due to "SCORE CRASH" but the frontend showed stable scores. Likely caused by a temporary data blip during scanning.

**Solution**: Added multiple safeguards:

1. **Score Stability Check** (`check_score_stability()`):
   - Before selling on score crash, checks last 3 scans' scores
   - If current score is much lower than recent average, flags as potential blip
   - **SKIPS THE SELL** if detected as a blip, waits for next scan to confirm

2. **Detailed Trade Logging**:
   - Trade reasons now include component breakdown: `[C:15/A:12/N:8/S:10/L:14/I:7/M:15]`
   - Shows recent average score for context: `(avg: 72)`
   - Flags low data confidence: `⚠️ Low data confidence`

3. **Enhanced Execute Trade Logging**:
   - Sells now log: cost basis, gain %, P/L amount, and full reason

**Files Changed**:
- `backend/ai_trader.py` - Added `check_score_stability()`, improved logging

### Scanner Data Blip Prevention

**Problem**: ARM showed a CANSLIM score of 20 when it should be ~75. API failures during scans caused missing earnings data, leading to artificially low scores being saved.

**Solution**: Added data blip detection in `save_stock_to_db()`:

1. **Detects blips when**:
   - Score drops > 25 points AND
   - 3+ component scores are 0 (missing data) OR
   - No earnings data returned OR
   - Multiple "Insufficient data" / "No data" in score details

2. **When blip detected**:
   - **KEEPS the old score** instead of saving the bad one
   - Preserves non-zero component scores
   - Logs warning: `DATA BLIP DETECTED for {ticker}...KEEPING OLD SCORE`

**Files Changed**:
- `backend/scheduler.py` - Added blip detection in `save_stock_to_db()`

**Test Results**:
```
Test 1: Clear data blip → Detected ✓
Test 2: Legitimate drop → Not flagged ✓
Test 3: Small change → Not flagged ✓
Test 4: ARM-like scenario → Detected ✓
```

**To investigate a stock's score on VPS**:
```bash
docker exec canslim-analyzer python3 -c "
import sys
sys.path.insert(0, '/app/backend')
from database import SessionLocal, AIPortfolioTrade, Stock, StockScore
from sqlalchemy import desc
db = SessionLocal()
# Recent trades
for t in db.query(AIPortfolioTrade).order_by(desc(AIPortfolioTrade.executed_at)).limit(10).all():
    print(f'{t.executed_at}: {t.action} {t.ticker} @ \${t.price:.2f} - {t.reason}')
print()
# Check B and BKR current scores
for ticker in ['B', 'BKR']:
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if stock:
        print(f'{ticker}: Score={stock.canslim_score}, C={stock.c_score}, A={stock.a_score}, N={stock.n_score}, S={stock.s_score}, L={stock.l_score}, I={stock.i_score}, M={stock.m_score}')
db.close()
"
```

---

### Backtest Cancellation Feature

Added the ability to cancel running/stuck backtests from the UI.

### Changes Made
1. **Backend (`database.py`)**: Added `cancel_requested` column to `BacktestRun` model
2. **Backend (`backtester.py`)**: Engine checks for cancellation every 10 simulated days and exits gracefully
3. **Backend (`main.py`)**: New `POST /api/backtests/{id}/cancel` endpoint
4. **Frontend (`api.js`)**: Added `cancelBacktest(id)` API function
5. **Frontend (`Backtest.jsx`)**: Added Cancel button for running/pending backtests, "Cancelled" status badge

### Deployment
```bash
ssh root@147.93.72.73 "cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build"
```

### Fixing Stuck Backtest (Manual)
If the backtest is stuck at 47% and won't cancel via UI, run this on the VPS:
```bash
docker exec canslim-analyzer python3 -c "
import sys
sys.path.insert(0, '/app/backend')
from database import SessionLocal, BacktestRun
db = SessionLocal()
# Mark all running backtests as failed
for bt in db.query(BacktestRun).filter(BacktestRun.status.in_(['running', 'pending'])).all():
    bt.status = 'failed'
    bt.error_message = 'Manually cancelled - stuck at 47%'
db.commit()
print('Done - stuck backtests marked as failed')
db.close()
"
```

---

## Previous Update - Jan 25, 2026

**Async Scanner Insider/Short Data Fix**

The async scanner now properly fetches insider trading and short interest data for all scanned stocks.

### What Was Fixed
- Previously, `async_scanner.py` used empty placeholder dicts `{}` for insider/short data
- Now fetches data asynchronously using new functions in `async_data_fetcher.py`
- Uses caching with proper freshness intervals (insider: 14 days, short: 3 days)

### Files Changed
1. `async_data_fetcher.py`:
   - Added `fetch_fmp_insider_trading_async()` - async FMP API call
   - Added `fetch_short_interest_async()` - wraps yfinance in executor

2. `async_scanner.py`:
   - Added `fetch_insider_short_batch_async()` - batch fetches for all tickers
   - Integrated into `analyze_stocks_async()` workflow

### Test Results (10 stocks)
```
Short interest: 10/10 stocks with data
Insider data: 0/10 (expected - large caps have minimal insider trading)
Scan time: ~9 seconds for 10 stocks
```

## Deployment Notes

To deploy this fix to the VPS:
```bash
ssh root@147.93.72.73 "cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build"
```

If changes don't apply after pull:
```bash
ssh root@147.93.72.73 "cd /opt/canslim_analyzer && git fetch origin && git reset --hard origin/main && docker-compose down && docker-compose build --no-cache && docker-compose up -d"
```

## Previous Issue (Resolved)

The yfinance blocking issue mentioned in the previous status has been addressed:
- `fetch_yahoo_supplement_async()` already used `run_in_executor`
- New `fetch_short_interest_async()` also uses `run_in_executor`
- Async performance is maintained

## Next Steps (Optional)

- Monitor next scan to verify insider/short data populates in the database
- Consider adding insider/short data to the frontend stock detail view
- Watchlist alerts feature still pending (target_price/alert_score fields exist but aren't monitored)
