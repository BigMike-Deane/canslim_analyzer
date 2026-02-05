#!/usr/bin/env python3
"""
One-time script to fix missing analyst targets for major stocks.
Run after a scan if big stocks are missing targets.
"""
import time
import yfinance as yf
from backend.database import StockDataCache, SessionLocal

# Major stocks that should definitely have analyst targets
PRIORITY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK.B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "LLY", "MCD", "TMO", "CSCO",
    "ACN", "ABT", "DHR", "ADBE", "NKE", "TXN", "NEE", "PM", "UNH"
]

def main():
    db = SessionLocal()
    updated = 0

    for ticker in PRIORITY_TICKERS:
        record = db.query(StockDataCache).filter(StockDataCache.ticker == ticker).first()

        # Skip if already has target
        if record and record.analyst_target_price:
            print(f"{ticker}: already has target ${record.analyst_target_price:.2f}")
            continue

        # Fetch from Yahoo
        time.sleep(1.5)  # Gentle rate limiting
        try:
            info = yf.Ticker(ticker).info
            target = info.get("targetMeanPrice")

            if target and record:
                record.analyst_target_price = target
                record.analyst_count = info.get("numberOfAnalystOpinions")
                updated += 1
                print(f"{ticker}: updated to ${target:.2f}")
            elif not record:
                print(f"{ticker}: not in database")
            else:
                print(f"{ticker}: no target from Yahoo")

        except Exception as e:
            print(f"{ticker}: error - {e}")

    db.commit()
    db.close()
    print(f"\nDone. Updated {updated} stocks.")

if __name__ == "__main__":
    main()
