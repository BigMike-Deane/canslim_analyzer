"""
Disk-based Cache for Historical Price Data

SQLite-based cache that persists historical price data across restarts.
Dramatically speeds up repeated backtests by avoiding redundant Yahoo API calls.

Key features:
- Cache key: ticker + start_date + end_date
- Auto-expiry: 30 days (configurable)
- Location: data/price_cache.db
- Thread-safe operations
"""

import os
import sqlite3
import logging
import json
import pandas as pd
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from pathlib import Path

# Add parent directory for config import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class PriceHistoryCache:
    """
    SQLite-based cache for historical price data.

    Stores price DataFrames as JSON for fast retrieval during backtesting.
    """

    def __init__(self, db_path: str = None, expiry_days: int = 30):
        """
        Initialize the price cache.

        Args:
            db_path: Path to SQLite database. Defaults to data/price_cache.db
            expiry_days: Days before cached data expires. Default 30.
        """
        if db_path is None:
            # Default to data/price_cache.db relative to project root
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "price_cache.db")

        self.db_path = db_path
        self.expiry_days = expiry_days

        # Initialize database
        self._init_db()

        # Stats
        self._hits = 0
        self._misses = 0

        logger.info(f"PriceHistoryCache initialized: {db_path}, expiry={expiry_days} days")

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    cache_key TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0
                )
            """)

            # Create index for cleanup queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_created
                ON price_cache(created_at)
            """)

            # Create index for ticker lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_cache_ticker
                ON price_cache(ticker)
            """)

            conn.commit()

    def _make_cache_key(self, ticker: str, start_date: date, end_date: date) -> str:
        """Generate a unique cache key for a ticker and date range."""
        return f"{ticker}_{start_date.isoformat()}_{end_date.isoformat()}"

    def get(self, ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Get cached price history for a ticker and date range.

        Args:
            ticker: Stock ticker
            start_date: Start date for price history
            end_date: End date for price history

        Returns:
            DataFrame with price history or None if not cached/expired
        """
        cache_key = self._make_cache_key(ticker, start_date, end_date)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data_json, created_at FROM price_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()

                if not row:
                    self._misses += 1
                    return None

                data_json, created_at_str = row

                # Check expiry
                created_at = datetime.fromisoformat(created_at_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)

                age_days = (datetime.now(timezone.utc) - created_at).days
                if age_days > self.expiry_days:
                    # Expired - delete and return None
                    conn.execute("DELETE FROM price_cache WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    self._misses += 1
                    logger.debug(f"Cache expired for {ticker} ({age_days} days old)")
                    return None

                # Parse JSON back to DataFrame
                data = json.loads(data_json)
                df = pd.DataFrame(data)

                # Convert date column back to date objects
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date

                self._hits += 1
                logger.debug(f"Cache HIT for {ticker} ({len(df)} rows, {age_days}d old)")
                return df

        except Exception as e:
            logger.warning(f"Cache read error for {ticker}: {e}")
            self._misses += 1
            return None

    def set(self, ticker: str, start_date: date, end_date: date, df: pd.DataFrame):
        """
        Store price history in cache.

        Args:
            ticker: Stock ticker
            start_date: Start date for price history
            end_date: End date for price history
            df: DataFrame with price history (columns: date, open, high, low, close, volume)
        """
        if df is None or df.empty:
            return

        cache_key = self._make_cache_key(ticker, start_date, end_date)

        try:
            # Convert DataFrame to JSON-serializable format
            df_copy = df.copy()

            # Convert date objects to strings for JSON
            if 'date' in df_copy.columns:
                df_copy['date'] = df_copy['date'].astype(str)

            data_json = df_copy.to_json(orient='records')

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO price_cache
                    (cache_key, ticker, start_date, end_date, data_json, created_at, row_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    ticker,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    data_json,
                    datetime.now(timezone.utc).isoformat(),
                    len(df)
                ))
                conn.commit()

            logger.debug(f"Cached {ticker}: {len(df)} rows ({start_date} to {end_date})")

        except Exception as e:
            logger.warning(f"Cache write error for {ticker}: {e}")

    def cleanup_expired(self) -> int:
        """
        Delete expired cache entries.

        Returns:
            Number of entries deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.expiry_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM price_cache WHERE created_at < ?",
                    (cutoff.isoformat(),)
                )
                deleted = cursor.rowcount
                conn.commit()

            if deleted > 0:
                logger.info(f"Cache cleanup: deleted {deleted} expired entries")

            return deleted

        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
            return 0

    def clear_ticker(self, ticker: str) -> int:
        """
        Clear all cached data for a specific ticker.

        Args:
            ticker: Stock ticker to clear

        Returns:
            Number of entries deleted
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM price_cache WHERE ticker = ?",
                    (ticker,)
                )
                deleted = cursor.rowcount
                conn.commit()

            logger.info(f"Cleared cache for {ticker}: {deleted} entries")
            return deleted

        except Exception as e:
            logger.warning(f"Cache clear error for {ticker}: {e}")
            return 0

    def clear_all(self):
        """Clear all cached data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM price_cache")
                conn.commit()

            logger.info("Cache cleared completely")

        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(row_count) FROM price_cache")
                entries, total_rows = cursor.fetchone()

                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT ticker) FROM price_cache"
                )
                unique_tickers = cursor.fetchone()[0]

                # Calculate database file size
                db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)

            hit_rate = (self._hits / (self._hits + self._misses) * 100
                       if (self._hits + self._misses) > 0 else 0)

            return {
                "entries": entries or 0,
                "total_rows": total_rows or 0,
                "unique_tickers": unique_tickers or 0,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 1),
                "db_size_mb": round(db_size_mb, 2),
                "expiry_days": self.expiry_days
            }

        except Exception as e:
            logger.warning(f"Stats error: {e}")
            return {
                "entries": 0,
                "error": str(e)
            }


# Global instance for convenience
_cache_instance = None


def get_price_cache(db_path: str = None, expiry_days: int = None) -> PriceHistoryCache:
    """
    Get or create the global price cache instance.

    Args:
        db_path: Optional custom database path
        expiry_days: Optional custom expiry (default from config)

    Returns:
        PriceHistoryCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        try:
            from config_loader import config
            if expiry_days is None:
                expiry_days = config.get('backtester.disk_cache.expiry_days', default=30)
            if db_path is None:
                db_path = config.get('backtester.disk_cache.database', default=None)
        except ImportError:
            if expiry_days is None:
                expiry_days = 30

        _cache_instance = PriceHistoryCache(db_path=db_path, expiry_days=expiry_days)

    return _cache_instance
