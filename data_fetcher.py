"""
Data Fetcher Module
Wrapper around yfinance with caching and error handling
Now with Financial Modeling Prep (FMP) API for earnings data
Includes Redis cache layer for improved performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from collections import OrderedDict
import time
import requests
import os
import threading
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Import Redis cache (lazy loaded to allow fallback)
try:
    from redis_cache import redis_cache
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis cache module not available")

# FMP API Configuration - using new /stable/ endpoints
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Rate limit tracking
_rate_limit_stats = {
    "errors_429": 0,
    "total_requests": 0,
    "last_reset": datetime.now()
}
_stats_lock = threading.Lock()

# Tiered data freshness cache
# Tracks when each data type was last fetched for each ticker
# Format: {ticker: {data_type: last_fetch_timestamp}}
_data_freshness_cache = {}
_freshness_lock = threading.Lock()

# Freshness intervals (in seconds)
# OPTIMIZED: Extended intervals for slow-changing data to reduce API calls
DATA_FRESHNESS_INTERVALS = {
    "price": 0,                       # Always fetch (real-time)
    "earnings": 7 * 24 * 3600,        # Once per week (only changes quarterly)
    "adjusted_eps": 7 * 24 * 3600,    # Once per week (Yahoo adjusted EPS - matches Fidelity/analysts)
    "revenue": 7 * 24 * 3600,         # Once per week (only changes quarterly)
    "balance_sheet": 7 * 24 * 3600,   # Once per week (only changes quarterly)
    "key_metrics": 7 * 24 * 3600,     # Once per week (derived from quarterly data)
    "analyst": 24 * 3600,             # Once per day (can change with upgrades/downgrades)
    "earnings_surprise": 7 * 24 * 3600,  # Once per week (only changes quarterly)
    "weekly_history": 24 * 3600,      # Once per day (for base detection)
    "institutional": 14 * 24 * 3600,  # Once per 2 weeks (13F filings are quarterly)
    "insider_trading": 14 * 24 * 3600,  # Once per 2 weeks (changes slowly)
    "short_interest": 7 * 24 * 3600,    # Once per week (bi-weekly FINRA updates)
    "yahoo_info": 7 * 24 * 3600,      # Once per week (comprehensive Yahoo data for key metrics, balance, analyst)
    # P1 Features (Feb 2026)
    "earnings_calendar": 7 * 24 * 3600,  # Once per week (earnings dates don't change frequently)
    "analyst_estimates": 7 * 24 * 3600,  # Once per week (revisions happen weekly, not daily)
}

# Cached data storage (stores the actual fetched data)
_cached_data = {}
_cached_data_lock = threading.Lock()
MAX_CACHED_TICKERS = 500  # Limit cache size

# DB-backed cache flag
_db_cache_loaded = False
_db_cache_lock = threading.Lock()


def _get_db_session():
    """Get a database session for cache operations"""
    try:
        from backend.database import SessionLocal
        return SessionLocal()
    except Exception:
        return None


def load_cache_from_db():
    """Load cached data from database into memory on startup"""
    global _db_cache_loaded

    with _db_cache_lock:
        if _db_cache_loaded:
            return

        db = _get_db_session()
        if not db:
            logger.warning("Could not connect to DB for cache loading")
            _db_cache_loaded = True
            return

        try:
            from backend.database import StockDataCache
            cache_records = db.query(StockDataCache).all()
            loaded_count = 0
            error_count = 0

            for record in cache_records:
                ticker = record.ticker
                try:
                    # Load earnings data
                    if record.quarterly_earnings and record.earnings_updated_at:
                        set_cached_data(ticker, "earnings", {
                            "quarterly": record.quarterly_earnings,
                            "annual": record.annual_earnings
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["earnings"] = record.earnings_updated_at

                    # Load revenue data
                    if record.quarterly_revenue and record.revenue_updated_at:
                        set_cached_data(ticker, "revenue", {
                            "quarterly": record.quarterly_revenue,
                            "annual": record.annual_revenue
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["revenue"] = record.revenue_updated_at

                    # Load balance sheet data
                    if record.balance_updated_at:
                        set_cached_data(ticker, "balance_sheet", {
                            "total_cash": record.total_cash,
                            "total_debt": record.total_debt,
                            "shares_outstanding": record.shares_outstanding
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["balance_sheet"] = record.balance_updated_at

                    # Load analyst data
                    if record.analyst_updated_at:
                        set_cached_data(ticker, "analyst", {
                            "target_price": record.analyst_target_price,
                            "count": record.analyst_count
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["analyst"] = record.analyst_updated_at

                    # Load institutional data
                    if record.institutional_updated_at:
                        set_cached_data(ticker, "institutional", record.institutional_holders_pct, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["institutional"] = record.institutional_updated_at

                    # Load key metrics
                    if record.metrics_updated_at:
                        set_cached_data(ticker, "key_metrics", {
                            "roe": record.roe,
                            "trailing_pe": record.trailing_pe,
                            "forward_pe": record.forward_pe,
                            "peg_ratio": record.peg_ratio
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["key_metrics"] = record.metrics_updated_at

                    # Load earnings calendar (P1 feature)
                    if hasattr(record, 'earnings_calendar_updated_at') and record.earnings_calendar_updated_at:
                        set_cached_data(ticker, "earnings_calendar", {
                            "next_earnings_date": record.next_earnings_date.isoformat() if record.next_earnings_date else None,
                            "days_to_earnings": record.days_to_earnings,
                            "earnings_beat_streak": record.earnings_beat_streak
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["earnings_calendar"] = record.earnings_calendar_updated_at

                    # Load analyst estimates (P1 feature)
                    if hasattr(record, 'analyst_estimates_updated_at') and record.analyst_estimates_updated_at:
                        set_cached_data(ticker, "analyst_estimates", {
                            "eps_estimate_current": record.eps_estimate_current,
                            "eps_estimate_prior": record.eps_estimate_prior,
                            "eps_estimate_revision_pct": record.eps_estimate_revision_pct
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["analyst_estimates"] = record.analyst_estimates_updated_at

                    # Load short interest (P1 feature)
                    if hasattr(record, 'short_updated_at') and record.short_updated_at:
                        set_cached_data(ticker, "short_interest", {
                            "short_interest_pct": record.short_interest_pct,
                            "short_ratio": record.short_ratio
                        }, persist_to_db=False)
                        with _freshness_lock:
                            if ticker not in _data_freshness_cache:
                                _data_freshness_cache[ticker] = {}
                            _data_freshness_cache[ticker]["short_interest"] = record.short_updated_at

                    loaded_count += 1
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Failed to load cache for {ticker}: {e}")
                    continue  # Continue with next record

            logger.info(f"Loaded {loaded_count} tickers from DB cache ({error_count} errors)")
            _db_cache_loaded = True

        except Exception as e:
            logger.warning(f"Failed to load DB cache: {e}")
            _db_cache_loaded = True
        finally:
            db.close()


def save_ticker_to_db_cache(ticker: str, data_type: str, data):
    """Save fetched data to database cache for persistence"""
    db = _get_db_session()
    if not db:
        return

    try:
        from backend.database import StockDataCache

        # Get or create cache record
        record = db.query(StockDataCache).filter(StockDataCache.ticker == ticker).first()
        if not record:
            record = StockDataCache(ticker=ticker)
            db.add(record)

        now = datetime.now()

        # Update the appropriate fields based on data_type
        if data_type == "earnings" and isinstance(data, dict):
            record.quarterly_earnings = data.get("quarterly")
            record.annual_earnings = data.get("annual")
            record.earnings_updated_at = now

        elif data_type == "revenue" and isinstance(data, dict):
            record.quarterly_revenue = data.get("quarterly")
            record.annual_revenue = data.get("annual")
            record.revenue_updated_at = now

        elif data_type == "balance_sheet" and isinstance(data, dict):
            record.total_cash = data.get("total_cash")
            record.total_debt = data.get("total_debt")
            record.shares_outstanding = data.get("shares_outstanding")
            record.balance_updated_at = now

        elif data_type == "analyst" and isinstance(data, dict):
            # Only update if we have actual data (don't overwrite with None)
            if data.get("target_price"):
                record.analyst_target_price = data.get("target_price")
            if data.get("count"):
                record.analyst_count = data.get("count")
            record.analyst_updated_at = now

        elif data_type == "institutional":
            # Only update if we have actual data
            if data and isinstance(data, (int, float)):
                record.institutional_holders_pct = data
                record.institutional_updated_at = now

        elif data_type == "key_metrics" and isinstance(data, dict):
            # Only update fields that have actual values
            if data.get("roe"):
                record.roe = data.get("roe")
            if data.get("trailing_pe"):
                record.trailing_pe = data.get("trailing_pe")
            if data.get("forward_pe"):
                record.forward_pe = data.get("forward_pe")
            if data.get("peg_ratio"):
                record.peg_ratio = data.get("peg_ratio")
            record.metrics_updated_at = now

        elif data_type == "yahoo_info" and isinstance(data, dict):
            # Comprehensive Yahoo data - update all relevant fields
            if data.get("roe"):
                record.roe = data.get("roe")
                record.metrics_updated_at = now
            if data.get("institutional_holders_pct"):
                record.institutional_holders_pct = data.get("institutional_holders_pct")
                record.institutional_updated_at = now
            if data.get("analyst_target_price"):
                record.analyst_target_price = data.get("analyst_target_price")
                record.analyst_count = data.get("num_analyst_opinions")
                record.analyst_updated_at = now
            if data.get("cash_and_equivalents") or data.get("total_debt"):
                record.total_cash = data.get("cash_and_equivalents")
                record.total_debt = data.get("total_debt")
                record.balance_updated_at = now

        # P1 Data Types
        elif data_type == "earnings_calendar" and isinstance(data, dict):
            # Convert string date to date object if needed
            next_date = data.get("next_earnings_date")
            if next_date and isinstance(next_date, str):
                try:
                    record.next_earnings_date = datetime.strptime(next_date, '%Y-%m-%d').date()
                except:
                    record.next_earnings_date = None
            else:
                record.next_earnings_date = next_date
            record.days_to_earnings = data.get("days_to_earnings")
            record.earnings_beat_streak = data.get("earnings_beat_streak")
            record.earnings_calendar_updated_at = now

        elif data_type == "analyst_estimates" and isinstance(data, dict):
            record.eps_estimate_current = data.get("eps_estimate_current")
            record.eps_estimate_prior = data.get("eps_estimate_prior")
            record.eps_estimate_revision_pct = data.get("eps_estimate_revision_pct")
            record.analyst_estimates_updated_at = now

        elif data_type == "short_interest" and isinstance(data, dict):
            record.short_interest_pct = data.get("short_interest_pct")
            record.short_ratio = data.get("short_ratio")
            record.short_updated_at = now

        record.updated_at = now
        db.commit()

    except Exception as e:
        logger.debug(f"Failed to save {ticker} {data_type} to DB cache: {e}")
        db.rollback()
    finally:
        db.close()


def compute_data_hash(ticker: str) -> str:
    """Compute a hash of critical data for delta detection"""
    import hashlib
    import json

    with _cached_data_lock:
        earnings = _cached_data.get(f"{ticker}:earnings", {})
        revenue = _cached_data.get(f"{ticker}:revenue", {})
        metrics = _cached_data.get(f"{ticker}:key_metrics", {})

    # Hash the critical scoring data
    data_str = json.dumps({
        "quarterly_earnings": earnings.get("quarterly", [])[:4] if isinstance(earnings, dict) else [],
        "quarterly_revenue": revenue.get("quarterly", [])[:4] if isinstance(revenue, dict) else [],
        "roe": metrics.get("roe") if isinstance(metrics, dict) else None
    }, sort_keys=True, default=str)

    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def is_data_fresh(ticker: str, data_type: str) -> bool:
    """Check if cached data is still fresh"""
    with _freshness_lock:
        if ticker not in _data_freshness_cache:
            return False
        if data_type not in _data_freshness_cache[ticker]:
            return False

        last_fetch = _data_freshness_cache[ticker][data_type]
        interval = DATA_FRESHNESS_INTERVALS.get(data_type, 0)

        return (datetime.now() - last_fetch).total_seconds() < interval


def mark_data_fetched(ticker: str, data_type: str):
    """Mark that data was just fetched"""
    with _freshness_lock:
        if ticker not in _data_freshness_cache:
            _data_freshness_cache[ticker] = {}
        _data_freshness_cache[ticker][data_type] = datetime.now()


def get_cached_data(ticker: str, data_type: str):
    """Get cached data if it exists"""
    with _cached_data_lock:
        key = f"{ticker}:{data_type}"
        return _cached_data.get(key)


def set_cached_data(ticker: str, data_type: str, data, persist_to_db: bool = True):
    """Store data in cache (memory + optionally DB)"""
    with _cached_data_lock:
        # Enforce cache size limit
        if len(_cached_data) >= MAX_CACHED_TICKERS * 10:  # ~10 data types per ticker
            # Remove oldest 20% of entries
            keys_to_remove = list(_cached_data.keys())[:int(len(_cached_data) * 0.2)]
            for key in keys_to_remove:
                del _cached_data[key]

        key = f"{ticker}:{data_type}"
        _cached_data[key] = data

    # Persist to DB for survival across restarts (async to not slow down)
    if persist_to_db and data_type in ["earnings", "revenue", "balance_sheet", "analyst", "institutional", "key_metrics", "yahoo_info",
                                        "earnings_calendar", "analyst_estimates", "short_interest"]:
        # Run DB save in background thread to not block
        import threading
        threading.Thread(target=save_ticker_to_db_cache, args=(ticker, data_type, data), daemon=True).start()


_cache_hit_count = 0
_cache_miss_count = 0

# Set to False to disable caching (for debugging)
CACHING_ENABLED = True

def fetch_with_cache(ticker: str, data_type: str, fetch_func, *args, **kwargs):
    """
    Wrapper that checks cache freshness before fetching.
    Cache hierarchy: Memory → Redis → Database → API fetch
    Returns cached data if fresh, otherwise fetches new data.
    """
    global _cache_hit_count, _cache_miss_count

    # Load DB cache on first access
    if not _db_cache_loaded:
        load_cache_from_db()

    # Skip caching if disabled
    if not CACHING_ENABLED:
        return fetch_func(*args, **kwargs)

    # 1. Check in-memory cache first (fastest)
    if is_data_fresh(ticker, data_type):
        cached = get_cached_data(ticker, data_type)
        if cached is not None:
            _cache_hit_count += 1
            if _cache_hit_count % 100 == 0:
                logger.info(f"Cache stats: {_cache_hit_count} hits, {_cache_miss_count} misses")
            return cached

    # 2. Check Redis cache (fast + persistent)
    if REDIS_AVAILABLE and redis_cache.enabled:
        try:
            redis_cached = redis_cache.get(ticker, data_type)
            if redis_cached is not None:
                # Store in memory cache for next time
                set_cached_data(ticker, data_type, redis_cached, persist_to_db=False)
                mark_data_fetched(ticker, data_type)
                _cache_hit_count += 1
                logger.debug(f"Redis cache hit: {ticker}:{data_type}")
                return redis_cached
        except Exception as e:
            logger.debug(f"Redis cache error: {e}")

    # 3. Fetch fresh data from API
    _cache_miss_count += 1
    data = fetch_func(*args, **kwargs)

    # 4. Cache the result in all layers
    if data:
        # Memory cache
        set_cached_data(ticker, data_type, data, persist_to_db=True)
        mark_data_fetched(ticker, data_type)

        # Redis cache (with automatic TTL)
        if REDIS_AVAILABLE and redis_cache.enabled:
            try:
                redis_cache.set(ticker, data_type, data)
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

    return data


def get_cache_hit_stats() -> dict:
    """Get cache hit/miss statistics"""
    return {"hits": _cache_hit_count, "misses": _cache_miss_count}


def get_cache_stats() -> dict:
    """Get cache statistics for debugging"""
    with _freshness_lock:
        freshness_count = sum(len(v) for v in _data_freshness_cache.values())
    with _cached_data_lock:
        data_count = len(_cached_data)

    stats = {
        "memory": {
            "tickers_tracked": len(_data_freshness_cache),
            "freshness_entries": freshness_count,
            "cached_data_entries": data_count
        },
        "hits": _cache_hit_count,
        "misses": _cache_miss_count,
    }

    # Add Redis stats if available
    if REDIS_AVAILABLE and redis_cache.enabled:
        try:
            stats["redis"] = redis_cache.get_stats()
        except Exception as e:
            stats["redis"] = {"error": str(e)}
    else:
        stats["redis"] = {"enabled": False}

    return stats


def get_rate_limit_stats() -> dict:
    """Get current rate limit statistics"""
    with _stats_lock:
        return _rate_limit_stats.copy()


def reset_rate_limit_stats():
    """Reset rate limit statistics"""
    with _stats_lock:
        _rate_limit_stats["errors_429"] = 0
        _rate_limit_stats["total_requests"] = 0
        _rate_limit_stats["last_reset"] = datetime.now()


def _track_request(status_code: int):
    """Track API request for rate limit stats"""
    with _stats_lock:
        _rate_limit_stats["total_requests"] += 1
        if status_code == 429:
            _rate_limit_stats["errors_429"] += 1


def _fmp_get(url: str, **kwargs) -> requests.Response:
    """Wrapper for FMP API requests with rate limit tracking"""
    resp = requests.get(url, **kwargs)
    _track_request(resp.status_code)
    return resp


# ============== DELISTED TICKER TRACKING ==============

def mark_ticker_as_delisted(ticker: str, reason: str = "no_data", source: str = None):
    """
    Mark a ticker as delisted/invalid so it's excluded from future scans.
    Uses database to persist across restarts.
    """
    db = _get_db_session()
    if not db:
        logger.warning(f"Could not mark {ticker} as delisted - no DB connection")
        return

    try:
        from backend.database import DelistedTicker
        from datetime import timedelta

        existing = db.query(DelistedTicker).filter(DelistedTicker.ticker == ticker).first()
        if existing:
            existing.failure_count += 1
            existing.last_failed_at = datetime.now()
            existing.reason = reason
            # After 3 failures, don't recheck for 30 days
            if existing.failure_count >= 3:
                existing.recheck_after = datetime.now() + timedelta(days=30)
        else:
            delisted = DelistedTicker(
                ticker=ticker,
                reason=reason,
                source=source,
                failure_count=1,
                recheck_after=datetime.now() + timedelta(days=7)  # Recheck after 7 days initially
            )
            db.add(delisted)

        db.commit()
        logger.info(f"Marked {ticker} as delisted/invalid: {reason}")
    except Exception as e:
        logger.debug(f"Failed to mark {ticker} as delisted: {e}")
        db.rollback()
    finally:
        db.close()


def get_delisted_tickers() -> set:
    """
    Get set of tickers that should be excluded from scans.
    Only excludes tickers with multiple confirmed failures to avoid false positives.
    """
    db = _get_db_session()
    if not db:
        return set()

    try:
        from backend.database import DelistedTicker

        # Only exclude tickers with 3+ failures (multiple confirmed issues)
        # This prevents temporary API issues from excluding valid stocks
        delisted = db.query(DelistedTicker.ticker).filter(
            DelistedTicker.failure_count >= 3
        ).all()

        return {t.ticker for t in delisted}
    except Exception as e:
        logger.debug(f"Failed to get delisted tickers: {e}")
        return set()
    finally:
        db.close()


def clear_delisted_ticker(ticker: str):
    """Remove a ticker from the delisted list (e.g., if it starts working again)"""
    db = _get_db_session()
    if not db:
        return

    try:
        from backend.database import DelistedTicker
        db.query(DelistedTicker).filter(DelistedTicker.ticker == ticker).delete()
        db.commit()
        logger.info(f"Removed {ticker} from delisted tickers")
    except Exception as e:
        logger.debug(f"Failed to clear delisted ticker {ticker}: {e}")
        db.rollback()
    finally:
        db.close()


def fetch_fmp_profile(ticker: str) -> dict:
    """Fetch company profile from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                profile = data[0]
                return {
                    "name": profile.get("companyName", ""),
                    "sector": profile.get("sector", ""),
                    "industry": profile.get("industry", ""),
                    "market_cap": profile.get("mktCap", 0),
                    "current_price": profile.get("price", 0),
                    "high_52w": profile.get("range", "").split("-")[-1].strip() if profile.get("range") else 0,
                    "shares_outstanding": profile.get("sharesOutstanding", 0) or 0,
                }
    except Exception as e:
        logger.debug(f"FMP profile error for {ticker}: {e}")
    return {}


def fetch_fmp_quote(ticker: str) -> dict:
    """Fetch current quote data from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                quote = data[0]
                return {
                    "current_price": quote.get("price", 0),
                    "high_52w": quote.get("yearHigh", 0),
                    "low_52w": quote.get("yearLow", 0),
                    "volume": quote.get("volume", 0),
                    "avg_volume": quote.get("avgVolume", 0),
                    "market_cap": quote.get("marketCap", 0),
                    "pe": quote.get("pe", 0),
                    "shares_outstanding": quote.get("sharesOutstanding", 0),
                }
    except Exception as e:
        logger.debug(f"FMP quote error for {ticker}: {e}")
    return {}


def fetch_fmp_key_metrics(ticker: str) -> dict:
    """Fetch key metrics including ROE, PE, and other valuation data from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                metrics = data[0]
                return {
                    "roe": metrics.get("returnOnEquity", 0) or 0,
                    "roa": metrics.get("returnOnAssets", 0) or 0,
                    "roic": metrics.get("returnOnInvestedCapital", 0) or 0,
                    "current_ratio": metrics.get("currentRatio", 0) or 0,
                    "earnings_yield": metrics.get("earningsYield", 0) or 0,
                    "fcf_yield": metrics.get("freeCashFlowYield", 0) or 0,
                }
    except Exception as e:
        logger.debug(f"FMP key metrics error for {ticker}: {e}")
    return {}


def fetch_fmp_earnings(ticker: str) -> dict:
    """Fetch quarterly and annual earnings from FMP"""
    if not FMP_API_KEY:
        logger.warning("FMP: No API key configured")
        return {}

    result = {"quarterly_eps": [], "annual_eps": []}

    try:
        # Quarterly income statement - using new /stable/ endpoint
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        logger.debug(f"FMP quarterly {ticker}: status={resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data:
                # EPS from income statement
                result["quarterly_eps"] = [q.get("eps", 0) or 0 for q in data]
                # Also get net income for backup
                result["quarterly_net_income"] = [q.get("netIncome", 0) or 0 for q in data]
                logger.debug(f"FMP quarterly {ticker}: got {len(data)} quarters, eps={result['quarterly_eps'][:3]}")
            else:
                logger.debug(f"FMP quarterly {ticker}: empty response")
        else:
            logger.debug(f"FMP quarterly {ticker}: error response: {resp.text[:200]}")
    except Exception as e:
        logger.debug(f"FMP quarterly earnings error for {ticker}: {e}")

    try:
        # Annual income statement - using new /stable/ endpoint
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result["annual_eps"] = [a.get("eps", 0) or 0 for a in data]
                result["annual_net_income"] = [a.get("netIncome", 0) or 0 for a in data]
    except Exception as e:
        logger.debug(f"FMP annual earnings error for {ticker}: {e}")

    return result


def fetch_finviz_institutional(ticker: str) -> float:
    """Fetch institutional ownership percentage from Finviz (scraping)"""
    import re
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # Extract institutional ownership percentage from HTML
            match = re.search(r'Inst Own</td><td[^>]*><b>([0-9.]+)%', resp.text)
            if match:
                pct = float(match.group(1))
                logger.debug(f"Finviz inst ownership for {ticker}: {pct:.1f}%")
                return pct
    except Exception as e:
        logger.debug(f"Finviz institutional error for {ticker}: {e}")
    return 0.0


def fetch_fmp_institutional(ticker: str) -> float:
    """Fetch institutional ownership percentage from FMP, fallback to Finviz"""
    # Try FMP first
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/institutional-holder?symbol={ticker}&apikey={FMP_API_KEY}"
            resp = _fmp_get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    # Sum up institutional shares and compare to outstanding
                    total_inst_shares = sum(h.get("shares", 0) or 0 for h in data[:50])  # Top 50 holders
                    if total_inst_shares > 0:
                        return total_inst_shares
        except Exception as e:
            logger.debug(f"FMP institutional error for {ticker}: {e}")

    # Fallback to Finviz (more reliable than Yahoo Finance from servers)
    return fetch_finviz_institutional(ticker)


def fetch_fmp_analyst(ticker: str) -> dict:
    """Fetch analyst ratings and price targets from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                est = data[0]
                return {
                    "estimated_eps_avg": est.get("estimatedEpsAvg", 0),
                    "estimated_eps_high": est.get("estimatedEpsHigh", 0),
                    "estimated_eps_low": est.get("estimatedEpsLow", 0),
                    "estimated_revenue_avg": est.get("estimatedRevenueAvg", 0),
                    "num_analysts": est.get("numberAnalystsEstimatedEps", 0),
                }
    except Exception as e:
        logger.debug(f"FMP analyst error for {ticker}: {e}")
    return {}


def fetch_fmp_price_target(ticker: str) -> dict:
    """Fetch analyst price targets from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                pt = data[0]
                return {
                    "target_high": pt.get("targetHigh", 0),
                    "target_low": pt.get("targetLow", 0),
                    "target_consensus": pt.get("targetConsensus", 0),
                    "target_median": pt.get("targetMedian", 0),
                }
    except Exception as e:
        logger.debug(f"FMP price target error for {ticker}: {e}")
    return {}


def fetch_fmp_earnings_surprise(ticker: str) -> dict:
    """
    Fetch earnings surprise history from FMP.
    IMPORTANT: This endpoint returns ADJUSTED EPS (actualEarningResult), which is what
    analysts track and what CANSLIM methodology uses. The income-statement endpoint
    returns GAAP EPS which includes stock-based compensation and is often negative
    for growth companies.
    """
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/earnings-surprises?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                # Get latest surprise and count consecutive beats
                latest = data[0]
                latest_surprise = 0
                if latest.get("estimatedEarning") and latest.get("actualEarningResult"):
                    estimated = latest.get("estimatedEarning", 0)
                    actual = latest.get("actualEarningResult", 0)
                    if estimated and estimated != 0:
                        latest_surprise = ((actual - estimated) / abs(estimated)) * 100

                # Count consecutive beats
                beat_streak = 0
                for record in data[:8]:  # Check last 8 quarters
                    estimated = record.get("estimatedEarning", 0)
                    actual = record.get("actualEarningResult", 0)
                    if estimated and actual and actual > estimated:
                        beat_streak += 1
                    else:
                        break

                # Extract quarterly ADJUSTED EPS (actualEarningResult) for CANSLIM scoring
                # This is the EPS that analysts track, not GAAP EPS
                quarterly_adjusted_eps = []
                for record in data[:8]:
                    actual = record.get("actualEarningResult")
                    if actual is not None:
                        quarterly_adjusted_eps.append(float(actual))

                return {
                    "latest_surprise_pct": latest_surprise,
                    "beat_streak": beat_streak,
                    "quarterly_adjusted_eps": quarterly_adjusted_eps,  # Adjusted EPS for CANSLIM
                }
    except Exception as e:
        logger.debug(f"FMP earnings surprise error for {ticker}: {e}")
    return {}


def fetch_fmp_earnings_calendar(ticker: str) -> dict:
    """
    Fetch earnings calendar data from FMP.
    Returns next earnings date and beat streak.

    Uses FMP /stable/earnings endpoint which has historical epsActual vs epsEstimated.
    Note: /stable/earnings-calendar only has 1 record per ticker, so we use /stable/earnings
    which has full historical data for calculating beat streaks.
    """
    if not FMP_API_KEY:
        return {}

    try:
        from datetime import date as date_type, timedelta

        # Fetch from FMP earnings (has full historical data for beat streak calculation)
        url = f"{FMP_BASE_URL}/earnings?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=15)

        if resp.status_code != 200:
            return {}

        all_data = resp.json()
        if not all_data:
            return {}

        # Filter for this ticker and sort by date descending
        data = [item for item in all_data if item.get('symbol') == ticker]
        data.sort(key=lambda x: x.get('date', ''), reverse=True)

        if not data:
            return {}

        next_earnings_date = None
        days_to_earnings = None
        beat_streak = 0
        today = date_type.today()

        # Find next earnings (where epsActual is None) and calculate beat streak
        for item in data:
            item_date_str = item.get('date')
            actual = item.get('epsActual')
            estimated = item.get('epsEstimated')

            if not item_date_str:
                continue

            item_date = datetime.strptime(item_date_str, '%Y-%m-%d').date()

            # Future earnings (no actual yet)
            if actual is None and item_date >= today:
                next_earnings_date = item_date_str
                days_to_earnings = (item_date - today).days
                continue

            # Past earnings - calculate beat streak
            if actual is not None and estimated is not None:
                if actual > estimated:
                    beat_streak += 1
                else:
                    break  # Stop at first non-beat

        # If no future earnings found, estimate from last earnings date
        if not next_earnings_date and data:
            last_date_str = data[0].get('date')
            if last_date_str:
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
                # Estimate next earnings ~90 days after last
                est_next = last_date + timedelta(days=90)
                while est_next <= today:
                    est_next += timedelta(days=90)
                next_earnings_date = est_next.strftime('%Y-%m-%d')
                days_to_earnings = (est_next - today).days

        if next_earnings_date or beat_streak > 0:
            return {
                "next_earnings_date": next_earnings_date,
                "days_to_earnings": days_to_earnings,
                "earnings_beat_streak": beat_streak,
            }
    except Exception as e:
        logger.debug(f"FMP earnings calendar error for {ticker}: {e}")
    return {}


def fetch_fmp_analyst_estimates(ticker: str) -> dict:
    """
    Fetch analyst estimate revisions from FMP.
    Compares current fiscal year estimate to prior year for revision tracking.

    API endpoint: /stable/analyst-estimates?symbol={ticker}&period=annual
    Returns: current estimate, prior estimate, revision %, and trend
    """
    if not FMP_API_KEY:
        return {}

    try:
        # Use period=annual (period=quarter requires premium)
        url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&period=annual&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) >= 2:
                # Find current year and prior year estimates
                from datetime import date
                current_year = date.today().year

                current = None
                prior = None

                for item in data:
                    item_date = item.get("date", "")
                    if item_date:
                        item_year = int(item_date[:4])
                        if item_year == current_year and current is None:
                            current = item
                        elif item_year == current_year - 1 and prior is None:
                            prior = item

                if not current:
                    current = data[0]  # Fallback to first record
                if not prior:
                    prior = data[1] if len(data) > 1 else None

                # Use epsAvg (not estimatedEpsAvg) based on actual API response
                current_eps = current.get("epsAvg", 0) or 0
                prior_eps = prior.get("epsAvg", 0) if prior else 0

                # Calculate revision percentage
                revision_pct = 0
                trend = "stable"
                if prior_eps and prior_eps != 0:
                    revision_pct = ((current_eps - prior_eps) / abs(prior_eps)) * 100
                    if revision_pct >= 5:
                        trend = "up"
                    elif revision_pct <= -5:
                        trend = "down"

                return {
                    "eps_estimate_current": current_eps,
                    "eps_estimate_prior": prior_eps,
                    "eps_estimate_revision_pct": round(revision_pct, 2),
                    "estimate_revision_trend": trend,
                    "num_analysts": current.get("numberAnalystsEstimatedEps", 0),
                }
    except Exception as e:
        logger.debug(f"FMP analyst estimates error for {ticker}: {e}")
    return {}


def fetch_fmp_revenue(ticker: str) -> dict:
    """Fetch quarterly and annual revenue from FMP income statements"""
    if not FMP_API_KEY:
        return {}

    result = {"quarterly_revenue": [], "annual_revenue": []}

    try:
        # Quarterly revenue
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result["quarterly_revenue"] = [q.get("revenue", 0) or 0 for q in data]
    except Exception as e:
        logger.debug(f"FMP quarterly revenue error for {ticker}: {e}")

    try:
        # Annual revenue
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result["annual_revenue"] = [a.get("revenue", 0) or 0 for a in data]
    except Exception as e:
        logger.debug(f"FMP annual revenue error for {ticker}: {e}")

    return result


def fetch_fmp_balance_sheet(ticker: str) -> dict:
    """Fetch balance sheet data for cash/debt analysis (growth stock funding)"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/balance-sheet-statement?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                bs = data[0]
                return {
                    "cash_and_equivalents": bs.get("cashAndCashEquivalents", 0) or 0,
                    "total_debt": bs.get("totalDebt", 0) or 0,
                    "total_assets": bs.get("totalAssets", 0) or 0,
                    "total_liabilities": bs.get("totalLiabilities", 0) or 0,
                }
    except Exception as e:
        logger.debug(f"FMP balance sheet error for {ticker}: {e}")
    return {}


def fetch_fmp_insider_trading(ticker: str) -> dict:
    """
    Fetch insider trading data from FMP API.
    Returns summary of insider buys/sells in last 3 months with $ values.

    API endpoint: /v4/insider-trading?symbol={ticker}

    Now tracks:
    - buy_value, sell_value, net_value ($ amounts)
    - largest_buy ($ amount of largest single purchase)
    - largest_buyer_title (CEO, CFO, etc.)
    """
    if not FMP_API_KEY:
        return {}

    try:
        # Use v4 endpoint for insider trading (not in /stable/)
        url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&limit=50&apikey={FMP_API_KEY}"
        resp = _fmp_get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            if not data:
                return {}

            # Filter to last 3 months
            cutoff_date = datetime.now() - timedelta(days=90)
            buy_count = 0
            sell_count = 0
            net_shares = 0
            buy_value = 0
            sell_value = 0
            largest_buy = 0
            largest_buyer_title = None

            for trade in data:
                # Parse transaction date
                trade_date_str = trade.get("transactionDate", "")
                if not trade_date_str:
                    continue

                try:
                    trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d")
                    if trade_date < cutoff_date:
                        continue
                except ValueError:
                    continue

                # Count buys and sells
                transaction_type = trade.get("transactionType", "").upper()
                shares = trade.get("securitiesTransacted", 0) or 0
                price = trade.get("price", 0) or 0
                trade_value = shares * price

                # Get insider title (reportingName often includes title, or use typeOfOwner)
                insider_title = trade.get("typeOfOwner", "") or ""
                reporting_name = trade.get("reportingName", "") or ""

                if "BUY" in transaction_type or "PURCHASE" in transaction_type or transaction_type == "P":
                    buy_count += 1
                    net_shares += shares
                    buy_value += trade_value

                    # Track largest buy
                    if trade_value > largest_buy:
                        largest_buy = trade_value
                        # Try to extract title from reporting name or type
                        if "CEO" in reporting_name.upper() or "CHIEF EXECUTIVE" in reporting_name.upper():
                            largest_buyer_title = "CEO"
                        elif "CFO" in reporting_name.upper() or "CHIEF FINANCIAL" in reporting_name.upper():
                            largest_buyer_title = "CFO"
                        elif "COO" in reporting_name.upper() or "CHIEF OPERATING" in reporting_name.upper():
                            largest_buyer_title = "COO"
                        elif "PRESIDENT" in reporting_name.upper():
                            largest_buyer_title = "PRESIDENT"
                        elif "DIRECTOR" in insider_title.upper() or "DIRECTOR" in reporting_name.upper():
                            largest_buyer_title = "DIRECTOR"
                        elif "10%" in insider_title:
                            largest_buyer_title = "10% OWNER"
                        else:
                            largest_buyer_title = insider_title or "OFFICER"

                elif "SELL" in transaction_type or "SALE" in transaction_type or transaction_type == "S":
                    sell_count += 1
                    net_shares -= shares
                    sell_value += trade_value

            net_value = buy_value - sell_value

            # Determine sentiment based on $ value (more meaningful than count)
            if net_value >= 100000:
                sentiment = "bullish"
            elif net_value <= -100000:
                sentiment = "bearish"
            elif buy_count > sell_count * 1.5:
                sentiment = "bullish"
            elif sell_count > buy_count * 1.5:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            logger.debug(f"{ticker}: Insider trading - {buy_count} buys (${buy_value:,.0f}), {sell_count} sells (${sell_value:,.0f}), net ${net_value:+,.0f}, {sentiment}")

            return {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "net_shares": net_shares,
                "sentiment": sentiment,
                # New value tracking fields
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_value": net_value,
                "largest_buy": largest_buy,
                "largest_buyer_title": largest_buyer_title,
            }

    except Exception as e:
        logger.debug(f"FMP insider trading error for {ticker}: {e}")

    return {}


def fetch_short_interest(ticker: str) -> dict:
    """
    Fetch short interest data from Yahoo Finance.
    Returns short interest percentage and days to cover (short ratio).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if info:
            short_pct = info.get("shortPercentOfFloat", 0) or 0
            short_ratio = info.get("shortRatio", 0) or 0

            # Convert to percentage if needed
            if short_pct > 0 and short_pct < 1:
                short_pct = short_pct * 100

            logger.debug(f"{ticker}: Short interest - {short_pct:.2f}% of float, {short_ratio:.1f} days to cover")

            return {
                "short_interest_pct": short_pct,
                "short_ratio": short_ratio
            }

    except Exception as e:
        logger.debug(f"Short interest error for {ticker}: {e}")

    return {}


def fetch_weekly_price_history(ticker: str) -> list:
    """Fetch weekly OHLC data for base pattern detection"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1wk", "range": "6mo"}  # 6 months of weekly data
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                timestamps = result[0].get("timestamp", [])
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]

                weekly_data = []
                for i, ts in enumerate(timestamps):
                    weekly_data.append({
                        "timestamp": ts,
                        "open": indicators.get("open", [])[i] if i < len(indicators.get("open", [])) else None,
                        "high": indicators.get("high", [])[i] if i < len(indicators.get("high", [])) else None,
                        "low": indicators.get("low", [])[i] if i < len(indicators.get("low", [])) else None,
                        "close": indicators.get("close", [])[i] if i < len(indicators.get("close", [])) else None,
                        "volume": indicators.get("volume", [])[i] if i < len(indicators.get("volume", [])) else None,
                    })
                return weekly_data
    except Exception as e:
        logger.debug(f"Weekly price history error for {ticker}: {e}")
    return []


def fetch_price_from_chart_api(ticker: str) -> dict:
    """Fallback: fetch basic price data from Yahoo chart API"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": "1y"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                timestamps = result[0].get("timestamp", [])

                return {
                    "current_price": meta.get("regularMarketPrice") or meta.get("previousClose"),
                    "high_52w": meta.get("fiftyTwoWeekHigh"),
                    "low_52w": meta.get("fiftyTwoWeekLow"),  # Added 52-week low
                    "market_cap": meta.get("marketCap"),  # Added market cap from Yahoo
                    "name": meta.get("longName") or meta.get("shortName") or ticker,
                    "close_prices": indicators.get("close", []),
                    "volumes": indicators.get("volume", []),
                    "timestamps": timestamps
                }
    except Exception:
        pass
    return {}


# ============== Multi-Index Market Direction ==============

# Index weights for market direction
MARKET_INDEX_WEIGHTS = {
    "SPY": 0.50,  # S&P 500 - broad market
    "QQQ": 0.30,  # NASDAQ 100 - tech/growth
    "DIA": 0.20,  # Dow Jones - blue chips
}


def calculate_index_signal(price: float, ma_50: float, ma_200: float) -> int:
    """
    Calculate signal for a single index based on MA positions.
    Returns: -1 (bearish), 0 (neutral), 1 (bullish), 2 (strong bullish)
    """
    if price <= 0 or ma_200 <= 0:
        return 0  # No data, neutral

    above_200 = price > ma_200
    above_50 = price > ma_50 if ma_50 > 0 else True

    if above_200 and above_50:
        return 2  # Strong bullish - above both MAs
    elif above_200 and not above_50:
        return 1  # Bullish - above 200 but below 50 (minor pullback)
    elif not above_200 and above_50:
        return 0  # Neutral - below 200 but above 50 (recovery attempt)
    else:
        return -1  # Bearish - below both MAs


def fetch_market_direction_data() -> dict:
    """
    Fetch market direction data for SPY, QQQ, and DIA.
    Returns comprehensive market analysis with weighted signal.

    This is designed to be called ONCE at startup or on manual refresh,
    then cached for several hours to avoid rate limiting during stock scans.
    """
    result = {
        "success": False,
        "indexes": {},
        "weighted_signal": 0,
        "market_score": 7.5,  # Default neutral M score (half of 15)
        "market_trend": "neutral",
        "error": None,
    }

    indexes_data = {}
    total_weight = 0
    weighted_sum = 0

    for ticker, weight in MARKET_INDEX_WEIGHTS.items():
        index_data = {
            "ticker": ticker,
            "price": 0,
            "ma_50": 0,
            "ma_200": 0,
            "signal": 0,
            "status": "unknown",
        }

        try:
            # Try Yahoo Chart API first (more reliable)
            chart_data = fetch_price_from_chart_api(ticker)

            if chart_data.get("close_prices") and len(chart_data["close_prices"]) >= 50:
                close_prices = [p for p in chart_data["close_prices"] if p is not None]

                if len(close_prices) >= 200:
                    index_data["price"] = close_prices[-1]
                    index_data["ma_50"] = sum(close_prices[-50:]) / 50
                    index_data["ma_200"] = sum(close_prices[-200:]) / 200
                    index_data["signal"] = calculate_index_signal(
                        index_data["price"],
                        index_data["ma_50"],
                        index_data["ma_200"]
                    )
                    index_data["status"] = "ok"

                    # Add to weighted calculation
                    weighted_sum += index_data["signal"] * weight
                    total_weight += weight

                    logger.info(f"Market {ticker}: ${index_data['price']:.2f}, "
                               f"50MA: ${index_data['ma_50']:.2f}, "
                               f"200MA: ${index_data['ma_200']:.2f}, "
                               f"signal: {index_data['signal']}")
                elif len(close_prices) >= 50:
                    # Partial data - use what we have
                    index_data["price"] = close_prices[-1]
                    index_data["ma_50"] = sum(close_prices[-50:]) / 50
                    # Estimate 200 MA from available data
                    index_data["ma_200"] = sum(close_prices) / len(close_prices)
                    index_data["signal"] = calculate_index_signal(
                        index_data["price"],
                        index_data["ma_50"],
                        index_data["ma_200"]
                    )
                    index_data["status"] = "partial"
                    weighted_sum += index_data["signal"] * weight
                    total_weight += weight
                    logger.warning(f"Market {ticker}: Using partial data ({len(close_prices)} days)")
                else:
                    index_data["status"] = "insufficient_data"
                    logger.warning(f"Market {ticker}: Insufficient price history")
            else:
                index_data["status"] = "fetch_failed"
                logger.warning(f"Market {ticker}: Failed to fetch chart data")

        except Exception as e:
            index_data["status"] = "error"
            index_data["error"] = str(e)
            logger.error(f"Market {ticker} error: {e}")

        indexes_data[ticker] = index_data

    result["indexes"] = indexes_data

    # Calculate weighted signal and market score
    if total_weight > 0:
        result["weighted_signal"] = weighted_sum / total_weight
        result["success"] = True

        # Convert weighted signal (-1 to +2) to M score (0 to 15)
        # -1 -> 0, 0 -> 7.5, 1 -> 11.25, 2 -> 15
        weighted_signal = result["weighted_signal"]
        if weighted_signal >= 1.5:
            result["market_score"] = 15.0
            result["market_trend"] = "bullish"
        elif weighted_signal >= 0.5:
            result["market_score"] = 12.0
            result["market_trend"] = "bullish"
        elif weighted_signal >= 0:
            result["market_score"] = 9.0
            result["market_trend"] = "neutral"
        elif weighted_signal >= -0.5:
            result["market_score"] = 5.0
            result["market_trend"] = "neutral"
        else:
            result["market_score"] = 2.0
            result["market_trend"] = "bearish"

        logger.info(f"Market Direction: weighted_signal={weighted_signal:.2f}, "
                   f"score={result['market_score']}, trend={result['market_trend']}")
    else:
        result["error"] = "Could not fetch any index data"
        logger.error("Market Direction: No index data available, using defaults")

    return result


# Cached market direction (refreshed at startup and periodically)
_cached_market_direction = None
_market_direction_timestamp = None
MARKET_CACHE_DURATION = 4 * 3600  # 4 hours


def get_cached_market_direction(force_refresh: bool = False) -> dict:
    """
    Get cached market direction data, refreshing if stale or forced.
    """
    global _cached_market_direction, _market_direction_timestamp

    now = datetime.now()

    # Check if cache is valid
    if not force_refresh and _cached_market_direction is not None:
        if _market_direction_timestamp is not None:
            age = (now - _market_direction_timestamp).total_seconds()
            if age < MARKET_CACHE_DURATION:
                logger.debug(f"Using cached market direction (age: {age/60:.0f} min)")
                return _cached_market_direction

    # Fetch fresh data
    logger.info("Fetching fresh market direction data...")
    _cached_market_direction = fetch_market_direction_data()
    _market_direction_timestamp = now

    return _cached_market_direction


class StockData:
    """Container for all stock data needed for CANSLIM analysis"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.name: str = ""
        self.sector: str = ""
        self.current_price: float = 0.0
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.quarterly_earnings: list[float] = []
        self.annual_earnings: list[float] = []
        self.institutional_holders_pct: float = 0.0
        self.shares_outstanding: int = 0
        self.avg_volume_50d: float = 0.0
        self.current_volume: float = 0.0
        self.high_52w: float = 0.0
        self.low_52w: float = 0.0
        self.market_cap: float = 0.0
        self.is_valid: bool = False
        self.error_message: str = ""

        # New fields for refined model
        self.analyst_target_price: float = 0.0
        self.analyst_target_low: float = 0.0
        self.analyst_target_high: float = 0.0
        self.analyst_recommendation: str = ""  # buy, hold, sell
        self.num_analyst_opinions: int = 0
        self.forward_pe: float = 0.0
        self.trailing_pe: float = 0.0
        self.peg_ratio: float = 0.0
        self.earnings_growth_estimate: float = 0.0  # Next year growth estimate

        # Key financial metrics for improved scoring
        self.roe: float = 0.0  # Return on Equity
        self.roa: float = 0.0  # Return on Assets
        self.roic: float = 0.0  # Return on Invested Capital
        self.earnings_yield: float = 0.0
        self.fcf_yield: float = 0.0

        # Enhanced earnings analysis
        self.quarterly_revenue: list[float] = []  # Last 8 quarters revenue
        self.annual_revenue: list[float] = []  # Last 5 years revenue
        self.earnings_surprise_pct: float = 0.0  # Latest earnings surprise %
        self.eps_beat_streak: int = 0  # Consecutive quarters of beating estimates

        # Technical analysis data
        self.weekly_price_history: list[dict] = []  # Weekly OHLC for base detection
        self.cash_and_equivalents: float = 0.0  # For growth stock funding analysis
        self.total_debt: float = 0.0  # For growth stock funding analysis

        # P1 Features (Feb 2026)
        # Earnings Calendar
        self.next_earnings_date: str = ""  # YYYY-MM-DD format
        self.days_to_earnings: int = 0  # Days until next earnings
        self.earnings_beat_streak: int = 0  # Consecutive quarters beating estimates

        # Analyst Estimate Revisions
        self.eps_estimate_current: float = 0.0
        self.eps_estimate_prior: float = 0.0
        self.eps_estimate_revision_pct: float = 0.0
        self.estimate_revision_trend: str = ""  # 'up', 'down', 'stable'

        # Insider Value Tracking
        self.insider_buy_value: float = 0.0
        self.insider_sell_value: float = 0.0
        self.insider_net_value: float = 0.0
        self.insider_largest_buy: float = 0.0
        self.insider_largest_buyer_title: str = ""


class DataFetcher:
    """Fetches and caches stock data using FMP API and Yahoo chart API"""

    # Maximum cache size to prevent memory growth
    MAX_CACHE_SIZE = 1000

    def __init__(self):
        self._cache: OrderedDict[str, StockData] = OrderedDict()
        self._sp500_history: Optional[pd.DataFrame] = None

    def get_stock_data(self, ticker: str, retries: int = 2) -> StockData:
        """
        Fetch all required data for a stock.
        Uses FMP API for earnings/fundamentals, Yahoo chart API for price history.
        """
        if ticker in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(ticker)
            return self._cache[ticker]

        stock_data = StockData(ticker)

        # 1. Get price history from Yahoo chart API (reliable)
        chart_data = fetch_price_from_chart_api(ticker)
        if chart_data.get("current_price"):
            stock_data.current_price = chart_data["current_price"]
            stock_data.high_52w = chart_data.get("high_52w", 0) or 0
            stock_data.low_52w = chart_data.get("low_52w", 0) or 0
            stock_data.market_cap = chart_data.get("market_cap", 0) or 0
            stock_data.name = chart_data.get("name", ticker)

            # Build price history from chart data
            close_prices = chart_data.get("close_prices", [])
            volumes = chart_data.get("volumes", [])
            timestamps = chart_data.get("timestamps", [])

            if len(close_prices) >= 50:
                dates = pd.to_datetime(timestamps, unit='s')
                stock_data.price_history = pd.DataFrame({
                    'Close': close_prices,
                    'Volume': volumes
                }, index=dates)

                # Calculate volume metrics from chart data
                if volumes:
                    recent_volumes = [v for v in volumes[-50:] if v]
                    stock_data.avg_volume_50d = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
                    stock_data.current_volume = volumes[-1] if volumes[-1] else 0

        # 2. Get company profile and quote from FMP
        if FMP_API_KEY:
            profile = fetch_fmp_profile(ticker)
            if profile:
                stock_data.name = profile.get("name") or stock_data.name
                stock_data.sector = profile.get("sector", "")
                stock_data.shares_outstanding = int(profile.get("shares_outstanding", 0) or 0)
                if not stock_data.current_price:
                    stock_data.current_price = profile.get("current_price", 0)
                if not stock_data.high_52w:
                    try:
                        stock_data.high_52w = float(profile.get("high_52w", 0) or 0)
                    except:
                        pass

            quote = fetch_fmp_quote(ticker)
            if quote:
                if not stock_data.current_price:
                    stock_data.current_price = quote.get("current_price", 0)
                if not stock_data.high_52w:
                    stock_data.high_52w = quote.get("high_52w", 0) or 0
                if not stock_data.low_52w:
                    stock_data.low_52w = quote.get("low_52w", 0) or 0
                if not stock_data.market_cap:
                    stock_data.market_cap = quote.get("market_cap", 0)
                if not stock_data.avg_volume_50d:
                    stock_data.avg_volume_50d = quote.get("avg_volume", 0)
                if not stock_data.current_volume:
                    stock_data.current_volume = quote.get("volume", 0)
                stock_data.trailing_pe = quote.get("pe", 0) or 0
                if not stock_data.shares_outstanding:
                    stock_data.shares_outstanding = int(quote.get("shares_outstanding", 0) or 0)

            # Fallback: Calculate 52-week high/low from price history if still missing
            # Only use if we have ~1 year of data (250+ trading days) to avoid incorrect values
            if (not stock_data.high_52w or not stock_data.low_52w) and not stock_data.price_history.empty:
                try:
                    closes = stock_data.price_history['Close'].dropna()
                    # Require at least 250 trading days for accurate 52-week calculation
                    if len(closes) >= 250:
                        if not stock_data.high_52w:
                            stock_data.high_52w = float(closes.max())
                        if not stock_data.low_52w:
                            stock_data.low_52w = float(closes.min())
                        logger.debug(f"Calculated 52w range for {ticker}: ${stock_data.low_52w:.2f} - ${stock_data.high_52w:.2f}")
                    elif len(closes) > 0:
                        logger.debug(f"{ticker}: Only {len(closes)} days of data, skipping 52w calculation")
                except Exception as e:
                    logger.debug(f"Error calculating 52w range for {ticker}: {e}")

            # 3. Get earnings data from FMP (critical for C and A scores)
            # TIERED: Cache for 24 hours (earnings don't change intraday)
            earnings = fetch_with_cache(ticker, "earnings", fetch_fmp_earnings, ticker)
            if earnings:
                stock_data.quarterly_earnings = earnings.get("quarterly_eps", [])
                stock_data.annual_earnings = earnings.get("annual_eps", [])
                logger.debug(f"FMP {ticker}: quarterly_earnings={stock_data.quarterly_earnings[:3] if stock_data.quarterly_earnings else 'EMPTY'}")

            # 3b. Get key metrics (ROE, etc.) from FMP
            # TIERED: Cache for 24 hours
            key_metrics = fetch_with_cache(ticker, "key_metrics", fetch_fmp_key_metrics, ticker)
            if key_metrics:
                stock_data.roe = key_metrics.get("roe", 0)
                stock_data.roa = key_metrics.get("roa", 0)
                stock_data.roic = key_metrics.get("roic", 0)
                stock_data.earnings_yield = key_metrics.get("earnings_yield", 0)
                stock_data.fcf_yield = key_metrics.get("fcf_yield", 0)

            # 4. Get institutional ownership from FMP (or Yahoo fallback)
            # TIERED: Cache for 7 days (institutional holdings rarely change)
            inst_result = fetch_with_cache(ticker, "institutional", fetch_fmp_institutional, ticker)
            if inst_result:
                # If result is > 100, it's likely shares from FMP - convert to percentage
                if inst_result > 100 and stock_data.shares_outstanding:
                    stock_data.institutional_holders_pct = (inst_result / stock_data.shares_outstanding) * 100
                else:
                    # Already a percentage (from Yahoo Finance fallback)
                    stock_data.institutional_holders_pct = inst_result
                # Cap at 100% in case of data issues
                stock_data.institutional_holders_pct = min(stock_data.institutional_holders_pct, 100)

            # 5. Get analyst data from FMP
            # TIERED: Cache for 24 hours
            # NOTE: Using distinct cache keys to avoid collision
            price_target = fetch_with_cache(ticker, "price_target", fetch_fmp_price_target, ticker)
            if price_target:
                stock_data.analyst_target_price = price_target.get("target_consensus", 0) or price_target.get("target_median", 0)
                stock_data.analyst_target_high = price_target.get("target_high", 0)
                stock_data.analyst_target_low = price_target.get("target_low", 0)

            analyst = fetch_with_cache(ticker, "analyst_estimates", fetch_fmp_analyst, ticker)
            if analyst:
                stock_data.num_analyst_opinions = analyst.get("num_analysts", 0)
                stock_data.earnings_growth_estimate = analyst.get("estimated_eps_avg", 0)

            # 5b. Get earnings surprise data (for enhanced C score)
            # TIERED: Cache for 24 hours
            # IMPORTANT: This also provides ADJUSTED EPS which is more accurate for CANSLIM
            # than GAAP EPS from income-statement (which includes stock-based comp)
            earnings_surprise = fetch_with_cache(ticker, "earnings_surprise", fetch_fmp_earnings_surprise, ticker)
            if earnings_surprise:
                stock_data.earnings_surprise_pct = earnings_surprise.get("latest_surprise_pct", 0)
                stock_data.eps_beat_streak = earnings_surprise.get("beat_streak", 0)

                # Use ADJUSTED EPS if available (preferred for CANSLIM scoring)
                # This is the EPS analysts track, not GAAP EPS which is often distorted by SBC
                adjusted_eps = earnings_surprise.get("quarterly_adjusted_eps", [])
                if adjusted_eps and len(adjusted_eps) >= 4:
                    logger.debug(f"{ticker}: Using ADJUSTED EPS from earnings-surprise: {adjusted_eps[:4]}")
                    stock_data.quarterly_earnings = adjusted_eps
                elif adjusted_eps:
                    # Partial data - merge with GAAP EPS
                    logger.debug(f"{ticker}: Partial adjusted EPS ({len(adjusted_eps)} quarters), supplementing with GAAP")
                    # Prefer adjusted for available quarters, fill rest with GAAP
                    gaap_eps = stock_data.quarterly_earnings
                    merged = adjusted_eps + gaap_eps[len(adjusted_eps):] if len(gaap_eps) > len(adjusted_eps) else adjusted_eps
                    stock_data.quarterly_earnings = merged[:8]

            # 5c. Get revenue data (for growth mode scoring)
            # TIERED: Cache for 24 hours
            revenue_data = fetch_with_cache(ticker, "revenue", fetch_fmp_revenue, ticker)
            if revenue_data:
                stock_data.quarterly_revenue = revenue_data.get("quarterly_revenue", [])
                stock_data.annual_revenue = revenue_data.get("annual_revenue", [])

            # 5d. Get balance sheet data (for growth stock funding analysis)
            # TIERED: Cache for 24 hours
            balance_sheet = fetch_with_cache(ticker, "balance_sheet", fetch_fmp_balance_sheet, ticker)
            if balance_sheet:
                stock_data.cash_and_equivalents = balance_sheet.get("cash_and_equivalents", 0)
                stock_data.total_debt = balance_sheet.get("total_debt", 0)

            # 5e. Get weekly price history (for technical analysis / base detection)
            # TIERED: Cache for 24 hours (pattern detection doesn't need real-time)
            stock_data.weekly_price_history = fetch_with_cache(ticker, "weekly_history", fetch_weekly_price_history, ticker) or []

            # 6. Yahoo Finance fallback for analyst/valuation data (rate-limited)
            # Only call if multiple critical fields are missing to minimize API calls
            missing_fields = sum([
                not stock_data.analyst_target_price,
                not stock_data.trailing_pe,
                not stock_data.num_analyst_opinions
            ])
            if missing_fields >= 2:  # Only fetch if 2+ fields missing
                try:
                    import time
                    time.sleep(0.3)  # Rate limit protection
                    yf_stock = yf.Ticker(ticker)
                    yf_info = yf_stock.info
                    if yf_info:
                        # Analyst target price fallback
                        if not stock_data.analyst_target_price:
                            stock_data.analyst_target_price = yf_info.get('targetMeanPrice', 0) or 0
                            stock_data.analyst_target_high = yf_info.get('targetHighPrice', 0) or 0
                            stock_data.analyst_target_low = yf_info.get('targetLowPrice', 0) or 0
                        # Number of analysts fallback
                        if not stock_data.num_analyst_opinions:
                            stock_data.num_analyst_opinions = yf_info.get('numberOfAnalystOpinions', 0) or 0
                        # Analyst recommendation fallback
                        if not stock_data.analyst_recommendation:
                            stock_data.analyst_recommendation = yf_info.get('recommendationKey', '') or ''
                        # P/E ratio fallback
                        if not stock_data.trailing_pe:
                            stock_data.trailing_pe = yf_info.get('trailingPE', 0) or 0
                        # PEG ratio fallback
                        if not stock_data.peg_ratio:
                            stock_data.peg_ratio = yf_info.get('pegRatio', 0) or 0
                        # Earnings growth estimate fallback
                        if not stock_data.earnings_growth_estimate:
                            growth = yf_info.get('earningsQuarterlyGrowth', 0) or yf_info.get('earningsGrowth', 0)
                            stock_data.earnings_growth_estimate = growth or 0
                except Exception:
                    pass  # Silent fallback failure - FMP data is primary

        # 3. Use yfinance to supplement/override FMP data
        # ALWAYS run to get adjusted EPS from earnings_history (FMP only has GAAP EPS)
        # This runs regardless of whether FMP_API_KEY is set
        if True:  # Always run to get adjusted EPS
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if info and info.get('regularMarketPrice'):
                    if not stock_data.sector:
                        stock_data.sector = info.get('sector', 'Unknown')
                    if not stock_data.shares_outstanding:
                        stock_data.shares_outstanding = info.get('sharesOutstanding', 0)
                    if not stock_data.institutional_holders_pct:
                        inst_pct = info.get('heldPercentInstitutions', 0)
                        stock_data.institutional_holders_pct = (inst_pct * 100) if inst_pct else 0
                    # Get ROE (critical for A score quality check)
                    # Store as decimal (e.g., 0.05 = 5%) - same format as FMP
                    if not stock_data.roe:
                        roe = info.get('returnOnEquity')
                        stock_data.roe = roe if roe else 0

                    # Get adjusted EPS from Yahoo - this is what analysts track (matches Fidelity)
                    # FMP returns GAAP EPS which includes stock-based comp and distorts growth metrics
                    # Check cache first to avoid excessive Yahoo calls
                    cached_adjusted = get_cached_data(ticker, "adjusted_eps")
                    if cached_adjusted and is_data_fresh(ticker, "adjusted_eps"):
                        # Use cached adjusted EPS
                        stock_data.quarterly_earnings = cached_adjusted
                        logger.debug(f"{ticker}: Using CACHED adjusted EPS: {cached_adjusted[:4]}")
                    else:
                        # Fetch fresh adjusted EPS from Yahoo
                        try:
                            earnings_hist = stock.earnings_history
                            if earnings_hist is not None and not earnings_hist.empty and 'epsActual' in earnings_hist.columns:
                                eps_actual = earnings_hist['epsActual'].dropna().tolist()
                                if eps_actual and len(eps_actual) >= 4:
                                    # Reverse to get most recent first
                                    yahoo_eps = eps_actual[::-1][:8]
                                    old_eps = stock_data.quarterly_earnings[:4] if stock_data.quarterly_earnings else []
                                    # ALWAYS prefer Yahoo adjusted EPS over FMP GAAP EPS
                                    stock_data.quarterly_earnings = yahoo_eps
                                    # Cache the adjusted EPS for 7 days
                                    set_cached_data(ticker, "adjusted_eps", yahoo_eps)
                                    mark_data_fetched(ticker, "adjusted_eps")
                                    logger.info(f"{ticker}: Using Yahoo ADJUSTED EPS {yahoo_eps[:4]} (was GAAP: {old_eps})")

                                    # Also get earnings surprise data
                                    if 'surprisePercent' in earnings_hist.columns and not stock_data.earnings_surprise_pct:
                                        latest_surprise = earnings_hist['surprisePercent'].iloc[-1]
                                        if pd.notna(latest_surprise):
                                            stock_data.earnings_surprise_pct = latest_surprise * 100
                        except Exception as e:
                            logger.debug(f"{ticker}: earnings_history failed: {e}")

                    # Fallback to Net Income calculation ONLY if we have NO earnings data at all
                    # Don't overwrite adjusted EPS with GAAP-based Net Income calculation
                    if not stock_data.quarterly_earnings:
                        try:
                            quarterly = stock.quarterly_financials
                            if quarterly is not None and not quarterly.empty:
                                if 'Net Income' in quarterly.index:
                                    net_income = quarterly.loc['Net Income'].dropna()
                                    shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                                    quarterly_eps = (net_income / shares).tolist()[:8]
                                    if len(quarterly_eps) >= 4:
                                        stock_data.quarterly_earnings = quarterly_eps
                                        logger.debug(f"{ticker}: Fallback to Yahoo Net Income EPS (no other data): {quarterly_eps[:4]}")
                        except Exception:
                            pass

                    # Annual earnings from yfinance
                    if not stock_data.annual_earnings:
                        try:
                            annual = stock.financials
                            if annual is not None and not annual.empty:
                                if 'Net Income' in annual.index:
                                    net_income = annual.loc['Net Income'].dropna()
                                    shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                                    stock_data.annual_earnings = (net_income / shares).tolist()[:5]
                        except Exception:
                            pass

            except Exception as e:
                stock_data.error_message = str(e)

        # Mark as valid if we have basic price data (from chart API or yfinance)
        if stock_data.current_price and not stock_data.price_history.empty and len(stock_data.price_history) >= 50:
            stock_data.is_valid = True
        elif stock_data.current_price:
            # Partial data - still somewhat usable
            stock_data.is_valid = True
            stock_data.error_message = "Limited data available"

        # Add to cache with LRU eviction
        self._cache[ticker] = stock_data
        if len(self._cache) > self.MAX_CACHE_SIZE:
            # Remove oldest (first) item
            self._cache.popitem(last=False)
        return stock_data

    def get_price_data_only(self, ticker: str) -> StockData:
        """
        Fetch ONLY price history for ETFs and indexes (no fundamentals)
        Much faster and avoids errors for securities without earnings data

        Use this for:
        - ETFs (XLK, XLV, SPY, QQQ, etc.)
        - Indexes (^GSPC, ^DJI, etc.)
        - Sector performance calculations
        """
        stock_data = StockData(ticker)

        # Only fetch price history from Yahoo chart API
        chart_data = fetch_price_from_chart_api(ticker)
        if chart_data.get("current_price"):
            stock_data.current_price = chart_data["current_price"]
            stock_data.high_52w = chart_data.get("high_52w", 0) or 0
            stock_data.low_52w = chart_data.get("low_52w", 0) or 0
            stock_data.market_cap = chart_data.get("market_cap", 0) or 0
            stock_data.name = chart_data.get("name", ticker)

            close_prices = chart_data.get("close_prices", [])
            volumes = chart_data.get("volumes", [])
            timestamps = chart_data.get("timestamps", [])

            if len(close_prices) >= 50:
                dates = pd.to_datetime(timestamps, unit='s')
                stock_data.price_history = pd.DataFrame({
                    'Close': close_prices,
                    'Volume': volumes
                }, index=dates)

                if volumes:
                    recent_volumes = [v for v in volumes[-50:] if v]
                    stock_data.avg_volume_50d = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
                    stock_data.current_volume = volumes[-1] if volumes[-1] else 0

        # Mark as valid if we have price data
        if stock_data.current_price and not stock_data.price_history.empty:
            stock_data.is_valid = True

        return stock_data

    def get_sp500_history(self) -> pd.DataFrame:
        """Fetch S&P 500 index price history for relative strength calculation"""
        if self._sp500_history is not None:
            return self._sp500_history

        # Try Yahoo chart API first (more reliable from servers)
        chart_data = fetch_price_from_chart_api("SPY")
        if chart_data.get("close_prices"):
            close_prices = chart_data.get("close_prices", [])
            timestamps = chart_data.get("timestamps", [])
            if len(close_prices) >= 50:
                dates = pd.to_datetime(timestamps, unit='s')
                self._sp500_history = pd.DataFrame({
                    'Close': close_prices,
                }, index=dates)
                return self._sp500_history

        # Fallback to yfinance
        try:
            spy = yf.Ticker("SPY")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self._sp500_history = spy.history(start=start_date, end=end_date)
        except Exception:
            self._sp500_history = pd.DataFrame()

        return self._sp500_history

    def get_market_direction(self) -> tuple[bool, float, float]:
        """
        Determine market direction based on S&P 500.
        Returns: (is_bullish, pct_above_200ma, pct_above_50ma)
        """
        history = self.get_sp500_history()

        if history.empty or len(history) < 200:
            return True, 0, 0  # Default to bullish if no data

        close = history['Close']
        current_price = close.iloc[-1]

        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()

        pct_above_200 = ((current_price - ma_200) / ma_200) * 100
        pct_above_50 = ((current_price - ma_50) / ma_50) * 100

        is_bullish = current_price > ma_200

        return is_bullish, pct_above_200, pct_above_50

    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        self._sp500_history = None


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()

    print("Testing with AAPL...")
    data = fetcher.get_stock_data("AAPL")
    print(f"Name: {data.name}")
    print(f"Sector: {data.sector}")
    print(f"Current Price: ${data.current_price:.2f}")
    print(f"52-week High: ${data.high_52w:.2f}")
    print(f"Institutional %: {data.institutional_holders_pct:.1f}%")
    print(f"Quarterly Earnings: {data.quarterly_earnings[:4]}")
    print(f"Is Valid: {data.is_valid}")

    # New analyst data
    print(f"\nAnalyst Data:")
    print(f"  Target Price: ${data.analyst_target_price:.2f}")
    print(f"  Target Range: ${data.analyst_target_low:.2f} - ${data.analyst_target_high:.2f}")
    print(f"  Recommendation: {data.analyst_recommendation}")
    print(f"  # of Analysts: {data.num_analyst_opinions}")
    print(f"  Forward P/E: {data.forward_pe:.1f}")
    print(f"  PEG Ratio: {data.peg_ratio:.2f}")

    is_bullish, pct_200, pct_50 = fetcher.get_market_direction()
    print(f"\nMarket: {'Bullish' if is_bullish else 'Bearish'}")
    print(f"S&P 500 vs 200-day MA: {pct_200:+.1f}%")
    print(f"S&P 500 vs 50-day MA: {pct_50:+.1f}%")
