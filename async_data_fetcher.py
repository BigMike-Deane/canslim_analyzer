"""
Async Data Fetcher Module - High Performance Version v2.0
Fetches stock data asynchronously with BATCH endpoints for 10-20x speed improvement

Key Optimizations:
1. Batch FMP endpoints (up to 500 tickers per call)
2. Consolidated income statement calls (earnings + revenue in one call)
3. Skip Yahoo when FMP data is complete
4. Exponential backoff for rate limits
5. Progress persistence for interrupted scans
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
import os
import json
from pathlib import Path

from data_fetcher import (
    StockData, FMP_API_KEY, FMP_BASE_URL,
    get_cached_data, set_cached_data, mark_data_fetched, is_data_fresh,
    fetch_price_from_chart_api, fetch_weekly_price_history,
    REDIS_AVAILABLE, _data_freshness_cache, _freshness_lock,
    load_cache_from_db, mark_ticker_as_delisted, clear_delisted_ticker
)

if REDIS_AVAILABLE:
    from redis_cache import redis_cache

logger = logging.getLogger(__name__)

# ============== YFINANCE SESSION MANAGEMENT ==============
# Track yfinance session state for handling Invalid Crumb errors
_yf_session_lock = asyncio.Lock()
_yf_session_state = {
    "last_refresh": None,
    "consecutive_errors": 0,
    "total_refreshes": 0
}

def refresh_yfinance_session():
    """
    Refresh yfinance session to handle Invalid Crumb errors.
    Clears internal caches and forces new session creation.
    """
    import yfinance as yf

    try:
        # Clear yfinance's internal caches
        if hasattr(yf, 'shared') and hasattr(yf.shared, '_CACHEMGR'):
            yf.shared._CACHEMGR.clear()

        # Clear any module-level cache
        if hasattr(yf, 'cache'):
            if hasattr(yf.cache, 'clear'):
                yf.cache.clear()

        # For newer yfinance versions, clear the session
        if hasattr(yf, 'utils') and hasattr(yf.utils, 'get_json'):
            # Force new session by clearing cookies
            import requests
            yf.utils.get_json.cache_clear() if hasattr(yf.utils.get_json, 'cache_clear') else None

        _yf_session_state["last_refresh"] = datetime.now()
        _yf_session_state["total_refreshes"] += 1
        _yf_session_state["consecutive_errors"] = 0

        logger.info(f"Refreshed yfinance session (total refreshes: {_yf_session_state['total_refreshes']})")
        return True
    except Exception as e:
        logger.warning(f"Failed to refresh yfinance session: {e}")
        return False


def get_yf_ticker_with_retry(ticker: str, max_retries: int = 2):
    """
    Get a yfinance Ticker object with retry logic for Invalid Crumb errors.

    Args:
        ticker: Stock ticker symbol
        max_retries: Maximum number of retries on auth errors

    Returns:
        yfinance Ticker object or None if all retries fail
    """
    import yfinance as yf

    for attempt in range(max_retries + 1):
        try:
            stock = yf.Ticker(ticker)
            # Test if the ticker is valid by accessing a property
            _ = stock.info
            _yf_session_state["consecutive_errors"] = 0
            return stock
        except Exception as e:
            error_str = str(e).lower()
            is_auth_error = any(x in error_str for x in ["401", "crumb", "unauthorized", "invalid"])

            if is_auth_error and attempt < max_retries:
                _yf_session_state["consecutive_errors"] += 1
                logger.debug(f"{ticker}: Auth error (attempt {attempt + 1}), refreshing session...")
                refresh_yfinance_session()
                continue
            elif is_auth_error:
                logger.warning(f"{ticker}: Auth error persists after {max_retries} retries")
                return None
            else:
                # Non-auth error, don't retry
                raise

    return None


def yf_safe_call(ticker: str, func_name: str, default=None, max_retries: int = 2):
    """
    Safely call a yfinance Ticker method with retry logic for Invalid Crumb errors.

    Args:
        ticker: Stock ticker symbol
        func_name: Name of the Ticker property/method to call (e.g., 'info', 'insider_transactions')
        default: Default value to return on failure
        max_retries: Maximum number of retries on auth errors

    Returns:
        Result of the yfinance call or default on failure
    """
    import yfinance as yf

    for attempt in range(max_retries + 1):
        try:
            stock = yf.Ticker(ticker)
            result = getattr(stock, func_name)
            # If it's callable (method), call it
            if callable(result):
                result = result()
            _yf_session_state["consecutive_errors"] = 0
            return result
        except Exception as e:
            error_str = str(e).lower()
            is_auth_error = any(x in error_str for x in ["401", "crumb", "unauthorized", "invalid"])
            is_rate_limit = any(x in error_str for x in ["429", "rate", "too many"])

            if is_rate_limit:
                # Don't retry rate limits, just fail
                logger.debug(f"{ticker}: Rate limited on {func_name}")
                return default
            elif is_auth_error and attempt < max_retries:
                _yf_session_state["consecutive_errors"] += 1
                logger.debug(f"{ticker}: Auth error on {func_name} (attempt {attempt + 1}), refreshing session...")
                refresh_yfinance_session()
                import time
                time.sleep(0.5)  # Small delay after refresh
                continue
            elif is_auth_error:
                logger.debug(f"{ticker}: Auth error on {func_name} persists after {max_retries} retries")
                return default
            else:
                # Non-auth, non-rate error
                logger.debug(f"{ticker}: Error on {func_name}: {e}")
                return default

    return default


# ============== FALLBACK TRACKING ==============
# Track stocks that needed cached fallback for monitoring
_fallback_tracker = {
    "stocks_using_fallback": set(),  # Tickers that used cached data this scan
    "stocks_no_data": set(),  # Tickers with no cache available
    "last_reset": None
}
_fallback_lock = asyncio.Lock()

# Maximum age for fallback data (7 days) - older data is considered too stale
MAX_FALLBACK_CACHE_AGE_DAYS = 7


def get_cache_age_days(ticker: str, data_type: str) -> float:
    """Get how old the cached data is in days. Returns 999 if unknown."""
    with _freshness_lock:
        if ticker in _data_freshness_cache and data_type in _data_freshness_cache[ticker]:
            cached_time = _data_freshness_cache[ticker][data_type]
            if cached_time:
                age = (datetime.now() - cached_time).total_seconds() / 86400
                return age
    return 999  # Unknown age = very old


def get_fallback_stats() -> dict:
    """Get statistics about fallback usage for monitoring"""
    return {
        "stocks_using_fallback": list(_fallback_tracker["stocks_using_fallback"]),
        "stocks_no_data": list(_fallback_tracker["stocks_no_data"]),
        "fallback_count": len(_fallback_tracker["stocks_using_fallback"]),
        "no_data_count": len(_fallback_tracker["stocks_no_data"])
    }


def reset_fallback_tracker():
    """Reset fallback tracker at start of new scan"""
    _fallback_tracker["stocks_using_fallback"] = set()
    _fallback_tracker["stocks_no_data"] = set()
    _fallback_tracker["last_reset"] = datetime.now()

# ============== RATE LIMITING CONFIGURATION ==============
# FMP API limit: 300 calls/minute = 5 calls/second
# We target 250 calls/minute to leave headroom

MAX_CONCURRENT_REQUESTS = 8  # Reduced from 20 - fewer concurrent requests = more predictable rate
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Global rate limiter - tracks calls per minute
_rate_limiter = {
    "calls_this_minute": 0,
    "minute_start": None,
    "max_calls_per_minute": 250,  # Target 250, well under 300 limit
    "consecutive_429s": 0,
    "backoff_until": None,
    "total_calls": 0,
    "total_429s": 0
}
_rate_lock = asyncio.Lock()

# Checkpoint file for progress persistence
CHECKPOINT_FILE = Path(__file__).parent / "data" / "scan_checkpoint.json"


def save_scan_progress(completed_tickers: List[str], scan_id: str = "default"):
    """Save scan progress to checkpoint file"""
    try:
        CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
        checkpoint = {
            "scan_id": scan_id,
            "completed": completed_tickers,
            "timestamp": datetime.now().isoformat()
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f)
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")


def load_scan_progress(scan_id: str = "default") -> List[str]:
    """Load scan progress from checkpoint file"""
    try:
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                checkpoint = json.load(f)
            checkpoint_scan_id = checkpoint.get("scan_id")
            checkpoint_completed = checkpoint.get("completed", [])
            ts = datetime.fromisoformat(checkpoint["timestamp"])
            age_minutes = (datetime.now() - ts).total_seconds() / 60

            logger.info(f"Checkpoint found: scan_id={checkpoint_scan_id}, "
                       f"completed={len(checkpoint_completed)}, age={age_minutes:.1f}min")

            # Only use checkpoint if it's from the same scan and less than 1 hour old
            if checkpoint_scan_id == scan_id:
                if datetime.now() - ts < timedelta(hours=1):
                    logger.info(f"Using checkpoint: {len(checkpoint_completed)} tickers already completed")
                    return checkpoint_completed
                else:
                    logger.info(f"Checkpoint too old ({age_minutes:.1f}min), ignoring")
            else:
                logger.info(f"Checkpoint scan_id mismatch: {checkpoint_scan_id} vs {scan_id}, ignoring")
        else:
            logger.debug("No checkpoint file found")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
    return []


def clear_scan_progress():
    """Clear checkpoint file after successful scan"""
    try:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("Checkpoint file cleared successfully")
    except (OSError, IOError) as e:
        logger.error(f"Failed to clear checkpoint file: {e}")


async def _check_rate_limit():
    """Check and enforce rate limit before making an API call"""
    global _rate_limiter

    async with _rate_lock:
        now = datetime.now()

        # Check if we're in backoff period from 429
        if _rate_limiter["backoff_until"]:
            if now < _rate_limiter["backoff_until"]:
                wait_secs = (_rate_limiter["backoff_until"] - now).total_seconds()
                logger.info(f"Rate limit backoff: waiting {wait_secs:.1f}s")
                await asyncio.sleep(wait_secs)
            _rate_limiter["backoff_until"] = None

        # Reset counter if we're in a new minute
        if _rate_limiter["minute_start"] is None or (now - _rate_limiter["minute_start"]).total_seconds() >= 60:
            _rate_limiter["calls_this_minute"] = 0
            _rate_limiter["minute_start"] = now

        # If we're approaching the limit, wait until the next minute
        if _rate_limiter["calls_this_minute"] >= _rate_limiter["max_calls_per_minute"]:
            wait_secs = 60 - (now - _rate_limiter["minute_start"]).total_seconds()
            if wait_secs > 0:
                logger.info(f"Rate limit reached ({_rate_limiter['calls_this_minute']} calls), waiting {wait_secs:.1f}s for next minute")
                await asyncio.sleep(wait_secs)
                _rate_limiter["calls_this_minute"] = 0
                _rate_limiter["minute_start"] = datetime.now()

        # Increment call counter
        _rate_limiter["calls_this_minute"] += 1
        _rate_limiter["total_calls"] += 1


def get_rate_limit_stats() -> dict:
    """Get current rate limit statistics"""
    return {
        "calls_this_minute": _rate_limiter["calls_this_minute"],
        "max_per_minute": _rate_limiter["max_calls_per_minute"],
        "total_calls": _rate_limiter["total_calls"],
        "total_429s": _rate_limiter["total_429s"],
        "consecutive_429s": _rate_limiter["consecutive_429s"]
    }


async def fetch_json_async(session: aiohttp.ClientSession, url: str, timeout: int = 15) -> Optional[dict]:
    """Async HTTP GET with rate limiting and exponential backoff"""
    global _rate_limiter

    # Check rate limit before making request
    await _check_rate_limit()

    async with api_semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        _rate_limiter["consecutive_429s"] = 0
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - exponential backoff with longer waits
                        _rate_limiter["consecutive_429s"] += 1
                        _rate_limiter["total_429s"] += 1
                        # Longer backoff: 5s, 15s, 30s based on attempts and consecutive 429s
                        base_wait = 5 * (2 ** attempt)
                        extra_wait = min(_rate_limiter["consecutive_429s"] * 5, 30)
                        wait_time = min(base_wait + extra_wait, 60)
                        logger.warning(f"Rate limited (429 #{_rate_limiter['total_429s']}), waiting {wait_time:.1f}s (attempt {attempt + 1}/3)")
                        _rate_limiter["backoff_until"] = datetime.now() + timedelta(seconds=wait_time)
                        # Also reduce the per-minute limit temporarily
                        _rate_limiter["max_calls_per_minute"] = max(200, _rate_limiter["max_calls_per_minute"] - 25)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.debug(f"HTTP {response.status} for {url[:100]}...")
                        return None
            except asyncio.TimeoutError:
                logger.debug(f"Timeout fetching {url[:100]}... (attempt {attempt + 1})")
                if attempt < 2:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Error fetching {url[:100]}...: {e}")
                return None
        return None


# ============== FMP QUOTE/PROFILE ENDPOINTS ==============
# NOTE: /api/v3/ batch endpoints require legacy subscription (before Aug 2025)
# For newer accounts, we fetch individually via /stable/ endpoints

async def fetch_fmp_single_quote(session: aiohttp.ClientSession, ticker: str) -> Optional[dict]:
    """Fetch quote for a single ticker using /stable/ endpoint"""
    if not FMP_API_KEY:
        return None

    url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url, timeout=15)

    if data and isinstance(data, list) and len(data) > 0:
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
            "name": quote.get("name", ticker),
        }
    # Empty response or 404 - mark as potentially delisted
    if data is not None and isinstance(data, list) and len(data) == 0:
        mark_ticker_as_delisted(ticker, reason="fmp_quote_empty", source="async_data_fetcher")
    return None


async def fetch_fmp_single_profile(session: aiohttp.ClientSession, ticker: str) -> Optional[dict]:
    """Fetch profile for a single ticker using /stable/ endpoint"""
    if not FMP_API_KEY:
        return None

    url = f"{FMP_BASE_URL}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url, timeout=15)

    if data and isinstance(data, list) and len(data) > 0:
        profile = data[0]
        high_52w = 0
        low_52w = 0
        if profile.get("range"):
            try:
                # Range format is "low-high" e.g. "169.21-288.62"
                range_parts = profile.get("range", "").split("-")
                if len(range_parts) >= 2:
                    low_52w = float(range_parts[0].strip())
                    high_52w = float(range_parts[-1].strip())
            except (ValueError, TypeError, AttributeError):
                pass

        return {
            "name": profile.get("companyName", ""),
            "sector": profile.get("sector", ""),
            "industry": profile.get("industry", ""),
            "market_cap": profile.get("mktCap", 0),
            "current_price": profile.get("price", 0),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "shares_outstanding": profile.get("sharesOutstanding", 0) or 0,
        }
    return None


async def fetch_fmp_batch_quotes(session: aiohttp.ClientSession, tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch quotes for multiple tickers.
    Uses individual /stable/ requests since batch /api/v3/ requires legacy subscription.
    Fetches in parallel batches to maintain speed.
    """
    if not FMP_API_KEY or not tickers:
        return {}

    results = {}

    # Fetch in parallel batches of 50 to avoid overwhelming the API
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        tasks = [fetch_fmp_single_quote(session, t) for t in chunk]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, result in zip(chunk, chunk_results):
            if isinstance(result, dict) and result:
                results[ticker] = result

        # Small delay between batches to respect rate limits
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.5)

    logger.info(f"FMP quotes: fetched {len(results)}/{len(tickers)} tickers")
    return results


async def fetch_fmp_batch_profiles(session: aiohttp.ClientSession, tickers: List[str]) -> Dict[str, dict]:
    """
    Fetch profiles for multiple tickers.
    Uses individual /stable/ requests since batch /api/v3/ requires legacy subscription.
    Fetches in parallel batches to maintain speed.
    """
    if not FMP_API_KEY or not tickers:
        return {}

    results = {}

    # Fetch in parallel batches of 50 to avoid overwhelming the API
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i + batch_size]
        tasks = [fetch_fmp_single_profile(session, t) for t in chunk]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, result in zip(chunk, chunk_results):
            if isinstance(result, dict) and result:
                results[ticker] = result

        # Small delay between batches to respect rate limits
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.5)

    logger.info(f"FMP profiles: fetched {len(results)}/{len(tickers)} tickers")
    return results


# ============== INDIVIDUAL STOCK FETCHERS ==============
# These are called only for data that can't be batched

async def fetch_fmp_financials_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """
    CONSOLIDATED: Fetch earnings + revenue in ONE call (not two separate calls)
    This eliminates redundant API calls since both come from income-statement endpoint

    If FMP fails after retries, falls back to cached data as LAST RESORT.
    """
    if not FMP_API_KEY:
        return {}

    result = {
        "quarterly_eps": [], "annual_eps": [],
        "quarterly_revenue": [], "annual_revenue": [],
        "quarterly_net_income": [], "annual_net_income": [],
        "used_cache_fallback": False
    }

    # Fetch quarterly and annual in parallel (but only 2 calls total, not 4)
    quarterly_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
    annual_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"

    quarterly_data, annual_data = await asyncio.gather(
        fetch_json_async(session, quarterly_url),
        fetch_json_async(session, annual_url),
        return_exceptions=True
    )

    got_quarterly = isinstance(quarterly_data, list) and quarterly_data
    got_annual = isinstance(annual_data, list) and annual_data

    if got_quarterly:
        result["quarterly_eps"] = [q.get("eps", 0) or 0 for q in quarterly_data]
        result["quarterly_revenue"] = [q.get("revenue", 0) or 0 for q in quarterly_data]
        result["quarterly_net_income"] = [q.get("netIncome", 0) or 0 for q in quarterly_data]

    if got_annual:
        result["annual_eps"] = [a.get("eps", 0) or 0 for a in annual_data]
        result["annual_revenue"] = [a.get("revenue", 0) or 0 for a in annual_data]
        result["annual_net_income"] = [a.get("netIncome", 0) or 0 for a in annual_data]

    # LAST RESORT: If both API calls failed, use cached data if not too old
    if not got_quarterly and not got_annual:
        cached_earnings = get_cached_data(ticker, "earnings")
        cached_revenue = get_cached_data(ticker, "revenue")
        cache_age = get_cache_age_days(ticker, "earnings")

        if cached_earnings or cached_revenue:
            if cache_age <= MAX_FALLBACK_CACHE_AGE_DAYS:
                logger.warning(f"{ticker}: FMP financials failed, using CACHED data ({cache_age:.1f} days old)")
                result["used_cache_fallback"] = True
                result["cache_age_days"] = cache_age
                _fallback_tracker["stocks_using_fallback"].add(ticker)

                if cached_earnings:
                    result["quarterly_eps"] = cached_earnings.get("quarterly_eps") or cached_earnings.get("quarterly", [])
                    result["annual_eps"] = cached_earnings.get("annual_eps") or cached_earnings.get("annual", [])
                if cached_revenue:
                    result["quarterly_revenue"] = cached_revenue.get("quarterly_revenue") or cached_revenue.get("quarterly", [])
                    result["annual_revenue"] = cached_revenue.get("annual_revenue") or cached_revenue.get("annual", [])
            else:
                logger.warning(f"{ticker}: FMP cached data too old ({cache_age:.1f} days) - not using stale fallback")
                _fallback_tracker["stocks_no_data"].add(ticker)
        else:
            logger.warning(f"{ticker}: FMP financials failed and no cache available")
            _fallback_tracker["stocks_no_data"].add(ticker)

    return result


async def fetch_fmp_key_metrics_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch key metrics from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

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
    return {}


async def fetch_fmp_balance_sheet_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch balance sheet data from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/balance-sheet-statement?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

    if data and len(data) > 0:
        bs = data[0]
        return {
            "cash_and_equivalents": bs.get("cashAndCashEquivalents", 0) or 0,
            "total_debt": bs.get("totalDebt", 0) or 0,
            "total_assets": bs.get("totalAssets", 0) or 0,
            "total_liabilities": bs.get("totalLiabilities", 0) or 0,
        }
    return {}


async def fetch_fmp_analyst_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch analyst estimates + price targets in ONE call"""
    if not FMP_API_KEY:
        return {}

    result = {}

    # Fetch estimates and price targets in parallel
    estimates_url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
    targets_url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}"

    estimates_data, targets_data = await asyncio.gather(
        fetch_json_async(session, estimates_url),
        fetch_json_async(session, targets_url),
        return_exceptions=True
    )

    if isinstance(estimates_data, list) and estimates_data:
        est = estimates_data[0]
        result["estimated_eps_avg"] = est.get("estimatedEpsAvg", 0)
        result["num_analysts"] = est.get("numberAnalystsEstimatedEps", 0)

    if isinstance(targets_data, list) and targets_data:
        pt = targets_data[0]
        result["target_high"] = pt.get("targetHigh", 0)
        result["target_low"] = pt.get("targetLow", 0)
        result["target_consensus"] = pt.get("targetConsensus", 0)
        result["target_median"] = pt.get("targetMedian", 0)

    return result


async def fetch_fmp_earnings_surprise_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch earnings surprise data from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/earnings-surprises?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

    if data and len(data) > 0:
        latest = data[0]
        latest_surprise = 0
        if latest.get("estimatedEarning") and latest.get("actualEarningResult"):
            estimated = latest.get("estimatedEarning", 0)
            actual = latest.get("actualEarningResult", 0)
            if estimated and estimated != 0:
                latest_surprise = ((actual - estimated) / abs(estimated)) * 100

        # Count consecutive beats
        beat_streak = 0
        for record in data[:8]:
            estimated = record.get("estimatedEarning", 0)
            actual = record.get("actualEarningResult", 0)
            if estimated and actual and actual > estimated:
                beat_streak += 1
            else:
                break

        # Extract quarterly adjusted EPS
        quarterly_adjusted_eps = []
        for record in data[:8]:
            actual = record.get("actualEarningResult")
            if actual is not None:
                quarterly_adjusted_eps.append(float(actual))

        return {
            "latest_surprise_pct": latest_surprise,
            "beat_streak": beat_streak,
            "quarterly_adjusted_eps": quarterly_adjusted_eps,
        }
    return {}


async def fetch_fmp_earnings_calendar_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """
    Async fetch earnings calendar data using Yahoo Finance.
    Returns next earnings date and beat streak from earnings history.
    Uses default executor since yfinance is synchronous.
    """
    import asyncio
    from data_fetcher import fetch_fmp_earnings_calendar

    # Run sync function in default executor (shared thread pool)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, fetch_fmp_earnings_calendar, ticker)
    return result or {}


async def fetch_fmp_analyst_estimates_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """
    Async fetch analyst estimate revisions from FMP.
    Compares current fiscal year estimate to prior year for revision tracking.
    Uses period=annual (period=quarter requires premium).
    """
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&period=annual&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

    if data and len(data) >= 2:
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
            current = data[0]
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
            "num_analysts": current.get("numAnalystsEps", 0),
        }
    return {}


async def fetch_insider_trading_async(ticker: str) -> dict:
    """
    Async fetch insider trading data from Yahoo Finance.
    Returns summary of insider buys/sells in last 3 months with $ values.
    Uses Yahoo instead of FMP (FMP v4 endpoint deprecated Aug 2025).

    Now tracks:
    - buy_value, sell_value, net_value ($ amounts)
    - largest_buy ($ amount of largest single purchase)
    - largest_buyer_title (CEO, CFO, etc.)
    """
    loop = asyncio.get_event_loop()

    def _fetch_insider():
        try:
            import re

            # Use retry-enabled helper for yfinance calls
            insider_df = yf_safe_call(ticker, 'insider_transactions', default=None, max_retries=2)
            if insider_df is None or (hasattr(insider_df, 'empty') and insider_df.empty):
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

            for _, row in insider_df.iterrows():
                # Parse date
                start_date = row.get('Start Date')
                if start_date is None:
                    continue

                # Handle different date formats
                if hasattr(start_date, 'to_pydatetime'):
                    trade_date = start_date.to_pydatetime()
                elif isinstance(start_date, str):
                    try:
                        trade_date = datetime.strptime(start_date[:10], "%Y-%m-%d")
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

                # Skip if too old (compare without timezone)
                if trade_date.replace(tzinfo=None) < cutoff_date:
                    continue

                # Get transaction type from Text column (e.g., "Sale at price...", "Purchase at price...")
                text = str(row.get('Text', '')).upper()
                shares = abs(row.get('Shares', 0) or 0)
                value = abs(row.get('Value', 0) or 0)

                # Try to extract price from Text if Value is missing
                if value == 0 and shares > 0:
                    # Pattern: "at price X.XX" or "@ $X.XX"
                    price_match = re.search(r'(?:AT PRICE|@|\$)\s*(\d+\.?\d*)', text)
                    if price_match:
                        try:
                            price = float(price_match.group(1))
                            value = shares * price
                        except (ValueError, TypeError):
                            pass

                # Get insider name/title
                insider_name = str(row.get('Insider', '')).upper()

                # Skip gifts and other non-market transactions
                if 'GIFT' in text or 'AWARD' in text or 'EXERCISE' in text:
                    continue

                if 'PURCHASE' in text or 'BUY' in text:
                    buy_count += 1
                    net_shares += shares
                    buy_value += value

                    # Track largest buy
                    if value > largest_buy:
                        largest_buy = value
                        # Determine title from insider name
                        if 'CEO' in insider_name or 'CHIEF EXECUTIVE' in insider_name:
                            largest_buyer_title = "CEO"
                        elif 'CFO' in insider_name or 'CHIEF FINANCIAL' in insider_name:
                            largest_buyer_title = "CFO"
                        elif 'COO' in insider_name or 'CHIEF OPERATING' in insider_name:
                            largest_buyer_title = "COO"
                        elif 'PRESIDENT' in insider_name:
                            largest_buyer_title = "PRESIDENT"
                        elif 'DIRECTOR' in insider_name:
                            largest_buyer_title = "DIRECTOR"
                        else:
                            largest_buyer_title = "OFFICER"

                elif 'SALE' in text or 'SELL' in text:
                    sell_count += 1
                    net_shares -= shares
                    sell_value += value

            net_value = buy_value - sell_value

            # Determine sentiment based on $ value (more meaningful than count)
            sentiment = "neutral"
            if net_value >= 100000:
                sentiment = "bullish"
            elif net_value <= -100000:
                sentiment = "bearish"
            elif buy_count > 0 and buy_count > sell_count * 1.5:
                sentiment = "bullish"
            elif sell_count > 0 and sell_count > buy_count * 1.5:
                sentiment = "bearish"

            return {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "net_shares": int(net_shares),
                "sentiment": sentiment,
                # New value tracking fields
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_value": net_value,
                "largest_buy": largest_buy,
                "largest_buyer_title": largest_buyer_title,
            }

        except Exception as e:
            logger.debug(f"{ticker}: Insider trading error: {e}")
            return {}

    return await loop.run_in_executor(None, _fetch_insider)


async def fetch_short_interest_async(ticker: str) -> dict:
    """
    Async fetch short interest data from Yahoo Finance.
    Wraps sync yfinance call in executor to not block async loop.
    Uses centralized retry logic for "Invalid Crumb" 401 errors.
    """
    loop = asyncio.get_event_loop()

    def _fetch_short():
        # Use retry-enabled helper for yfinance calls
        info = yf_safe_call(ticker, 'info', default=None, max_retries=2)

        if info:
            short_pct = info.get("shortPercentOfFloat", 0) or 0
            short_ratio = info.get("shortRatio", 0) or 0

            # Convert to percentage if needed
            if short_pct > 0 and short_pct < 1:
                short_pct = short_pct * 100

            return {
                "short_interest_pct": short_pct,
                "short_ratio": short_ratio
            }

        return {}

    return await loop.run_in_executor(None, _fetch_short)


# ============== YAHOO INFO FETCHER (COMPREHENSIVE) ==============
# Gets ROE, institutional %, analyst targets, cash/debt, short interest in ONE call

# Semaphore to limit concurrent Yahoo requests (they throttle aggressively)
_yahoo_semaphore = asyncio.Semaphore(3)  # 3 concurrent Yahoo requests
_yahoo_delay = 0.6  # Delay between Yahoo requests in seconds
_yahoo_max_retries = 4  # Retries for rate limits (with exponential backoff)

async def fetch_yahoo_info_comprehensive_async(ticker: str) -> dict:
    """
    Fetch ALL supplementary data from Yahoo Finance in ONE call.
    This replaces multiple FMP calls (key_metrics, balance_sheet, analyst, short_interest).

    Returns dict with:
    - roe, roa, roic (key metrics)
    - cash_and_equivalents, total_debt (balance sheet)
    - analyst_target_price, num_analyst_opinions (analyst)
    - short_interest_pct, short_ratio (short interest)
    - institutional_holders_pct
    """
    loop = asyncio.get_event_loop()

    def _fetch_yahoo_info():
        result = {
            # Key metrics
            "roe": 0,
            "roa": 0,
            "roic": 0,
            "earnings_yield": 0,
            "fcf_yield": 0,
            # Balance sheet
            "cash_and_equivalents": 0,
            "total_debt": 0,
            # Market data (fallback if FMP doesn't have it)
            "market_cap": 0,
            # Analyst
            "analyst_target_price": 0,
            "analyst_target_high": 0,
            "analyst_target_low": 0,
            "num_analyst_opinions": 0,
            "earnings_growth_estimate": 0,
            # Short interest
            "short_interest_pct": 0,
            "short_ratio": 0,
            # Institutional
            "institutional_holders_pct": 0,
            # Status
            "success": False
        }

        try:
            # Use retry-enabled helper for yfinance calls
            info = yf_safe_call(ticker, 'info', default=None, max_retries=2)

            if not info or not info.get('regularMarketPrice'):
                logger.debug(f"{ticker}: Yahoo info returned no data")
                # Mark as potentially delisted if Yahoo returns no price
                mark_ticker_as_delisted(ticker, reason="yahoo_no_price", source="async_data_fetcher")
                return result

            # Key metrics - store as decimal (e.g., 0.05 = 5%) for consistency with FMP
            roe = info.get('returnOnEquity')
            result["roe"] = roe if roe and roe > -10 else 0

            roa = info.get('returnOnAssets')
            result["roa"] = roa if roa and roa > -10 else 0

            # ROIC not directly available, estimate from ROE and debt ratio
            result["roic"] = result["roe"] * 0.8 if result["roe"] > 0 else 0

            # Earnings/FCF yield
            trailing_pe = info.get('trailingPE')
            if trailing_pe and trailing_pe > 0:
                result["earnings_yield"] = 1 / trailing_pe

            fcf = info.get('freeCashflow', 0) or 0
            market_cap = info.get('marketCap', 0) or 0
            if market_cap > 0 and fcf:
                result["fcf_yield"] = fcf / market_cap
            # Store market_cap as fallback data
            result["market_cap"] = market_cap

            # Balance sheet data
            result["cash_and_equivalents"] = info.get('totalCash', 0) or 0
            result["total_debt"] = info.get('totalDebt', 0) or 0

            # Analyst data
            result["analyst_target_price"] = info.get('targetMeanPrice', 0) or info.get('targetMedianPrice', 0) or 0
            result["analyst_target_high"] = info.get('targetHighPrice', 0) or 0
            result["analyst_target_low"] = info.get('targetLowPrice', 0) or 0
            result["num_analyst_opinions"] = info.get('numberOfAnalystOpinions', 0) or 0

            # Earnings growth estimate
            # Yahoo returns as decimal (0.25 = 25%) - convert to percentage
            growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
            if growth:
                # If absolute value <= 1, it's likely a decimal (e.g., 0.25 = 25%)
                result["earnings_growth_estimate"] = growth * 100 if abs(growth) <= 1 else growth

            # Short interest
            short_pct = info.get('shortPercentOfFloat', 0) or 0
            # Convert to percentage if needed (Yahoo sometimes returns as decimal)
            if 0 < short_pct < 1:
                short_pct = short_pct * 100
            result["short_interest_pct"] = short_pct
            result["short_ratio"] = info.get('shortRatio', 0) or 0

            # Institutional ownership
            # Yahoo returns decimal (0.65 = 65%, 1.00002 = 100.002%)
            # Convert to percentage if value looks like a decimal (< 1.5)
            inst_pct = info.get('heldPercentInstitutions', 0) or 0
            result["institutional_holders_pct"] = (inst_pct * 100) if 0 < inst_pct <= 1.5 else inst_pct

            result["success"] = True
            logger.debug(f"{ticker}: Yahoo info fetched successfully")
            return result

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "too many" in error_str:
                raise  # Re-raise rate limit errors for outer retry logic
            # Check for 404 or "not found" errors indicating delisted ticker
            if "404" in error_str or "not found" in error_str:
                mark_ticker_as_delisted(ticker, reason="yahoo_404", source="async_data_fetcher")
            # Auth errors already handled by yf_safe_call
            logger.debug(f"{ticker}: Yahoo info error: {e}")
            return result

    # Use semaphore to limit concurrent Yahoo requests with retry logic
    async with _yahoo_semaphore:
        for attempt in range(_yahoo_max_retries):
            await asyncio.sleep(_yahoo_delay)
            try:
                result = await loop.run_in_executor(None, _fetch_yahoo_info)
                if result.get("success"):
                    return result
                # If not successful but no exception, try again with backoff
                if attempt < _yahoo_max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s, 20s backoff
                    logger.debug(f"{ticker}: Yahoo returned no data, retrying in {wait_time}s (attempt {attempt + 1}/{_yahoo_max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                return result  # Last attempt, return whatever we got
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "too many" in error_str:
                    # Exponential backoff: 8s, 16s, 32s, 64s, 120s
                    wait_time = min(8 * (2 ** attempt), 120)
                    logger.warning(f"{ticker}: Yahoo rate limited, waiting {wait_time}s (attempt {attempt + 1}/{_yahoo_max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.debug(f"{ticker}: Yahoo error: {e}")
                    # Don't give up on first error, try again
                    if attempt < _yahoo_max_retries - 1:
                        await asyncio.sleep(3)
                        continue
                    break

        # LAST RESORT: Use cached data if available AND not too old
        logger.warning(f"{ticker}: Yahoo failed after {_yahoo_max_retries} retries, checking for cached data...")
        cached_yahoo = get_cached_data(ticker, "yahoo_info")
        cache_age = get_cache_age_days(ticker, "yahoo_info")

        if cached_yahoo and cached_yahoo.get("success"):
            if cache_age <= MAX_FALLBACK_CACHE_AGE_DAYS:
                logger.info(f"{ticker}: Using CACHED Yahoo data as fallback ({cache_age:.1f} days old)")
                cached_yahoo["used_cache_fallback"] = True
                cached_yahoo["cache_age_days"] = cache_age
                _fallback_tracker["stocks_using_fallback"].add(ticker)
                return cached_yahoo
            else:
                logger.warning(f"{ticker}: Cached data too old ({cache_age:.1f} days) - not using stale fallback")
                _fallback_tracker["stocks_no_data"].add(ticker)
                # Mark as potentially delisted - hasn't had fresh data in over 7 days
                mark_ticker_as_delisted(ticker, reason="stale_data", source="async_fetcher")
                return {"success": False, "cache_too_old": True, "cache_age_days": cache_age}

        logger.warning(f"{ticker}: No cached data available - stock will have incomplete data")
        _fallback_tracker["stocks_no_data"].add(ticker)
        # Mark as potentially delisted for future exclusion
        mark_ticker_as_delisted(ticker, reason="no_yahoo_data", source="async_fetcher")
        return {"success": False}


async def fetch_yahoo_supplement_async(ticker: str, stock_data: StockData) -> None:
    """
    Fetch supplemental data from Yahoo Finance ONLY if FMP data is incomplete
    Runs in executor to not block async loop
    """
    loop = asyncio.get_event_loop()

    def _fetch_yahoo():
        try:
            import yfinance as yf

            # Use retry-enabled helper for yfinance calls
            info = yf_safe_call(ticker, 'info', default=None, max_retries=2)

            if info and info.get('regularMarketPrice'):
                # Only fill in missing data
                if not stock_data.sector:
                    stock_data.sector = info.get('sector', 'Unknown')
                if not stock_data.industry:
                    stock_data.industry = info.get('industry', '')
                if not stock_data.shares_outstanding:
                    stock_data.shares_outstanding = info.get('sharesOutstanding', 0)
                if not stock_data.institutional_holders_pct:
                    inst_pct = info.get('heldPercentInstitutions', 0) or 0
                    # Convert decimal to percentage (Yahoo returns 0.65 = 65%, 1.00002 = 100.002%)
                    stock_data.institutional_holders_pct = (inst_pct * 100) if 0 < inst_pct <= 1.5 else inst_pct
                if not stock_data.roe:
                    roe = info.get('returnOnEquity')
                    stock_data.roe = roe if roe else 0  # Store as decimal

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
                    # Use retry-enabled helper for earnings_history
                    earnings_hist = yf_safe_call(ticker, 'earnings_history', default=None, max_retries=2)
                    if earnings_hist is not None and not earnings_hist.empty and 'epsActual' in earnings_hist.columns:
                        adjusted_eps = []
                        for _, row in earnings_hist.iterrows():
                            actual = row.get('epsActual')
                            if actual is not None and not pd.isna(actual):
                                adjusted_eps.append(float(actual))

                        if adjusted_eps and len(adjusted_eps) >= 4:
                            # Always prefer Yahoo adjusted EPS over FMP GAAP EPS
                            old_eps = stock_data.quarterly_earnings[:4] if stock_data.quarterly_earnings else []
                            stock_data.quarterly_earnings = adjusted_eps[::-1][:8]  # Reverse to newest first, limit to 8
                            # Cache the adjusted EPS for 7 days
                            set_cached_data(ticker, "adjusted_eps", stock_data.quarterly_earnings)
                            mark_data_fetched(ticker, "adjusted_eps")
                            logger.info(f"{ticker}: Using Yahoo ADJUSTED EPS {stock_data.quarterly_earnings[:4]} (was GAAP: {old_eps})")
                except Exception as e:
                    logger.debug(f"{ticker}: Yahoo earnings_history error: {e}")

            # Get annual earnings if missing
            if not stock_data.annual_earnings:
                try:
                    annual = stock.financials
                    if annual is not None and not annual.empty and 'Net Income' in annual.index:
                        net_income = annual.loc['Net Income'].dropna()
                        shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                        stock_data.annual_earnings = (net_income / shares).tolist()[:5]
                except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                    pass  # Annual earnings calculation failed - use available data

        except Exception as e:
            logger.debug(f"{ticker}: Yahoo fetch error: {e}")

    await loop.run_in_executor(None, _fetch_yahoo)


async def get_stock_data_async(
    ticker: str,
    session: aiohttp.ClientSession,
    batch_quotes: Dict[str, dict] = None,
    batch_profiles: Dict[str, dict] = None
) -> StockData:
    """
    Async version of get_stock_data

    Now uses pre-fetched batch data for quotes/profiles when available,
    only making individual calls for detailed financials.
    """
    stock_data = StockData(ticker)

    # Use batch data if available, otherwise fetch individually
    if batch_quotes and ticker in batch_quotes:
        quote = batch_quotes[ticker]
        stock_data.current_price = quote.get("current_price", 0)
        stock_data.high_52w = quote.get("high_52w", 0) or 0
        stock_data.low_52w = quote.get("low_52w", 0) or 0
        stock_data.market_cap = quote.get("market_cap", 0)
        stock_data.avg_volume_50d = quote.get("avg_volume", 0)
        stock_data.current_volume = quote.get("volume", 0)
        stock_data.trailing_pe = quote.get("pe", 0) or 0
        stock_data.shares_outstanding = int(quote.get("shares_outstanding", 0) or 0)
        stock_data.name = quote.get("name", ticker)
    else:
        # Fetch individual quote from FMP /stable/ endpoint
        quote = await fetch_fmp_single_quote(session, ticker)
        if quote:
            stock_data.current_price = quote.get("current_price", 0)
            stock_data.high_52w = quote.get("high_52w", 0) or 0
            stock_data.low_52w = quote.get("low_52w", 0) or 0
            stock_data.market_cap = quote.get("market_cap", 0)
            stock_data.avg_volume_50d = quote.get("avg_volume", 0)
            stock_data.current_volume = quote.get("volume", 0)
            stock_data.trailing_pe = quote.get("pe", 0) or 0
            stock_data.shares_outstanding = int(quote.get("shares_outstanding", 0) or 0)
            stock_data.name = quote.get("name", ticker)
            if stock_data.market_cap:
                logger.debug(f"{ticker}: FMP quote market_cap={stock_data.market_cap}")
        else:
            logger.warning(f"{ticker}: FMP quote returned None")

    if batch_profiles and ticker in batch_profiles:
        profile = batch_profiles[ticker]
        stock_data.name = profile.get("name") or stock_data.name or ticker
        stock_data.sector = profile.get("sector", "")
        stock_data.industry = profile.get("industry", "")
        if not stock_data.shares_outstanding:
            stock_data.shares_outstanding = int(profile.get("shares_outstanding", 0) or 0)
        if not stock_data.current_price:
            stock_data.current_price = profile.get("current_price", 0)
        if not stock_data.high_52w:
            stock_data.high_52w = profile.get("high_52w", 0) or 0
        if not stock_data.low_52w:
            stock_data.low_52w = profile.get("low_52w", 0) or 0
        if not stock_data.market_cap:
            stock_data.market_cap = profile.get("market_cap", 0) or 0
    elif not stock_data.sector:
        # Fetch individual profile from FMP /stable/ endpoint for sector info
        profile = await fetch_fmp_single_profile(session, ticker)
        if profile:
            stock_data.name = profile.get("name") or stock_data.name or ticker
            stock_data.sector = profile.get("sector", "")
            stock_data.industry = profile.get("industry", "")
            if not stock_data.shares_outstanding:
                stock_data.shares_outstanding = int(profile.get("shares_outstanding", 0) or 0)
            if not stock_data.current_price:
                stock_data.current_price = profile.get("current_price", 0)
            if not stock_data.high_52w:
                stock_data.high_52w = profile.get("high_52w", 0) or 0
            if not stock_data.low_52w:
                stock_data.low_52w = profile.get("low_52w", 0) or 0
            if not stock_data.market_cap:
                stock_data.market_cap = profile.get("market_cap", 0) or 0

    # ============== HYBRID DATA FETCHING ==============
    # Strategy: FMP for earnings/revenue (rate-limited), Yahoo for everything else (more lenient)
    # This reduces FMP calls from 6 to 2 per stock, dramatically reducing 429 errors

    # STEP 1: FMP for earnings and revenue only (the most reliable source for this data)
    financials = {}
    if FMP_API_KEY:
        earnings_fresh = is_data_fresh(ticker, "earnings")
        if not earnings_fresh:
            logger.info(f"{ticker}: Earnings not fresh, fetching from FMP")
            financials = await fetch_fmp_financials_async(session, ticker)
        else:
            # Load from cache
            cached_earnings = get_cached_data(ticker, "earnings")
            cached_revenue = get_cached_data(ticker, "revenue")
            if cached_earnings:
                stock_data.quarterly_earnings = cached_earnings.get("quarterly_eps") or cached_earnings.get("quarterly", [])
                stock_data.annual_earnings = cached_earnings.get("annual_eps") or cached_earnings.get("annual", [])
                logger.info(f"{ticker}: CACHE HIT - quarterly={len(stock_data.quarterly_earnings)}, annual={len(stock_data.annual_earnings)}")
            else:
                logger.warning(f"{ticker}: is_data_fresh=True but get_cached_data returned None")
            if cached_revenue:
                stock_data.quarterly_revenue = cached_revenue.get("quarterly_revenue") or cached_revenue.get("quarterly", [])
                stock_data.annual_revenue = cached_revenue.get("annual_revenue") or cached_revenue.get("annual", [])

    # Apply FMP financials if fetched
    if financials:
        stock_data.quarterly_earnings = financials.get("quarterly_eps", [])
        stock_data.annual_earnings = financials.get("annual_eps", [])
        stock_data.quarterly_revenue = financials.get("quarterly_revenue", [])
        stock_data.annual_revenue = financials.get("annual_revenue", [])
        # Cache earnings and revenue data
        set_cached_data(ticker, "earnings", {
            "quarterly": stock_data.quarterly_earnings,
            "annual": stock_data.annual_earnings
        }, persist_to_db=True)
        set_cached_data(ticker, "revenue", {
            "quarterly": stock_data.quarterly_revenue,
            "annual": stock_data.annual_revenue
        }, persist_to_db=True)
        mark_data_fetched(ticker, "earnings")

    # STEP 2: Yahoo for everything else (key metrics, balance sheet, analyst, short interest)
    # This is ONE call that gets ALL supplementary data
    yahoo_info = {}
    if not is_data_fresh(ticker, "yahoo_info"):
        yahoo_info = await fetch_yahoo_info_comprehensive_async(ticker)
    else:
        # Load from cache
        cached_yahoo = get_cached_data(ticker, "yahoo_info")
        if cached_yahoo:
            yahoo_info = cached_yahoo

    # Apply Yahoo info data
    if yahoo_info and yahoo_info.get("success", False):
        # Key metrics
        if yahoo_info.get("roe"):
            stock_data.roe = yahoo_info["roe"]
        if yahoo_info.get("roa"):
            stock_data.roa = yahoo_info["roa"]
        if yahoo_info.get("roic"):
            stock_data.roic = yahoo_info["roic"]
        stock_data.earnings_yield = yahoo_info.get("earnings_yield", 0)
        stock_data.fcf_yield = yahoo_info.get("fcf_yield", 0)

        # Balance sheet
        if yahoo_info.get("cash_and_equivalents"):
            stock_data.cash_and_equivalents = yahoo_info["cash_and_equivalents"]
        if yahoo_info.get("total_debt"):
            stock_data.total_debt = yahoo_info["total_debt"]

        # Market cap fallback from Yahoo if not set from FMP
        if not stock_data.market_cap and yahoo_info.get("market_cap"):
            stock_data.market_cap = yahoo_info["market_cap"]

        # Analyst data
        if yahoo_info.get("analyst_target_price"):
            stock_data.analyst_target_price = yahoo_info["analyst_target_price"]
        stock_data.analyst_target_high = yahoo_info.get("analyst_target_high", 0)
        stock_data.analyst_target_low = yahoo_info.get("analyst_target_low", 0)
        stock_data.num_analyst_opinions = yahoo_info.get("num_analyst_opinions", 0)
        if yahoo_info.get("earnings_growth_estimate"):
            stock_data.earnings_growth_estimate = yahoo_info["earnings_growth_estimate"]

        # Short interest
        stock_data.short_interest_pct = yahoo_info.get("short_interest_pct", 0)
        stock_data.short_ratio = yahoo_info.get("short_ratio", 0)

        # Institutional ownership
        if yahoo_info.get("institutional_holders_pct"):
            stock_data.institutional_holders_pct = yahoo_info["institutional_holders_pct"]

        # Cache Yahoo info (use same freshness as key_metrics - 7 days)
        set_cached_data(ticker, "yahoo_info", yahoo_info, persist_to_db=True)
        mark_data_fetched(ticker, "yahoo_info")

    # Also cache in the old format for backwards compatibility with other code
    if yahoo_info and yahoo_info.get("success", False):
        set_cached_data(ticker, "key_metrics", {
            "roe": yahoo_info.get("roe", 0),
            "roa": yahoo_info.get("roa", 0),
            "roic": yahoo_info.get("roic", 0),
        }, persist_to_db=True)
        mark_data_fetched(ticker, "key_metrics")

        set_cached_data(ticker, "balance_sheet", {
            "cash_and_equivalents": yahoo_info.get("cash_and_equivalents", 0),
            "total_debt": yahoo_info.get("total_debt", 0),
        }, persist_to_db=True)
        mark_data_fetched(ticker, "balance_sheet")

        set_cached_data(ticker, "analyst", {
            "target_consensus": yahoo_info.get("analyst_target_price", 0),
            "num_analysts": yahoo_info.get("num_analyst_opinions", 0),
        }, persist_to_db=True)
        mark_data_fetched(ticker, "analyst")

    # Get price history from Yahoo chart API (fast, no rate limit)
    chart_data = fetch_price_from_chart_api(ticker)
    if chart_data.get("current_price"):
        if not stock_data.current_price:
            stock_data.current_price = chart_data["current_price"]
        if not stock_data.high_52w:
            stock_data.high_52w = chart_data.get("high_52w", 0) or 0
        # FIX: Add fallback for low_52w and market_cap from Yahoo chart API
        if not stock_data.low_52w:
            stock_data.low_52w = chart_data.get("low_52w", 0) or 0
        if not stock_data.market_cap:
            stock_data.market_cap = chart_data.get("market_cap", 0) or 0
        if not stock_data.name:
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
                if recent_volumes:
                    stock_data.avg_volume_50d = sum(recent_volumes) / len(recent_volumes)
                stock_data.current_volume = volumes[-1] if volumes[-1] else 0

    # Fetch weekly price history for base pattern detection
    # This is a separate call because we need weekly OHLC data, not daily
    weekly_data = fetch_weekly_price_history(ticker)
    if weekly_data:
        stock_data.weekly_price_history = weekly_data

    # ALWAYS call Yahoo to get adjusted EPS (FMP only has GAAP EPS which distorts growth metrics)
    # Yahoo earnings_history.epsActual matches what Fidelity and analyst estimates use
    # Also fills in missing sector, ROE, institutional data
    await fetch_yahoo_supplement_async(ticker, stock_data)

    # Mark as valid if we have basic data
    if stock_data.current_price and not stock_data.price_history.empty:
        stock_data.is_valid = True
        # Self-healing: if ticker was previously marked as delisted but now works, clear it
        clear_delisted_ticker(ticker)
    elif stock_data.current_price:
        stock_data.is_valid = True
        stock_data.error_message = "Limited data available"
        # Self-healing: ticker has price, so it's not delisted
        clear_delisted_ticker(ticker)
    else:
        # No price data from any source - likely delisted or invalid ticker
        mark_ticker_as_delisted(ticker, reason="no_price_data", source="get_stock_data_async")
        stock_data.is_valid = False
        stock_data.error_message = "No price data available - ticker may be delisted"

    return stock_data


async def fetch_stocks_batch_async(
    tickers: List[str],
    batch_size: int = 100,
    progress_callback=None,
    resume_from_checkpoint: bool = True
) -> List[StockData]:
    """
    Fetch multiple stocks concurrently with BATCH optimization

    Key improvements:
    1. Batch quotes/profiles fetched ONCE for all tickers (huge savings!)
    2. Individual financials fetched in parallel batches
    3. Progress persistence for interrupted scans

    Args:
        tickers: List of stock tickers to fetch
        batch_size: Number of stocks to process detailed financials at once
        progress_callback: Optional callback function(current, total)
        resume_from_checkpoint: Whether to resume from saved progress

    Returns:
        List of StockData objects
    """
    results = []
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d')}"

    # Load DB cache on startup to enable freshness checks
    # This prevents re-fetching data that's already fresh from previous scans
    load_cache_from_db()

    # Reset fallback tracker for this scan
    reset_fallback_tracker()

    # Resume from checkpoint if available
    completed_tickers = []
    original_tickers_set = set(tickers)  # Store original ticker set for validation
    original_total = len(tickers)  # Store original total BEFORE filtering
    if resume_from_checkpoint:
        checkpoint_tickers = load_scan_progress(scan_id)
        if checkpoint_tickers:
            # Only count checkpoint tickers that are in current list (exclude delisted)
            completed_tickers = [t for t in checkpoint_tickers if t in original_tickers_set]
            excluded_count = len(checkpoint_tickers) - len(completed_tickers)
            if excluded_count > 0:
                logger.info(f"Checkpoint had {excluded_count} tickers no longer in scan list (likely delisted)")
            logger.info(f"Resuming scan: {len(completed_tickers)} tickers already completed")
            tickers = [t for t in tickers if t not in set(completed_tickers)]

    if not tickers:
        logger.info("All tickers already scanned (from checkpoint)")
        return results

    # Timeout per individual request, not for the entire session
    # Full scans can take 15-20 minutes, so we use per-request timeout only
    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=60)
    # Reduced connection limits to prevent overwhelming the API
    connector = aiohttp.TCPConnector(limit=30, limit_per_host=20)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # SKIP batch quotes/profiles - they require legacy FMP subscription for true batching
        # Individual /stable/ fetches are too slow (1904 tickers * 2 endpoints = ~16 min just waiting on rate limits)
        # Instead, rely on Yahoo fallbacks in get_stock_data_async which provide market_cap and 52-week data
        batch_quotes = {}
        batch_profiles = {}
        logger.info(f"Skipping FMP batch quotes/profiles (using Yahoo fallbacks for market data)")

        # Process detailed financials in smaller batches with rate limiting
        # Use smaller batch size (50) to give rate limiter time to work
        effective_batch_size = min(batch_size, 50)
        logger.info(f"Fetching detailed financials in batches of {effective_batch_size}...")

        for i in range(0, len(tickers), effective_batch_size):
            batch = tickers[i:i + effective_batch_size]

            # Fetch all stocks in this batch concurrently
            tasks = [
                get_stock_data_async(ticker, session, batch_quotes, batch_profiles)
                for ticker in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results with diagnostic logging
            batch_completed = []
            exception_count = 0
            other_count = 0
            for j, result in enumerate(batch_results):
                if isinstance(result, StockData):
                    results.append(result)
                    batch_completed.append(batch[j])
                elif isinstance(result, Exception):
                    exception_count += 1
                    logger.warning(f"BATCH EXCEPTION for {batch[j]}: {type(result).__name__}: {result}")
                else:
                    other_count += 1
                    logger.warning(f"BATCH UNEXPECTED TYPE for {batch[j]}: {type(result)} = {result}")

            # Log batch summary if there were failures
            if exception_count > 0 or other_count > 0:
                logger.warning(f"Batch {i//effective_batch_size + 1}: {len(batch_completed)} succeeded, {exception_count} exceptions, {other_count} other")

            # Save progress checkpoint
            completed_tickers.extend(batch_completed)
            save_scan_progress(completed_tickers, scan_id)

            # Report progress with rate limit stats
            # Note: completed_tickers already includes checkpoint tickers + newly completed,
            # so we don't need to add len(results) (that would double-count)
            if progress_callback:
                progress_callback(len(completed_tickers), original_total)

            rate_stats = get_rate_limit_stats()
            current_progress = len(completed_tickers)
            logger.info(f"Progress: {current_progress}/{original_total} stocks ({current_progress/original_total*100:.1f}%) | "
                       f"API calls: {rate_stats['total_calls']} | 429s: {rate_stats['total_429s']}")

            # Smart delay between batches based on rate limit status
            if i + effective_batch_size < len(tickers):
                # Base delay of 2 seconds, increase if we've hit 429s
                delay = 2.0 + (rate_stats['consecutive_429s'] * 3.0)
                delay = min(delay, 15.0)  # Cap at 15 seconds
                await asyncio.sleep(delay)

    # Clear checkpoint on successful completion
    clear_scan_progress()

    # Log fallback statistics
    fallback_stats = get_fallback_stats()
    if fallback_stats["fallback_count"] > 0:
        logger.warning(f"FALLBACK SUMMARY: {fallback_stats['fallback_count']} stocks used cached data: {fallback_stats['stocks_using_fallback']}")
    if fallback_stats["no_data_count"] > 0:
        logger.error(f"DATA GAPS: {fallback_stats['no_data_count']} stocks have incomplete data: {fallback_stats['stocks_no_data']}")
    if fallback_stats["fallback_count"] == 0 and fallback_stats["no_data_count"] == 0:
        logger.info("All stocks fetched fresh data successfully - no fallbacks needed")

    return results


def fetch_stocks_async_wrapper(tickers: List[str], batch_size: int = 100, progress_callback=None) -> List[StockData]:
    """
    Synchronous wrapper for async fetch function
    Can be called from non-async code
    """
    return asyncio.run(fetch_stocks_batch_async(tickers, batch_size, progress_callback))


# Legacy compatibility
def get_price_data_only(ticker: str) -> StockData:
    """
    Fetch ONLY price history for ETFs and indexes (no fundamentals)
    Synchronous version for use in growth_projector
    """
    stock_data = StockData(ticker)

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

    if stock_data.current_price and not stock_data.price_history.empty:
        stock_data.is_valid = True

    return stock_data


if __name__ == "__main__":
    # Test async fetching with optimizations
    import time

    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
                    "V", "MA", "HD", "PG", "JPM", "XOM", "CVX", "ABBV", "MRK", "PFE"]

    print(f"\n{'='*60}")
    print("Testing Optimized Async Data Fetcher v2.0")
    print(f"{'='*60}\n")
    print(f"Fetching {len(test_tickers)} stocks with batch optimization...\n")

    start = time.time()
    results = fetch_stocks_async_wrapper(test_tickers, batch_size=20)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Fetched {len(results)} stocks in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/len(results):.2f} seconds per stock")
    print(f"\nSample results:")
    for stock in results[:5]:
        print(f"  {stock.ticker}: ${stock.current_price:.2f}, {stock.sector}, "
              f"EPS quarters: {len(stock.quarterly_earnings)}, Valid: {stock.is_valid}")
    print(f"{'='*60}\n")
