"""
Data Fetcher Module
Wrapper around yfinance with caching and error handling
Now with Financial Modeling Prep (FMP) API for earnings data
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
DATA_FRESHNESS_INTERVALS = {
    "price": 0,              # Always fetch (real-time)
    "earnings": 24 * 3600,   # Once per day (24 hours)
    "revenue": 24 * 3600,    # Once per day
    "balance_sheet": 24 * 3600,  # Once per day
    "key_metrics": 24 * 3600,    # Once per day
    "analyst": 24 * 3600,        # Once per day
    "earnings_surprise": 24 * 3600,  # Once per day
    "weekly_history": 24 * 3600,     # Once per day (for base detection)
    "institutional": 7 * 24 * 3600,  # Once per week
}

# Cached data storage (stores the actual fetched data)
_cached_data = {}
_cached_data_lock = threading.Lock()
MAX_CACHED_TICKERS = 500  # Limit cache size


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


def set_cached_data(ticker: str, data_type: str, data):
    """Store data in cache"""
    with _cached_data_lock:
        # Enforce cache size limit
        if len(_cached_data) >= MAX_CACHED_TICKERS * 10:  # ~10 data types per ticker
            # Remove oldest 20% of entries
            keys_to_remove = list(_cached_data.keys())[:int(len(_cached_data) * 0.2)]
            for key in keys_to_remove:
                del _cached_data[key]

        key = f"{ticker}:{data_type}"
        _cached_data[key] = data


_cache_hit_count = 0
_cache_miss_count = 0

# Set to False to disable caching (for debugging)
CACHING_ENABLED = True

def fetch_with_cache(ticker: str, data_type: str, fetch_func, *args, **kwargs):
    """
    Wrapper that checks cache freshness before fetching.
    Returns cached data if fresh, otherwise fetches new data.
    """
    global _cache_hit_count, _cache_miss_count

    # Skip caching if disabled
    if not CACHING_ENABLED:
        return fetch_func(*args, **kwargs)

    # Check if we have fresh cached data
    if is_data_fresh(ticker, data_type):
        cached = get_cached_data(ticker, data_type)
        if cached is not None:
            _cache_hit_count += 1
            # Log every 100 hits to avoid spam
            if _cache_hit_count % 100 == 0:
                logger.info(f"Cache stats: {_cache_hit_count} hits, {_cache_miss_count} misses")
            return cached

    # Fetch fresh data
    _cache_miss_count += 1
    data = fetch_func(*args, **kwargs)

    # Cache the result
    if data:
        set_cached_data(ticker, data_type, data)
        mark_data_fetched(ticker, data_type)

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

    return {
        "tickers_tracked": len(_data_freshness_cache),
        "freshness_entries": freshness_count,
        "cached_data_entries": data_count
    }


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
    """Fetch earnings surprise history from FMP"""
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

                return {
                    "latest_surprise_pct": latest_surprise,
                    "beat_streak": beat_streak,
                }
    except Exception as e:
        logger.debug(f"FMP earnings surprise error for {ticker}: {e}")
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
                    "name": meta.get("longName") or meta.get("shortName") or ticker,
                    "close_prices": indicators.get("close", []),
                    "volumes": indicators.get("volume", []),
                    "timestamps": timestamps
                }
    except Exception:
        pass
    return {}


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
            if (not stock_data.high_52w or not stock_data.low_52w) and not stock_data.price_history.empty:
                try:
                    closes = stock_data.price_history['Close'].dropna()
                    if len(closes) > 0:
                        if not stock_data.high_52w:
                            stock_data.high_52w = float(closes.max())
                        if not stock_data.low_52w:
                            stock_data.low_52w = float(closes.min())
                        logger.debug(f"Calculated 52w range for {ticker}: ${stock_data.low_52w:.2f} - ${stock_data.high_52w:.2f}")
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
            price_target = fetch_with_cache(ticker, "analyst", fetch_fmp_price_target, ticker)
            if price_target:
                stock_data.analyst_target_price = price_target.get("target_consensus", 0) or price_target.get("target_median", 0)
                stock_data.analyst_target_high = price_target.get("target_high", 0)
                stock_data.analyst_target_low = price_target.get("target_low", 0)

            analyst = fetch_with_cache(ticker, "analyst", fetch_fmp_analyst, ticker)
            if analyst:
                stock_data.num_analyst_opinions = analyst.get("num_analysts", 0)
                stock_data.earnings_growth_estimate = analyst.get("estimated_eps_avg", 0)

            # 5b. Get earnings surprise data (for enhanced C score)
            # TIERED: Cache for 24 hours
            earnings_surprise = fetch_with_cache(ticker, "earnings_surprise", fetch_fmp_earnings_surprise, ticker)
            if earnings_surprise:
                stock_data.earnings_surprise_pct = earnings_surprise.get("latest_surprise_pct", 0)
                stock_data.eps_beat_streak = earnings_surprise.get("beat_streak", 0)

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

        # 3. Fallback to yfinance ONLY if FMP didn't provide critical data
        # Skip yfinance if we already have earnings from FMP (to avoid rate limits)
        if not stock_data.quarterly_earnings and not FMP_API_KEY:
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

                    # Quarterly earnings from yfinance
                    if not stock_data.quarterly_earnings:
                        try:
                            quarterly = stock.quarterly_financials
                            if quarterly is not None and not quarterly.empty:
                                if 'Net Income' in quarterly.index:
                                    net_income = quarterly.loc['Net Income'].dropna()
                                    shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                                    stock_data.quarterly_earnings = (net_income / shares).tolist()[:8]
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
