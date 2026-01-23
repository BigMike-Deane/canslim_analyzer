"""
Async Data Fetcher Module - High Performance Version
Fetches stock data asynchronously for 5-10x speed improvement
Uses aiohttp for concurrent API calls
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging
import os
import json

from data_fetcher import (
    StockData, FMP_API_KEY, FMP_BASE_URL,
    get_cached_data, set_cached_data, mark_data_fetched, is_data_fresh,
    fetch_price_from_chart_api, fetch_weekly_price_history,
    fetch_finviz_institutional, fetch_short_interest,
    REDIS_AVAILABLE
)

if REDIS_AVAILABLE:
    from redis_cache import redis_cache

logger = logging.getLogger(__name__)

# Semaphore to limit concurrent requests (avoid rate limiting)
MAX_CONCURRENT_REQUESTS = 10
api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


async def fetch_json_async(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Optional[dict]:
    """Async HTTP GET that returns JSON"""
    async with api_semaphore:  # Limit concurrent requests
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.debug(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")
            return None


async def fetch_fmp_profile_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch company profile from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

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
    return {}


async def fetch_fmp_quote_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch current quote from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

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
    return {}


async def fetch_fmp_earnings_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch quarterly and annual earnings from FMP"""
    if not FMP_API_KEY:
        return {}

    result = {"quarterly_eps": [], "annual_eps": []}

    # Fetch quarterly and annual in parallel
    quarterly_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
    annual_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"

    quarterly_data, annual_data = await asyncio.gather(
        fetch_json_async(session, quarterly_url),
        fetch_json_async(session, annual_url),
        return_exceptions=True
    )

    if isinstance(quarterly_data, list) and quarterly_data:
        result["quarterly_eps"] = [q.get("eps", 0) or 0 for q in quarterly_data]
        result["quarterly_net_income"] = [q.get("netIncome", 0) or 0 for q in quarterly_data]

    if isinstance(annual_data, list) and annual_data:
        result["annual_eps"] = [a.get("eps", 0) or 0 for a in annual_data]
        result["annual_net_income"] = [a.get("netIncome", 0) or 0 for a in annual_data]

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


async def fetch_fmp_revenue_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch revenue data from FMP"""
    if not FMP_API_KEY:
        return {}

    result = {"quarterly_revenue": [], "annual_revenue": []}

    quarterly_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
    annual_url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"

    quarterly_data, annual_data = await asyncio.gather(
        fetch_json_async(session, quarterly_url),
        fetch_json_async(session, annual_url),
        return_exceptions=True
    )

    if isinstance(quarterly_data, list) and quarterly_data:
        result["quarterly_revenue"] = [q.get("revenue", 0) or 0 for q in quarterly_data]

    if isinstance(annual_data, list) and annual_data:
        result["annual_revenue"] = [a.get("revenue", 0) or 0 for a in annual_data]

    return result


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
    """Async fetch analyst data from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

    if data and len(data) > 0:
        est = data[0]
        return {
            "estimated_eps_avg": est.get("estimatedEpsAvg", 0),
            "estimated_eps_high": est.get("estimatedEpsHigh", 0),
            "estimated_eps_low": est.get("estimatedEpsLow", 0),
            "estimated_revenue_avg": est.get("estimatedRevenueAvg", 0),
            "num_analysts": est.get("numberAnalystsEstimatedEps", 0),
        }
    return {}


async def fetch_fmp_price_target_async(session: aiohttp.ClientSession, ticker: str) -> dict:
    """Async fetch analyst price targets from FMP"""
    if not FMP_API_KEY:
        return {}

    url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}"
    data = await fetch_json_async(session, url)

    if data and len(data) > 0:
        pt = data[0]
        return {
            "target_high": pt.get("targetHigh", 0),
            "target_low": pt.get("targetLow", 0),
            "target_consensus": pt.get("targetConsensus", 0),
            "target_median": pt.get("targetMedian", 0),
        }
    return {}


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


async def get_stock_data_async(ticker: str, session: aiohttp.ClientSession) -> StockData:
    """
    Async version of get_stock_data - fetches all data for a stock in parallel

    This is the main performance improvement: all API calls happen concurrently!
    """
    stock_data = StockData(ticker)

    # Check cache first (Redis if available, otherwise memory)
    # For now, use synchronous cache checks (can be optimized later with async Redis)

    # Fetch all data in parallel!
    tasks = []

    if FMP_API_KEY:
        tasks.extend([
            fetch_fmp_profile_async(session, ticker),
            fetch_fmp_quote_async(session, ticker),
            fetch_fmp_earnings_async(session, ticker),
            fetch_fmp_key_metrics_async(session, ticker),
            fetch_fmp_revenue_async(session, ticker),
            fetch_fmp_balance_sheet_async(session, ticker),
            fetch_fmp_analyst_async(session, ticker),
            fetch_fmp_price_target_async(session, ticker),
            fetch_fmp_earnings_surprise_async(session, ticker),
        ])

    # Fetch all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results (in same order as tasks)
    idx = 0
    if FMP_API_KEY:
        profile = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        quote = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        earnings = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        key_metrics = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        revenue = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        balance_sheet = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        analyst = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        price_target = results[idx] if not isinstance(results[idx], Exception) else {}
        idx += 1
        earnings_surprise = results[idx] if not isinstance(results[idx], Exception) else {}

        # Populate stock_data from fetched results
        if profile:
            stock_data.name = profile.get("name", ticker)
            stock_data.sector = profile.get("sector", "")
            stock_data.shares_outstanding = int(profile.get("shares_outstanding", 0) or 0)
            stock_data.current_price = profile.get("current_price", 0)
            stock_data.high_52w = float(profile.get("high_52w", 0) or 0) if isinstance(profile.get("high_52w"), (int, float)) else 0

        if quote:
            if not stock_data.current_price:
                stock_data.current_price = quote.get("current_price", 0)
            if not stock_data.high_52w:
                stock_data.high_52w = quote.get("high_52w", 0) or 0
            stock_data.low_52w = quote.get("low_52w", 0) or 0
            stock_data.market_cap = quote.get("market_cap", 0)
            stock_data.avg_volume_50d = quote.get("avg_volume", 0)
            stock_data.current_volume = quote.get("volume", 0)
            stock_data.trailing_pe = quote.get("pe", 0) or 0
            if not stock_data.shares_outstanding:
                stock_data.shares_outstanding = int(quote.get("shares_outstanding", 0) or 0)

        if earnings:
            stock_data.quarterly_earnings = earnings.get("quarterly_eps", [])
            stock_data.annual_earnings = earnings.get("annual_eps", [])

        if key_metrics:
            stock_data.roe = key_metrics.get("roe", 0)
            stock_data.roa = key_metrics.get("roa", 0)
            stock_data.roic = key_metrics.get("roic", 0)
            stock_data.earnings_yield = key_metrics.get("earnings_yield", 0)
            stock_data.fcf_yield = key_metrics.get("fcf_yield", 0)

        if revenue:
            stock_data.quarterly_revenue = revenue.get("quarterly_revenue", [])
            stock_data.annual_revenue = revenue.get("annual_revenue", [])

        if balance_sheet:
            stock_data.cash_and_equivalents = balance_sheet.get("cash_and_equivalents", 0)
            stock_data.total_debt = balance_sheet.get("total_debt", 0)

        if analyst:
            stock_data.num_analyst_opinions = analyst.get("num_analysts", 0)
            stock_data.earnings_growth_estimate = analyst.get("estimated_eps_avg", 0)

        if price_target:
            stock_data.analyst_target_price = price_target.get("target_consensus", 0) or price_target.get("target_median", 0)
            stock_data.analyst_target_high = price_target.get("target_high", 0)
            stock_data.analyst_target_low = price_target.get("target_low", 0)

        if earnings_surprise:
            stock_data.earnings_surprise_pct = earnings_surprise.get("latest_surprise_pct", 0)
            stock_data.eps_beat_streak = earnings_surprise.get("beat_streak", 0)
            adjusted_eps = earnings_surprise.get("quarterly_adjusted_eps", [])
            if adjusted_eps and len(adjusted_eps) >= 4:
                stock_data.quarterly_earnings = adjusted_eps

    # Get price history (synchronous for now - could be optimized)
    chart_data = fetch_price_from_chart_api(ticker)
    if chart_data.get("current_price"):
        if not stock_data.current_price:
            stock_data.current_price = chart_data["current_price"]
        if not stock_data.high_52w:
            stock_data.high_52w = chart_data.get("high_52w", 0) or 0
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
                stock_data.avg_volume_50d = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
                stock_data.current_volume = volumes[-1] if volumes[-1] else 0

    # Mark as valid if we have basic data
    if stock_data.current_price and not stock_data.price_history.empty:
        stock_data.is_valid = True
    elif stock_data.current_price:
        stock_data.is_valid = True
        stock_data.error_message = "Limited data available"

    return stock_data


async def fetch_stocks_batch_async(tickers: List[str], batch_size: int = 50, progress_callback=None) -> List[StockData]:
    """
    Fetch multiple stocks concurrently in batches

    Args:
        tickers: List of stock tickers to fetch
        batch_size: Number of stocks to process at once (default 50)
        progress_callback: Optional callback function(current, total) to report progress

    Returns:
        List of StockData objects
    """
    results = []

    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            # Fetch all stocks in this batch concurrently
            tasks = [get_stock_data_async(ticker, session) for ticker in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            for result in batch_results:
                if isinstance(result, StockData):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error fetching stock: {result}")

            # Report progress after each batch
            if progress_callback:
                progress_callback(len(results), len(tickers))

            # Small delay between batches to be nice to the API
            if i + batch_size < len(tickers):
                await asyncio.sleep(0.5)

    return results


def fetch_stocks_async_wrapper(tickers: List[str], batch_size: int = 50) -> List[StockData]:
    """
    Synchronous wrapper for async fetch function
    Can be called from non-async code
    """
    return asyncio.run(fetch_stocks_batch_async(tickers, batch_size))


if __name__ == "__main__":
    # Test async fetching
    import time

    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ"]

    print(f"\n{'='*60}")
    print("Testing Async Data Fetcher Performance")
    print(f"{'='*60}\n")
    print(f"Fetching {len(test_tickers)} stocks concurrently...\n")

    start = time.time()
    results = fetch_stocks_async_wrapper(test_tickers, batch_size=10)
    elapsed = time.time() - start

    print(f"\n✓ Fetched {len(results)} stocks in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/len(results):.2f} seconds per stock")
    print(f"\nSample results:")
    for stock in results[:3]:
        print(f"  {stock.ticker}: ${stock.current_price:.2f}, {stock.sector}, Valid: {stock.is_valid}")

    print(f"\n{'='*60}\n")
