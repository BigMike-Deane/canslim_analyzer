"""
Async Scanner Integration for CANSLIM Analyzer
Replaces ThreadPoolExecutor with async batch processing for 5-10x speed boost
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import List, Dict
from async_data_fetcher import (
    fetch_stocks_batch_async,
    fetch_fmp_insider_trading_async,
    fetch_short_interest_async
)

logger = logging.getLogger(__name__)


async def fetch_insider_short_batch_async(
    tickers: List[str],
    session: aiohttp.ClientSession
) -> Dict[str, Dict]:
    """
    Batch fetch insider trading and short interest data for multiple tickers.
    Respects freshness intervals to avoid redundant API calls.

    Returns dict mapping ticker -> {insider_data, short_data}
    """
    from data_fetcher import is_data_fresh, mark_data_fetched, get_cached_data, set_cached_data

    results = {}
    insider_tasks = []
    short_tasks = []
    tickers_needing_insider = []
    tickers_needing_short = []

    # Check which tickers need fresh data
    for ticker in tickers:
        results[ticker] = {"insider": {}, "short": {}}

        # Check insider data freshness (14 days)
        if not is_data_fresh(ticker, "insider_trading"):
            tickers_needing_insider.append(ticker)
            insider_tasks.append(fetch_fmp_insider_trading_async(session, ticker))
        else:
            # Use cached data if available
            cached = get_cached_data(ticker, "insider_trading")
            if cached:
                results[ticker]["insider"] = cached

        # Check short interest freshness (3 days)
        if not is_data_fresh(ticker, "short_interest"):
            tickers_needing_short.append(ticker)
            short_tasks.append(fetch_short_interest_async(ticker))
        else:
            # Use cached data if available
            cached = get_cached_data(ticker, "short_interest")
            if cached:
                results[ticker]["short"] = cached

    # Fetch insider data in parallel
    if insider_tasks:
        logger.info(f"Fetching insider trading data for {len(insider_tasks)} tickers...")
        insider_results = await asyncio.gather(*insider_tasks, return_exceptions=True)
        for ticker, data in zip(tickers_needing_insider, insider_results):
            if isinstance(data, dict) and data:
                results[ticker]["insider"] = data
                mark_data_fetched(ticker, "insider_trading")
                set_cached_data(ticker, "insider_trading", data, persist_to_db=False)

    # Fetch short interest in parallel (uses executor, so limit concurrency)
    if short_tasks:
        logger.info(f"Fetching short interest data for {len(short_tasks)} tickers...")
        # Process in smaller batches to avoid overwhelming the executor
        BATCH_SIZE = 20
        for i in range(0, len(short_tasks), BATCH_SIZE):
            batch = short_tasks[i:i + BATCH_SIZE]
            batch_tickers = tickers_needing_short[i:i + BATCH_SIZE]
            short_results = await asyncio.gather(*batch, return_exceptions=True)
            for ticker, data in zip(batch_tickers, short_results):
                if isinstance(data, dict) and data:
                    results[ticker]["short"] = data
                    mark_data_fetched(ticker, "short_interest")
                    set_cached_data(ticker, "short_interest", data, persist_to_db=False)

    return results


async def analyze_stocks_async(tickers: List[str], batch_size: int = 100, progress_callback=None) -> List[Dict]:
    """
    Analyze multiple stocks asynchronously

    Args:
        tickers: List of stock tickers to analyze
        batch_size: Number of stocks to fetch concurrently (default 100)
        progress_callback: Optional function to call with progress updates (current_count, total_count)

    Returns:
        List of analysis results (dicts with CANSLIM scores, etc.)
    """
    from canslim_scorer import CANSLIMScorer, GrowthModeScorer, TechnicalAnalyzer
    from data_fetcher import DataFetcher
    from growth_projector import GrowthProjector

    logger.info(f"Starting async analysis of {len(tickers)} stocks...")

    # Fetch all stock data asynchronously in batches
    start_time = datetime.now()
    stock_data_list = await fetch_stocks_batch_async(tickers, batch_size=batch_size, progress_callback=progress_callback)
    fetch_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"✓ Fetched {len(stock_data_list)} stocks in {fetch_time:.1f}s "
               f"({fetch_time/len(stock_data_list):.2f}s per stock)")

    # Fetch insider/short data for all valid tickers
    valid_tickers = [sd.ticker for sd in stock_data_list if sd and sd.is_valid]
    insider_short_data = {}

    if valid_tickers:
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=25)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            insider_short_data = await fetch_insider_short_batch_async(valid_tickers, session)

    # Now analyze each stock (this is fast, no API calls)
    data_fetcher = DataFetcher()
    canslim_scorer = CANSLIMScorer(data_fetcher)
    growth_mode_scorer = GrowthModeScorer(data_fetcher, canslim_scorer)
    growth_projector = GrowthProjector(data_fetcher)

    results = []
    for stock_data in stock_data_list:
        try:
            if not stock_data or not stock_data.is_valid:
                continue

            # Score the stock (fast - no API calls)
            canslim_result = canslim_scorer.score_stock(stock_data)
            projection = growth_projector.project_growth(
                stock_data=stock_data,
                canslim_score=canslim_result
            )

            # Growth Mode scoring
            growth_mode_result = None
            is_growth_stock = growth_mode_scorer.should_use_growth_mode(stock_data)
            if is_growth_stock:
                growth_mode_result = growth_mode_scorer.score_stock(stock_data)

            # Technical analysis (fast)
            base_pattern = TechnicalAnalyzer.detect_base_pattern(stock_data.weekly_price_history)
            volume_ratio = TechnicalAnalyzer.calculate_volume_ratio(stock_data)
            is_breaking_out, breakout_vol = TechnicalAnalyzer.is_breaking_out(stock_data, base_pattern)

            # Get pre-fetched insider/short data for this ticker
            ticker_supplemental = insider_short_data.get(stock_data.ticker, {})
            insider_data = ticker_supplemental.get("insider", {})
            short_data = ticker_supplemental.get("short", {})

            # Calculate revenue growth
            revenue_growth_pct = None
            if stock_data.quarterly_revenue and len(stock_data.quarterly_revenue) >= 5:
                current = stock_data.quarterly_revenue[0]
                prior = stock_data.quarterly_revenue[4]
                if prior > 0:
                    revenue_growth_pct = round(((current - prior) / prior) * 100, 1)

            result = {
                "ticker": stock_data.ticker,
                "company_name": stock_data.name,
                "sector": stock_data.sector,
                "industry": None,
                "current_price": stock_data.current_price,
                "market_cap": stock_data.market_cap,
                "canslim_score": canslim_result.total_score,
                "c_score": canslim_result.c_score,
                "a_score": canslim_result.a_score,
                "n_score": canslim_result.n_score,
                "s_score": canslim_result.s_score,
                "l_score": canslim_result.l_score,
                "i_score": canslim_result.i_score,
                "m_score": canslim_result.m_score,
                "score_details": {
                    "c": {
                        "summary": canslim_result.c_detail,
                        "quarterly_eps": stock_data.quarterly_earnings[:4] if stock_data.quarterly_earnings else [],
                        "earnings_surprise_pct": stock_data.earnings_surprise_pct,
                    },
                    "a": {
                        "summary": canslim_result.a_detail,
                        "annual_eps": stock_data.annual_earnings[:3] if stock_data.annual_earnings else [],
                        "roe": stock_data.roe,
                    },
                    "n": {
                        "summary": canslim_result.n_detail,
                        "current_price": stock_data.current_price,
                        "week_52_high": stock_data.high_52w,
                        "pct_from_high": round(((stock_data.high_52w - stock_data.current_price) / stock_data.high_52w) * 100, 1) if stock_data.high_52w and stock_data.current_price else None,
                    },
                    "s": {
                        "summary": canslim_result.s_detail,
                        "volume_ratio": volume_ratio,
                        "avg_volume": stock_data.avg_volume_50d,
                        "shares_outstanding": stock_data.shares_outstanding,
                    },
                    "l": {
                        "summary": canslim_result.l_detail,
                        "relative_strength": stock_data.relative_strength if hasattr(stock_data, 'relative_strength') else None,
                    },
                    "i": {
                        "summary": canslim_result.i_detail,
                        "institutional_pct": stock_data.institutional_holders_pct,
                    },
                    "m": {
                        "summary": canslim_result.m_detail,
                    },
                },
                "projected_growth": projection.projected_growth_pct,
                "confidence": projection.confidence,
                "analyst_target": projection.analyst_target,
                "pe_ratio": stock_data.trailing_pe,
                "week_52_high": stock_data.high_52w,
                "week_52_low": stock_data.low_52w,
                "relative_strength": None,
                "institutional_ownership": stock_data.institutional_holders_pct,
                # Growth Mode
                "is_growth_stock": is_growth_stock,
                "growth_mode_score": growth_mode_result.total_score if growth_mode_result else None,
                "growth_mode_details": {
                    "r": growth_mode_result.r_detail,
                    "f": growth_mode_result.f_detail,
                    "n": growth_mode_result.n_detail,
                    "s": growth_mode_result.s_detail,
                    "l": growth_mode_result.l_detail,
                    "i": growth_mode_result.i_detail,
                    "m": growth_mode_result.m_detail,
                } if growth_mode_result else None,
                # Enhanced earnings
                "eps_acceleration": len(stock_data.quarterly_earnings) >= 5 and canslim_result.c_detail and "+accel" in canslim_result.c_detail,
                "earnings_surprise_pct": stock_data.earnings_surprise_pct,
                "revenue_growth_pct": revenue_growth_pct,
                "quarterly_earnings": stock_data.quarterly_earnings[:8] if stock_data.quarterly_earnings else [],
                "annual_earnings": stock_data.annual_earnings[:5] if stock_data.annual_earnings else [],
                "quarterly_revenue": stock_data.quarterly_revenue[:8] if stock_data.quarterly_revenue else [],
                # Technical
                "volume_ratio": volume_ratio,
                "weeks_in_base": base_pattern.get("weeks", 0),
                "base_type": base_pattern.get("type", "none"),
                "is_breaking_out": is_breaking_out,
                "breakout_volume_ratio": breakout_vol if is_breaking_out else None,
                # Insider/Short (fetched asynchronously with caching)
                "insider_buy_count": insider_data.get("buy_count"),
                "insider_sell_count": insider_data.get("sell_count"),
                "insider_net_shares": insider_data.get("net_shares"),
                "insider_sentiment": insider_data.get("sentiment"),
                "short_interest_pct": short_data.get("short_interest_pct"),
                "short_ratio": short_data.get("short_ratio"),
            }

            results.append(result)

        except Exception as e:
            logger.error(f"Error analyzing {stock_data.ticker}: {e}")
            continue

    total_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"✓ Analyzed {len(results)} stocks in {total_time:.1f}s total")

    return results


def run_async_scan(tickers: List[str], batch_size: int = 100, progress_callback=None) -> List[Dict]:
    """
    Synchronous wrapper for async scanner
    Can be called from non-async code (like the scheduler)

    Args:
        tickers: List of stock tickers to scan
        batch_size: Number of stocks to process concurrently
        progress_callback: Optional callback function(current, total) to report progress

    Returns:
        List of analysis results
    """
    return asyncio.run(analyze_stocks_async(tickers, batch_size=batch_size, progress_callback=progress_callback))


if __name__ == "__main__":
    # Test async scanner
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ"]

    print(f"\n{'='*60}")
    print("Testing Async Scanner with Insider/Short Data")
    print(f"{'='*60}\n")

    import time
    start = time.time()
    results = run_async_scan(test_tickers, batch_size=10)
    elapsed = time.time() - start

    print(f"\n✓ Scanned {len(results)} stocks in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/len(results):.2f} seconds per stock")

    print(f"\nTop 3 by CANSLIM score:")
    sorted_results = sorted(results, key=lambda x: x['canslim_score'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result['ticker']}: {result['canslim_score']:.1f} points")

    print(f"\nInsider/Short Data Sample:")
    for result in results[:5]:
        insider = result.get('insider_sentiment', 'N/A')
        short_pct = result.get('short_interest_pct')
        short_str = f"{short_pct:.1f}%" if short_pct else "N/A"
        buys = result.get('insider_buy_count', 0) or 0
        sells = result.get('insider_sell_count', 0) or 0
        print(f"  {result['ticker']}: Insider={insider} (B:{buys}/S:{sells}), Short={short_str}")

    # Count how many have data
    with_insider = sum(1 for r in results if r.get('insider_sentiment'))
    with_short = sum(1 for r in results if r.get('short_interest_pct'))
    print(f"\nData Coverage: {with_insider}/{len(results)} with insider data, {with_short}/{len(results)} with short interest")

    print(f"\n{'='*60}\n")
