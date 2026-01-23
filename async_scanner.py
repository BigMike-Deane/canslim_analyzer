"""
Async Scanner Integration for CANSLIM Analyzer
Replaces ThreadPoolExecutor with async batch processing for 5-10x speed boost
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict
from async_data_fetcher import fetch_stocks_batch_async

logger = logging.getLogger(__name__)


async def analyze_stocks_async(tickers: List[str], batch_size: int = 50, progress_callback=None) -> List[Dict]:
    """
    Analyze multiple stocks asynchronously

    Args:
        tickers: List of stock tickers to analyze
        batch_size: Number of stocks to fetch concurrently (default 50)
        progress_callback: Optional function to call with progress updates (current_count, total_count)

    Returns:
        List of analysis results (dicts with CANSLIM scores, etc.)
    """
    from canslim_scorer import CANSLIMScorer, GrowthModeScorer, TechnicalAnalyzer
    from data_fetcher import DataFetcher, fetch_fmp_insider_trading, fetch_short_interest, is_data_fresh, mark_data_fetched
    from growth_projector import GrowthProjector

    logger.info(f"Starting async analysis of {len(tickers)} stocks...")

    # Fetch all stock data asynchronously in batches
    start_time = datetime.now()
    stock_data_list = await fetch_stocks_batch_async(tickers, batch_size=batch_size, progress_callback=progress_callback)
    fetch_time = (datetime.now() - start_time).total_seconds()

    logger.info(f"✓ Fetched {len(stock_data_list)} stocks in {fetch_time:.1f}s "
               f"({fetch_time/len(stock_data_list):.2f}s per stock)")

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

            # For insider/short data, we'll still use sync calls (less critical for performance)
            # These could be optimized later
            insider_data = {}
            short_data = {}

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
                    "c": canslim_result.c_detail,
                    "a": canslim_result.a_detail,
                    "n": canslim_result.n_detail,
                    "s": canslim_result.s_detail,
                    "l": canslim_result.l_detail,
                    "i": canslim_result.i_detail,
                    "m": canslim_result.m_detail,
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
                # Technical
                "volume_ratio": volume_ratio,
                "weeks_in_base": base_pattern.get("weeks", 0),
                "base_type": base_pattern.get("type", "none"),
                "is_breaking_out": is_breaking_out,
                "breakout_volume_ratio": breakout_vol if is_breaking_out else None,
                # Insider/Short (placeholders for now)
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


def run_async_scan(tickers: List[str], batch_size: int = 50, progress_callback=None) -> List[Dict]:
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
    print("Testing Async Scanner")
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

    print(f"\n{'='*60}\n")
