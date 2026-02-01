"""
CANSLIM Continuous Scanning Scheduler

Runs automatic scans at configurable intervals to keep stock data fresh.
Stays within FMP API rate limits (300 calls/min).
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import logging
import os
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scheduler instance
scheduler = BackgroundScheduler()

# State tracking
_scan_config = {
    "enabled": False,
    "source": "sp500",  # sp500, top50, russell, all
    "interval_minutes": 15,
    "last_scan_start": None,
    "last_scan_end": None,
    "stocks_scanned": 0,
    "total_stocks": 0,
    "is_scanning": False
}


def get_scan_status():
    """Get current scheduler status"""
    next_run = None
    job = scheduler.get_job("continuous_scan")
    if job and job.next_run_time:
        # APScheduler returns timezone-aware datetime, isoformat() includes offset
        next_run = job.next_run_time.isoformat()

    return {
        **_scan_config,
        "scheduler_running": scheduler.running,
        "next_run": next_run
    }


def run_continuous_scan():
    """Execute a scan of the configured stock universe"""
    from backend.database import SessionLocal, Stock
    from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers
    import time

    if _scan_config["is_scanning"]:
        logger.info("Scan already in progress, skipping...")
        return

    _scan_config["is_scanning"] = True
    _scan_config["last_scan_start"] = datetime.utcnow().isoformat() + 'Z'
    _scan_config["stocks_scanned"] = 0
    _scan_config["total_stocks"] = 0

    logger.info(f"Starting continuous scan ({_scan_config['source']})...")

    # Get tickers based on source
    # Always include portfolio tickers first (they're most important)
    from sp500_tickers import get_portfolio_tickers
    portfolio_tickers = get_portfolio_tickers()

    source = _scan_config["source"]
    if source == "top50":
        base_tickers = get_sp500_tickers()[:50]
    elif source == "sp500":
        base_tickers = get_sp500_tickers()
    elif source == "russell":
        base_tickers = get_russell2000_tickers()
    elif source == "all":
        base_tickers = get_all_tickers(include_portfolio=False)  # Portfolio added separately
    else:
        base_tickers = get_sp500_tickers()

    # Combine: portfolio first (priority), then base tickers, deduplicated
    seen = set()
    tickers = []
    for t in portfolio_tickers + base_tickers:
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    # Shuffle ONLY the non-portfolio tickers to avoid Yahoo warmup issues
    # Keep portfolio tickers at the front (they're most important)
    num_portfolio = len(portfolio_tickers)
    portfolio_section = tickers[:num_portfolio]
    rest_section = tickers[num_portfolio:]
    random.shuffle(rest_section)
    tickers = portfolio_section + rest_section

    logger.info(f"Including {len(portfolio_tickers)} portfolio tickers in scan")

    _scan_config["total_stocks"] = len(tickers)
    logger.info(f"Scanning {len(tickers)} stocks...")

    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from canslim_scorer import CANSLIMScorer, GrowthModeScorer, TechnicalAnalyzer
    from data_fetcher import (
        DataFetcher, get_cached_market_direction,
        fetch_fmp_insider_trading, fetch_short_interest,
        is_data_fresh, mark_data_fetched
    )
    from growth_projector import GrowthProjector

    # IMPORTANT: Fetch market direction FIRST, before any stock analysis
    # This ensures the M score uses fresh data and avoids rate limiting during stock scans
    logger.info("Fetching market direction data (SPY, QQQ, DIA)...")
    market_data = get_cached_market_direction(force_refresh=True)
    if market_data.get("success"):
        logger.info(f"Market direction: {market_data.get('market_trend')} "
                   f"(score: {market_data.get('market_score')}, "
                   f"signal: {market_data.get('weighted_signal', 0):.2f})")
    else:
        logger.warning(f"Failed to fetch market direction: {market_data.get('error')}")

    data_fetcher = DataFetcher()
    canslim_scorer = CANSLIMScorer(data_fetcher)
    growth_mode_scorer = GrowthModeScorer(data_fetcher, canslim_scorer)
    growth_projector = GrowthProjector(data_fetcher)

    def analyze_stock(ticker: str) -> dict:
        """Analyze a single stock"""
        try:
            stock_data = data_fetcher.get_stock_data(ticker)
            if not stock_data or not stock_data.is_valid:
                return None

            canslim_result = canslim_scorer.score_stock(stock_data)
            projection = growth_projector.project_growth(
                stock_data=stock_data,
                canslim_score=canslim_result
            )

            # Calculate Growth Mode score if applicable
            growth_mode_result = None
            is_growth_stock = growth_mode_scorer.should_use_growth_mode(stock_data)
            if is_growth_stock:
                growth_mode_result = growth_mode_scorer.score_stock(stock_data)

            # Technical analysis
            base_pattern = TechnicalAnalyzer.detect_base_pattern(stock_data.weekly_price_history)
            volume_ratio = TechnicalAnalyzer.calculate_volume_ratio(stock_data)
            is_breaking_out, breakout_vol = TechnicalAnalyzer.is_breaking_out(stock_data, base_pattern)

            # Insider trading signals (fetch weekly)
            insider_data = {}
            if not is_data_fresh(ticker, "insider_trading"):
                insider_data = fetch_fmp_insider_trading(ticker)
                if insider_data:
                    mark_data_fetched(ticker, "insider_trading")

            # Short interest data (fetch daily)
            short_data = {}
            if not is_data_fresh(ticker, "short_interest"):
                short_data = fetch_short_interest(ticker)
                if short_data:
                    mark_data_fetched(ticker, "short_interest")

            return {
                "ticker": ticker,
                "company_name": stock_data.name,
                "sector": stock_data.sector,
                "industry": None,  # StockData doesn't have industry
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
                "analyst_target": stock_data.analyst_target_price,
                "pe_ratio": stock_data.trailing_pe,
                "week_52_high": stock_data.high_52w,
                "week_52_low": stock_data.low_52w,
                "relative_strength": None,  # Could parse from l_detail if needed
                "institutional_ownership": stock_data.institutional_holders_pct,
                # Growth Mode scoring
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
                "revenue_growth_pct": _calc_revenue_growth(stock_data),
                # Technical analysis
                "volume_ratio": volume_ratio,
                "weeks_in_base": base_pattern.get("weeks", 0),
                "base_type": base_pattern.get("type", "none"),
                "pivot_price": base_pattern.get("pivot_price", 0),
                "is_breaking_out": is_breaking_out,
                "breakout_volume_ratio": breakout_vol if is_breaking_out else None,
                # Insider trading signals
                "insider_buy_count": insider_data.get("buy_count"),
                "insider_sell_count": insider_data.get("sell_count"),
                "insider_net_shares": insider_data.get("net_shares"),
                "insider_sentiment": insider_data.get("sentiment"),
                # Short interest
                "short_interest_pct": short_data.get("short_interest_pct"),
                "short_ratio": short_data.get("short_ratio"),
                # Raw earnings data (needed for database save)
                "quarterly_earnings": stock_data.quarterly_earnings,
                "annual_earnings": stock_data.annual_earnings,
                "quarterly_revenue": stock_data.quarterly_revenue,
            }
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None

    def _calc_revenue_growth(stock_data) -> float:
        """Calculate YoY revenue growth percentage"""
        if stock_data.quarterly_revenue and len(stock_data.quarterly_revenue) >= 5:
            current = stock_data.quarterly_revenue[0]
            prior = stock_data.quarterly_revenue[4]
            if prior > 0:
                return round(((current - prior) / prior) * 100, 1)
        return None

    def save_stock_to_db(db, analysis: dict):
        """Save analysis to database with historical tracking"""
        from backend.database import Stock, StockScore
        from datetime import date

        stock = db.query(Stock).filter(Stock.ticker == analysis["ticker"]).first()
        if not stock:
            stock = Stock(ticker=analysis["ticker"])
            db.add(stock)
            db.flush()  # Get the ID

        # Track score change
        old_score = stock.canslim_score
        new_score = analysis.get("canslim_score")

        # SAFEGUARD: Detect potential data blips before saving
        # If score dropped dramatically AND key data is missing, keep the old score
        if old_score is not None and new_score is not None:
            score_drop = old_score - new_score

            # Check for signs of incomplete data
            has_earnings_data = bool(analysis.get("quarterly_earnings"))
            has_price_data = analysis.get("current_price") is not None and analysis.get("current_price") > 0
            has_52w_high = analysis.get("week_52_high") is not None and analysis.get("week_52_high") > 0

            # Count how many component scores are 0 (suspicious if many are 0)
            component_scores = [
                analysis.get("c_score", 0),
                analysis.get("a_score", 0),
                analysis.get("n_score", 0),
                analysis.get("s_score", 0),
                analysis.get("l_score", 0),
                analysis.get("i_score", 0),
            ]
            zero_components = sum(1 for s in component_scores if s == 0)

            # Check score details for "Insufficient data" or "No data" indicators
            score_details = analysis.get("score_details", {})
            data_issues = []
            for component, details in score_details.items():
                if isinstance(details, dict):
                    summary = details.get("summary", "")
                    if any(x in summary.lower() for x in ["insufficient", "no data", "no price", "no volume"]):
                        data_issues.append(component.upper())

            # BLIP DETECTION: Score dropped >25 points AND looks like missing data
            is_likely_blip = (
                score_drop > 25 and  # Big drop
                (zero_components >= 3 or  # Too many zero components
                 not has_earnings_data or  # Missing earnings
                 len(data_issues) >= 2)  # Multiple data issues
            )

            if is_likely_blip:
                logger.warning(f"DATA BLIP DETECTED for {analysis['ticker']}: "
                              f"Score would drop {old_score:.0f} → {new_score:.0f} (-{score_drop:.0f}). "
                              f"Issues: {data_issues}, Zero components: {zero_components}. "
                              f"KEEPING OLD SCORE.")
                # Keep the old score - don't update
                new_score = old_score
                analysis["canslim_score"] = old_score
                # Also preserve component scores if they were non-zero before
                if stock.c_score and analysis.get("c_score") == 0:
                    analysis["c_score"] = stock.c_score
                if stock.a_score and analysis.get("a_score") == 0:
                    analysis["a_score"] = stock.a_score
                if stock.n_score and analysis.get("n_score") == 0:
                    analysis["n_score"] = stock.n_score
                if stock.s_score and analysis.get("s_score") == 0:
                    analysis["s_score"] = stock.s_score
                if stock.l_score and analysis.get("l_score") == 0:
                    analysis["l_score"] = stock.l_score
                if stock.i_score and analysis.get("i_score") == 0:
                    analysis["i_score"] = stock.i_score

            stock.previous_score = old_score
            stock.score_change = round(new_score - old_score, 2)
        else:
            stock.previous_score = None
            stock.score_change = None

        # Update stock data
        stock.name = analysis.get("company_name")
        stock.sector = analysis.get("sector")
        stock.industry = analysis.get("industry")
        stock.current_price = analysis.get("current_price")
        stock.market_cap = analysis.get("market_cap")
        # Debug log for market_cap
        logger.info(f"{analysis['ticker']}: market_cap in analysis={analysis.get('market_cap')}, assigned to stock.market_cap")
        stock.canslim_score = new_score
        stock.c_score = analysis.get("c_score")
        stock.a_score = analysis.get("a_score")
        stock.n_score = analysis.get("n_score")
        stock.s_score = analysis.get("s_score")
        stock.l_score = analysis.get("l_score")
        stock.i_score = analysis.get("i_score")
        stock.m_score = analysis.get("m_score")
        stock.score_details = analysis.get("score_details")
        stock.projected_growth = analysis.get("projected_growth")
        stock.confidence = analysis.get("confidence")
        stock.analyst_target = analysis.get("analyst_target")
        stock.pe_ratio = analysis.get("pe_ratio")
        stock.week_52_high = analysis.get("week_52_high")
        stock.week_52_low = analysis.get("week_52_low")
        stock.relative_strength = analysis.get("relative_strength")
        stock.institutional_ownership = analysis.get("institutional_ownership")
        stock.last_updated = datetime.utcnow()

        # Growth Mode scoring
        stock.growth_mode_score = analysis.get("growth_mode_score")
        stock.growth_mode_details = analysis.get("growth_mode_details")
        stock.is_growth_stock = analysis.get("is_growth_stock", False)

        # Enhanced earnings analysis
        stock.eps_acceleration = analysis.get("eps_acceleration")
        stock.earnings_surprise_pct = analysis.get("earnings_surprise_pct")
        stock.revenue_growth_pct = analysis.get("revenue_growth_pct")
        stock.quarterly_earnings = analysis.get("quarterly_earnings")
        stock.annual_earnings = analysis.get("annual_earnings")
        stock.quarterly_revenue = analysis.get("quarterly_revenue")

        # Technical analysis
        stock.volume_ratio = analysis.get("volume_ratio")
        stock.weeks_in_base = analysis.get("weeks_in_base")
        stock.base_type = analysis.get("base_type")
        stock.pivot_price = analysis.get("pivot_price")
        stock.is_breaking_out = analysis.get("is_breaking_out", False)
        stock.breakout_volume_ratio = analysis.get("breakout_volume_ratio")

        # Insider trading signals (only update if we have data)
        if analysis.get("insider_sentiment"):
            stock.insider_buy_count = analysis.get("insider_buy_count")
            stock.insider_sell_count = analysis.get("insider_sell_count")
            stock.insider_net_shares = analysis.get("insider_net_shares")
            stock.insider_sentiment = analysis.get("insider_sentiment")
            stock.insider_updated_at = datetime.utcnow()

        # Short interest (only update if we have data)
        if analysis.get("short_interest_pct") is not None:
            stock.short_interest_pct = analysis.get("short_interest_pct")
            stock.short_ratio = analysis.get("short_ratio")
            stock.short_updated_at = datetime.utcnow()

        # Save historical score (one per scan for granular backtesting data)
        today = date.today()
        historical_score = StockScore(
            stock_id=stock.id,
            timestamp=datetime.utcnow(),
            date=today,
            total_score=new_score,
            c_score=analysis.get("c_score"),
            a_score=analysis.get("a_score"),
            n_score=analysis.get("n_score"),
            s_score=analysis.get("s_score"),
            l_score=analysis.get("l_score"),
            i_score=analysis.get("i_score"),
            m_score=analysis.get("m_score"),
            projected_growth=analysis.get("projected_growth"),
            current_price=analysis.get("current_price"),
            week_52_high=analysis.get("week_52_high")
        )
        db.add(historical_score)

        # NOTE: Don't commit here - batched commits happen in the main loop for performance
        return stock

    try:
        # ASYNC SCANNING: Use async batch processing for 10x performance boost
        # Fetches 50 stocks concurrently, then saves to database
        # Performance: ~10 stocks in 3s vs 30-50s with old method
        logger.info(f"Starting ASYNC scan of {len(tickers)} stocks (batch_size=100)...")

        from async_scanner import run_async_scan

        # Progress callback to update frontend in real-time
        def update_progress(current, total):
            _scan_config["stocks_scanned"] = current
            # Also update total if it differs (e.g., from checkpoint resume)
            if total != _scan_config["total_stocks"]:
                _scan_config["total_stocks"] = total
            if current % 100 == 0:  # Log every 100 stocks
                logger.info(f"Progress: {current}/{total} stocks fetched ({current/total*100:.1f}%)")

        # Fetch and analyze all stocks asynchronously (this is the fast part!)
        start_time = time.time()
        analysis_results = run_async_scan(tickers, batch_size=100, progress_callback=update_progress)
        fetch_time = time.time() - start_time

        logger.info(f"✓ Async fetching complete: {len(analysis_results)} stocks in {fetch_time:.1f}s ({fetch_time/len(analysis_results):.2f}s per stock)")

        # Save results to database with BATCHED COMMITS for performance
        # Commit every 50 stocks instead of per-stock (reduces I/O overhead)
        BATCH_SIZE = 50
        db = SessionLocal()
        successful = 0
        for i, analysis in enumerate(analysis_results):
            try:
                save_stock_to_db(db, analysis)
                successful += 1
                # Update progress in real-time
                _scan_config["stocks_scanned"] = successful

                # Batch commit every BATCH_SIZE stocks
                if successful % BATCH_SIZE == 0:
                    db.commit()
                    logger.debug(f"DB batch commit: {successful} stocks saved")
            except Exception as e:
                logger.error(f"Error saving {analysis.get('ticker', 'unknown')}: {e}")
                db.rollback()  # Rollback on error, continue with next batch

        # Final commit for remaining stocks
        try:
            db.commit()
        except Exception as e:
            logger.error(f"Final commit error: {e}")
            db.rollback()

        db.close()

        total_time = time.time() - start_time
        _scan_config["stocks_scanned"] = successful
        logger.info(f"Continuous scan complete: {successful}/{len(tickers)} stocks in {total_time:.1f}s total")
        logger.info(f"Performance: {total_time/successful:.2f}s per stock (10x faster than old method!)")

        # Log rate limit stats and cache stats
        try:
            from data_fetcher import get_rate_limit_stats, reset_rate_limit_stats, get_cache_stats, get_cache_hit_stats
            stats = get_rate_limit_stats()
            error_rate = (stats['errors_429'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            logger.info(f"Rate limit stats: {stats['errors_429']} 429 errors / {stats['total_requests']} requests ({error_rate:.1f}%)")

            cache_stats = get_cache_stats()
            hit_stats = get_cache_hit_stats()
            hit_rate = (hit_stats['hits'] / (hit_stats['hits'] + hit_stats['misses']) * 100) if (hit_stats['hits'] + hit_stats['misses']) > 0 else 0
            logger.info(f"Cache stats: {hit_stats['hits']} hits, {hit_stats['misses']} misses ({hit_rate:.1f}% hit rate)")
            mem_stats = cache_stats.get('memory', cache_stats)  # Handle both nested and flat formats
            logger.info(f"Cache size: {mem_stats.get('tickers_tracked', 0)} tickers, {mem_stats.get('cached_data_entries', 0)} data entries")

            reset_rate_limit_stats()  # Reset for next scan
        except Exception as e:
            logger.error(f"Rate limit stats error: {e}")

        # Update market snapshot (SPY price, MAs, trend)
        try:
            from backend.main import update_market_snapshot
            market_db = SessionLocal()
            update_market_snapshot(market_db)
            market_db.close()
            logger.info("Market snapshot updated")
        except Exception as e:
            logger.error(f"Market snapshot error: {e}")

        # Run AI trading cycle after scan completes
        try:
            from backend.ai_trader import run_ai_trading_cycle, get_or_create_config, take_portfolio_snapshot
            ai_db = SessionLocal()
            config = get_or_create_config(ai_db)
            if config.is_active:
                logger.info("Running AI trading cycle...")
                result = run_ai_trading_cycle(ai_db)
                logger.info(f"AI trading: {len(result.get('buys_executed', []))} buys, {len(result.get('sells_executed', []))} sells")
                # Note: run_ai_trading_cycle already takes a snapshot
            else:
                # Only take snapshot if trading didn't run (trading cycle takes its own)
                take_portfolio_snapshot(ai_db)
            ai_db.close()
        except Exception as e:
            logger.error(f"AI trading error: {e}")

    except Exception as e:
        logger.error(f"Scan error: {e}")
    finally:
        _scan_config["is_scanning"] = False
        _scan_config["last_scan_end"] = datetime.utcnow().isoformat() + 'Z'


def start_continuous_scanning(source: str = "sp500", interval_minutes: int = 15):
    """Start continuous scanning with specified interval"""
    _scan_config["enabled"] = True
    _scan_config["source"] = source
    _scan_config["interval_minutes"] = interval_minutes

    # Remove existing job if any
    if scheduler.get_job("continuous_scan"):
        scheduler.remove_job("continuous_scan")

    # Add the scan job
    scheduler.add_job(
        run_continuous_scan,
        IntervalTrigger(minutes=interval_minutes),
        id="continuous_scan",
        name=f"Continuous CANSLIM Scan ({source})",
        replace_existing=True
    )

    if not scheduler.running:
        scheduler.start()

    logger.info(f"Continuous scanning started: {source} every {interval_minutes} minutes")

    # Run first scan immediately
    from threading import Thread
    Thread(target=run_continuous_scan).start()

    return get_scan_status()


def stop_continuous_scanning():
    """Stop continuous scanning"""
    _scan_config["enabled"] = False

    if scheduler.get_job("continuous_scan"):
        scheduler.remove_job("continuous_scan")

    logger.info("Continuous scanning stopped")
    return get_scan_status()


def update_scan_config(source: str = None, interval_minutes: int = None):
    """Update scan configuration"""
    if source:
        _scan_config["source"] = source
    if interval_minutes:
        _scan_config["interval_minutes"] = interval_minutes

    # If enabled, restart with new config
    if _scan_config["enabled"]:
        return start_continuous_scanning(
            _scan_config["source"],
            _scan_config["interval_minutes"]
        )

    return get_scan_status()
