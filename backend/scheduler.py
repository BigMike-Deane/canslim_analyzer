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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    import random

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

    logger.info(f"Including {len(portfolio_tickers)} portfolio tickers in scan")

    _scan_config["total_stocks"] = len(tickers)
    logger.info(f"Scanning {len(tickers)} stocks...")

    # Import here to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from canslim_scorer import CANSLIMScorer, GrowthModeScorer, TechnicalAnalyzer
    from data_fetcher import DataFetcher
    from growth_projector import GrowthProjector

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
                "is_breaking_out": is_breaking_out,
                "breakout_volume_ratio": breakout_vol if is_breaking_out else None,
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

        if old_score is not None and new_score is not None:
            stock.previous_score = old_score
            stock.score_change = round(new_score - old_score, 2)
        else:
            stock.previous_score = None
            stock.score_change = None

        # Update stock data
        stock.company_name = analysis.get("company_name")
        stock.sector = analysis.get("sector")
        stock.industry = analysis.get("industry")
        stock.current_price = analysis.get("current_price")
        stock.market_cap = analysis.get("market_cap")
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

        # Technical analysis
        stock.volume_ratio = analysis.get("volume_ratio")
        stock.weeks_in_base = analysis.get("weeks_in_base")
        stock.base_type = analysis.get("base_type")
        stock.is_breaking_out = analysis.get("is_breaking_out", False)
        stock.breakout_volume_ratio = analysis.get("breakout_volume_ratio")

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

        db.commit()
        return stock

    # Thread-safe counters
    import threading
    counter_lock = threading.Lock()
    counters = {"processed": 0, "successful": 0}

    def process_single_stock(ticker):
        """Process a single stock with rate limiting"""
        # Delay to stay under FMP's 300 calls/min limit (4 calls/stock)
        # Using 2.5-4.0s with 4 workers = ~60-90 stocks/min, well under limit
        time.sleep(random.uniform(2.5, 4.0))

        try:
            thread_db = SessionLocal()
            analysis = analyze_stock(ticker)
            if analysis:
                save_stock_to_db(thread_db, analysis)
                with counter_lock:
                    counters["successful"] += 1
                    counters["processed"] += 1
                    # Update progress in real-time
                    _scan_config["stocks_scanned"] = counters["successful"]
            else:
                with counter_lock:
                    counters["processed"] += 1
            thread_db.close()
            return ticker, True
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            with counter_lock:
                counters["processed"] += 1
            return ticker, False

    try:
        # Use 4 workers to stay within rate limits (reduces 429 errors)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_single_stock, t): t for t in tickers}

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Thread error for {ticker}: {e}")

        _scan_config["stocks_scanned"] = counters["successful"]
        logger.info(f"Continuous scan complete: {counters['successful']}/{len(tickers)} stocks")

        # Log rate limit stats
        try:
            from data_fetcher import get_rate_limit_stats, reset_rate_limit_stats
            stats = get_rate_limit_stats()
            error_rate = (stats['errors_429'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0
            logger.info(f"Rate limit stats: {stats['errors_429']} 429 errors / {stats['total_requests']} requests ({error_rate:.1f}%)")
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
