"""
AI Portfolio Trading Logic

Automatically manages a simulated portfolio based on CANSLIM scores.
Makes buy/sell decisions after each scan cycle.
"""

from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
import threading
import sys
import os

# Add parent directory to path for config_loader import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config

# Timezone constants - use zoneinfo for proper DST handling
EASTERN_TZ = ZoneInfo("America/New_York")
CENTRAL_TZ = ZoneInfo("America/Chicago")


def get_cst_now():
    """Get current time in Central Time (handles CST/CDT automatically)"""
    return datetime.now(CENTRAL_TZ).replace(tzinfo=None)  # Store without tz for SQLite compatibility


def get_eastern_now():
    """Get current time in Eastern Time (handles EST/EDT automatically)"""
    return datetime.now(EASTERN_TZ)


def is_market_open() -> bool:
    """
    Check if US stock market is currently open.
    Market hours: Monday-Friday, 9:30 AM - 4:00 PM Eastern Time.

    Note: Does not account for market holidays - will return True on holidays
    that fall on weekdays during market hours.

    Returns:
        True if market is open, False otherwise
    """
    # Use zoneinfo for proper DST handling (EST/EDT automatic)
    now = datetime.now(EASTERN_TZ)

    # Weekday check: Monday=0, Friday=4, Saturday=5, Sunday=6
    if now.weekday() > 4:
        return False

    # Market hours: 9:30 AM - 4:00 PM Eastern
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


from backend.database import (
    Stock, AIPortfolioConfig, AIPortfolioPosition,
    AIPortfolioTrade, AIPortfolioSnapshot, StockScore, CoiledSpringAlert
)
from canslim_scorer import calculate_coiled_spring_score, CANSLIMScore

logger = logging.getLogger(__name__)

# Lock to prevent concurrent trading cycles
# Use RLock so we can safely acquire multiple times if needed
_trading_cycle_lock = threading.RLock()
_trading_cycle_started = None  # Track when cycle started for timeout
_trading_cycle_meta_lock = threading.Lock()  # Protects access to _trading_cycle_started

# Minimum cash reserve as percentage of portfolio (stop buying below this)
# Trading allocation limits - loaded from config with fallbacks
MIN_CASH_RESERVE_PCT = config.get('ai_trader.allocation.min_cash_reserve_pct', default=0.10)


def _get_cycle_started():
    """Thread-safe getter for cycle start time"""
    with _trading_cycle_meta_lock:
        return _trading_cycle_started


def _set_cycle_started(value):
    """Thread-safe setter for cycle start time"""
    global _trading_cycle_started
    with _trading_cycle_meta_lock:
        _trading_cycle_started = value


def get_effective_score(stock_or_position, use_current: bool = True) -> float:
    """
    Get the effective score for a stock or position based on its type.
    - Growth stocks use growth_mode_score
    - Traditional stocks use canslim_score

    Args:
        stock_or_position: Stock or AIPortfolioPosition object
        use_current: If True, use current scores; if False, use purchase scores (for positions)

    Returns:
        The appropriate score for this stock's type
    """
    is_growth = getattr(stock_or_position, 'is_growth_stock', False)

    if use_current:
        if is_growth:
            return getattr(stock_or_position, 'current_growth_score', None) or \
                   getattr(stock_or_position, 'growth_mode_score', None) or 0
        else:
            return getattr(stock_or_position, 'current_score', None) or \
                   getattr(stock_or_position, 'canslim_score', None) or 0
    else:
        # For positions, get purchase score
        if is_growth:
            return getattr(stock_or_position, 'purchase_growth_score', None) or 0
        else:
            return getattr(stock_or_position, 'purchase_score', None) or 0


def get_or_create_config(db: Session) -> AIPortfolioConfig:
    """Get or create AI portfolio configuration"""
    config = db.query(AIPortfolioConfig).first()
    if not config:
        config = AIPortfolioConfig(
            starting_cash=25000.0,
            current_cash=25000.0,
            max_positions=20,  # More positions for diversification
            max_position_pct=12.0,  # Larger positions for conviction picks
            min_score_to_buy=65,  # Lower threshold to catch more growth
            sell_score_threshold=45,  # Hold slightly longer
            take_profit_pct=40.0,  # Let winners run
            stop_loss_pct=10.0,  # Cut losers faster
            is_active=True
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


# Sector concentration and correlation settings - loaded from config
MAX_SECTOR_ALLOCATION = config.get('ai_trader.allocation.max_sector_allocation', default=0.30)
MAX_STOCKS_PER_SECTOR = config.get('ai_trader.allocation.max_stocks_per_sector', default=4)
MAX_POSITION_ALLOCATION = config.get('ai_trader.allocation.max_single_position', default=0.15)


def calculate_coiled_spring_score_for_stock(stock: Stock) -> dict:
    """
    Calculate Coiled Spring score for a database Stock object.

    Builds a mock StockData and CANSLIMScore from the Stock model fields
    and calls the calculate_coiled_spring_score function.

    Returns dict with:
    - is_coiled_spring: bool
    - cs_score: float (bonus points)
    - cs_details: str (explanation)
    - allow_pre_earnings_buy: bool
    """
    # Create a simple namespace to hold data fields (simulates StockData)
    class MockData:
        pass

    data = MockData()
    data.weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
    data.earnings_beat_streak = getattr(stock, 'earnings_beat_streak', 0) or 0
    data.days_to_earnings = getattr(stock, 'days_to_earnings', None)
    data.institutional_holders_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

    # Create a mock score object
    class MockScore:
        pass

    score = MockScore()
    score.c_score = getattr(stock, 'c_score', 0) or 0
    score.l_score = getattr(stock, 'l_score', 0) or 0
    score.total_score = getattr(stock, 'canslim_score', 0) or 0

    # Get config
    cs_config = config.get('coiled_spring', {})

    return calculate_coiled_spring_score(data, score, cs_config)


def record_coiled_spring_alert(db: Session, ticker: str, cs_result: dict, stock: Stock) -> bool:
    """
    Record a Coiled Spring alert, respecting daily limits and cooldown.

    Args:
        db: Database session
        ticker: Stock ticker
        cs_result: Result from calculate_coiled_spring_score
        stock: Stock model with current data

    Returns:
        True if alert was recorded, False if limits exceeded
    """
    today = date.today()

    # Get config limits
    cs_config = config.get('coiled_spring', {})
    alerts_config = cs_config.get('alerts', {})
    max_per_day = alerts_config.get('max_per_day', 3)
    cooldown_hours = alerts_config.get('cooldown_hours', 8)

    if not alerts_config.get('enabled', True):
        return False

    # Check daily limit
    today_count = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.alert_date == today
    ).count()

    if today_count >= max_per_day:
        logger.debug(f"CS alert limit reached ({today_count}/{max_per_day}), skipping {ticker}")
        return False

    # Check cooldown for this ticker
    cutoff = datetime.utcnow() - timedelta(hours=cooldown_hours)
    recent = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.ticker == ticker,
        CoiledSpringAlert.created_at >= cutoff
    ).first()

    if recent:
        logger.debug(f"CS cooldown active for {ticker}, skipping")
        return False

    # Record the alert
    alert = CoiledSpringAlert(
        ticker=ticker,
        alert_date=today,
        days_to_earnings=getattr(stock, 'days_to_earnings', None),
        weeks_in_base=getattr(stock, 'weeks_in_base', 0) or 0,
        beat_streak=getattr(stock, 'earnings_beat_streak', 0) or 0,
        c_score=getattr(stock, 'c_score', 0) or 0,
        total_score=getattr(stock, 'canslim_score', 0) or 0,
        cs_bonus=cs_result.get('cs_score', 0),
        price_at_alert=getattr(stock, 'current_price', 0) or 0,
        base_type=getattr(stock, 'base_type', None),
        institutional_pct=(stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0,
        l_score=getattr(stock, 'l_score', 0) or 0,
        email_sent=False
    )

    db.add(alert)
    db.commit()

    logger.info(f"COILED SPRING ALERT: {ticker} - {cs_result.get('cs_details', '')}")
    return True


def check_score_stability(db: Session, ticker: str, current_score: float, threshold: float = 50) -> dict:
    """
    Check if a low score is consistent across recent scans (not a one-time blip).

    Returns:
        dict with:
        - is_stable: True if score has been consistently low (not a blip)
        - recent_scores: list of recent scores
        - avg_score: average of recent scores
        - warning: any warning message
    """
    from datetime import timedelta

    # Get the stock
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        return {"is_stable": True, "recent_scores": [], "avg_score": current_score, "warning": "Stock not found"}

    # Get scores from last 3 scans (roughly last 4-5 hours if scanning every 90 min)
    recent_scores = db.query(StockScore).filter(
        StockScore.stock_id == stock.id
    ).order_by(StockScore.timestamp.desc()).limit(3).all()

    if len(recent_scores) < 2:
        # Not enough history, trust current score
        return {"is_stable": True, "recent_scores": [current_score], "avg_score": current_score, "warning": "Limited history"}

    scores = [s.total_score for s in recent_scores if s.total_score is not None]
    if not scores:
        return {"is_stable": True, "recent_scores": [], "avg_score": current_score, "warning": "No score history"}

    avg_score = sum(scores) / len(scores)

    # Check if current score is significantly lower than average (potential blip)
    score_variance = abs(current_score - avg_score)

    # If current score is much lower than recent average, it might be a blip
    is_blip = current_score < threshold and avg_score > threshold + 10 and score_variance > 15

    if is_blip:
        return {
            "is_stable": False,
            "recent_scores": scores,
            "avg_score": avg_score,
            "warning": f"Possible data blip: current {current_score:.0f} vs avg {avg_score:.0f}"
        }

    return {
        "is_stable": True,
        "recent_scores": scores,
        "avg_score": avg_score,
        "warning": None
    }


def get_sector_allocations(db: Session) -> dict:
    """Calculate current allocation by sector (batch optimized)"""
    positions = db.query(AIPortfolioPosition).all()
    portfolio_value = get_portfolio_value(db)["total_value"]

    if portfolio_value <= 0:
        return {}

    # Batch fetch all stocks in one query (fixes N+1)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_sector = {s.ticker: s.sector for s in stocks}
    else:
        ticker_to_sector = {}

    sector_values = {}
    sector_counts = {}

    for position in positions:
        sector = ticker_to_sector.get(position.ticker) or "Unknown"
        sector_values[sector] = sector_values.get(sector, 0) + (position.current_value or 0)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return {
        "allocations": {s: v / portfolio_value for s, v in sector_values.items()},
        "counts": sector_counts,
        "portfolio_value": portfolio_value
    }


def check_sector_limit(db: Session, ticker: str, buy_amount: float) -> tuple[float, str]:
    """
    Check if a buy would exceed sector limits.
    Returns: (adjusted_amount, reason)
    - If sector limit would be exceeded, reduce the buy amount
    - If already at max stocks in sector, return 0
    """
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    sector = stock.sector if stock and stock.sector else "Unknown"

    sector_info = get_sector_allocations(db)
    portfolio_value = sector_info.get("portfolio_value", 0)
    current_allocation = sector_info.get("allocations", {}).get(sector, 0)
    current_count = sector_info.get("counts", {}).get(sector, 0)

    # Check stock count limit
    if current_count >= MAX_STOCKS_PER_SECTOR:
        return 0, f"Max {MAX_STOCKS_PER_SECTOR} stocks in {sector}"

    # Check allocation limit
    if portfolio_value > 0:
        new_allocation = current_allocation + (buy_amount / portfolio_value)
        if new_allocation > MAX_SECTOR_ALLOCATION:
            # Calculate how much we can still buy
            remaining_room = MAX_SECTOR_ALLOCATION - current_allocation
            if remaining_room <= 0.02:  # Less than 2% room
                return 0, f"Sector {sector} at {current_allocation*100:.0f}% (max {MAX_SECTOR_ALLOCATION*100:.0f}%)"
            adjusted_amount = remaining_room * portfolio_value
            return adjusted_amount, f"Reduced for sector limit"

    return buy_amount, ""


def check_correlation(db: Session, ticker: str) -> tuple[str, str]:
    """
    Check correlation with existing positions.
    Returns: (status, detail)
    - "ok": Low correlation, proceed normally
    - "high_correlation": Already have 3+ stocks in same sector
    """
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    sector = stock.sector if stock and stock.sector else "Unknown"

    sector_info = get_sector_allocations(db)
    current_count = sector_info.get("counts", {}).get(sector, 0)

    if current_count >= 3:
        return "high_correlation", f"Already own {current_count} stocks in {sector}"

    return "ok", ""


def evaluate_pyramids(db: Session) -> list:
    """
    Evaluate existing positions for pyramid opportunities.
    Pyramid = add to a winning position that's still strong.

    Criteria:
    - Position is profitable (5%+ gain)
    - Current effective score is still high (70+)
    - Not already at max position size (15%)
    - Stock is showing accumulation (good volume)
    """
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    portfolio = get_portfolio_value(db)
    portfolio_value = portfolio["total_value"]

    # Batch fetch all stocks in one query (fixes N+1)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_stock = {s.ticker: s for s in stocks}
    else:
        ticker_to_stock = {}

    pyramids = []

    for position in positions:
        # Use effective score based on stock type
        score = get_effective_score(position, use_current=True)

        if not position.current_price or score == 0:
            continue

        gain_pct = position.gain_loss_pct or 0
        current_allocation = (position.current_value or 0) / portfolio_value if portfolio_value > 0 else 0

        # Skip if position is losing or score is weak
        if gain_pct < 5 or score < 70:
            continue

        # Skip if already at max position size
        if current_allocation >= MAX_POSITION_ALLOCATION:
            continue

        # Skip if not enough cash
        if config.current_cash < 200:
            continue

        # Get stock data for additional checks (from batch-fetched dict)
        stock = ticker_to_stock.get(position.ticker)
        if not stock:
            continue

        # Prefer stocks that are breaking out or showing accumulation
        is_breaking_out = getattr(stock, 'is_breaking_out', False)
        volume_ratio = getattr(stock, 'volume_ratio', 1.0) or 1.0

        # Calculate pyramid amount (50% of original position)
        original_cost = position.shares * position.cost_basis
        pyramid_amount = original_cost * 0.5

        # Cap by remaining room in position
        remaining_room = (MAX_POSITION_ALLOCATION - current_allocation) * portfolio_value
        pyramid_amount = min(pyramid_amount, remaining_room, config.current_cash * 0.5)

        if pyramid_amount < 100:
            continue

        # Check sector limits
        adjusted_amount, limit_reason = check_sector_limit(db, position.ticker, pyramid_amount)
        if adjusted_amount < 100:
            continue
        pyramid_amount = adjusted_amount

        # Priority: higher for breakouts and strong volume
        priority = 0
        reason_parts = [f"Winner +{gain_pct:.0f}%", f"Score {score:.0f}"]

        if is_breaking_out:
            priority -= 20
            reason_parts.append("Breakout!")
        if volume_ratio >= 1.5:
            priority -= 10
            reason_parts.append(f"Vol {volume_ratio:.1f}x")

        pyramids.append({
            "position": position,
            "amount": pyramid_amount,
            "shares": pyramid_amount / position.current_price,
            "reason": " | ".join(reason_parts),
            "priority": priority
        })

    # Sort by priority (breakouts first)
    pyramids.sort(key=lambda x: x["priority"])
    return pyramids


def get_portfolio_value(db: Session) -> dict:
    """Calculate current portfolio value"""
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()

    positions_value = sum(p.current_value or 0 for p in positions)
    total_value = config.current_cash + positions_value

    return {
        "cash": config.current_cash,
        "positions_value": positions_value,
        "total_value": total_value,
        "positions_count": len(positions),
        "starting_cash": config.starting_cash,
        "total_return": total_value - config.starting_cash,
        "total_return_pct": ((total_value / config.starting_cash) - 1) * 100 if config.starting_cash > 0 else 0
    }


def fetch_live_price(ticker: str) -> float | None:
    """Fetch current/live price - tries FMP first, then Yahoo as fallback"""
    import requests
    import os

    # Try FMP first (more reliable, you have API key)
    fmp_api_key = os.environ.get('FMP_API_KEY', '')
    if not fmp_api_key:
        logger.warning(f"FMP_API_KEY not set, will use Yahoo for {ticker}")

    if fmp_api_key:
        try:
            url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={fmp_api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    price = data[0].get("price")
                    if price:
                        logger.info(f"FMP live price for {ticker}: ${price}")
                        return float(price)
        except Exception as e:
            logger.warning(f"FMP price error for {ticker}: {e}")

    # Fallback to Yahoo chart API
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "1d"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice") or meta.get("previousClose")
                if price:
                    logger.info(f"Yahoo live price for {ticker}: ${price}")
                    return float(price)
    except Exception as e:
        logger.warning(f"Yahoo price error for {ticker}: {e}")

    return None


def update_position_prices(db: Session, use_live_prices: bool = True):
    """Update all position prices - fetches live prices by default.
    Also tracks peak price for trailing stop loss.
    """
    import time

    positions = db.query(AIPortfolioPosition).all()
    updated = 0

    # Batch fetch all stocks in one query (fixes N+1)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_stock = {s.ticker: s for s in stocks}
    else:
        ticker_to_stock = {}

    for position in positions:
        current_price = None

        # Try to get live price first
        if use_live_prices:
            current_price = fetch_live_price(position.ticker)
            time.sleep(0.5)  # Delay to stay under FMP 300 calls/min limit

        # Fallback to Stock table if live fetch fails (from batch-fetched dict)
        if not current_price:
            stock = ticker_to_stock.get(position.ticker)
            current_price = stock.current_price if stock else None

        if current_price:
            position.current_price = current_price
            position.current_value = position.shares * current_price
            position.gain_loss = position.current_value - (position.shares * position.cost_basis)
            position.gain_loss_pct = ((current_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

            # Track peak price for trailing stop loss
            # Initialize peak_price if not set (new position or migration)
            if position.peak_price is None:
                position.peak_price = max(current_price, position.cost_basis)
                position.peak_date = get_cst_now()
            elif current_price > position.peak_price:
                # New high reached
                position.peak_price = current_price
                position.peak_date = get_cst_now()
                logger.debug(f"{position.ticker}: New peak ${current_price:.2f}")

            # Update scores from Stock table (from batch-fetched dict)
            stock = ticker_to_stock.get(position.ticker)
            if stock:
                position.current_score = stock.canslim_score
                position.current_growth_score = stock.growth_mode_score
                position.is_growth_stock = stock.is_growth_stock or False

            updated += 1

    db.commit()
    return updated


def refresh_ai_portfolio(db: Session) -> dict:
    """Refresh position prices and take snapshot without trading"""
    updated = update_position_prices(db)
    take_portfolio_snapshot(db)
    portfolio = get_portfolio_value(db)

    return {
        "message": f"Refreshed {updated} positions",
        "summary": portfolio
    }


def check_and_execute_stop_losses(db: Session) -> dict:
    """
    Check stop loss conditions and execute sells if triggered.
    This runs independently of the full trading cycle for faster stop loss response.

    Only evaluates:
    - Fixed stop loss (e.g., -10%)
    - Trailing stop loss (protect gains from peak)

    Does NOT evaluate:
    - Score-based sells
    - Take profit sells
    - Partial profit taking
    """
    from backend.database import Stock

    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()

    if not positions:
        return {"message": "No positions to check", "sells_executed": []}

    # First update prices to get current values
    logger.info(f"Checking stop losses for {len(positions)} positions...")
    update_position_prices(db, use_live_prices=True)

    # Batch fetch all stocks in one query (fixes N+1)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_stock = {s.ticker: s for s in stocks}
    else:
        ticker_to_stock = {}

    sells_executed = []

    for position in positions:
        if not position.current_price:
            continue

        gain_pct = position.gain_loss_pct or 0

        # Check fixed stop loss
        if gain_pct <= -config.stop_loss_pct:
            logger.warning(f"{position.ticker}: STOP LOSS TRIGGERED at {gain_pct:.1f}%")

            # Execute the sell (from batch-fetched dict)
            stock = ticker_to_stock.get(position.ticker)
            score = stock.canslim_score if stock else 0
            growth_score = stock.growth_mode_score if stock else None

            execute_trade(
                db=db,
                ticker=position.ticker,
                action="SELL",
                shares=position.shares,
                price=position.current_price,
                reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%",
                score=score,
                growth_score=growth_score,
                is_growth_stock=position.is_growth_stock or False,
                cost_basis=position.cost_basis,
                realized_gain=position.gain_loss
            )

            # Update cash and remove position
            config.current_cash += position.current_value
            sells_executed.append({
                "ticker": position.ticker,
                "shares": position.shares,
                "price": position.current_price,
                "gain_loss": position.gain_loss,
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%"
            })
            db.delete(position)
            continue

        # Check trailing stop loss
        if position.peak_price and position.peak_price > 0:
            drop_from_peak = ((position.peak_price - position.current_price) / position.peak_price) * 100
            peak_gain_pct = ((position.peak_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

            # Dynamic trailing stop thresholds
            trailing_stop_pct = None
            if peak_gain_pct >= 50:
                trailing_stop_pct = 15
            elif peak_gain_pct >= 30:
                trailing_stop_pct = 12
            elif peak_gain_pct >= 20:
                trailing_stop_pct = 10
            elif peak_gain_pct >= 10:
                trailing_stop_pct = 8

            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct:
                logger.warning(f"{position.ticker}: TRAILING STOP TRIGGERED - Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)")

                # Execute the sell (from batch-fetched dict)
                stock = ticker_to_stock.get(position.ticker)
                score = stock.canslim_score if stock else 0
                growth_score = stock.growth_mode_score if stock else None

                execute_trade(
                    db=db,
                    ticker=position.ticker,
                    action="SELL",
                    shares=position.shares,
                    price=position.current_price,
                    reason=f"TRAILING STOP: Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)",
                    score=score,
                    growth_score=growth_score,
                    is_growth_stock=position.is_growth_stock or False,
                    cost_basis=position.cost_basis,
                    realized_gain=position.gain_loss
                )

                # Update cash and remove position
                config.current_cash += position.current_value
                sells_executed.append({
                    "ticker": position.ticker,
                    "shares": position.shares,
                    "price": position.current_price,
                    "gain_loss": position.gain_loss,
                    "reason": f"TRAILING STOP: Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)"
                })
                db.delete(position)

    db.commit()

    if sells_executed:
        logger.info(f"Stop loss check complete: {len(sells_executed)} sells executed")
        take_portfolio_snapshot(db)
    else:
        logger.info("Stop loss check complete: no stops triggered")

    return {
        "message": f"Checked {len(positions)} positions, {len(sells_executed)} stop losses triggered",
        "sells_executed": sells_executed
    }


def execute_trade(db: Session, ticker: str, action: str, shares: float,
                  price: float, reason: str, score: float = None,
                  growth_score: float = None, is_growth_stock: bool = False,
                  cost_basis: float = None, realized_gain: float = None):
    """Record a trade in the database with detailed logging"""
    trade = AIPortfolioTrade(
        ticker=ticker,
        action=action,
        shares=shares,
        price=price,
        total_value=shares * price,
        reason=reason,
        canslim_score=score,
        growth_mode_score=growth_score,
        is_growth_stock=is_growth_stock,
        cost_basis=cost_basis,
        realized_gain=realized_gain,
        executed_at=get_cst_now()  # Use CST timezone
    )
    db.add(trade)

    stock_type = "Growth" if is_growth_stock else "CANSLIM"
    effective = growth_score if is_growth_stock else score

    # Enhanced logging for sells
    if action == "SELL":
        gain_pct = ((price / cost_basis) - 1) * 100 if cost_basis and cost_basis > 0 else 0
        logger.info(f"AI SELL: {ticker} - {shares:.2f} shares @ ${price:.2f} "
                    f"(cost: ${cost_basis:.2f}, gain: {gain_pct:+.1f}%, P/L: ${realized_gain:.2f}) "
                    f"Score: {effective:.0f} - {reason}")
    else:
        logger.info(f"AI {action}: {ticker} - {shares:.2f} shares @ ${price:.2f} "
                    f"({stock_type} {effective:.0f}) - {reason}")

    return trade


def evaluate_sells(db: Session) -> list:
    """Evaluate positions for potential sells - uses effective score based on stock type.

    Includes trailing stop loss logic:
    - Tracks peak price since purchase
    - Triggers sell when price drops X% from peak
    - Uses tighter trailing stops for bigger winners
    """
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    sells = []

    # Batch fetch all stocks in one query (fixes N+1 for score crash checks)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_stock = {s.ticker: s for s in stocks}
    else:
        ticker_to_stock = {}

    for position in positions:
        # Use effective score based on stock type
        score = get_effective_score(position, use_current=True)
        purchase_score = get_effective_score(position, use_current=False)

        if not position.current_price or score == 0:
            continue

        gain_pct = position.gain_loss_pct or 0
        stock_type = "Growth" if position.is_growth_stock else "CANSLIM"

        # Stop loss - sell if down more than threshold (tight stops)
        if gain_pct <= -config.stop_loss_pct:
            sells.append({
                "position": position,
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%",
                "priority": 1  # High priority - cut losers fast
            })
            continue

        # TRAILING STOP LOSS - protect gains from peak
        # Uses dynamic trailing stop based on how much we're up
        if position.peak_price and position.peak_price > 0:
            drop_from_peak = ((position.peak_price - position.current_price) / position.peak_price) * 100
            peak_gain_pct = ((position.peak_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

            # Dynamic trailing stop thresholds based on peak gain:
            # - Up 50%+: 15% trailing stop (protect big winner)
            # - Up 30-50%: 12% trailing stop
            # - Up 20-30%: 10% trailing stop
            # - Up 10-20%: 8% trailing stop
            # - Up 0-10%: No trailing stop (use regular stop loss)
            trailing_stop_pct = None
            if peak_gain_pct >= 50:
                trailing_stop_pct = 15
            elif peak_gain_pct >= 30:
                trailing_stop_pct = 12
            elif peak_gain_pct >= 20:
                trailing_stop_pct = 10
            elif peak_gain_pct >= 10:
                trailing_stop_pct = 8

            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct:
                sells.append({
                    "position": position,
                    "reason": f"TRAILING STOP: Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)",
                    "priority": 2  # High priority - protect gains
                })
                logger.info(f"{position.ticker}: Trailing stop triggered - peak ${position.peak_price:.2f}, now ${position.current_price:.2f} (-{drop_from_peak:.1f}%)")
                continue

        # Score crashed - sell if score dropped dramatically (>20 points)
        # BUT add safeguards against data blips
        score_drop = purchase_score - score
        if score_drop > 20 and score < 50:
            # Get the stock to check data quality and component scores (from batch-fetched dict)
            stock = ticker_to_stock.get(position.ticker)

            # SAFEGUARD: Check score stability - is this a consistent low or a one-time blip?
            stability = check_score_stability(db, position.ticker, score, threshold=50)

            if not stability["is_stable"]:
                # This looks like a data blip - DON'T SELL, just log warning
                logger.warning(f"{position.ticker}: SKIPPING SELL - {stability['warning']}. "
                               f"Recent scores: {stability['recent_scores']}")
                continue  # Skip this sell, wait for next scan to confirm

            # Build detailed reason with component breakdown
            reason_parts = [f"SCORE CRASH: {purchase_score:.0f} → {score:.0f}"]

            if stock:
                # Check data quality - if low confidence, might be a data blip
                data_quality = getattr(stock, 'growth_confidence', 'unknown')

                # Get component scores for debugging
                components = []
                if stock.c_score is not None:
                    components.append(f"C:{stock.c_score:.0f}")
                if stock.a_score is not None:
                    components.append(f"A:{stock.a_score:.0f}")
                if stock.n_score is not None:
                    components.append(f"N:{stock.n_score:.0f}")
                if stock.s_score is not None:
                    components.append(f"S:{stock.s_score:.0f}")
                if stock.l_score is not None:
                    components.append(f"L:{stock.l_score:.0f}")
                if stock.i_score is not None:
                    components.append(f"I:{stock.i_score:.0f}")
                if stock.m_score is not None:
                    components.append(f"M:{stock.m_score:.0f}")

                if components:
                    reason_parts.append(f"[{'/'.join(components)}]")

                # Add recent score history for context
                if stability["recent_scores"]:
                    reason_parts.append(f"(avg: {stability['avg_score']:.0f})")

                # Flag if data quality is low
                if data_quality in ('low', 'unknown', None):
                    reason_parts.append("⚠️ Low data confidence")
                    logger.warning(f"{position.ticker}: Score crash with low data confidence. "
                                   f"Score: {score:.0f}, Components: {components}")

            sells.append({
                "position": position,
                "reason": " ".join(reason_parts),
                "priority": 3
            })
            logger.info(f"{position.ticker}: Score crash confirmed - {purchase_score:.0f} → {score:.0f}, "
                        f"recent avg: {stability['avg_score']:.0f}")
            continue

        # PARTIAL PROFIT TAKING - let winners run while locking in gains
        # Only take partial profits if score remains decent (>= 60)
        partial_taken = getattr(position, 'partial_profit_taken', 0) or 0

        # Check for 50% partial at +40% gain (highest priority partial)
        if gain_pct >= 40 and score >= 60 and partial_taken < 50:
            take_pct = 50 - partial_taken  # Take what's left to get to 50%
            if take_pct > 0:
                sells.append({
                    "position": position,
                    "reason": f"PARTIAL PROFIT 50%: Up {gain_pct:.1f}%, score {score:.0f} still strong",
                    "priority": 4,
                    "is_partial": True,
                    "sell_pct": take_pct
                })
                continue  # Don't add more sell signals for this position

        # Check for 25% partial at +25% gain
        elif gain_pct >= 25 and score >= 60 and partial_taken < 25:
            sells.append({
                "position": position,
                "reason": f"PARTIAL PROFIT 25%: Up {gain_pct:.1f}%, score {score:.0f} still strong",
                "priority": 5,
                "is_partial": True,
                "sell_pct": 25
            })
            continue  # Don't add more sell signals for this position

        # For winners, use additional score-based logic
        if gain_pct >= 20:
            # If up 20%+, only sell if score is weak AND gains are fading
            if score < config.sell_score_threshold:
                sells.append({
                    "position": position,
                    "reason": f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({score:.0f})",
                    "priority": 6
                })
            # Take full profits at 40%+ if score is declining significantly
            elif gain_pct >= config.take_profit_pct and score < purchase_score - 15:
                sells.append({
                    "position": position,
                    "reason": f"TAKE PROFIT: Up {gain_pct:.1f}%, score declining significantly",
                    "priority": 7
                })

        # For losing or flat positions with weak scores - cut and redeploy capital
        elif gain_pct < 10 and score < config.sell_score_threshold:
            sells.append({
                "position": position,
                "reason": f"WEAK POSITION: {gain_pct:+.1f}%, score {score:.0f}",
                "priority": 6
            })

    # Sort by priority (stop losses first, then trailing stops, etc.)
    sells.sort(key=lambda x: x["priority"])
    return sells


def evaluate_buys(db: Session) -> list:
    """
    Evaluate stocks for potential buys - considers both CANSLIM and Growth Mode stocks.
    Uses appropriate score based on stock type for a balanced portfolio.
    """
    config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    logger.info(f"evaluate_buys: cash=${config.current_cash:.2f}, portfolio_value=${portfolio['total_value']:.2f}")

    positions = db.query(AIPortfolioPosition).all()
    current_tickers = {p.ticker for p in positions}

    # Define duplicate ticker groups (same company, different share classes)
    DUPLICATE_TICKERS = [
        {'GOOGL', 'GOOG'},  # Alphabet Class A vs Class C
        # Add more pairs here if needed (e.g., BRK.A/BRK.B)
    ]

    # Build set of tickers to exclude (already own or own a duplicate)
    excluded_tickers = set(current_tickers)
    for ticker in current_tickers:
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                excluded_tickers.update(group)  # Exclude all in the group

    # Get traditional CANSLIM stocks that meet minimum score threshold
    canslim_candidates = db.query(Stock).filter(
        Stock.canslim_score >= config.min_score_to_buy,
        Stock.current_price > 0,
        Stock.projected_growth != None,  # Must have growth projection
        ~Stock.ticker.in_(excluded_tickers) if excluded_tickers else True
    ).all()

    # Get Growth Mode stocks that meet minimum threshold
    growth_candidates = db.query(Stock).filter(
        Stock.growth_mode_score >= config.min_score_to_buy,
        Stock.is_growth_stock == True,
        Stock.current_price > 0,
        ~Stock.ticker.in_(excluded_tickers) if excluded_tickers else True
    ).all()

    # Combine candidates, avoiding duplicates
    seen_tickers = set()
    candidates = []
    for stock in canslim_candidates:
        if stock.ticker not in seen_tickers:
            seen_tickers.add(stock.ticker)
            candidates.append(stock)
    for stock in growth_candidates:
        if stock.ticker not in seen_tickers:
            seen_tickers.add(stock.ticker)
            candidates.append(stock)

    logger.info(f"Buy candidates: {len(canslim_candidates)} CANSLIM, {len(growth_candidates)} Growth Mode, {len(candidates)} total unique")

    buys = []
    for stock in candidates:
        # Determine if this is a growth stock and get effective score
        is_growth = stock.is_growth_stock or False
        effective_score = stock.growth_mode_score if is_growth else stock.canslim_score
        if not effective_score or effective_score < config.min_score_to_buy:
            continue

        # Earnings proximity check with Coiled Spring exception
        days_to_earnings = getattr(stock, 'days_to_earnings', None)
        cs_config = config.get('coiled_spring', {})
        allow_buy_days = cs_config.get('earnings_window', {}).get('allow_buy_days', 7)
        block_days = cs_config.get('earnings_window', {}).get('block_days', 1)

        # Initialize CS result
        cs_result = None

        if days_to_earnings is not None and 0 < days_to_earnings <= allow_buy_days:
            # Check for Coiled Spring qualification
            cs_result = calculate_coiled_spring_score_for_stock(stock)

            if cs_result["is_coiled_spring"] and days_to_earnings > block_days:
                # ALLOW - high conviction earnings catalyst
                logger.info(f"COILED SPRING: {stock.ticker} ({cs_result['cs_details']})")
                # Attach CS result for scoring bonus and alert recording
                stock._cs_result = cs_result
                # Record alert (respects daily limits)
                record_coiled_spring_alert(db, stock.ticker, cs_result, stock)
            else:
                # Standard block for stocks without CS qualification (within 3 days)
                if days_to_earnings <= 3:
                    logger.info(f"Skipping {stock.ticker}: {days_to_earnings}d to earnings (not CS qualified)")
                    continue

        # Check if stock is breaking out (best buying opportunity)
        is_breaking_out = getattr(stock, 'is_breaking_out', False)
        breakout_volume_ratio = getattr(stock, 'breakout_volume_ratio', 1.0) or 1.0
        volume_ratio = getattr(stock, 'volume_ratio', 1.0) or 1.0

        # Base pattern data for pre-breakout detection
        base_type = getattr(stock, 'base_type', 'none') or 'none'
        weeks_in_base = getattr(stock, 'weeks_in_base', 0) or 0
        pivot_price = getattr(stock, 'pivot_price', 0) or 0
        has_base = base_type not in ('none', '', None)

        # Insider trading signals
        insider_sentiment = getattr(stock, 'insider_sentiment', 'neutral') or 'neutral'
        insider_buy_count = getattr(stock, 'insider_buy_count', 0) or 0

        # Short interest data
        short_interest_pct = getattr(stock, 'short_interest_pct', 0) or 0
        short_ratio = getattr(stock, 'short_ratio', 0) or 0

        # Calculate momentum, breakout, pre-breakout, and extended scores
        momentum_score = 0
        breakout_bonus = 0
        pre_breakout_bonus = 0
        extended_penalty = 0
        base_quality_bonus = 0

        # Base pattern quality bonus (up to 15 points)
        if has_base:
            if base_type == "cup_with_handle":
                base_quality_bonus = 10
            elif base_type == "cup":
                base_quality_bonus = 8
            elif base_type == "double_bottom":
                base_quality_bonus = 7
            elif base_type == "flat":
                base_quality_bonus = 6
            # Extra bonus for longer consolidation (max +5)
            if weeks_in_base >= 8:
                base_quality_bonus += 5
            elif weeks_in_base >= 6:
                base_quality_bonus += 3
            elif weeks_in_base >= 5:
                base_quality_bonus += 1

        if stock.week_52_high and stock.current_price:
            pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100

            # Calculate pct_from_pivot if we have a valid pivot price
            pct_from_pivot = 0
            if pivot_price > 0:
                pct_from_pivot = ((pivot_price - stock.current_price) / pivot_price) * 100

            # PRE-BREAKOUT: 5-15% below pivot with valid base pattern
            # This is the BEST entry - optimal risk/reward before the crowd notices
            if has_base and pivot_price > 0 and 5 <= pct_from_pivot <= 15:
                pre_breakout_bonus = 30  # Highest bonus for pre-breakout position
                momentum_score = 30
                if volume_ratio >= 1.3:
                    pre_breakout_bonus += 5  # Accumulation volume bonus

            # AT PIVOT ZONE: 0-5% below pivot with base pattern (ready to break out)
            elif has_base and pivot_price > 0 and 0 <= pct_from_pivot < 5:
                pre_breakout_bonus = 25  # Strong bonus near pivot
                momentum_score = 27
                if volume_ratio >= 1.5:
                    momentum_score += 5

            # BREAKOUT STOCKS - buying after the pivot point (slightly extended)
            elif is_breaking_out:
                breakout_bonus = 20  # Good bonus for confirmed breakouts
                if breakout_volume_ratio >= 2.0:
                    breakout_bonus += 10  # Extra bonus for strong volume breakout
                momentum_score = 25

            # EXTENDED: More than 5% above pivot - the easy money is gone (matches backtester)
            elif has_base and pivot_price > 0 and pct_from_pivot < -5:
                if pct_from_pivot < -10:
                    extended_penalty = -20  # Heavily penalize extended stocks
                    momentum_score = 5
                else:
                    extended_penalty = -10  # Moderate penalty
                    momentum_score = 10

            # NO BASE PATTERN at high = chasing
            elif not has_base and pct_from_high <= 2:
                if effective_score < 85:
                    extended_penalty = -15  # Penalize buying at high without base
                    momentum_score = 5
                else:
                    momentum_score = 12  # Very high score justifies the entry

            # Good zone: 5-15% from high (whether or not has base)
            elif pct_from_high <= 15:
                if volume_ratio >= 1.3:  # Accumulation pattern
                    momentum_score = 18
                else:
                    momentum_score = 12

            elif pct_from_high <= 25:
                momentum_score = 5  # Still acceptable
            else:
                momentum_score = -10  # Too far from highs, may be in downtrend

        # Calculate insider sentiment bonus/penalty (P1 Feature: Scale by $ value)
        insider_bonus = 0
        insider_net_value = getattr(stock, 'insider_net_value', 0) or 0
        insider_largest_buyer_title = getattr(stock, 'insider_largest_buyer_title', '') or ''

        if insider_sentiment == "bullish":
            # Scale bonus by $ value of net buying
            if insider_net_value >= 500000:  # $500K+ net buying
                insider_bonus = 10
            elif insider_net_value >= 100000:  # $100K+ net buying
                insider_bonus = 7
            elif insider_buy_count >= 2:
                insider_bonus = 5  # Fallback to count-based if no value data

            # Extra +3 for C-suite buying (CEO, CFO, COO, President)
            if insider_largest_buyer_title.upper() in ('CEO', 'CFO', 'COO', 'PRESIDENT', 'CHIEF EXECUTIVE OFFICER', 'CHIEF FINANCIAL OFFICER'):
                insider_bonus += 3
        elif insider_sentiment == "bearish":
            insider_bonus = -3  # Insiders selling = caution

        # Calculate short interest adjustment
        # High short interest can be bullish (short squeeze potential) or bearish (smart money betting against)
        # For CANSLIM, we prefer lower short interest (less risk)
        short_penalty = 0
        if short_interest_pct > 20:
            short_penalty = -5  # Very high short interest = risky
        elif short_interest_pct > 10:
            short_penalty = -2  # Elevated short interest = slight caution

        # MOMENTUM CONFIRMATION: Penalize stocks where recent momentum is fading
        # If 3-month RS is significantly weaker than 12-month RS, momentum is weakening
        rs_12m = getattr(stock, 'rs_12m', 1.0) or 1.0
        rs_3m = getattr(stock, 'rs_3m', 1.0) or 1.0
        momentum_penalty = 0
        if rs_12m > 0 and rs_3m < rs_12m * 0.95:
            # Recent momentum fading - apply 15% penalty to composite
            momentum_penalty = -0.15

        # Coiled Spring bonus (from earlier calculation)
        coiled_spring_bonus = 0
        if hasattr(stock, '_cs_result') and stock._cs_result.get('is_coiled_spring'):
            coiled_spring_bonus = stock._cs_result.get('cs_score', 0)

        # Calculate composite score with breakout and pre-breakout weighting
        # 25% growth, 25% score, 20% momentum, 20% breakout/pre-breakout, 10% base quality
        growth_projection = min(stock.projected_growth or 0, 50)  # Cap at 50 for scoring
        composite_score = (
            (growth_projection * 0.25) +
            (effective_score * 0.25) +
            (momentum_score * 0.20) +
            ((breakout_bonus + pre_breakout_bonus) * 0.20) +
            (base_quality_bonus * 0.10) +
            extended_penalty +
            insider_bonus +
            short_penalty +
            coiled_spring_bonus  # Earnings catalyst bonus
        )

        # Apply momentum penalty after base composite calculation
        if momentum_penalty < 0:
            composite_score *= (1 + momentum_penalty)  # Reduce by 15%

        # Skip if composite score is too low
        if composite_score < 25:
            continue

        # Calculate position size - MORE DYNAMIC range based on conviction
        portfolio_value = get_portfolio_value(db)["total_value"]

        # Check correlation - reduce position for highly correlated sectors
        correlation_status, correlation_detail = check_correlation(db, stock.ticker)
        correlation_penalty = 0.7 if correlation_status == "high_correlation" else 1.0

        # Position sizing: 4-20% based on conviction (wider range for more flexibility)
        # Base: 4% minimum, scale up to 20% for highest conviction picks
        conviction_multiplier = min(composite_score / 50, 1.5)  # 0.5 to 1.5
        position_pct = 4.0 + (conviction_multiplier * 10.67)  # 4% to ~20%

        # Pre-breakout stocks get largest positions (optimal entry point)
        # Breakout stocks get smaller boost (slightly extended)
        if pre_breakout_bonus >= 25 and has_base:
            position_pct *= 1.30  # 30% larger for pre-breakout with base (best entry)
        elif pre_breakout_bonus >= 20 and has_base:
            position_pct *= 1.20  # 20% larger for at-pivot entries
        elif is_breaking_out and breakout_volume_ratio >= 1.5:
            position_pct *= 1.15  # 15% larger position for confirmed breakouts

        # Coiled Spring position boost
        if hasattr(stock, '_cs_result') and stock._cs_result.get('is_coiled_spring'):
            cs_multiplier = cs_config.get('position_multiplier', 1.25)
            position_pct *= cs_multiplier

        # Apply correlation penalty
        position_pct *= correlation_penalty

        # Cap at 20% max
        position_pct = min(position_pct, 20.0)

        max_position_value = portfolio_value * (position_pct / 100)

        # Don't exceed available cash (allow more for high conviction entries)
        if is_breaking_out:
            cash_limit = config.current_cash * 0.85
        elif pre_breakout_bonus >= 15:
            cash_limit = config.current_cash * 0.80
        else:
            cash_limit = config.current_cash * 0.70
        position_value = min(max_position_value, cash_limit)

        # Check sector limits
        adjusted_value, sector_reason = check_sector_limit(db, stock.ticker, position_value)
        if adjusted_value < 100:
            continue  # Skip if sector limit would be exceeded
        position_value = adjusted_value

        if position_value < 100:  # Minimum $100 position
            continue

        shares = position_value / stock.current_price

        reason_parts = []
        # Coiled Spring indicator (highest priority)
        if hasattr(stock, '_cs_result') and stock._cs_result.get('is_coiled_spring'):
            days_to_earn = getattr(stock, 'days_to_earnings', 0) or 0
            reason_parts.append(f"🌀 COILED SPRING ({days_to_earn}d to earnings)")
        if is_breaking_out:
            reason_parts.append(f"🚀 BREAKOUT {breakout_volume_ratio:.1f}x vol")
        elif pre_breakout_bonus >= 15:
            reason_parts.append(f"📈 PRE-BREAKOUT ({base_type}) {pct_from_pivot:.0f}% below pivot")
        elif extended_penalty < 0:
            if has_base and pivot_price > 0:
                reason_parts.append(f"⚠️ Extended {abs(pct_from_pivot):.0f}% above pivot")
            else:
                reason_parts.append(f"⚠️ At high without base")
        if stock.projected_growth and stock.projected_growth > 15:
            reason_parts.append(f"+{stock.projected_growth:.0f}% growth")
        stock_type_label = "Growth" if is_growth else "CANSLIM"
        reason_parts.append(f"{stock_type_label} {effective_score:.0f}")
        if has_base and not is_breaking_out and pre_breakout_bonus < 15:
            reason_parts.append(f"Base: {base_type} {weeks_in_base}w")
        if momentum_score >= 20 and not is_breaking_out and pre_breakout_bonus < 15:
            reason_parts.append("Strong momentum")
        if volume_ratio >= 1.5 and not is_breaking_out:
            reason_parts.append(f"Vol {volume_ratio:.1f}x")
        if insider_sentiment == "bullish" and insider_buy_count >= 2:
            reason_parts.append(f"👔 Insiders buying ({insider_buy_count})")
        if short_interest_pct > 15:
            reason_parts.append(f"⚠️ Short {short_interest_pct:.0f}%")

        buys.append({
            "stock": stock,
            "shares": shares,
            "value": position_value,
            "reason": " | ".join(reason_parts),
            "priority": -composite_score,  # Higher composite = lower priority number (buy first)
            "composite_score": composite_score,
            "is_growth_stock": is_growth,
            "effective_score": effective_score,
            "is_breaking_out": is_breaking_out,
            "position_pct": position_pct
        })

    # Sort by composite score (highest first)
    buys.sort(key=lambda x: x["priority"])

    # Log first few buy candidates for debugging
    for b in buys[:5]:
        breakout_flag = " 🚀BREAKOUT" if b.get('is_breaking_out') else ""
        logger.info(f"Buy candidate: {b['stock'].ticker}{breakout_flag}, ${b['value']:.0f} ({b['position_pct']:.1f}%), score={b['composite_score']:.1f}")

    # Remove duplicates from buy candidates (keep only highest scoring from each group)
    final_buys = []
    seen_groups = set()
    for buy in buys:
        ticker = buy["stock"].ticker

        # Check if this ticker belongs to a duplicate group
        in_group = None
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                in_group = frozenset(group)
                break

        if in_group:
            if in_group in seen_groups:
                # Already have a higher-scored ticker from this group, skip
                logger.info(f"Skipping {ticker} - already have another share class from same company")
                continue
            seen_groups.add(in_group)

        final_buys.append(buy)

    return final_buys


def run_ai_trading_cycle(db: Session) -> dict:
    """
    Run a complete AI trading cycle:
    1. Update existing position prices (if any)
    2. Evaluate and execute sells
    3. Evaluate and execute buys (using cached data for decisions, live prices for execution)
    4. Take a portfolio snapshot
    """
    import time

    # Try to acquire lock without blocking
    if not _trading_cycle_lock.acquire(blocking=False):
        # Lock is held - check for timeout using thread-safe accessor
        cycle_started = _get_cycle_started()
        if cycle_started:
            elapsed = (datetime.now() - cycle_started).total_seconds()
            if elapsed < 300:  # 5 minute timeout
                logger.warning(f"Trading cycle already in progress (started {elapsed:.0f}s ago), skipping")
                return {"status": "busy", "message": f"Trading cycle already running ({elapsed:.0f}s elapsed)"}
            else:
                # Timeout - force acquire (blocking) to reset
                logger.warning(f"Previous cycle timed out after {elapsed:.0f}s, forcing new cycle")
                _trading_cycle_lock.acquire(blocking=True)
        else:
            # Lock held but no start time - force acquire
            logger.warning("Lock held but no start time, forcing new cycle")
            _trading_cycle_lock.acquire(blocking=True)

    # Lock acquired - record start time using thread-safe setter
    _set_cycle_started(datetime.now())

    try:
        config = get_or_create_config(db)
        logger.info(f"Starting AI trading cycle. Active: {config.is_active}, Cash: ${config.current_cash:.2f}")

        if not config.is_active:
            logger.info("AI Portfolio is not active, skipping cycle")
            return {"status": "inactive", "message": "AI Portfolio is not active"}

        results = {
            "sells_executed": [],
            "buys_executed": [],
            "sells_considered": 0,
            "buys_considered": 0
        }

        # Get current positions
        positions = db.query(AIPortfolioPosition).all()
        position_count = len(positions)
        logger.info(f"Current positions: {position_count}")

        # Only fetch live prices for existing positions (skip if none)
        if position_count > 0:
            logger.info("Updating existing position prices...")
            update_position_prices(db, use_live_prices=True)

        # Evaluate and execute sells
        sells = evaluate_sells(db)
        results["sells_considered"] = len(sells)
        logger.info(f"Sells to consider: {len(sells)}")

        for sell in sells:
            position = sell["position"]
            is_partial = sell.get("is_partial", False)
            sell_pct = sell.get("sell_pct", 100)

            if is_partial:
                # PARTIAL SELL - only sell a percentage of shares
                shares_to_sell = position.shares * (sell_pct / 100)
                value_to_sell = shares_to_sell * position.current_price
                realized_gain = (position.current_price - position.cost_basis) * shares_to_sell

                # Execute partial sell
                execute_trade(
                    db=db,
                    ticker=position.ticker,
                    action="SELL",
                    shares=shares_to_sell,
                    price=position.current_price,
                    reason=sell["reason"],
                    score=position.current_score,
                    growth_score=position.current_growth_score,
                    is_growth_stock=position.is_growth_stock or False,
                    cost_basis=position.cost_basis,
                    realized_gain=realized_gain
                )

                # Add partial cash back
                config.current_cash += value_to_sell

                # Update position (reduce shares, track partial profit taken)
                position.shares -= shares_to_sell
                position.current_value = position.shares * position.current_price
                position.gain_loss = position.current_value - (position.shares * position.cost_basis)

                # Track cumulative partial profit percentage taken
                current_partial = getattr(position, 'partial_profit_taken', 0) or 0
                position.partial_profit_taken = current_partial + sell_pct

                logger.info(f"PARTIAL SELL {position.ticker}: {sell_pct}% ({shares_to_sell:.2f} shares) @ ${position.current_price:.2f}")

                results["sells_executed"].append({
                    "ticker": position.ticker,
                    "shares": shares_to_sell,
                    "price": position.current_price,
                    "gain_loss": realized_gain,
                    "is_growth_stock": position.is_growth_stock or False,
                    "reason": sell["reason"],
                    "is_partial": True,
                    "remaining_shares": position.shares
                })
            else:
                # FULL SELL - sell entire position
                execute_trade(
                    db=db,
                    ticker=position.ticker,
                    action="SELL",
                    shares=position.shares,
                    price=position.current_price,
                    reason=sell["reason"],
                    score=position.current_score,
                    growth_score=position.current_growth_score,
                    is_growth_stock=position.is_growth_stock or False,
                    cost_basis=position.cost_basis,
                    realized_gain=position.gain_loss
                )

                # Add cash back
                config.current_cash += position.current_value

                # Remove position
                db.delete(position)
                position_count -= 1

                results["sells_executed"].append({
                    "ticker": position.ticker,
                    "shares": position.shares,
                    "price": position.current_price,
                    "gain_loss": position.gain_loss,
                    "is_growth_stock": position.is_growth_stock or False,
                    "reason": sell["reason"]
                })

        db.commit()

        # Evaluate and execute pyramid trades (add to winners)
        pyramids = evaluate_pyramids(db)
        results["pyramids_executed"] = []
        results["pyramids_considered"] = len(pyramids)

        if pyramids:
            logger.info(f"Pyramid opportunities found: {len(pyramids)}")

        for pyramid in pyramids[:3]:  # Limit to 3 pyramids per cycle
            position = pyramid["position"]

            # Fetch live price
            live_price = fetch_live_price(position.ticker)
            if not live_price:
                live_price = position.current_price
            if not live_price or live_price <= 0:
                continue

            time.sleep(0.3)

            # Recalculate with live price
            actual_value = min(pyramid["amount"], config.current_cash * 0.5)
            if actual_value < 100:
                continue

            actual_shares = actual_value / live_price

            if config.current_cash < actual_value:
                continue

            # Execute the pyramid buy with both scores
            execute_trade(
                db=db,
                ticker=position.ticker,
                action="BUY",
                shares=actual_shares,
                price=live_price,
                reason=f"PYRAMID: {pyramid['reason']}",
                score=position.current_score,
                growth_score=position.current_growth_score,
                is_growth_stock=position.is_growth_stock or False
            )

            # Deduct cash
            config.current_cash -= actual_value

            # Update position (add shares, recalculate cost basis)
            total_cost = (position.shares * position.cost_basis) + actual_value
            position.shares += actual_shares
            position.cost_basis = total_cost / position.shares
            position.current_value = position.shares * live_price
            position.gain_loss = position.current_value - total_cost
            position.gain_loss_pct = ((live_price / position.cost_basis) - 1) * 100

            results["pyramids_executed"].append({
                "ticker": position.ticker,
                "shares_added": actual_shares,
                "price": live_price,
                "value": actual_value,
                "reason": pyramid["reason"]
            })

            logger.info(f"PYRAMID {position.ticker}: +{actual_shares:.2f} shares @ ${live_price:.2f}")

        db.commit()

        # Check cash reserve - stop buying if below 10% of portfolio
        portfolio = get_portfolio_value(db)
        min_cash_reserve = portfolio["total_value"] * MIN_CASH_RESERVE_PCT
        if config.current_cash < min_cash_reserve:
            logger.info(f"Cash ${config.current_cash:.2f} below 10% reserve (${min_cash_reserve:.2f}), skipping buys")
        elif position_count < config.max_positions:
            # Evaluate and execute buys (only if we have room for more positions)
            logger.info("Evaluating buy candidates from Stock table...")
            buys = evaluate_buys(db)
            results["buys_considered"] = len(buys)
            logger.info(f"Buy candidates found: {len(buys)}")

            if not buys:
                logger.warning("No buy candidates found! Check if Stock table has data with scores >= 65")

            for buy in buys:
                # Re-check cash reserve on each buy
                portfolio = get_portfolio_value(db)
                min_cash_reserve = portfolio["total_value"] * MIN_CASH_RESERVE_PCT
                if config.current_cash < min_cash_reserve:
                    logger.info(f"Cash ${config.current_cash:.2f} below 10% reserve, stopping buys")
                    break

                if position_count >= config.max_positions:
                    logger.info(f"Max positions ({config.max_positions}) reached, stopping buys")
                    break

                stock = buy["stock"]
                logger.info(f"Processing buy: {stock.ticker} (score: {stock.canslim_score}, cached price: ${stock.current_price})")

                # Fetch live price for this specific stock
                live_price = fetch_live_price(stock.ticker)
                if live_price:
                    logger.info(f"{stock.ticker}: Live price ${live_price:.2f}")
                else:
                    live_price = stock.current_price  # Fallback to cached
                    logger.warning(f"{stock.ticker}: Using cached price ${live_price} (live fetch failed)")

                if not live_price or live_price <= 0:
                    logger.error(f"{stock.ticker}: No valid price, skipping")
                    continue

                time.sleep(0.3)  # Rate limit delay

                # Recalculate position value and shares with live price
                logger.info(f"{stock.ticker}: buy_value=${buy['value']:.2f}, cash=${config.current_cash:.2f}")
                actual_value = min(buy["value"], config.current_cash - min_cash_reserve)  # Leave 10% reserve
                if actual_value < 100:
                    logger.info(f"{stock.ticker}: Position too small (${actual_value:.2f}), skipping")
                    continue

                actual_shares = actual_value / live_price

                if config.current_cash < actual_value:
                    logger.info(f"{stock.ticker}: Not enough cash (${config.current_cash:.2f} < ${actual_value:.2f})")
                    continue

                # Get growth stock info from buy candidate
                is_growth = buy.get("is_growth_stock", False)

                # Execute the buy at live price
                execute_trade(
                    db=db,
                    ticker=stock.ticker,
                    action="BUY",
                    shares=actual_shares,
                    price=live_price,
                    reason=buy["reason"],
                    score=stock.canslim_score,
                    growth_score=stock.growth_mode_score,
                    is_growth_stock=is_growth
                )

                # Deduct cash
                config.current_cash -= actual_value

                # Create position at live price with both scores
                new_position = AIPortfolioPosition(
                    ticker=stock.ticker,
                    shares=actual_shares,
                    cost_basis=live_price,
                    purchase_score=stock.canslim_score,
                    current_price=live_price,
                    current_value=actual_value,
                    gain_loss=0,
                    gain_loss_pct=0,
                    current_score=stock.canslim_score,
                    # Growth Mode fields
                    is_growth_stock=is_growth,
                    purchase_growth_score=stock.growth_mode_score,
                    current_growth_score=stock.growth_mode_score,
                    # Trailing stop loss tracking
                    peak_price=live_price,
                    peak_date=get_cst_now()
                )
                db.add(new_position)
                position_count += 1

                stock_type = "Growth" if is_growth else "CANSLIM"
                effective = buy.get("effective_score", stock.canslim_score)
                logger.info(f"BOUGHT {stock.ticker}: {actual_shares:.2f} shares @ ${live_price:.2f} = ${actual_value:.2f} ({stock_type} {effective:.0f})")

                results["buys_executed"].append({
                    "ticker": stock.ticker,
                    "shares": actual_shares,
                    "price": live_price,
                    "value": actual_value,
                    "canslim_score": stock.canslim_score,
                    "growth_mode_score": stock.growth_mode_score,
                    "is_growth_stock": is_growth,
                    "reason": buy["reason"]
                })
        else:
            logger.info(f"Already at max positions ({position_count}), skipping buys")

        try:
            db.commit()
            logger.info("Commit successful")

            # Verify positions were saved
            saved_positions = db.query(AIPortfolioPosition).count()
            saved_config = db.query(AIPortfolioConfig).first()
            logger.info(f"After commit: {saved_positions} positions, cash=${saved_config.current_cash:.2f}")
        except Exception as e:
            logger.error(f"Commit failed: {e}")
            db.rollback()

        # Take daily snapshot
        take_portfolio_snapshot(db)

        logger.info(f"Trading cycle complete: {len(results['buys_executed'])} buys, {len(results['sells_executed'])} sells")
        return results

    finally:
        # Always release the lock - clear timestamp first using thread-safe setter
        _set_cycle_started(None)
        _trading_cycle_lock.release()
        logger.info("Trading cycle lock released")


def take_portfolio_snapshot(db: Session):
    """Take a snapshot of current portfolio state - called after each scan"""
    from datetime import datetime as dt

    portfolio = get_portfolio_value(db)
    config = get_or_create_config(db)

    # Get previous snapshot for change calculation
    prev_snapshot = db.query(AIPortfolioSnapshot).order_by(
        desc(AIPortfolioSnapshot.timestamp)
    ).first()

    value_change = 0
    value_change_pct = 0
    prev_value = None
    if prev_snapshot:
        prev_value = prev_snapshot.total_value
        value_change = portfolio["total_value"] - prev_snapshot.total_value
        value_change_pct = (value_change / prev_snapshot.total_value) * 100 if prev_snapshot.total_value > 0 else 0

    # Create new snapshot (one per scan)
    snapshot = AIPortfolioSnapshot(
        timestamp=get_cst_now(),  # Use CST timezone
        date=date.today(),
        total_value=portfolio["total_value"],
        cash=portfolio["cash"],
        positions_value=portfolio["positions_value"],
        positions_count=portfolio["positions_count"],
        total_return=portfolio["total_return"],
        total_return_pct=portfolio["total_return_pct"],
        prev_value=prev_value,
        value_change=value_change,
        value_change_pct=value_change_pct
    )
    db.add(snapshot)
    db.commit()

    logger.info(f"Portfolio snapshot taken: ${portfolio['total_value']:.2f} ({portfolio['positions_count']} positions)")


def initialize_ai_portfolio(db: Session, starting_cash: float = 25000.0):
    """Initialize or reset the AI portfolio with aggressive growth settings"""
    # Clear existing data
    db.query(AIPortfolioPosition).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioSnapshot).delete()
    db.query(AIPortfolioConfig).delete()

    # Create new config - aggressive growth strategy
    config = AIPortfolioConfig(
        starting_cash=starting_cash,
        current_cash=starting_cash,
        max_positions=20,  # More diversification
        max_position_pct=12.0,  # Larger conviction positions
        min_score_to_buy=65,  # Lower threshold for more opportunities
        sell_score_threshold=45,  # Hold a bit longer
        take_profit_pct=40.0,  # Let winners run
        stop_loss_pct=10.0,  # Cut losers fast
        is_active=True
    )
    db.add(config)
    db.commit()

    # Take initial snapshot
    take_portfolio_snapshot(db)

    return {
        "message": "AI Portfolio reset. Click 'Run Trading Cycle' to build positions.",
        "starting_cash": starting_cash,
        "strategy": {
            "min_score": 65,
            "max_positions": 20,
            "stop_loss": "10%",
            "take_profit": "40%",
            "focus": "High growth + momentum stocks"
        }
    }
