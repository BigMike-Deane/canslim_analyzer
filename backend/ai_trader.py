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

# Import email utils with fallback for testing
try:
    from email_utils import send_coiled_spring_alert_webhook
except ImportError:
    try:
        from backend.email_utils import send_coiled_spring_alert_webhook
    except ImportError:
        # Function not available (e.g., in tests without backend context)
        def send_coiled_spring_alert_webhook(*args, **kwargs):
            return False

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
from canslim_scorer import calculate_coiled_spring_score, CANSLIMScore, TechnicalAnalyzer

# Import historical data provider for market timing, RS line, VIX, correlation
try:
    from backend.historical_data import HistoricalDataProvider
except ImportError:
    HistoricalDataProvider = None

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
            min_score_to_buy=72,  # CANSLIM quality threshold
            sell_score_threshold=45,  # Hold slightly longer
            take_profit_pct=40.0,  # Let winners run
            stop_loss_pct=8.0,  # O'Neil standard 8% stop
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


def get_portfolio_drawdown(db: Session) -> dict:
    """
    Calculate current drawdown from portfolio high water mark.
    Returns dict with:
    - drawdown_pct: current drawdown from peak (0 = at peak, 15 = down 15%)
    - position_multiplier: 0.5 to 1.0 (reduce positions when in drawdown)
    - high_water_mark: highest portfolio value ever
    - detail: explanation string
    """
    from sqlalchemy import func

    drawdown_config = config.get('ai_trader.drawdown_protection', {})
    if not drawdown_config.get('enabled', True):
        return {
            "drawdown_pct": 0,
            "position_multiplier": 1.0,
            "high_water_mark": 0,
            "detail": "Drawdown protection disabled"
        }

    # Get current portfolio value
    portfolio = get_portfolio_value(db)
    current_value = portfolio["total_value"]

    # Get high water mark from snapshots
    high_water_mark = db.query(func.max(AIPortfolioSnapshot.total_value)).scalar()

    if not high_water_mark or high_water_mark <= 0:
        return {
            "drawdown_pct": 0,
            "position_multiplier": 1.0,
            "high_water_mark": current_value,
            "detail": "No historical data for drawdown"
        }

    # Calculate drawdown
    drawdown_pct = 0
    if current_value < high_water_mark:
        drawdown_pct = ((high_water_mark - current_value) / high_water_mark) * 100

    # Get thresholds from config
    level_1_threshold = drawdown_config.get('level_1_threshold', 10)  # 10% drawdown
    level_2_threshold = drawdown_config.get('level_2_threshold', 15)  # 15% drawdown
    level_1_multiplier = drawdown_config.get('level_1_multiplier', 0.75)
    level_2_multiplier = drawdown_config.get('level_2_multiplier', 0.50)

    # Determine position multiplier based on drawdown level
    if drawdown_pct >= level_2_threshold:
        position_multiplier = level_2_multiplier
        detail = f"Level 2 drawdown: {drawdown_pct:.1f}% from peak (50% smaller positions)"
    elif drawdown_pct >= level_1_threshold:
        position_multiplier = level_1_multiplier
        detail = f"Level 1 drawdown: {drawdown_pct:.1f}% from peak (25% smaller positions)"
    else:
        position_multiplier = 1.0
        detail = f"No significant drawdown ({drawdown_pct:.1f}%)"

    return {
        "drawdown_pct": round(drawdown_pct, 2),
        "position_multiplier": position_multiplier,
        "high_water_mark": high_water_mark,
        "current_value": current_value,
        "detail": detail
    }


def calculate_sector_momentum(db: Session) -> dict:
    """
    Calculate average L score by sector to identify leading/lagging sectors.
    Returns dict with sector -> {avg_l_score, is_leading, is_lagging, stock_count}
    """
    from sqlalchemy import func

    sector_config = config.get('ai_trader.sector_rotation', {})
    if not sector_config.get('enabled', True):
        return {}

    leading_threshold = sector_config.get('leading_threshold', 10)
    lagging_threshold = sector_config.get('lagging_threshold', 5)

    # Calculate average L score by sector
    sector_stats = db.query(
        Stock.sector,
        func.avg(Stock.l_score).label('avg_l'),
        func.count(Stock.id).label('count')
    ).filter(
        Stock.l_score.isnot(None),
        Stock.sector.isnot(None)
    ).group_by(Stock.sector).all()

    result = {}
    for sector, avg_l, count in sector_stats:
        if not sector or count < 5:  # Skip small sectors
            continue

        avg_l_score = float(avg_l) if avg_l else 0
        result[sector] = {
            "avg_l_score": round(avg_l_score, 2),
            "is_leading": avg_l_score >= leading_threshold,
            "is_lagging": avg_l_score < lagging_threshold,
            "stock_count": count
        }

    return result


def get_sector_rotation_bonus(db: Session, sector: str) -> tuple[int, str]:
    """
    Get sector rotation bonus/penalty for a specific sector.
    Returns (bonus_points, detail_string)
    """
    sector_config = config.get('ai_trader.sector_rotation', {})
    if not sector_config.get('enabled', True):
        return 0, ""

    leading_bonus = sector_config.get('leading_bonus', 5)
    lagging_penalty = sector_config.get('lagging_penalty', -3)

    sector_momentum = calculate_sector_momentum(db)
    sector_info = sector_momentum.get(sector, {})

    if sector_info.get("is_leading"):
        return leading_bonus, f"Leading sector (L avg: {sector_info['avg_l_score']:.1f})"
    elif sector_info.get("is_lagging"):
        return lagging_penalty, f"Lagging sector (L avg: {sector_info['avg_l_score']:.1f})"

    return 0, ""


def get_market_regime(db: Session = None) -> dict:
    """
    Determine current market regime based on multi-index weighted signal.
    Returns dict with:
    - regime: "bullish", "neutral", or "bearish"
    - max_position_pct: adjusted max position size (8-15%)
    - min_score_adjustment: adjustment to min_score_to_buy threshold
    - detail: explanation string
    """
    from data_fetcher import get_cached_market_direction

    market_data = get_cached_market_direction()
    regime_config = config.get('ai_trader.market_regime', {})

    if not regime_config.get('enabled', True):
        return {
            "regime": "neutral",
            "max_position_pct": 12.0,
            "min_score_adjustment": 0,
            "detail": "Regime detection disabled"
        }

    if not market_data.get("success"):
        return {
            "regime": "neutral",
            "max_position_pct": 12.0,
            "min_score_adjustment": 0,
            "detail": "No market data available"
        }

    weighted_signal = market_data.get("weighted_signal", 0)

    # Thresholds from config
    bullish_threshold = regime_config.get('bullish_threshold', 1.5)
    bearish_threshold = regime_config.get('bearish_threshold', -0.5)
    bullish_max_pct = regime_config.get('bullish_max_position_pct', 15.0)
    bearish_max_pct = regime_config.get('bearish_max_position_pct', 8.0)
    neutral_max_pct = regime_config.get('neutral_max_position_pct', 12.0)

    if weighted_signal >= bullish_threshold:
        return {
            "regime": "bullish",
            "max_position_pct": bullish_max_pct,
            "min_score_adjustment": -5,  # Slightly lower threshold in bull markets
            "detail": f"Bullish regime (signal: {weighted_signal:.2f})"
        }
    elif weighted_signal <= bearish_threshold:
        return {
            "regime": "bearish",
            "max_position_pct": bearish_max_pct,
            "min_score_adjustment": 5,  # Higher quality required in bear markets
            "detail": f"Bearish regime (signal: {weighted_signal:.2f})"
        }
    else:
        return {
            "regime": "neutral",
            "max_position_pct": neutral_max_pct,
            "min_score_adjustment": 0,
            "detail": f"Neutral regime (signal: {weighted_signal:.2f})"
        }


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
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
    recent = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.ticker == ticker,
        CoiledSpringAlert.created_at >= cutoff
    ).first()

    if recent:
        logger.debug(f"CS cooldown active for {ticker}, skipping")
        return False

    # Price stability check - require minimum price movement to re-alert
    # This prevents duplicate alerts when price hasn't moved significantly
    min_price_change_pct = alerts_config.get('min_price_change_pct', 3)
    most_recent_alert = db.query(CoiledSpringAlert).filter(
        CoiledSpringAlert.ticker == ticker
    ).order_by(CoiledSpringAlert.created_at.desc()).first()

    if most_recent_alert and most_recent_alert.price_at_alert:
        current_price = getattr(stock, 'current_price', 0) or 0
        if current_price > 0 and most_recent_alert.price_at_alert > 0:
            price_change_pct = abs(current_price - most_recent_alert.price_at_alert) / most_recent_alert.price_at_alert * 100
            if price_change_pct < min_price_change_pct:
                logger.debug(f"CS alert skipped for {ticker}: price only moved {price_change_pct:.1f}% (need {min_price_change_pct}%)")
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

    # Send webhook notification if configured
    try:
        webhook_config = config.get('notifications.webhook', {})
        if webhook_config.get('enabled', True):
            send_coiled_spring_alert_webhook(stock, cs_result)
    except Exception as e:
        logger.warning(f"Failed to send CS webhook for {ticker}: {e}")

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


def check_portfolio_correlation(db: Session, ticker: str) -> dict:
    """
    Enhanced correlation check that evaluates both sector AND industry concentration.
    Returns dict with:
    - status: "ok", "sector_concentrated", "industry_concentrated", "both_concentrated"
    - position_multiplier: 0.5 to 1.0 (reduce position for high correlation)
    - detail: explanation string
    """
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        return {"status": "ok", "position_multiplier": 1.0, "detail": "Stock not found"}

    sector = stock.sector if stock.sector else "Unknown"
    industry = stock.industry if stock.industry else "Unknown"

    positions = db.query(AIPortfolioPosition).all()
    if not positions:
        return {"status": "ok", "position_multiplier": 1.0, "detail": "No existing positions"}

    # Batch fetch all stocks in one query
    position_tickers = [pos.ticker for pos in positions]
    position_stocks = db.query(Stock).filter(Stock.ticker.in_(position_tickers)).all()
    ticker_to_stock = {s.ticker: s for s in position_stocks}

    # Count positions by sector and industry
    sector_count = 0
    industry_count = 0
    sector_value = 0
    industry_value = 0
    total_value = sum(pos.current_value or 0 for pos in positions)

    for pos in positions:
        pos_stock = ticker_to_stock.get(pos.ticker)
        if not pos_stock:
            continue

        if pos_stock.sector == sector:
            sector_count += 1
            sector_value += pos.current_value or 0

        if pos_stock.industry == industry and industry != "Unknown":
            industry_count += 1
            industry_value += pos.current_value or 0

    # Determine correlation level and position multiplier
    position_multiplier = 1.0
    issues = []

    # Sector concentration check (more than 30% in same sector or 4+ stocks)
    sector_pct = (sector_value / total_value * 100) if total_value > 0 else 0
    if sector_count >= 4 or sector_pct >= 30:
        position_multiplier *= 0.7  # 30% reduction
        issues.append(f"Sector: {sector_count} stocks, {sector_pct:.0f}%")

    # Industry concentration check (2+ stocks in same industry is risky)
    if industry_count >= 2 and industry != "Unknown":
        industry_pct = (industry_value / total_value * 100) if total_value > 0 else 0
        position_multiplier *= 0.8  # Additional 20% reduction
        issues.append(f"Industry: {industry_count} stocks, {industry_pct:.0f}%")

    # Determine status
    if sector_count >= 4 and industry_count >= 2:
        status = "both_concentrated"
    elif sector_count >= 4 or sector_pct >= 30:
        status = "sector_concentrated"
    elif industry_count >= 2:
        status = "industry_concentrated"
    else:
        status = "ok"

    # Cap minimum multiplier at 0.5
    position_multiplier = max(position_multiplier, 0.5)

    return {
        "status": status,
        "position_multiplier": position_multiplier,
        "detail": "; ".join(issues) if issues else "Low correlation",
        "sector": sector,
        "sector_count": sector_count,
        "industry": industry,
        "industry_count": industry_count
    }


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

    from config_loader import config as yaml_config

    for position in positions:
        # Use effective score based on stock type
        score = get_effective_score(position, use_current=True)

        if not position.current_price or score == 0:
            continue

        gain_pct = position.gain_loss_pct or 0
        current_allocation = (position.current_value or 0) / portfolio_value if portfolio_value > 0 else 0
        current_pyramid_count = getattr(position, 'pyramid_count', 0) or 0

        # Get stock data for additional checks (from batch-fetched dict)
        stock = ticker_to_stock.get(position.ticker)
        if not stock:
            continue

        volume_ratio = getattr(stock, 'volume_ratio', 1.0) or 1.0

        # SCALE-IN ON PULLBACKS: Add to mature winners pulling back on low volume (matches backtester)
        scale_in_config = yaml_config.get('scale_in_pullbacks', {})
        if (scale_in_config.get('enabled', True) and current_pyramid_count >= 2 and
                current_allocation < MAX_POSITION_ALLOCATION and config.current_cash >= 200):
            min_gain = scale_in_config.get('min_gain_pct', 10.0)
            pullback_min = scale_in_config.get('pullback_pct', 3.0)
            pullback_max = scale_in_config.get('max_pullback_pct', 5.0)
            low_vol_ratio = scale_in_config.get('low_volume_ratio', 0.8)
            min_score_pullback = scale_in_config.get('min_score', 70)

            if gain_pct >= min_gain and score >= min_score_pullback:
                peak_price = getattr(position, 'peak_price', 0) or 0
                if peak_price > 0:
                    drop_from_peak = ((peak_price - position.current_price) / peak_price) * 100
                    if pullback_min <= drop_from_peak <= pullback_max and volume_ratio < low_vol_ratio:
                        add_pct = scale_in_config.get('add_pct', 30.0) / 100
                        original_cost = position.shares * position.cost_basis
                        scale_amount = min(original_cost * add_pct, config.current_cash * 0.3)
                        remaining_room = (MAX_POSITION_ALLOCATION - current_allocation) * portfolio_value
                        scale_amount = min(scale_amount, remaining_room)
                        if scale_amount >= 100:
                            pyramids.append({
                                "position": position,
                                "amount": scale_amount,
                                "shares": scale_amount / position.current_price,
                                "reason": f"SCALE-IN PULLBACK: +{gain_pct:.0f}%, -{drop_from_peak:.1f}% from peak, vol {volume_ratio:.1f}x",
                                "priority": -30  # High priority for disciplined entries
                            })
                            continue  # Don't also evaluate as regular pyramid

        # Pyramid threshold: must be up at least 2.5% (matching backtester)
        if gain_pct < 2.5 or score < 70:
            continue

        # Max 2 pyramids per position (O'Neil: decreasing position sizes)
        if current_pyramid_count >= 2:
            continue

        # Skip if already at max position size
        if current_allocation >= MAX_POSITION_ALLOCATION:
            continue

        # Skip if not enough cash
        if config.current_cash < 200:
            continue

        # Prefer stocks that are breaking out or showing accumulation
        is_breaking_out = getattr(stock, 'is_breaking_out', False)

        # O'Neil 60/40 pyramid sizing (decreasing adds)
        # First pyramid: 60% of original cost, Second: 40%
        original_cost = position.shares * position.cost_basis
        if current_pyramid_count == 0:
            pyramid_amount = original_cost * 0.60  # First add: 60%
        else:
            pyramid_amount = original_cost * 0.40  # Second add: 40%

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


def calculate_atr_stop(ticker: str, current_price: float, base_stop_pct: float) -> float:
    """
    Calculate ATR-based adaptive stop loss.
    Volatile stocks get wider stops to avoid being shaken out on normal movement.
    Returns the effective stop loss percentage (always >= base_stop_pct, capped at max_stop_pct).
    """
    from config_loader import config as yaml_config
    stop_config = yaml_config.get('ai_trader.stops', {})

    if not stop_config.get('use_atr_stops', True):
        return base_stop_pct

    atr_multiplier = stop_config.get('atr_multiplier', 2.5)
    max_stop_pct = stop_config.get('max_stop_pct', 20.0)

    # Try to get ATR from recent price data via Yahoo
    try:
        import requests
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "1mo"}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                highs = indicators.get("high", [])
                lows = indicators.get("low", [])
                closes = indicators.get("close", [])

                if len(highs) >= 15 and len(lows) >= 15 and len(closes) >= 15:
                    # Calculate 14-day ATR
                    atr_period = stop_config.get('atr_period', 14)
                    true_ranges = []
                    for i in range(-atr_period, 0):
                        if highs[i] and lows[i] and closes[i-1]:
                            tr = max(
                                highs[i] - lows[i],
                                abs(highs[i] - closes[i-1]),
                                abs(lows[i] - closes[i-1])
                            )
                            true_ranges.append(tr)

                    if true_ranges:
                        atr = sum(true_ranges) / len(true_ranges)
                        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
                        atr_stop = atr_pct * atr_multiplier
                        effective_stop = max(base_stop_pct, atr_stop)
                        effective_stop = min(effective_stop, max_stop_pct)
                        if effective_stop > base_stop_pct:
                            logger.debug(f"{ticker}: ATR stop {effective_stop:.1f}% (ATR={atr:.2f}, base={base_stop_pct}%)")
                        return effective_stop
    except Exception as e:
        logger.debug(f"{ticker}: ATR calculation failed ({e}), using base stop {base_stop_pct}%")

    return base_stop_pct


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

    # Get market condition for market-aware stop losses
    from data_fetcher import get_cached_market_direction
    market_data = get_cached_market_direction()
    is_bearish_market = False
    if market_data.get("success"):
        spy_data = market_data.get("spy", {})
        spy_price = spy_data.get("price", 0)
        spy_ma_50 = spy_data.get("ma_50", 0)
        is_bearish_market = spy_price < spy_ma_50 if spy_price > 0 and spy_ma_50 > 0 else False

    # Get stop loss config
    from config_loader import config as yaml_config
    stop_loss_config = yaml_config.get('ai_trader.stops', {})
    normal_stop_loss_pct = stop_loss_config.get('normal_stop_loss_pct', config.stop_loss_pct)
    bearish_stop_loss_pct = stop_loss_config.get('bearish_stop_loss_pct', 7.0)

    # Use tighter stop loss in bearish market
    effective_stop_loss_pct = bearish_stop_loss_pct if is_bearish_market else normal_stop_loss_pct

    # VIX-regime stop adjustment (matches backtester)
    vix_config = yaml_config.get('vix_stops', {})
    if vix_config.get('enabled', True) and HistoricalDataProvider:
        try:
            from backend.historical_data import HistoricalDataProvider as HDP
            provider = HDP([])
            provider.preload_data(date.today() - timedelta(days=30), date.today())
            vix_proxy = provider.get_vix_proxy(date.today())
            low_vix = vix_config.get('low_vix_threshold', 15)
            high_vix = vix_config.get('high_vix_threshold', 25)
            if vix_proxy < low_vix:
                effective_stop_loss_pct *= vix_config.get('low_vix_stop_tighten', 0.80)
            elif vix_proxy > high_vix:
                effective_stop_loss_pct *= vix_config.get('high_vix_stop_widen', 1.20)
        except Exception as e:
            logger.debug(f"VIX proxy calculation failed in stop check: {e}")

    # Partial trailing stop config (matches backtester)
    partial_trailing_config = yaml_config.get('ai_trader.trailing_stops', {})
    partial_on_trailing = partial_trailing_config.get('partial_on_trailing', True)
    partial_min_pyramid = partial_trailing_config.get('partial_min_pyramid_count', 2)
    partial_min_score = partial_trailing_config.get('partial_min_score', 65)
    partial_sell_pct_config = partial_trailing_config.get('partial_sell_pct', 50)

    for position in positions:
        if not position.current_price:
            continue

        gain_pct = position.gain_loss_pct or 0

        # ATR-based adaptive stop loss - volatile stocks get wider stops
        position_stop_pct = calculate_atr_stop(position.ticker, position.current_price, effective_stop_loss_pct)

        # Market-aware stop loss check (with ATR adaptation)
        if gain_pct <= -position_stop_pct:
            market_note = " (bearish market)" if is_bearish_market else ""
            atr_note = f" (ATR-adjusted {position_stop_pct:.1f}%)" if position_stop_pct > effective_stop_loss_pct else ""
            logger.warning(f"{position.ticker}: STOP LOSS TRIGGERED at {gain_pct:.1f}%{market_note}{atr_note}")

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
                reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}",
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
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}"
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

            # Pyramid-aware trailing stop widening: +2% per pyramid (max +6%)
            pyramid_count = getattr(position, 'pyramid_count', 0) or 0
            if trailing_stop_pct and pyramid_count > 0:
                pyramid_widening = min(pyramid_count * 2, 6)
                trailing_stop_pct += pyramid_widening

            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct:
                stock = ticker_to_stock.get(position.ticker)
                score = stock.canslim_score if stock else 0
                growth_score = stock.growth_mode_score if stock else None

                # Partial trailing stop: high conviction positions sell 50%, reset peak
                if (partial_on_trailing and
                        pyramid_count >= partial_min_pyramid and
                        score >= partial_min_score):
                    shares_to_sell = position.shares * (partial_sell_pct_config / 100)
                    partial_value = shares_to_sell * position.current_price
                    logger.warning(f"{position.ticker}: PARTIAL TRAILING STOP - selling {partial_sell_pct_config}% ({shares_to_sell:.2f} shares)")

                    execute_trade(
                        db=db,
                        ticker=position.ticker,
                        action="SELL",
                        shares=shares_to_sell,
                        price=position.current_price,
                        reason=f"PARTIAL TRAILING STOP ({partial_sell_pct_config}%): Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)",
                        score=score,
                        growth_score=growth_score,
                        is_growth_stock=position.is_growth_stock or False,
                        cost_basis=position.cost_basis,
                        realized_gain=(position.current_price - position.cost_basis) * shares_to_sell
                    )

                    # Update position: reduce shares, reset peak
                    position.shares -= shares_to_sell
                    position.current_value = position.shares * position.current_price
                    position.gain_loss = position.current_value - (position.shares * position.cost_basis)
                    position.peak_price = position.current_price
                    position.partial_profit_taken = (getattr(position, 'partial_profit_taken', 0) or 0) + partial_sell_pct_config
                    config.current_cash += partial_value
                    sells_executed.append({
                        "ticker": position.ticker,
                        "shares": shares_to_sell,
                        "price": position.current_price,
                        "gain_loss": (position.current_price - position.cost_basis) * shares_to_sell,
                        "reason": f"PARTIAL TRAILING STOP ({partial_sell_pct_config}%): Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)"
                    })
                else:
                    # Standard: full sell
                    logger.warning(f"{position.ticker}: TRAILING STOP TRIGGERED - Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)")

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
                  cost_basis: float = None, realized_gain: float = None,
                  signal_factors: dict = None):
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
        executed_at=get_cst_now(),  # Use CST timezone
        signal_factors=signal_factors  # Trade journal: what drove this decision
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

    Includes:
    - Market-aware stop losses (wider in bearish markets)
    - Trailing stop loss (protects gains from peak)
    - Score crash detection with profitability exception
    """
    portfolio_config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    sells = []

    # Batch fetch all stocks in one query (fixes N+1 for score crash checks)
    tickers = [pos.ticker for pos in positions]
    if tickers:
        stocks = db.query(Stock).filter(Stock.ticker.in_(tickers)).all()
        ticker_to_stock = {s.ticker: s for s in stocks}
    else:
        ticker_to_stock = {}

    # Get market condition for market-aware stop losses
    from data_fetcher import get_cached_market_direction
    market_data = get_cached_market_direction()
    is_bearish_market = False
    if market_data.get("success"):
        spy_data = market_data.get("spy", {})
        spy_price = spy_data.get("price", 0)
        spy_ma_50 = spy_data.get("ma_50", 0)
        is_bearish_market = spy_price < spy_ma_50 if spy_price > 0 and spy_ma_50 > 0 else False

    # Get stop loss config from YAML
    from config_loader import config as yaml_config
    stop_loss_config = yaml_config.get('ai_trader.stops', {})
    normal_stop_loss_pct = stop_loss_config.get('normal_stop_loss_pct', portfolio_config.stop_loss_pct)
    bearish_stop_loss_pct = stop_loss_config.get('bearish_stop_loss_pct', 7.0)

    # Partial profit config from YAML
    partial_profit_config = yaml_config.get('ai_trader.partial_profits', {})
    pp_25_gain = partial_profit_config.get('threshold_25pct', {}).get('gain_pct', 25)
    pp_25_sell = partial_profit_config.get('threshold_25pct', {}).get('sell_pct', 25)
    pp_25_min_score = partial_profit_config.get('threshold_25pct', {}).get('min_score', 60)
    pp_40_gain = partial_profit_config.get('threshold_40pct', {}).get('gain_pct', 40)
    pp_40_sell = partial_profit_config.get('threshold_40pct', {}).get('sell_pct', 50)
    pp_40_min_score = partial_profit_config.get('threshold_40pct', {}).get('min_score', 60)

    # Use tighter stop loss in bearish market (7% vs 8% normal)
    effective_stop_loss_pct = bearish_stop_loss_pct if is_bearish_market else normal_stop_loss_pct

    # VIX-regime stop adjustment (matches backtester)
    vix_config = yaml_config.get('vix_stops', {})
    if vix_config.get('enabled', True) and HistoricalDataProvider:
        try:
            from backend.historical_data import HistoricalDataProvider as HDP
            provider = HDP([])
            provider.preload_data(date.today() - timedelta(days=30), date.today())
            vix_proxy = provider.get_vix_proxy(date.today())
            low_vix = vix_config.get('low_vix_threshold', 15)
            high_vix = vix_config.get('high_vix_threshold', 25)
            if vix_proxy < low_vix:
                effective_stop_loss_pct *= vix_config.get('low_vix_stop_tighten', 0.80)
                logger.debug(f"VIX proxy {vix_proxy:.1f} < {low_vix}: tightening stops to {effective_stop_loss_pct:.1f}%")
            elif vix_proxy > high_vix:
                effective_stop_loss_pct *= vix_config.get('high_vix_stop_widen', 1.20)
                logger.debug(f"VIX proxy {vix_proxy:.1f} > {high_vix}: widening stops to {effective_stop_loss_pct:.1f}%")
        except Exception as e:
            logger.debug(f"VIX proxy calculation failed: {e}")

    for position in positions:
        # Use effective score based on stock type
        score = get_effective_score(position, use_current=True)
        purchase_score = get_effective_score(position, use_current=False)

        if not position.current_price:
            continue

        # P0 FIX: When score==0 (data missing), still evaluate stop loss and trailing stop
        # Only skip score-dependent sells (score crash, partial profit, weak position)
        score_available = score > 0

        gain_pct = position.gain_loss_pct or 0
        stock_type = "Growth" if position.is_growth_stock else "CANSLIM"

        # ATR-based adaptive stop loss - volatile stocks get wider stops
        position_stop_pct = calculate_atr_stop(position.ticker, position.current_price, effective_stop_loss_pct)

        # Market-aware stop loss (with ATR adaptation)
        if gain_pct <= -position_stop_pct:
            market_note = " (bearish market)" if is_bearish_market else ""
            atr_note = f" (ATR-adjusted {position_stop_pct:.1f}%)" if position_stop_pct > effective_stop_loss_pct else ""
            sells.append({
                "position": position,
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}",
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

            # Pyramid-aware trailing stop widening: +2% per pyramid (max +6%)
            # High-conviction positions (pyramided) get more room to run
            pyramid_count = getattr(position, 'pyramid_count', 0) or 0
            if trailing_stop_pct and pyramid_count > 0:
                pyramid_widening = min(pyramid_count * 2, 6)
                trailing_stop_pct += pyramid_widening

            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct:
                # Partial trailing stop for high-conviction positions
                # If pyramided 2+ times and score still strong, sell 50% and keep running
                partial_trailing_config = yaml_config.get('ai_trader.trailing_stops', {})
                partial_on_trailing = partial_trailing_config.get('partial_on_trailing', True)
                partial_min_pyramids = partial_trailing_config.get('partial_min_pyramid_count', 2)
                partial_min_score = partial_trailing_config.get('partial_min_score', 65)
                partial_sell_pct_config = partial_trailing_config.get('partial_sell_pct', 50)

                if (partial_on_trailing and pyramid_count >= partial_min_pyramids
                        and score_available and score >= partial_min_score):
                    sells.append({
                        "position": position,
                        "reason": f"PARTIAL TRAILING STOP: Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%), keeping {100-partial_sell_pct_config}%",
                        "priority": 2,
                        "is_partial": True,
                        "sell_pct": partial_sell_pct_config,
                        "reset_peak": True  # Signal to reset peak price after partial sell
                    })
                    logger.info(f"{position.ticker}: Partial trailing stop ({partial_sell_pct_config}%) - "
                               f"pyramids={pyramid_count}, score={score:.0f}")
                    continue

                sells.append({
                    "position": position,
                    "reason": f"TRAILING STOP: Peak ${position.peak_price:.2f} → ${position.current_price:.2f} (-{drop_from_peak:.1f}%)",
                    "priority": 2  # High priority - protect gains
                })
                logger.info(f"{position.ticker}: Trailing stop triggered - peak ${position.peak_price:.2f}, now ${position.current_price:.2f} (-{drop_from_peak:.1f}%)")
                continue

        # Skip all score-dependent sells if score data is missing
        if not score_available:
            logger.debug(f"{position.ticker}: Score=0 (data missing), skipping score-dependent sells but stops still active")
            continue

        # Score crashed - sell if score dropped dramatically
        # WITH safeguards: profitability exception + consecutive requirement
        # Get score crash config
        score_crash_config = yaml_config.get('ai_trader.score_crash', {})
        consecutive_required = score_crash_config.get('consecutive_required', 3)
        score_threshold = score_crash_config.get('threshold', 50)
        drop_required = score_crash_config.get('drop_required', 20)
        ignore_if_profitable_pct = score_crash_config.get('ignore_if_profitable_pct', 10)

        score_drop = purchase_score - score
        if score_drop > drop_required and score < score_threshold:
            # Skip score crash sell if position is profitable enough
            if gain_pct >= ignore_if_profitable_pct:
                logger.debug(f"{position.ticker}: SKIP score crash - profitable (+{gain_pct:.1f}%)")
                continue

            # Get the stock to check data quality and component scores (from batch-fetched dict)
            stock = ticker_to_stock.get(position.ticker)

            # SAFEGUARD: Check score stability - is this a consistent low or a one-time blip?
            stability = check_score_stability(db, position.ticker, score, threshold=score_threshold)

            if not stability["is_stable"]:
                # This looks like a data blip - DON'T SELL, just log warning
                logger.warning(f"{position.ticker}: SKIPPING SELL - {stability['warning']}. "
                               f"Recent scores: {stability['recent_scores']}")
                continue  # Skip this sell, wait for next scan to confirm

            # Require N consecutive low scores before selling (configurable, default 3)
            # Count from most recent — stop at first score above threshold
            recent_scores = stability.get("recent_scores", [])
            consecutive_low = 0
            for s in recent_scores:
                if s < score_threshold:
                    consecutive_low += 1
                else:
                    break
            if consecutive_low < consecutive_required:
                logger.debug(f"{position.ticker}: SKIPPING SELL - only {consecutive_low} "
                            f"consecutive low score(s), need {consecutive_required}+")
                continue

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
                    reason_parts.append(f"(avg: {stability['avg_score']:.0f}, {consecutive_low} low scans)")

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
                        f"recent avg: {stability['avg_score']:.0f}, {consecutive_low} low scans")
            continue

        # PARTIAL PROFIT TAKING - let winners run while locking in gains
        # Thresholds from YAML config
        partial_taken = getattr(position, 'partial_profit_taken', 0) or 0

        # Check for higher tier partial at configured gain threshold
        if gain_pct >= pp_40_gain and score >= pp_40_min_score and partial_taken < pp_40_sell:
            take_pct = pp_40_sell - partial_taken  # Take what's left to get to target
            if take_pct > 0:
                sells.append({
                    "position": position,
                    "reason": f"PARTIAL PROFIT {pp_40_sell}%: Up {gain_pct:.1f}%, score {score:.0f} still strong",
                    "priority": 4,
                    "is_partial": True,
                    "sell_pct": take_pct
                })
                continue  # Don't add more sell signals for this position

        # Check for lower tier partial at configured gain threshold
        elif gain_pct >= pp_25_gain and score >= pp_25_min_score and partial_taken < pp_25_sell:
            sells.append({
                "position": position,
                "reason": f"PARTIAL PROFIT {pp_25_sell}%: Up {gain_pct:.1f}%, score {score:.0f} still strong",
                "priority": 5,
                "is_partial": True,
                "sell_pct": pp_25_sell
            })
            continue  # Don't add more sell signals for this position

        # For winners, use additional score-based logic
        if gain_pct >= 20:
            # If up 20%+, only sell if score is weak AND gains are fading
            if score < portfolio_config.sell_score_threshold:
                sells.append({
                    "position": position,
                    "reason": f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({score:.0f})",
                    "priority": 6
                })
            # Take full profits at 40%+ if score is declining significantly
            elif gain_pct >= portfolio_config.take_profit_pct and score < purchase_score - 15:
                sells.append({
                    "position": position,
                    "reason": f"TAKE PROFIT: Up {gain_pct:.1f}%, score declining significantly",
                    "priority": 7
                })

        # For losing or flat positions with weak scores - cut and redeploy capital
        elif gain_pct < 10 and score < portfolio_config.sell_score_threshold:
            sells.append({
                "position": position,
                "reason": f"WEAK POSITION: {gain_pct:+.1f}%, score {score:.0f}",
                "priority": 6
            })

    # Sort by priority (stop losses first, then trailing stops, etc.)
    sells.sort(key=lambda x: x["priority"])
    return sells


def evaluate_buys(db: Session, ftd_penalty_active: bool = False, heat_penalty_active: bool = False) -> list:
    """
    Evaluate stocks for potential buys - considers both CANSLIM and Growth Mode stocks.
    Uses appropriate score based on stock type for a balanced portfolio.
    """
    portfolio_config = get_or_create_config(db)
    portfolio = get_portfolio_value(db)
    logger.info(f"evaluate_buys: cash=${portfolio_config.current_cash:.2f}, portfolio_value=${portfolio['total_value']:.2f}")

    positions = db.query(AIPortfolioPosition).all()
    current_tickers = {p.ticker for p in positions}

    # Define duplicate ticker groups (same company, different share classes)
    DUPLICATE_TICKERS = [
        {'GOOGL', 'GOOG'},  # Alphabet Class A vs Class C
        # Add more pairs here if needed (e.g., BRK.A/BRK.B)
    ]

    # RE-ENTRY COOLDOWN: Prevent whipsaw losses from rapid re-buys after stops
    # Check recent sells and build cooldown set
    from config_loader import config as yaml_config
    cooldown_config = yaml_config.get('ai_trader.re_entry_cooldown', {})
    stop_loss_cooldown_days = cooldown_config.get('stop_loss_days', 5)
    trailing_stop_cooldown_days = cooldown_config.get('trailing_stop_days', 3)

    cooldown_tickers = set()
    # Query recent SELL trades to check for cooldowns
    cooldown_lookback = datetime.now(timezone.utc) - timedelta(days=max(stop_loss_cooldown_days, trailing_stop_cooldown_days))
    recent_sells = db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.action == "SELL",
        AIPortfolioTrade.executed_at >= cooldown_lookback
    ).all()

    for trade in recent_sells:
        if not trade.reason:
            continue
        days_since = (datetime.now(timezone.utc) - trade.executed_at.replace(tzinfo=timezone.utc)).days if trade.executed_at else 999
        if "STOP LOSS" in trade.reason and days_since < stop_loss_cooldown_days:
            cooldown_tickers.add(trade.ticker)
            logger.debug(f"Cooldown: {trade.ticker} - stop loss {days_since}d ago (need {stop_loss_cooldown_days}d)")
        elif "TRAILING STOP" in trade.reason and "PARTIAL" not in trade.reason and days_since < trailing_stop_cooldown_days:
            cooldown_tickers.add(trade.ticker)
            logger.debug(f"Cooldown: {trade.ticker} - trailing stop {days_since}d ago (need {trailing_stop_cooldown_days}d)")

    if cooldown_tickers:
        logger.info(f"Re-entry cooldown active for: {', '.join(sorted(cooldown_tickers))}")

    # Build set of tickers to exclude (already own or own a duplicate)
    excluded_tickers = set(current_tickers) | cooldown_tickers
    for ticker in current_tickers:
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                excluded_tickers.update(group)  # Exclude all in the group

    # Read min_score from YAML config with DB fallback
    min_score_to_buy = yaml_config.get('ai_trader.allocation.min_score_to_buy', portfolio_config.min_score_to_buy)

    # Get traditional CANSLIM stocks that meet minimum score threshold
    canslim_candidates = db.query(Stock).filter(
        Stock.canslim_score >= min_score_to_buy,
        Stock.current_price > 0,
        Stock.projected_growth != None,  # Must have growth projection
        ~Stock.ticker.in_(excluded_tickers) if excluded_tickers else True
    ).all()

    # Get Growth Mode stocks that meet minimum threshold
    growth_candidates = db.query(Stock).filter(
        Stock.growth_mode_score >= min_score_to_buy,
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

    # REGIME SCORE ADJUSTMENT: Raise the bar in bear markets, lower in bull
    market_regime = get_market_regime(db)
    regime_config = yaml_config.get('ai_trader.market_regime', {})
    regime_adj = 0
    if market_regime["regime"] == "bullish":
        regime_adj = -5
    elif market_regime["regime"] == "bearish":
        regime_adj = regime_config.get('bearish_min_score_adj', 10)

    effective_min_score = min_score_to_buy + regime_adj
    logger.info(f"Regime: {market_regime['regime']}, min_score: {min_score_to_buy} + {regime_adj} = {effective_min_score}")

    # Bear market exception config
    bear_exception_min_cal = regime_config.get('bear_exception_min_cal', 35)
    bear_exception_position_mult = regime_config.get('bear_exception_position_mult', 0.50)

    buys = []
    bear_exception_candidates = []

    for stock in candidates:
        # Determine if this is a growth stock and get effective score
        is_growth = stock.is_growth_stock or False
        effective_score = stock.growth_mode_score if is_growth else stock.canslim_score
        if not effective_score or effective_score < effective_min_score:
            # BEAR MARKET EXCEPTION: Allow strong fundamental stocks at half position size
            if market_regime["regime"] == "bearish" and effective_score and effective_score >= min_score_to_buy:
                score_details = stock.score_details or {}
                c_val = score_details.get('c', {}).get('score', 0) if isinstance(score_details.get('c'), dict) else score_details.get('c', 0)
                a_val = score_details.get('a', {}).get('score', 0) if isinstance(score_details.get('a'), dict) else score_details.get('a', 0)
                l_val = score_details.get('l', {}).get('score', 0) if isinstance(score_details.get('l'), dict) else score_details.get('l', 0)
                cal_sum = c_val + a_val + l_val
                if cal_sum >= bear_exception_min_cal:
                    bear_exception_candidates.append(stock)
                    logger.info(f"Bear exception: {stock.ticker} C+A+L={cal_sum:.0f} (score {effective_score:.0f})")
                    continue
            continue

        # QUALITY FILTERS: Only buy stocks with strong fundamentals
        from config_loader import config as yaml_config
        quality_config = yaml_config.get('ai_trader.quality_filters', {})
        min_c_score = quality_config.get('min_c_score', 10)
        min_l_score = quality_config.get('min_l_score', 8)
        min_volume_ratio = quality_config.get('min_volume_ratio', 1.2)
        skip_growth = quality_config.get('skip_in_growth_mode', True)

        # Get individual scores from score_details
        score_details = stock.score_details or {}
        c_score = score_details.get('c', {}).get('score', 0) if isinstance(score_details.get('c'), dict) else score_details.get('c', 0)
        l_score = score_details.get('l', {}).get('score', 0) if isinstance(score_details.get('l'), dict) else score_details.get('l', 0)
        volume_ratio = getattr(stock, 'volume_ratio', 1.0) or 1.0

        # Skip if not meeting quality thresholds (unless growth stock)
        if not (is_growth and skip_growth):
            if c_score < min_c_score:
                logger.debug(f"Skipping {stock.ticker}: C score {c_score} < {min_c_score}")
                continue
            if l_score < min_l_score:
                logger.debug(f"Skipping {stock.ticker}: L score {l_score} < {min_l_score}")
                continue

        # VOLUME GATE: Context-aware volume thresholds (matches backtester)
        is_breaking_out = getattr(stock, 'is_breaking_out', False)
        vol_gate_config = yaml_config.get('volume_gate', {})
        if vol_gate_config.get('enabled', True):
            if is_breaking_out:
                vol_threshold = vol_gate_config.get('breakout_min_volume_ratio', 1.5)
            elif has_base and pivot_price > 0:
                pct_check = ((pivot_price - stock.current_price) / pivot_price) * 100 if pivot_price > 0 else 0
                if 0 <= pct_check <= 15:
                    vol_threshold = vol_gate_config.get('pre_breakout_min_volume_ratio', 0.8)
                else:
                    vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
            else:
                vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
            if volume_ratio < vol_threshold:
                logger.debug(f"Skipping {stock.ticker}: Volume ratio {volume_ratio:.2f} < {vol_threshold} (volume gate)")
                continue
        elif volume_ratio < min_volume_ratio and not is_breaking_out:
            logger.debug(f"Skipping {stock.ticker}: Volume ratio {volume_ratio:.2f} < {min_volume_ratio}")
            continue

        # Earnings proximity check with Coiled Spring exception
        # CS stocks EMBRACE earnings (catalyst), non-CS stocks AVOID earnings (binary risk)
        days_to_earnings = getattr(stock, 'days_to_earnings', None)
        cs_config = yaml_config.get('coiled_spring', {})
        earnings_config = yaml_config.get('ai_trader.earnings', {})

        # CS-specific settings
        allow_buy_days = cs_config.get('earnings_window', {}).get('allow_buy_days', 7)
        block_days = cs_config.get('earnings_window', {}).get('block_days', 1)

        # Non-CS earnings avoidance settings
        avoidance_days = earnings_config.get('avoidance_days', 5)
        cs_override_enabled = earnings_config.get('cs_override', True)

        # Initialize CS result
        cs_result = None

        if days_to_earnings is not None and 0 < days_to_earnings <= max(allow_buy_days, avoidance_days):
            # Check for Coiled Spring qualification
            cs_result = calculate_coiled_spring_score_for_stock(stock)

            if cs_result["is_coiled_spring"] and days_to_earnings > block_days and cs_override_enabled:
                # ALLOW - high conviction earnings catalyst (CS stocks embrace earnings)
                logger.info(f"COILED SPRING: {stock.ticker} ({cs_result['cs_details']})")
                # Attach CS result for scoring bonus and alert recording
                stock._cs_result = cs_result
                # Record alert (respects daily limits)
                record_coiled_spring_alert(db, stock.ticker, cs_result, stock)
            else:
                # NON-CS: Skip if within avoidance window (binary earnings risk)
                if days_to_earnings <= avoidance_days:
                    logger.info(f"Skipping {stock.ticker}: {days_to_earnings}d to earnings (not CS qualified, avoidance={avoidance_days}d)")
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
            # This is the BEST entry - PREDICTIVE approach before the crowd notices
            if has_base and pivot_price > 0 and 5 <= pct_from_pivot <= 15:
                pre_breakout_bonus = 40  # Highest bonus - ideal entry point
                momentum_score = 35
                if volume_ratio >= 1.3:
                    pre_breakout_bonus += 5  # Accumulation volume bonus
                if weeks_in_base >= 10:
                    pre_breakout_bonus += 5  # Longer base = more stored energy

            # AT PIVOT ZONE: 0-5% below pivot with base pattern (ready to break out)
            elif has_base and pivot_price > 0 and 0 <= pct_from_pivot < 5:
                pre_breakout_bonus = 35  # Strong bonus near pivot
                momentum_score = 30
                if volume_ratio >= 1.5:
                    momentum_score += 5

            # BREAKOUT STOCKS - buying AFTER the pivot point (already moved - less ideal)
            # Once a stock has broken out, the easy money is made - we're late
            elif is_breaking_out:
                breakout_bonus = 10  # Reduced bonus - prefer pre-breakout entries
                if breakout_volume_ratio >= 2.0:
                    breakout_bonus += 5  # Small bonus for strong volume
                momentum_score = 15  # Lower score - already extended

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

        # Get insider signal config
        insider_config = config.get('ai_trader.insider_signals', {})
        cluster_bonus = insider_config.get('cluster_bonus', 8)
        high_value_cluster_bonus = insider_config.get('high_value_cluster_bonus', 12)

        if insider_sentiment == "bullish":
            # INSIDER CLUSTER DETECTION: Multiple insiders buying is stronger signal
            if insider_buy_count >= 3:
                # Cluster of insider buying - very bullish signal
                if insider_net_value >= 1_000_000:  # $1M+ cluster
                    insider_bonus = high_value_cluster_bonus
                else:
                    insider_bonus = cluster_bonus
            elif insider_net_value >= 500000:  # $500K+ net buying
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

        # ACCUMULATION/DISTRIBUTION: Proxy using volume_ratio and L score
        # High volume with rising RS suggests institutional accumulation
        accum_bonus = 0
        l_score = getattr(stock, 'l_score', 0) or 0
        if volume_ratio >= 1.5 and l_score >= 10:
            accum_bonus = 5  # Strong volume with strong RS = accumulation
        elif volume_ratio >= 1.3 and l_score >= 8:
            accum_bonus = 3  # Moderate accumulation signal
        elif volume_ratio >= 2.0 and l_score < 5:
            accum_bonus = -3  # High volume with weak RS = possible distribution

        # SHORT SQUEEZE DETECTION
        # High short interest can be bullish (short squeeze potential) or bearish (smart money betting against)
        # Squeeze setup: high short + strong RS + base pattern + breaking out or pre-breakout
        short_adjustment = 0
        is_squeeze_setup = False
        squeeze_config = config.get('ai_trader.short_squeeze', {})
        squeeze_enabled = squeeze_config.get('enabled', True)
        min_short_pct = squeeze_config.get('min_short_pct', 20)
        min_l_for_squeeze = squeeze_config.get('min_l_score', 10)
        squeeze_bonus = squeeze_config.get('squeeze_bonus', 5)

        if short_interest_pct >= min_short_pct:
            # Check if this is a squeeze OPPORTUNITY (not just risk)
            if squeeze_enabled and l_score >= min_l_for_squeeze and has_base and (is_breaking_out or pre_breakout_bonus >= 15):
                # Squeeze setup: high short with strong technicals = potential squeeze
                short_adjustment = squeeze_bonus
                is_squeeze_setup = True
            else:
                # Just high short interest without setup = risky
                short_adjustment = -5
        elif short_interest_pct > 10:
            short_adjustment = -2  # Elevated short interest = slight caution

        # MOMENTUM CONFIRMATION: Penalize stocks where recent momentum is fading
        # If 3-month RS is significantly weaker than 12-month RS, momentum is weakening
        rs_12m = getattr(stock, 'rs_12m', 1.0) or 1.0
        rs_3m = getattr(stock, 'rs_3m', 1.0) or 1.0
        momentum_penalty = 0
        if rs_12m > 0 and rs_3m < rs_12m * 0.95:
            # Recent momentum fading - apply 15% penalty to composite
            momentum_penalty = -0.15

        # ANALYST REVISION BONUS: Reward stocks where analysts are raising estimates
        estimate_revision_bonus = 0
        revision_pct = getattr(stock, 'eps_estimate_revision_pct', None)
        revision_config = config.get('ai_trader.analyst_revisions', {})
        if revision_pct is not None:
            strong_up_threshold = revision_config.get('strong_up_threshold', 10)
            strong_up_bonus = revision_config.get('strong_up_bonus', 5)
            mod_up_bonus = revision_config.get('mod_up_bonus', 3)
            strong_down_penalty = revision_config.get('strong_down_penalty', -5)
            mod_down_penalty = revision_config.get('mod_down_penalty', -2)

            if revision_pct >= strong_up_threshold:
                estimate_revision_bonus = strong_up_bonus
            elif revision_pct >= 5:
                estimate_revision_bonus = mod_up_bonus
            elif revision_pct <= -10:
                estimate_revision_bonus = strong_down_penalty
            elif revision_pct <= -5:
                estimate_revision_bonus = mod_down_penalty

        # Coiled Spring bonus (from earlier calculation)
        coiled_spring_bonus = 0
        if hasattr(stock, '_cs_result') and stock._cs_result.get('is_coiled_spring'):
            coiled_spring_bonus = stock._cs_result.get('cs_score', 0)

        # SECTOR ROTATION: Bonus for leading sectors, penalty for lagging
        sector = getattr(stock, 'sector', None)
        sector_bonus, sector_detail = get_sector_rotation_bonus(db, sector) if sector else (0, "")

        # RS LINE NEW HIGH: Leading indicator when RS makes new high before price (matches backtester)
        rs_line_bonus = 0
        rs_line_config = yaml_config.get('rs_line', {})
        if rs_line_config.get('enabled', True):
            # Use L score as proxy for RS line in live trader (avoids expensive historical calc)
            # RS new high = L score >= 13 (top quartile) and stock is NOT at price new high
            if l_score >= 13 and stock.week_52_high and stock.current_price:
                pct_from_high_check = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
                if pct_from_high_check > 2:  # RS strong but price hasn't caught up
                    rs_line_bonus = rs_line_config.get('bonus_points', 8)
            elif l_score < 5:  # RS lagging badly
                rs_line_bonus = rs_line_config.get('penalty_divergence', -5)

        # EARNINGS SURPRISE DRIFT: Post-earnings momentum for big beats (matches backtester)
        earnings_drift_bonus = 0
        drift_config = yaml_config.get('earnings_drift', {})
        if drift_config.get('enabled', True):
            beat_streak = getattr(stock, 'earnings_beat_streak', 0) or 0
            if beat_streak >= 4:
                earnings_drift_bonus = drift_config.get('bonus_points', 5)
            elif beat_streak >= 3:
                earnings_drift_bonus = 3

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
            short_adjustment +  # Short squeeze potential (+) or risk (-)
            accum_bonus +  # Accumulation/distribution signal
            estimate_revision_bonus +  # Analyst estimate revisions
            sector_bonus +  # Sector rotation signal
            coiled_spring_bonus +  # Earnings catalyst bonus
            rs_line_bonus +  # RS line new high bonus
            earnings_drift_bonus  # Post-earnings drift bonus
        )

        # Apply momentum penalty after base composite calculation
        if momentum_penalty < 0:
            composite_score *= (1 + momentum_penalty)  # Reduce by 15%

        # FTD penalty: no confirmed uptrend → reduce score (advisory, not blocking)
        if ftd_penalty_active:
            composite_score -= 15

        # Heat penalty: too much risk exposure → reduce score
        if heat_penalty_active:
            composite_score -= 10

        # Skip if composite score is too low
        if composite_score < 25:
            continue

        # Calculate position size - MORE DYNAMIC range based on conviction
        portfolio_value = get_portfolio_value(db)["total_value"]

        # Check correlation - reduce position for highly correlated sectors/industries
        correlation_info = check_portfolio_correlation(db, stock.ticker)
        correlation_multiplier = correlation_info["position_multiplier"]

        # Get market regime for position size limits
        market_regime = get_market_regime(db)
        regime_max_pct = market_regime["max_position_pct"]

        # Get drawdown protection multiplier
        drawdown_info = get_portfolio_drawdown(db)
        drawdown_multiplier = drawdown_info["position_multiplier"]

        # Position sizing: 4-regime_max_pct based on conviction
        # Base: 4% minimum, scale up to regime_max_pct for highest conviction picks
        conviction_multiplier = min(composite_score / 50, 1.5)  # 0.5 to 1.5
        position_pct = 4.0 + (conviction_multiplier * (regime_max_pct - 4) / 1.5)  # Dynamic range

        # Half-size positions when portfolio heat is elevated
        if heat_penalty_active:
            position_pct *= 0.50

        # PREDICTIVE POSITION SIZING: Pre-breakout stocks get largest positions
        # These are the ideal entries - before the crowd notices
        if pre_breakout_bonus >= 35 and has_base:
            position_pct *= 1.40  # 40% larger for best pre-breakout entries
        elif pre_breakout_bonus >= 25 and has_base:
            position_pct *= 1.30  # 30% larger for good pre-breakout entries
        elif is_breaking_out and breakout_volume_ratio >= 1.5:
            position_pct *= 1.0   # No boost - already extended, entry is late

        # Coiled Spring position boost
        if hasattr(stock, '_cs_result') and stock._cs_result.get('is_coiled_spring'):
            cs_multiplier = cs_config.get('position_multiplier', 1.25)
            position_pct *= cs_multiplier

        # Apply correlation penalty
        position_pct *= correlation_multiplier

        # Apply drawdown protection (reduce positions when portfolio is down)
        position_pct *= drawdown_multiplier

        # Cap at market regime max (varies by market conditions)
        position_pct = min(position_pct, regime_max_pct)

        max_position_value = portfolio_value * (position_pct / 100)

        # Don't exceed available cash (allow more for high conviction entries)
        if is_breaking_out:
            cash_limit = portfolio_config.current_cash * 0.85
        elif pre_breakout_bonus >= 15:
            cash_limit = portfolio_config.current_cash * 0.80
        else:
            cash_limit = portfolio_config.current_cash * 0.70
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
            if insider_buy_count >= 3:
                # Insider cluster indicator
                if insider_net_value >= 1_000_000:
                    reason_parts.append(f"👔💰 Insider cluster ({insider_buy_count}, ${insider_net_value/1_000_000:.1f}M)")
                else:
                    reason_parts.append(f"👔 Insider cluster ({insider_buy_count})")
            else:
                reason_parts.append(f"👔 Insiders buying ({insider_buy_count})")
        if short_interest_pct > 15:
            if is_squeeze_setup:
                reason_parts.append(f"⚡ Squeeze ({short_interest_pct:.0f}%)")
            else:
                reason_parts.append(f"⚠️ Short {short_interest_pct:.0f}%")
        if estimate_revision_bonus >= 5:
            reason_parts.append(f"📊 Est↑ {revision_pct:+.0f}%")
        elif estimate_revision_bonus <= -5:
            reason_parts.append(f"📉 Est↓ {revision_pct:+.0f}%")
        if rs_line_bonus > 0:
            reason_parts.append("RS new high")
        if earnings_drift_bonus > 0:
            reason_parts.append(f"Drift +{earnings_drift_bonus}")

        # Build trade journal signal_factors (matches backtester)
        buy_signal_factors = {
            "entry_type": "breakout" if is_breaking_out else ("pre-breakout" if pre_breakout_bonus >= 15 else "standard"),
            "market_regime": market_regime["regime"],
            "rs_line_bonus": rs_line_bonus,
            "earnings_drift_bonus": earnings_drift_bonus,
            "composite_score": round(composite_score, 1),
        }
        if coiled_spring_bonus > 0:
            buy_signal_factors["coiled_spring"] = True

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
            "position_pct": position_pct,
            "signal_factors": buy_signal_factors  # Trade journal
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

    # Add bear market exception candidates at reduced position size
    # These are stocks that meet base min_score but not the regime-adjusted threshold
    if bear_exception_candidates and not final_buys:
        logger.info(f"No regular candidates in bear market, adding {len(bear_exception_candidates)} exception candidates at {bear_exception_position_mult*100:.0f}% position size")
        for stock in bear_exception_candidates[:3]:  # Max 3 exceptions
            is_growth = stock.is_growth_stock or False
            effective_score = stock.growth_mode_score if is_growth else stock.canslim_score
            portfolio_value = get_portfolio_value(db)["total_value"]
            # Half position size for bear exceptions
            position_value = portfolio_value * 0.05 * bear_exception_position_mult  # ~2.5% positions
            position_value = min(position_value, portfolio_config.current_cash * 0.50)
            if position_value < 100 or not stock.current_price:
                continue
            shares = position_value / stock.current_price
            final_buys.append({
                "stock": stock,
                "shares": shares,
                "value": position_value,
                "reason": f"BEAR EXCEPTION: Strong C+A+L at {bear_exception_position_mult*100:.0f}% size | {'Growth' if is_growth else 'CANSLIM'} {effective_score:.0f}",
                "priority": 0,
                "composite_score": effective_score,
                "is_growth_stock": is_growth,
                "effective_score": effective_score,
                "is_breaking_out": False,
                "position_pct": 5.0 * bear_exception_position_mult
            })

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

                # Reset peak price after partial trailing stop (give remainder room to recover)
                if sell.get("reset_peak"):
                    position.peak_price = position.current_price
                    position.peak_date = get_cst_now()
                    logger.info(f"PARTIAL TRAILING STOP {position.ticker}: peak reset to ${position.current_price:.2f}")

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

        # ===== DRAWDOWN CIRCUIT BREAKER (before pyramids and buys) =====
        # Re-fetch positions after sells (some may have been deleted)
        positions = db.query(AIPortfolioPosition).all()
        position_count = len(positions)

        portfolio = get_portfolio_value(db)
        total_value = portfolio["total_value"]

        # Update peak portfolio value
        peak_value = config.peak_portfolio_value or config.starting_cash
        if total_value > peak_value:
            config.peak_portfolio_value = total_value
            peak_value = total_value

        current_drawdown = ((peak_value - total_value) / peak_value) * 100 if peak_value > 0 else 0

        from config_loader import config as yaml_config
        drawdown_config = yaml_config.get('ai_trader.drawdown_protection', {})
        halt_threshold = drawdown_config.get('halt_new_buys_pct', 15.0)
        liquidate_threshold = drawdown_config.get('liquidate_all_pct', 25.0)
        recovery_threshold = drawdown_config.get('recovery_pct', 10.0)
        drawdown_halt = False

        if current_drawdown >= liquidate_threshold and position_count > 0:
            logger.warning(f"CIRCUIT BREAKER: {current_drawdown:.1f}% drawdown >= {liquidate_threshold}% - LIQUIDATING ALL POSITIONS")
            for position in positions:
                if position.current_price and position.current_price > 0:
                    execute_trade(
                        db=db, ticker=position.ticker, action="SELL",
                        shares=position.shares, price=position.current_price,
                        reason=f"CIRCUIT BREAKER: Portfolio drawdown {current_drawdown:.1f}%",
                        score=position.current_score, cost_basis=position.cost_basis,
                        realized_gain=position.gain_loss,
                        is_growth_stock=position.is_growth_stock or False
                    )
                    config.current_cash += position.current_value
                    results["sells_executed"].append({
                        "ticker": position.ticker, "shares": position.shares,
                        "price": position.current_price, "gain_loss": position.gain_loss,
                        "reason": f"CIRCUIT BREAKER: Portfolio drawdown {current_drawdown:.1f}%"
                    })
                    db.delete(position)
            db.commit()
            logger.warning(f"CIRCUIT BREAKER: All positions liquidated. Cash: ${config.current_cash:.2f}")
            return results
        elif current_drawdown >= halt_threshold:
            logger.warning(f"CIRCUIT BREAKER: {current_drawdown:.1f}% drawdown - halting new buys and pyramids")
            drawdown_halt = True
        elif current_drawdown < recovery_threshold:
            # Recovery: only resume if below recovery threshold (conservative)
            drawdown_halt = False

        # Evaluate and execute pyramid trades (add to winners) - blocked by circuit breaker
        results["pyramids_executed"] = []
        if not drawdown_halt:
            pyramids = evaluate_pyramids(db)
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

                # Update position (add shares, recalculate cost basis, increment pyramid count)
                total_cost = (position.shares * position.cost_basis) + actual_value
                position.shares += actual_shares
                position.cost_basis = total_cost / position.shares
                position.current_value = position.shares * live_price
                position.gain_loss = position.current_value - total_cost
                position.gain_loss_pct = ((live_price / position.cost_basis) - 1) * 100
                position.pyramid_count = (getattr(position, 'pyramid_count', 0) or 0) + 1

                results["pyramids_executed"].append({
                    "ticker": position.ticker,
                    "shares_added": actual_shares,
                    "price": live_price,
                    "value": actual_value,
                    "reason": pyramid["reason"]
                })

                logger.info(f"PYRAMID {position.ticker}: +{actual_shares:.2f} shares @ ${live_price:.2f}")

            db.commit()
        else:
            results["pyramids_considered"] = 0
            logger.info("Pyramids skipped - circuit breaker active")

        # Dynamic cash reserves based on market regime (matches backtester)
        # Strong bull: 5%, Bull: 10%, Neutral: 20%, Bear: 40%, Strong bear: 60%
        market_regime = get_market_regime(db)

        from config_loader import config as yaml_config
        alloc_config = yaml_config.get('ai_trader.allocation', {})
        if market_regime["regime"] == "bullish":
            # Check if strong bull (signal >= 2.0) vs regular bull
            from data_fetcher import get_cached_market_direction
            _mkt = get_cached_market_direction()
            _signal = _mkt.get("weighted_signal", 0) if _mkt.get("success") else 0
            if _signal >= 2.0:
                dynamic_reserve_pct = alloc_config.get('cash_reserve_strong_bull', 0.05)
            else:
                dynamic_reserve_pct = alloc_config.get('cash_reserve_bull', 0.10)
        elif market_regime["regime"] == "bearish":
            from data_fetcher import get_cached_market_direction
            _mkt = get_cached_market_direction()
            _signal = _mkt.get("weighted_signal", 0) if _mkt.get("success") else 0
            if _signal <= -1.0:
                dynamic_reserve_pct = alloc_config.get('cash_reserve_strong_bear', 0.60)
            else:
                dynamic_reserve_pct = alloc_config.get('cash_reserve_bear', 0.40)
        else:
            dynamic_reserve_pct = alloc_config.get('cash_reserve_neutral', 0.20)

        min_cash_reserve = portfolio["total_value"] * dynamic_reserve_pct

        # ===== MARKET TIMING: Follow-Through Day check (advisory penalty, matches backtester) =====
        ftd_penalty_active = False
        market_timing_config = yaml_config.get('market_timing', {})
        if market_timing_config.get('follow_through_day', {}).get('enabled', True) and HistoricalDataProvider:
            try:
                from backend.historical_data import HistoricalDataProvider as HDP
                ftd_provider = HDP(['SPY'])
                ftd_provider.preload_data(date.today() - timedelta(days=60), date.today())
                ftd_status = ftd_provider.get_follow_through_day_status(date.today())
                ftd_can_buy = ftd_status.get("can_buy", True)
                if not ftd_can_buy:
                    ftd_penalty_active = True
                    logger.info(f"Market timing: {ftd_status['state']} - applying score penalty to buys")
            except Exception as e:
                logger.debug(f"FTD check failed: {e}")

        # ===== PORTFOLIO HEAT check (advisory penalty, matches backtester) =====
        heat_penalty_active = False
        heat_config = yaml_config.get('portfolio_heat', {})
        if heat_config.get('enabled', True):
            max_heat = heat_config.get('max_heat_pct', 15.0)
            # Calculate heat: sum of (position_pct × distance_to_stop)
            stop_cfg = yaml_config.get('ai_trader.stops', {})
            base_stop = stop_cfg.get('normal_stop_loss_pct', 8.0)
            total_heat = 0.0
            pv = portfolio["total_value"]
            if pv > 0:
                for pos in db.query(AIPortfolioPosition).all():
                    if pos.current_price and pos.current_price > 0 and pos.cost_basis and pos.cost_basis > 0:
                        pos_pct = ((pos.current_value or 0) / pv) * 100
                        g_pct = ((pos.current_price - pos.cost_basis) / pos.cost_basis) * 100
                        dist = base_stop + g_pct
                        total_heat += pos_pct * (dist / 100)
            if total_heat > max_heat:
                heat_penalty_active = True
                logger.info(f"Portfolio heat {total_heat:.1f}% > {max_heat}% - applying score penalty + half-size buys")

        if drawdown_halt:
            logger.warning(f"CIRCUIT BREAKER active ({current_drawdown:.1f}% drawdown) - skipping all buys")
        elif config.current_cash < min_cash_reserve:
            logger.info(f"Cash ${config.current_cash:.2f} below {dynamic_reserve_pct*100:.0f}% dynamic reserve (${min_cash_reserve:.2f}), regime={market_regime['regime']}, skipping buys")
        elif position_count < config.max_positions:
            # Evaluate and execute buys (only if we have room for more positions)
            logger.info("Evaluating buy candidates from Stock table...")
            buys = evaluate_buys(db, ftd_penalty_active=ftd_penalty_active, heat_penalty_active=heat_penalty_active)
            results["buys_considered"] = len(buys)
            logger.info(f"Buy candidates found: {len(buys)}")

            if not buys:
                logger.warning("No buy candidates found! Check if Stock table has data with scores >= 65")

            for buy in buys:
                # Re-check dynamic cash reserve on each buy
                portfolio = get_portfolio_value(db)
                min_cash_reserve = portfolio["total_value"] * dynamic_reserve_pct
                if config.current_cash < min_cash_reserve:
                    logger.info(f"Cash ${config.current_cash:.2f} below {dynamic_reserve_pct*100:.0f}% dynamic reserve, stopping buys")
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
                    is_growth_stock=is_growth,
                    signal_factors=buy.get("signal_factors")
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

    # Create new config - CANSLIM concentrated strategy
    config = AIPortfolioConfig(
        starting_cash=starting_cash,
        current_cash=starting_cash,
        max_positions=20,
        max_position_pct=12.0,
        min_score_to_buy=72,  # CANSLIM quality threshold
        sell_score_threshold=45,
        take_profit_pct=40.0,
        stop_loss_pct=8.0,  # O'Neil standard 8% stop
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
            "min_score": 72,
            "max_positions": 20,
            "stop_loss": "8%",
            "take_profit": "40%",
            "focus": "High growth + momentum stocks"
        }
    }
