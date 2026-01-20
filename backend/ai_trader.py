"""
AI Portfolio Trading Logic

Automatically manages a simulated portfolio based on CANSLIM scores.
Makes buy/sell decisions after each scan cycle.
"""

from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from backend.database import (
    Stock, AIPortfolioConfig, AIPortfolioPosition,
    AIPortfolioTrade, AIPortfolioSnapshot
)

logger = logging.getLogger(__name__)


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
    """Update all position prices - fetches live prices by default"""
    import time

    positions = db.query(AIPortfolioPosition).all()
    updated = 0

    for position in positions:
        current_price = None

        # Try to get live price first
        if use_live_prices:
            current_price = fetch_live_price(position.ticker)
            time.sleep(0.5)  # Delay to stay under FMP 300 calls/min limit

        # Fallback to Stock table if live fetch fails
        if not current_price:
            stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
            if stock and stock.current_price:
                current_price = stock.current_price

        if current_price:
            position.current_price = current_price
            position.current_value = position.shares * current_price
            position.gain_loss = position.current_value - (position.shares * position.cost_basis)
            position.gain_loss_pct = ((current_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

            # Update score from Stock table (this doesn't need to be live)
            stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
            if stock:
                position.current_score = stock.canslim_score

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


def execute_trade(db: Session, ticker: str, action: str, shares: float,
                  price: float, reason: str, score: float = None,
                  cost_basis: float = None, realized_gain: float = None):
    """Record a trade in the database"""
    trade = AIPortfolioTrade(
        ticker=ticker,
        action=action,
        shares=shares,
        price=price,
        total_value=shares * price,
        reason=reason,
        canslim_score=score,
        cost_basis=cost_basis,
        realized_gain=realized_gain,
        executed_at=datetime.utcnow()
    )
    db.add(trade)
    logger.info(f"AI Trade: {action} {shares:.2f} shares of {ticker} @ ${price:.2f} - {reason}")
    return trade


def evaluate_sells(db: Session) -> list:
    """Evaluate positions for potential sells - aggressive growth strategy"""
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    sells = []

    for position in positions:
        if not position.current_price or not position.current_score:
            continue

        gain_pct = position.gain_loss_pct or 0
        score = position.current_score or 0
        purchase_score = position.purchase_score or score

        # Stop loss - sell if down more than threshold (tight stops)
        if gain_pct <= -config.stop_loss_pct:
            sells.append({
                "position": position,
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%",
                "priority": 1  # High priority - cut losers fast
            })
            continue

        # Score crashed - sell if score dropped dramatically (>20 points)
        score_drop = purchase_score - score
        if score_drop > 20 and score < 50:
            sells.append({
                "position": position,
                "reason": f"SCORE CRASH: {purchase_score:.0f} → {score:.0f}",
                "priority": 2
            })
            continue

        # For winners, use trailing stop logic
        if gain_pct >= 20:
            # If up 20%+, only sell if score is weak AND gains are fading
            if score < config.sell_score_threshold:
                sells.append({
                    "position": position,
                    "reason": f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({score:.0f})",
                    "priority": 3
                })
            # Take partial profits at 40%+ if score is declining
            elif gain_pct >= config.take_profit_pct and score < purchase_score - 10:
                sells.append({
                    "position": position,
                    "reason": f"TAKE PROFIT: Up {gain_pct:.1f}%, score declining",
                    "priority": 4
                })

        # For losing or flat positions with weak scores - cut and redeploy capital
        elif gain_pct < 10 and score < config.sell_score_threshold:
            sells.append({
                "position": position,
                "reason": f"WEAK POSITION: {gain_pct:+.1f}%, score {score:.0f}",
                "priority": 5
            })

    # Sort by priority (stop losses first, then score crashes, etc.)
    sells.sort(key=lambda x: x["priority"])
    return sells


def evaluate_buys(db: Session) -> list:
    """Evaluate stocks for potential buys - prioritize high growth momentum stocks"""
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

    # Get stocks that meet minimum score threshold
    candidates = db.query(Stock).filter(
        Stock.canslim_score >= config.min_score_to_buy,
        Stock.current_price > 0,
        Stock.projected_growth != None,  # Must have growth projection
        ~Stock.ticker.in_(excluded_tickers) if excluded_tickers else True
    ).all()

    buys = []
    for stock in candidates:
        # Calculate momentum score (how close to 52-week high)
        momentum_score = 0
        if stock.week_52_high and stock.current_price:
            pct_from_high = ((stock.week_52_high - stock.current_price) / stock.week_52_high) * 100
            # Prefer stocks within 15% of 52-week high (showing strength)
            if pct_from_high <= 5:
                momentum_score = 30  # Near highs = strong momentum
            elif pct_from_high <= 10:
                momentum_score = 20
            elif pct_from_high <= 15:
                momentum_score = 10
            elif pct_from_high > 30:
                momentum_score = -10  # Too far from highs, may be in downtrend

        # Calculate composite score: 40% growth, 40% CANSLIM, 20% momentum
        growth_score = min(stock.projected_growth or 0, 50)  # Cap at 50 for scoring
        composite_score = (
            (growth_score * 0.4) +
            (stock.canslim_score * 0.4) +
            (momentum_score * 0.2)
        )

        # Skip if composite score is too low
        if composite_score < 30:
            continue

        # Calculate position size - larger for higher conviction
        portfolio_value = get_portfolio_value(db)["total_value"]

        # Scale position size by conviction (6-12% based on composite score)
        conviction_multiplier = min(composite_score / 60, 1.0)  # 0.5 to 1.0
        position_pct = 6.0 + (conviction_multiplier * 6.0)  # 6% to 12%
        max_position_value = portfolio_value * (position_pct / 100)

        # Don't exceed available cash
        position_value = min(max_position_value, config.current_cash * 0.90)

        if position_value < 100:  # Minimum $100 position
            continue

        shares = position_value / stock.current_price

        reason_parts = []
        if stock.projected_growth and stock.projected_growth > 15:
            reason_parts.append(f"+{stock.projected_growth:.0f}% growth")
        reason_parts.append(f"Score {stock.canslim_score:.0f}")
        if momentum_score >= 20:
            reason_parts.append("Strong momentum")

        buys.append({
            "stock": stock,
            "shares": shares,
            "value": position_value,
            "reason": " | ".join(reason_parts),
            "priority": -composite_score,  # Higher composite = lower priority number (buy first)
            "composite_score": composite_score
        })

    # Sort by composite score (highest first)
    buys.sort(key=lambda x: x["priority"])

    # Log first few buy candidates for debugging
    for b in buys[:3]:
        logger.info(f"Buy candidate: {b['stock'].ticker}, value=${b['value']:.2f}, shares={b['shares']:.2f}")

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

        # Execute the sell
        execute_trade(
            db=db,
            ticker=position.ticker,
            action="SELL",
            shares=position.shares,
            price=position.current_price,
            reason=sell["reason"],
            score=position.current_score,
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
            "reason": sell["reason"]
        })

    db.commit()

    # Evaluate and execute buys (only if we have room for more positions)
    if position_count < config.max_positions:
        logger.info("Evaluating buy candidates from Stock table...")
        buys = evaluate_buys(db)
        results["buys_considered"] = len(buys)
        logger.info(f"Buy candidates found: {len(buys)}")

        if not buys:
            logger.warning("No buy candidates found! Check if Stock table has data with scores >= 65")

        for buy in buys:
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
            actual_value = min(buy["value"], config.current_cash * 0.95)
            if actual_value < 100:
                logger.info(f"{stock.ticker}: Position too small (${actual_value:.2f}), skipping")
                continue

            actual_shares = actual_value / live_price

            if config.current_cash < actual_value:
                logger.info(f"{stock.ticker}: Not enough cash (${config.current_cash:.2f} < ${actual_value:.2f})")
                continue

            # Execute the buy at live price
            execute_trade(
                db=db,
                ticker=stock.ticker,
                action="BUY",
                shares=actual_shares,
                price=live_price,
                reason=buy["reason"],
                score=stock.canslim_score
            )

            # Deduct cash
            config.current_cash -= actual_value

            # Create position at live price
            new_position = AIPortfolioPosition(
                ticker=stock.ticker,
                shares=actual_shares,
                cost_basis=live_price,
                purchase_score=stock.canslim_score,
                current_price=live_price,
                current_value=actual_value,
                gain_loss=0,
                gain_loss_pct=0,
                current_score=stock.canslim_score
            )
            db.add(new_position)
            position_count += 1
            logger.info(f"BOUGHT {stock.ticker}: {actual_shares:.2f} shares @ ${live_price:.2f} = ${actual_value:.2f}")

            results["buys_executed"].append({
                "ticker": stock.ticker,
                "shares": actual_shares,
                "price": live_price,
                "value": actual_value,
                "score": stock.canslim_score,
                "reason": buy["reason"]
            })
    else:
        logger.info(f"Already at max positions ({position_count}), skipping buys")

    db.commit()

    # Take daily snapshot
    take_portfolio_snapshot(db)

    logger.info(f"Trading cycle complete: {len(results['buys_executed'])} buys, {len(results['sells_executed'])} sells")
    return results


def take_portfolio_snapshot(db: Session):
    """Take a snapshot of current portfolio state"""
    today = date.today()

    # Check if we already have a snapshot for today
    existing = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.date == today
    ).first()

    portfolio = get_portfolio_value(db)
    config = get_or_create_config(db)

    # Get yesterday's snapshot for day change calculation
    yesterday_snapshot = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.date < today
    ).order_by(desc(AIPortfolioSnapshot.date)).first()

    day_change = 0
    day_change_pct = 0
    if yesterday_snapshot:
        day_change = portfolio["total_value"] - yesterday_snapshot.total_value
        day_change_pct = (day_change / yesterday_snapshot.total_value) * 100 if yesterday_snapshot.total_value > 0 else 0

    if existing:
        # Update existing snapshot
        existing.total_value = portfolio["total_value"]
        existing.cash = portfolio["cash"]
        existing.positions_value = portfolio["positions_value"]
        existing.positions_count = portfolio["positions_count"]
        existing.total_return = portfolio["total_return"]
        existing.total_return_pct = portfolio["total_return_pct"]
        existing.day_change = day_change
        existing.day_change_pct = day_change_pct
    else:
        # Create new snapshot
        snapshot = AIPortfolioSnapshot(
            date=today,
            total_value=portfolio["total_value"],
            cash=portfolio["cash"],
            positions_value=portfolio["positions_value"],
            positions_count=portfolio["positions_count"],
            total_return=portfolio["total_return"],
            total_return_pct=portfolio["total_return_pct"],
            day_change=day_change,
            day_change_pct=day_change_pct
        )
        db.add(snapshot)

    db.commit()


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
