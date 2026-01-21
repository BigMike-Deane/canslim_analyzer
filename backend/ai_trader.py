"""
AI Portfolio Trading Logic

Automatically manages a simulated portfolio based on CANSLIM scores.
Makes buy/sell decisions after each scan cycle.
"""

from datetime import datetime, date, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging


def get_cst_now():
    """Get current time in CST (UTC-6)"""
    cst = timezone(timedelta(hours=-6))
    return datetime.now(cst).replace(tzinfo=None)  # Store without tz for SQLite compatibility

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


# Sector concentration and correlation settings
MAX_SECTOR_ALLOCATION = 0.30  # 30% max per sector
MAX_STOCKS_PER_SECTOR = 4  # Max stocks in same sector
MAX_POSITION_ALLOCATION = 0.15  # 15% max single position (for pyramiding)


def get_sector_allocations(db: Session) -> dict:
    """Calculate current allocation by sector"""
    positions = db.query(AIPortfolioPosition).all()
    portfolio_value = get_portfolio_value(db)["total_value"]

    if portfolio_value <= 0:
        return {}

    sector_values = {}
    sector_counts = {}

    for position in positions:
        stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
        sector = stock.sector if stock and stock.sector else "Unknown"

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
    - Current score is still high (70+)
    - Not already at max position size (15%)
    - Stock is showing accumulation (good volume)
    """
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    portfolio = get_portfolio_value(db)
    portfolio_value = portfolio["total_value"]

    pyramids = []

    for position in positions:
        if not position.current_price or not position.current_score:
            continue

        gain_pct = position.gain_loss_pct or 0
        score = position.current_score or 0
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

        # Get stock data for additional checks
        stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
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
        executed_at=get_cst_now()  # Use CST timezone
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

        # Check correlation - reduce position for highly correlated sectors
        correlation_status, correlation_detail = check_correlation(db, stock.ticker)
        correlation_penalty = 0.7 if correlation_status == "high_correlation" else 1.0

        # Scale position size by conviction (6-12% based on composite score)
        conviction_multiplier = min(composite_score / 60, 1.0)  # 0.5 to 1.0
        position_pct = 6.0 + (conviction_multiplier * 6.0)  # 6% to 12%
        position_pct *= correlation_penalty  # Reduce for high correlation
        max_position_value = portfolio_value * (position_pct / 100)

        # Don't exceed available cash
        position_value = min(max_position_value, config.current_cash * 0.90)

        # Check sector limits
        adjusted_value, sector_reason = check_sector_limit(db, stock.ticker, position_value)
        if adjusted_value < 100:
            continue  # Skip if sector limit would be exceeded
        position_value = adjusted_value

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

        # Execute the pyramid buy
        execute_trade(
            db=db,
            ticker=position.ticker,
            action="BUY",
            shares=actual_shares,
            price=live_price,
            reason=f"PYRAMID: {pyramid['reason']}",
            score=position.current_score
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
