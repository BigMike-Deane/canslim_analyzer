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
            max_positions=15,
            max_position_pct=10.0,
            min_score_to_buy=75,
            sell_score_threshold=50,
            take_profit_pct=25.0,
            stop_loss_pct=15.0,
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


def update_position_prices(db: Session):
    """Update all position prices from latest stock data"""
    positions = db.query(AIPortfolioPosition).all()

    for position in positions:
        stock = db.query(Stock).filter(Stock.ticker == position.ticker).first()
        if stock and stock.current_price:
            position.current_price = stock.current_price
            position.current_value = position.shares * stock.current_price
            position.gain_loss = position.current_value - (position.shares * position.cost_basis)
            position.gain_loss_pct = ((position.current_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0
            position.current_score = stock.canslim_score

    db.commit()


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
    """Evaluate positions for potential sells"""
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    sells = []

    for position in positions:
        if not position.current_price or not position.current_score:
            continue

        gain_pct = position.gain_loss_pct or 0
        score = position.current_score or 0

        # Stop loss - sell if down more than threshold
        if gain_pct <= -config.stop_loss_pct:
            sells.append({
                "position": position,
                "reason": f"STOP LOSS: Down {abs(gain_pct):.1f}%",
                "priority": 1  # High priority
            })

        # Take profit - sell if up more than threshold
        elif gain_pct >= config.take_profit_pct:
            sells.append({
                "position": position,
                "reason": f"TAKE PROFIT: Up {gain_pct:.1f}%",
                "priority": 2
            })

        # Score degradation - sell if score dropped significantly
        elif score < config.sell_score_threshold:
            sells.append({
                "position": position,
                "reason": f"WEAK SCORE: Score dropped to {score:.1f}",
                "priority": 3
            })

    # Sort by priority (stop losses first)
    sells.sort(key=lambda x: x["priority"])
    return sells


def evaluate_buys(db: Session) -> list:
    """Evaluate stocks for potential buys"""
    config = get_or_create_config(db)
    positions = db.query(AIPortfolioPosition).all()
    current_tickers = {p.ticker for p in positions}

    # Get top-scoring stocks we don't already own
    candidates = db.query(Stock).filter(
        Stock.canslim_score >= config.min_score_to_buy,
        Stock.current_price > 0,
        ~Stock.ticker.in_(current_tickers) if current_tickers else True
    ).order_by(desc(Stock.canslim_score)).limit(20).all()

    buys = []
    for stock in candidates:
        # Calculate position size (max % of portfolio)
        portfolio_value = get_portfolio_value(db)["total_value"]
        max_position_value = portfolio_value * (config.max_position_pct / 100)

        # Don't exceed available cash
        position_value = min(max_position_value, config.current_cash * 0.95)  # Keep 5% buffer

        if position_value < 100:  # Minimum $100 position
            continue

        shares = position_value / stock.current_price

        buys.append({
            "stock": stock,
            "shares": shares,
            "value": position_value,
            "reason": f"HIGH SCORE: {stock.canslim_score:.1f} CANSLIM score",
            "priority": 100 - stock.canslim_score  # Higher scores = lower priority number
        })

    # Sort by score (highest first)
    buys.sort(key=lambda x: x["priority"])
    return buys


def run_ai_trading_cycle(db: Session) -> dict:
    """
    Run a complete AI trading cycle:
    1. Update all position prices
    2. Evaluate and execute sells
    3. Evaluate and execute buys
    4. Take a portfolio snapshot
    """
    config = get_or_create_config(db)

    if not config.is_active:
        return {"status": "inactive", "message": "AI Portfolio is not active"}

    results = {
        "sells_executed": [],
        "buys_executed": [],
        "sells_considered": 0,
        "buys_considered": 0
    }

    # Update position prices first
    update_position_prices(db)

    # Get current position count
    positions = db.query(AIPortfolioPosition).all()
    position_count = len(positions)

    # Evaluate and execute sells
    sells = evaluate_sells(db)
    results["sells_considered"] = len(sells)

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
        buys = evaluate_buys(db)
        results["buys_considered"] = len(buys)

        for buy in buys:
            if position_count >= config.max_positions:
                break

            if config.current_cash < buy["value"]:
                continue

            stock = buy["stock"]

            # Execute the buy
            execute_trade(
                db=db,
                ticker=stock.ticker,
                action="BUY",
                shares=buy["shares"],
                price=stock.current_price,
                reason=buy["reason"],
                score=stock.canslim_score
            )

            # Deduct cash
            config.current_cash -= buy["value"]

            # Create position
            new_position = AIPortfolioPosition(
                ticker=stock.ticker,
                shares=buy["shares"],
                cost_basis=stock.current_price,
                purchase_score=stock.canslim_score,
                current_price=stock.current_price,
                current_value=buy["value"],
                gain_loss=0,
                gain_loss_pct=0,
                current_score=stock.canslim_score
            )
            db.add(new_position)
            position_count += 1

            results["buys_executed"].append({
                "ticker": stock.ticker,
                "shares": buy["shares"],
                "price": stock.current_price,
                "value": buy["value"],
                "score": stock.canslim_score,
                "reason": buy["reason"]
            })

    db.commit()

    # Take daily snapshot
    take_portfolio_snapshot(db)

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
    """Initialize or reset the AI portfolio"""
    # Clear existing data
    db.query(AIPortfolioPosition).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioSnapshot).delete()
    db.query(AIPortfolioConfig).delete()

    # Create new config
    config = AIPortfolioConfig(
        starting_cash=starting_cash,
        current_cash=starting_cash,
        max_positions=15,
        max_position_pct=10.0,
        min_score_to_buy=75,
        sell_score_threshold=50,
        take_profit_pct=25.0,
        stop_loss_pct=15.0,
        is_active=True
    )
    db.add(config)
    db.commit()

    # Run initial trading cycle to build positions
    result = run_ai_trading_cycle(db)

    return {
        "message": "AI Portfolio initialized",
        "starting_cash": starting_cash,
        "initial_trades": result
    }
