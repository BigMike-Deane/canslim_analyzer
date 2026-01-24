"""
CANSLIM Backtesting Engine

Simulates historical trading using the AI Portfolio logic.
Runs day-by-day simulation over a historical period.
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from backend.database import (
    BacktestRun, BacktestSnapshot, BacktestTrade, BacktestPosition, Stock
)
from backend.historical_data import HistoricalDataProvider, HistoricalStockData

logger = logging.getLogger(__name__)

# Trading thresholds (matching ai_trader.py)
MIN_CASH_RESERVE_PCT = 0.10
MAX_SECTOR_ALLOCATION = 0.30
MAX_STOCKS_PER_SECTOR = 4
MAX_POSITION_ALLOCATION = 0.15


@dataclass
class SimulatedPosition:
    """In-memory position for simulation"""
    ticker: str
    shares: float
    cost_basis: float
    purchase_date: date
    purchase_score: float
    peak_price: float
    peak_date: date
    is_growth_stock: bool = False
    sector: str = ""


@dataclass
class SimulatedTrade:
    """Trade to be executed"""
    ticker: str
    action: str  # BUY, SELL, PYRAMID
    shares: float
    price: float
    reason: str
    score: float = 0.0
    priority: int = 0
    is_growth_stock: bool = False


class BacktestEngine:
    """
    Runs a historical simulation of the CANSLIM AI trading strategy.
    """

    def __init__(self, db: Session, backtest_id: int):
        self.db = db
        self.backtest = db.query(BacktestRun).get(backtest_id)

        if not self.backtest:
            raise ValueError(f"Backtest {backtest_id} not found")

        # Data provider (initialized in run())
        self.data_provider: Optional[HistoricalDataProvider] = None

        # In-memory portfolio state
        self.cash: float = self.backtest.starting_cash
        self.positions: Dict[str, SimulatedPosition] = {}

        # Static data cache (sector, earnings, etc. from database)
        self.static_data: Dict[str, dict] = {}

        # Track metrics
        self.peak_portfolio_value: float = self.backtest.starting_cash
        self.max_drawdown_pct: float = 0.0
        self.daily_returns: List[float] = []
        self.trades_executed: int = 0
        self.profitable_trades: int = 0

        # SPY tracking for benchmark
        self.spy_start_price: float = 0.0
        self.spy_shares: float = 0.0  # Hypothetical SPY buy-and-hold

    def run(self) -> BacktestRun:
        """
        Execute the backtest day by day.
        """
        try:
            self.backtest.status = "running"
            self.db.commit()

            # Get stock universe
            tickers = self._get_universe_tickers()
            logger.info(f"Backtest {self.backtest.id}: {len(tickers)} tickers, "
                        f"{self.backtest.start_date} to {self.backtest.end_date}")

            # Initialize data provider
            self.data_provider = HistoricalDataProvider(tickers)

            # Preload historical data
            def update_progress(pct):
                self.backtest.progress_pct = pct * 0.3  # Loading is 30% of total
                self.db.commit()

            success = self.data_provider.preload_data(
                self.backtest.start_date,
                self.backtest.end_date,
                progress_callback=update_progress
            )

            if not success:
                raise ValueError("Failed to load historical data")

            # Load static data (sector, earnings) from database
            self._load_static_data()

            # Get trading days
            trading_days = self.data_provider.get_trading_days()
            if not trading_days:
                raise ValueError("No trading days in period")

            # Initialize SPY benchmark
            self.spy_start_price = self.data_provider.get_spy_price_on_date(trading_days[0])
            if self.spy_start_price > 0:
                self.spy_shares = self.backtest.starting_cash / self.spy_start_price

            # Simulate each trading day
            total_days = len(trading_days)
            for i, current_date in enumerate(trading_days):
                self._simulate_day(current_date)

                # Update progress (30% loading + 70% simulation)
                progress = 30 + (i / total_days * 70)
                self.backtest.progress_pct = progress
                if i % 10 == 0:  # Commit every 10 days
                    self.db.commit()

            # Calculate final metrics
            self._calculate_final_metrics()

            self.backtest.status = "completed"
            self.backtest.completed_at = datetime.utcnow()
            self.backtest.progress_pct = 100
            self.db.commit()

            logger.info(f"Backtest {self.backtest.id} completed: "
                        f"{self.backtest.total_return_pct:.1f}% return, "
                        f"{self.backtest.total_trades} trades")

            return self.backtest

        except Exception as e:
            logger.error(f"Backtest {self.backtest.id} failed: {e}")
            self.backtest.status = "failed"
            self.backtest.error_message = str(e)
            self.db.commit()
            raise

    def _get_universe_tickers(self) -> List[str]:
        """Get list of tickers to include in backtest"""
        if self.backtest.stock_universe == "custom" and self.backtest.custom_tickers:
            return self.backtest.custom_tickers

        # Try to get tickers from the ticker module (more reliable than database)
        try:
            from sp500_tickers import get_sp500_tickers, get_all_tickers

            if self.backtest.stock_universe == "sp500":
                tickers = get_sp500_tickers()
                logger.info(f"Loaded {len(tickers)} S&P 500 tickers from module")
            else:
                tickers = get_all_tickers()
                logger.info(f"Loaded {len(tickers)} tickers from module")

            if tickers:
                return tickers
        except Exception as e:
            logger.warning(f"Could not load tickers from module: {e}")

        # Fallback to database
        query = self.db.query(Stock.ticker)

        if self.backtest.stock_universe == "sp500":
            query = query.filter(Stock.market_cap >= 10_000_000_000)

        tickers = [row[0] for row in query.all()]
        return tickers

    def _load_static_data(self):
        """Load static stock data (sector, earnings) from database"""
        stocks = self.db.query(Stock).all()
        for stock in stocks:
            self.static_data[stock.ticker] = {
                "sector": stock.sector or "Unknown",
                "name": stock.name or stock.ticker,
                "institutional_holders_pct": 0.0,  # Would need historical data
                "roe": 0.0,
                "analyst_target_price": 0.0,
                "num_analyst_opinions": 0,
                "quarterly_earnings": stock.quarterly_earnings or [],
                "annual_earnings": stock.annual_earnings or [],
                "quarterly_revenue": stock.quarterly_revenue or [],
            }

    def _simulate_day(self, current_date: date):
        """Simulate one trading day"""
        # Update position prices and peak tracking
        self._update_positions(current_date)

        # Calculate scores for universe (simplified - use price momentum as proxy)
        scores = self._calculate_scores(current_date)

        # Evaluate and execute sells first
        sells = self._evaluate_sells(current_date, scores)
        for trade in sells:
            self._execute_sell(current_date, trade)

        # Evaluate and execute pyramids
        pyramids = self._evaluate_pyramids(current_date, scores)
        for trade in pyramids[:3]:  # Max 3 pyramids per day
            self._execute_pyramid(current_date, trade)

        # Check cash reserve before buys
        portfolio_value = self._get_portfolio_value()
        if self.cash / portfolio_value >= MIN_CASH_RESERVE_PCT:
            # Evaluate and execute buys
            buys = self._evaluate_buys(current_date, scores)
            for trade in buys:
                if self.cash < 100:
                    break
                if len(self.positions) >= self.backtest.max_positions:
                    break
                self._execute_buy(current_date, trade)

        # Take daily snapshot
        self._take_snapshot(current_date)

    def _update_positions(self, current_date: date):
        """Update position prices and track peak for trailing stops"""
        for ticker, position in self.positions.items():
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if price and price > 0:
                # Track peak price for trailing stop
                if price > position.peak_price:
                    position.peak_price = price
                    position.peak_date = current_date

    def _calculate_scores(self, current_date: date) -> Dict[str, dict]:
        """
        Calculate simplified scores for all stocks.

        In a full implementation, this would run CANSLIM scoring.
        For now, we use a simplified scoring based on:
        - Price momentum (relative strength)
        - Proximity to 52-week high
        - Volume trends
        """
        scores = {}

        for ticker in self.data_provider.get_available_tickers():
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            high_52w, low_52w = self.data_provider.get_52_week_high_low(ticker, current_date)
            rs_12m = self.data_provider.get_relative_strength(ticker, current_date, 12)
            rs_3m = self.data_provider.get_relative_strength(ticker, current_date, 3)

            # Get market direction
            market = self.data_provider.get_market_direction(current_date)

            # Calculate N score (proximity to 52-week high)
            n_score = 0
            if high_52w > 0:
                pct_from_high = ((high_52w - price) / high_52w) * 100
                if pct_from_high <= 5:
                    n_score = 15
                elif pct_from_high <= 10:
                    n_score = 12
                elif pct_from_high <= 15:
                    n_score = 10
                elif pct_from_high <= 25:
                    n_score = 5
                else:
                    n_score = 0

            # Calculate L score (relative strength)
            # Weight: 60% 12-month, 40% 3-month
            combined_rs = (rs_12m * 0.6) + (rs_3m * 0.4)
            if combined_rs >= 1.3:
                l_score = 15
            elif combined_rs >= 1.15:
                l_score = 12
            elif combined_rs >= 1.0:
                l_score = 8
            elif combined_rs >= 0.85:
                l_score = 4
            else:
                l_score = 0

            # Calculate M score (market direction)
            weighted_signal = market.get("weighted_signal", 0)
            if weighted_signal >= 1.5:
                m_score = 15
            elif weighted_signal >= 1.0:
                m_score = 12
            elif weighted_signal >= 0.5:
                m_score = 8
            elif weighted_signal >= 0:
                m_score = 5
            else:
                m_score = 0

            # Simplified total score (N + L + M only, out of 45 scaled to 100)
            # In production, would include C, A, S, I scores too
            raw_score = n_score + l_score + m_score
            total_score = (raw_score / 45) * 100

            # Get breakout status for better buy decisions
            is_breaking_out, volume_ratio, base_pattern = self.data_provider.is_breaking_out(
                ticker, current_date
            )

            scores[ticker] = {
                "total_score": total_score,
                "n_score": n_score,
                "l_score": l_score,
                "m_score": m_score,
                "rs_12m": rs_12m,
                "rs_3m": rs_3m,
                "pct_from_high": ((high_52w - price) / high_52w * 100) if high_52w > 0 else 100,
                "is_growth_stock": False,  # Would need earnings analysis
                "is_breaking_out": is_breaking_out,
                "volume_ratio": volume_ratio,
                "base_pattern": base_pattern,
            }

        return scores

    def _evaluate_sells(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """Evaluate positions for sells using ai_trader logic"""
        sells = []

        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100
            score_data = scores.get(ticker, {})
            current_score = score_data.get("total_score", 0)

            # Stop loss check
            if gain_pct <= -self.backtest.stop_loss_pct:
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%",
                    score=current_score,
                    priority=1
                ))
                continue

            # Trailing stop check
            if position.peak_price > 0:
                drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                peak_gain_pct = ((position.peak_price - position.cost_basis) / position.cost_basis) * 100

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
                    sells.append(SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=position.shares,
                        price=price,
                        reason=f"TRAILING STOP: Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%)",
                        score=current_score,
                        priority=2
                    ))
                    continue

            # Score crash check
            score_drop = position.purchase_score - current_score
            if score_drop > 20 and current_score < 50:
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"SCORE CRASH: {position.purchase_score:.0f} -> {current_score:.0f}",
                    score=current_score,
                    priority=3
                ))
                continue

            # Protect gains - winners with weak scores
            if gain_pct >= 20 and current_score < self.backtest.sell_score_threshold:
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({current_score:.0f})",
                    score=current_score,
                    priority=4
                ))
                continue

            # Weak flat positions
            if gain_pct < 10 and current_score < self.backtest.sell_score_threshold:
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"WEAK POSITION: {gain_pct:+.1f}%, score {current_score:.0f}",
                    score=current_score,
                    priority=6
                ))

        # Sort by priority
        sells.sort(key=lambda x: x.priority)
        return sells

    def _evaluate_buys(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """
        Evaluate stocks for buys with breakout detection.

        Key improvements over simple score-based buying:
        - Breakout bonus: +25 for confirmed breakouts, +35 for strong volume breakouts
        - Chase penalty: -15 for buying at highs without breakout confirmation
        - Volume confirmation: Larger positions for breakouts with strong volume
        """
        buys = []
        current_tickers = set(self.positions.keys())

        # Get candidates above minimum score
        candidates = [
            (ticker, data) for ticker, data in scores.items()
            if data["total_score"] >= self.backtest.min_score_to_buy
            and ticker not in current_tickers
        ]

        portfolio_value = self._get_portfolio_value()

        for ticker, score_data in candidates:
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            score = score_data["total_score"]
            pct_from_high = score_data.get("pct_from_high", 100)

            # Check sector limits
            sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
            if not self._check_sector_limit(sector):
                continue

            # Get breakout status and volume ratio from cached scores
            is_breaking_out = score_data.get("is_breaking_out", False)
            volume_ratio = score_data.get("volume_ratio", 1.0)
            base_pattern = score_data.get("base_pattern", {"type": "none"})

            # Calculate momentum and breakout scores
            momentum_score = 0
            breakout_bonus = 0
            at_high_penalty = 0

            # BREAKOUT STOCKS get highest priority - buying the pivot point
            if is_breaking_out:
                breakout_bonus = 25  # Big bonus for confirmed breakouts
                if volume_ratio >= 2.0:
                    breakout_bonus += 10  # Extra bonus for strong volume breakout
                momentum_score = 30
            # Within 5% of high but NOT breaking out - risky entry point
            elif pct_from_high <= 2:
                # At the high without breakout = chasing, penalize unless extremely high score
                if score < 85:
                    at_high_penalty = -15  # Penalize buying at 52-week high
                    momentum_score = 5
                else:
                    momentum_score = 15
            # 5-10% from high - decent entry
            elif pct_from_high <= 10:
                momentum_score = 20
                if volume_ratio >= 1.5:
                    momentum_score += 5  # Bonus for accumulation volume
            # 10-25% from high - pullback zone
            elif pct_from_high <= 25:
                momentum_score = 10
            # More than 25% from high - weak momentum
            else:
                momentum_score = -10

            # Composite score: 30% growth, 30% score, 25% momentum, 15% breakout bonus
            growth_projection = min(score_data.get("projected_growth", score * 0.3), 50)
            composite_score = (
                (growth_projection * 0.30) +
                (score * 0.30) +
                (momentum_score * 0.25) +
                (breakout_bonus * 0.15) +
                at_high_penalty
            )

            if composite_score < 25:
                continue

            # Position sizing (4-20% based on conviction)
            conviction_multiplier = min(composite_score / 50, 1.5)
            position_pct = 4.0 + (conviction_multiplier * 10.67)
            position_pct = min(position_pct, 20.0)

            # Breakout stocks get larger positions (high confidence entry)
            if is_breaking_out and volume_ratio >= 1.5:
                position_pct *= 1.25  # 25% larger position for confirmed breakouts

            position_value = portfolio_value * (position_pct / 100)

            # Allow more cash for breakout stocks
            cash_limit = self.cash * 0.85 if is_breaking_out else self.cash * 0.70
            position_value = min(position_value, cash_limit)

            if position_value < 100:
                continue

            shares = position_value / price

            # Build reason string
            reason_parts = []
            if is_breaking_out:
                base_type = base_pattern.get("type", "none")
                reason_parts.append(f"🚀 BREAKOUT ({base_type}) {volume_ratio:.1f}x vol")
            reason_parts.append(f"Score {score:.0f}")
            if not is_breaking_out:
                reason_parts.append(f"{pct_from_high:.1f}% from high")
            if volume_ratio >= 1.5 and not is_breaking_out:
                reason_parts.append(f"Vol {volume_ratio:.1f}x")

            buys.append(SimulatedTrade(
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                reason=" | ".join(reason_parts),
                score=score,
                priority=-int(composite_score)  # Higher score = lower priority number
            ))

        buys.sort(key=lambda x: x.priority)

        # Log breakout candidates for debugging
        breakout_buys = [b for b in buys[:10] if "BREAKOUT" in b.reason]
        if breakout_buys:
            logger.info(f"Breakout candidates on {current_date}: "
                       f"{[b.ticker for b in breakout_buys]}")

        return buys

    def _evaluate_pyramids(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """Evaluate positions for pyramid opportunities, prioritizing breakouts"""
        pyramids = []
        portfolio_value = self._get_portfolio_value()

        for ticker, position in self.positions.items():
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            score_data = scores.get(ticker, {})
            current_score = score_data.get("total_score", 0)

            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100
            current_value = position.shares * price
            current_allocation = current_value / portfolio_value if portfolio_value > 0 else 0

            # Skip if not profitable enough or score too low
            if gain_pct < 5 or current_score < 70:
                continue

            # Skip if at max allocation
            if current_allocation >= MAX_POSITION_ALLOCATION:
                continue

            # Skip if not enough cash
            if self.cash < 200:
                continue

            # Check if breaking out (prefer pyramiding into breakouts)
            is_breaking_out = score_data.get("is_breaking_out", False)
            volume_ratio = score_data.get("volume_ratio", 1.0)

            # Calculate pyramid amount (50% of original position)
            original_cost = position.shares * position.cost_basis
            pyramid_amount = original_cost * 0.5

            # Cap by remaining room
            remaining_room = (MAX_POSITION_ALLOCATION - current_allocation) * portfolio_value
            pyramid_amount = min(pyramid_amount, remaining_room, self.cash * 0.5)

            if pyramid_amount < 100:
                continue

            shares = pyramid_amount / price

            # Build reason and set priority
            reason_parts = [f"Winner +{gain_pct:.0f}%", f"Score {current_score:.0f}"]
            priority = 0

            if is_breaking_out:
                priority = -20  # Higher priority for breakouts
                reason_parts.append("Breakout!")
            if volume_ratio >= 1.5:
                priority -= 5
                reason_parts.append(f"Vol {volume_ratio:.1f}x")

            pyramids.append(SimulatedTrade(
                ticker=ticker,
                action="PYRAMID",
                shares=shares,
                price=price,
                reason=" | ".join(reason_parts),
                score=current_score,
                priority=priority
            ))

        # Sort by priority (breakouts first)
        pyramids.sort(key=lambda x: x.priority)
        return pyramids

    def _execute_buy(self, current_date: date, trade: SimulatedTrade):
        """Execute a buy trade"""
        total_cost = trade.shares * trade.price

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        sector = self.static_data.get(trade.ticker, {}).get("sector", "Unknown")

        self.positions[trade.ticker] = SimulatedPosition(
            ticker=trade.ticker,
            shares=trade.shares,
            cost_basis=trade.price,
            purchase_date=current_date,
            purchase_score=trade.score,
            peak_price=trade.price,
            peak_date=current_date,
            is_growth_stock=trade.is_growth_stock,
            sector=sector
        )

        self._record_trade(current_date, trade, cost_basis=trade.price)
        self.trades_executed += 1

    def _execute_sell(self, current_date: date, trade: SimulatedTrade):
        """Execute a sell trade"""
        if trade.ticker not in self.positions:
            return

        position = self.positions[trade.ticker]
        total_value = trade.shares * trade.price
        cost_basis = position.cost_basis
        realized_gain = total_value - (trade.shares * cost_basis)

        self.cash += total_value
        del self.positions[trade.ticker]

        self._record_trade(
            current_date, trade,
            cost_basis=cost_basis,
            realized_gain=realized_gain
        )

        self.trades_executed += 1
        if realized_gain > 0:
            self.profitable_trades += 1

    def _execute_pyramid(self, current_date: date, trade: SimulatedTrade):
        """Execute a pyramid (add to position)"""
        if trade.ticker not in self.positions:
            return

        position = self.positions[trade.ticker]
        total_cost = trade.shares * trade.price

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        # Update cost basis (weighted average)
        old_value = position.shares * position.cost_basis
        new_value = trade.shares * trade.price
        position.shares += trade.shares
        position.cost_basis = (old_value + new_value) / position.shares

        self._record_trade(current_date, trade)
        self.trades_executed += 1

    def _record_trade(self, current_date: date, trade: SimulatedTrade,
                      cost_basis: float = None, realized_gain: float = None):
        """Record trade in database"""
        position = self.positions.get(trade.ticker)
        holding_days = None
        realized_gain_pct = None

        if trade.action == "SELL" and cost_basis:
            holding_days = (current_date - position.purchase_date).days if position else 0
            realized_gain_pct = (realized_gain / (trade.shares * cost_basis)) * 100 if cost_basis else 0

        db_trade = BacktestTrade(
            backtest_id=self.backtest.id,
            date=current_date,
            ticker=trade.ticker,
            action=trade.action,
            shares=trade.shares,
            price=trade.price,
            total_value=trade.shares * trade.price,
            reason=trade.reason,
            canslim_score=trade.score,
            is_growth_stock=trade.is_growth_stock,
            cost_basis=cost_basis,
            realized_gain=realized_gain,
            realized_gain_pct=realized_gain_pct,
            holding_days=holding_days
        )
        self.db.add(db_trade)

    def _take_snapshot(self, current_date: date):
        """Take daily portfolio snapshot"""
        positions_value = sum(
            pos.shares * (self.data_provider.get_price_on_date(pos.ticker, current_date) or 0)
            for pos in self.positions.values()
        )
        total_value = self.cash + positions_value

        # Track peak and drawdown
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        drawdown = ((self.peak_portfolio_value - total_value) / self.peak_portfolio_value) * 100
        if drawdown > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown

        # Calculate returns
        cumulative_return_pct = ((total_value / self.backtest.starting_cash) - 1) * 100

        if self.daily_returns:
            prev_value = self.backtest.starting_cash * (1 + self.daily_returns[-1] / 100)
            daily_return_pct = ((total_value / prev_value) - 1) * 100
        else:
            daily_return_pct = cumulative_return_pct

        self.daily_returns.append(cumulative_return_pct)

        # SPY benchmark
        spy_price = self.data_provider.get_spy_price_on_date(current_date)
        spy_value = self.spy_shares * spy_price if spy_price else 0
        spy_return_pct = ((spy_price / self.spy_start_price) - 1) * 100 if self.spy_start_price else 0

        snapshot = BacktestSnapshot(
            backtest_id=self.backtest.id,
            date=current_date,
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            positions_count=len(self.positions),
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
            spy_price=spy_price,
            spy_value=spy_value,
            spy_return_pct=spy_return_pct
        )
        self.db.add(snapshot)

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # Use most recent snapshot or calculate from positions
        positions_value = sum(
            pos.shares * pos.peak_price  # Use peak as approximation
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def _check_sector_limit(self, sector: str) -> bool:
        """Check if we can add another position in this sector"""
        sector_count = sum(1 for p in self.positions.values() if p.sector == sector)
        return sector_count < MAX_STOCKS_PER_SECTOR

    def _calculate_final_metrics(self):
        """Calculate final backtest metrics"""
        # Get final snapshot
        final_value = self._get_portfolio_value()
        self.backtest.final_value = final_value
        self.backtest.total_return_pct = ((final_value / self.backtest.starting_cash) - 1) * 100
        self.backtest.max_drawdown_pct = self.max_drawdown_pct
        self.backtest.total_trades = self.trades_executed

        # Win rate
        if self.trades_executed > 0:
            self.backtest.win_rate = (self.profitable_trades / self.trades_executed) * 100
        else:
            self.backtest.win_rate = 0

        # Sharpe ratio (simplified - using daily returns)
        if len(self.daily_returns) > 1:
            import numpy as np
            daily_returns_array = np.diff(self.daily_returns)  # Convert cumulative to daily
            if len(daily_returns_array) > 0 and np.std(daily_returns_array) > 0:
                avg_return = np.mean(daily_returns_array)
                std_return = np.std(daily_returns_array)
                # Annualized Sharpe (assuming 252 trading days)
                self.backtest.sharpe_ratio = (avg_return / std_return) * math.sqrt(252)
            else:
                self.backtest.sharpe_ratio = 0
        else:
            self.backtest.sharpe_ratio = 0

        # SPY benchmark
        trading_days = self.data_provider.get_trading_days()
        if trading_days:
            spy_end_price = self.data_provider.get_spy_price_on_date(trading_days[-1])
            if spy_end_price and self.spy_start_price:
                self.backtest.spy_final_value = self.spy_shares * spy_end_price
                self.backtest.spy_return_pct = ((spy_end_price / self.spy_start_price) - 1) * 100


def run_backtest(db: Session, backtest_id: int) -> BacktestRun:
    """Run a backtest by ID"""
    engine = BacktestEngine(db, backtest_id)
    return engine.run()
