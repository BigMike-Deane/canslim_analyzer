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
                # Check for cancellation request every 10 days
                if i % 10 == 0:
                    self.db.refresh(self.backtest)
                    if self.backtest.cancel_requested:
                        logger.info(f"Backtest {self.backtest.id} cancelled by user at {i}/{total_days} days")
                        self.backtest.status = "cancelled"
                        self.backtest.error_message = f"Cancelled by user at {progress:.0f}% progress"
                        self.db.commit()
                        return self.backtest

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
        - Proximity to PIVOT POINT (from base pattern) - not just 52-week high
        - Volume trends
        - Base pattern quality
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

            # Get base pattern for pivot-based scoring
            base_pattern = self.data_provider.detect_base_pattern(ticker, current_date)
            has_base = base_pattern["type"] != "none"
            pivot_price = base_pattern.get("pivot_price", 0) if has_base else high_52w
            weeks_in_base = base_pattern.get("weeks", 0)

            # Calculate N score based on pivot proximity (not just 52-week high)
            # Stocks with proper base patterns get scored on pivot, others on 52-week high
            n_score = 0
            pct_from_pivot = 0
            pct_from_high = ((high_52w - price) / high_52w) * 100 if high_52w > 0 else 100

            if pivot_price > 0:
                pct_from_pivot = ((pivot_price - price) / pivot_price) * 100

                if has_base:
                    # Proper base pattern: score based on pivot proximity
                    # BEST entry: 0-5% BELOW pivot (pre-breakout zone)
                    if 0 < pct_from_pivot <= 5:
                        n_score = 15  # Optimal pre-breakout position
                    # Good entry: at or just above pivot (breakout zone)
                    elif -3 <= pct_from_pivot <= 0:
                        n_score = 14  # At the breakout point
                    # Acceptable: 5-10% below pivot (building toward breakout)
                    elif 5 < pct_from_pivot <= 10:
                        n_score = 12
                    # Extended: 3-8% above pivot (still ok but less ideal)
                    elif -8 <= pct_from_pivot < -3:
                        n_score = 10
                    # Too extended: >8% above pivot (chasing)
                    elif pct_from_pivot < -8:
                        n_score = 5
                    # Too far from pivot: >10% below
                    elif pct_from_pivot > 10:
                        n_score = 6
                    else:
                        n_score = 4
                else:
                    # No base pattern: use 52-week high but with reduced scores
                    # Penalize stocks at highs without proper consolidation
                    if pct_from_high <= 5:
                        n_score = 10  # Reduced from 15 - no base pattern
                    elif pct_from_high <= 10:
                        n_score = 9
                    elif pct_from_high <= 15:
                        n_score = 7
                    elif pct_from_high <= 25:
                        n_score = 4
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
            is_breaking_out, volume_ratio, _ = self.data_provider.is_breaking_out(
                ticker, current_date
            )

            scores[ticker] = {
                "total_score": total_score,
                "n_score": n_score,
                "l_score": l_score,
                "m_score": m_score,
                "rs_12m": rs_12m,
                "rs_3m": rs_3m,
                "pct_from_high": pct_from_high,
                "pct_from_pivot": pct_from_pivot,
                "pivot_price": pivot_price,
                "has_base_pattern": has_base,
                "base_pattern": base_pattern,
                "weeks_in_base": weeks_in_base,
                "is_growth_stock": False,  # Would need earnings analysis
                "is_breaking_out": is_breaking_out,
                "volume_ratio": volume_ratio,
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
        Evaluate stocks for buys with enhanced breakout and base pattern detection.

        Key improvements:
        - Pre-breakout bonus: +20 for stocks 5-15% below pivot with valid base
        - Breakout bonus: +25 for confirmed breakouts, +35 for strong volume breakouts
        - Extended penalty: -10 to -20 for stocks >5% above pivot (chasing)
        - Chase penalty: -15 for buying at highs without base pattern
        - Volume confirmation: Larger positions for breakouts with strong volume
        - Base pattern bonus: Extra points for stocks with proper consolidation
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
            pct_from_pivot = score_data.get("pct_from_pivot", pct_from_high)
            has_base = score_data.get("has_base_pattern", False)
            weeks_in_base = score_data.get("weeks_in_base", 0)

            # Check sector limits
            sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
            if not self._check_sector_limit(sector):
                continue

            # Get breakout status and volume ratio from cached scores
            is_breaking_out = score_data.get("is_breaking_out", False)
            volume_ratio = score_data.get("volume_ratio", 1.0)
            base_pattern = score_data.get("base_pattern", {"type": "none"})

            # Calculate momentum, breakout, pre-breakout, and extended scores
            momentum_score = 0
            breakout_bonus = 0
            pre_breakout_bonus = 0
            extended_penalty = 0
            base_quality_bonus = 0

            # Base pattern quality bonus (up to 10 points)
            if has_base:
                base_type = base_pattern.get("type", "none")
                if base_type == "cup_with_handle":
                    base_quality_bonus = 10  # Best pattern
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

            # BREAKOUT STOCKS get highest priority - buying the pivot point
            if is_breaking_out:
                breakout_bonus = 25  # Big bonus for confirmed breakouts
                if volume_ratio >= 2.0:
                    breakout_bonus += 10  # Extra bonus for strong volume breakout
                momentum_score = 30

            # PRE-BREAKOUT: 5-15% below pivot with valid base pattern
            # This is often the BEST entry - before the crowd notices
            elif has_base and 5 <= pct_from_pivot <= 15:
                pre_breakout_bonus = 20  # Big bonus for pre-breakout position
                momentum_score = 25
                if volume_ratio >= 1.3:
                    pre_breakout_bonus += 5  # Accumulation volume bonus

            # AT PIVOT: 0-5% from pivot with base pattern
            elif has_base and 0 <= pct_from_pivot < 5:
                pre_breakout_bonus = 15  # Good entry near pivot
                momentum_score = 22
                if volume_ratio >= 1.5:
                    momentum_score += 5

            # EXTENDED: More than 5% above pivot - the easy money is gone
            elif pct_from_pivot < -5:
                if pct_from_pivot < -10:
                    extended_penalty = -20  # Heavily penalize extended stocks
                    momentum_score = 5
                else:
                    extended_penalty = -10  # Moderate penalty
                    momentum_score = 10

            # NO BASE PATTERN: Use 52-week high with penalties
            elif not has_base:
                if pct_from_high <= 2:
                    # At 52-week high without base = chasing
                    if score < 85:
                        extended_penalty = -15
                        momentum_score = 5
                    else:
                        momentum_score = 12
                elif pct_from_high <= 10:
                    momentum_score = 15
                    if volume_ratio >= 1.5:
                        momentum_score += 3
                elif pct_from_high <= 25:
                    momentum_score = 8
                else:
                    momentum_score = -5

            # Composite score: 25% growth, 25% score, 20% momentum, 20% breakout/pre-breakout, 10% base quality
            growth_projection = min(score_data.get("projected_growth", score * 0.3), 50)
            composite_score = (
                (growth_projection * 0.25) +
                (score * 0.25) +
                (momentum_score * 0.20) +
                ((breakout_bonus + pre_breakout_bonus) * 0.20) +
                (base_quality_bonus * 0.10) +
                extended_penalty
            )

            if composite_score < 25:
                continue

            # Position sizing (4-20% based on conviction)
            conviction_multiplier = min(composite_score / 50, 1.5)
            position_pct = 4.0 + (conviction_multiplier * 10.67)
            position_pct = min(position_pct, 20.0)

            # Breakout and pre-breakout stocks get larger positions (high confidence entry)
            if is_breaking_out and volume_ratio >= 1.5:
                position_pct *= 1.25  # 25% larger position for confirmed breakouts
            elif pre_breakout_bonus >= 15 and has_base:
                position_pct *= 1.15  # 15% larger for pre-breakout with base

            position_value = portfolio_value * (position_pct / 100)

            # Allow more cash for breakout/pre-breakout stocks
            if is_breaking_out:
                cash_limit = self.cash * 0.85
            elif pre_breakout_bonus >= 15:
                cash_limit = self.cash * 0.80
            else:
                cash_limit = self.cash * 0.70
            position_value = min(position_value, cash_limit)

            if position_value < 100:
                continue

            shares = position_value / price

            # Build reason string
            reason_parts = []
            if is_breaking_out:
                base_type = base_pattern.get("type", "none")
                reason_parts.append(f"🚀 BREAKOUT ({base_type}) {volume_ratio:.1f}x vol")
            elif pre_breakout_bonus >= 15:
                base_type = base_pattern.get("type", "none")
                reason_parts.append(f"📈 PRE-BREAKOUT ({base_type}) {pct_from_pivot:.0f}% below pivot")
            elif extended_penalty < 0:
                reason_parts.append(f"⚠️ Extended {abs(pct_from_pivot):.0f}% above pivot")
            reason_parts.append(f"Score {score:.0f}")
            if has_base and not is_breaking_out and pre_breakout_bonus < 15:
                reason_parts.append(f"Base: {base_pattern.get('type', 'none')} {weeks_in_base}w")
            if not is_breaking_out and pre_breakout_bonus < 15:
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
