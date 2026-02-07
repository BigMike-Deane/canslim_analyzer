"""
CANSLIM Backtesting Engine

Simulates historical trading using the AI Portfolio logic.
Runs day-by-day simulation over a historical period.
"""

import logging
import math
import sys
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

# Add parent directory to path for config_loader import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config

from backend.database import (
    BacktestRun, BacktestSnapshot, BacktestTrade, BacktestPosition, Stock
)
from backend.historical_data import HistoricalDataProvider, HistoricalStockData
from canslim_scorer import calculate_coiled_spring_score

logger = logging.getLogger(__name__)

# Trading thresholds - loaded from config to match ai_trader.py
MIN_CASH_RESERVE_PCT = config.get('ai_trader.allocation.min_cash_reserve_pct', default=0.10)
MAX_SECTOR_ALLOCATION = config.get('ai_trader.allocation.max_sector_allocation', default=0.30)
MAX_STOCKS_PER_SECTOR = config.get('ai_trader.allocation.max_stocks_per_sector', default=4)
MAX_POSITION_ALLOCATION = config.get('ai_trader.allocation.max_single_position', default=0.15)


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
    partial_profit_taken: float = 0.0  # Cumulative % of position sold as partial profits
    pyramid_count: int = 0  # Number of times pyramided into


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
    is_partial: bool = False  # Whether this is a partial sell
    sell_pct: float = 100.0  # Percentage to sell (for partial sells)


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

        # Score history for stability checks (prevent selling on single score blips)
        self.score_history: Dict[str, List[float]] = {}

        # Static data cache (sector, earnings, etc. from database)
        self.static_data: Dict[str, dict] = {}

        # Track metrics
        self.peak_portfolio_value: float = self.backtest.starting_cash
        self.max_drawdown_pct: float = 0.0
        self.daily_returns: List[float] = []
        self.trades_executed: int = 0
        self.sells_executed: int = 0  # Track sells separately for accurate win rate
        self.profitable_trades: int = 0

        # Re-entry cooldown tracking: ticker -> (date, reason)
        self.recently_sold: Dict[str, tuple] = {}

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

            # Pre-compute market direction for all trading days (performance optimization)
            self.data_provider.precompute_market_direction()

            # Load static data (sector, earnings) from database
            self._load_static_data()

            # Filter for survivorship bias: only keep stocks with price data at backtest start
            # This prevents inflated returns from stocks that didn't exist or weren't tradeable
            available_tickers = self.data_provider.get_available_tickers()
            original_count = len(available_tickers)

            # Get the first trading day to check for data availability
            trading_days = self.data_provider.get_trading_days()
            if not trading_days:
                raise ValueError("No trading days in period")

            # Survivorship bias filter: exclude stocks that don't have price data at the START
            # of the backtest. This prevents us from trading stocks that IPO'd mid-backtest
            # or weren't actively traded at the beginning.
            first_day = trading_days[0]
            valid_tickers = []
            excluded_tickers = []

            for ticker in available_tickers:
                start_price = self.data_provider.get_price_on_date(ticker, first_day)
                if start_price and start_price > 0:
                    valid_tickers.append(ticker)
                else:
                    excluded_tickers.append(ticker)

            # Update the data provider to only use valid tickers
            self.data_provider.filter_tickers(valid_tickers)

            excluded_count = len(excluded_tickers)
            if excluded_count > 0:
                logger.info(f"Survivorship filter: excluded {excluded_count} stocks without "
                           f"price data at backtest start ({first_day})")
                if excluded_count <= 20:
                    logger.debug(f"Excluded tickers: {excluded_tickers}")

            logger.info(f"Universe after survivorship filter: {len(valid_tickers)} stocks "
                       f"(was {original_count})")

            # Initialize SPY benchmark
            self.spy_start_price = self.data_provider.get_spy_price_on_date(trading_days[0])
            if self.spy_start_price > 0:
                self.spy_shares = self.backtest.starting_cash / self.spy_start_price

            # Seed initial positions on day 1 based on historical CANSLIM scores
            self._seed_initial_positions(trading_days[0])

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
            self.backtest.completed_at = datetime.now(timezone.utc)
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
                "institutional_holders_pct": getattr(stock, 'institutional_holders_pct', 0) or 0,
                "roe": 0.0,
                "analyst_target_price": 0.0,
                "num_analyst_opinions": 0,
                "quarterly_earnings": stock.quarterly_earnings or [],
                "annual_earnings": stock.annual_earnings or [],
                "quarterly_revenue": stock.quarterly_revenue or [],
                # Coiled Spring fields
                "weeks_in_base": getattr(stock, 'weeks_in_base', 0) or 0,
                "earnings_beat_streak": getattr(stock, 'earnings_beat_streak', 0) or 0,
                "days_to_earnings": getattr(stock, 'days_to_earnings', None),
            }

    def _calculate_coiled_spring_for_backtest(self, ticker: str, score_data: dict, static_data: dict, cs_config: dict) -> dict:
        """
        Calculate Coiled Spring score for backtesting using score_data and static_data dicts.

        This is a simplified version that uses the pre-calculated score data from
        _calculate_scores() rather than live Stock objects.

        Returns dict with:
        - is_coiled_spring: bool
        - cs_score: float (bonus points)
        - cs_details: str (explanation)
        - allow_pre_earnings_buy: bool
        """
        # Create mock objects for calculate_coiled_spring_score
        class MockData:
            pass

        class MockScore:
            pass

        data = MockData()
        data.weeks_in_base = static_data.get('weeks_in_base', 0)
        data.earnings_beat_streak = static_data.get('earnings_beat_streak', 0)
        data.days_to_earnings = static_data.get('days_to_earnings')
        data.institutional_holders_pct = static_data.get('institutional_holders_pct', 0)

        score = MockScore()
        score.c_score = score_data.get('c_score', 0)
        score.l_score = score_data.get('l_score', 0)
        score.total_score = score_data.get('total_score', 0)

        return calculate_coiled_spring_score(data, score, cs_config)

    def _simulate_day(self, current_date: date):
        """Simulate one trading day with two-tier scoring for performance."""
        # Update position prices and peak tracking
        self._update_positions(current_date)

        # Tier 1: Score HELD positions every day (needed for sell/pyramid triggers)
        held_tickers = list(self.positions.keys())
        held_scores = self._calculate_scores(current_date, tickers=held_tickers) if held_tickers else {}

        # Evaluate and execute sells first
        sells = self._evaluate_sells(current_date, held_scores)
        for trade in sells:
            self._execute_sell(current_date, trade)

        # Tier 2: Check if we have cash to buy new positions
        portfolio_value = self._get_portfolio_value(current_date)
        can_buy = (self.cash / portfolio_value >= MIN_CASH_RESERVE_PCT and
                   len(self.positions) < self.backtest.max_positions and
                   self.cash >= 100)

        if can_buy:
            # Score full universe when we can buy
            all_scores = self._calculate_scores(current_date)
            all_scores.update(held_scores)  # Ensure held positions have scores

            # Evaluate pyramids with full scores
            pyramids = self._evaluate_pyramids(current_date, all_scores)
            for trade in pyramids[:3]:
                self._execute_pyramid(current_date, trade)

            # Evaluate and execute buys
            buys = self._evaluate_buys(current_date, all_scores)
            for trade in buys:
                if self.cash < 100:
                    break
                if len(self.positions) >= self.backtest.max_positions:
                    break
                self._execute_buy(current_date, trade)
        else:
            # Only evaluate pyramids for held positions
            pyramids = self._evaluate_pyramids(current_date, held_scores)
            for trade in pyramids[:3]:
                self._execute_pyramid(current_date, trade)

        # Take daily snapshot
        self._take_snapshot(current_date)

    def _seed_initial_positions(self, first_day: date):
        """
        Seed the portfolio with top CANSLIM stocks on day 1.

        Instead of waiting months for breakout/pre-breakout setups, establish
        initial positions based on the best-scoring stocks at backtest start.
        Uses quality filters but bypasses breakout/base pattern requirements.
        Allocates up to 50% of capital across top picks (max 3 positions).
        """
        scores = self._calculate_scores(first_day)
        if not scores:
            return

        # Use relaxed thresholds for seeding — we're establishing initial exposure,
        # not making high-conviction breakout entries. The M score drags down totals
        # during corrections, so we lower the bar by 7 points.
        seed_min_score = self.backtest.min_score_to_buy - 7  # e.g., 65 instead of 72

        quality_config = config.get('ai_trader.quality_filters', {})
        min_c_score = quality_config.get('min_c_score', 10) - 2  # Relax C by 2 (e.g., 8)
        min_l_score = quality_config.get('min_l_score', 8) - 2   # Relax L by 2 (e.g., 6)
        skip_growth = quality_config.get('skip_in_growth_mode', True)

        # Filter by relaxed quality and score, rank by total score
        candidates = []
        for ticker, data in scores.items():
            total_score = data.get("total_score", 0)
            if total_score < seed_min_score:
                continue

            c_score = data.get('c', 0) or data.get('c_score', 0)
            l_score = data.get('l', 0) or data.get('l_score', 0)
            is_growth = data.get('is_growth_stock', False)

            if not (is_growth and skip_growth):
                if c_score < min_c_score or l_score < min_l_score:
                    continue

            price = self.data_provider.get_price_on_date(ticker, first_day)
            if not price or price <= 0:
                continue

            candidates.append((ticker, data, price, total_score))

        if not candidates:
            logger.info(f"Backtest {self.backtest.id}: No stocks qualify for initial seeding on {first_day}")
            return

        # Sort by total score descending, take top 3
        candidates.sort(key=lambda x: x[3], reverse=True)
        max_seed_positions = 3
        seed_pct_per_position = 15.0  # 15% each, up to 45% total
        seeds = candidates[:max_seed_positions]

        logger.info(f"Backtest {self.backtest.id}: Seeding {len(seeds)} initial positions on {first_day}: "
                    f"{[s[0] for s in seeds]}")

        for ticker, data, price, total_score in seeds:
            portfolio_value = self._get_portfolio_value(first_day)
            position_value = portfolio_value * (seed_pct_per_position / 100)
            position_value = min(position_value, self.cash * 0.70)

            if position_value < 100 or self.cash < 100:
                break

            shares = position_value / price
            has_base = data.get("has_base_pattern", False)
            base_type = data.get("base_pattern", {}).get("type", "none") if has_base else "none"
            pct_from_high = data.get("pct_from_high", 0)

            reason = f"🏁 INITIAL SEED | Score {total_score:.0f} | {pct_from_high:.1f}% from high"
            if has_base:
                reason += f" | Base: {base_type}"

            trade = SimulatedTrade(
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                reason=reason,
                score=total_score,
                priority=0
            )
            self._execute_buy(first_day, trade)

        self._take_snapshot(first_day)

    def _update_positions(self, current_date: date):
        """Update position prices and track peak for trailing stops"""
        for ticker, position in self.positions.items():
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if price and price > 0:
                # Track peak price for trailing stop
                if price > position.peak_price:
                    position.peak_price = price
                    position.peak_date = current_date

    def _calculate_scores(self, current_date: date, tickers: List[str] = None) -> Dict[str, dict]:
        """
        Calculate full CANSLIM scores for stocks.

        Args:
            current_date: Date to calculate scores for
            tickers: Optional list of tickers to score. If None, scores all available.

        CANSLIM Components (100 points total):
        - C (15 pts): Current quarterly earnings growth + acceleration
        - A (15 pts): Annual earnings growth (3-year CAGR) + ROE quality
        - N (15 pts): New highs - proximity to pivot/52-week high
        - S (15 pts): Supply/Demand - volume analysis
        - L (15 pts): Leader/Laggard - relative strength
        - I (10 pts): Institutional ownership quality
        - M (15 pts): Market direction
        """
        scores = {}

        # Get market direction once (same for all stocks)
        market = self.data_provider.get_market_direction(current_date)

        ticker_list = tickers if tickers is not None else self.data_provider.get_available_tickers()
        for ticker in ticker_list:
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            # Get historical stock data with point-in-time earnings
            static_data = self.static_data.get(ticker, {})
            stock_data = self.data_provider.get_stock_data_on_date(ticker, current_date, static_data)

            high_52w, low_52w = self.data_provider.get_52_week_high_low(ticker, current_date)
            rs_12m = self.data_provider.get_relative_strength(ticker, current_date, 12)
            rs_3m = self.data_provider.get_relative_strength(ticker, current_date, 3)

            # ===== C SCORE (15 pts): Current Quarterly Earnings =====
            c_score = 0
            quarterly_earnings = stock_data.quarterly_earnings or []
            if len(quarterly_earnings) >= 2:
                # Calculate YoY growth (compare current quarter to same quarter last year)
                current_eps = quarterly_earnings[0] if quarterly_earnings else 0
                year_ago_eps = quarterly_earnings[4] if len(quarterly_earnings) > 4 else 0

                if year_ago_eps > 0 and current_eps > 0:
                    eps_growth = ((current_eps - year_ago_eps) / year_ago_eps) * 100
                    if eps_growth >= 50:
                        c_score = 15
                    elif eps_growth >= 25:
                        c_score = 12
                    elif eps_growth >= 15:
                        c_score = 9
                    elif eps_growth >= 5:
                        c_score = 6
                    elif eps_growth > 0:
                        c_score = 3
                    # Bonus for acceleration (each quarter better than last)
                    if len(quarterly_earnings) >= 3:
                        q1, q2, q3 = quarterly_earnings[0], quarterly_earnings[1], quarterly_earnings[2]
                        if q1 > q2 > q3 > 0:
                            c_score = min(15, c_score + 2)

            # ===== A SCORE (15 pts): Annual Earnings Growth =====
            a_score = 0
            annual_earnings = stock_data.annual_earnings or []
            roe = static_data.get("roe", 0)

            if len(annual_earnings) >= 3:
                # Calculate 3-year CAGR
                current_annual = annual_earnings[0]
                three_years_ago = annual_earnings[2] if len(annual_earnings) > 2 else annual_earnings[-1]

                if three_years_ago > 0 and current_annual > 0:
                    cagr = ((current_annual / three_years_ago) ** (1/3) - 1) * 100
                    if cagr >= 25:
                        a_score = 12
                    elif cagr >= 15:
                        a_score = 9
                    elif cagr >= 10:
                        a_score = 6
                    elif cagr > 0:
                        a_score = 3

                    # ROE bonus (17%+ is quality threshold)
                    if roe >= 0.17:
                        a_score = min(15, a_score + 3)
                    elif roe >= 0.12:
                        a_score = min(15, a_score + 1)

            # ===== N SCORE (15 pts): New Highs / Pivot Proximity =====
            base_pattern = self.data_provider.detect_base_pattern(ticker, current_date)
            has_base = base_pattern["type"] != "none"
            pivot_price = base_pattern.get("pivot_price", 0) if has_base else high_52w
            weeks_in_base = base_pattern.get("weeks", 0)

            n_score = 0
            pct_from_pivot = 0
            pct_from_high = ((high_52w - price) / high_52w) * 100 if high_52w > 0 else 100

            if pivot_price > 0:
                pct_from_pivot = ((pivot_price - price) / pivot_price) * 100

                if has_base:
                    # Proper base pattern scoring
                    if 0 < pct_from_pivot <= 5:
                        n_score = 15  # Optimal pre-breakout
                    elif -3 <= pct_from_pivot <= 0:
                        n_score = 14  # At breakout point
                    elif 5 < pct_from_pivot <= 10:
                        n_score = 12
                    elif -8 <= pct_from_pivot < -3:
                        n_score = 10
                    elif pct_from_pivot < -8:
                        n_score = 5  # Too extended
                    elif pct_from_pivot > 10:
                        n_score = 6
                    else:
                        n_score = 4
                else:
                    # No base pattern: reduced scores
                    if pct_from_high <= 5:
                        n_score = 10
                    elif pct_from_high <= 10:
                        n_score = 9
                    elif pct_from_high <= 15:
                        n_score = 7
                    elif pct_from_high <= 25:
                        n_score = 4
                    else:
                        n_score = 0

            # ===== S SCORE (15 pts): Supply/Demand (Volume) =====
            s_score = 0
            avg_volume = self.data_provider.get_50_day_avg_volume(ticker, current_date)
            current_volume = self.data_provider.get_volume_on_date(ticker, current_date) or 0

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume if current_volume else 1.0

                # Volume surge indicates institutional interest
                if volume_ratio >= 2.0:
                    s_score = 15
                elif volume_ratio >= 1.5:
                    s_score = 12
                elif volume_ratio >= 1.2:
                    s_score = 9
                elif volume_ratio >= 0.8:
                    s_score = 6
                else:
                    s_score = 3  # Low volume is concerning

            # ===== L SCORE (15 pts): Leader/Laggard (Relative Strength) =====
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

            # ===== I SCORE (10 pts): Institutional Ownership =====
            i_score = 0
            inst_pct = static_data.get("institutional_holders_pct", 0)
            # Ideal is 30-70% institutional ownership
            if 0.30 <= inst_pct <= 0.70:
                i_score = 10
            elif 0.20 <= inst_pct < 0.30 or 0.70 < inst_pct <= 0.80:
                i_score = 7
            elif 0.10 <= inst_pct < 0.20 or 0.80 < inst_pct <= 0.90:
                i_score = 4
            elif inst_pct > 0:
                i_score = 2
            # If no data, give benefit of doubt with middle score
            else:
                i_score = 5

            # ===== M SCORE (15 pts): Market Direction =====
            weighted_signal = market.get("weighted_signal", 0) if market else 0
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

            # ===== TOTAL SCORE (100 pts) =====
            total_score = c_score + a_score + n_score + s_score + l_score + i_score + m_score

            # Get breakout status for buy decisions
            is_breaking_out, volume_ratio_breakout, _ = self.data_provider.is_breaking_out(
                ticker, current_date
            )

            # Determine if this is a growth stock (negative/no earnings but high revenue growth)
            is_growth_stock = False
            if len(quarterly_earnings) >= 2 and quarterly_earnings[0] <= 0:
                quarterly_revenue = stock_data.quarterly_revenue or []
                if len(quarterly_revenue) >= 5:
                    current_rev = quarterly_revenue[0]
                    year_ago_rev = quarterly_revenue[4]
                    if year_ago_rev > 0 and current_rev > 0:
                        rev_growth = ((current_rev - year_ago_rev) / year_ago_rev) * 100
                        if rev_growth >= 30:
                            is_growth_stock = True

            scores[ticker] = {
                "total_score": total_score,
                "c_score": c_score,
                "a_score": a_score,
                "n_score": n_score,
                "s_score": s_score,
                "l_score": l_score,
                "i_score": i_score,
                "m_score": m_score,
                "rs_12m": rs_12m,
                "rs_3m": rs_3m,
                "pct_from_high": pct_from_high,
                "pct_from_pivot": pct_from_pivot,
                "pivot_price": pivot_price,
                "has_base_pattern": has_base,
                "base_pattern": base_pattern,
                "weeks_in_base": weeks_in_base,
                "is_growth_stock": is_growth_stock,
                "is_breaking_out": is_breaking_out,
                "volume_ratio": volume_ratio_breakout if volume_ratio_breakout else (current_volume / avg_volume if avg_volume > 0 else 1.0),
            }

        return scores

    def _update_score_history(self, ticker: str, score: float):
        """Track score history for stability checks"""
        if ticker not in self.score_history:
            self.score_history[ticker] = []
        self.score_history[ticker].append(score)
        # Keep last 5 scores (about 5 trading days worth)
        if len(self.score_history[ticker]) > 5:
            self.score_history[ticker] = self.score_history[ticker][-5:]

    def _check_score_stability(self, ticker: str, current_score: float, threshold: float = 50) -> dict:
        """
        Check if a low score is consistent across recent scans (not a one-time blip).
        Matches ai_trader.py behavior to prevent whipsaw sells.

        Returns:
            dict with:
            - is_stable: True if score has been consistently low (not a blip)
            - recent_scores: list of recent scores
            - avg_score: average of recent scores
            - consecutive_low: count of consecutive low scores
        """
        history = self.score_history.get(ticker, [])

        if len(history) < 2:
            # Not enough history, trust current score
            return {
                "is_stable": True,
                "recent_scores": [current_score],
                "avg_score": current_score,
                "consecutive_low": 1 if current_score < threshold else 0
            }

        # Calculate average of recent scores
        avg_score = sum(history) / len(history)

        # Count consecutive low scores from most recent
        consecutive_low = 0
        for score in reversed(history):
            if score < threshold:
                consecutive_low += 1
            else:
                break

        # Check if current score is significantly lower than average (potential blip)
        score_variance = abs(current_score - avg_score)

        # If current score is much lower than recent average, it might be a blip
        is_blip = (current_score < threshold and
                   avg_score > threshold + 10 and
                   score_variance > 15 and
                   consecutive_low < 2)  # Require 2+ consecutive low scans

        return {
            "is_stable": not is_blip,
            "recent_scores": history,
            "avg_score": avg_score,
            "consecutive_low": consecutive_low
        }

    def _evaluate_sells(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """Evaluate positions for sells using ai_trader logic"""
        sells = []

        # Update score history for all positions
        for ticker in self.positions:
            score_data = scores.get(ticker, {})
            current_score = score_data.get("total_score", 0)
            self._update_score_history(ticker, current_score)

        # Get market condition for market-aware stop losses
        market = self.data_provider.get_market_direction(current_date)
        spy_data = market.get('spy', {})
        is_bearish_market = spy_data.get('price', 0) < spy_data.get('ma_50', 0)

        # Get stop loss config
        stop_loss_config = config.get('ai_trader.stops', {})
        normal_stop_loss_pct = stop_loss_config.get('normal_stop_loss_pct', self.backtest.stop_loss_pct)
        bearish_stop_loss_pct = stop_loss_config.get('bearish_stop_loss_pct', 15.0)
        use_atr_stops = stop_loss_config.get('use_atr_stops', True)
        atr_multiplier = stop_loss_config.get('atr_multiplier', 2.5)
        atr_period = stop_loss_config.get('atr_period', 14)
        max_stop_pct = stop_loss_config.get('max_stop_pct', 20.0)

        # Partial trailing stop config
        partial_trailing_config = config.get('ai_trader.trailing_stops', {})
        partial_on_trailing = partial_trailing_config.get('partial_on_trailing', True)
        partial_min_pyramid = partial_trailing_config.get('partial_min_pyramid_count', 2)
        partial_min_score = partial_trailing_config.get('partial_min_score', 65)
        partial_sell_pct = partial_trailing_config.get('partial_sell_pct', 50)

        # Use wider stop loss in bearish market
        base_stop_loss_pct = bearish_stop_loss_pct if is_bearish_market else normal_stop_loss_pct

        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100
            score_data = scores.get(ticker, {})
            current_score = score_data.get("total_score", 0)

            # ATR-based stop loss: volatile stocks get wider stops
            effective_stop_loss_pct = base_stop_loss_pct
            if use_atr_stops:
                atr_pct = self.data_provider.get_atr(ticker, current_date, period=atr_period)
                if atr_pct > 0:
                    atr_stop_pct = atr_multiplier * atr_pct
                    effective_stop_loss_pct = max(base_stop_loss_pct, atr_stop_pct)
                    effective_stop_loss_pct = min(effective_stop_loss_pct, max_stop_pct)  # Cap

            # Market-aware stop loss check
            if gain_pct <= -effective_stop_loss_pct:
                market_note = " (bearish market)" if is_bearish_market else ""
                atr_note = f" (ATR-adj {effective_stop_loss_pct:.1f}%)" if use_atr_stops and effective_stop_loss_pct != base_stop_loss_pct else ""
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}",
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

                if trailing_stop_pct:
                    # Widen trailing stop for pyramided positions (high conviction)
                    pyramid_widening = min(position.pyramid_count * 2.0, 6.0)  # +2% per pyramid, max +6%
                    effective_trailing_stop = trailing_stop_pct + pyramid_widening

                    if drop_from_peak >= effective_trailing_stop:
                        # High conviction: partial sell + widen stop on remainder
                        if (partial_on_trailing and
                                position.pyramid_count >= partial_min_pyramid and
                                current_score >= partial_min_score):
                            shares_to_sell = position.shares * (partial_sell_pct / 100)
                            sells.append(SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=shares_to_sell,
                                price=price,
                                reason=f"TRAILING STOP (PARTIAL {partial_sell_pct}%): Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%)",
                                score=current_score,
                                priority=2,
                                is_partial=True,
                                sell_pct=partial_sell_pct
                            ))
                            # Reset peak to current price so remaining shares get a fresh wider stop
                            position.peak_price = price
                            position.peak_date = current_date
                        else:
                            # Standard: full sell
                            pyramid_note = f" (pyramid +{pyramid_widening:.0f}%)" if pyramid_widening > 0 else ""
                            sells.append(SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=position.shares,
                                price=price,
                                reason=f"TRAILING STOP: Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%){pyramid_note}",
                                score=current_score,
                                priority=2
                            ))
                        continue

            # Score crash check with stability verification and profitability exception
            # Get score crash config
            score_crash_config = config.get('ai_trader.score_crash', {})
            consecutive_required = score_crash_config.get('consecutive_required', 3)
            score_threshold = score_crash_config.get('threshold', 50)
            drop_required = score_crash_config.get('drop_required', 20)
            ignore_if_profitable_pct = score_crash_config.get('ignore_if_profitable_pct', 10)

            score_drop = position.purchase_score - current_score
            if score_drop > drop_required and current_score < score_threshold:
                # Skip score crash sell if position is profitable enough
                if gain_pct >= ignore_if_profitable_pct:
                    logger.debug(f"{ticker}: SKIP score crash - profitable (+{gain_pct:.1f}%)")
                    continue

                stability = self._check_score_stability(ticker, current_score, threshold=score_threshold)

                if not stability["is_stable"]:
                    # This looks like a data blip - skip this sell, wait for confirmation
                    logger.debug(f"{ticker}: SKIPPING SELL - possible blip. "
                                f"Current: {current_score:.0f}, Avg: {stability['avg_score']:.0f}, "
                                f"Consecutive low: {stability['consecutive_low']}")
                    continue

                # Require N consecutive low scores before selling (configurable, default 3)
                if stability["consecutive_low"] < consecutive_required:
                    logger.debug(f"{ticker}: SKIPPING SELL - only {stability['consecutive_low']} "
                                f"consecutive low score(s), need {consecutive_required}+")
                    continue

                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"SCORE CRASH: {position.purchase_score:.0f} -> {current_score:.0f} "
                           f"(avg: {stability['avg_score']:.0f}, {stability['consecutive_low']} low scans)",
                    score=current_score,
                    priority=3
                ))
                continue

            # PARTIAL PROFIT TAKING - let winners run while locking in gains
            # Only take partial profits if score remains decent (>= 60)
            partial_taken = position.partial_profit_taken

            # Check for 50% partial at +40% gain (highest priority partial)
            if gain_pct >= 40 and current_score >= 60 and partial_taken < 50:
                take_pct = 50 - partial_taken  # Take what's left to get to 50%
                if take_pct > 0:
                    shares_to_sell = position.shares * (take_pct / 100)
                    sells.append(SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=shares_to_sell,
                        price=price,
                        reason=f"PARTIAL PROFIT 50%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                        score=current_score,
                        priority=4,
                        is_partial=True,
                        sell_pct=take_pct
                    ))
                    continue  # Don't add more sell signals for this position

            # Check for 25% partial at +25% gain
            elif gain_pct >= 25 and current_score >= 60 and partial_taken < 25:
                shares_to_sell = position.shares * 0.25
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=shares_to_sell,
                    price=price,
                    reason=f"PARTIAL PROFIT 25%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                    score=current_score,
                    priority=5,
                    is_partial=True,
                    sell_pct=25
                ))
                continue  # Don't add more sell signals for this position

            # Protect gains - winners with weak scores
            if gain_pct >= 20 and current_score < self.backtest.sell_score_threshold:
                sells.append(SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({current_score:.0f})",
                    score=current_score,
                    priority=6
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

        # Apply market regime score adjustment (match ai_trader logic)
        market = self.data_provider.get_market_direction(current_date)
        weighted_signal = market.get("weighted_signal", 0)

        regime_config = config.get('ai_trader.market_regime', {})
        bullish_threshold = regime_config.get('bullish_threshold', 1.5)
        bearish_threshold = regime_config.get('bearish_threshold', -0.5)
        bearish_adj = regime_config.get('bearish_min_score_adj', 10)
        bear_exception_min_cal = regime_config.get('bear_exception_min_cal', 35)
        bear_exception_position_mult = regime_config.get('bear_exception_position_mult', 0.50)

        if weighted_signal >= bullish_threshold:
            effective_min_score = self.backtest.min_score_to_buy - 5  # Easier in bull
        elif weighted_signal <= bearish_threshold:
            effective_min_score = self.backtest.min_score_to_buy + bearish_adj  # Harder in bear
        else:
            effective_min_score = self.backtest.min_score_to_buy

        # Get candidates with regime-adjusted threshold
        candidates = [
            (ticker, data) for ticker, data in scores.items()
            if data["total_score"] >= effective_min_score
            and ticker not in current_tickers
        ]

        # BEAR MARKET EXCEPTION: Allow very strong stocks at reduced position size
        # If C+A+L >= 35 (out of 45 max), the stock has excellent fundamentals
        # regardless of market conditions
        is_bearish = weighted_signal <= bearish_threshold
        if not candidates and is_bearish:
            bear_candidates = []
            for ticker, data in scores.items():
                if ticker in current_tickers:
                    continue
                if data["total_score"] < self.backtest.min_score_to_buy:
                    continue
                cal_score = (data.get("c_score", 0) + data.get("a_score", 0) + data.get("l_score", 0))
                if cal_score >= bear_exception_min_cal:
                    data["_bear_market_entry"] = True
                    bear_candidates.append((ticker, data))
            candidates = bear_candidates

        portfolio_value = self._get_portfolio_value(current_date)

        for ticker, score_data in candidates:
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            score = score_data["total_score"]
            pct_from_high = score_data.get("pct_from_high", 100)
            pct_from_pivot = score_data.get("pct_from_pivot", pct_from_high)
            has_base = score_data.get("has_base_pattern", False)
            weeks_in_base = score_data.get("weeks_in_base", 0)

            # Re-entry cooldown: don't re-buy recently stopped-out stocks
            cooldown_config = config.get('ai_trader.re_entry_cooldown', {})
            stop_loss_cooldown = cooldown_config.get('stop_loss_days', 5)
            trailing_stop_cooldown = cooldown_config.get('trailing_stop_days', 3)

            if ticker in self.recently_sold:
                sold_date, sold_reason = self.recently_sold[ticker]
                days_since = (current_date - sold_date).days
                if "STOP LOSS" in sold_reason:
                    cooldown = stop_loss_cooldown
                elif "TRAILING STOP" in sold_reason and "PARTIAL" not in sold_reason:
                    cooldown = trailing_stop_cooldown
                else:
                    cooldown = 0  # No cooldown for score crash, protect gains, etc.
                if cooldown > 0 and days_since < cooldown:
                    continue

            # Check sector limits
            sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
            if not self._check_sector_limit(sector):
                continue

            # QUALITY FILTERS: Only buy stocks with strong fundamentals
            quality_config = config.get('ai_trader.quality_filters', {})
            min_c_score = quality_config.get('min_c_score', 10)
            min_l_score = quality_config.get('min_l_score', 8)
            min_volume_ratio = quality_config.get('min_volume_ratio', 1.2)
            skip_growth = quality_config.get('skip_in_growth_mode', True)

            # Get individual scores from score_data
            c_score = score_data.get('c', 0) or score_data.get('c_score', 0)
            l_score = score_data.get('l', 0) or score_data.get('l_score', 0)
            volume_ratio = score_data.get('volume_ratio', 1.0) or 1.0
            is_growth_stock = score_data.get('is_growth_stock', False)

            # Skip if not meeting quality thresholds (unless growth stock)
            if not (is_growth_stock and skip_growth):
                if c_score < min_c_score:
                    logger.debug(f"Skipping {ticker}: C score {c_score} < {min_c_score}")
                    continue
                if l_score < min_l_score:
                    logger.debug(f"Skipping {ticker}: L score {l_score} < {min_l_score}")
                    continue

            # Volume confirmation - accumulation signal
            if volume_ratio < min_volume_ratio and not score_data.get('is_breaking_out', False):
                logger.debug(f"Skipping {ticker}: Volume ratio {volume_ratio:.2f} < {min_volume_ratio}")
                continue

            # Earnings proximity check with Coiled Spring exception
            static_data = self.static_data.get(ticker, {})
            days_to_earnings = static_data.get('days_to_earnings')
            cs_config = config.get('coiled_spring', {})
            allow_buy_days = cs_config.get('earnings_window', {}).get('allow_buy_days', 7)
            block_days = cs_config.get('earnings_window', {}).get('block_days', 1)

            # Initialize CS result
            cs_result = None
            coiled_spring_bonus = 0

            if days_to_earnings is not None and 0 < days_to_earnings <= allow_buy_days:
                # Check for Coiled Spring qualification using static_data
                cs_result = self._calculate_coiled_spring_for_backtest(ticker, score_data, static_data, cs_config)

                if cs_result["is_coiled_spring"] and days_to_earnings > block_days:
                    # ALLOW - high conviction earnings catalyst
                    coiled_spring_bonus = cs_result.get('cs_score', 0)
                    logger.debug(f"Backtest CS: {ticker} ({cs_result['cs_details']})")
                else:
                    # Standard block for stocks without CS qualification (within 3 days)
                    if days_to_earnings <= 3:
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

            # PRE-BREAKOUT: 5-15% below pivot with valid base pattern
            # This is the BEST entry - optimal risk/reward BEFORE the crowd notices
            # PREDICTIVE: We want to catch stocks before they move
            if has_base and 5 <= pct_from_pivot <= 15:
                pre_breakout_bonus = 40  # Highest bonus - ideal entry point
                momentum_score = 35
                if volume_ratio >= 1.3:
                    pre_breakout_bonus += 5  # Accumulation volume bonus
                if weeks_in_base >= 10:
                    pre_breakout_bonus += 5  # Longer base = more stored energy

            # AT PIVOT ZONE: 0-5% below pivot with base pattern (ready to break out)
            elif has_base and 0 <= pct_from_pivot < 5:
                pre_breakout_bonus = 35  # Strong bonus near pivot
                momentum_score = 30
                if volume_ratio >= 1.5:
                    momentum_score += 5

            # BREAKOUT STOCKS - buying AFTER the pivot point (already moved - less ideal)
            # Once a stock has broken out, the easy money is made - we're late to the party
            elif is_breaking_out:
                breakout_bonus = 10  # Reduced bonus - we prefer pre-breakout entries
                if volume_ratio >= 2.0:
                    breakout_bonus += 5  # Small bonus for strong volume
                momentum_score = 15  # Lower score - already extended

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

            # MOMENTUM CONFIRMATION: Penalize stocks where recent momentum is fading
            # If 3-month RS is significantly weaker than 12-month RS, momentum is weakening
            rs_12m = score_data.get("rs_12m", 1.0)
            rs_3m = score_data.get("rs_3m", 1.0)
            momentum_penalty = 0
            if rs_12m > 0 and rs_3m < rs_12m * 0.95:
                # Recent momentum fading - apply 15% penalty to composite
                momentum_penalty = -0.15

            # Composite score: 25% growth, 25% score, 20% momentum, 20% breakout/pre-breakout, 10% base quality
            growth_projection = min(score_data.get("projected_growth", score * 0.3), 50)
            composite_score = (
                (growth_projection * 0.25) +
                (score * 0.25) +
                (momentum_score * 0.20) +
                ((breakout_bonus + pre_breakout_bonus) * 0.20) +
                (base_quality_bonus * 0.10) +
                extended_penalty +
                coiled_spring_bonus  # Earnings catalyst bonus
            )

            # Apply momentum penalty after base composite calculation
            if momentum_penalty < 0:
                composite_score *= (1 + momentum_penalty)  # Reduce by 15%

            if composite_score < 25:
                continue

            # Position sizing (4-20% based on conviction)
            conviction_multiplier = min(composite_score / 50, 1.5)
            position_pct = 4.0 + (conviction_multiplier * 10.67)
            position_pct = min(position_pct, 20.0)

            # PREDICTIVE POSITION SIZING: Pre-breakout stocks get largest positions
            # These are the ideal entries - before the crowd notices
            if pre_breakout_bonus >= 35 and has_base:
                position_pct *= 1.40  # 40% larger for best pre-breakout entries
            elif pre_breakout_bonus >= 25 and has_base:
                position_pct *= 1.30  # 30% larger for good pre-breakout entries
            elif is_breaking_out and volume_ratio >= 1.5:
                position_pct *= 1.0   # No boost - already extended, entry is late

            # Coiled Spring position boost
            if coiled_spring_bonus > 0:
                cs_multiplier = cs_config.get('position_multiplier', 1.25)
                position_pct *= cs_multiplier

            # Reduce position size for bear market exception entries
            if score_data.get("_bear_market_entry"):
                position_pct *= bear_exception_position_mult

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
            # Coiled Spring indicator (highest priority)
            if coiled_spring_bonus > 0:
                days_to_earn = static_data.get('days_to_earnings', 0) or 0
                reason_parts.append(f"🌀 COILED SPRING ({days_to_earn}d to earnings)")
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
        portfolio_value = self._get_portfolio_value(current_date)

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
        """Execute a sell trade (full or partial)"""
        if trade.ticker not in self.positions:
            return

        position = self.positions[trade.ticker]
        cost_basis = position.cost_basis
        total_value = trade.shares * trade.price
        realized_gain = total_value - (trade.shares * cost_basis)

        self.cash += total_value

        if trade.is_partial:
            # PARTIAL SELL - reduce position but don't delete
            position.shares -= trade.shares
            position.partial_profit_taken += trade.sell_pct

            logger.debug(f"PARTIAL SELL {trade.ticker}: {trade.sell_pct}% "
                        f"({trade.shares:.2f} shares), remaining: {position.shares:.2f}")
        else:
            # FULL SELL - remove position entirely and record for cooldown
            del self.positions[trade.ticker]
            self.recently_sold[trade.ticker] = (current_date, trade.reason)

        self._record_trade(
            current_date, trade,
            cost_basis=cost_basis,
            realized_gain=realized_gain
        )

        self.trades_executed += 1
        self.sells_executed += 1  # Track sells for win rate calculation
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
        position.pyramid_count += 1

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

    def _get_portfolio_value(self, current_date: date = None) -> float:
        """Get current portfolio value using actual prices, not peak prices"""
        if current_date and self.data_provider:
            # Use actual current prices for accurate valuation
            positions_value = sum(
                pos.shares * (self.data_provider.get_price_on_date(pos.ticker, current_date) or pos.cost_basis)
                for pos in self.positions.values()
            )
        else:
            # Fallback to cost basis if no date provided
            positions_value = sum(
                pos.shares * pos.cost_basis
                for pos in self.positions.values()
            )
        return self.cash + positions_value

    def _check_sector_limit(self, sector: str) -> bool:
        """Check if we can add another position in this sector"""
        sector_count = sum(1 for p in self.positions.values() if p.sector == sector)
        return sector_count < MAX_STOCKS_PER_SECTOR

    def _calculate_final_metrics(self):
        """Calculate final backtest metrics"""
        # Get final snapshot using end date for accurate valuation
        final_value = self._get_portfolio_value(self.backtest.end_date)
        self.backtest.final_value = final_value
        self.backtest.total_return_pct = ((final_value / self.backtest.starting_cash) - 1) * 100
        self.backtest.max_drawdown_pct = self.max_drawdown_pct
        self.backtest.total_trades = self.trades_executed

        # Win rate - calculated from sells only (not all trades including buys)
        if self.sells_executed > 0:
            self.backtest.win_rate = (self.profitable_trades / self.sells_executed) * 100
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
