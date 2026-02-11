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


def get_strategy_profile(strategy_name: str = "balanced") -> dict:
    """Load strategy profile from YAML config, falling back to balanced defaults."""
    profiles = config.get('strategy_profiles', {})
    profile = profiles.get(strategy_name, profiles.get('balanced', {}))
    return profile


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
    signal_factors: dict = field(default_factory=dict)  # Trade journal: factors that drove the buy


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
        self.drawdown_halt: bool = False  # Circuit breaker: no new buys when True

        # Re-entry cooldown tracking: ticker -> (date, reason)
        self.recently_sold: Dict[str, tuple] = {}

        # Market timing state (O'Neil A/D + FTD)
        self.market_timing_state: str = "CONFIRMED_UPTREND"
        self.ftd_can_buy: bool = True
        self.ftd_penalty_active: bool = False
        self.heat_penalty_active: bool = False

        # Track consecutive days with no positions (for re-seeding)
        self.idle_days: int = 0

        # Strategy profile (balanced or growth)
        self.strategy = self.backtest.strategy or "balanced"
        self.profile = get_strategy_profile(self.strategy)

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
                # Check for cancellation during data loading
                self.db.refresh(self.backtest)
                if self.backtest.cancel_requested or self.backtest.status == "cancelled":
                    raise ValueError("Cancelled by user")

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

            # Fetch missing historical earnings for old backtest periods
            self._fetch_missing_earnings()

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
                    if self.backtest.cancel_requested or self.backtest.status == "cancelled":
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
            # Store data fingerprint for reproducibility tracking
            if hasattr(self, '_data_fingerprint'):
                self.backtest.error_message = f"data_fp:{self._data_fingerprint}"
            self.db.commit()

            logger.info(f"Backtest {self.backtest.id} completed: "
                        f"{self.backtest.total_return_pct:.1f}% return, "
                        f"{self.backtest.total_trades} trades")

            return self.backtest

        except Exception as e:
            if "Cancelled by user" in str(e):
                logger.info(f"Backtest {self.backtest.id} cancelled during data loading")
                self.backtest.status = "cancelled"
                self.backtest.error_message = "Cancelled by user"
                self.db.commit()
                return self.backtest
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
        """Load static stock data (sector, earnings, ROE) from database"""
        from backend.database import StockDataCache

        stocks = self.db.query(Stock).all()

        # Batch-load ROE and analyst data from StockDataCache
        cache_entries = self.db.query(StockDataCache).all()
        cache_by_ticker = {c.ticker: c for c in cache_entries}

        for stock in stocks:
            cache = cache_by_ticker.get(stock.ticker)

            # Extract ROE from StockDataCache (stored as decimal, e.g. 0.25 = 25%)
            roe = 0.0
            if cache and cache.roe is not None:
                roe = cache.roe
            elif stock.score_details:
                # Fallback: extract from score_details JSON if available
                a_details = stock.score_details.get('a', {})
                if isinstance(a_details, dict):
                    roe = a_details.get('roe', 0.0) or 0.0

            # Extract analyst data from cache
            analyst_target = 0.0
            num_analysts = 0
            if cache:
                analyst_target = getattr(cache, 'analyst_target_price', 0) or 0
                num_analysts = getattr(cache, 'num_analyst_opinions', 0) or 0

            self.static_data[stock.ticker] = {
                "sector": stock.sector or "Unknown",
                "name": stock.name or stock.ticker,
                "institutional_holders_pct": getattr(stock, 'institutional_holders_pct', 0) or 0,
                "roe": roe,
                "analyst_target_price": analyst_target,
                "num_analyst_opinions": num_analysts,
                "quarterly_earnings": stock.quarterly_earnings or [],
                "annual_earnings": stock.annual_earnings or [],
                "quarterly_revenue": stock.quarterly_revenue or [],
                # Coiled Spring fields
                "weeks_in_base": getattr(stock, 'weeks_in_base', 0) or 0,
                "earnings_beat_streak": getattr(stock, 'earnings_beat_streak', 0) or 0,
                "days_to_earnings": getattr(stock, 'days_to_earnings', None),
                "eps_estimate_revision_pct": getattr(stock, 'eps_estimate_revision_pct', None),
            }

    def _fetch_missing_earnings(self):
        """
        Fetch earnings from FMP for ALL tickers to ensure deterministic backtests.

        The DB cache changes with every scan, causing different C/A scores and
        different stock picks between backtest runs. By always fetching from FMP,
        we get consistent historical data regardless of DB state.

        Only updates self.static_data (NOT the DB) to avoid interfering with scans.
        """
        import requests
        import time
        import hashlib

        fmp_api_key = os.environ.get('FMP_API_KEY', '')
        if not fmp_api_key:
            logger.warning("Backtest: No FMP_API_KEY, cannot fetch earnings")
            return

        # Calculate how many quarters we need
        today = date.today()
        days_diff = (today - self.backtest.start_date).days

        # quarters_to_skip = how many the filter will skip
        # We need at least 8 usable quarters after skipping (for C score YoY + buffer)
        quarters_to_skip = max(0, (days_diff + 45) // 91) if days_diff > 0 else 0
        quarters_needed = quarters_to_skip + 8
        annuals_to_skip = max(0, (days_diff + 45) // 365) if days_diff > 0 else 0
        annuals_needed = annuals_to_skip + 5

        # Fetch for ALL tickers (deterministic — ignores DB cache)
        all_tickers = list(self.static_data.keys())

        logger.info(f"Backtest {self.backtest.id}: Fetching FMP earnings for "
                     f"{len(all_tickers)} tickers (need {quarters_needed}q/{annuals_needed}a, "
                     f"backtest starts {self.backtest.start_date})")

        fmp_base = "https://financialmodelingprep.com/stable"
        fetched_count = 0
        failed_count = 0
        batch_size = 50  # Respect FMP rate limits

        for i, ticker in enumerate(all_tickers):
            try:
                # Fetch quarterly (enough for the backtest period)
                q_limit = min(quarters_needed + 4, 40)  # small buffer, cap at 40
                q_url = f"{fmp_base}/income-statement?symbol={ticker}&period=quarter&limit={q_limit}&apikey={fmp_api_key}"
                q_resp = requests.get(q_url, timeout=10)

                # Fetch annual
                a_limit = min(annuals_needed + 2, 15)
                a_url = f"{fmp_base}/income-statement?symbol={ticker}&limit={a_limit}&apikey={fmp_api_key}"
                a_resp = requests.get(a_url, timeout=10)

                updated = False

                if q_resp.status_code == 200:
                    q_data = q_resp.json()
                    if isinstance(q_data, list) and q_data:
                        q_eps = [q.get("eps", 0) or 0 for q in q_data]
                        q_rev = [q.get("revenue", 0) or 0 for q in q_data]
                        # Always overwrite — FMP is the canonical source for determinism
                        self.static_data[ticker]["quarterly_earnings"] = q_eps
                        self.static_data[ticker]["quarterly_revenue"] = q_rev
                        updated = True

                if a_resp.status_code == 200:
                    a_data = a_resp.json()
                    if isinstance(a_data, list) and a_data:
                        a_eps = [a.get("eps", 0) or 0 for a in a_data]
                        self.static_data[ticker]["annual_earnings"] = a_eps
                        updated = True

                if updated:
                    fetched_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                failed_count += 1
                if i < 3:  # Only log first few errors
                    logger.debug(f"Backtest earnings fetch failed for {ticker}: {e}")

            # Rate limiting: pause every batch_size requests
            if (i + 1) % batch_size == 0:
                time.sleep(1.0)
                # Log progress every 200 tickers
                if (i + 1) % 200 == 0:
                    logger.info(f"Backtest {self.backtest.id}: FMP earnings fetch progress: "
                               f"{i + 1}/{len(all_tickers)} tickers")

        # Compute data fingerprint for reproducibility tracking
        fingerprint_data = ""
        for ticker in sorted(self.static_data.keys()):
            q = self.static_data[ticker].get("quarterly_earnings", [])
            a = self.static_data[ticker].get("annual_earnings", [])
            fingerprint_data += f"{ticker}:{q[:8]}:{a[:5]}|"
        self._data_fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]

        logger.info(f"Backtest {self.backtest.id}: FMP earnings fetched for {fetched_count} tickers "
                     f"({failed_count} failed), data fingerprint: {self._data_fingerprint}")

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

    def _calculate_portfolio_heat(self, current_date: date) -> float:
        """
        Calculate total portfolio heat: sum of (position_size_pct × distance_to_stop_pct).
        Higher heat = more total risk exposure.
        """
        portfolio_value = self._get_portfolio_value(current_date)
        if portfolio_value <= 0:
            return 0.0

        stop_loss_config = config.get('ai_trader.stops', {})
        base_stop = stop_loss_config.get('normal_stop_loss_pct', 8.0)

        total_heat = 0.0
        for ticker, position in self.positions.items():
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue
            position_value = position.shares * price
            position_pct = (position_value / portfolio_value) * 100
            # Distance to stop = current gain% + stop loss%
            # e.g., stock up 5% with 8% stop = 13% distance
            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100 if position.cost_basis > 0 else 0
            distance_to_stop = base_stop + gain_pct if gain_pct > 0 else base_stop + gain_pct
            total_heat += position_pct * (distance_to_stop / 100)

        return total_heat

    def _simulate_day(self, current_date: date):
        """Simulate one trading day with two-tier scoring for performance."""
        # Update position prices and peak tracking
        self._update_positions(current_date)

        # ===== MARKET TIMING: A/D Days + FTD =====
        market_timing_config = config.get('market_timing', {})
        if market_timing_config.get('follow_through_day', {}).get('enabled', True):
            ftd_status = self.data_provider.get_follow_through_day_status(current_date)
            self.market_timing_state = ftd_status["state"]
            self.ftd_can_buy = ftd_status["can_buy"]
        else:
            self.ftd_can_buy = True

        # ===== DRAWDOWN CIRCUIT BREAKER =====
        portfolio_value = self._get_portfolio_value(current_date)
        current_drawdown = 0.0
        if self.peak_portfolio_value > 0:
            current_drawdown = ((self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value) * 100

        drawdown_config = config.get('ai_trader.drawdown_protection', {})
        halt_threshold = drawdown_config.get('halt_new_buys_pct', 15.0)
        liquidate_threshold = drawdown_config.get('liquidate_all_pct', 25.0)
        recovery_threshold = drawdown_config.get('recovery_pct', 10.0)

        if current_drawdown >= liquidate_threshold and self.positions:
            # Emergency: liquidate all positions
            logger.warning(f"CIRCUIT BREAKER: {current_drawdown:.1f}% drawdown >= {liquidate_threshold}% threshold - LIQUIDATING ALL")
            self.drawdown_halt = True
            held_tickers = list(self.positions.keys())
            held_scores = self._calculate_scores(current_date, tickers=held_tickers) if held_tickers else {}
            for ticker in list(self.positions.keys()):
                position = self.positions[ticker]
                price = self.data_provider.get_price_on_date(ticker, current_date)
                if price and price > 0:
                    score = held_scores.get(ticker, {}).get("total_score", 0)
                    cb_gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100 if position.cost_basis > 0 else 0
                    trade = SimulatedTrade(
                        ticker=ticker, action="SELL", shares=position.shares,
                        price=price, reason=f"CIRCUIT BREAKER: Portfolio drawdown {current_drawdown:.1f}%",
                        score=score, priority=0
                    )
                    trade._signal_factors = {"sell_reason": "CIRCUIT BREAKER", "gain_pct": round(cb_gain_pct, 1), "drawdown_pct": round(current_drawdown, 1)}
                    self._execute_sell(current_date, trade)
            self._take_snapshot(current_date)
            return

        if current_drawdown >= halt_threshold:
            if not self.drawdown_halt:
                logger.warning(f"CIRCUIT BREAKER: {current_drawdown:.1f}% drawdown - halting new buys")
            self.drawdown_halt = True
        elif self.drawdown_halt and current_drawdown < recovery_threshold:
            logger.info(f"CIRCUIT BREAKER LIFTED: Drawdown recovered to {current_drawdown:.1f}%")
            self.drawdown_halt = False

        # Tier 1: Score HELD positions every day (needed for sell/pyramid triggers)
        held_tickers = list(self.positions.keys())
        held_scores = self._calculate_scores(current_date, tickers=held_tickers) if held_tickers else {}

        # Evaluate and execute sells first
        sells = self._evaluate_sells(current_date, held_scores)
        for trade in sells:
            self._execute_sell(current_date, trade)

        # Tier 2: Check if we have cash to buy new positions
        portfolio_value = self._get_portfolio_value(current_date)

        # Dynamic cash reserve based on market regime
        market_for_cash = self.data_provider.get_market_direction(current_date)
        weighted_signal_cash = market_for_cash.get("weighted_signal", 0) if market_for_cash else 0
        alloc_config = config.get('ai_trader.allocation', {})
        if weighted_signal_cash >= 1.5:
            min_cash_pct = alloc_config.get('cash_reserve_strong_bull', 0.05)
        elif weighted_signal_cash >= 1.0:
            min_cash_pct = alloc_config.get('cash_reserve_bull', 0.10)
        elif weighted_signal_cash >= 0:
            min_cash_pct = alloc_config.get('cash_reserve_neutral', 0.20)
        elif weighted_signal_cash >= -0.5:
            min_cash_pct = alloc_config.get('cash_reserve_bear', 0.40)
        else:
            min_cash_pct = alloc_config.get('cash_reserve_strong_bear', 0.60)

        # Portfolio heat check — advisory penalty, not hard block
        heat_config = config.get('portfolio_heat', {})
        self.heat_penalty_active = False
        if heat_config.get('enabled', True):
            portfolio_heat = self._calculate_portfolio_heat(current_date)
            max_heat = heat_config.get('max_heat_pct', 15.0)
            if portfolio_heat > max_heat:
                self.heat_penalty_active = True
                logger.debug(f"Portfolio heat {portfolio_heat:.1f}% > {max_heat}% - applying score penalty + half-size buys")

        # FTD: advisory penalty, not hard block
        self.ftd_penalty_active = not self.ftd_can_buy

        profile_max_positions = self.profile.get('max_positions', self.backtest.max_positions)
        can_buy = (not self.drawdown_halt and
                   self.cash / portfolio_value >= min_cash_pct and
                   len(self.positions) < profile_max_positions and
                   self.cash >= 100)

        # Track idle days (no positions, all cash) for re-seeding
        if not self.positions:
            self.idle_days += 1
        else:
            self.idle_days = 0

        # Re-seed if portfolio has been 100% cash for 10+ trading days
        # This handles the case where all seeds were sold and the buy filters
        # are too strict for the available historical data
        if can_buy and not self.positions and self.idle_days >= 10:
            logger.info(f"Backtest {self.backtest.id}: Portfolio idle {self.idle_days} days, re-seeding on {current_date}")
            self._seed_initial_positions(current_date)
            self._take_snapshot(current_date)
            return

        # MARKET REGIME GATE: Don't buy when SPY is below 50MA (bearish regime)
        # This prevents the whack-a-mole pattern of buying into corrections
        regime_gate_config = config.get('ai_trader.market_regime_gate', {})
        if regime_gate_config.get('enabled', True) and can_buy:
            spy_data = market_for_cash.get('spy', {}) if market_for_cash else {}
            spy_price = spy_data.get('price', 0)
            spy_ma50 = spy_data.get('ma_50', 0)
            if spy_price and spy_ma50 and spy_price < spy_ma50:
                logger.debug(f"REGIME GATE: SPY ${spy_price:.2f} below 50MA ${spy_ma50:.2f}, skipping buys on {current_date}")
                can_buy = False

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
                if len(self.positions) >= profile_max_positions:
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

        NOTE: On day 1 of a 1-year backtest, _filter_available_earnings skips
        ~4 quarters of earnings data. With only 8 quarters in the DB, the C score
        YoY comparison (needs index 4) fails for ALL stocks, giving C=0 universally.
        A score similarly gets 0. So we skip C/A quality filters for seeding and
        rank by N+S+L+I+M (technical/institutional quality).
        """
        scores = self._calculate_scores(first_day)
        if not scores:
            return

        # Low floor since C and A scores are typically 0 on backtest day 1
        # (earnings data gets filtered out for historical dates).
        # N+S+L+I+M max = 70, good stocks score 30-50 in this range.
        seed_min_score = 35

        # Skip C/A quality filters — earnings data is unavailable for day 1 scoring.
        # Instead rely on L score (relative strength) as the primary quality signal.
        candidates = []
        for ticker, data in scores.items():
            total_score = data.get("total_score", 0)
            if total_score < seed_min_score:
                continue

            # Require decent relative strength (the one reliable signal on day 1)
            l_score = data.get('l_score', 0)
            if l_score < 8:
                continue

            price = self.data_provider.get_price_on_date(ticker, first_day)
            if not price or price <= 0:
                continue

            candidates.append((ticker, data, price, total_score))

        if not candidates:
            logger.info(f"Backtest {self.backtest.id}: No stocks qualify for initial seeding on {first_day}")
            return

        # Sort by total score descending, take top N based on strategy profile
        # O'Neil/Minervini: start with half-size pilot positions, let winners prove themselves
        candidates.sort(key=lambda x: x[3], reverse=True)
        max_seed_positions = self.profile.get('seed_count', 5)
        seed_pct_per_position = self.profile.get('seed_pct', 10.0)
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
            eps_growth = 0.0  # Track for projected_growth calculation
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
            annual_cagr = 0.0  # Track for projected_growth calculation
            annual_earnings = stock_data.annual_earnings or []
            roe = static_data.get("roe", 0)

            if len(annual_earnings) >= 3:
                # Calculate 3-year CAGR
                current_annual = annual_earnings[0]
                three_years_ago = annual_earnings[2] if len(annual_earnings) > 2 else annual_earnings[-1]

                if three_years_ago > 0 and current_annual > 0:
                    cagr = ((current_annual / three_years_ago) ** (1/3) - 1) * 100
                    annual_cagr = cagr  # Save for projected_growth
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

            # ===== PROJECTED GROWTH (independent of CANSLIM score) =====
            # Combine EPS growth rate, annual CAGR, and price momentum
            # This gives the composite score a truly independent growth signal
            momentum_pct = (combined_rs - 1.0) * 100  # RS ratio to %
            projected_growth = (
                eps_growth * 0.40 +      # 40% quarterly EPS growth rate
                annual_cagr * 0.30 +     # 30% annual earnings CAGR
                momentum_pct * 0.30      # 30% price momentum vs market
            )

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
                "projected_growth": projected_growth,
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

        # Get stop loss config — strategy profile overrides YAML defaults
        stop_loss_config = config.get('ai_trader.stops', {})
        normal_stop_loss_pct = self.profile.get('stop_loss_pct', stop_loss_config.get('normal_stop_loss_pct', self.backtest.stop_loss_pct))
        bearish_stop_loss_pct = self.profile.get('bearish_stop_loss_pct', stop_loss_config.get('bearish_stop_loss_pct', 7.0))
        use_atr_stops = stop_loss_config.get('use_atr_stops', True)
        atr_multiplier = stop_loss_config.get('atr_multiplier', 2.5)
        atr_period = stop_loss_config.get('atr_period', 14)
        max_stop_pct = stop_loss_config.get('max_stop_pct', 20.0)

        # Partial profit config from YAML
        partial_profit_config = config.get('ai_trader.partial_profits', {})
        pp_25_gain = partial_profit_config.get('threshold_25pct', {}).get('gain_pct', 25)
        pp_25_sell = partial_profit_config.get('threshold_25pct', {}).get('sell_pct', 25)
        pp_25_min_score = partial_profit_config.get('threshold_25pct', {}).get('min_score', 60)
        pp_40_gain = partial_profit_config.get('threshold_40pct', {}).get('gain_pct', 40)
        pp_40_sell = partial_profit_config.get('threshold_40pct', {}).get('sell_pct', 50)
        pp_40_min_score = partial_profit_config.get('threshold_40pct', {}).get('min_score', 60)

        # Partial trailing stop config
        partial_trailing_config = config.get('ai_trader.trailing_stops', {})
        partial_on_trailing = partial_trailing_config.get('partial_on_trailing', True)
        partial_min_pyramid = partial_trailing_config.get('partial_min_pyramid_count', 2)
        partial_min_score = partial_trailing_config.get('partial_min_score', 65)
        partial_sell_pct = partial_trailing_config.get('partial_sell_pct', 50)

        # Use tighter stop loss in bearish market (7% vs 8% normal)
        base_stop_loss_pct = bearish_stop_loss_pct if is_bearish_market else normal_stop_loss_pct

        # VIX-regime stop adjustment
        vix_config = config.get('vix_stops', {})
        if vix_config.get('enabled', True):
            vix_proxy = self.data_provider.get_vix_proxy(current_date)
            low_vix = vix_config.get('low_vix_threshold', 15)
            high_vix = vix_config.get('high_vix_threshold', 25)
            if vix_proxy < low_vix:
                vix_multiplier = vix_config.get('low_vix_stop_tighten', 0.80)
                base_stop_loss_pct *= vix_multiplier
            elif vix_proxy > high_vix:
                vix_multiplier = vix_config.get('high_vix_stop_widen', 1.20)
                base_stop_loss_pct *= vix_multiplier

        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            # Guard against division by zero on cost_basis
            if position.cost_basis <= 0:
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
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}",
                    score=current_score,
                    priority=1
                )
                trade._signal_factors = {"sell_reason": "STOP LOSS", "gain_pct": round(gain_pct, 1), "stop_pct": round(effective_stop_loss_pct, 1)}
                sells.append(trade)
                continue

            # Trailing stop check
            if position.peak_price > 0:
                drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                peak_gain_pct = ((position.peak_price - position.cost_basis) / position.cost_basis) * 100

                # Dynamic trailing stop thresholds — strategy profile overrides
                profile_trailing = self.profile.get('trailing_stops', {})
                trailing_stop_pct = None
                if peak_gain_pct >= 50:
                    trailing_stop_pct = profile_trailing.get('gain_50_plus', 15)
                elif peak_gain_pct >= 30:
                    trailing_stop_pct = profile_trailing.get('gain_30_to_50', 12)
                elif peak_gain_pct >= 20:
                    trailing_stop_pct = profile_trailing.get('gain_20_to_30', 10)
                elif peak_gain_pct >= 10:
                    trailing_stop_pct = profile_trailing.get('gain_10_to_20', 8)

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
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=shares_to_sell,
                                price=price,
                                reason=f"TRAILING STOP (PARTIAL {partial_sell_pct}%): Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%)",
                                score=current_score,
                                priority=2,
                                is_partial=True,
                                sell_pct=partial_sell_pct
                            )
                            trade._signal_factors = {"sell_reason": "PARTIAL TRAILING", "gain_pct": round(gain_pct, 1), "drop_from_peak": round(drop_from_peak, 1), "sell_pct": partial_sell_pct}
                            sells.append(trade)
                            # Reset peak to current price so remaining shares get a fresh wider stop
                            position.peak_price = price
                            position.peak_date = current_date
                        else:
                            # Standard: full sell
                            pyramid_note = f" (pyramid +{pyramid_widening:.0f}%)" if pyramid_widening > 0 else ""
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=position.shares,
                                price=price,
                                reason=f"TRAILING STOP: Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%){pyramid_note}",
                                score=current_score,
                                priority=2
                            )
                            trade._signal_factors = {"sell_reason": "TRAILING STOP", "gain_pct": round(gain_pct, 1), "drop_from_peak": round(drop_from_peak, 1)}
                            sells.append(trade)
                        continue

            # Score crash check with stability verification and profitability exception
            # Get score crash config
            score_crash_config = config.get('ai_trader.score_crash', {})
            consecutive_required = score_crash_config.get('consecutive_required', 3)
            score_threshold = score_crash_config.get('threshold', 50)
            drop_required = self.profile.get('score_crash_drop_required', score_crash_config.get('drop_required', 20))
            ignore_if_profitable_pct = self.profile.get('score_crash_ignore_if_profitable', score_crash_config.get('ignore_if_profitable_pct', 10))

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

                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"SCORE CRASH: {position.purchase_score:.0f} -> {current_score:.0f} "
                           f"(avg: {stability['avg_score']:.0f}, {stability['consecutive_low']} low scans)",
                    score=current_score,
                    priority=3
                )
                trade._signal_factors = {"sell_reason": "SCORE CRASH", "gain_pct": round(gain_pct, 1), "score_drop": round(score_drop, 1)}
                sells.append(trade)
                continue

            # PARTIAL PROFIT TAKING - let winners run while locking in gains
            # Thresholds from YAML config
            partial_taken = position.partial_profit_taken

            # Check for higher tier partial at +40% gain (highest priority partial)
            if gain_pct >= pp_40_gain and current_score >= pp_40_min_score and partial_taken < pp_40_sell:
                take_pct = pp_40_sell - partial_taken  # Take what's left to get to target
                if take_pct > 0:
                    shares_to_sell = position.shares * (take_pct / 100)
                    trade = SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=shares_to_sell,
                        price=price,
                        reason=f"PARTIAL PROFIT {pp_40_sell}%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                        score=current_score,
                        priority=4,
                        is_partial=True,
                        sell_pct=take_pct
                    )
                    trade._signal_factors = {"sell_reason": "PARTIAL PROFIT", "gain_pct": round(gain_pct, 1), "sell_pct": take_pct}
                    sells.append(trade)
                    continue  # Don't add more sell signals for this position

            # Check for lower tier partial at +25% gain
            elif gain_pct >= pp_25_gain and current_score >= pp_25_min_score and partial_taken < pp_25_sell:
                shares_to_sell = position.shares * (pp_25_sell / 100)
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=shares_to_sell,
                    price=price,
                    reason=f"PARTIAL PROFIT {pp_25_sell}%: Up {gain_pct:.1f}%, score {current_score:.0f} still strong",
                    score=current_score,
                    priority=5,
                    is_partial=True,
                    sell_pct=pp_25_sell
                )
                trade._signal_factors = {"sell_reason": "PARTIAL PROFIT", "gain_pct": round(gain_pct, 1), "sell_pct": pp_25_sell}
                sells.append(trade)
                continue  # Don't add more sell signals for this position

            # TAKE PROFIT: Full sell at profile threshold if score declining significantly
            take_profit_pct = self.profile.get('take_profit_pct', 40.0)
            if gain_pct >= take_profit_pct and current_score < position.purchase_score - 15:
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"TAKE PROFIT: Up {gain_pct:.1f}%, score declining significantly ({position.purchase_score:.0f} -> {current_score:.0f})",
                    score=current_score,
                    priority=7
                )
                trade._signal_factors = {"sell_reason": "TAKE PROFIT", "gain_pct": round(gain_pct, 1)}
                sells.append(trade)
                continue

            # Protect gains - winners with weak scores
            if gain_pct >= 20 and current_score < self.backtest.sell_score_threshold:
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({current_score:.0f})",
                    score=current_score,
                    priority=6
                )
                trade._signal_factors = {"sell_reason": "PROTECT GAINS", "gain_pct": round(gain_pct, 1)}
                sells.append(trade)
                continue

            # Weak flat positions
            if gain_pct < 10 and current_score < self.backtest.sell_score_threshold:
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"WEAK POSITION: {gain_pct:+.1f}%, score {current_score:.0f}",
                    score=current_score,
                    priority=6
                )
                trade._signal_factors = {"sell_reason": "WEAK POSITION", "gain_pct": round(gain_pct, 1)}
                sells.append(trade)

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

        # Regime-based position sizing (matches live trader get_market_regime())
        bullish_max_pct = regime_config.get('bullish_max_position_pct', 15.0)
        bearish_max_pct = regime_config.get('bearish_max_position_pct', 8.0)
        neutral_max_pct = regime_config.get('neutral_max_position_pct', 12.0)

        # Use strategy profile min_score (falls back to backtest record value)
        profile_min_score = self.profile.get('min_score', self.backtest.min_score_to_buy)

        if weighted_signal >= bullish_threshold:
            effective_min_score = profile_min_score - 5  # Easier in bull
            regime_max_pct = bullish_max_pct
        elif weighted_signal <= bearish_threshold:
            effective_min_score = profile_min_score + bearish_adj  # Harder in bear
            regime_max_pct = bearish_max_pct
        else:
            effective_min_score = profile_min_score
            regime_max_pct = neutral_max_pct

        # Detect if earnings data is limited (historical backtests lose C/A scores)
        # When _filter_available_earnings strips too many quarters, C=0 and A=0 for
        # all stocks. Without these 30 points, the normal threshold (72) is unreachable.
        max_c_in_universe = max((data.get('c_score', 0) for _, data in scores.items()), default=0)
        max_a_in_universe = max((data.get('a_score', 0) for _, data in scores.items()), default=0)
        earnings_data_limited = (max_c_in_universe < 5 and max_a_in_universe < 5)

        if earnings_data_limited:
            # C+A worth up to 30 points — reduce threshold to match available range
            # This mirrors seeding logic which uses threshold 35 instead of 72
            effective_min_score = max(effective_min_score - 30, 35)
            logger.debug(f"Earnings data limited (max C={max_c_in_universe:.0f}, A={max_a_in_universe:.0f}), "
                         f"adjusted min_score to {effective_min_score:.0f}")

        # PERCENTILE-BASED THRESHOLD: Adapt to score distribution
        # Use the lower of (regime-adjusted threshold, top 5% percentile)
        # This ensures we always have candidates even when scores are compressed
        percentile_pct = config.get('ai_trader.allocation.percentile_threshold_pct', 5)
        score_floor = config.get('ai_trader.allocation.percentile_score_floor', 45)

        all_scores_list = sorted([data["total_score"] for _, data in scores.items()
                                  if data["total_score"] > 0], reverse=True)
        if all_scores_list:
            top_idx = max(1, len(all_scores_list) * percentile_pct // 100)
            percentile_threshold = all_scores_list[min(top_idx - 1, len(all_scores_list) - 1)]
            # Use the lower of regime-adjusted and percentile, floored at score_floor
            effective_min_score = max(score_floor, min(effective_min_score, percentile_threshold))

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
            pivot_price = score_data.get("pivot_price", 0)
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
            # Strategy profile overrides YAML defaults
            quality_config = config.get('ai_trader.quality_filters', {})
            profile_quality = self.profile.get('quality_filters', {})
            min_c_score = profile_quality.get('min_c_score', quality_config.get('min_c_score', 10))
            min_l_score = profile_quality.get('min_l_score', quality_config.get('min_l_score', 8))
            min_volume_ratio = quality_config.get('min_volume_ratio', 1.2)
            skip_growth = quality_config.get('skip_in_growth_mode', True)

            # Get individual scores from score_data
            c_score = score_data.get('c', 0) or score_data.get('c_score', 0)
            l_score = score_data.get('l', 0) or score_data.get('l_score', 0)
            volume_ratio = score_data.get('volume_ratio', 1.0) or 1.0
            is_growth_stock = score_data.get('is_growth_stock', False)

            # Skip if not meeting quality thresholds (unless growth stock or limited earnings)
            if not (is_growth_stock and skip_growth) and not earnings_data_limited:
                if c_score < min_c_score:
                    logger.debug(f"Skipping {ticker}: C score {c_score} < {min_c_score}")
                    continue
                if l_score < min_l_score:
                    logger.debug(f"Skipping {ticker}: L score {l_score} < {min_l_score}")
                    continue
            elif earnings_data_limited:
                if l_score < min_l_score:
                    continue

            # VOLUME GATE: Only buy when volume confirms interest
            vol_gate_config = config.get('volume_gate', {})
            if vol_gate_config.get('enabled', True):
                if score_data.get('is_breaking_out', False):
                    vol_threshold = vol_gate_config.get('breakout_min_volume_ratio', 1.5)
                elif has_base and pct_from_pivot is not None and 0 <= pct_from_pivot <= 15:
                    vol_threshold = vol_gate_config.get('pre_breakout_min_volume_ratio', 0.8)
                else:
                    vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
                if volume_ratio < vol_threshold:
                    logger.debug(f"Skipping {ticker}: Volume ratio {volume_ratio:.2f} < {vol_threshold} (volume gate)")
                    continue
            elif volume_ratio < min_volume_ratio and not score_data.get('is_breaking_out', False):
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
            # Once a stock has broken out, the easy money is made - we're late to the party
            elif is_breaking_out:
                breakout_bonus = 10  # Reduced bonus - we prefer pre-breakout entries
                if volume_ratio >= 2.0:
                    breakout_bonus += 5  # Small bonus for strong volume
                momentum_score = 15  # Lower score - already extended

            # EXTENDED: More than 5% above pivot - the easy money is gone
            elif has_base and pivot_price > 0 and pct_from_pivot < -5:
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

            # RS LINE NEW HIGH: Leading indicator when RS makes new high before price
            rs_line_bonus = 0
            rs_line_config = config.get('rs_line', {})
            if rs_line_config.get('enabled', True):
                rs_line_data = self.data_provider.get_rs_line_new_high(ticker, current_date)
                if rs_line_data.get("rs_leading"):
                    rs_line_bonus = rs_line_config.get('bonus_points', 8)
                elif rs_line_data.get("rs_lagging"):
                    rs_line_bonus = rs_line_config.get('penalty_divergence', -5)

            # EARNINGS SURPRISE DRIFT: Post-earnings momentum for big beats
            earnings_drift_bonus = 0
            drift_config = config.get('earnings_drift', {})
            if drift_config.get('enabled', True):
                static_data = self.static_data.get(ticker, {})
                days_to_earnings = static_data.get('days_to_earnings')
                beat_streak = static_data.get('earnings_beat_streak', 0)
                # If stock has recent strong earnings beat (proxy: high beat streak + near earnings)
                if beat_streak >= 3 and days_to_earnings is not None:
                    # Post-earnings drift window: within 60 days after earnings
                    # Proxy: if days_to_earnings < 0 (past), we're in drift window
                    # Since we don't have negative days_to_earnings, use beat_streak as signal
                    if beat_streak >= 4:
                        earnings_drift_bonus = drift_config.get('bonus_points', 5)
                    elif beat_streak >= 3:
                        earnings_drift_bonus = 3

            # ANALYST REVISION BONUS: Reward stocks where analysts are raising estimates
            estimate_revision_bonus = 0
            revision_pct = self.static_data.get(ticker, {}).get('eps_estimate_revision_pct')
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

            # MOMENTUM CONFIRMATION: Penalize stocks where recent momentum is fading
            # If 3-month RS is significantly weaker than 12-month RS, momentum is weakening
            rs_12m = score_data.get("rs_12m", 1.0)
            rs_3m = score_data.get("rs_3m", 1.0)
            momentum_penalty = 0
            if rs_12m > 0 and rs_3m < rs_12m * 0.95:
                # Recent momentum fading - apply 15% penalty to composite
                momentum_penalty = -0.15

            # Composite score with strategy-specific weights
            scoring_weights = self.profile.get('scoring_weights', {})
            w_growth = scoring_weights.get('growth_projection', 0.25)
            w_score = scoring_weights.get('canslim_score', 0.25)
            w_momentum = scoring_weights.get('momentum', 0.20)
            w_breakout = scoring_weights.get('breakout', 0.20)
            w_base = scoring_weights.get('base_quality', 0.10)

            # Growth mode: weight C/L/N scores higher in the CANSLIM component
            effective_score = score
            if self.strategy == "growth":
                c_weight = self.profile.get('c_score_weight', 1.0)
                l_weight = self.profile.get('l_score_weight', 1.0)
                n_weight = self.profile.get('n_score_weight', 1.0)
                c_sc = score_data.get('c_score', 0)
                l_sc = score_data.get('l_score', 0)
                n_sc = score_data.get('n_score', 0)
                # Add the weighted bonus on top of total score
                effective_score += c_sc * (c_weight - 1.0) + l_sc * (l_weight - 1.0) + n_sc * (n_weight - 1.0)

            growth_projection = min(score_data.get("projected_growth", score * 0.3), 50)
            composite_score = (
                (growth_projection * w_growth) +
                (effective_score * w_score) +
                (momentum_score * w_momentum) +
                ((breakout_bonus + pre_breakout_bonus) * w_breakout) +
                (base_quality_bonus * w_base) +
                extended_penalty +
                coiled_spring_bonus +   # Earnings catalyst bonus
                rs_line_bonus +         # RS line new high bonus
                earnings_drift_bonus +  # Post-earnings drift bonus
                estimate_revision_bonus # Analyst estimate revisions
            )

            # Apply momentum penalty after base composite calculation
            if momentum_penalty < 0:
                composite_score *= (1 + momentum_penalty)  # Reduce by 15%

            # FTD penalty: no confirmed uptrend → mild score reduction (advisory, not blocking)
            if self.ftd_penalty_active:
                composite_score -= 8

            # Heat penalty: too much risk exposure → mild score reduction
            if self.heat_penalty_active:
                composite_score -= 5

            # Composite score used for ranking/priority only — CANSLIM score is the quality gate

            # Position sizing: conviction-based (higher scores get larger positions)
            max_positions = self.profile.get('max_positions', self.backtest.max_positions)
            conv_config = config.get('ai_trader.conviction_sizing', {})
            if conv_config.get('enabled', False):
                # Linear interpolation: score_floor → min_pct, score_ceiling → max_pct
                conv_min = conv_config.get('min_pct', 8.0)
                conv_max = conv_config.get('max_pct', 20.0)
                score_floor = conv_config.get('score_floor', 72)
                score_ceiling = conv_config.get('score_ceiling', 95)
                score_ratio = max(0, min(1, (composite_score - score_floor) / max(1, score_ceiling - score_floor)))
                position_pct = conv_min + score_ratio * (conv_max - conv_min)
            else:
                # Fallback: equal-weight floor with conviction scaling
                min_position_pct = 90.0 / max_positions
                conviction_multiplier = min(composite_score / 50, 1.5)
                conviction_pct = min_position_pct + (conviction_multiplier * (regime_max_pct - min_position_pct) / 1.5)
                position_pct = max(min_position_pct, conviction_pct)

            # Half-size positions when portfolio heat is elevated
            if self.heat_penalty_active:
                position_pct *= 0.50

            # PREDICTIVE POSITION SIZING: Pre-breakout stocks get largest positions
            # These are the ideal entries - before the crowd notices
            pre_breakout_mult = self.profile.get('pre_breakout_multiplier', 1.40)
            if pre_breakout_bonus >= 35 and has_base:
                position_pct *= pre_breakout_mult  # Larger for best pre-breakout entries
            elif pre_breakout_bonus >= 25 and has_base:
                position_pct *= (pre_breakout_mult * 0.93)  # Slightly less for good pre-breakout
            elif is_breaking_out and volume_ratio >= 1.5:
                position_pct *= 1.0   # No boost - already extended, entry is late

            # Coiled Spring position boost
            if coiled_spring_bonus > 0:
                cs_multiplier = cs_config.get('position_multiplier', 1.25)
                position_pct *= cs_multiplier

            # Reduce position size for bear market exception entries
            if score_data.get("_bear_market_entry"):
                position_pct *= bear_exception_position_mult

            # CORRELATION-AWARE SIZING: Reduce position if highly correlated with existing holdings
            corr_config = config.get('correlation_sizing', {})
            if corr_config.get('enabled', True) and self.positions:
                corr_threshold = corr_config.get('high_correlation_threshold', 0.70)
                corr_multiplier = corr_config.get('high_correlation_multiplier', 0.75)
                max_correlated = corr_config.get('max_correlated_positions', 3)
                lookback = corr_config.get('lookback_days', 30)

                high_corr_count = 0
                for held_ticker in list(self.positions.keys())[:5]:  # Check top 5 for performance
                    corr = self.data_provider.get_stock_correlation(
                        ticker, held_ticker, current_date, lookback
                    )
                    if corr > corr_threshold:
                        high_corr_count += 1

                if high_corr_count >= max_correlated:
                    logger.debug(f"{ticker}: {high_corr_count} correlated positions, skipping")
                    continue
                elif high_corr_count > 0:
                    position_pct *= corr_multiplier

            # Cap at profile max or market regime max AFTER all multipliers (matches live trader)
            profile_max_pct = self.profile.get('max_single_position_pct', 25)
            position_pct = min(position_pct, regime_max_pct, profile_max_pct)

            position_value = portfolio_value * (position_pct / 100)

            # Budget cash evenly across remaining position slots (matches live trader)
            remaining_slots = max(1, max_positions - len(self.positions))
            available_cash = self.cash * 0.90  # Keep 10% liquid buffer
            per_slot_budget = available_cash / remaining_slots
            # Allow high-conviction entries up to 1.3x the per-slot budget
            if pre_breakout_bonus >= 35 or is_breaking_out:
                cash_limit = per_slot_budget * 1.3
            else:
                cash_limit = per_slot_budget
            cash_limit = max(cash_limit, 500)  # Floor
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
            if estimate_revision_bonus >= 5:
                reason_parts.append(f"📊 Est↑ {revision_pct:+.0f}%")
            elif estimate_revision_bonus <= -5:
                reason_parts.append(f"📉 Est↓ {revision_pct:+.0f}%")

            # Build trade journal signal factors
            signal_factors = {
                "entry_type": "breakout" if is_breaking_out else ("pre-breakout" if pre_breakout_bonus >= 15 else "standard"),
                "market_regime": "bullish" if weighted_signal >= bullish_threshold else ("bearish" if weighted_signal <= bearish_threshold else "neutral"),
                "market_timing_state": self.market_timing_state,
                "rs_line_bonus": rs_line_bonus,
                "earnings_drift_bonus": earnings_drift_bonus,
                "estimate_revision_bonus": estimate_revision_bonus,
                "composite_score": round(composite_score, 1),
            }
            if coiled_spring_bonus > 0:
                signal_factors["coiled_spring"] = True
            if score_data.get("_bear_market_entry"):
                signal_factors["bear_exception"] = True

            buys.append(SimulatedTrade(
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                reason=" | ".join(reason_parts),
                score=score,
                priority=-composite_score  # Higher score = lower priority number
            ))
            # Stash signal_factors on trade for recording
            buys[-1]._signal_factors = signal_factors

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

            # SCALE-IN ON PULLBACKS: Add to winners pulling back on low volume
            scale_in_config = config.get('scale_in_pullbacks', {})
            if scale_in_config.get('enabled', True) and position.pyramid_count >= 2:
                min_gain = scale_in_config.get('min_gain_pct', 10.0)
                pullback_min = scale_in_config.get('pullback_pct', 3.0)
                pullback_max = scale_in_config.get('max_pullback_pct', 5.0)
                low_vol_ratio = scale_in_config.get('low_volume_ratio', 0.8)
                min_score_pullback = scale_in_config.get('min_score', 70)

                if gain_pct >= min_gain and current_score >= min_score_pullback:
                    # Check if pulling back from peak on low volume
                    if position.peak_price > 0:
                        drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                        vol_ratio = self.data_provider.get_volume_ratio(ticker, current_date)
                        if pullback_min <= drop_from_peak <= pullback_max and vol_ratio < low_vol_ratio:
                            add_pct = scale_in_config.get('add_pct', 30.0) / 100
                            original_cost = position.shares * position.cost_basis
                            scale_amount = min(original_cost * add_pct, self.cash * 0.3)
                            if scale_amount >= 100:
                                shares = scale_amount / price
                                pyramids.append(SimulatedTrade(
                                    ticker=ticker, action="PYRAMID", shares=shares,
                                    price=price,
                                    reason=f"SCALE-IN PULLBACK: +{gain_pct:.0f}% winner, -{drop_from_peak:.1f}% pullback, {vol_ratio:.1f}x vol",
                                    score=current_score, priority=-25
                                ))
                                continue  # Don't evaluate standard pyramid

            # O'Neil: add after 2-3% confirmation (lowered from 5%)
            if gain_pct < 2.5 or current_score < 70:
                continue

            # Max 2 pyramids per O'Neil 50/30/20 method
            if position.pyramid_count >= 2:
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

            # O'Neil 50/30/20: second add = 60% of original, third add = 40%
            original_cost = position.shares * position.cost_basis
            if position.pyramid_count == 0:
                pyramid_pct = 0.60  # ~30% of full position
            else:
                pyramid_pct = 0.40  # ~20% of full position
            pyramid_amount = original_cost * pyramid_pct

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
            sector=sector,
            signal_factors=getattr(trade, '_signal_factors', {})
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

        purchase_date = position.purchase_date

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
            realized_gain=realized_gain,
            purchase_date=purchase_date
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
                      cost_basis: float = None, realized_gain: float = None,
                      purchase_date: date = None):
        """Record trade in database"""
        position = self.positions.get(trade.ticker)
        holding_days = None
        realized_gain_pct = None

        if trade.action == "SELL" and cost_basis:
            sell_purchase_date = purchase_date or (position.purchase_date if position else None)
            holding_days = (current_date - sell_purchase_date).days if sell_purchase_date else 0
            realized_gain_pct = (realized_gain / (trade.shares * cost_basis)) * 100 if cost_basis else 0

        # Get signal_factors from trade (stashed during evaluation)
        signal_factors = getattr(trade, '_signal_factors', None)
        # For sells, get from position if available
        if trade.action == "SELL" and not signal_factors and position:
            signal_factors = position.signal_factors

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
            holding_days=holding_days,
            signal_factors=signal_factors
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
