"""
Historical Data Provider for Backtesting

Provides point-in-time stock data for historical simulations.
Key principle: Only use data that would have been available on each historical date.
"""

import logging
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Yahoo Finance chart API URL
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Market indexes for M score
MARKET_INDEXES = ["SPY", "QQQ", "DIA"]
MARKET_INDEX_WEIGHTS = {"SPY": 0.50, "QQQ": 0.30, "DIA": 0.20}

# Earnings report delay (days after quarter end when data becomes available)
EARNINGS_REPORT_DELAY_DAYS = 45


@dataclass
class HistoricalStockData:
    """Simplified stock data for backtesting"""
    ticker: str
    price: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    volume: float = 0.0
    avg_volume_50d: float = 0.0
    sector: str = ""
    name: str = ""

    # For CANSLIM scoring
    quarterly_earnings: List[float] = field(default_factory=list)
    annual_earnings: List[float] = field(default_factory=list)
    institutional_holders_pct: float = 0.0
    roe: float = 0.0
    analyst_target_price: float = 0.0
    num_analyst_opinions: int = 0

    # For growth mode
    quarterly_revenue: List[float] = field(default_factory=list)
    is_growth_stock: bool = False


class HistoricalDataProvider:
    """
    Provides historical stock data for backtesting.
    All data is preloaded and cached for efficient day-by-day simulation.
    """

    def __init__(self, tickers: List[str]):
        """
        Initialize with list of tickers to track.

        Args:
            tickers: List of stock tickers to include in the backtest
        """
        self.tickers = tickers

        # Price history cache: {ticker: DataFrame with columns [date, open, high, low, close, volume]}
        self._price_cache: Dict[str, pd.DataFrame] = {}

        # Market index price history (for M score)
        self._index_cache: Dict[str, pd.DataFrame] = {}

        # Static data cache (earnings, sector, etc.)
        self._static_cache: Dict[str, dict] = {}

        # Trading days in the backtest period
        self._trading_days: List[date] = []

        # Preloaded flag
        self._is_loaded = False

    def preload_data(self, start_date: date, end_date: date,
                     progress_callback=None) -> bool:
        """
        Preload all historical data for the backtest period.
        Should be called once before running the backtest.

        Args:
            start_date: First day of backtest
            end_date: Last day of backtest
            progress_callback: Optional callback(pct) for progress updates

        Returns:
            True if data loaded successfully
        """
        logger.info(f"Preloading historical data for {len(self.tickers)} tickers "
                    f"from {start_date} to {end_date}")

        # Need extra lookback for calculating 52-week high/low and moving averages
        lookback_start = start_date - timedelta(days=400)  # ~1.5 years for 52-week calcs

        total_tickers = len(self.tickers) + len(MARKET_INDEXES)
        loaded = 0

        # Load market indexes first (needed for M score)
        for index in MARKET_INDEXES:
            df = self._fetch_price_history(index, lookback_start, end_date)
            if df is not None and not df.empty:
                self._index_cache[index] = df
                logger.debug(f"Loaded {len(df)} days of {index} history")
            loaded += 1
            if progress_callback:
                progress_callback(loaded / total_tickers * 100)

        # Load stock price history in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_price_history, ticker, lookback_start, end_date): ticker
                for ticker in self.tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        self._price_cache[ticker] = df
                except Exception as e:
                    logger.warning(f"Failed to load {ticker}: {e}")

                loaded += 1
                if progress_callback:
                    progress_callback(loaded / total_tickers * 100)

        # Determine trading days from SPY (most complete data)
        if "SPY" in self._index_cache:
            spy_df = self._index_cache["SPY"]
            mask = (spy_df["date"] >= start_date) & (spy_df["date"] <= end_date)
            self._trading_days = sorted(spy_df.loc[mask, "date"].tolist())

        self._is_loaded = True
        logger.info(f"Loaded price data for {len(self._price_cache)} stocks, "
                    f"{len(self._trading_days)} trading days")

        return len(self._price_cache) > 0

    def _fetch_price_history(self, ticker: str, start_date: date,
                             end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())

            url = YAHOO_CHART_URL.format(ticker=ticker)
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
                "events": "history"
            }

            resp = requests.get(url, params=params, headers=YAHOO_HEADERS, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None

            timestamps = result[0].get("timestamp", [])
            quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
            adj_close = result[0].get("indicators", {}).get("adjclose", [{}])

            if not timestamps or not quotes:
                return None

            # Use adjusted close if available, otherwise regular close
            close_prices = quotes.get("close", [])
            if adj_close and adj_close[0].get("adjclose"):
                close_prices = adj_close[0]["adjclose"]

            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s").date,
                "open": quotes.get("open", []),
                "high": quotes.get("high", []),
                "low": quotes.get("low", []),
                "close": close_prices,
                "volume": quotes.get("volume", [])
            })

            # Remove rows with None values
            df = df.dropna(subset=["close"])
            df = df.reset_index(drop=True)

            return df

        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")
            return None

    def get_trading_days(self) -> List[date]:
        """Get list of trading days in the backtest period"""
        return self._trading_days.copy()

    def get_price_on_date(self, ticker: str, as_of_date: date) -> Optional[float]:
        """
        Get closing price for a specific date.

        Args:
            ticker: Stock ticker
            as_of_date: The date to get price for

        Returns:
            Closing price or None if not available
        """
        df = self._price_cache.get(ticker)
        if df is None:
            return None

        mask = df["date"] == as_of_date
        if mask.any():
            return float(df.loc[mask, "close"].iloc[0])

        # If exact date not found, get most recent prior date
        prior = df[df["date"] < as_of_date]
        if not prior.empty:
            return float(prior.iloc[-1]["close"])

        return None

    def get_volume_on_date(self, ticker: str, as_of_date: date) -> Optional[float]:
        """Get volume for a specific date"""
        df = self._price_cache.get(ticker)
        if df is None:
            return None

        mask = df["date"] == as_of_date
        if mask.any():
            return float(df.loc[mask, "volume"].iloc[0])

        return None

    def get_price_history_up_to(self, ticker: str, as_of_date: date,
                                 lookback_days: int = 252) -> pd.DataFrame:
        """
        Get price history up to (and including) as_of_date.

        Args:
            ticker: Stock ticker
            as_of_date: End date for the history
            lookback_days: Number of trading days to include

        Returns:
            DataFrame with price history
        """
        df = self._price_cache.get(ticker)
        if df is None:
            return pd.DataFrame()

        mask = df["date"] <= as_of_date
        history = df[mask].tail(lookback_days).copy()
        return history

    def get_52_week_high_low(self, ticker: str, as_of_date: date) -> Tuple[float, float]:
        """
        Calculate 52-week high and low as of a specific date.

        Returns:
            (high_52w, low_52w) or (0.0, 0.0) if not available
        """
        history = self.get_price_history_up_to(ticker, as_of_date, lookback_days=252)
        if history.empty:
            return 0.0, 0.0

        return float(history["high"].max()), float(history["low"].min())

    def get_50_day_avg_volume(self, ticker: str, as_of_date: date) -> float:
        """Calculate 50-day average volume as of a specific date"""
        history = self.get_price_history_up_to(ticker, as_of_date, lookback_days=50)
        if len(history) < 20:  # Need at least 20 days
            return 0.0

        return float(history["volume"].mean())

    def get_moving_average(self, ticker: str, as_of_date: date, period: int) -> float:
        """Calculate moving average as of a specific date"""
        history = self.get_price_history_up_to(ticker, as_of_date, lookback_days=period)
        if len(history) < period:
            return 0.0

        return float(history["close"].tail(period).mean())

    def get_market_direction(self, as_of_date: date) -> dict:
        """
        Calculate market direction (M score components) for a specific date.
        Uses historical MAs from index price history.

        Returns:
            {
                "spy": {"price": float, "ma_50": float, "ma_200": float, "signal": int},
                "qqq": {"price": float, "ma_50": float, "ma_200": float, "signal": int},
                "dia": {"price": float, "ma_50": float, "ma_200": float, "signal": int},
                "weighted_signal": float,
                "is_bullish": bool
            }
        """
        result = {"weighted_signal": 0.0, "is_bullish": False}

        for index in MARKET_INDEXES:
            df = self._index_cache.get(index)
            if df is None:
                result[index.lower()] = {"price": 0, "ma_50": 0, "ma_200": 0, "signal": 0}
                continue

            history = df[df["date"] <= as_of_date].tail(252)
            if len(history) < 50:
                result[index.lower()] = {"price": 0, "ma_50": 0, "ma_200": 0, "signal": 0}
                continue

            price = float(history.iloc[-1]["close"])
            ma_50 = float(history["close"].tail(50).mean())
            ma_200 = float(history["close"].tail(200).mean()) if len(history) >= 200 else ma_50

            # Calculate signal: -1 (bearish), 0 (neutral), 1 (bullish), 2 (strong bullish)
            signal = self._calculate_index_signal(price, ma_50, ma_200)

            result[index.lower()] = {
                "price": price,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "signal": signal
            }

        # Calculate weighted signal
        weighted = sum(
            result.get(idx.lower(), {}).get("signal", 0) * weight
            for idx, weight in MARKET_INDEX_WEIGHTS.items()
        )
        result["weighted_signal"] = weighted
        result["is_bullish"] = weighted > 0.5

        return result

    def _calculate_index_signal(self, price: float, ma_50: float, ma_200: float) -> int:
        """
        Calculate signal for a single index.
        Returns: -1 (bearish), 0 (neutral), 1 (bullish), 2 (strong bullish)
        """
        if price <= 0 or ma_200 <= 0:
            return 0

        above_200 = price > ma_200
        above_50 = price > ma_50 if ma_50 > 0 else True

        if above_200 and above_50:
            return 2  # Strong bullish
        elif above_200:
            return 1  # Bullish
        elif above_50:
            return 0  # Neutral
        else:
            return -1  # Bearish

    def get_relative_strength(self, ticker: str, as_of_date: date,
                               period_months: int = 12) -> float:
        """
        Calculate relative strength vs S&P 500 over a period.

        Returns:
            Relative strength ratio (1.0 = market performance, >1.0 = outperforming)
        """
        lookback_days = period_months * 21  # Approximate trading days per month

        stock_history = self.get_price_history_up_to(ticker, as_of_date, lookback_days)
        spy_history = self._index_cache.get("SPY")

        if stock_history.empty or spy_history is None:
            return 1.0

        spy_subset = spy_history[spy_history["date"] <= as_of_date].tail(lookback_days)

        if len(stock_history) < 20 or len(spy_subset) < 20:
            return 1.0

        # Calculate returns
        stock_start = float(stock_history.iloc[0]["close"])
        stock_end = float(stock_history.iloc[-1]["close"])
        spy_start = float(spy_subset.iloc[0]["close"])
        spy_end = float(spy_subset.iloc[-1]["close"])

        if stock_start <= 0 or spy_start <= 0:
            return 1.0

        stock_return = (stock_end - stock_start) / stock_start
        spy_return = (spy_end - spy_start) / spy_start

        # Relative strength: stock return / SPY return
        if spy_return == 0:
            return 1.0

        return (1 + stock_return) / (1 + spy_return)

    def get_stock_data_on_date(self, ticker: str, as_of_date: date,
                                static_data: dict = None) -> HistoricalStockData:
        """
        Get complete stock data for a specific date.

        Args:
            ticker: Stock ticker
            as_of_date: The date to get data for
            static_data: Optional pre-fetched static data (earnings, sector, etc.)

        Returns:
            HistoricalStockData with all fields populated
        """
        data = HistoricalStockData(ticker=ticker)

        # Price data
        data.price = self.get_price_on_date(ticker, as_of_date) or 0.0
        data.volume = self.get_volume_on_date(ticker, as_of_date) or 0.0
        data.high_52w, data.low_52w = self.get_52_week_high_low(ticker, as_of_date)
        data.avg_volume_50d = self.get_50_day_avg_volume(ticker, as_of_date)

        # Static data (from cache or provided)
        if static_data:
            data.sector = static_data.get("sector", "")
            data.name = static_data.get("name", ticker)
            data.institutional_holders_pct = static_data.get("institutional_holders_pct", 0.0)
            data.roe = static_data.get("roe", 0.0)
            data.analyst_target_price = static_data.get("analyst_target_price", 0.0)
            data.num_analyst_opinions = static_data.get("num_analyst_opinions", 0)

            # Filter earnings to only include data available on as_of_date
            all_quarterly = static_data.get("quarterly_earnings", [])
            all_annual = static_data.get("annual_earnings", [])

            data.quarterly_earnings = self._filter_available_earnings(
                all_quarterly, as_of_date, quarterly=True
            )
            data.annual_earnings = self._filter_available_earnings(
                all_annual, as_of_date, quarterly=False
            )

            # Revenue (same filtering)
            all_quarterly_rev = static_data.get("quarterly_revenue", [])
            data.quarterly_revenue = self._filter_available_earnings(
                all_quarterly_rev, as_of_date, quarterly=True
            )

        return data

    def _filter_available_earnings(self, earnings_list: List, as_of_date: date,
                                    quarterly: bool = True) -> List[float]:
        """
        Filter earnings to only include data that would have been available on as_of_date.

        Earnings are typically reported ~45 days after quarter end.
        The earnings list is ordered from most recent to oldest (index 0 = most recent).
        We need to skip earnings that wouldn't have been reported yet on as_of_date.

        For quarterly earnings: 4 quarters per year
        For annual earnings: 1 year per entry
        """
        if not earnings_list:
            return []

        # First, extract numeric values from the list
        numeric_values = []
        for entry in earnings_list:
            if isinstance(entry, (int, float)):
                numeric_values.append(float(entry))
            elif isinstance(entry, dict) and "eps" in entry:
                numeric_values.append(float(entry["eps"]))

        if not numeric_values:
            return []

        # Calculate how many periods to skip based on the time difference
        # The database has current earnings - we need to remove recent ones
        # that wouldn't have been available on as_of_date
        today = date.today()
        days_diff = (today - as_of_date).days

        if days_diff <= 0:
            # as_of_date is today or in the future, return all earnings
            return numeric_values

        if quarterly:
            # Quarterly earnings: ~91 days per quarter + 45 days reporting delay
            # So each quarter's earnings become available ~136 days after the previous
            # Simplified: skip 1 quarter for every ~90 days in the past
            periods_to_skip = max(0, (days_diff + EARNINGS_REPORT_DELAY_DAYS) // 91)
        else:
            # Annual earnings: 365 days per year + 45-90 days reporting delay
            # Skip 1 year for every ~365 days in the past
            periods_to_skip = max(0, (days_diff + EARNINGS_REPORT_DELAY_DAYS) // 365)

        # Skip the most recent entries that wouldn't have been available
        if periods_to_skip >= len(numeric_values):
            # No earnings would have been available - return empty
            return []

        # Return earnings that would have been available (skip recent ones)
        return numeric_values[periods_to_skip:]

    def has_data_for_ticker(self, ticker: str) -> bool:
        """Check if we have price data for a ticker"""
        return ticker in self._price_cache

    def get_available_tickers(self) -> List[str]:
        """Get list of tickers with available data"""
        return list(self._price_cache.keys())

    def filter_tickers(self, valid_tickers: List[str]):
        """
        Filter the price cache to only include specified tickers.
        Used for survivorship bias filtering - removes stocks that
        didn't have data at the start of the backtest period.
        """
        tickers_to_remove = [t for t in self._price_cache.keys() if t not in valid_tickers]
        for ticker in tickers_to_remove:
            del self._price_cache[ticker]
        logger.debug(f"Filtered price cache: {len(self._price_cache)} tickers remaining")

    def get_spy_price_on_date(self, as_of_date: date) -> float:
        """Get SPY price on a specific date (for benchmark calculations)"""
        df = self._index_cache.get("SPY")
        if df is None:
            return 0.0

        mask = df["date"] == as_of_date
        if mask.any():
            return float(df.loc[mask, "close"].iloc[0])

        # If exact date not found, get most recent prior date
        prior = df[df["date"] < as_of_date]
        if not prior.empty:
            return float(prior.iloc[-1]["close"])

        return 0.0

    def get_volume_ratio(self, ticker: str, as_of_date: date) -> float:
        """
        Calculate volume ratio (current volume / 50-day average).
        Values > 1.5 indicate unusual volume, good for breakout confirmation.
        """
        current_vol = self.get_volume_on_date(ticker, as_of_date)
        avg_vol = self.get_50_day_avg_volume(ticker, as_of_date)

        if not current_vol or not avg_vol or avg_vol <= 0:
            return 1.0

        return current_vol / avg_vol

    def get_weekly_price_history(self, ticker: str, as_of_date: date,
                                  weeks: int = 26) -> List[dict]:
        """
        Get weekly OHLC data for base pattern detection.
        Returns list of dicts with keys: high, low, close, volume

        Args:
            ticker: Stock ticker
            as_of_date: End date for the history
            weeks: Number of weeks to include
        """
        # Get daily data and aggregate to weekly
        lookback_days = weeks * 7  # More days than needed to ensure enough data
        daily = self.get_price_history_up_to(ticker, as_of_date, lookback_days)

        if daily.empty:
            return []

        # Group by week and aggregate
        daily = daily.copy()
        daily['week'] = pd.to_datetime(daily['date']).dt.to_period('W')

        weekly_data = []
        for week, group in daily.groupby('week'):
            if len(group) > 0:
                weekly_data.append({
                    'high': float(group['high'].max()),
                    'low': float(group['low'].min()),
                    'close': float(group.iloc[-1]['close']),
                    'volume': float(group['volume'].sum())
                })

        return weekly_data[-weeks:]  # Return most recent N weeks

    def detect_base_pattern(self, ticker: str, as_of_date: date) -> dict:
        """
        Detect consolidation base patterns for breakout detection.

        Returns:
            {
                "type": "flat"|"cup"|"cup_with_handle"|"double_bottom"|"none",
                "weeks": int,
                "pivot_price": float
            }
        """
        weekly_data = self.get_weekly_price_history(ticker, as_of_date, weeks=26)

        if len(weekly_data) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        # Try to detect flat base first (most common for CANSLIM)
        flat_base = self._detect_flat_base(weekly_data)
        if flat_base["type"] != "none":
            return flat_base

        # Try cup pattern
        cup = self._detect_cup_pattern(weekly_data)
        if cup["type"] != "none":
            return cup

        return {"type": "none", "weeks": 0, "pivot_price": 0.0}

    def _detect_flat_base(self, weekly_data: List[dict]) -> dict:
        """
        Detect flat base pattern: 5+ weeks where the OVERALL price range is tight (<15%).

        Per O'Neil's CANSLIM methodology:
        - A flat base is a sideways consolidation with <15% total range
        - Calculates (highest_high - lowest_low) / lowest_low across the window
        - Returns the longest valid flat base found
        """
        if len(weekly_data) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        recent_weeks = weekly_data[-15:]  # Look at last 15 weeks for longer bases

        if len(recent_weeks) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        best_base = {"type": "none", "weeks": 0, "pivot_price": 0.0}

        # Try different window sizes from 5 to len(recent_weeks)
        # Find the longest valid flat base
        for window_size in range(5, len(recent_weeks) + 1):
            # Slide the window across recent weeks
            for start_idx in range(len(recent_weeks) - window_size + 1):
                window = recent_weeks[start_idx:start_idx + window_size]

                # Calculate the OVERALL range across all weeks in the window
                highest_high = max(w["high"] for w in window)
                lowest_low = min(w["low"] for w in window)

                if lowest_low <= 0:
                    continue

                # Total consolidation range as percentage
                total_range_pct = (highest_high - lowest_low) / lowest_low

                # A valid flat base has < 15% total range
                if total_range_pct < 0.15:
                    # Prefer longer bases (more significant)
                    if window_size > best_base["weeks"]:
                        best_base = {
                            "type": "flat",
                            "weeks": window_size,
                            "pivot_price": highest_high,
                            "base_low": lowest_low,
                            "base_depth": round(total_range_pct * 100, 1)
                        }

        return best_base

    def _detect_cup_pattern(self, weekly_data: List[dict]) -> dict:
        """
        Detect cup pattern: 15%+ decline, then recovery to within 15% of prior high.
        """
        if len(weekly_data) < 10:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        # Find the high in first third of data
        first_third = weekly_data[:len(weekly_data) // 3]
        if not first_third:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        left_high = max(w["high"] for w in first_third)

        # Find lowest point in middle
        middle_data = weekly_data[len(weekly_data) // 4: 3 * len(weekly_data) // 4]
        if not middle_data:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        cup_low = min(w["low"] for w in middle_data)

        # Check for sufficient depth (15%+)
        if left_high > 0:
            depth_pct = (left_high - cup_low) / left_high
            if depth_pct < 0.15:
                return {"type": "none", "weeks": 0, "pivot_price": 0.0}
        else:
            return {"type": "none", "weeks": 0, "pivot_price": 0.0}

        # Check if price has recovered to within 15% of left high
        recent_high = max(w["high"] for w in weekly_data[-4:])
        recovery_pct = (left_high - recent_high) / left_high if left_high > 0 else 1.0

        if recovery_pct <= 0.15:
            return {
                "type": "cup",
                "weeks": len(weekly_data),
                "pivot_price": left_high
            }

        return {"type": "none", "weeks": 0, "pivot_price": 0.0}

    def is_breaking_out(self, ticker: str, as_of_date: date) -> Tuple[bool, float, dict]:
        """
        Check if stock is breaking out of a base pattern or near 52-week high with volume.

        Returns:
            (is_breaking_out, volume_ratio, base_pattern)
        """
        price = self.get_price_on_date(ticker, as_of_date)
        vol_ratio = self.get_volume_ratio(ticker, as_of_date)
        base_pattern = self.detect_base_pattern(ticker, as_of_date)

        if not price or price <= 0:
            return False, vol_ratio, base_pattern

        # Check for multi-day volume confirmation (last 3 days)
        history = self.get_price_history_up_to(ticker, as_of_date, lookback_days=5)
        avg_vol = self.get_50_day_avg_volume(ticker, as_of_date)

        multi_day_vol_score = 0
        if len(history) >= 3 and avg_vol > 0:
            recent_vols = history['volume'].tail(3).tolist()
            above_avg_days = sum(1 for v in recent_vols if v > avg_vol)
            multi_day_vol_score = above_avg_days * 25  # 0, 25, 50, or 75

        # Effective volume score
        effective_vol_score = max(vol_ratio * 50, multi_day_vol_score)

        # Check breakout from base pattern
        if base_pattern["type"] != "none" and base_pattern["pivot_price"] > 0:
            pivot = base_pattern["pivot_price"]
            pct_from_pivot = (price - pivot) / pivot if pivot > 0 else 0

            # EXTENDED CHECK: If stock is >5% above pivot, it's NO LONGER a breakout
            # The buy point has passed - don't chase extended stocks
            if pct_from_pivot > 0.05:
                return False, vol_ratio, base_pattern

            # Active breakout: 0-5% above pivot with volume confirmation
            # This is the optimal buy zone per CANSLIM methodology
            if 0 <= pct_from_pivot <= 0.05 and effective_vol_score >= 50:
                return True, vol_ratio, base_pattern

            # Pre-breakout: within 3% below pivot, building volume
            if -0.03 <= pct_from_pivot < 0 and effective_vol_score >= 40:
                return True, vol_ratio, base_pattern

        # Check for breakout near 52-week high (no base pattern)
        # Be VERY strict - only mark as breakout with exceptional volume at new highs
        high_52w, _ = self.get_52_week_high_low(ticker, as_of_date)
        if high_52w > 0:
            pct_from_high = (high_52w - price) / high_52w
            # Require: within 2% of 52-week high AND exceptional volume (2.5x+)
            # True breakouts without detected bases should be rare
            if pct_from_high <= 0.02 and vol_ratio >= 2.5:
                return True, vol_ratio, base_pattern

        return False, vol_ratio, base_pattern
