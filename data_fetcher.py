"""
Data Fetcher Module
Wrapper around yfinance with caching and error handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import requests


def fetch_price_from_chart_api(ticker: str) -> dict:
    """Fallback: fetch basic price data from Yahoo chart API"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": "1y"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                meta = result[0].get("meta", {})
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                timestamps = result[0].get("timestamp", [])

                return {
                    "current_price": meta.get("regularMarketPrice") or meta.get("previousClose"),
                    "high_52w": meta.get("fiftyTwoWeekHigh"),
                    "name": meta.get("longName") or meta.get("shortName") or ticker,
                    "close_prices": indicators.get("close", []),
                    "volumes": indicators.get("volume", []),
                    "timestamps": timestamps
                }
    except Exception:
        pass
    return {}


class StockData:
    """Container for all stock data needed for CANSLIM analysis"""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.name: str = ""
        self.sector: str = ""
        self.current_price: float = 0.0
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.quarterly_earnings: list[float] = []
        self.annual_earnings: list[float] = []
        self.institutional_holders_pct: float = 0.0
        self.shares_outstanding: int = 0
        self.avg_volume_50d: float = 0.0
        self.current_volume: float = 0.0
        self.high_52w: float = 0.0
        self.is_valid: bool = False
        self.error_message: str = ""

        # New fields for refined model
        self.analyst_target_price: float = 0.0
        self.analyst_target_low: float = 0.0
        self.analyst_target_high: float = 0.0
        self.analyst_recommendation: str = ""  # buy, hold, sell
        self.num_analyst_opinions: int = 0
        self.forward_pe: float = 0.0
        self.trailing_pe: float = 0.0
        self.peg_ratio: float = 0.0
        self.earnings_growth_estimate: float = 0.0  # Next year growth estimate


class DataFetcher:
    """Fetches and caches stock data from yfinance"""

    def __init__(self):
        self._cache: dict[str, StockData] = {}
        self._sp500_history: Optional[pd.DataFrame] = None

    def get_stock_data(self, ticker: str, retries: int = 2) -> StockData:
        """
        Fetch all required data for a stock.
        Uses caching to avoid redundant API calls.
        Falls back to chart API if yfinance fails.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        stock_data = StockData(ticker)

        # First try the direct chart API (more reliable from servers)
        chart_data = fetch_price_from_chart_api(ticker)
        if chart_data.get("current_price"):
            stock_data.current_price = chart_data["current_price"]
            stock_data.high_52w = chart_data.get("high_52w", 0) or 0
            stock_data.name = chart_data.get("name", ticker)

            # Build price history from chart data
            close_prices = chart_data.get("close_prices", [])
            volumes = chart_data.get("volumes", [])
            timestamps = chart_data.get("timestamps", [])

            if len(close_prices) >= 50:
                dates = pd.to_datetime(timestamps, unit='s')
                stock_data.price_history = pd.DataFrame({
                    'Close': close_prices,
                    'Volume': volumes
                }, index=dates)

                # Calculate volume metrics from chart data
                if volumes:
                    recent_volumes = [v for v in volumes[-50:] if v]
                    stock_data.avg_volume_50d = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
                    stock_data.current_volume = volumes[-1] if volumes[-1] else 0

        # Try yfinance for additional data (earnings, institutional, etc.)
        for attempt in range(retries):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Only update if we got valid data
                if info and info.get('regularMarketPrice'):
                    # Basic info (update if not set from chart API)
                    if not stock_data.name or stock_data.name == ticker:
                        stock_data.name = info.get('longName', info.get('shortName', ticker))
                    stock_data.sector = info.get('sector', 'Unknown')
                    if not stock_data.current_price:
                        stock_data.current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    stock_data.shares_outstanding = info.get('sharesOutstanding', 0)
                    if not stock_data.high_52w:
                        stock_data.high_52w = info.get('fiftyTwoWeekHigh', 0)

                    # Institutional ownership
                    inst_pct = info.get('heldPercentInstitutions', 0)
                    stock_data.institutional_holders_pct = (inst_pct * 100) if inst_pct else 0

                    # Volume data (if not set from chart)
                    if not stock_data.avg_volume_50d:
                        stock_data.avg_volume_50d = info.get('averageVolume', 0)
                    if not stock_data.current_volume:
                        stock_data.current_volume = info.get('volume', info.get('regularMarketVolume', 0))

                    # Analyst data
                    stock_data.analyst_target_price = info.get('targetMeanPrice', 0) or 0
                    stock_data.analyst_target_low = info.get('targetLowPrice', 0) or 0
                    stock_data.analyst_target_high = info.get('targetHighPrice', 0) or 0
                    stock_data.num_analyst_opinions = info.get('numberOfAnalystOpinions', 0) or 0

                    # Map recommendation key to readable string
                    rec_key = info.get('recommendationKey', '')
                    rec_map = {'strong_buy': 'strong buy', 'buy': 'buy', 'hold': 'hold',
                               'sell': 'sell', 'strong_sell': 'strong sell'}
                    stock_data.analyst_recommendation = rec_map.get(rec_key, rec_key)

                    # Valuation metrics
                    stock_data.forward_pe = info.get('forwardPE', 0) or 0
                    stock_data.trailing_pe = info.get('trailingPE', 0) or 0
                    stock_data.peg_ratio = info.get('pegRatio', 0) or 0
                    stock_data.earnings_growth_estimate = info.get('earningsGrowth', 0) or 0

                # Price history from yfinance if chart API didn't work
                if stock_data.price_history.empty:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    history = stock.history(start=start_date, end=end_date)

                    if len(history) >= 50:
                        stock_data.price_history = history

                # Quarterly earnings (EPS)
                try:
                    quarterly = stock.quarterly_financials
                    if quarterly is not None and not quarterly.empty:
                        # Try to get Net Income and divide by shares
                        if 'Net Income' in quarterly.index:
                            net_income = quarterly.loc['Net Income'].dropna()
                            shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                            stock_data.quarterly_earnings = (net_income / shares).tolist()[:8]  # Last 8 quarters
                except Exception:
                    pass

                # If quarterly earnings failed, try earnings history
                if not stock_data.quarterly_earnings:
                    try:
                        earnings = stock.earnings_history
                        if earnings is not None and not earnings.empty and 'epsActual' in earnings.columns:
                            stock_data.quarterly_earnings = earnings['epsActual'].dropna().tolist()[-8:]
                    except Exception:
                        pass

                # Annual earnings
                try:
                    annual = stock.financials
                    if annual is not None and not annual.empty:
                        if 'Net Income' in annual.index:
                            net_income = annual.loc['Net Income'].dropna()
                            shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                            stock_data.annual_earnings = (net_income / shares).tolist()[:5]  # Last 5 years
                except Exception:
                    pass

                break  # Success with yfinance data

            except Exception as e:
                stock_data.error_message = str(e)
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retry

        # Mark as valid if we have basic price data (from chart API or yfinance)
        if stock_data.current_price and not stock_data.price_history.empty and len(stock_data.price_history) >= 50:
            stock_data.is_valid = True
        elif stock_data.current_price:
            # Partial data - still somewhat usable
            stock_data.is_valid = True
            stock_data.error_message = "Limited data available"

        self._cache[ticker] = stock_data
        return stock_data

    def get_sp500_history(self) -> pd.DataFrame:
        """Fetch S&P 500 index price history for relative strength calculation"""
        if self._sp500_history is not None:
            return self._sp500_history

        try:
            spy = yf.Ticker("SPY")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            self._sp500_history = spy.history(start=start_date, end=end_date)
        except Exception:
            self._sp500_history = pd.DataFrame()

        return self._sp500_history

    def get_market_direction(self) -> tuple[bool, float, float]:
        """
        Determine market direction based on S&P 500.
        Returns: (is_bullish, pct_above_200ma, pct_above_50ma)
        """
        history = self.get_sp500_history()

        if history.empty or len(history) < 200:
            return True, 0, 0  # Default to bullish if no data

        close = history['Close']
        current_price = close.iloc[-1]

        ma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        ma_50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.mean()

        pct_above_200 = ((current_price - ma_200) / ma_200) * 100
        pct_above_50 = ((current_price - ma_50) / ma_50) * 100

        is_bullish = current_price > ma_200

        return is_bullish, pct_above_200, pct_above_50

    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        self._sp500_history = None


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()

    print("Testing with AAPL...")
    data = fetcher.get_stock_data("AAPL")
    print(f"Name: {data.name}")
    print(f"Sector: {data.sector}")
    print(f"Current Price: ${data.current_price:.2f}")
    print(f"52-week High: ${data.high_52w:.2f}")
    print(f"Institutional %: {data.institutional_holders_pct:.1f}%")
    print(f"Quarterly Earnings: {data.quarterly_earnings[:4]}")
    print(f"Is Valid: {data.is_valid}")

    # New analyst data
    print(f"\nAnalyst Data:")
    print(f"  Target Price: ${data.analyst_target_price:.2f}")
    print(f"  Target Range: ${data.analyst_target_low:.2f} - ${data.analyst_target_high:.2f}")
    print(f"  Recommendation: {data.analyst_recommendation}")
    print(f"  # of Analysts: {data.num_analyst_opinions}")
    print(f"  Forward P/E: {data.forward_pe:.1f}")
    print(f"  PEG Ratio: {data.peg_ratio:.2f}")

    is_bullish, pct_200, pct_50 = fetcher.get_market_direction()
    print(f"\nMarket: {'Bullish' if is_bullish else 'Bearish'}")
    print(f"S&P 500 vs 200-day MA: {pct_200:+.1f}%")
    print(f"S&P 500 vs 50-day MA: {pct_50:+.1f}%")
