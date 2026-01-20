"""
Data Fetcher Module
Wrapper around yfinance with caching and error handling
Now with Financial Modeling Prep (FMP) API for earnings data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import requests
import os

# FMP API Configuration - using new /stable/ endpoints
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_fmp_profile(ticker: str) -> dict:
    """Fetch company profile from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/profile?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                profile = data[0]
                return {
                    "name": profile.get("companyName", ""),
                    "sector": profile.get("sector", ""),
                    "industry": profile.get("industry", ""),
                    "market_cap": profile.get("mktCap", 0),
                    "current_price": profile.get("price", 0),
                    "high_52w": profile.get("range", "").split("-")[-1].strip() if profile.get("range") else 0,
                    "shares_outstanding": profile.get("sharesOutstanding", 0) or 0,
                }
    except Exception as e:
        print(f"FMP profile error for {ticker}: {e}")
    return {}


def fetch_fmp_quote(ticker: str) -> dict:
    """Fetch current quote data from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/quote?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                quote = data[0]
                return {
                    "current_price": quote.get("price", 0),
                    "high_52w": quote.get("yearHigh", 0),
                    "low_52w": quote.get("yearLow", 0),
                    "volume": quote.get("volume", 0),
                    "avg_volume": quote.get("avgVolume", 0),
                    "market_cap": quote.get("marketCap", 0),
                    "pe": quote.get("pe", 0),
                    "shares_outstanding": quote.get("sharesOutstanding", 0),
                }
    except Exception as e:
        print(f"FMP quote error for {ticker}: {e}")
    return {}


def fetch_fmp_key_metrics(ticker: str) -> dict:
    """Fetch key metrics including ROE, PE, and other valuation data from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                metrics = data[0]
                return {
                    "roe": metrics.get("returnOnEquity", 0) or 0,
                    "roa": metrics.get("returnOnAssets", 0) or 0,
                    "roic": metrics.get("returnOnInvestedCapital", 0) or 0,
                    "current_ratio": metrics.get("currentRatio", 0) or 0,
                    "earnings_yield": metrics.get("earningsYield", 0) or 0,
                    "fcf_yield": metrics.get("freeCashFlowYield", 0) or 0,
                }
    except Exception as e:
        print(f"FMP key metrics error for {ticker}: {e}")
    return {}


def fetch_fmp_earnings(ticker: str) -> dict:
    """Fetch quarterly and annual earnings from FMP"""
    if not FMP_API_KEY:
        print(f"FMP: No API key configured")
        return {}

    result = {"quarterly_eps": [], "annual_eps": []}

    try:
        # Quarterly income statement - using new /stable/ endpoint
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&period=quarter&limit=8&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        print(f"FMP quarterly {ticker}: status={resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if data:
                # EPS from income statement
                result["quarterly_eps"] = [q.get("eps", 0) or 0 for q in data]
                # Also get net income for backup
                result["quarterly_net_income"] = [q.get("netIncome", 0) or 0 for q in data]
                print(f"FMP quarterly {ticker}: got {len(data)} quarters, eps={result['quarterly_eps'][:3]}")
            else:
                print(f"FMP quarterly {ticker}: empty response")
        else:
            print(f"FMP quarterly {ticker}: error response: {resp.text[:200]}")
    except Exception as e:
        print(f"FMP quarterly earnings error for {ticker}: {e}")

    try:
        # Annual income statement - using new /stable/ endpoint
        url = f"{FMP_BASE_URL}/income-statement?symbol={ticker}&limit=5&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result["annual_eps"] = [a.get("eps", 0) or 0 for a in data]
                result["annual_net_income"] = [a.get("netIncome", 0) or 0 for a in data]
    except Exception as e:
        print(f"FMP annual earnings error for {ticker}: {e}")

    return result


def fetch_finviz_institutional(ticker: str) -> float:
    """Fetch institutional ownership percentage from Finviz (scraping)"""
    import re
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # Extract institutional ownership percentage from HTML
            match = re.search(r'Inst Own</td><td[^>]*><b>([0-9.]+)%', resp.text)
            if match:
                pct = float(match.group(1))
                print(f"Finviz inst ownership for {ticker}: {pct:.1f}%")
                return pct
    except Exception as e:
        print(f"Finviz institutional error for {ticker}: {e}")
    return 0.0


def fetch_fmp_institutional(ticker: str) -> float:
    """Fetch institutional ownership percentage from FMP, fallback to Finviz"""
    # Try FMP first
    if FMP_API_KEY:
        try:
            url = f"{FMP_BASE_URL}/institutional-holder?symbol={ticker}&apikey={FMP_API_KEY}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    # Sum up institutional shares and compare to outstanding
                    total_inst_shares = sum(h.get("shares", 0) or 0 for h in data[:50])  # Top 50 holders
                    if total_inst_shares > 0:
                        return total_inst_shares
        except Exception as e:
            print(f"FMP institutional error for {ticker}: {e}")

    # Fallback to Finviz (more reliable than Yahoo Finance from servers)
    return fetch_finviz_institutional(ticker)


def fetch_fmp_analyst(ticker: str) -> dict:
    """Fetch analyst ratings and price targets from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/analyst-estimates?symbol={ticker}&limit=1&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                est = data[0]
                return {
                    "estimated_eps_avg": est.get("estimatedEpsAvg", 0),
                    "estimated_eps_high": est.get("estimatedEpsHigh", 0),
                    "estimated_eps_low": est.get("estimatedEpsLow", 0),
                    "estimated_revenue_avg": est.get("estimatedRevenueAvg", 0),
                    "num_analysts": est.get("numberAnalystsEstimatedEps", 0),
                }
    except Exception as e:
        print(f"FMP analyst error for {ticker}: {e}")
    return {}


def fetch_fmp_price_target(ticker: str) -> dict:
    """Fetch analyst price targets from FMP"""
    if not FMP_API_KEY:
        return {}

    try:
        url = f"{FMP_BASE_URL}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                pt = data[0]
                return {
                    "target_high": pt.get("targetHigh", 0),
                    "target_low": pt.get("targetLow", 0),
                    "target_consensus": pt.get("targetConsensus", 0),
                    "target_median": pt.get("targetMedian", 0),
                }
    except Exception as e:
        print(f"FMP price target error for {ticker}: {e}")
    return {}


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
        self.low_52w: float = 0.0
        self.market_cap: float = 0.0
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

        # Key financial metrics for improved scoring
        self.roe: float = 0.0  # Return on Equity
        self.roa: float = 0.0  # Return on Assets
        self.roic: float = 0.0  # Return on Invested Capital
        self.earnings_yield: float = 0.0
        self.fcf_yield: float = 0.0


class DataFetcher:
    """Fetches and caches stock data using FMP API and Yahoo chart API"""

    def __init__(self):
        self._cache: dict[str, StockData] = {}
        self._sp500_history: Optional[pd.DataFrame] = None

    def get_stock_data(self, ticker: str, retries: int = 2) -> StockData:
        """
        Fetch all required data for a stock.
        Uses FMP API for earnings/fundamentals, Yahoo chart API for price history.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        stock_data = StockData(ticker)

        # 1. Get price history from Yahoo chart API (reliable)
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

        # 2. Get company profile and quote from FMP
        if FMP_API_KEY:
            profile = fetch_fmp_profile(ticker)
            if profile:
                stock_data.name = profile.get("name") or stock_data.name
                stock_data.sector = profile.get("sector", "")
                stock_data.shares_outstanding = int(profile.get("shares_outstanding", 0) or 0)
                if not stock_data.current_price:
                    stock_data.current_price = profile.get("current_price", 0)
                if not stock_data.high_52w:
                    try:
                        stock_data.high_52w = float(profile.get("high_52w", 0) or 0)
                    except:
                        pass

            quote = fetch_fmp_quote(ticker)
            if quote:
                if not stock_data.current_price:
                    stock_data.current_price = quote.get("current_price", 0)
                if not stock_data.high_52w:
                    stock_data.high_52w = quote.get("high_52w", 0)
                if not stock_data.low_52w:
                    stock_data.low_52w = quote.get("low_52w", 0)
                if not stock_data.market_cap:
                    stock_data.market_cap = quote.get("market_cap", 0)
                if not stock_data.avg_volume_50d:
                    stock_data.avg_volume_50d = quote.get("avg_volume", 0)
                if not stock_data.current_volume:
                    stock_data.current_volume = quote.get("volume", 0)
                stock_data.trailing_pe = quote.get("pe", 0) or 0
                if not stock_data.shares_outstanding:
                    stock_data.shares_outstanding = int(quote.get("shares_outstanding", 0) or 0)

            # 3. Get earnings data from FMP (critical for C and A scores)
            earnings = fetch_fmp_earnings(ticker)
            if earnings:
                stock_data.quarterly_earnings = earnings.get("quarterly_eps", [])
                stock_data.annual_earnings = earnings.get("annual_eps", [])
                print(f"FMP {ticker}: quarterly_earnings={stock_data.quarterly_earnings[:3] if stock_data.quarterly_earnings else 'EMPTY'}")

            # 3b. Get key metrics (ROE, etc.) from FMP
            key_metrics = fetch_fmp_key_metrics(ticker)
            if key_metrics:
                stock_data.roe = key_metrics.get("roe", 0)
                stock_data.roa = key_metrics.get("roa", 0)
                stock_data.roic = key_metrics.get("roic", 0)
                stock_data.earnings_yield = key_metrics.get("earnings_yield", 0)
                stock_data.fcf_yield = key_metrics.get("fcf_yield", 0)

            # 4. Get institutional ownership from FMP (or Yahoo fallback)
            inst_result = fetch_fmp_institutional(ticker)
            if inst_result:
                # If result is > 100, it's likely shares from FMP - convert to percentage
                if inst_result > 100 and stock_data.shares_outstanding:
                    stock_data.institutional_holders_pct = (inst_result / stock_data.shares_outstanding) * 100
                else:
                    # Already a percentage (from Yahoo Finance fallback)
                    stock_data.institutional_holders_pct = inst_result
                # Cap at 100% in case of data issues
                stock_data.institutional_holders_pct = min(stock_data.institutional_holders_pct, 100)

            # 5. Get analyst data from FMP
            price_target = fetch_fmp_price_target(ticker)
            if price_target:
                stock_data.analyst_target_price = price_target.get("target_consensus", 0) or price_target.get("target_median", 0)
                stock_data.analyst_target_high = price_target.get("target_high", 0)
                stock_data.analyst_target_low = price_target.get("target_low", 0)

            analyst = fetch_fmp_analyst(ticker)
            if analyst:
                stock_data.num_analyst_opinions = analyst.get("num_analysts", 0)
                stock_data.earnings_growth_estimate = analyst.get("estimated_eps_avg", 0)

        # 3. Fallback to yfinance ONLY if FMP didn't provide critical data
        # Skip yfinance if we already have earnings from FMP (to avoid rate limits)
        if not stock_data.quarterly_earnings and not FMP_API_KEY:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                if info and info.get('regularMarketPrice'):
                    if not stock_data.sector:
                        stock_data.sector = info.get('sector', 'Unknown')
                    if not stock_data.shares_outstanding:
                        stock_data.shares_outstanding = info.get('sharesOutstanding', 0)
                    if not stock_data.institutional_holders_pct:
                        inst_pct = info.get('heldPercentInstitutions', 0)
                        stock_data.institutional_holders_pct = (inst_pct * 100) if inst_pct else 0

                    # Quarterly earnings from yfinance
                    if not stock_data.quarterly_earnings:
                        try:
                            quarterly = stock.quarterly_financials
                            if quarterly is not None and not quarterly.empty:
                                if 'Net Income' in quarterly.index:
                                    net_income = quarterly.loc['Net Income'].dropna()
                                    shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                                    stock_data.quarterly_earnings = (net_income / shares).tolist()[:8]
                        except Exception:
                            pass

                    # Annual earnings from yfinance
                    if not stock_data.annual_earnings:
                        try:
                            annual = stock.financials
                            if annual is not None and not annual.empty:
                                if 'Net Income' in annual.index:
                                    net_income = annual.loc['Net Income'].dropna()
                                    shares = stock_data.shares_outstanding if stock_data.shares_outstanding > 0 else 1
                                    stock_data.annual_earnings = (net_income / shares).tolist()[:5]
                        except Exception:
                            pass

            except Exception as e:
                stock_data.error_message = str(e)

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

        # Try Yahoo chart API first (more reliable from servers)
        chart_data = fetch_price_from_chart_api("SPY")
        if chart_data.get("close_prices"):
            close_prices = chart_data.get("close_prices", [])
            timestamps = chart_data.get("timestamps", [])
            if len(close_prices) >= 50:
                dates = pd.to_datetime(timestamps, unit='s')
                self._sp500_history = pd.DataFrame({
                    'Close': close_prices,
                }, index=dates)
                return self._sp500_history

        # Fallback to yfinance
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
