"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


@pytest.fixture
def mock_stock_data():
    """Create mock StockData for testing"""
    from data_fetcher import StockData
    import pandas as pd

    stock = StockData("AAPL")
    stock.ticker = "AAPL"
    stock.name = "Apple Inc."
    stock.sector = "Technology"
    stock.current_price = 150.0
    stock.high_52w = 180.0
    stock.low_52w = 120.0
    stock.avg_volume_50d = 50_000_000
    stock.current_volume = 60_000_000
    stock.shares_outstanding = 16_000_000_000
    stock.institutional_holders_pct = 45.0

    # Earnings data
    stock.quarterly_earnings = [1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90]
    stock.annual_earnings = [5.50, 4.80, 4.20, 3.60, 3.00]

    # Revenue data
    stock.quarterly_revenue = [90_000_000_000, 85_000_000_000, 80_000_000_000, 75_000_000_000]
    stock.annual_revenue = [350_000_000_000, 320_000_000_000, 290_000_000_000]

    # Financial metrics
    stock.roe = 0.28  # 28% ROE
    stock.trailing_pe = 25.0
    stock.peg_ratio = 1.5

    # Price history
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    prices = [145 + i * 0.1 for i in range(252)]
    volumes = [50_000_000] * 252

    stock.price_history = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    stock.is_valid = True

    return stock


@pytest.fixture
def mock_growth_stock_data():
    """Create mock growth stock (pre-revenue) for testing"""
    from data_fetcher import StockData
    import pandas as pd

    stock = StockData("LCTX")
    stock.ticker = "LCTX"
    stock.name = "Lineage Cell Therapeutics"
    stock.sector = "Healthcare"
    stock.current_price = 2.50
    stock.high_52w = 3.00
    stock.low_52w = 1.50
    stock.avg_volume_50d = 1_000_000
    stock.current_volume = 1_500_000
    stock.shares_outstanding = 100_000_000
    stock.institutional_holders_pct = 35.0

    # No earnings (pre-revenue)
    stock.quarterly_earnings = [-0.05, -0.06, -0.07, -0.08]
    stock.annual_earnings = [-0.25, -0.30, -0.35]

    # Revenue growing fast
    stock.quarterly_revenue = [5_000_000, 3_000_000, 2_000_000, 1_000_000]
    stock.annual_revenue = [10_000_000, 5_000_000, 2_000_000]

    # Balance sheet
    stock.cash_and_equivalents = 50_000_000
    stock.total_debt = 5_000_000

    # Price history
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    prices = [2.0 + i * 0.002 for i in range(252)]
    volumes = [1_000_000] * 252

    stock.price_history = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    }, index=dates)

    stock.is_valid = True

    return stock


@pytest.fixture
def mock_data_fetcher(monkeypatch):
    """Create mock DataFetcher for testing"""
    from data_fetcher import DataFetcher
    import pandas as pd

    fetcher = DataFetcher()

    # Mock S&P 500 history
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    spy_prices = [400 + i * 0.5 for i in range(252)]

    fetcher._sp500_history = pd.DataFrame({
        'Close': spy_prices
    }, index=dates)

    # Mock market direction
    def mock_market_direction():
        return True, 5.0, 3.0  # Bullish, +5% above 200MA, +3% above 50MA

    monkeypatch.setattr(fetcher, 'get_market_direction', mock_market_direction)

    return fetcher
