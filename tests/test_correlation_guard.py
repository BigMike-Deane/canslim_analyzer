"""
Tests for position correlation guard functionality.

Tests the check_position_correlation function in ai_trader.py
and the _check_correlation method in backtester.py.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestCheckPositionCorrelation:
    """Test the live trader correlation check."""

    def test_empty_holdings_returns_neutral(self):
        """No held positions = no correlation to check."""
        from backend.ai_trader import check_position_correlation
        result = check_position_correlation('AAPL', [], threshold=0.70)
        assert result['max_corr'] == 0.0
        assert result['multiplier'] == 1.0

    @patch('yfinance.download')
    def test_download_failure_returns_neutral(self, mock_download):
        """If yfinance fails, return neutral (don't block the trade)."""
        from backend.ai_trader import check_position_correlation
        mock_download.side_effect = Exception("Network error")
        result = check_position_correlation('AAPL', ['MSFT'], threshold=0.70)
        assert result['max_corr'] == 0.0
        assert result['multiplier'] == 1.0

    @patch('yfinance.download')
    def test_high_correlation_reduces_multiplier(self, mock_download):
        """Stocks with >0.7 correlation should have multiplier < 1."""
        from backend.ai_trader import check_position_correlation
        import pandas as pd

        dates = pd.date_range(end='2026-03-20', periods=40, freq='B')
        base_prices = np.arange(100.0, 140.0)

        mock_data = pd.DataFrame({
            ('Close', 'AAPL'): base_prices,
            ('Close', 'MSFT'): base_prices * 1.05,  # Highly correlated
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data

        result = check_position_correlation('AAPL', ['MSFT'], threshold=0.70)
        assert result['max_corr'] > 0.9
        assert result['multiplier'] < 1.0
        assert result['most_correlated'] == 'MSFT'

    @patch('yfinance.download')
    def test_multiplier_floor_at_025(self, mock_download):
        """Even with perfect correlation, multiplier shouldn't go below 0.25."""
        from backend.ai_trader import check_position_correlation
        import pandas as pd

        dates = pd.date_range(end='2026-03-20', periods=40, freq='B')
        base_prices = np.arange(100.0, 140.0)

        mock_data = pd.DataFrame({
            ('Close', 'A'): base_prices,
            ('Close', 'B'): base_prices,  # Identical = correlation 1.0
        }, index=dates)
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_download.return_value = mock_data

        result = check_position_correlation('A', ['B'], threshold=0.70, reduction=0.50)
        assert result['multiplier'] >= 0.25


class TestBacktesterCorrelation:
    """Test the backtester's _check_correlation method."""

    def test_check_correlation_with_no_data(self):
        """Should return 0 correlation when price data unavailable."""
        from backend.backtester import BacktestEngine

        mock_db = MagicMock()
        mock_bt = MagicMock()
        mock_bt.stock_universe = "all"
        mock_bt.starting_cash = 25000
        mock_bt.min_score_to_buy = 72
        mock_bt.start_date = MagicMock()
        mock_bt.end_date = MagicMock()
        mock_bt.strategy = "nostate_optimized"

        mock_db.query.return_value.filter.return_value.first.return_value = mock_bt

        engine = BacktestEngine.__new__(BacktestEngine)
        engine.positions = {'MSFT': MagicMock()}
        engine.data_provider = MagicMock()
        engine.data_provider.get_price_series.return_value = None

        from datetime import date
        max_corr, ticker = engine._check_correlation('AAPL', date(2026, 3, 20), 30)
        assert max_corr == 0.0
        assert ticker is None

    def test_check_correlation_with_correlated_data(self):
        """Should detect high correlation between similar price series."""
        from backend.backtester import BacktestEngine
        from datetime import date

        engine = BacktestEngine.__new__(BacktestEngine)
        base = list(range(100, 130))
        engine.positions = {'MSFT': MagicMock()}
        engine.data_provider = MagicMock()
        engine.data_provider.get_price_series.side_effect = lambda ticker, d, lb: (
            base if ticker == 'AAPL' else [x * 1.5 for x in base]
        )

        max_corr, most_corr = engine._check_correlation('AAPL', date(2026, 3, 20), 30)
        assert max_corr > 0.99  # Nearly perfectly correlated
        assert most_corr == 'MSFT'
