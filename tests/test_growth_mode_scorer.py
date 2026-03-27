"""
Comprehensive tests for GrowthModeScorer — alternative scoring for pre-revenue stocks.

Tests R (Revenue), F (Funding), should_use_growth_mode detection,
and integration with shared CANSLIM components (N, S, L, I, M).
"""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from canslim_scorer import GrowthModeScorer, CANSLIMScorer


def _make_growth_stock(**overrides):
    """Create a mock pre-revenue/growth stock."""
    from data_fetcher import StockData
    stock = StockData("GRWTH")
    stock.ticker = overrides.get("ticker", "GRWTH")
    stock.name = "Growth Corp"
    stock.sector = "Technology"
    stock.current_price = overrides.get("current_price", 15.0)
    stock.high_52w = overrides.get("high_52w", 20.0)
    stock.low_52w = overrides.get("low_52w", 5.0)
    stock.avg_volume_50d = overrides.get("avg_volume_50d", 2_000_000)
    stock.current_volume = overrides.get("current_volume", 3_000_000)
    stock.shares_outstanding = overrides.get("shares_outstanding", 100_000_000)
    stock.institutional_holders_pct = overrides.get("institutional_holders_pct", 30.0)
    stock.roe = overrides.get("roe", -0.20)  # Negative ROE = losing money
    stock.trailing_pe = None  # No PE for pre-revenue
    stock.earnings_growth_estimate = 0

    # Pre-revenue: negative earnings, strong revenue growth
    stock.quarterly_earnings = overrides.get("quarterly_earnings", [-0.15, -0.18, -0.20, -0.25])
    stock.annual_earnings = overrides.get("annual_earnings", [-0.70, -0.85, -1.00])
    stock.quarterly_revenue = overrides.get("quarterly_revenue",
        [50e6, 40e6, 32e6, 25e6, 20e6])  # 150% YoY growth
    stock.annual_revenue = overrides.get("annual_revenue", [150e6, 80e6, 40e6])

    # Price history
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    prices = [5 + i * 0.04 for i in range(252)]
    stock.price_history = pd.DataFrame({
        'Close': prices, 'Volume': [2_000_000] * 252
    }, index=dates)
    stock.weekly_price_history = pd.DataFrame()
    stock.is_valid = overrides.get("is_valid", True)
    return stock


def _make_scorer():
    mock_fetcher = MagicMock()
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    sp500 = pd.DataFrame({
        'Close': [100.0] * 252, 'Volume': [1e9] * 252
    }, index=dates)
    mock_fetcher.get_sp500_history.return_value = sp500
    canslim = CANSLIMScorer(mock_fetcher)
    return GrowthModeScorer(mock_fetcher, canslim)


class TestShouldUseGrowthMode:
    """Test the growth mode detection logic."""

    def test_pre_revenue_uses_growth_mode(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_earnings=[-0.1, -0.2, -0.3, -0.4])
        assert scorer.should_use_growth_mode(stock) is True

    def test_profitable_company_uses_canslim(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_earnings=[1.5, 1.4, 1.3, 1.2])
        assert scorer.should_use_growth_mode(stock) is False

    def test_no_earnings_data_uses_growth_mode(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_earnings=[])
        assert scorer.should_use_growth_mode(stock) is True

    def test_mixed_earnings_uses_canslim(self):
        """If some quarters are positive, use CANSLIM."""
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_earnings=[0.5, -0.1, 0.2, -0.3])
        assert scorer.should_use_growth_mode(stock) is False

    def test_all_zero_earnings_uses_growth_mode(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_earnings=[0, 0, 0, 0])
        assert scorer.should_use_growth_mode(stock) is True


class TestRevenueScore:
    """Test R (Revenue Growth) scoring — max 20 pts."""

    def test_strong_revenue_growth_scores_high(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_revenue=[100e6, 80e6, 65e6, 50e6, 40e6])
        score = scorer.score_stock(stock)
        assert score.r_score >= 12  # 150% YoY growth

    def test_no_revenue_data_scores_low(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_revenue=[], annual_revenue=[])
        score = scorer.score_stock(stock)
        assert score.r_score <= 5  # Little or no data = low score

    def test_declining_revenue_scores_low(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(quarterly_revenue=[30e6, 40e6, 50e6, 60e6, 70e6])
        score = scorer.score_stock(stock)
        assert score.r_score < 5


class TestFundingHealth:
    """Test F (Funding Health) scoring — max 15 pts."""

    def test_score_returns_float(self):
        scorer = _make_scorer()
        stock = _make_growth_stock()
        score = scorer.score_stock(stock)
        assert isinstance(score.f_score, (int, float))
        assert 0 <= score.f_score <= 15


class TestGrowthModeTotalScore:
    """Test total score computation."""

    def test_total_is_sum_of_components(self):
        scorer = _make_scorer()
        stock = _make_growth_stock()
        score = scorer.score_stock(stock)
        expected = (score.r_score + score.f_score + score.n_score +
                    score.s_score + score.l_score + score.i_score + score.m_score)
        assert score.total_score == pytest.approx(expected, abs=0.1)

    def test_max_score_is_100(self):
        """R(20)+F(15)+N(15)+S(15)+L(15)+I(10)+M(10) = 100"""
        scorer = _make_scorer()
        stock = _make_growth_stock()
        score = scorer.score_stock(stock)
        assert score.total_score <= 100

    def test_invalid_stock_gets_zero(self):
        scorer = _make_scorer()
        stock = _make_growth_stock(is_valid=False)
        score = scorer.score_stock(stock)
        assert score.total_score == 0

    def test_reuses_canslim_components(self):
        """N, S, L, I should use CANSLIM scorer internally."""
        scorer = _make_scorer()
        stock = _make_growth_stock()
        growth_score = scorer.score_stock(stock)
        canslim_score = scorer.canslim_scorer.score_stock(stock)
        # N, S, L, I should match CANSLIM
        assert growth_score.n_score == canslim_score.n_score
        assert growth_score.s_score == canslim_score.s_score
        assert growth_score.l_score == canslim_score.l_score
        assert growth_score.i_score == canslim_score.i_score

    def test_m_score_scaled_to_10(self):
        """Growth mode M score is scaled from 15 to 10 max."""
        scorer = _make_scorer()
        stock = _make_growth_stock()
        score = scorer.score_stock(stock)
        assert score.m_score <= 10
