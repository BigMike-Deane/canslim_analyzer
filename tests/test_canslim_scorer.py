"""
Unit tests for CANSLIM scoring logic
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from canslim_scorer import CANSLIMScorer, GrowthModeScorer
from data_fetcher import DataFetcher, StockData


class TestCANSLIMScorer:
    """Test CANSLIM scoring functionality"""

    def test_score_current_earnings_positive_growth(self, mock_stock_data, mock_data_fetcher):
        """Test C score with positive earnings growth"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Stock with good TTM growth (1.50+1.40+1.30+1.20 vs 1.10+1.00+0.95+0.90)
        score, detail = scorer._score_current_earnings(mock_stock_data)

        assert score > 0, "Should have positive C score for growing earnings"
        assert "TTM" in detail, "Detail should mention TTM"
        assert score <= 15, "C score should not exceed max (15 points)"

    def test_score_current_earnings_negative(self, mock_stock_data, mock_data_fetcher):
        """Test C score with negative earnings"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set all earnings to negative and worsening
        mock_stock_data.quarterly_earnings = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45]

        score, detail = scorer._score_current_earnings(mock_stock_data)

        # Score can be negative for worsening losses or 0
        assert score <= 0, "Worsening losses should result in 0 or negative score"
        assert "loss" in detail.lower(), "Detail should mention losses"

    def test_score_annual_earnings_good_cagr(self, mock_stock_data, mock_data_fetcher):
        """Test A score with good 3-year CAGR"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Annual earnings show strong growth: 5.50, 4.80, 4.20, 3.60, 3.00
        score, detail = scorer._score_annual_earnings(mock_stock_data)

        assert score > 0, "Should have positive A score"
        assert "CAGR" in detail, "Detail should mention CAGR"
        assert score <= 15, "A score should not exceed max (15 points)"

    def test_score_annual_earnings_with_high_roe(self, mock_stock_data, mock_data_fetcher):
        """Test A score bonus for high ROE"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set high ROE (28%)
        mock_stock_data.roe = 0.28

        score, detail = scorer._score_annual_earnings(mock_stock_data)

        # Score is combination of CAGR + ROE, so just check it's positive and mentions ROE
        assert score > 0, "High ROE with positive CAGR should give positive score"
        assert "ROE" in detail, "Detail should mention ROE"

    def test_score_new_highs_at_52w_high(self, mock_stock_data, mock_data_fetcher):
        """Test N score when at 52-week high"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set price at 52-week high
        mock_stock_data.current_price = 180.0
        mock_stock_data.high_52w = 180.0

        score, detail = scorer._score_new_highs(mock_stock_data)

        assert score >= 12, "At 52-week high should score high"
        assert "high" in detail.lower(), "Detail should mention high"

    def test_score_new_highs_far_from_high(self, mock_stock_data, mock_data_fetcher):
        """Test N score when far from 52-week high"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set price 40% below high
        mock_stock_data.current_price = 108.0  # 40% below 180
        mock_stock_data.high_52w = 180.0

        score, detail = scorer._score_new_highs(mock_stock_data)

        assert score == 0, "Far from high should score 0"

    def test_score_supply_demand_with_volume_surge(self, mock_stock_data, mock_data_fetcher):
        """Test S score with high volume"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set current volume to 2x average
        mock_stock_data.current_volume = 100_000_000
        mock_stock_data.avg_volume_50d = 50_000_000

        score, detail = scorer._score_supply_demand(mock_stock_data)

        assert score > 5, "High volume should increase S score"
        assert "vol" in detail.lower(), "Detail should mention volume"

    def test_score_leader_strong_rs(self, mock_stock_data, mock_data_fetcher):
        """Test L score with strong relative strength"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Stock outperforming (prices go from 145 to 170)
        # SPY goes from 400 to 525
        # Stock return: (170-145)/145 = 17%
        # SPY return: (525-400)/400 = 31%
        # RS = 1.17/1.31 = 0.89

        score, detail = scorer._score_leader(mock_stock_data)

        assert score >= 0, "Should have non-negative L score"
        assert score <= 15, "L score should not exceed max (15 points)"
        assert "RS" in detail, "Detail should show RS value"

    def test_score_institutional_optimal_range(self, mock_stock_data, mock_data_fetcher):
        """Test I score in optimal institutional ownership range"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set to optimal range (20-60%)
        mock_stock_data.institutional_holders_pct = 45.0

        score, detail = scorer._score_institutional(mock_stock_data)

        assert score == 10, "Optimal range should score max (10 points)"
        assert "45" in detail or "inst" in detail.lower(), "Detail should show percentage"

    def test_score_institutional_too_high(self, mock_stock_data, mock_data_fetcher):
        """Test I score when institutional ownership too high"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        # Set to very high (overcrowded)
        mock_stock_data.institutional_holders_pct = 90.0

        score, detail = scorer._score_institutional(mock_stock_data)

        assert score < 10, "Too high institutional should reduce score"

    def test_full_canslim_score(self, mock_stock_data, mock_data_fetcher):
        """Test complete CANSLIM scoring"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        canslim_score = scorer.score_stock(mock_stock_data)

        # Verify structure
        assert canslim_score.ticker == "AAPL"
        assert canslim_score.total_score > 0, "Total score should be positive"
        assert canslim_score.total_score <= 100, "Total score should not exceed 100"

        # Verify all components are scored
        assert canslim_score.c_score >= 0
        assert canslim_score.a_score >= 0
        assert canslim_score.n_score >= 0
        assert canslim_score.s_score >= 0
        assert canslim_score.l_score >= 0
        assert canslim_score.i_score >= 0
        assert canslim_score.m_score >= 0

        # Verify total is sum of components
        total = (canslim_score.c_score + canslim_score.a_score + canslim_score.n_score +
                 canslim_score.s_score + canslim_score.l_score + canslim_score.i_score +
                 canslim_score.m_score)
        assert abs(canslim_score.total_score - total) < 0.1, "Total should equal sum of components"


class TestGrowthModeScorer:
    """Test Growth Mode scoring for pre-revenue stocks"""

    def test_should_use_growth_mode_for_negative_earnings(self, mock_growth_stock_data, mock_data_fetcher):
        """Test that growth mode is selected for loss-making stocks"""
        scorer = GrowthModeScorer(mock_data_fetcher)

        should_use = scorer.should_use_growth_mode(mock_growth_stock_data)

        assert should_use, "Should use growth mode for negative earnings"

    def test_should_not_use_growth_mode_for_low_growth(self, mock_stock_data, mock_data_fetcher):
        """Test that growth mode is NOT used for low revenue growth stocks"""
        scorer = GrowthModeScorer(mock_data_fetcher)

        # Set low revenue growth to ensure growth mode isn't triggered
        mock_stock_data.quarterly_revenue = [100, 98, 95, 92, 90]  # Declining revenue

        should_use = scorer.should_use_growth_mode(mock_stock_data)

        assert should_use == False, "Low growth profitable stocks should not use growth mode"

    def test_score_revenue_growth(self, mock_growth_stock_data, mock_data_fetcher):
        """Test R score for revenue growth"""
        scorer = GrowthModeScorer(mock_data_fetcher)

        # Revenue: 5M, 3M, 2M, 1M (YoY: 5M vs 1M = 400% growth)
        score, detail = scorer._score_revenue_growth(mock_growth_stock_data)

        assert score > 0, "Should have positive R score for growing revenue"
        assert score <= 20, "R score should not exceed max (20 points)"
        assert "Rev" in detail or "rev" in detail, "Detail should mention revenue"

    def test_score_funding_health(self, mock_growth_stock_data, mock_data_fetcher):
        """Test F score for funding/financial health"""
        scorer = GrowthModeScorer(mock_data_fetcher)

        # Has $50M cash, $5M debt
        score, detail = scorer._score_funding_health(mock_growth_stock_data)

        assert score > 0, "Should have positive F score with cash"
        assert score <= 15, "F score should not exceed max (15 points)"

    def test_full_growth_mode_score(self, mock_growth_stock_data, mock_data_fetcher):
        """Test complete Growth Mode scoring"""
        canslim_scorer = CANSLIMScorer(mock_data_fetcher)
        scorer = GrowthModeScorer(mock_data_fetcher, canslim_scorer)

        growth_score = scorer.score_stock(mock_growth_stock_data)

        # Verify structure
        assert growth_score.ticker == "LCTX"
        assert growth_score.is_growth_stock == True
        assert growth_score.total_score > 0, "Total score should be positive"
        assert growth_score.total_score <= 100, "Total score should not exceed 100"

        # Verify all components
        assert growth_score.r_score >= 0
        assert growth_score.f_score >= 0
        assert growth_score.n_score >= 0
        assert growth_score.s_score >= 0
        assert growth_score.l_score >= 0
        assert growth_score.i_score >= 0
        assert growth_score.m_score >= 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_insufficient_data(self, mock_data_fetcher):
        """Test scoring with insufficient data"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        stock = StockData("TEST")
        stock.quarterly_earnings = [1.0, 1.1]  # Only 2 quarters
        stock.annual_earnings = [4.0]  # Only 1 year
        stock.current_price = 100.0
        stock.high_52w = 120.0
        stock.is_valid = True

        score, detail = scorer._score_current_earnings(stock)

        assert score == 0, "Insufficient data should result in 0 score"

    def test_zero_price_data(self, mock_data_fetcher):
        """Test N score with zero prices"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        stock = StockData("TEST")
        stock.current_price = 0
        stock.high_52w = 0
        stock.is_valid = True

        score, detail = scorer._score_new_highs(stock)

        assert score == 0, "Zero prices should result in 0 score"
        assert "No price data" in detail

    def test_invalid_stock_data(self, mock_data_fetcher):
        """Test scoring with invalid stock"""
        scorer = CANSLIMScorer(mock_data_fetcher)

        stock = StockData("INVALID")
        stock.is_valid = False

        canslim_score = scorer.score_stock(stock)

        assert canslim_score.total_score == 0, "Invalid stock should score 0"
