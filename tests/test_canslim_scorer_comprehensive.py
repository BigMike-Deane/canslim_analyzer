"""
Comprehensive tests for canslim_scorer.py — the brain of the CANSLIM scoring system.

Covers all 7 CANSLIM components with edge cases:
- C: TTM growth, acceleration, surprise, beat streak, revisions, negative earnings
- A: CAGR, sector-adjusted thresholds, ROE bonus, turnarounds
- N: 52-week high proximity, breakout detection
- S: Supply/demand, volume ratio, shares outstanding
- L: Multi-timeframe RS, industry group bonus integration
- I: Institutional ownership
- M: Market direction
"""
import math
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from canslim_scorer import CANSLIMScorer, CANSLIMScore, _clean_earnings


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_stock_data(**overrides):
    """Create a mock StockData with sensible defaults."""
    from data_fetcher import StockData
    stock = StockData("TEST")
    stock.ticker = overrides.get("ticker", "TEST")
    stock.name = overrides.get("name", "Test Corp")
    stock.sector = overrides.get("sector", "Technology")
    stock.industry = overrides.get("industry", "Software")
    stock.current_price = overrides.get("current_price", 100.0)
    stock.high_52w = overrides.get("high_52w", 110.0)
    stock.low_52w = overrides.get("low_52w", 60.0)
    stock.avg_volume_50d = overrides.get("avg_volume_50d", 1_000_000)
    stock.current_volume = overrides.get("current_volume", 1_200_000)
    stock.shares_outstanding = overrides.get("shares_outstanding", 50_000_000)
    stock.institutional_holders_pct = overrides.get("institutional_holders_pct", 40.0)
    stock.roe = overrides.get("roe", 0.25)
    stock.trailing_pe = overrides.get("trailing_pe", 20.0)
    stock.earnings_growth_estimate = overrides.get("earnings_growth_estimate", 0.15)
    stock.earnings_surprise_pct = overrides.get("earnings_surprise_pct", 0)
    stock.eps_beat_streak = overrides.get("eps_beat_streak", 0)
    stock.eps_estimate_revision_pct = overrides.get("eps_estimate_revision_pct", None)

    # Earnings data
    stock.quarterly_earnings = overrides.get("quarterly_earnings",
        [1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90])
    stock.annual_earnings = overrides.get("annual_earnings", [5.50, 4.80, 4.20])
    stock.quarterly_revenue = overrides.get("quarterly_revenue",
        [10e9, 9.5e9, 9e9, 8.5e9, 7e9])

    # Price history (252 days, upward trending)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    base = overrides.get("price_base", 80.0)
    trend = overrides.get("price_trend", 0.1)
    prices = [base + i * trend for i in range(252)]
    volumes = [1_000_000] * 252
    stock.price_history = pd.DataFrame({
        'Close': prices, 'Volume': volumes
    }, index=dates)

    stock.weekly_price_history = pd.DataFrame()
    stock.is_valid = overrides.get("is_valid", True)
    return stock


def _make_scorer():
    """Create a scorer with mocked SP500 history."""
    mock_fetcher = MagicMock()
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
    # SP500 flat — makes RS calculations predictable
    sp500 = pd.DataFrame({
        'Close': [100.0] * 252, 'Volume': [1e9] * 252
    }, index=dates)
    mock_fetcher.get_sp500_history.return_value = sp500
    return CANSLIMScorer(mock_fetcher)


# ── _clean_earnings ───────────────────────────────────────────────────────────

class TestCleanEarnings:
    """Test the NaN/None cleaning utility."""

    def test_removes_none(self):
        assert _clean_earnings([1.0, None, 2.0]) == [1.0, 2.0]

    def test_removes_nan(self):
        assert _clean_earnings([1.0, float('nan'), 2.0]) == [1.0, 2.0]

    def test_keeps_zeros(self):
        assert _clean_earnings([1.0, 0.0, 2.0]) == [1.0, 0.0, 2.0]

    def test_keeps_negatives(self):
        assert _clean_earnings([-0.5, -1.0]) == [-0.5, -1.0]

    def test_empty_list(self):
        assert _clean_earnings([]) == []

    def test_all_nan(self):
        assert _clean_earnings([None, float('nan'), None]) == []


# ── C Score (Current Quarterly Earnings) ──────────────────────────────────────

class TestCScore:
    """Test Current Quarterly Earnings scoring (max 15 pts)."""

    def test_strong_ttm_growth_gets_high_score(self):
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[2.0, 1.9, 1.8, 1.7, 1.0, 0.9, 0.8, 0.7]
        )
        score = scorer.score_stock(stock)
        # TTM: 2.0+1.9+1.8+1.7 = 7.4 vs prior 1.0+0.9+0.8+0.7 = 3.4
        # Growth: ~118% — excellent for any sector
        assert score.c_score >= 10

    def test_negative_ttm_gets_zero(self):
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[-0.5, -0.6, -0.7, -0.8, -0.3, -0.2, -0.1, 0.1]
        )
        score = scorer.score_stock(stock)
        # Current TTM = -2.6, losses worsening
        assert score.c_score == 0

    def test_shrinking_losses_get_partial_credit(self):
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
        )
        score = scorer.score_stock(stock)
        # TTM: -1.0 vs prior -2.6 — losses are shrinking
        assert 0 < score.c_score < 7

    def test_acceleration_bonus(self):
        """EPS acceleration (current Q growth > prior Q growth) should add bonus."""
        scorer = _make_scorer()
        # Accelerating: Q0 grew 100% YoY, Q1 grew 50% YoY
        stock_accel = _make_stock_data(
            quarterly_earnings=[2.0, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0, 1.0]
        )
        # Decelerating: Q0 grew 20% YoY, Q1 grew 50% YoY
        stock_decel = _make_stock_data(
            quarterly_earnings=[1.2, 1.5, 1.3, 1.2, 1.0, 1.0, 1.0, 1.0]
        )
        score_accel = scorer.score_stock(stock_accel)
        score_decel = scorer.score_stock(stock_decel)
        assert score_accel.c_score >= score_decel.c_score

    def test_earnings_surprise_bonus(self):
        """Beating estimates by 10%+ should add bonus points."""
        scorer = _make_scorer()
        stock = _make_stock_data(earnings_surprise_pct=15)
        score = scorer.score_stock(stock)
        stock_no_surprise = _make_stock_data(earnings_surprise_pct=0)
        score_no = scorer.score_stock(stock_no_surprise)
        assert score.c_score >= score_no.c_score

    def test_beat_streak_bonus(self):
        """4+ consecutive beats should add bonus."""
        scorer = _make_scorer()
        stock = _make_stock_data(eps_beat_streak=5)
        score = scorer.score_stock(stock)
        stock_no_streak = _make_stock_data(eps_beat_streak=0)
        score_no = scorer.score_stock(stock_no_streak)
        assert score.c_score >= score_no.c_score

    def test_estimate_revision_bonus(self):
        """Strong upward revision (>= 10%) should boost C score."""
        scorer = _make_scorer()
        stock = _make_stock_data(eps_estimate_revision_pct=15)
        score = scorer.score_stock(stock)
        stock_down = _make_stock_data(eps_estimate_revision_pct=-15)
        score_down = scorer.score_stock(stock_down)
        assert score.c_score > score_down.c_score

    def test_insufficient_data(self):
        scorer = _make_scorer()
        stock = _make_stock_data(quarterly_earnings=[1.0])
        score = scorer.score_stock(stock)
        assert score.c_score == 0
        assert "Insufficient" in score.c_detail

    def test_turnaround_from_zero(self):
        """Stock with zero prior TTM but positive current should get credit."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[0.5, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]
        )
        score = scorer.score_stock(stock)
        assert score.c_score > 5  # "Turnaround" credit

    def test_sector_adjusted_thresholds(self):
        """Tech should need 30%+ for excellent, Utilities only 12%."""
        scorer = _make_scorer()
        # Moderate growth (20%)
        stock_tech = _make_stock_data(
            sector="Technology",
            quarterly_earnings=[1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.90, 0.85]
        )
        stock_util = _make_stock_data(
            sector="Utilities",
            quarterly_earnings=[1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.90, 0.85]
        )
        score_tech = scorer.score_stock(stock_tech)
        score_util = scorer.score_stock(stock_util)
        # Same growth should score higher for Utilities (lower bar)
        assert score_util.c_score >= score_tech.c_score


# ── A Score (Annual Earnings Growth) ──────────────────────────────────────────

class TestAScore:
    """Test Annual Earnings Growth scoring (max 15 pts)."""

    def test_strong_cagr_gets_high_score(self):
        scorer = _make_scorer()
        stock = _make_stock_data(annual_earnings=[10.0, 7.0, 5.0])
        score = scorer.score_stock(stock)
        # CAGR ~41% — excellent
        assert score.a_score >= 10

    def test_flat_earnings_gets_low_score(self):
        scorer = _make_scorer()
        stock = _make_stock_data(annual_earnings=[5.0, 5.0, 5.0])
        score = scorer.score_stock(stock)
        assert score.a_score < 5

    def test_negative_earnings_gets_zero(self):
        scorer = _make_scorer()
        stock = _make_stock_data(annual_earnings=[-1.0, -2.0, -3.0])
        score = scorer.score_stock(stock)
        assert score.a_score == 0

    def test_turnaround_gets_credit(self):
        """Positive current, negative older should get turnaround credit."""
        scorer = _make_scorer()
        stock = _make_stock_data(annual_earnings=[3.0, 1.0, -2.0])
        score = scorer.score_stock(stock)
        assert score.a_score > 5  # Turnaround credit

    def test_roe_bonus(self):
        """Strong ROE (>=25%) should add bonus to A score."""
        scorer = _make_scorer()
        stock_high_roe = _make_stock_data(roe=0.30)
        stock_low_roe = _make_stock_data(roe=0.05)
        score_high = scorer.score_stock(stock_high_roe)
        score_low = scorer.score_stock(stock_low_roe)
        assert score_high.a_score >= score_low.a_score

    def test_insufficient_annual_data(self):
        scorer = _make_scorer()
        stock = _make_stock_data(annual_earnings=[5.0, 4.0])
        score = scorer.score_stock(stock)
        assert score.a_score == 0


# ── L Score (Leader vs Laggard) ───────────────────────────────────────────────

class TestLScore:
    """Test Leader scoring (max 15 pts) with RS and industry group bonus."""

    def test_outperforming_stock_scores_high(self):
        """Stock with uptrend should beat flat SP500 and score well."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=60, price_trend=0.2)  # Strong uptrend
        score = scorer.score_stock(stock)
        assert score.l_score >= 8

    def test_underperforming_stock_scores_low(self):
        """Stock with strong downtrend vs flat SP500 should score poorly."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=150, price_trend=-0.3)  # Strong downtrend
        score = scorer.score_stock(stock)
        assert score.l_score <= 7

    def test_industry_group_rank_boosts_l_score(self):
        """Top group rank should add to L score."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=80, price_trend=0.08)
        score_no_rank = scorer.score_stock(stock, industry_group_rank=None)
        score_top = scorer.score_stock(stock, industry_group_rank=95)
        assert score_top.l_score >= score_no_rank.l_score

    def test_industry_group_rank_penalizes_bottom(self):
        """Bottom group rank should reduce L score."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=80, price_trend=0.08)
        score_no_rank = scorer.score_stock(stock, industry_group_rank=None)
        score_bottom = scorer.score_stock(stock, industry_group_rank=5)
        assert score_bottom.l_score <= score_no_rank.l_score

    def test_l_score_capped_at_max(self):
        """Even with group bonus, L shouldn't exceed 15."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=50, price_trend=0.3)  # Very strong
        score = scorer.score_stock(stock, industry_group_rank=100)
        assert score.l_score <= 15

    def test_l_score_floored_at_zero(self):
        """Even with group penalty, L shouldn't go below 0."""
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=140, price_trend=-0.2)  # Weak
        score = scorer.score_stock(stock, industry_group_rank=1)
        assert score.l_score >= 0

    def test_insufficient_price_data(self):
        scorer = _make_scorer()
        stock = _make_stock_data()
        stock.price_history = pd.DataFrame({'Close': [100.0] * 10},
            index=pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D'))
        score = scorer.score_stock(stock)
        assert score.l_score == 0


# ── N Score (New Highs) ───────────────────────────────────────────────────────

class TestNScore:
    """Test New Highs scoring (max 15 pts)."""

    def test_at_52w_high_scores_well(self):
        scorer = _make_scorer()
        stock = _make_stock_data(current_price=110.0, high_52w=110.0)
        score = scorer.score_stock(stock)
        assert score.n_score >= 12

    def test_far_from_high_scores_low(self):
        scorer = _make_scorer()
        stock = _make_stock_data(current_price=50.0, high_52w=110.0)
        score = scorer.score_stock(stock)
        assert score.n_score < 5


# ── S Score (Supply/Demand) ──────────────────────────────────────────────────

class TestSScore:
    """Test Supply/Demand scoring (max 15 pts)."""

    def test_above_avg_volume_scores_well(self):
        scorer = _make_scorer()
        stock = _make_stock_data(current_volume=2_000_000, avg_volume_50d=1_000_000)
        score = scorer.score_stock(stock)
        assert score.s_score > 5

    def test_low_float_gets_bonus(self):
        """Small share count = tighter supply = higher score."""
        scorer = _make_scorer()
        stock_small = _make_stock_data(shares_outstanding=10_000_000)
        stock_large = _make_stock_data(shares_outstanding=5_000_000_000)
        score_small = scorer.score_stock(stock_small)
        score_large = scorer.score_stock(stock_large)
        assert score_small.s_score >= score_large.s_score


# ── I Score (Institutional Ownership) ─────────────────────────────────────────

class TestIScore:
    """Test Institutional Ownership scoring (max 10 pts)."""

    def test_moderate_ownership_is_ideal(self):
        """O'Neil prefers 25-60% institutional ownership."""
        scorer = _make_scorer()
        stock = _make_stock_data(institutional_holders_pct=40.0)
        score = scorer.score_stock(stock)
        assert score.i_score >= 5

    def test_too_high_ownership_penalized(self):
        """90%+ institutional = over-owned, limited upside."""
        scorer = _make_scorer()
        stock_high = _make_stock_data(institutional_holders_pct=95.0)
        stock_ideal = _make_stock_data(institutional_holders_pct=45.0)
        score_high = scorer.score_stock(stock_high)
        score_ideal = scorer.score_stock(stock_ideal)
        assert score_ideal.i_score >= score_high.i_score


# ── M Score (Market Direction) ────────────────────────────────────────────────

class TestMScore:
    """Test Market Direction scoring (max 15 pts)."""

    def test_m_score_is_consistent(self):
        """M score should be the same for all stocks (it's market-wide)."""
        scorer = _make_scorer()
        stock1 = _make_stock_data(ticker="AAPL")
        stock2 = _make_stock_data(ticker="MSFT")
        score1 = scorer.score_stock(stock1)
        score2 = scorer.score_stock(stock2)
        assert score1.m_score == score2.m_score


# ── Total Score ──────────────────────────────────────────────────────────────

class TestTotalScore:
    """Test total score computation."""

    def test_total_is_sum_of_components(self):
        scorer = _make_scorer()
        stock = _make_stock_data()
        score = scorer.score_stock(stock)
        expected = (score.c_score + score.a_score + score.n_score +
                    score.s_score + score.l_score + score.i_score + score.m_score)
        assert score.total_score == pytest.approx(expected, abs=0.1)

    def test_max_possible_score_is_100(self):
        """Total max is C(15)+A(15)+N(15)+S(15)+L(15)+I(10)+M(15) = 100."""
        scorer = _make_scorer()
        stock = _make_stock_data()
        score = scorer.score_stock(stock)
        assert score.total_score <= 100

    def test_invalid_stock_gets_zero(self):
        scorer = _make_scorer()
        stock = _make_stock_data(is_valid=False)
        score = scorer.score_stock(stock)
        assert score.total_score == 0

    def test_nan_in_earnings_doesnt_crash(self):
        """NaN values in earnings should be cleaned, not cause crashes."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[1.5, float('nan'), 1.3, None, 1.1, 1.0, 0.9, 0.8]
        )
        score = scorer.score_stock(stock)
        assert isinstance(score.total_score, float)
        assert not math.isnan(score.total_score)


# ── RS Value Extraction ──────────────────────────────────────────────────────

class TestRSExtraction:
    """Test the extract_rs_values method for persistence."""

    def test_extracts_rs_values(self):
        scorer = _make_scorer()
        stock = _make_stock_data(price_base=80, price_trend=0.1)
        rs = scorer.extract_rs_values(stock)
        assert "rs_12m" in rs
        assert "rs_3m" in rs
        assert rs["rs_12m"] is not None
        assert rs["rs_3m"] is not None

    def test_insufficient_data_returns_none(self):
        scorer = _make_scorer()
        stock = _make_stock_data()
        stock.price_history = pd.DataFrame({'Close': [100.0] * 10},
            index=pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D'))
        rs = scorer.extract_rs_values(stock)
        assert rs["rs_12m"] is None
        assert rs["rs_3m"] is None
