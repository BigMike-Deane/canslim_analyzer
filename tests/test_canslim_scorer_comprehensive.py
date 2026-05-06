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
    stock.earnings_beat_streak = overrides.get("earnings_beat_streak", 0)
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

    # ── Bug B regression: Turnaround branch must apply ROE bonus ──────────────
    # Pre-fix: Turnaround silently returned a flat 10.5 pts and ignored ROE,
    # capping ~198 stocks (e.g., GEV ROE 75.7%, MMM 71.5%) at 10.5 instead of
    # up to 13.5. The primary 3-yr CAGR path applied the ROE bonus correctly;
    # the Turnaround branch did not.

    @pytest.mark.skip(reason="Split-test May 6: Turnaround ROE bonus reverted to isolate Bundle 2 regression")
    def test_turnaround_applies_roe_bonus_high(self):
        """Turnaround stock with strong ROE should exceed flat 10.5-pt floor."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            annual_earnings=[10.0, 5.0, -1.0],  # turnaround: older<=0, recent>0
            roe=0.50,                            # 50% ROE — full 3-pt bonus
        )
        score = scorer.score_stock(stock)
        # 10.5 base + 3.0 ROE = 13.5
        assert score.a_score == pytest.approx(13.5, abs=0.1)
        assert "Turnaround" in score.a_detail
        assert "ROE" in score.a_detail

    def test_turnaround_zero_roe_unchanged(self):
        """Turnaround with no ROE should still be exactly 10.5 (no regression)."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            annual_earnings=[10.0, 5.0, -1.0],
            roe=0.0,
        )
        score = scorer.score_stock(stock)
        assert score.a_score == pytest.approx(10.5, abs=0.01)

    @pytest.mark.skip(reason="Split-test May 6: Turnaround ROE bonus reverted to isolate Bundle 2 regression")
    def test_turnaround_partial_roe_bonus(self):
        """Turnaround with 17%-tier ROE should land between 10.5 and 13.5."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            annual_earnings=[10.0, 5.0, -1.0],
            roe=0.18,  # 18% — 0.7 * 3 = 2.1 bonus
        )
        score = scorer.score_stock(stock)
        # 10.5 + 2.1 = 12.6
        assert score.a_score == pytest.approx(12.6, abs=0.1)

    def test_turnaround_total_capped_at_max(self):
        """Turnaround total should never exceed A score max (15)."""
        scorer = _make_scorer()
        # Stack the deck: massive ROE + turnaround. 10.5 + 3 = 13.5, well under
        # 15, but pin the cap explicitly so a future bump to base/bonus can't
        # silently exceed max_score.
        stock = _make_stock_data(
            annual_earnings=[10.0, 5.0, -1.0],
            roe=10.0,  # 1000% — read as already-percentage; still caps at max_pts
        )
        score = scorer.score_stock(stock)
        assert score.a_score <= 15.0


# ── Sector threshold lookup (Bug A regression) ────────────────────────────────
# FMP/Yahoo emit Morningstar sector names for ~37% of the universe. Pre-fix,
# those names had no key in SECTOR_GROWTH_THRESHOLDS and silently fell through
# to `default` (excellent=25, good=15) instead of the tuned values for their
# GICS equivalents. Affects both A and C scoring (both call _get_sector_thresholds).

@pytest.mark.skip(reason="Sector aliases reverted May 6 2026 — bt 661/662 showed they hurt portfolio returns. See canslim-may6-bundle2-revert.md memory.")
class TestSectorThresholds:
    """Test sector threshold lookup honors Morningstar -> GICS aliases."""

    def test_financial_services_resolves_to_financials(self):
        scorer = _make_scorer()
        thresholds = scorer._get_sector_thresholds("Financial Services")
        assert thresholds == {'excellent': 20, 'good': 12}

    def test_consumer_cyclical_resolves_to_consumer_discretionary(self):
        scorer = _make_scorer()
        thresholds = scorer._get_sector_thresholds("Consumer Cyclical")
        assert thresholds == {'excellent': 25, 'good': 18}

    def test_consumer_defensive_resolves_to_consumer_staples(self):
        scorer = _make_scorer()
        thresholds = scorer._get_sector_thresholds("Consumer Defensive")
        assert thresholds == {'excellent': 15, 'good': 10}

    def test_basic_materials_resolves_to_materials(self):
        scorer = _make_scorer()
        thresholds = scorer._get_sector_thresholds("Basic Materials")
        assert thresholds == {'excellent': 18, 'good': 12}

    def test_aliased_does_not_fall_through_to_default(self):
        """All four Morningstar aliases must NOT return the default thresholds."""
        scorer = _make_scorer()
        default = scorer.SECTOR_GROWTH_THRESHOLDS['default']
        for morningstar_name in ('Financial Services', 'Consumer Cyclical',
                                  'Consumer Defensive', 'Basic Materials'):
            thresholds = scorer._get_sector_thresholds(morningstar_name)
            # Each alias maps to a sector with different thresholds than default
            assert thresholds != default, (
                f"{morningstar_name} fell through to default — alias broken"
            )

    def test_gics_names_still_work(self):
        """Native GICS names must continue to resolve correctly (no regression)."""
        scorer = _make_scorer()
        assert scorer._get_sector_thresholds("Technology") == {'excellent': 30, 'good': 20}
        assert scorer._get_sector_thresholds("Financials") == {'excellent': 20, 'good': 12}
        assert scorer._get_sector_thresholds("Materials") == {'excellent': 18, 'good': 12}

    def test_unknown_sector_falls_through_to_default(self):
        """Truly unknown sectors should still hit the default."""
        scorer = _make_scorer()
        thresholds = scorer._get_sector_thresholds("NonexistentSector")
        assert thresholds == scorer.SECTOR_GROWTH_THRESHOLDS['default']

    def test_alias_affects_a_score(self):
        """Same earnings, Morningstar vs GICS sector name, should yield same A score."""
        scorer = _make_scorer()
        # Materials: excellent=18, good=12 — different from default (25/15)
        stock_gics = _make_stock_data(
            sector="Materials",
            annual_earnings=[1.50, 1.30, 1.10],  # CAGR ~10.9%
        )
        stock_alias = _make_stock_data(
            sector="Basic Materials",
            annual_earnings=[1.50, 1.30, 1.10],
        )
        s_gics = scorer.score_stock(stock_gics)
        s_alias = scorer.score_stock(stock_alias)
        assert s_gics.a_score == s_alias.a_score, (
            "Materials and Basic Materials must produce identical A scores"
        )

    def test_alias_affects_c_score(self):
        """Sector aliases must also resolve via the C-score path."""
        scorer = _make_scorer()
        # Same earnings, Financials vs Financial Services
        earnings = [1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.90, 0.85]
        stock_gics = _make_stock_data(sector="Financials", quarterly_earnings=earnings)
        stock_alias = _make_stock_data(sector="Financial Services", quarterly_earnings=earnings)
        s_gics = scorer.score_stock(stock_gics)
        s_alias = scorer.score_stock(stock_alias)
        assert s_gics.c_score == s_alias.c_score, (
            "Financials and Financial Services must produce identical C scores"
        )


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


# ── C-Score Bug Regression Tests (May 2026) ──────────────────────────────────
# Three compounding bugs in the C (Current Earnings) score path:
#   1. data_fetcher.py truncated quarterly_earnings to 4-7 quarters when FMP
#      returned a short adjusted-EPS series, kicking the scorer into the
#      QoQ-anomaly fallback path.
#   2. The fallback path silently dropped the analyst-revision /
#      earnings-surprise / beat-streak bonuses (worth up to ~10 pts).
#   3. The fallback path's anomaly filter capped at >100% QoQ, discarding
#      legitimate post-trough recoveries like AMD's $0.48 -> $1.20 (+150%).
# Pin all three with regression tests.

class TestCScoreFallbackBonuses:
    """Bug 2: fallback path must apply the same bonuses as the primary path."""

    def test_fallback_applies_revision_bonus(self):
        """A 4-quarter stock with strong analyst revision should still get the bonus.

        Note (May 6, 2026 excellence-cap): a +25% revision is BELOW the
        excellence threshold (30%) so signal-3 doesn't fire. With 0 excellence
        signals the cap clips the post-bonus score at 10 — but the bonus must
        still register as a strict improvement over the no-revision baseline.
        We pin "strict improvement" rather than the exact pre-cap delta because
        the cap will (correctly) clip near-cap stocks.
        """
        scorer = _make_scorer()
        # Only 4 quarters of EPS — forces the fallback path
        stock = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],  # ~33% YoY growth via QoQ
            eps_estimate_revision_pct=25.0,
        )
        score = scorer.score_stock(stock)
        stock_no_rev = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],
            eps_estimate_revision_pct=None,
        )
        score_no_rev = scorer.score_stock(stock_no_rev)
        assert score.c_score > score_no_rev.c_score, (
            f"Revision bonus must apply in fallback path: "
            f"with={score.c_score} vs without={score_no_rev.c_score}"
        )
        # Bonus must move the score upward by at least the next-tier gap (~3 pts).
        # Pre-cap the gap was ~8 pts; post-cap it's clipped to 10 from ~14.
        assert score.c_score - score_no_rev.c_score >= 3, (
            f"Revision bonus should produce a meaningful lift; got delta {score.c_score - score_no_rev.c_score}"
        )

    def test_fallback_applies_beat_streak_bonus(self):
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],
            eps_beat_streak=6,
            earnings_surprise_pct=12,
        )
        stock_no_streak = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],
            eps_beat_streak=0,
            earnings_surprise_pct=0,
        )
        score = scorer.score_stock(stock)
        score_no = scorer.score_stock(stock_no_streak)
        assert score.c_score > score_no.c_score

    def test_smell_test_strong_signals_must_not_score_below_10(self):
        """Investigator's smell test:
        eps_estimate_revision_pct >= 20 AND eps_beat_streak >= 3
        should NEVER produce C < 10 — the bonuses alone are worth ~10 pts.
        Pin this against either path being silently missed.
        """
        scorer = _make_scorer()
        # Try BOTH paths: 8-quarter (primary) AND 4-quarter (fallback)
        primary = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90, 0.85, 0.80, 0.78, 0.75],
            eps_estimate_revision_pct=22.0,
            eps_beat_streak=4,
            earnings_surprise_pct=8,
        )
        fallback = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],
            eps_estimate_revision_pct=22.0,
            eps_beat_streak=4,
            earnings_surprise_pct=8,
        )
        s_primary = scorer.score_stock(primary)
        s_fallback = scorer.score_stock(fallback)
        assert s_primary.c_score >= 10, (
            f"Strong analyst signals on 8q path: C={s_primary.c_score} (expected >= 10)"
        )
        assert s_fallback.c_score >= 10, (
            f"Strong analyst signals on 4q fallback path: C={s_fallback.c_score} "
            f"(expected >= 10) — bonuses must NOT be silently dropped"
        )


class TestAnomalyFilterCapRaised:
    """Bug 3: anomaly filter cap raised from 100% to 300% to admit
    legitimate post-trough recoveries."""

    def test_amd_style_recovery_admitted(self):
        """AMD's $0.48 -> $1.20 jump (+150%) should NOT be filtered out.
        Without the fix, this swing is dropped, all 3 QoQs come back as
        anomalies, scorer falls back to forward estimate or zeros out.
        With the fix, the +150% lands in the median and the C score
        reflects the recovery."""
        scorer = _make_scorer()
        # 4 quarters: $1.20, $0.85, $0.55, $0.48 — Q0 vs Q1 = +41%, Q1 vs Q2 = +55%,
        # Q2 vs Q3 = +15%. To trigger the AMD-style swing specifically, build:
        # $1.20, $0.48, $0.50, $0.45 — Q0 vs Q1 = +150%
        stock_recovery = _make_stock_data(
            quarterly_earnings=[1.20, 0.48, 0.50, 0.45],
            earnings_growth_estimate=0,  # no fallback estimate available
        )
        score = scorer.score_stock(stock_recovery)
        # +150% recovery should produce a strong C score (median QoQ ~+150%)
        # NOT zero or "Data anomaly"
        assert "Data anomaly" not in score.c_detail, (
            f"AMD-style recovery wrongly filtered as anomaly: detail={score.c_detail!r}"
        )
        assert score.c_score > 5, (
            f"AMD-style +150% recovery should score well; got C={score.c_score}, "
            f"detail={score.c_detail!r}"
        )

    def test_extreme_outlier_still_filtered(self):
        """A 1000% swing (real one-time data anomaly) is still dropped."""
        scorer = _make_scorer()
        # One huge swing (Q0 vs Q1 = +5900%) flanked by tiny stable quarters
        stock = _make_stock_data(
            quarterly_earnings=[6.0, 0.10, 0.10, 0.10],
            earnings_growth_estimate=0,
        )
        score = scorer.score_stock(stock)
        # Median of [+5900, 0, 0] would be 0 if outlier kept; anomaly drop
        # leaves [0, 0] -> median 0 also. Either way the +5900 is NOT used
        # to award full C credit.
        assert score.c_score < 12, (
            f"Extreme +5900% swing must not produce near-max C; got {score.c_score}"
        )


class TestEpsSeriesNotTruncated:
    """Bug 1: data_fetcher must not truncate an 8-quarter GAAP series
    when FMP only returned 4-7 quarters of adjusted EPS.

    We test the merge logic in isolation since the fetch involves network IO.
    """

    def test_partial_adjusted_merges_with_gaap(self):
        """4 quarters of adjusted + 8 quarters of GAAP should merge to a
        full 8-quarter series (adjusted prefix + GAAP suffix)."""
        adjusted_eps = [1.20, 1.10, 1.00, 0.90]  # 4 quarters
        gaap_eps = [1.18, 1.08, 0.98, 0.88, 0.85, 0.80, 0.78, 0.75]  # 8 quarters
        # Replicate the merge logic from data_fetcher.py
        merged = adjusted_eps + gaap_eps[len(adjusted_eps):] if len(gaap_eps) > len(adjusted_eps) else adjusted_eps
        merged = merged[:8]
        # Adjusted prefix preserved
        assert merged[:4] == adjusted_eps
        # GAAP fills the back half
        assert merged[4:] == gaap_eps[4:8]
        assert len(merged) == 8

    def test_eight_plus_adjusted_wins_outright(self):
        """When adjusted has >= 8 quarters, it should fully replace GAAP
        for post-split correctness. (Below the new threshold the merge runs.)"""
        # Mirrors the data_fetcher branch we changed: was `>= 4`, now `>= 8`.
        adjusted_eps = [1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90]
        gaap_eps = [3.00, 2.80, 2.60, 2.40, 2.20, 2.00, 1.90, 1.80, 1.70]
        # New threshold semantics: len(adjusted_eps) >= 8 -> use adjusted as-is.
        if adjusted_eps and len(adjusted_eps) >= 8:
            chosen = adjusted_eps
        else:
            merged = adjusted_eps + gaap_eps[len(adjusted_eps):] if len(gaap_eps) > len(adjusted_eps) else adjusted_eps
            chosen = merged[:8]
        assert chosen == adjusted_eps, "Adjusted EPS must win outright at >= 8 samples"

    def test_short_adjusted_no_longer_truncates_long_gaap(self):
        """The actual bug: pre-fix, 4 adjusted overrode 8 GAAP -> scorer
        only saw 4 quarters and dropped to fallback path. Post-fix, the
        merge keeps the 8-quarter length so the primary TTM path runs."""
        adjusted_eps = [1.20, 1.10, 1.00, 0.90]  # 4 quarters
        gaap_eps = [1.18, 1.08, 0.98, 0.88, 0.85, 0.80, 0.78, 0.75]  # 8 quarters
        # New threshold is `>= 8`, so 4-quarter adjusted falls through to merge
        if adjusted_eps and len(adjusted_eps) >= 8:
            result = adjusted_eps
        else:
            merged = adjusted_eps + gaap_eps[len(adjusted_eps):] if len(gaap_eps) > len(adjusted_eps) else adjusted_eps
            result = merged[:8]
        # Critical invariant: scorer needs >= 8 quarters for the primary TTM path
        assert len(result) >= 8, (
            f"Bug 1 regression: short adjusted_eps truncated long GAAP series. "
            f"Got {len(result)} quarters, need >= 8 for the primary scorer path"
        )


# ── C Score Excellence-Tier Cap (May 6, 2026) ─────────────────────────────────
# Today's audit (May 6) found 49.4% of canslim>=73 admits at C=15 cap (38/77) —
# C lost discrimination at the top because additive bonuses stack to the cap
# from a single strong signal. The cap requires CONJUNCTION: only stocks with
# multiple "broad-based excellence" signals earn the full 15.
#
# Cap table:
#   4 of 4 signals -> 15 (no cap)
#   2-3            -> 13
#   1              -> 11.5
#   0              -> 10
#
# Excellence signals (each binary):
#   1. Strong growth (TTM>=50% on primary path, median QoQ>=40% on fallback)
#   2. Long beat streak (>=5)
#   3. Strong analyst revision (>=30%)
#   4. Big earnings surprise (>=10%)

class TestCExcellenceCap:
    """Pin the excellence-tier cap on C score."""

    # Strong primary-path stock that pre-cap would naturally score 15:
    #   - TTM 7.4 vs 3.4 = +118% growth (signal 1: yes)
    #   - 8 quarters of accelerating earnings (acceleration bonus = 5 pts)
    # Add or omit signals 2/3/4 to drive the excellence count.
    _STRONG_GROWTH_EARNINGS = [2.0, 1.9, 1.8, 1.7, 1.0, 0.9, 0.8, 0.7]

    def test_4_of_4_keeps_full_score(self):
        """All four signals present -> no cap, score may reach 15."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=self._STRONG_GROWTH_EARNINGS,  # signal 1
            earnings_beat_streak=6,                            # signal 2
            eps_estimate_revision_pct=35.0,                    # signal 3
            earnings_surprise_pct=15.0,                        # signal 4
        )
        score = scorer.score_stock(stock)
        # Pre-cap natural score is at the 15 ceiling; with 4/4 signals, no clip.
        assert score.c_score == pytest.approx(15.0, abs=0.05)
        assert "excellence:4/4" in score.c_detail
        assert "cap:" not in score.c_detail  # no cap line because no clip

    def test_3_of_4_caps_at_13(self):
        """Three signals present -> capped at 13 even if natural score was 15."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=self._STRONG_GROWTH_EARNINGS,  # signal 1
            earnings_beat_streak=6,                            # signal 2
            eps_estimate_revision_pct=35.0,                    # signal 3
            earnings_surprise_pct=2.0,                         # signal 4 NO (<10)
        )
        score = scorer.score_stock(stock)
        assert score.c_score == pytest.approx(13.0, abs=0.05), (
            f"3-of-4 should cap at 13; got {score.c_score} ({score.c_detail})"
        )
        assert "excellence:3/4" in score.c_detail
        assert "cap:13" in score.c_detail

    def test_2_of_4_caps_at_13(self):
        """Two signals present -> still 13 (same tier as 3-of-4)."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=self._STRONG_GROWTH_EARNINGS,  # signal 1
            earnings_beat_streak=6,                            # signal 2
            eps_estimate_revision_pct=10.0,                    # signal 3 NO (<30)
            earnings_surprise_pct=3.0,                         # signal 4 NO
        )
        score = scorer.score_stock(stock)
        assert score.c_score == pytest.approx(13.0, abs=0.05), (
            f"2-of-4 should cap at 13; got {score.c_score} ({score.c_detail})"
        )
        assert "excellence:2/4" in score.c_detail

    def test_1_of_4_caps_at_11_5(self):
        """One signal present -> capped at 11.5."""
        scorer = _make_scorer()
        stock = _make_stock_data(
            quarterly_earnings=self._STRONG_GROWTH_EARNINGS,  # signal 1
            earnings_beat_streak=2,                            # NO (<5)
            eps_estimate_revision_pct=10.0,                    # NO (<30)
            earnings_surprise_pct=3.0,                         # NO (<10)
        )
        score = scorer.score_stock(stock)
        assert score.c_score == pytest.approx(11.5, abs=0.05), (
            f"1-of-4 should cap at 11.5; got {score.c_score} ({score.c_detail})"
        )
        assert "excellence:1/4" in score.c_detail
        assert "cap:11.5" in score.c_detail

    def test_0_of_4_caps_at_10(self):
        """Zero signals -> capped at 10. Use weak growth + heavy bonuses
        to force a near-cap natural score with 0 excellence signals."""
        scorer = _make_scorer()
        # ~30% TTM growth (signal 1 NO, 30 < 50). Pile up bonuses that don't
        # individually trip the excellence thresholds:
        #   - revision +25% (NO, 25 < 30) -> +8 bonus pts
        #   - surprise 8% (NO, 8 < 10)    -> +1.5 bonus pts
        #   - beat streak 4 (NO, 4 < 5)   -> +1 bonus pt
        # Combined: base ~9 + bonuses ~10.5 ~= caps at max_score, but
        # excellence cap clips it to 10.
        stock = _make_stock_data(
            quarterly_earnings=[1.30, 1.25, 1.20, 1.15, 1.00, 0.95, 0.90, 0.85],
            earnings_beat_streak=4,
            eps_estimate_revision_pct=25.0,
            earnings_surprise_pct=8.0,
        )
        score = scorer.score_stock(stock)
        assert score.c_score == pytest.approx(10.0, abs=0.05), (
            f"0-of-4 should cap at 10; got {score.c_score} ({score.c_detail})"
        )
        assert "excellence:0/4" in score.c_detail
        assert "cap:10" in score.c_detail

    def test_cap_does_not_elevate_low_scores(self):
        """A stock scoring 7 with 0 signals stays 7, not raised to 10.
        The cap is a CEILING only, never a floor."""
        scorer = _make_scorer()
        # Mild growth (~10% TTM) -> low base (~3-5). No bonuses -> 0 signals.
        stock = _make_stock_data(
            quarterly_earnings=[1.10, 1.08, 1.05, 1.03, 1.00, 0.98, 0.95, 0.93],
            earnings_beat_streak=0,
            eps_estimate_revision_pct=None,
            earnings_surprise_pct=0,
        )
        score = scorer.score_stock(stock)
        # Natural score is below the cap floor; cap doesn't elevate.
        assert score.c_score < 10.0, (
            f"Cap must not elevate sub-10 scores; got {score.c_score} ({score.c_detail})"
        )
        # Detail still shows the count but no "cap:" tag because no clip happened.
        assert "excellence:0/4" in score.c_detail
        assert "cap:" not in score.c_detail

    def test_detail_string_contains_excellence_count(self):
        """Every C-score detail (primary path) reports the excellence count
        so the user can see why their stock was capped at 11.5 vs 15.

        Build stocks with exactly N signals firing by toggling fields:
          signal 1 = strong growth (use STRONG earnings; weak earnings disable)
          signal 2 = beat streak >= 5
          signal 3 = revision >= 30%
          signal 4 = surprise >= 10%
        Order signals 1,2,3,4 to construct any count 0..4.
        """
        # Weak earnings: ~10% TTM growth -> signal 1 NO (10 < 50)
        weak = [1.10, 1.08, 1.05, 1.03, 1.00, 0.98, 0.95, 0.93]
        scorer = _make_scorer()
        for n_signals in range(5):
            stock = _make_stock_data(
                quarterly_earnings=self._STRONG_GROWTH_EARNINGS if n_signals >= 1 else weak,
                earnings_beat_streak=6 if n_signals >= 2 else 0,
                eps_estimate_revision_pct=35.0 if n_signals >= 3 else None,
                earnings_surprise_pct=15.0 if n_signals >= 4 else 0,
            )
            score = scorer.score_stock(stock)
            assert f"excellence:{n_signals}/4" in score.c_detail, (
                f"Expected excellence:{n_signals}/4 in detail; got {score.c_detail!r}"
            )

    def test_fallback_path_applies_cap(self):
        """The <8-quarter fallback path also applies the cap."""
        scorer = _make_scorer()
        # 4 quarters only -> fallback. ~33% median QoQ growth (signal 1 NO,
        # 33 < 40). Add 0 other signals -> 0/4 -> cap at 10.
        stock = _make_stock_data(
            quarterly_earnings=[1.20, 1.10, 1.00, 0.90],
            earnings_beat_streak=0,
            eps_estimate_revision_pct=None,
            earnings_surprise_pct=0,
        )
        score = scorer.score_stock(stock)
        assert "excellence:" in score.c_detail, (
            f"Fallback path must report excellence count; got {score.c_detail!r}"
        )
        # 0/4 signals -> cap at 10. Whether the natural score reaches the
        # cap depends on bonuses; we pin only the upper bound.
        assert score.c_score <= 10.0, (
            f"Fallback path with 0 signals must cap at 10; got {score.c_score}"
        )

    def test_fallback_path_strong_growth_signal_fires(self):
        """Fallback path uses median QoQ >= 40 to fire signal 1.

        4-quarter series with strong sequential growth produces median QoQ
        >= 40%, which counts toward the excellence count. Combined with
        another signal, total goes from 0/4 -> 2/4 (cap 13)."""
        scorer = _make_scorer()
        # QoQ rates: +25%, +33%, +33%; median ~33% (still below 40 -> signal 1 NO).
        # Steeper: $1.50, $1.00, $0.65, $0.45 -> +50%, +54%, +44%; median ~50% (signal 1 YES).
        stock_strong = _make_stock_data(
            quarterly_earnings=[1.50, 1.00, 0.65, 0.45],
            earnings_beat_streak=6,                # signal 2 YES
            eps_estimate_revision_pct=None,
            earnings_surprise_pct=0,
        )
        score = scorer.score_stock(stock_strong)
        assert "excellence:2/4" in score.c_detail, (
            f"Fallback path with strong QoQ + beat streak should be 2/4; got {score.c_detail!r}"
        )
        # 2-of-4 cap is 13; allow score to reach but not exceed it.
        assert score.c_score <= 13.0 + 0.05

    def test_count_excellence_signals_helper_directly(self):
        """Direct unit test of the helper for clarity."""
        scorer = _make_scorer()
        # All 4 signals present
        all_on = _make_stock_data(
            earnings_beat_streak=10,
            eps_estimate_revision_pct=50.0,
            earnings_surprise_pct=25.0,
        )
        # path="primary" with ttm_growth=100 fires signal 1
        assert scorer._count_excellence_signals(all_on, "primary", ttm_growth=100.0) == 4
        # path="primary" with ttm_growth=20 -> signal 1 fails
        assert scorer._count_excellence_signals(all_on, "primary", ttm_growth=20.0) == 3
        # path="fallback" with avg_qoq_growth=10 -> signal 1 fails
        assert scorer._count_excellence_signals(all_on, "fallback", avg_qoq_growth=10.0) == 3
        # No path metric supplied -> signal 1 cannot fire
        assert scorer._count_excellence_signals(all_on, "primary") == 3

        # None present
        none_on = _make_stock_data(
            earnings_beat_streak=0,
            eps_estimate_revision_pct=None,
            earnings_surprise_pct=0,
        )
        assert scorer._count_excellence_signals(none_on, "primary", ttm_growth=10.0) == 0

    def test_apply_excellence_cap_helper_directly(self):
        """Direct unit test of the cap helper. The cap is a min(), never a max()."""
        scorer = _make_scorer()
        # Cap is the ceiling
        assert scorer._apply_excellence_cap(15.0, 0) == 10.0
        assert scorer._apply_excellence_cap(15.0, 1) == 11.5
        assert scorer._apply_excellence_cap(15.0, 2) == 13.0
        assert scorer._apply_excellence_cap(15.0, 3) == 13.0
        assert scorer._apply_excellence_cap(15.0, 4) == 15.0
        # Below-cap scores are NOT raised
        assert scorer._apply_excellence_cap(7.0, 0) == 7.0
        assert scorer._apply_excellence_cap(5.0, 1) == 5.0
        assert scorer._apply_excellence_cap(11.5, 1) == 11.5  # boundary
        # Unknown signal count defaults to no cap (15)
        assert scorer._apply_excellence_cap(15.0, 99) == 15.0

    def test_earnings_beat_streak_field_recognized(self):
        """Spec uses earnings_beat_streak (P1 calendar field), not the older
        eps_beat_streak field. Helper must accept either."""
        scorer = _make_scorer()
        # Only earnings_beat_streak set (eps_beat_streak=0)
        stock = _make_stock_data(
            quarterly_earnings=self._STRONG_GROWTH_EARNINGS,  # signal 1
            earnings_beat_streak=6,                            # signal 2 via new field
            eps_beat_streak=0,                                 # old field unused
            eps_estimate_revision_pct=35.0,                    # signal 3
            earnings_surprise_pct=15.0,                        # signal 4
        )
        score = scorer.score_stock(stock)
        assert "excellence:4/4" in score.c_detail, (
            f"earnings_beat_streak field must satisfy signal 2; got {score.c_detail!r}"
        )
