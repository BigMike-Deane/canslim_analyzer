"""
Tests for bear_base.py — Bear Market Base-Building Watchlist

Tests the readiness score computation and candidate selection logic.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── compute_readiness_score ───────────────────────────────────────────────────

class TestComputeReadinessScore:
    """Test the 0-100 readiness score computation."""

    def _make_stock(self, **kwargs):
        """Create a mock stock with given attributes."""
        stock = MagicMock()
        defaults = {
            'c_score': 12, 'a_score': 10, 'l_score': 11,
            'rs_3m': 1.2, 'rs_12m': 1.1,
            'base_type': 'cup', 'weeks_in_base': 8,
            'atr_pct': 2.5,
            'volume_dry_up': True,
            'institutional_accumulation': False,
            'insider_sentiment': 'neutral',
            'industry_group_rank': 75,
            'pivot_price': 50.0, 'current_price': 47.0,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(stock, k, v)
        return stock

    def test_high_quality_stock_scores_high(self):
        from backend.bear_base import compute_readiness_score
        stock = self._make_stock(
            c_score=14, a_score=13, rs_3m=1.4, rs_12m=1.2,
            base_type='cup_with_handle', weeks_in_base=10,
            atr_pct=1.5, volume_dry_up=True,
            institutional_accumulation=True,
            insider_sentiment='bullish',
            industry_group_rank=90,
            pivot_price=50.0, current_price=48.0,
        )
        score, factors = compute_readiness_score(stock)
        assert score >= 70
        assert factors['fundamentals'] > 0
        assert factors['relative_strength'] > 0
        assert factors['base_quality'] > 0

    def test_weak_stock_scores_low(self):
        from backend.bear_base import compute_readiness_score
        stock = self._make_stock(
            c_score=3, a_score=2, rs_3m=0.6, rs_12m=0.5,
            base_type='none', weeks_in_base=0,
            atr_pct=8.0, volume_dry_up=False,
            institutional_accumulation=False,
            insider_sentiment='bearish',
            industry_group_rank=10,
            pivot_price=0, current_price=20.0,
        )
        score, factors = compute_readiness_score(stock)
        assert score < 30

    def test_fundamentals_component_max_25(self):
        from backend.bear_base import compute_readiness_score
        stock = self._make_stock(c_score=15, a_score=15)
        score, factors = compute_readiness_score(stock)
        assert factors['fundamentals'] == 25.0

    def test_rs_improving_gets_bonus(self):
        """3m RS > 12m RS should get bonus (momentum turning)."""
        from backend.bear_base import compute_readiness_score
        stock_improving = self._make_stock(rs_3m=1.3, rs_12m=1.0)
        stock_flat = self._make_stock(rs_3m=1.0, rs_12m=1.3)
        s1, f1 = compute_readiness_score(stock_improving)
        s2, f2 = compute_readiness_score(stock_flat)
        assert f1['relative_strength'] > f2['relative_strength']

    def test_tight_base_scores_higher(self):
        """Low ATR% = tight base = more potential energy."""
        from backend.bear_base import compute_readiness_score
        tight = self._make_stock(atr_pct=1.5)
        loose = self._make_stock(atr_pct=6.0)
        _, f_tight = compute_readiness_score(tight)
        _, f_loose = compute_readiness_score(loose)
        assert f_tight['base_quality'] > f_loose['base_quality']

    def test_cup_with_handle_scores_best(self):
        """Cup-with-handle is the highest quality base pattern."""
        from backend.bear_base import compute_readiness_score
        cwh = self._make_stock(base_type='cup_with_handle')
        flat = self._make_stock(base_type='flat_base')
        _, f1 = compute_readiness_score(cwh)
        _, f2 = compute_readiness_score(flat)
        assert f1['base_quality'] >= f2['base_quality']

    def test_volume_dry_up_adds_accumulation_points(self):
        from backend.bear_base import compute_readiness_score
        dry = self._make_stock(volume_dry_up=True)
        wet = self._make_stock(volume_dry_up=False)
        _, f1 = compute_readiness_score(dry)
        _, f2 = compute_readiness_score(wet)
        assert f1['accumulation'] > f2['accumulation']

    def test_near_pivot_gets_bonus(self):
        """Stock within 5% of pivot should get max pivot proximity bonus."""
        from backend.bear_base import compute_readiness_score
        near = self._make_stock(pivot_price=50.0, current_price=48.0)  # 4% from pivot
        far = self._make_stock(pivot_price=50.0, current_price=35.0)   # 30% from pivot
        _, f_near = compute_readiness_score(near)
        _, f_far = compute_readiness_score(far)
        assert f_near['pivot_proximity'] > f_far['pivot_proximity']

    def test_top_industry_group_gets_bonus(self):
        from backend.bear_base import compute_readiness_score
        top = self._make_stock(industry_group_rank=90)
        bottom = self._make_stock(industry_group_rank=15)
        _, f_top = compute_readiness_score(top)
        _, f_bottom = compute_readiness_score(bottom)
        assert f_top['industry_group'] > f_bottom['industry_group']

    def test_score_capped_at_100(self):
        """Even with perfect scores across all factors, cap at 100."""
        from backend.bear_base import compute_readiness_score
        perfect = self._make_stock(
            c_score=15, a_score=15,
            rs_3m=2.0, rs_12m=1.5,
            base_type='cup_with_handle', weeks_in_base=10,
            atr_pct=1.0,
            volume_dry_up=True, institutional_accumulation=True,
            insider_sentiment='bullish',
            industry_group_rank=100,
            pivot_price=50.0, current_price=49.0,
        )
        score, _ = compute_readiness_score(perfect)
        assert score <= 100

    def test_none_values_dont_crash(self):
        """Handle None gracefully for optional fields."""
        from backend.bear_base import compute_readiness_score
        stock = self._make_stock(
            rs_3m=None, rs_12m=None,
            atr_pct=None, industry_group_rank=None,
            pivot_price=None,
        )
        score, factors = compute_readiness_score(stock)
        assert isinstance(score, float)
        assert score >= 0


# ── Notification functions ────────────────────────────────────────────────────

class TestBearBaseNotifications:
    """Test ntfy notification functions for bear base features."""

    @patch('backend.email_utils.WEBHOOK_URL', '')
    def test_bear_base_update_push_no_url(self):
        from backend.email_utils import send_bear_base_update_push
        # Should return False when no URL configured
        result = send_bear_base_update_push(5, [
            {"ticker": "AAPL", "readiness_score": 80},
            {"ticker": "MSFT", "readiness_score": 75},
        ])
        assert result is False

    @patch('backend.email_utils.WEBHOOK_URL', '')
    def test_bear_market_report_push_no_url(self):
        from backend.email_utils import send_bear_market_report_push
        result = send_bear_market_report_push({
            "bases_forming": 10,
            "improving_groups": [{"industry": "Software"}],
            "top_ready": [{"ticker": "AAPL", "readiness_score": 80}],
        })
        assert result is False

    @patch('backend.email_utils.WEBHOOK_URL', '')
    def test_market_turn_ready_push_no_url(self):
        from backend.email_utils import send_market_turn_ready_push
        result = send_market_turn_ready_push(
            [{"ticker": "AAPL", "readiness_score": 80, "base_type": "cup", "weeks_in_base": 8}],
            550.0, 540.0
        )
        assert result is False

    @patch('backend.email_utils.WEBHOOK_URL', 'https://ntfy.sh/test')
    @patch('backend.email_utils.requests.post')
    def test_bear_base_update_push_sends(self, mock_post):
        """Verify notification is actually sent when URL is configured."""
        from backend.email_utils import send_bear_base_update_push
        mock_post.return_value.status_code = 200
        result = send_bear_base_update_push(5, [
            {"ticker": "AAPL", "readiness_score": 80},
        ])
        assert result is True
        mock_post.assert_called_once()
