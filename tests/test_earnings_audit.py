"""
Tests for the Earnings Audit feature.

Tests cover:
- Fundamental confidence score calculation
- Component sub-score calculations
- Bonus/penalty mapping
- Graceful degradation on missing data
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class TestComputeFundamentalConfidence:
    """Test the core confidence scoring algorithm."""

    def test_perfect_score(self):
        """Stock with excellent fundamentals across all dimensions."""
        from backend.earnings_audit import compute_fundamental_confidence

        audit_data = {
            "analyst_upside_pct": 35,     # >30% upside
            "analyst_num": 15,             # Many analysts
            "beat_streak": 6,              # Long streak
            "avg_beat_magnitude": 25,      # Big beats
            "last_beat_pct": 30,
            "roe": 0.30,                   # High ROE
            "debt_to_equity": 0.2,         # Low debt
            "free_cash_flow_per_share": 8, # Strong FCF
            "current_ratio": 2.5,
        }
        stock_data = {
            "insider_net_value": 600000,   # >500K insider buys
            "insider_buy_count": 4,        # Cluster
            "eps_estimate_revision_pct": 12,  # Strong revisions
            "estimate_revision_trend": "strong_up",
        }

        confidence, breakdown = compute_fundamental_confidence(audit_data, stock_data)

        assert confidence >= 90, f"Expected >= 90 for perfect fundamentals, got {confidence}"
        assert breakdown["analyst_upside"] == 100
        assert breakdown["beat_quality"] == 100
        assert breakdown["financial_health"] >= 90
        assert breakdown["insider_conviction"] == 100
        assert breakdown["estimate_revisions"] == 100

    def test_weak_score(self):
        """Stock with poor fundamentals."""
        from backend.earnings_audit import compute_fundamental_confidence

        audit_data = {
            "analyst_upside_pct": -5,      # Downside
            "analyst_num": 2,              # Few analysts
            "beat_streak": 0,              # No beats
            "avg_beat_magnitude": 0,
            "last_beat_pct": -10,
            "roe": -0.05,                  # Negative ROE
            "debt_to_equity": 3.0,         # High debt
            "free_cash_flow_per_share": -2, # Negative FCF
            "current_ratio": 0.5,
        }
        stock_data = {
            "insider_net_value": -100000,  # Insider selling
            "insider_buy_count": 0,
            "eps_estimate_revision_pct": -15,  # Downward revisions
            "estimate_revision_trend": "down",
        }

        confidence, breakdown = compute_fundamental_confidence(audit_data, stock_data)

        assert confidence < 20, f"Expected < 20 for weak fundamentals, got {confidence}"
        assert breakdown["analyst_upside"] == 0
        assert breakdown["beat_quality"] == 0

    def test_medium_confidence(self):
        """Stock with mixed fundamentals — moderate confidence."""
        from backend.earnings_audit import compute_fundamental_confidence

        audit_data = {
            "analyst_upside_pct": 12,      # Moderate upside
            "analyst_num": 8,
            "beat_streak": 3,              # Decent streak
            "avg_beat_magnitude": 7,
            "last_beat_pct": 5,
            "roe": 0.18,                   # Above 17% threshold
            "debt_to_equity": 0.8,         # Moderate debt
            "free_cash_flow_per_share": 3,
            "current_ratio": 1.5,
        }
        stock_data = {
            "insider_net_value": 50000,
            "insider_buy_count": 1,
            "eps_estimate_revision_pct": 3,
            "estimate_revision_trend": "neutral",
        }

        confidence, breakdown = compute_fundamental_confidence(audit_data, stock_data)

        assert 40 <= confidence <= 70, f"Expected 40-70 for mixed, got {confidence}"

    def test_missing_stock_data(self):
        """Graceful handling when stock_data is None."""
        from backend.earnings_audit import compute_fundamental_confidence

        audit_data = {
            "analyst_upside_pct": 15,
            "analyst_num": 10,
            "beat_streak": 4,
            "avg_beat_magnitude": 10,
            "last_beat_pct": 8,
            "roe": 0.20,
            "debt_to_equity": 0.5,
            "free_cash_flow_per_share": 4,
            "current_ratio": 2.0,
        }

        # stock_data = None should not raise
        confidence, breakdown = compute_fundamental_confidence(audit_data, None)
        assert isinstance(confidence, float)
        assert isinstance(breakdown, dict)
        # Insider and revision components should be 0 without stock data
        assert breakdown["insider_conviction"] == 0
        assert breakdown["estimate_revisions"] == 25  # eps_rev 0 → score 25

    def test_empty_audit_data(self):
        """Graceful handling of completely empty audit data."""
        from backend.earnings_audit import compute_fundamental_confidence

        confidence, breakdown = compute_fundamental_confidence({}, {})
        # With empty data:
        # - analyst_upside: 0 (no upside)
        # - beat_quality: 0 (no beats)
        # - financial_health: 30 (debt_to_equity 0 → low debt score 30)
        # - insider_conviction: 0 (no data)
        # - estimate_revisions: 25 (0% revision → neutral band score 25)
        # Weighted total: 30*0.20 + 25*0.20 = 11.0
        assert isinstance(confidence, float)
        assert confidence == 11.0  # Baseline for unknown/empty data
        assert breakdown["analyst_upside"] == 0
        assert breakdown["beat_quality"] == 0
        assert breakdown["insider_conviction"] == 0


class TestGetAuditBonus:
    """Test the confidence → composite bonus mapping."""

    def test_high_confidence_bonus(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(85) == 10  # >= 70 → +10

    def test_medium_confidence_bonus(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(55) == 5   # >= 50 → +5

    def test_no_bonus_zone(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(40) == 0   # 30-50 → 0

    def test_low_confidence_penalty(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(20) == -5  # < 30 → -5

    def test_none_confidence(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(None) == 0  # Graceful degradation

    def test_boundary_high(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(70) == 10  # Exactly 70

    def test_boundary_medium(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(50) == 5   # Exactly 50

    def test_boundary_low(self):
        from backend.earnings_audit import get_audit_bonus
        assert get_audit_bonus(30) == 0   # Exactly 30 (not < 30)


class TestAnalystSubScore:
    """Test analyst upside scoring component."""

    def test_few_analysts_penalty(self):
        """Less than 3 analysts should halve the score."""
        from backend.earnings_audit import compute_fundamental_confidence

        # Same upside but different analyst counts
        audit_many = {"analyst_upside_pct": 25, "analyst_num": 10,
                      "beat_streak": 0, "avg_beat_magnitude": 0, "roe": 0,
                      "debt_to_equity": 0, "free_cash_flow_per_share": 0}
        audit_few = {"analyst_upside_pct": 25, "analyst_num": 2,
                     "beat_streak": 0, "avg_beat_magnitude": 0, "roe": 0,
                     "debt_to_equity": 0, "free_cash_flow_per_share": 0}

        _, breakdown_many = compute_fundamental_confidence(audit_many, {})
        _, breakdown_few = compute_fundamental_confidence(audit_few, {})

        # Few analysts should be exactly half
        assert breakdown_few["analyst_upside"] == breakdown_many["analyst_upside"] * 0.5


class TestBeatQualitySubScore:
    """Test earnings beat quality scoring."""

    def test_streak_scoring(self):
        """Higher beat streaks should score progressively higher."""
        from backend.earnings_audit import compute_fundamental_confidence

        scores = []
        for streak in range(6):
            audit = {"analyst_upside_pct": 0, "analyst_num": 0,
                     "beat_streak": streak, "avg_beat_magnitude": 0, "roe": 0,
                     "debt_to_equity": 0, "free_cash_flow_per_share": 0}
            _, breakdown = compute_fundamental_confidence(audit, {})
            scores.append(breakdown["beat_quality"])

        # Each streak level should be >= previous
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], \
                f"Streak {i} ({scores[i]}) should be >= streak {i - 1} ({scores[i - 1]})"


class TestFinancialHealthSubScore:
    """Test financial health scoring."""

    def test_high_roe_low_debt(self):
        from backend.earnings_audit import compute_fundamental_confidence

        audit = {"analyst_upside_pct": 0, "analyst_num": 0,
                 "beat_streak": 0, "avg_beat_magnitude": 0,
                 "roe": 0.30, "debt_to_equity": 0.2, "free_cash_flow_per_share": 6}
        _, breakdown = compute_fundamental_confidence(audit, {})
        assert breakdown["financial_health"] == 100  # 50 + 30 + 20

    def test_negative_roe_high_debt(self):
        from backend.earnings_audit import compute_fundamental_confidence

        audit = {"analyst_upside_pct": 0, "analyst_num": 0,
                 "beat_streak": 0, "avg_beat_magnitude": 0,
                 "roe": -0.10, "debt_to_equity": 5.0, "free_cash_flow_per_share": -3}
        _, breakdown = compute_fundamental_confidence(audit, {})
        assert breakdown["financial_health"] == 0


class TestGetLatestAudit:
    """Test audit freshness lookup."""

    def test_returns_none_when_no_audit(self):
        """Should return None when no audit exists for ticker."""
        from backend.earnings_audit import get_latest_audit

        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = None

        result = get_latest_audit(mock_db, "AAPL")
        assert result is None
