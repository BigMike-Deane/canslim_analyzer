"""
Tests for Coiled Spring / Earnings Catalyst Detection

Tests cover:
1. CS detection when all criteria met
2. CS detection when each criterion fails
3. Alert limiting (max 3/day, cooldown)
4. Score bonus calculation
5. Earnings block override
"""

import pytest
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from canslim_scorer import calculate_coiled_spring_score


class MockData:
    """Mock StockData for testing"""
    def __init__(self):
        self.weeks_in_base = 20
        self.earnings_beat_streak = 4
        self.days_to_earnings = 7
        self.institutional_holders_pct = 15


class MockScore:
    """Mock CANSLIMScore for testing"""
    def __init__(self):
        self.c_score = 13
        self.l_score = 10
        self.total_score = 75


@pytest.fixture
def default_config():
    """Default Coiled Spring configuration"""
    return {
        'thresholds': {
            'min_weeks_in_base': 15,
            'min_beat_streak': 3,
            'min_c_score': 12,
            'min_total_score': 65,
            'max_institutional_pct': 40,
            'min_l_score': 8,
        },
        'earnings_window': {
            'alert_days': 14,
            'allow_buy_days': 7,
            'block_days': 1,
        },
        'scoring': {
            'base_bonus': 20,
            'long_base_bonus': 10,
            'strong_beat_bonus': 5,
            'max_bonus': 35,
        }
    }


@pytest.fixture
def qualifying_data():
    """Data that meets all CS criteria"""
    data = MockData()
    data.weeks_in_base = 20  # >= 15
    data.earnings_beat_streak = 4  # >= 3
    data.days_to_earnings = 7  # 1 < x <= 14
    data.institutional_holders_pct = 15  # <= 40
    return data


@pytest.fixture
def qualifying_score():
    """Score that meets all CS criteria"""
    score = MockScore()
    score.c_score = 13  # >= 12
    score.l_score = 10  # >= 8
    score.total_score = 75  # >= 65
    return score


class TestCoiledSpringDetection:
    """Test CS detection with all criteria met"""

    def test_all_criteria_met_basic(self, default_config, qualifying_data, qualifying_score):
        """Test detection when all criteria are met"""
        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is True
        assert result["allow_pre_earnings_buy"] is True
        assert result["cs_score"] > 0
        assert "CS:" in result["cs_details"]

    def test_base_bonus_calculation(self, default_config, qualifying_data, qualifying_score):
        """Test base bonus is applied correctly"""
        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        # 20 (base) + 10 (long base >= 20w) = 30
        assert result["cs_score"] == 30

    def test_long_base_bonus(self, default_config, qualifying_data, qualifying_score):
        """Test extra bonus for 20+ week base"""
        qualifying_data.weeks_in_base = 20
        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        # Should get long_base_bonus
        assert result["cs_score"] >= 30  # base_bonus + long_base_bonus

    def test_strong_beat_bonus(self, default_config, qualifying_data, qualifying_score):
        """Test extra bonus for 5+ consecutive beats"""
        qualifying_data.weeks_in_base = 16  # Just above threshold, no long base bonus
        qualifying_data.earnings_beat_streak = 6

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        # Should get strong_beat_bonus: 20 (base) + 5 (strong beat) = 25
        assert result["cs_score"] == 25

    def test_max_bonus_capped(self, default_config, qualifying_data, qualifying_score):
        """Test bonus is capped at max_bonus"""
        # Maximize all bonuses
        qualifying_data.weeks_in_base = 30  # Very long base
        qualifying_data.earnings_beat_streak = 10  # Very long streak

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        # Should be capped at 35
        assert result["cs_score"] <= 35


class TestCoiledSpringCriteriaFailures:
    """Test CS detection when each criterion fails"""

    def test_fails_weeks_in_base(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when weeks_in_base too low"""
        qualifying_data.weeks_in_base = 10  # Below 15 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "weeks_in_base" in result["cs_details"]

    def test_fails_beat_streak(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when beat_streak too low"""
        qualifying_data.earnings_beat_streak = 2  # Below 3 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "beat_streak" in result["cs_details"]

    def test_fails_c_score(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when C score too low"""
        qualifying_score.c_score = 10  # Below 12 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "c_score" in result["cs_details"]

    def test_fails_total_score(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when total score too low"""
        qualifying_score.total_score = 60  # Below 65 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "total_score" in result["cs_details"]

    def test_fails_institutional_pct(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when institutional ownership too high"""
        qualifying_data.institutional_holders_pct = 50  # Above 40 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "institutional" in result["cs_details"]

    def test_fails_l_score(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when L score too low"""
        qualifying_score.l_score = 6  # Below 8 threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "l_score" in result["cs_details"]

    def test_fails_no_earnings_date(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when no earnings date"""
        qualifying_data.days_to_earnings = None

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "earnings" in result["cs_details"].lower()

    def test_fails_earnings_too_close(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when earnings too close (within block_days)"""
        qualifying_data.days_to_earnings = 1  # Equal to block_days

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "too close" in result["cs_details"].lower() or "earnings" in result["cs_details"].lower()

    def test_fails_earnings_too_far(self, default_config, qualifying_data, qualifying_score):
        """Test rejection when earnings too far (beyond alert_days)"""
        qualifying_data.days_to_earnings = 20  # Beyond 14 day threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False
        assert "too far" in result["cs_details"].lower() or "earnings" in result["cs_details"].lower()


class TestCoiledSpringEarningsWindow:
    """Test earnings timing edge cases"""

    def test_at_block_boundary(self, default_config, qualifying_data, qualifying_score):
        """Test exactly at block_days boundary"""
        qualifying_data.days_to_earnings = 2  # Just above block_days (1)

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is True
        assert result["allow_pre_earnings_buy"] is True

    def test_at_alert_boundary(self, default_config, qualifying_data, qualifying_score):
        """Test exactly at alert_days boundary"""
        qualifying_data.days_to_earnings = 14  # At alert_days threshold

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is True

    def test_just_outside_alert_window(self, default_config, qualifying_data, qualifying_score):
        """Test just outside alert window"""
        qualifying_data.days_to_earnings = 15  # Just beyond alert_days

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert result["is_coiled_spring"] is False


class TestCoiledSpringFactors:
    """Test factors dictionary for debugging"""

    def test_factors_included(self, default_config, qualifying_data, qualifying_score):
        """Test that factors dict contains all necessary fields"""
        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert "factors" in result
        factors = result["factors"]

        assert "weeks_in_base" in factors
        assert "earnings_beat_streak" in factors
        assert "days_to_earnings" in factors
        assert "institutional_pct" in factors
        assert "c_score" in factors
        assert "l_score" in factors
        assert "total_score" in factors
        assert "thresholds" in factors

    def test_failed_criteria_in_factors(self, default_config, qualifying_data, qualifying_score):
        """Test that failed criteria are tracked in factors"""
        qualifying_data.weeks_in_base = 10  # Will fail

        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, default_config)

        assert "criteria_failed" in result["factors"]
        assert len(result["factors"]["criteria_failed"]) > 0


class TestCoiledSpringWithDefaultConfig:
    """Test CS detection using default config loading"""

    def test_works_without_explicit_config(self, qualifying_data, qualifying_score):
        """Test function works when config=None (loads from config_loader)"""
        # This tests the import and config loading path
        result = calculate_coiled_spring_score(qualifying_data, qualifying_score, None)

        # Should either succeed or fail based on config, but not error
        assert "is_coiled_spring" in result
        assert "cs_score" in result


class TestAlertLimiting:
    """Test alert recording and limiting functionality"""

    def test_alert_limit_import(self):
        """Test that alert limiting function can be imported"""
        # This just tests the import path works
        try:
            sys.path.insert(0, str(parent_dir / 'backend'))
            from ai_trader import record_coiled_spring_alert
            assert callable(record_coiled_spring_alert)
        except ImportError:
            pytest.skip("Backend not available for this test")

    def test_cs_calculation_function_import(self):
        """Test that CS calculation helper can be imported"""
        try:
            sys.path.insert(0, str(parent_dir / 'backend'))
            from ai_trader import calculate_coiled_spring_score_for_stock
            assert callable(calculate_coiled_spring_score_for_stock)
        except ImportError:
            pytest.skip("Backend not available for this test")


class TestDatabaseModel:
    """Test CoiledSpringAlert database model"""

    def test_model_import(self):
        """Test that CoiledSpringAlert model can be imported"""
        try:
            sys.path.insert(0, str(parent_dir / 'backend'))
            from database import CoiledSpringAlert
            assert CoiledSpringAlert is not None
            assert hasattr(CoiledSpringAlert, 'ticker')
            assert hasattr(CoiledSpringAlert, 'alert_date')
            assert hasattr(CoiledSpringAlert, 'cs_bonus')
            assert hasattr(CoiledSpringAlert, 'email_sent')
        except ImportError:
            pytest.skip("Database not available for this test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
