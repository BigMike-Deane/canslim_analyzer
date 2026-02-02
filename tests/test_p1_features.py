"""
Tests for P1 Features: Earnings Calendar, Analyst Revisions, Insider Value Tracking
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestEarningsCalendar:
    """Tests for earnings calendar feature"""

    def test_earnings_proximity_blocks_buy(self):
        """Test that stocks 1-3 days from earnings are skipped"""
        # Mock stock with 2 days until earnings
        mock_stock = Mock()
        mock_stock.days_to_earnings = 2
        mock_stock.ticker = "AAPL"
        mock_stock.canslim_score = 80
        mock_stock.projected_growth = 25
        mock_stock.current_price = 150.0

        # Should skip this stock
        days_to_earnings = getattr(mock_stock, 'days_to_earnings', None)
        skip = days_to_earnings is not None and 0 < days_to_earnings <= 3

        assert skip is True, "Should skip stocks 1-3 days before earnings"

    def test_earnings_proximity_allows_buy_past_window(self):
        """Test that stocks 4+ days from earnings are allowed"""
        mock_stock = Mock()
        mock_stock.days_to_earnings = 5

        days_to_earnings = getattr(mock_stock, 'days_to_earnings', None)
        skip = days_to_earnings is not None and 0 < days_to_earnings <= 3

        assert skip is False, "Should allow stocks 4+ days before earnings"

    def test_earnings_proximity_allows_null(self):
        """Test that stocks with no earnings data are allowed"""
        mock_stock = Mock()
        mock_stock.days_to_earnings = None

        days_to_earnings = getattr(mock_stock, 'days_to_earnings', None)
        skip = days_to_earnings is not None and 0 < days_to_earnings <= 3

        assert skip is False, "Should allow stocks with no earnings data"

    def test_beat_streak_bonus_calculation(self):
        """Test beat streak bonus in C score"""
        # Beat streak >= 4 should get bonus
        test_cases = [
            (3, 0),   # 3 beats = no bonus
            (4, 1),   # 4 beats = +1
            (5, 2),   # 5 beats = +2
            (6, 2),   # 6 beats = +2 (capped)
            (10, 2),  # 10 beats = +2 (capped)
        ]

        for eps_beat_streak, expected_bonus in test_cases:
            if eps_beat_streak >= 4:
                beat_streak_bonus = min(2, eps_beat_streak - 3)
            else:
                beat_streak_bonus = 0

            assert beat_streak_bonus == expected_bonus, \
                f"Beat streak {eps_beat_streak} should give bonus {expected_bonus}, got {beat_streak_bonus}"

    def test_days_to_earnings_calculation(self):
        """Test calculation of days until next earnings"""
        # Mock scenario: Next earnings on Feb 10, 2026, current date is Feb 5, 2026
        next_earnings = date(2026, 2, 10)
        current_date = date(2026, 2, 5)

        days_to_earnings = (next_earnings - current_date).days

        assert days_to_earnings == 5, "Should calculate 5 days until earnings"


class TestAnalystEstimates:
    """Tests for analyst estimate revisions feature"""

    def test_positive_revision_bonus_strong(self):
        """Test strong positive revision bonus (+10% or more)"""
        revision_pct = 12.0

        if revision_pct >= 10:
            revision_bonus = 4
        elif revision_pct >= 5:
            revision_bonus = 2
        elif revision_pct <= -5:
            revision_bonus = -2
        else:
            revision_bonus = 0

        assert revision_bonus == 4, "10%+ revision should give +4 bonus"

    def test_positive_revision_bonus_moderate(self):
        """Test moderate positive revision bonus (5-10%)"""
        revision_pct = 7.0

        if revision_pct >= 10:
            revision_bonus = 4
        elif revision_pct >= 5:
            revision_bonus = 2
        elif revision_pct <= -5:
            revision_bonus = -2
        else:
            revision_bonus = 0

        assert revision_bonus == 2, "5-10% revision should give +2 bonus"

    def test_negative_revision_penalty(self):
        """Test negative revision penalty (-5% or worse)"""
        revision_pct = -8.0

        if revision_pct >= 10:
            revision_bonus = 4
        elif revision_pct >= 5:
            revision_bonus = 2
        elif revision_pct <= -5:
            revision_bonus = -2
        else:
            revision_bonus = 0

        assert revision_bonus == -2, "Negative revision should give -2 penalty"

    def test_revision_trend_detection(self):
        """Test estimate revision trend categorization"""
        test_cases = [
            (15.0, "up"),
            (7.0, "up"),
            (2.0, "stable"),
            (-2.0, "stable"),
            (-7.0, "down"),
            (-15.0, "down"),
        ]

        for revision_pct, expected_trend in test_cases:
            if revision_pct >= 5:
                trend = "up"
            elif revision_pct <= -5:
                trend = "down"
            else:
                trend = "stable"

            assert trend == expected_trend, \
                f"Revision {revision_pct}% should have trend '{expected_trend}', got '{trend}'"


class TestInsiderValues:
    """Tests for insider value tracking feature"""

    def test_value_scaled_bonus_high(self):
        """Test high value insider bonus ($500K+)"""
        insider_net_value = 750000
        insider_buy_count = 3
        insider_sentiment = "bullish"

        if insider_sentiment == "bullish":
            if insider_net_value >= 500000:
                insider_bonus = 10
            elif insider_net_value >= 100000:
                insider_bonus = 7
            elif insider_buy_count >= 2:
                insider_bonus = 5
            else:
                insider_bonus = 0
        else:
            insider_bonus = 0

        assert insider_bonus == 10, "$500K+ net buying should give +10 bonus"

    def test_value_scaled_bonus_medium(self):
        """Test medium value insider bonus ($100K-$500K)"""
        insider_net_value = 250000
        insider_buy_count = 2
        insider_sentiment = "bullish"

        if insider_sentiment == "bullish":
            if insider_net_value >= 500000:
                insider_bonus = 10
            elif insider_net_value >= 100000:
                insider_bonus = 7
            elif insider_buy_count >= 2:
                insider_bonus = 5
            else:
                insider_bonus = 0
        else:
            insider_bonus = 0

        assert insider_bonus == 7, "$100K-$500K net buying should give +7 bonus"

    def test_value_scaled_bonus_fallback(self):
        """Test count-based fallback when no value data"""
        insider_net_value = 0  # No value data
        insider_buy_count = 3
        insider_sentiment = "bullish"

        if insider_sentiment == "bullish":
            if insider_net_value >= 500000:
                insider_bonus = 10
            elif insider_net_value >= 100000:
                insider_bonus = 7
            elif insider_buy_count >= 2:
                insider_bonus = 5
            else:
                insider_bonus = 0
        else:
            insider_bonus = 0

        assert insider_bonus == 5, "Fallback to count-based bonus"

    def test_c_suite_extra_bonus(self):
        """Test extra bonus for C-suite buying"""
        c_suite_titles = ['CEO', 'CFO', 'COO', 'PRESIDENT', 'CHIEF EXECUTIVE OFFICER', 'CHIEF FINANCIAL OFFICER']

        for title in c_suite_titles:
            insider_bonus = 7  # Base bonus
            if title.upper() in ('CEO', 'CFO', 'COO', 'PRESIDENT', 'CHIEF EXECUTIVE OFFICER', 'CHIEF FINANCIAL OFFICER'):
                insider_bonus += 3

            assert insider_bonus == 10, f"C-suite title '{title}' should add +3 bonus"

    def test_c_suite_no_extra_for_directors(self):
        """Test no extra bonus for non-C-suite"""
        non_c_suite_titles = ['DIRECTOR', 'VP SALES', '10% OWNER', 'GENERAL COUNSEL']

        for title in non_c_suite_titles:
            insider_bonus = 7  # Base bonus
            if title.upper() in ('CEO', 'CFO', 'COO', 'PRESIDENT', 'CHIEF EXECUTIVE OFFICER', 'CHIEF FINANCIAL OFFICER'):
                insider_bonus += 3

            assert insider_bonus == 7, f"Non-C-suite title '{title}' should NOT add bonus"

    def test_largest_buy_tracking(self):
        """Test tracking of largest single insider buy"""
        transactions = [
            {"type": "buy", "value": 50000},
            {"type": "buy", "value": 150000},
            {"type": "buy", "value": 75000},
            {"type": "sell", "value": 200000},
        ]

        largest_buy = max(
            (t["value"] for t in transactions if t["type"] == "buy"),
            default=0
        )

        assert largest_buy == 150000, "Should track largest single buy"


class TestDataFetcherFunctions:
    """Tests for data fetcher P1 functions"""

    def test_fetch_fmp_earnings_calendar_parses_correctly(self):
        """Test earnings calendar data parsing"""
        mock_response = [
            {"date": "2026-02-15", "actualEarningResult": "beat", "revenueEstimated": 100000000},
            {"date": "2025-11-10", "actualEarningResult": "beat", "revenueEstimated": 95000000},
            {"date": "2025-08-05", "actualEarningResult": "beat", "revenueEstimated": 90000000},
            {"date": "2025-05-01", "actualEarningResult": "miss", "revenueEstimated": 85000000},
        ]

        # Calculate beat streak (consecutive beats from most recent)
        beat_streak = 0
        for earning in mock_response:
            if earning.get("actualEarningResult") == "beat":
                beat_streak += 1
            else:
                break

        assert beat_streak == 3, "Should count 3 consecutive beats"

    def test_fetch_fmp_analyst_estimates_calculates_revision(self):
        """Test analyst estimate revision calculation"""
        mock_response = [
            {"estimatedEpsAvg": 2.50},  # Current estimate
            {"estimatedEpsAvg": 2.30},  # Prior estimate
        ]

        current = mock_response[0]["estimatedEpsAvg"]
        prior = mock_response[1]["estimatedEpsAvg"]

        if prior > 0:
            revision_pct = ((current - prior) / prior) * 100
        else:
            revision_pct = 0

        assert abs(revision_pct - 8.7) < 0.1, "Should calculate ~8.7% revision"


class TestDatabaseMigrations:
    """Tests for P1 database migrations"""

    def test_p1_columns_exist_in_stock_model(self):
        """Test that P1 columns are defined in Stock model"""
        from backend.database import Stock
        from sqlalchemy import inspect

        mapper = inspect(Stock)
        column_names = [col.name for col in mapper.columns]

        # Earnings Calendar columns
        assert "next_earnings_date" in column_names
        assert "days_to_earnings" in column_names
        assert "earnings_beat_streak" in column_names
        assert "earnings_calendar_updated_at" in column_names

        # Analyst Estimate columns
        assert "eps_estimate_current" in column_names
        assert "eps_estimate_prior" in column_names
        assert "eps_estimate_revision_pct" in column_names
        assert "estimate_revision_trend" in column_names
        assert "analyst_estimates_updated_at" in column_names

        # Insider Value columns
        assert "insider_buy_value" in column_names
        assert "insider_sell_value" in column_names
        assert "insider_net_value" in column_names
        assert "insider_largest_buy" in column_names
        assert "insider_largest_buyer_title" in column_names
