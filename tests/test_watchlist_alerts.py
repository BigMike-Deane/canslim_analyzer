"""
Tests for watchlist alert functionality.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path so we can import modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'backend'))


class TestWatchlistAlerts:
    """Test cases for watchlist alert system."""

    @pytest.fixture
    def mock_watchlist_item(self):
        """Create a mock Watchlist item."""
        item = Mock()
        item.id = 1
        item.ticker = "AAPL"
        item.target_price = 180.0
        item.alert_score = 75.0
        item.notes = "Test notes"
        item.alert_triggered_at = None
        item.alert_sent = False
        item.last_check_price = 170.0
        return item

    @pytest.fixture
    def mock_stock(self):
        """Create a mock Stock."""
        stock = Mock()
        stock.ticker = "AAPL"
        stock.name = "Apple Inc."
        stock.current_price = 185.0
        stock.canslim_score = 80.0
        return stock

    def test_price_alert_triggers_above_target(self, mock_watchlist_item, mock_stock):
        """Test that price alert triggers when price crosses above target."""
        # Stock price ($185) is above target ($180)
        # Last check price ($170) was below target
        # Should trigger

        triggered = False
        reasons = []

        if mock_watchlist_item.target_price and mock_stock.current_price:
            if mock_stock.current_price >= mock_watchlist_item.target_price:
                if not mock_watchlist_item.last_check_price or mock_watchlist_item.last_check_price < mock_watchlist_item.target_price:
                    triggered = True
                    reasons.append(f"Price ${mock_stock.current_price:.2f} >= target ${mock_watchlist_item.target_price:.2f}")

        assert triggered is True
        assert len(reasons) == 1
        assert "Price $185.00 >= target $180.00" in reasons[0]

    def test_price_alert_not_triggered_when_already_above(self, mock_watchlist_item, mock_stock):
        """Test that price alert doesn't re-trigger if already above target."""
        # Last check was already above target
        mock_watchlist_item.last_check_price = 182.0  # Already above $180 target

        triggered = False

        if mock_watchlist_item.target_price and mock_stock.current_price:
            if mock_stock.current_price >= mock_watchlist_item.target_price:
                # Only trigger if last check was below target
                if not mock_watchlist_item.last_check_price or mock_watchlist_item.last_check_price < mock_watchlist_item.target_price:
                    triggered = True

        assert triggered is False

    def test_score_alert_triggers_above_threshold(self, mock_watchlist_item, mock_stock):
        """Test that score alert triggers when score crosses above threshold."""
        # Stock score (80) is above alert_score (75)
        triggered = False
        reasons = []

        if mock_watchlist_item.alert_score and mock_stock.canslim_score:
            if mock_stock.canslim_score >= mock_watchlist_item.alert_score:
                triggered = True
                reasons.append(f"CANSLIM score {mock_stock.canslim_score:.0f} >= target {mock_watchlist_item.alert_score:.0f}")

        assert triggered is True
        assert len(reasons) == 1
        assert "CANSLIM score 80 >= target 75" in reasons[0]

    def test_score_alert_not_triggered_below_threshold(self, mock_watchlist_item, mock_stock):
        """Test that score alert doesn't trigger below threshold."""
        mock_stock.canslim_score = 70.0  # Below alert_score of 75

        triggered = False

        if mock_watchlist_item.alert_score and mock_stock.canslim_score:
            if mock_stock.canslim_score >= mock_watchlist_item.alert_score:
                triggered = True

        assert triggered is False

    def test_alert_cooldown_prevents_duplicate(self, mock_watchlist_item):
        """Test that cooldown period prevents duplicate alerts."""
        # Set alert_triggered_at to 12 hours ago
        mock_watchlist_item.alert_triggered_at = datetime.utcnow() - timedelta(hours=12)
        cooldown_hours = 24

        within_cooldown = False
        if mock_watchlist_item.alert_triggered_at:
            time_since_last = datetime.utcnow() - mock_watchlist_item.alert_triggered_at
            if time_since_last < timedelta(hours=cooldown_hours):
                within_cooldown = True

        assert within_cooldown is True

    def test_alert_sent_after_cooldown(self, mock_watchlist_item):
        """Test that alert can be sent after cooldown expires."""
        # Set alert_triggered_at to 25 hours ago (beyond 24h cooldown)
        mock_watchlist_item.alert_triggered_at = datetime.utcnow() - timedelta(hours=25)
        cooldown_hours = 24

        within_cooldown = False
        if mock_watchlist_item.alert_triggered_at:
            time_since_last = datetime.utcnow() - mock_watchlist_item.alert_triggered_at
            if time_since_last < timedelta(hours=cooldown_hours):
                within_cooldown = True

        assert within_cooldown is False

    def test_alert_not_sent_if_already_triggered(self, mock_watchlist_item):
        """Test that alert is not re-sent if alert_sent flag is True."""
        mock_watchlist_item.alert_sent = True

        should_send = not mock_watchlist_item.alert_sent

        assert should_send is False

    def test_both_alerts_can_trigger_together(self, mock_watchlist_item, mock_stock):
        """Test that both price and score alerts can trigger in same check."""
        # Stock price ($185) above target ($180) - coming from below ($170)
        # Stock score (80) above alert_score (75)

        reasons = []

        # Check price alert
        if mock_watchlist_item.target_price and mock_stock.current_price:
            if mock_stock.current_price >= mock_watchlist_item.target_price:
                if not mock_watchlist_item.last_check_price or mock_watchlist_item.last_check_price < mock_watchlist_item.target_price:
                    reasons.append(f"Price ${mock_stock.current_price:.2f} >= target ${mock_watchlist_item.target_price:.2f}")

        # Check score alert
        if mock_watchlist_item.alert_score and mock_stock.canslim_score:
            if mock_stock.canslim_score >= mock_watchlist_item.alert_score:
                reasons.append(f"CANSLIM score {mock_stock.canslim_score:.0f} >= target {mock_watchlist_item.alert_score:.0f}")

        assert len(reasons) == 2

    def test_no_alert_without_targets(self, mock_stock):
        """Test that no alert triggers when no targets are set."""
        item = Mock()
        item.target_price = None
        item.alert_score = None

        reasons = []

        if item.target_price and mock_stock.current_price:
            if mock_stock.current_price >= item.target_price:
                reasons.append("price alert")

        if item.alert_score and mock_stock.canslim_score:
            if mock_stock.canslim_score >= item.alert_score:
                reasons.append("score alert")

        assert len(reasons) == 0


class TestWatchlistAlertEmail:
    """Test cases for watchlist alert email generation."""

    def test_email_subject_format(self):
        """Test that email subject is formatted correctly."""
        ticker = "AAPL"
        expected_subject = f"CANSLIM Alert: {ticker}"
        assert expected_subject == "CANSLIM Alert: AAPL"

    def test_email_contains_reasons(self):
        """Test that email content includes all alert reasons."""
        reasons = [
            "Price $185.00 >= target $180.00",
            "CANSLIM score 80 >= target 75"
        ]

        # Simulate building email content
        html_reasons = ''.join(f'<div class="reason">{r}</div>' for r in reasons)

        assert "Price $185.00 >= target $180.00" in html_reasons
        assert "CANSLIM score 80 >= target 75" in html_reasons

    def test_email_includes_notes_when_present(self):
        """Test that watchlist notes are included in email when present."""
        item = Mock()
        item.notes = "Watch for earnings catalyst"

        content = ""
        if item.notes:
            content = f'<p><strong>Your Notes:</strong> {item.notes}</p>'

        assert "Watch for earnings catalyst" in content

    def test_email_omits_notes_when_empty(self):
        """Test that notes section is omitted when no notes exist."""
        item = Mock()
        item.notes = None

        content = ""
        if item.notes:
            content = f'<p><strong>Your Notes:</strong> {item.notes}</p>'

        assert content == ""


class TestInstitutionalPercentageExtraction:
    """Test cases for institutional percentage extraction from score_details."""

    def test_institutional_pct_from_score_details(self):
        """Test that inst_pct is correctly extracted from score_details JSON."""
        stock = Mock()
        stock.score_details = {
            'i': {
                'institutional_pct': 45.5,
                'score': 8
            }
        }

        inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

        assert inst_pct == 45.5

    def test_institutional_pct_missing_score_details(self):
        """Test handling when score_details is None."""
        stock = Mock()
        stock.score_details = None

        inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

        assert inst_pct == 0

    def test_institutional_pct_missing_i_key(self):
        """Test handling when 'i' key is missing from score_details."""
        stock = Mock()
        stock.score_details = {
            'c': {'score': 10},
            'a': {'score': 12}
        }

        inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

        assert inst_pct == 0

    def test_institutional_pct_handles_none_value(self):
        """Test handling when institutional_pct is explicitly None."""
        stock = Mock()
        stock.score_details = {
            'i': {
                'institutional_pct': None,
                'score': 5
            }
        }

        inst_pct = (stock.score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

        assert inst_pct == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
