"""Per-user webhook routing tests (Phase 1).

Verifies that send_*_webhook helpers route to each user's own webhook_url and
silently no-op when a user has no URL configured. Also verifies that the
explicit `url` parameter on send_webhook_notification overrides the global env.
"""
from unittest.mock import patch, MagicMock

import pytest


class TestSendWebhookNotificationOverride:
    @patch("backend.email_utils.requests.post")
    def test_explicit_url_overrides_global(self, mock_post):
        from backend.email_utils import send_webhook_notification
        mock_post.return_value = MagicMock(status_code=200, text="ok")

        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_webhook_notification(
                "Test", "Hello", url="https://user-specific.example/abc"
            )

        assert ok is True
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://user-specific.example/abc"

    @patch("backend.email_utils.requests.post")
    def test_empty_url_silences_even_when_global_set(self, mock_post):
        from backend.email_utils import send_webhook_notification
        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_webhook_notification("Test", "Hello", url="")
        assert ok is False
        assert mock_post.call_count == 0

    @patch("backend.email_utils.requests.post")
    def test_none_url_falls_back_to_global(self, mock_post):
        from backend.email_utils import send_webhook_notification
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_webhook_notification("Test", "Hello", url=None)
        assert ok is True
        assert mock_post.call_args[0][0] == "https://global.example/topic"


class TestGetUserWebhookUrl:
    def test_returns_url_when_user_has_one(self):
        from backend.email_utils import get_user_webhook_url
        fake_user = MagicMock(webhook_url="https://ntfy.sh/user1-private")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.email_utils.SessionLocal" if False else "backend.database.SessionLocal",
                   return_value=fake_db):
            url = get_user_webhook_url(1)
        assert url == "https://ntfy.sh/user1-private"

    def test_returns_empty_string_when_user_has_no_url(self):
        from backend.email_utils import get_user_webhook_url
        fake_user = MagicMock(webhook_url=None)
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.database.SessionLocal", return_value=fake_db):
            url = get_user_webhook_url(2)
        assert url == ""

    def test_returns_empty_string_when_user_not_found(self):
        from backend.email_utils import get_user_webhook_url
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = None

        with patch("backend.database.SessionLocal", return_value=fake_db):
            url = get_user_webhook_url(999)
        assert url == ""

    def test_returns_empty_string_on_db_error(self):
        from backend.email_utils import get_user_webhook_url
        with patch("backend.database.SessionLocal", side_effect=RuntimeError("db down")):
            url = get_user_webhook_url(1)
        assert url == ""

    def test_returns_empty_string_for_falsy_user_id(self):
        from backend.email_utils import get_user_webhook_url
        assert get_user_webhook_url(None) == ""
        assert get_user_webhook_url(0) == ""


class TestPerUserTradeRouting:
    @patch("backend.email_utils.requests.post")
    def test_trade_webhook_routes_to_user_url(self, mock_post):
        """User 1 with their own URL receives the notification at THAT URL."""
        from backend.email_utils import send_trade_webhook
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        fake_user = MagicMock(webhook_url="https://ntfy.sh/user1-secret")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_trade_webhook("AAPL", "BUY", 10, 150.0, "test reason", user_id=1)

        assert ok is True
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://ntfy.sh/user1-secret"

    @patch("backend.email_utils.requests.post")
    def test_trade_webhook_silent_when_user_has_no_url(self, mock_post):
        """User 2 with no URL gets nothing — even if global URL is set."""
        from backend.email_utils import send_trade_webhook
        fake_user = MagicMock(webhook_url=None)
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_trade_webhook("AAPL", "BUY", 10, 150.0, "test reason", user_id=2)

        assert ok is False
        assert mock_post.call_count == 0

    @patch("backend.email_utils.requests.post")
    def test_trade_webhook_legacy_no_user_id_uses_global(self, mock_post):
        """Backwards compat: callers that don't pass user_id still use global env URL."""
        from backend.email_utils import send_trade_webhook
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            ok = send_trade_webhook("AAPL", "BUY", 10, 150.0, "test reason")
        assert ok is True
        assert mock_post.call_args[0][0] == "https://global.example/topic"

    @patch("backend.email_utils.requests.post")
    def test_stop_loss_webhook_routes_per_user(self, mock_post):
        from backend.email_utils import send_stop_loss_webhook
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        fake_user = MagicMock(webhook_url="https://ntfy.sh/user1-secret")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            send_stop_loss_webhook("AAPL", 10, 140.0, "STOP LOSS", -7.0, user_id=1)

        assert mock_post.call_args[0][0] == "https://ntfy.sh/user1-secret"

    @patch("backend.email_utils.requests.post")
    def test_score_crash_warning_routes_per_user(self, mock_post):
        from backend.email_utils import send_score_crash_warning_push
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        fake_user = MagicMock(webhook_url="https://ntfy.sh/user1-secret")
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.first.return_value = fake_user

        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            send_score_crash_warning_push("AAPL", 80, 60, 5.0, 1, 3, user_id=1)

        assert mock_post.call_args[0][0] == "https://ntfy.sh/user1-secret"
