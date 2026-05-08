"""Tests for backend/email_utils.py — closes uncovered ranges left after the
existing webhook-routing / notifications / ab_eval suites.

Focus areas (lines uncovered against 67% baseline as of commit 296013c):
- send_email SMTP body 53-81 (recipient= kwarg routing, error path)
- send_watchlist_alert_email 95-153 (full render + dispatch)
- send_webhook_notification 209-229 (ntfy headers + Timeout/RequestException/non-200)
- create_notification 279-292 (DB exception, push exception)
- _send_web_push_to_subscriptions 311-360 (ImportError, generic Exception, prune err)
- send_web_push_* 381-402 (DB exception paths)
- broadcast_notification 434-446 (DB + push exception)
- send_trade_webhook 512-519 (gain >= 0 vs < 0 tag branches)
- send_risk_alert_webhook 572-589 (each alert_type)
- send_morning_briefing_push 601-647 (full render + branches)
- send_scan_completion_push 664-683 (with/without trades)
- send_spy_gate_change_push 697-714 (bullish + bearish)
- send_morning_briefing_email 869-914 (alert suffix + earnings_this_week section)

All tests are eval-safe — no scoring writes, no trading, no real network. Push
delivery (pywebpush.webpush) is patched module-wide so VAPID keys never matter.
SMTP is patched via smtplib.SMTP context manager.
"""

import os
import sys
from unittest.mock import patch, MagicMock, ANY

import pytest
import requests

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")


# ============================================================================
# send_email — SMTP body, recipient kwarg routing, error path
# ============================================================================
class TestSendEmail:
    """The post-296013c recipient= kwarg is the load-bearing change here. It
    must route to BOTH msg['To'] (header the recipient sees) AND
    server.sendmail's to-addr (envelope SMTP gets). Earlier ab_eval tests only
    mocked the wrapper — these go through the real smtplib.SMTP path."""

    def _make_smtp_mock(self):
        """Build a MagicMock that satisfies the `with smtplib.SMTP(...) as server:`
        context-manager protocol."""
        server = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = server
        ctx.__exit__.return_value = False
        smtp_class = MagicMock(return_value=ctx)
        return smtp_class, server

    @patch("backend.email_utils.smtplib.SMTP")
    def test_happy_path_returns_true_and_calls_sendmail(self, mock_smtp):
        from backend.email_utils import send_email
        smtp_class, server = self._make_smtp_mock()
        # Replace the patched class with our context-manager-aware mock by
        # configuring the patched object's call to return our context.
        ctx = MagicMock()
        ctx.__enter__.return_value = server
        ctx.__exit__.return_value = False
        mock_smtp.return_value = ctx

        ok = send_email("Subj", "<p>hi</p>", "hi")
        assert ok is True
        # STARTTLS + login + sendmail all fired.
        server.starttls.assert_called_once()
        server.login.assert_called_once()
        server.sendmail.assert_called_once()

    @patch("backend.email_utils.smtplib.SMTP")
    def test_recipient_kwarg_routes_to_msg_To_and_sendmail_to_addr(self, mock_smtp):
        """recipient='ops@example.com' MUST appear (a) on the MIME header and
        (b) as the second positional arg to server.sendmail. This is what
        296013c actually changed and the regression risk is silently sending
        to the default mailbox."""
        from backend.email_utils import send_email
        server = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = server
        ctx.__exit__.return_value = False
        mock_smtp.return_value = ctx

        ok = send_email("Subj", "<p>x</p>", "x", recipient="ops@example.com")
        assert ok is True

        # sendmail(from_addr, to_addr, msg.as_string())
        sent_args = server.sendmail.call_args[0]
        assert sent_args[1] == "ops@example.com"
        assert "To: ops@example.com" in sent_args[2]

    @patch("backend.email_utils.smtplib.SMTP")
    def test_default_recipient_falls_back_to_module_constant(self, mock_smtp):
        """recipient=None → falls back to RECIPIENT_EMAIL. Patch the module
        constant so the test doesn't depend on env."""
        from backend import email_utils
        server = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = server
        ctx.__exit__.return_value = False
        mock_smtp.return_value = ctx

        with patch.object(email_utils, "RECIPIENT_EMAIL", "default@inbox.test"):
            ok = email_utils.send_email("S", "<p>h</p>", "h")
        assert ok is True
        assert server.sendmail.call_args[0][1] == "default@inbox.test"

    @patch("backend.email_utils.smtplib.SMTP", side_effect=OSError("network unreachable"))
    def test_returns_false_on_smtp_exception(self, _mock_smtp):
        """Exceptions are caught and logged — function MUST return False
        rather than raise. Trade pipeline relies on this."""
        from backend.email_utils import send_email
        ok = send_email("S", "<p>h</p>", "h")
        assert ok is False


# ============================================================================
# send_watchlist_alert_email — render + dispatch
# ============================================================================
class TestSendWatchlistAlertEmail:
    def test_renders_and_dispatches_via_send_email(self):
        from backend import email_utils
        item = MagicMock()
        item.ticker = "AAPL"
        item.notes = "watching for breakout"
        stock = MagicMock()
        stock.name = "Apple Inc."
        stock.current_price = 150.50
        stock.canslim_score = 88.0
        with patch.object(email_utils, "send_email", return_value=True) as mock_send:
            ok = email_utils.send_watchlist_alert_email(
                item, stock, ["Above 50MA", "Volume surge"]
            )
        assert ok is True
        args, _ = mock_send.call_args
        subject, html_body, text_body = args[0], args[1], args[2]
        assert "AAPL" in subject
        assert "AAPL" in html_body and "Apple Inc." in html_body
        assert "Above 50MA" in html_body and "Volume surge" in html_body
        assert "150.50" in html_body
        assert "88" in html_body  # canslim_score rendered
        assert "watching for breakout" in html_body
        # Plain-text fallback also present.
        assert "AAPL" in text_body
        assert "Above 50MA" in text_body

    def test_handles_missing_optional_fields(self):
        """name=None, current_price=None, canslim_score=None, notes=None must
        all render gracefully — these are real DB-state edge cases."""
        from backend import email_utils
        item = MagicMock(); item.ticker = "ZZZ"; item.notes = None
        stock = MagicMock(); stock.name = None; stock.current_price = None
        stock.canslim_score = None
        with patch.object(email_utils, "send_email", return_value=True) as mock_send:
            ok = email_utils.send_watchlist_alert_email(item, stock, ["reason"])
        assert ok is True
        html_body = mock_send.call_args[0][1]
        assert "Unknown" in html_body  # fallback for null name
        assert "$0.00" in html_body
        # No "Your Notes" block when item.notes is falsy.
        assert "Your Notes" not in html_body


# ============================================================================
# send_webhook_notification — ntfy header path, error paths
# ============================================================================
class TestSendWebhookNotification:
    @patch("backend.email_utils.requests.post")
    def test_ntfy_path_sends_tags_click_and_markdown_headers(self, mock_post):
        from backend.email_utils import send_webhook_notification
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        ok = send_webhook_notification(
            "Title", "msg", priority="urgent",
            tags=["fire", "warning"], click="https://x/y", markdown=True,
            url="https://ntfy.sh/topic-abc",
        )
        assert ok is True
        kwargs = mock_post.call_args.kwargs
        headers = kwargs["headers"]
        assert headers["Title"] == "Title"
        assert headers["Priority"] == "5"  # urgent maps to "5"
        assert headers["Tags"] == "fire,warning"
        assert headers["Click"] == "https://x/y"
        assert headers["Markdown"] == "yes"

    @patch("backend.email_utils.requests.post")
    def test_non_ntfy_url_uses_json_payload(self, mock_post):
        from backend.email_utils import send_webhook_notification
        mock_post.return_value = MagicMock(status_code=204, text="")
        ok = send_webhook_notification(
            "T", "M", priority="default", data={"k": "v"},
            url="https://discord.example/webhook",
        )
        assert ok is True
        kwargs = mock_post.call_args.kwargs
        # Standard JSON webhook path: json=, no header dict.
        assert kwargs["json"]["title"] == "T"
        assert kwargs["json"]["data"] == {"k": "v"}
        assert "headers" not in kwargs

    @patch("backend.email_utils.requests.post")
    def test_non_2xx_status_returns_false(self, mock_post):
        from backend.email_utils import send_webhook_notification
        mock_post.return_value = MagicMock(status_code=500, text="boom" * 100)
        ok = send_webhook_notification("T", "M", url="https://discord.example/wh")
        assert ok is False

    @patch("backend.email_utils.requests.post",
           side_effect=requests.exceptions.Timeout())
    def test_timeout_returns_false_and_doesnt_raise(self, _mock_post):
        from backend.email_utils import send_webhook_notification
        ok = send_webhook_notification("T", "M", url="https://discord.example/wh")
        assert ok is False

    @patch("backend.email_utils.requests.post",
           side_effect=requests.exceptions.RequestException("conn reset"))
    def test_request_exception_returns_false(self, _mock_post):
        from backend.email_utils import send_webhook_notification
        ok = send_webhook_notification("T", "M", url="https://discord.example/wh")
        assert ok is False

    def test_zero_user_id_returns_empty_url_short_circuit(self):
        """Defensive guard at line 235-236 — user_id=0 (or None) skips DB
        lookup entirely so an authless event doesn't query users.id == 0."""
        from backend.email_utils import get_user_webhook_url
        assert get_user_webhook_url(0) == ""
        assert get_user_webhook_url(None) == ""


# ============================================================================
# create_notification — fail-soft on DB and push
# ============================================================================
class TestCreateNotification:
    def test_zero_user_id_returns_false_short_circuit(self):
        from backend.email_utils import create_notification
        assert create_notification(0, "trade", "T", "B") is False

    def test_db_exception_returns_false_but_still_attempts_push(self):
        """If the DB insert raises, we still want to fire the push so the
        user gets the alert during a degraded-DB window. The return value
        reports DB success only."""
        from backend import email_utils
        with patch.object(email_utils, "SessionLocal" if False else "send_web_push_to_user",
                          return_value=1) as mock_push, \
             patch("backend.database.SessionLocal", side_effect=RuntimeError("db down")):
            ok = email_utils.create_notification(1, "trade", "T", "B",
                                                 data={"ticker": "AAPL"})
        assert ok is False
        # Push fan-out still ran with the kind+ticker URL injected.
        assert mock_push.call_count == 1
        push_data = mock_push.call_args.kwargs["data"]
        assert push_data["kind"] == "trade"
        assert push_data["url"] == "/stock/AAPL"

    def test_push_exception_does_not_raise(self):
        """Push errors must be caught; create_notification still returns the
        DB insert result."""
        from backend import email_utils
        fake_db = MagicMock()
        fake_db.add = MagicMock()
        fake_db.commit = MagicMock()
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "send_web_push_to_user",
                          side_effect=RuntimeError("push down")):
            ok = email_utils.create_notification(1, "trade", "T", "B")
        # DB succeeded, push raised but was swallowed.
        assert ok is True

    def test_url_defaults_to_notifications_when_no_ticker(self):
        from backend import email_utils
        fake_db = MagicMock()
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "send_web_push_to_user", return_value=0) as mock_push:
            email_utils.create_notification(1, "system", "T", "B", data={})
        assert mock_push.call_args.kwargs["data"]["url"] == "/notifications"

    def test_high_priority_routes_to_high_urgency_push(self):
        from backend import email_utils
        fake_db = MagicMock()
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "send_web_push_to_user", return_value=0) as mock_push:
            email_utils.create_notification(1, "stop_loss", "T", "B", priority="urgent")
        assert mock_push.call_args.kwargs["urgency"] == "high"


# ============================================================================
# _send_web_push_to_subscriptions — ImportError, expired, generic exception
# ============================================================================
class TestSendWebPushToSubscriptions:
    def test_empty_subscription_list_returns_zero(self):
        from backend.email_utils import _send_web_push_to_subscriptions
        assert _send_web_push_to_subscriptions([], "T", "B") == 0

    def test_pywebpush_missing_returns_zero_gracefully(self):
        """ImportError path 311-313 — pywebpush isn't always installed in CI.
        Force ImportError by removing it from sys.modules and patching the
        finder so a re-import fails."""
        from backend.email_utils import _send_web_push_to_subscriptions
        sub = MagicMock(id=1, endpoint="https://e", p256dh_key="k", auth_key="a")

        # Save and remove cached pywebpush so the local import hits real import.
        saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                      if k.startswith("pywebpush")}
        try:
            with patch.dict(sys.modules, {"pywebpush": None}):
                # sys.modules entry of None is treated as ImportError on import
                count = _send_web_push_to_subscriptions([sub], "T", "B")
            assert count == 0
        finally:
            sys.modules.update(saved_mods)

    def test_no_vapid_key_returns_zero(self):
        from backend.email_utils import _send_web_push_to_subscriptions
        sub = MagicMock(id=1, endpoint="https://e", p256dh_key="k", auth_key="a")
        with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": ""}, clear=False):
            count = _send_web_push_to_subscriptions([sub], "T", "B")
        assert count == 0

    def test_successful_delivery_counts_sent(self):
        """Happy path: 2 subs, both succeed → count == 2."""
        from backend import email_utils
        subs = [
            MagicMock(id=1, endpoint="https://e1", p256dh_key="k", auth_key="a"),
            MagicMock(id=2, endpoint="https://e2", p256dh_key="k", auth_key="a"),
        ]
        fake_pywebpush = MagicMock()
        fake_pywebpush.webpush = MagicMock(return_value=None)
        fake_pywebpush.WebPushException = type("WebPushException", (Exception,), {})
        with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": "fake-key",
                                     "VAPID_SUBJECT": "mailto:a@b"}), \
             patch.dict(sys.modules, {"pywebpush": fake_pywebpush}):
            count = email_utils._send_web_push_to_subscriptions(subs, "T", "B")
        assert count == 2

    def test_410_gone_subscription_is_pruned(self):
        """status_code=410 means the browser deleted the subscription. Helper
        must batch-delete the row(s) in a fresh DB session."""
        from backend import email_utils
        sub_alive = MagicMock(id=1, endpoint="https://e1", p256dh_key="k", auth_key="a")
        sub_dead = MagicMock(id=2, endpoint="https://e2", p256dh_key="k", auth_key="a")

        WebPushExc = type("WebPushException", (Exception,), {})
        gone_exc = WebPushExc("gone")
        gone_exc.response = MagicMock(status_code=410)

        def _push(subscription_info, **kwargs):
            if subscription_info["endpoint"] == "https://e2":
                raise gone_exc
            return None

        fake_pywebpush = MagicMock()
        fake_pywebpush.webpush = MagicMock(side_effect=_push)
        fake_pywebpush.WebPushException = WebPushExc

        fake_db = MagicMock()
        with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": "k"}), \
             patch.dict(sys.modules, {"pywebpush": fake_pywebpush}), \
             patch("backend.database.SessionLocal", return_value=fake_db):
            count = email_utils._send_web_push_to_subscriptions(
                [sub_alive, sub_dead], "T", "B"
            )
        assert count == 1
        # Prune query went out.
        assert fake_db.commit.called

    def test_generic_exception_is_swallowed(self):
        """Non-WebPushException errors hit the bare except at 348-349 — must
        not raise, just log + skip. Returns 0 because no subs succeeded."""
        from backend import email_utils
        sub = MagicMock(id=1, endpoint="https://e", p256dh_key="k", auth_key="a")
        fake_pywebpush = MagicMock()
        fake_pywebpush.webpush = MagicMock(side_effect=RuntimeError("boom"))
        fake_pywebpush.WebPushException = type("WebPushException", (Exception,), {})
        with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": "k"}), \
             patch.dict(sys.modules, {"pywebpush": fake_pywebpush}):
            count = email_utils._send_web_push_to_subscriptions([sub], "T", "B")
        assert count == 0


# ============================================================================
# send_web_push_to_user / send_web_push_broadcast
# ============================================================================
class TestSendWebPushRouting:
    def test_send_web_push_to_user_zero_id_returns_zero(self):
        from backend.email_utils import send_web_push_to_user
        assert send_web_push_to_user(0, "T", "B") == 0

    def test_send_web_push_to_user_db_exception_returns_zero(self):
        from backend.email_utils import send_web_push_to_user
        with patch("backend.database.SessionLocal", side_effect=RuntimeError("db")):
            assert send_web_push_to_user(1, "T", "B") == 0

    def test_send_web_push_broadcast_db_exception_returns_zero(self):
        from backend.email_utils import send_web_push_broadcast
        with patch("backend.database.SessionLocal", side_effect=RuntimeError("db")):
            assert send_web_push_broadcast("T", "B") == 0

    def test_send_web_push_to_user_delegates_to_internal_helper(self):
        from backend import email_utils
        fake_subs = [MagicMock()]
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.all.return_value = fake_subs
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "_send_web_push_to_subscriptions",
                          return_value=1) as mock_inner:
            count = email_utils.send_web_push_to_user(1, "T", "B")
        assert count == 1
        # The subs list and title/body are passed through.
        called_subs = mock_inner.call_args[0][0]
        assert called_subs is fake_subs


# ============================================================================
# broadcast_notification — fan-out + fail-soft
# ============================================================================
class TestBroadcastNotification:
    def test_db_exception_does_not_raise(self):
        """DB exception path 434-435 — broadcast still returns and the push
        fan-out fires (push happens regardless of DB)."""
        from backend import email_utils
        with patch("backend.database.SessionLocal", side_effect=RuntimeError("db")), \
             patch.object(email_utils, "send_web_push_broadcast", return_value=0):
            inserted = email_utils.broadcast_notification("system", "T", "B")
        assert inserted == 0  # nothing was written

    def test_push_fanout_exception_swallowed(self):
        from backend import email_utils
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.all.return_value = [(1,), (2,)]
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "send_web_push_broadcast",
                          side_effect=RuntimeError("push")):
            inserted = email_utils.broadcast_notification("system", "T", "B",
                                                          data={"ticker": "AAPL"})
        # 2 rows still got inserted; push exception didn't poison the count.
        assert inserted == 2

    def test_url_defaults_to_notifications_without_ticker(self):
        from backend import email_utils
        fake_db = MagicMock()
        fake_db.query.return_value.filter.return_value.all.return_value = []
        with patch("backend.database.SessionLocal", return_value=fake_db), \
             patch.object(email_utils, "send_web_push_broadcast", return_value=0) as mock_push:
            email_utils.broadcast_notification("system", "T", "B")
        assert mock_push.call_args.kwargs["data"]["url"] == "/notifications"


# ============================================================================
# send_trade_webhook — tag branches
# ============================================================================
class TestSendTradeWebhook:
    def test_buy_uses_moneybag_tags(self):
        from backend import email_utils
        with patch.object(email_utils, "create_notification") as mock_create, \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_trade_webhook(
                "AAPL", "BUY", 10.0, 150.0, "high score", user_id=1,
            )
        tags = mock_wh.call_args.kwargs["tags"]
        assert "moneybag" in tags

    def test_sell_winner_uses_white_check_mark(self):
        from backend import email_utils
        with patch.object(email_utils, "create_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh, \
             patch.object(email_utils, "get_user_webhook_url",
                          return_value="https://ntfy.sh/x"):
            email_utils.send_trade_webhook(
                "AAPL", "SELL", 10.0, 175.0, "trail stop", gain_pct=12.5, user_id=1,
            )
        tags = mock_wh.call_args.kwargs["tags"]
        assert "white_check_mark" in tags

    def test_sell_loser_uses_chart_with_downwards_trend(self):
        from backend import email_utils
        with patch.object(email_utils, "create_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh, \
             patch.object(email_utils, "get_user_webhook_url", return_value=""):
            email_utils.send_trade_webhook(
                "AAPL", "SELL", 10.0, 90.0, "stop", gain_pct=-7.0, user_id=1,
            )
        tags = mock_wh.call_args.kwargs["tags"]
        assert "chart_with_downwards_trend" in tags

    def test_user_id_none_skips_per_user_url_lookup(self):
        """user_id=None preserves legacy behavior — falls back to global env
        webhook (via url=None)."""
        from backend import email_utils
        with patch.object(email_utils, "create_notification") as mock_create, \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh, \
             patch.object(email_utils, "get_user_webhook_url") as mock_lookup:
            email_utils.send_trade_webhook("AAPL", "BUY", 1.0, 1.0, "r", user_id=None)
        # No DB lookup when user_id is None.
        mock_lookup.assert_not_called()
        # url kwarg was None (legacy global-fallback path).
        assert mock_wh.call_args.kwargs["url"] is None


# ============================================================================
# send_stop_loss_webhook
# ============================================================================
class TestSendStopLossWebhook:
    def test_creates_notification_and_sends_urgent_webhook(self):
        from backend import email_utils
        with patch.object(email_utils, "create_notification") as mock_create, \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh, \
             patch.object(email_utils, "get_user_webhook_url", return_value=""):
            email_utils.send_stop_loss_webhook(
                "AAPL", 10.0, 95.0, "TRAILING STOP", -8.0, user_id=1,
            )
        # Notification persisted with kind=stop_loss + urgent priority.
        kw = mock_create.call_args.kwargs
        assert kw["kind"] == "stop_loss"
        assert kw["priority"] == "urgent"
        # Webhook tagged rotating_light.
        tags = mock_wh.call_args.kwargs["tags"]
        assert "rotating_light" in tags


# ============================================================================
# send_risk_alert_webhook — every alert_type
# ============================================================================
class TestSendRiskAlertWebhook:
    @pytest.mark.parametrize("alert_type,expected_title,expected_tag", [
        ("heat", "Portfolio Heat Warning", "fire"),
        ("sector_concentration", "Sector Concentration Alert", "pie"),
        ("position_size", "Position Size Alert", "heavy_dollar_sign"),
        ("drawdown", "Drawdown Warning", "rotating_light"),
        ("unknown_kind", "Risk Alert: unknown_kind", "warning"),
    ])
    def test_alert_type_dispatches_correct_title_and_tags(
        self, alert_type, expected_title, expected_tag
    ):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            ok = email_utils.send_risk_alert_webhook(alert_type, "details here")
        assert ok is True
        args, kwargs = mock_wh.call_args
        assert args[0] == expected_title
        assert expected_tag in kwargs["tags"]


# ============================================================================
# send_coiled_spring_alert_webhook — body shape
# ============================================================================
class TestSendCoiledSpringAlertWebhook:
    def test_renders_stock_fields_in_payload(self):
        from backend import email_utils
        stock = MagicMock()
        stock.ticker = "NVDA"
        stock.name = "NVIDIA Corp"
        stock.current_price = 500.0
        stock.canslim_score = 92.0
        cs_result = {
            "days_to_earnings": 7, "base_type": "cup-with-handle",
            "weeks_in_base": 9, "beat_streak": 4,
        }
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            ok = email_utils.send_coiled_spring_alert_webhook(stock, cs_result)
        assert ok is True
        # Webhook called with title containing ticker + body containing the
        # ingredients.
        args, kwargs = mock_wh.call_args
        assert "NVDA" in args[0]
        assert "cup-with-handle" in args[1]
        assert "4 beat streak" in args[1]
        assert kwargs["data"]["ticker"] == "NVDA"
        assert "cyclone" in kwargs["tags"]


# ============================================================================
# send_morning_briefing_push — full render path
# ============================================================================
class TestSendMorningBriefingPush:
    def test_rendered_payload_contains_value_regime_heat(self):
        from backend import email_utils
        briefing = {
            "portfolio": {"total_value": 27500.0, "total_return_pct": 10.0, "cash": 1500.0},
            "market_regime": {"regime": "bullish"},
            "positions": [
                {"ticker": "AAPL", "gain_pct": 5.0},
                {"ticker": "MSFT", "gain_pct": -2.0},
            ],
            "top_candidates": [{"ticker": "NVDA", "score": 90.0}],
            "portfolio_heat": 18.0,
            "near_stop_positions": [{"ticker": "TSLA", "pct_to_stop": 2}],
            "earnings_warnings": [{"ticker": "META", "days": 3}],
        }
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            ok = email_utils.send_morning_briefing_push(briefing)
        assert ok is True
        title, message = mock_wh.call_args.args[0], mock_wh.call_args.args[1]
        assert "27,500" in title
        assert "+10.0%" in title
        assert "BULLISH" in message
        assert "Heat: 18%" in message
        assert "NEAR STOP: TSLA" in message
        assert "EARNINGS: META 3d" in message
        # Top positions line includes both tickers.
        assert "Top:" in message and "AAPL" in message and "MSFT" in message
        assert "Buy:" in message and "NVDA" in message

    def test_bearish_regime_uses_red_circle_tag(self):
        from backend import email_utils
        briefing = {
            "portfolio": {"total_value": 1000.0},
            "market_regime": {"regime": "bearish"},
        }
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_morning_briefing_push(briefing)
        assert "red_circle" in mock_wh.call_args.kwargs["tags"]


# ============================================================================
# send_scan_completion_push
# ============================================================================
class TestSendScanCompletionPush:
    def test_no_trades_omits_trade_line(self):
        from backend import email_utils
        with patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_scan_completion_push(2000, 2050, 90.5)
        msg = mock_wh.call_args.args[1]
        assert "90s" in msg
        assert "Trades" not in msg
        # Bell tag absent when no trades.
        assert "bell" not in mock_wh.call_args.kwargs["tags"]

    def test_with_buys_and_sells_includes_count_and_bell_tag(self):
        from backend import email_utils
        with patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_scan_completion_push(
                10, 10, 5.0, buys=[{"t": "A"}, {"t": "B"}], sells=[{"t": "C"}],
            )
        title, msg = mock_wh.call_args.args[0], mock_wh.call_args.args[1]
        assert "10/10" in title
        assert "2 buys" in msg
        assert "1 sell" in msg
        assert "bell" in mock_wh.call_args.kwargs["tags"]


# ============================================================================
# send_spy_gate_change_push
# ============================================================================
class TestSendSpyGateChangePush:
    def test_bullish_flip_uses_green_circle_and_says_enabled(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_spy_gate_change_push("bullish", 450.0, 440.0)
        title, msg = mock_wh.call_args.args[0], mock_wh.call_args.args[1]
        assert "BULLISH" in title
        assert "ABOVE" in msg
        assert "ENABLED" in msg
        assert "green_circle" in mock_wh.call_args.kwargs["tags"]

    def test_bearish_flip_uses_red_circle_and_says_blocked(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_spy_gate_change_push("bearish", 430.0, 440.0)
        msg = mock_wh.call_args.args[1]
        assert "BELOW" in msg
        assert "BLOCKED" in msg
        assert "red_circle" in mock_wh.call_args.kwargs["tags"]


# ============================================================================
# send_score_crash_warning_push
# ============================================================================
class TestSendScoreCrashWarningPush:
    def test_message_contains_remaining_scan_count(self):
        from backend import email_utils
        with patch.object(email_utils, "create_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh, \
             patch.object(email_utils, "get_user_webhook_url", return_value=""):
            email_utils.send_score_crash_warning_push(
                "AAPL", purchase_score=85, current_score=60,
                gain_pct=4.0, consecutive_low=2, consecutive_required=3,
                user_id=1,
            )
        msg = mock_wh.call_args.args[1]
        assert "AAPL" in msg
        assert "85" in msg and "60" in msg
        assert "2/3" in msg
        assert "1 more before auto-sell" in msg


# ============================================================================
# send_bear_base_update_push & send_market_turn_ready_push & bear_market_report
# ============================================================================
class TestBearBaseAndMarketTurn:
    def test_bear_base_update_renders_top_candidates(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_bear_base_update_push(
                total=42,
                top_candidates=[
                    {"ticker": "AAPL", "readiness_score": 88},
                    {"ticker": "MSFT", "readiness_score": 82},
                ],
            )
        title, msg = mock_wh.call_args.args[0], mock_wh.call_args.args[1]
        assert "42 candidates" in title
        assert "AAPL(88)" in msg

    def test_market_turn_ready_uses_urgent_priority_and_markdown(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_market_turn_ready_push(
                candidates=[{"ticker": "AAPL", "readiness_score": 90,
                             "base_type": "cup", "weeks_in_base": 8}],
                spy_price=455.0, spy_ma50=440.0,
            )
        kwargs = mock_wh.call_args.kwargs
        assert kwargs["priority"] == "urgent"
        assert kwargs["markdown"] is True

    def test_bear_market_report_with_no_data_renders_fallback_message(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_bear_market_report_push({"bases_forming": 0})
        msg = mock_wh.call_args.args[1]
        assert "No notable changes" in msg

    def test_bear_market_report_renders_top_ready_and_improving(self):
        from backend import email_utils
        with patch.object(email_utils, "broadcast_notification"), \
             patch.object(email_utils, "send_webhook_notification",
                          return_value=True) as mock_wh:
            email_utils.send_bear_market_report_push({
                "bases_forming": 5,
                "top_ready": [{"ticker": "AAPL", "readiness_score": 90}],
                "improving_groups": [{"industry": "Semiconductors"}],
            })
        msg = mock_wh.call_args.args[1]
        assert "Ready: AAPL(90)" in msg
        assert "Improving: Semiconductors" in msg


# ============================================================================
# send_morning_briefing_email — alert subject suffix + earnings_this_week
# ============================================================================
class TestSendMorningBriefingEmail:
    def test_alert_suffix_appears_when_near_stop_or_earnings_present(self):
        """Lines 869, 871 — when near_stop_positions or earnings_warnings is
        non-empty, subject gets '[1 near stop, 1 earnings]' suffix."""
        from backend import email_utils
        briefing = {
            "portfolio": {"total_value": 25000.0, "total_return_pct": 0.0, "cash": 1000.0},
            "market_regime": {"regime": "bullish"},
            "near_stop_positions": [{"ticker": "AAPL", "pct_to_stop": 2}],
            "earnings_warnings": [
                {"ticker": "META", "days": 5, "date": "2026-05-12"},
            ],
            "positions": [], "top_candidates": [], "earnings_this_week": [],
            "market_timing": {"state": "GREEN", "can_buy": True},
            "portfolio_heat": 12.0,
        }
        with patch.object(email_utils, "send_email", return_value=True) as mock_send:
            ok = email_utils.send_morning_briefing_email(briefing)
        assert ok is True
        subject = mock_send.call_args.args[0]
        assert "1 near stop" in subject
        assert "1 earnings" in subject

    def test_earnings_this_week_section_renders_table(self):
        """Lines 904-905, 914 — earnings_this_week populated → table rows
        include ticker, days, beats, gain_pct."""
        from backend import email_utils
        briefing = {
            "portfolio": {"total_value": 25000.0, "total_return_pct": 0.0, "cash": 1000.0},
            "market_regime": {"regime": "bullish"},
            "positions": [], "top_candidates": [],
            "near_stop_positions": [], "earnings_warnings": [],
            "earnings_this_week": [
                {"ticker": "AAPL", "days_to_earnings": 3, "beat_streak": 4, "gain_pct": 8.5},
            ],
            "market_timing": {"state": "GREEN", "can_buy": True},
            "portfolio_heat": 0.0,
        }
        with patch.object(email_utils, "send_email", return_value=True) as mock_send:
            email_utils.send_morning_briefing_email(briefing)
        html_body = mock_send.call_args.args[1]
        assert "Earnings This Week" in html_body
        assert "AAPL" in html_body
        assert "4 beats" in html_body
        assert "+8.5%" in html_body

    def test_positions_section_renders_only_when_positions_present(self):
        from backend import email_utils
        # No positions → no "Current Positions" header.
        briefing_empty = {
            "portfolio": {"total_value": 25000.0, "total_return_pct": 0.0, "cash": 0.0},
            "market_regime": {"regime": "bullish"},
            "positions": [], "top_candidates": [],
            "near_stop_positions": [], "earnings_warnings": [],
            "earnings_this_week": [],
            "market_timing": {}, "portfolio_heat": 0.0,
        }
        with patch.object(email_utils, "send_email", return_value=True) as mock_send:
            email_utils.send_morning_briefing_email(briefing_empty)
        html_body = mock_send.call_args.args[1]
        assert "Current Positions" not in html_body
        # Empty buy candidates → fallback "No buy candidates today" copy.
        assert "No buy candidates today" in html_body
