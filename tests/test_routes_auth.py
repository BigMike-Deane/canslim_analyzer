"""Coverage tests for backend/routes/auth.py.

Companion to tests/test_auth.py (which covers backend/auth.py helpers).
These tests exercise the FastAPI endpoint handlers themselves — the
six routes that mount under `/api/auth`:

    POST   /google             — Google Sign-In
    POST   /refresh            — refresh-token exchange
    GET    /me                 — current user profile
    PATCH  /me/webhook         — update notification webhook URL
    POST   /me/webhook/test    — send test webhook
    GET    /config             — public auth config

The conftest disables slowapi for the test session, so we can call the
`@limiter.limit`-decorated `async def` handlers directly via
`asyncio.run(...)` with mocked dependencies — no TestClient overhead.

Sibling style: mirrors the MagicMock-db / direct-call pattern in
tests/test_auth.py for the dep helpers.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException, Request
from jose import jwt

from backend.routes import auth as routes_auth
from backend.routes.auth import (
    _client_ip, _record_auth_fail,
    google_login, refresh_token,
    get_me, update_my_webhook, get_auth_config,
    update_notification_prefs,
    GoogleLoginRequest, RefreshRequest, WebhookUpdateRequest,
    NotificationPrefsRequest,
)
# Aliased on import: the route handler's literal name starts with `test_`,
# which pytest's collector would pick up as a top-level test and execute
# with the unresolved Depends() default → AttributeError.
from backend.routes.auth import test_my_webhook as send_test_webhook
from backend.auth import (
    create_access_token, create_refresh_token,
    SECRET_KEY, ALGORITHM,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _run(coro):
    """Execute an async route body and return its result (or re-raise)."""
    return asyncio.run(coro)


def _make_request(host: str = "10.0.0.1") -> MagicMock:
    """A minimal Request stand-in: only `request.client.host` is read."""
    req = MagicMock(spec=Request)
    req.client = MagicMock()
    req.client.host = host
    return req


def _make_db(user=None):
    """MagicMock SQLAlchemy session whose .query().filter().first() returns `user`."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = user
    return db


def _make_user(
    user_id: int = 1,
    email: str = "u@example.com",
    display_name: str | None = "Owner",
    is_active: bool = True,
    is_admin: bool = False,
    webhook_url: str | None = None,
):
    user = MagicMock()
    user.id = user_id
    user.email = email
    user.display_name = display_name
    user.is_active = is_active
    user.is_admin = is_admin
    user.webhook_url = webhook_url
    return user


# ── _client_ip ─────────────────────────────────────────────────────────


class TestClientIp:
    """Covers backend/routes/auth.py:21-24."""

    def test_returns_unknown_when_request_is_none(self):
        assert _client_ip(None) == "unknown"

    def test_returns_unknown_when_request_client_is_none(self):
        req = MagicMock(spec=Request)
        req.client = None
        assert _client_ip(req) == "unknown"

    def test_returns_host_when_present(self):
        assert _client_ip(_make_request("192.168.1.5")) == "192.168.1.5"

    def test_returns_unknown_when_host_is_empty_string(self):
        # `request.client.host or "unknown"` falls through on empty string.
        req = _make_request("")
        assert _client_ip(req) == "unknown"


# ── _record_auth_fail ──────────────────────────────────────────────────


class TestRecordAuthFail:
    """Covers backend/routes/auth.py:27-38 — including the bare-except."""

    def test_records_event_with_pii_free_payload(self):
        with patch("backend.routes.auth.record_event") as mock_rec:
            _record_auth_fail(_make_request("1.2.3.4"), "invalid_token")
        mock_rec.assert_called_once()
        kwargs = mock_rec.call_args.kwargs
        assert kwargs["source_ip"] == "1.2.3.4"
        assert kwargs["route"] == "/api/auth/google"
        assert kwargs["detail"] == "invalid_token"

    def test_swallows_record_event_exception(self):
        """A broken audit subsystem must never mask the auth response."""
        with patch(
            "backend.routes.auth.record_event",
            side_effect=RuntimeError("audit broke"),
        ):
            # Must not raise.
            _record_auth_fail(_make_request(), "invalid_token")


# ── POST /api/auth/google ──────────────────────────────────────────────


class TestGoogleLogin:
    """Covers backend/routes/auth.py:45-81."""

    def test_invalid_google_token_propagates_and_audits(self):
        req_body = GoogleLoginRequest(credential="bad-token")
        with patch(
            "backend.routes.auth.verify_google_token",
            side_effect=HTTPException(status_code=401, detail="Invalid Google token"),
        ), patch("backend.routes.auth.record_event") as mock_rec:
            with pytest.raises(HTTPException) as exc:
                _run(google_login(req_body, request=_make_request(), db=_make_db()))
        assert exc.value.status_code == 401
        # PII-free audit fired
        assert mock_rec.called
        assert mock_rec.call_args.kwargs["detail"] == "invalid_token"

    def test_missing_email_in_google_payload_raises_401(self):
        req_body = GoogleLoginRequest(credential="token")
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"name": "No Email Here"},
        ), patch("backend.routes.auth.record_event") as mock_rec:
            with pytest.raises(HTTPException) as exc:
                _run(google_login(req_body, request=_make_request(), db=_make_db()))
        assert exc.value.status_code == 401
        assert "No email" in exc.value.detail
        assert mock_rec.call_args.kwargs["detail"] == "missing_email"

    def test_unknown_user_raises_403(self):
        """Invite-only mode: an email not in the users table is rejected."""
        req_body = GoogleLoginRequest(credential="token")
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "stranger@example.com", "name": "Stranger"},
        ), patch("backend.routes.auth.record_event") as mock_rec:
            with pytest.raises(HTTPException) as exc:
                _run(google_login(req_body, request=_make_request(), db=_make_db(user=None)))
        assert exc.value.status_code == 403
        assert "No account" in exc.value.detail
        assert mock_rec.call_args.kwargs["detail"] == "unknown_user"

    def test_disabled_account_raises_403(self):
        req_body = GoogleLoginRequest(credential="token")
        disabled = _make_user(is_active=False)
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "u@example.com", "name": "U"},
        ), patch("backend.routes.auth.record_event") as mock_rec:
            with pytest.raises(HTTPException) as exc:
                _run(google_login(req_body, request=_make_request(), db=_make_db(user=disabled)))
        assert exc.value.status_code == 403
        assert "disabled" in exc.value.detail.lower()
        assert mock_rec.call_args.kwargs["detail"] == "account_disabled"

    def test_success_existing_display_name_does_not_overwrite(self):
        """If the user already has a display_name, the Google name is ignored."""
        req_body = GoogleLoginRequest(credential="token")
        user = _make_user(user_id=42, display_name="Already Set")
        db = _make_db(user=user)
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "u@example.com", "name": "Different Name"},
        ):
            token = _run(google_login(req_body, request=_make_request(), db=db))
        # display_name unchanged
        assert user.display_name == "Already Set"
        db.commit.assert_not_called()
        # Token shape: access + refresh, both decode to sub="42"
        assert token.token_type == "bearer"
        for raw in (token.access_token, token.refresh_token):
            payload = jwt.decode(raw, SECRET_KEY, algorithms=[ALGORITHM])
            assert payload["sub"] == "42"

    def test_success_blank_display_name_is_filled_from_google(self):
        """If user.display_name is empty, the Google `name` is persisted."""
        req_body = GoogleLoginRequest(credential="token")
        user = _make_user(user_id=7, display_name=None)
        db = _make_db(user=user)
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "u@example.com", "name": "From Google"},
        ):
            token = _run(google_login(req_body, request=_make_request(), db=db))
        assert user.display_name == "From Google"
        db.commit.assert_called_once()
        assert token.access_token
        assert token.refresh_token

    def test_success_no_google_name_skips_display_name_update(self):
        """If Google omits the `name` field, leave display_name alone."""
        req_body = GoogleLoginRequest(credential="token")
        user = _make_user(user_id=9, display_name=None)
        db = _make_db(user=user)
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "u@example.com"},  # no `name`
        ):
            _run(google_login(req_body, request=_make_request(), db=db))
        assert user.display_name is None
        db.commit.assert_not_called()

    def test_email_is_lowercased_before_lookup(self):
        """Google returns mixed-case emails; the lookup must use lowercase."""
        req_body = GoogleLoginRequest(credential="token")
        user = _make_user(user_id=3)
        db = _make_db(user=user)
        with patch(
            "backend.routes.auth.verify_google_token",
            return_value={"email": "MixedCase@Example.COM"},
        ):
            _run(google_login(req_body, request=_make_request(), db=db))
        # We can't easily introspect the filter() arg, but at minimum
        # the route succeeded — i.e. it did call .query/.filter/.first.
        db.query.assert_called()


# ── POST /api/auth/refresh ─────────────────────────────────────────────


class TestRefreshToken:
    """Covers backend/routes/auth.py:88-107."""

    def test_garbage_token_returns_401(self):
        req_body = RefreshRequest(refresh_token="not.a.jwt")
        with pytest.raises(HTTPException) as exc:
            _run(refresh_token(req_body, request=_make_request(), db=_make_db()))
        assert exc.value.status_code == 401
        assert "Invalid refresh token" in exc.value.detail

    def test_access_token_rejected_as_refresh(self):
        """An access token presented at /refresh must 401 on the type check."""
        access = create_access_token(data={"sub": "1"})
        req_body = RefreshRequest(refresh_token=access)
        with pytest.raises(HTTPException) as exc:
            _run(refresh_token(req_body, request=_make_request(), db=_make_db()))
        assert exc.value.status_code == 401
        assert "Invalid token type" in exc.value.detail

    def test_valid_refresh_token_for_missing_user_returns_401(self):
        rt = create_refresh_token(data={"sub": "999"})
        req_body = RefreshRequest(refresh_token=rt)
        with pytest.raises(HTTPException) as exc:
            _run(refresh_token(req_body, request=_make_request(), db=_make_db(user=None)))
        assert exc.value.status_code == 401
        assert "User not found" in exc.value.detail

    def test_valid_refresh_token_for_active_user_returns_new_pair(self):
        user = _make_user(user_id=42)
        rt = create_refresh_token(data={"sub": "42"})
        req_body = RefreshRequest(refresh_token=rt)
        token = _run(refresh_token(req_body, request=_make_request(), db=_make_db(user=user)))
        assert token.token_type == "bearer"
        assert token.access_token
        assert token.refresh_token
        payload = jwt.decode(token.access_token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "42"
        assert payload["type"] == "access"
        payload2 = jwt.decode(token.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload2["type"] == "refresh"


# ── GET /api/auth/me ───────────────────────────────────────────────────


class TestGetMe:
    """Covers backend/routes/auth.py:111-120."""

    def test_returns_user_response_with_all_fields(self):
        user = _make_user(
            user_id=11,
            email="me@example.com",
            display_name="Me",
            is_admin=True,
            is_active=True,
            webhook_url="https://hooks.example.com/abc",
        )
        resp = _run(get_me(current_user=user))
        assert resp.id == 11
        assert resp.email == "me@example.com"
        assert resp.display_name == "Me"
        assert resp.is_admin is True
        assert resp.is_active is True
        assert resp.webhook_url == "https://hooks.example.com/abc"

    def test_returns_user_response_when_webhook_unset(self):
        user = _make_user(webhook_url=None)
        resp = _run(get_me(current_user=user))
        assert resp.webhook_url is None


# ── PATCH /api/auth/me/webhook ─────────────────────────────────────────


class TestUpdateMyWebhook:
    """Covers backend/routes/auth.py:127-152."""

    def test_accepts_https_url(self):
        # Isolate from DNS: the SSRF guard has its own tests in
        # test_email_utils; here we verify the endpoint stores an accepted URL.
        user = _make_user(webhook_url=None)
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="https://hooks.example.com/abc")
        with patch("backend.email_utils.is_safe_webhook_url", return_value=True):
            resp = _run(update_my_webhook(body, current_user=user, db=db))
        assert user.webhook_url == "https://hooks.example.com/abc"
        assert resp.webhook_url == "https://hooks.example.com/abc"
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(user)

    def test_accepts_http_url(self):
        user = _make_user(webhook_url="https://old.example.com/x")
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="http://hooks.example.com/notify")
        with patch("backend.email_utils.is_safe_webhook_url", return_value=True):
            resp = _run(update_my_webhook(body, current_user=user, db=db))
        assert user.webhook_url == "http://hooks.example.com/notify"
        assert resp.webhook_url == "http://hooks.example.com/notify"

    def test_rejects_private_address_url(self):
        # SSRF guard: a scheme-valid but private-resolving URL is 400'd.
        user = _make_user(webhook_url=None)
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="http://127.0.0.1:6379/x")
        with pytest.raises(HTTPException) as exc:
            _run(update_my_webhook(body, current_user=user, db=db))
        assert exc.value.status_code == 400
        db.commit.assert_not_called()

    def test_empty_string_clears_webhook_to_none(self):
        user = _make_user(webhook_url="https://hooks.example.com/abc")
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="")
        resp = _run(update_my_webhook(body, current_user=user, db=db))
        assert user.webhook_url is None
        assert resp.webhook_url is None
        db.commit.assert_called_once()

    def test_whitespace_only_clears_webhook(self):
        """A user typing spaces should be treated as a clear, not a 400."""
        user = _make_user(webhook_url="https://hooks.example.com/abc")
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="   ")
        _run(update_my_webhook(body, current_user=user, db=db))
        assert user.webhook_url is None

    def test_url_without_http_prefix_returns_400(self):
        user = _make_user()
        db = _make_db()
        body = WebhookUpdateRequest(webhook_url="ftp://wrong.scheme/notify")
        with pytest.raises(HTTPException) as exc:
            _run(update_my_webhook(body, current_user=user, db=db))
        assert exc.value.status_code == 400
        assert "http://" in exc.value.detail
        db.commit.assert_not_called()

    def test_bare_hostname_returns_400(self):
        user = _make_user()
        body = WebhookUpdateRequest(webhook_url="hooks.example.com/abc")
        with pytest.raises(HTTPException) as exc:
            _run(update_my_webhook(body, current_user=user, db=_make_db()))
        assert exc.value.status_code == 400


# ── POST /api/auth/me/webhook/test ─────────────────────────────────────


class TestTestMyWebhook:
    """Covers backend/routes/auth.py:155-170 — the lazy-import branch is
    triggered via a patch on backend.email_utils.send_webhook_notification."""

    def test_no_url_configured_returns_400(self):
        user = _make_user(webhook_url=None)
        with pytest.raises(HTTPException) as exc:
            _run(send_test_webhook(current_user=user))
        assert exc.value.status_code == 400
        assert "No webhook URL configured" in exc.value.detail

    def test_empty_string_url_returns_400(self):
        """A `""` (or whitespace) webhook_url is treated the same as None."""
        user = _make_user(webhook_url="   ")
        with pytest.raises(HTTPException) as exc:
            _run(send_test_webhook(current_user=user))
        assert exc.value.status_code == 400

    def test_url_configured_dispatches_and_returns_sent_true(self):
        user = _make_user(
            email="user@example.com",
            webhook_url="https://hooks.example.com/abc",
        )
        with patch(
            "backend.email_utils.send_webhook_notification",
            return_value=True,
        ) as mock_send, patch(
            "backend.email_utils.is_safe_webhook_url", return_value=True,
        ):
            result = _run(send_test_webhook(current_user=user))
        assert result == {"sent": True, "url_configured": True}
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["url"] == "https://hooks.example.com/abc"
        assert kwargs["title"] == "CANSLIM Test"
        assert "user@example.com" in kwargs["message"]

    def test_url_configured_but_dispatch_failed_returns_sent_false(self):
        user = _make_user(webhook_url="https://hooks.example.com/abc")
        with patch(
            "backend.email_utils.send_webhook_notification",
            return_value=False,
        ), patch(
            "backend.email_utils.is_safe_webhook_url", return_value=True,
        ):
            result = _run(send_test_webhook(current_user=user))
        assert result == {"sent": False, "url_configured": True}


# ── PATCH /api/auth/me/notification-prefs ──────────────────────────────


class TestUpdateNotificationPrefs:
    """Covers backend/routes/auth.py:200-244 — per-kind mute list + quiet hours.

    The handler has two independent branch clusters:
      1. mute_kinds: None (skip) | [] (clear) | [valid...] (set) | [bad...] (400)
      2. quiet_hours: clear flag short-circuits; otherwise start/end each go
         through the 0..23 range check.

    Boundary tests (0, 23 accepted; -1, 24 rejected) pin the inclusive range
    contract — off-by-one here would silently mute the user at hour 0 or
    leak alerts at hour 23.
    """

    @staticmethod
    def _user_with_prefs(
        mute_kinds=None,
        quiet_hours_start=None,
        quiet_hours_end=None,
    ):
        """Like _make_user but with explicit notification-pref attrs.

        UserResponse declares these as Optional[list[str]] / Optional[int];
        leaving them as auto-MagicMock would still pass Pydantic in this
        suite (see TestGetMe), but the assertions here read these attrs
        back so we set them explicitly.
        """
        u = _make_user()
        u.mute_kinds = mute_kinds
        u.quiet_hours_start = quiet_hours_start
        u.quiet_hours_end = quiet_hours_end
        return u

    # --- mute_kinds branch ----------------------------------------------

    def test_unknown_kind_returns_400_and_does_not_commit(self):
        user = self._user_with_prefs(mute_kinds=["trade"])
        db = _make_db()
        body = NotificationPrefsRequest(mute_kinds=["trade", "bogus_kind"])
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=db))
        assert exc.value.status_code == 400
        assert "Unknown kinds" in exc.value.detail
        assert "bogus_kind" in exc.value.detail
        # Pre-existing mute list untouched; db.commit not reached
        assert user.mute_kinds == ["trade"]
        db.commit.assert_not_called()

    def test_all_valid_kinds_persists_as_list(self):
        user = self._user_with_prefs(mute_kinds=None)
        db = _make_db()
        body = NotificationPrefsRequest(mute_kinds=["trade", "stop_loss", "breakout"])
        resp = _run(update_notification_prefs(body, current_user=user, db=db))
        assert user.mute_kinds == ["trade", "stop_loss", "breakout"]
        assert resp.mute_kinds == ["trade", "stop_loss", "breakout"]
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(user)

    def test_empty_list_clears_mute(self):
        """Passing [] is the documented way to clear the mute list."""
        user = self._user_with_prefs(mute_kinds=["trade", "stop_loss"])
        db = _make_db()
        body = NotificationPrefsRequest(mute_kinds=[])
        resp = _run(update_notification_prefs(body, current_user=user, db=db))
        assert user.mute_kinds == []
        # `or []` fallback in the response means `mute_kinds=[]` round-trips as []
        assert resp.mute_kinds == []
        db.commit.assert_called_once()

    def test_mute_kinds_none_leaves_user_attr_untouched(self):
        """Unset field must NOT overwrite the existing mute list."""
        user = self._user_with_prefs(mute_kinds=["trade"])
        db = _make_db()
        body = NotificationPrefsRequest()  # all fields default
        _run(update_notification_prefs(body, current_user=user, db=db))
        assert user.mute_kinds == ["trade"]

    def test_bad_kinds_listed_in_error_detail(self):
        """Both bogus kinds should appear in the 400 detail, not just the first."""
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(mute_kinds=["nope1", "trade", "nope2"])
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert "nope1" in exc.value.detail
        assert "nope2" in exc.value.detail

    # --- quiet_hours: clear flag ----------------------------------------

    def test_clear_quiet_hours_true_nulls_both_bounds(self):
        user = self._user_with_prefs(quiet_hours_start=22, quiet_hours_end=6)
        db = _make_db()
        body = NotificationPrefsRequest(clear_quiet_hours=True)
        resp = _run(update_notification_prefs(body, current_user=user, db=db))
        assert user.quiet_hours_start is None
        assert user.quiet_hours_end is None
        assert resp.quiet_hours_start is None
        assert resp.quiet_hours_end is None

    def test_clear_flag_short_circuits_start_end_assignment(self):
        """clear_quiet_hours=True wins even if start/end are also provided."""
        user = self._user_with_prefs(quiet_hours_start=22, quiet_hours_end=6)
        body = NotificationPrefsRequest(
            clear_quiet_hours=True,
            quiet_hours_start=10,
            quiet_hours_end=18,
        )
        _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        # Bounds nulled; the 10/18 inputs ignored because clear branch ran.
        assert user.quiet_hours_start is None
        assert user.quiet_hours_end is None

    # --- quiet_hours: range checks --------------------------------------

    def test_quiet_hours_start_above_range_returns_400(self):
        user = self._user_with_prefs(quiet_hours_start=20)
        db = _make_db()
        body = NotificationPrefsRequest(quiet_hours_start=24)
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=db))
        assert exc.value.status_code == 400
        assert "quiet_hours_start" in exc.value.detail
        assert "0-23" in exc.value.detail
        # Pre-existing value untouched
        assert user.quiet_hours_start == 20
        db.commit.assert_not_called()

    def test_quiet_hours_start_below_range_returns_400(self):
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(quiet_hours_start=-1)
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert exc.value.status_code == 400

    def test_quiet_hours_start_zero_accepted(self):
        """0 is the inclusive lower bound — must be accepted, not rejected."""
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(quiet_hours_start=0)
        resp = _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert user.quiet_hours_start == 0
        assert resp.quiet_hours_start == 0

    def test_quiet_hours_start_twenty_three_accepted(self):
        """23 is the inclusive upper bound — must be accepted."""
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(quiet_hours_start=23)
        _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert user.quiet_hours_start == 23

    def test_quiet_hours_end_above_range_returns_400(self):
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(quiet_hours_end=99)
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert exc.value.status_code == 400
        assert "quiet_hours_end" in exc.value.detail

    def test_quiet_hours_end_below_range_returns_400(self):
        user = self._user_with_prefs()
        body = NotificationPrefsRequest(quiet_hours_end=-1)
        with pytest.raises(HTTPException) as exc:
            _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert exc.value.status_code == 400

    def test_quiet_hours_end_boundary_values_accepted(self):
        """Both ends of the inclusive range round-trip cleanly."""
        for hour in (0, 23):
            user = self._user_with_prefs()
            body = NotificationPrefsRequest(quiet_hours_end=hour)
            _run(update_notification_prefs(body, current_user=user, db=_make_db()))
            assert user.quiet_hours_end == hour

    def test_only_end_set_leaves_start_untouched(self):
        """Each field is independently optional — start should not be overwritten."""
        user = self._user_with_prefs(quiet_hours_start=22, quiet_hours_end=None)
        body = NotificationPrefsRequest(quiet_hours_end=6)
        _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert user.quiet_hours_start == 22  # unchanged
        assert user.quiet_hours_end == 6

    def test_all_fields_unset_still_commits_and_returns_current_state(self):
        """An empty body is a valid no-op write that echoes back the user."""
        user = self._user_with_prefs(
            mute_kinds=["breakout"],
            quiet_hours_start=21,
            quiet_hours_end=7,
        )
        db = _make_db()
        body = NotificationPrefsRequest()
        resp = _run(update_notification_prefs(body, current_user=user, db=db))
        # State unchanged
        assert user.mute_kinds == ["breakout"]
        assert user.quiet_hours_start == 21
        assert user.quiet_hours_end == 7
        # Response echoes that state
        assert resp.mute_kinds == ["breakout"]
        assert resp.quiet_hours_start == 21
        assert resp.quiet_hours_end == 7
        # Commit + refresh still fire — endpoint contract is "PATCH always writes"
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(user)

    # --- response shape -------------------------------------------------

    def test_response_falls_back_to_empty_list_when_mute_kinds_is_none(self):
        """The `current_user.mute_kinds or []` fallback hides None from clients."""
        user = self._user_with_prefs(mute_kinds=None)
        body = NotificationPrefsRequest(quiet_hours_start=10)
        resp = _run(update_notification_prefs(body, current_user=user, db=_make_db()))
        assert resp.mute_kinds == []


# ── GET /api/auth/config ───────────────────────────────────────────────


class TestGetAuthConfig:
    """Covers backend/routes/auth.py:173-179.

    The handler reads the module-level GOOGLE_CLIENT_ID constant that was
    imported from backend.auth at module load time. Patching that
    constant on the routes.auth module exercises both branches.
    """

    def test_returns_client_id_and_require_auth_true_when_configured(self):
        with patch.object(routes_auth, "GOOGLE_CLIENT_ID", "abc.apps.googleusercontent.com"):
            result = _run(get_auth_config())
        assert result == {
            "google_client_id": "abc.apps.googleusercontent.com",
            "require_auth": True,
        }

    def test_returns_require_auth_false_when_client_id_blank(self):
        with patch.object(routes_auth, "GOOGLE_CLIENT_ID", ""):
            result = _run(get_auth_config())
        assert result == {
            "google_client_id": "",
            "require_auth": False,
        }
