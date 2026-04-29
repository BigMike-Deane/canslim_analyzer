"""Web Push subscription + delivery tests.

Covers: VAPID public key endpoint, subscribe (insert + idempotent re-sub),
list, delete, cross-user isolation, and the email_utils delivery path —
specifically that 410/404 responses prune the subscription and that other
errors are logged but never raised.

We mock pywebpush.webpush since real delivery requires the browser vendor's
push service. The contract being tested is the WIRING, not the cryptography.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")
os.environ["VAPID_PUBLIC_KEY"] = "TEST_PUBLIC_KEY_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["VAPID_PRIVATE_KEY"] = "TEST_PRIVATE_KEY_xxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["VAPID_SUBJECT"] = "mailto:test@test.com"

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, SessionLocal, User, PushSubscription
from backend.auth import get_current_active_user

USER_A_ID = 7001
USER_B_ID = 7002

_user_a = User(id=USER_A_ID, email="a@push.test", display_name="A",
               is_active=True, is_admin=False, hashed_password="")
_user_b = User(id=USER_B_ID, email="b@push.test", display_name="B",
               is_active=True, is_admin=False, hashed_password="")

app.dependency_overrides[get_current_active_user] = lambda: _user_a

init_db()
client = TestClient(app)


def _ensure_users():
    db = SessionLocal()
    try:
        for u in (_user_a, _user_b):
            if not db.query(User).filter_by(id=u.id).first():
                db.add(User(id=u.id, email=u.email, display_name=u.display_name,
                            is_active=True, is_admin=False, hashed_password=""))
        db.commit()
    finally:
        db.close()


def _wipe_subs():
    db = SessionLocal()
    try:
        db.query(PushSubscription).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _setup_each():
    _ensure_users()
    _wipe_subs()
    app.dependency_overrides[get_current_active_user] = lambda: _user_a
    yield
    _wipe_subs()


def _act_as(user):
    app.dependency_overrides[get_current_active_user] = lambda: user


def _sub_payload(endpoint="https://fcm.googleapis.com/fcm/send/abc123"):
    return {"endpoint": endpoint, "keys": {"p256dh": "P256_KEY", "auth": "AUTH_KEY"}}


# ── VAPID public key endpoint ───────────────────────────────────────

class TestVapidPublicKey:
    def test_returns_public_key(self):
        r = client.get("/api/push/vapid-public-key")
        assert r.status_code == 200
        assert r.json()["public_key"].startswith("TEST_PUBLIC_KEY_")

    def test_503_when_unconfigured(self):
        with patch.dict(os.environ, {"VAPID_PUBLIC_KEY": ""}):
            r = client.get("/api/push/vapid-public-key")
            assert r.status_code == 503


# ── Subscribe / list / delete ────────────────────────────────────────

class TestSubscribe:
    def test_subscribe_inserts_row(self):
        r = client.post("/api/push/subscribe", json=_sub_payload())
        assert r.status_code == 200
        db = SessionLocal()
        try:
            rows = db.query(PushSubscription).filter_by(user_id=USER_A_ID).all()
            assert len(rows) == 1
            assert rows[0].endpoint == "https://fcm.googleapis.com/fcm/send/abc123"
            assert rows[0].p256dh_key == "P256_KEY"
            assert rows[0].auth_key == "AUTH_KEY"
        finally:
            db.close()

    def test_resubscribe_same_endpoint_updates_in_place(self):
        """Phone clears site data → permission re-granted → same endpoint comes
        back with possibly-new keys. Must not create a duplicate row."""
        client.post("/api/push/subscribe", json=_sub_payload())
        # Same endpoint, new keys
        client.post("/api/push/subscribe", json={
            "endpoint": "https://fcm.googleapis.com/fcm/send/abc123",
            "keys": {"p256dh": "NEW_P256", "auth": "NEW_AUTH"},
        })
        db = SessionLocal()
        try:
            rows = db.query(PushSubscription).filter_by(
                endpoint="https://fcm.googleapis.com/fcm/send/abc123").all()
            assert len(rows) == 1
            assert rows[0].p256dh_key == "NEW_P256"
            assert rows[0].auth_key == "NEW_AUTH"
        finally:
            db.close()

    def test_user_agent_recorded(self):
        client.post("/api/push/subscribe", json=_sub_payload(),
                    headers={"User-Agent": "Mozilla/5.0 iPhone Test"})
        db = SessionLocal()
        try:
            row = db.query(PushSubscription).filter_by(user_id=USER_A_ID).first()
            assert row.user_agent and "iPhone" in row.user_agent
        finally:
            db.close()

    def test_list_returns_only_own_subscriptions(self):
        _act_as(_user_a)
        client.post("/api/push/subscribe", json=_sub_payload("https://example/a1"))
        client.post("/api/push/subscribe", json=_sub_payload("https://example/a2"))
        _act_as(_user_b)
        client.post("/api/push/subscribe", json=_sub_payload("https://example/b1"))

        _act_as(_user_a)
        a_subs = client.get("/api/push/subscriptions").json()
        assert len(a_subs) == 2

        _act_as(_user_b)
        b_subs = client.get("/api/push/subscriptions").json()
        assert len(b_subs) == 1

    def test_delete_only_own_subscription(self):
        _act_as(_user_b)
        client.post("/api/push/subscribe", json=_sub_payload("https://example/b1"))
        b_id = client.get("/api/push/subscriptions").json()[0]["id"]

        _act_as(_user_a)
        # User A trying to delete user B's subscription gets 404
        r = client.delete(f"/api/push/subscriptions/{b_id}")
        assert r.status_code == 404

        # Subscription still exists
        _act_as(_user_b)
        assert len(client.get("/api/push/subscriptions").json()) == 1


# ── Delivery: send_web_push_to_user / broadcast ──────────────────────

class TestDelivery:
    def _seed_sub(self, user_id, endpoint="https://example/sub1"):
        db = SessionLocal()
        try:
            sub = PushSubscription(
                user_id=user_id, endpoint=endpoint,
                p256dh_key="K1", auth_key="A1", user_agent="Test",
            )
            db.add(sub)
            db.commit()
            db.refresh(sub)
            return sub.id
        finally:
            db.close()

    @patch("pywebpush.webpush")
    def test_send_to_user_fans_out_to_all_devices(self, mock_webpush):
        from backend.email_utils import send_web_push_to_user
        for ep in ("https://e/1", "https://e/2", "https://e/3"):
            self._seed_sub(USER_A_ID, ep)
        # User B's device should NOT be called
        self._seed_sub(USER_B_ID, "https://e/other")

        sent = send_web_push_to_user(USER_A_ID, "T", "B", data={"x": 1})
        assert sent == 3
        # webpush called once per A device (not once for B)
        assert mock_webpush.call_count == 3
        called_endpoints = {c.kwargs["subscription_info"]["endpoint"]
                            for c in mock_webpush.call_args_list}
        assert called_endpoints == {"https://e/1", "https://e/2", "https://e/3"}

    @patch("pywebpush.webpush")
    def test_410_response_prunes_subscription(self, mock_webpush):
        from backend.email_utils import send_web_push_to_user
        from pywebpush import WebPushException
        sub_id = self._seed_sub(USER_A_ID, "https://gone")
        # Simulate a "Gone" response from the push service
        resp = MagicMock(status_code=410)
        mock_webpush.side_effect = WebPushException("Gone", response=resp)

        sent = send_web_push_to_user(USER_A_ID, "t", "b")
        assert sent == 0
        # Subscription was pruned
        db = SessionLocal()
        try:
            assert db.query(PushSubscription).filter_by(id=sub_id).first() is None
        finally:
            db.close()

    @patch("pywebpush.webpush")
    def test_500_error_does_not_prune(self, mock_webpush):
        """Transient server errors must not destroy the subscription."""
        from backend.email_utils import send_web_push_to_user
        from pywebpush import WebPushException
        sub_id = self._seed_sub(USER_A_ID, "https://flaky")
        resp = MagicMock(status_code=500)
        mock_webpush.side_effect = WebPushException("Server error", response=resp)

        send_web_push_to_user(USER_A_ID, "t", "b")
        db = SessionLocal()
        try:
            assert db.query(PushSubscription).filter_by(id=sub_id).first() is not None
        finally:
            db.close()

    @patch("pywebpush.webpush")
    def test_no_private_key_skips_silently(self, mock_webpush):
        from backend.email_utils import send_web_push_to_user
        self._seed_sub(USER_A_ID)
        with patch.dict(os.environ, {"VAPID_PRIVATE_KEY": ""}):
            sent = send_web_push_to_user(USER_A_ID, "t", "b")
        assert sent == 0
        assert mock_webpush.call_count == 0

    @patch("pywebpush.webpush")
    def test_broadcast_only_active_users(self, mock_webpush):
        from backend.email_utils import send_web_push_broadcast
        self._seed_sub(USER_A_ID, "https://a")
        self._seed_sub(USER_B_ID, "https://b")
        # Disable user B
        db = SessionLocal()
        try:
            db.query(User).filter_by(id=USER_B_ID).update({User.is_active: False})
            db.commit()
        finally:
            db.close()
        try:
            send_web_push_broadcast("t", "b")
            endpoints = {c.kwargs["subscription_info"]["endpoint"]
                         for c in mock_webpush.call_args_list}
            # Only A's endpoint is hit; B is inactive
            assert "https://a" in endpoints
            assert "https://b" not in endpoints
        finally:
            # restore
            db = SessionLocal()
            try:
                db.query(User).filter_by(id=USER_B_ID).update({User.is_active: True})
                db.commit()
            finally:
                db.close()
