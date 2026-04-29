"""In-app notification tests — dual-write, isolation, list/read/delete routes.

Mirrors the pattern in test_api_routes.py: dependency_overrides for auth,
real SessionLocal/init_db DB, seed users explicitly. Two users are seeded so
cross-user isolation can be exercised by swapping the auth override.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, SessionLocal, User, Notification
from backend.auth import get_current_active_user
from backend.email_utils import (
    create_notification,
    broadcast_notification,
    send_trade_webhook,
    send_stop_loss_webhook,
    send_score_crash_warning_push,
)


# ── Setup ────────────────────────────────────────────────────────────

USER_A_ID = 9001
USER_B_ID = 9002

_user_a = User(id=USER_A_ID, email="a@test.com", display_name="A",
               is_active=True, is_admin=False, hashed_password="")
_user_b = User(id=USER_B_ID, email="b@test.com", display_name="B",
               is_active=True, is_admin=False, hashed_password="")

# Default override: act as user A. Individual tests swap to B as needed.
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


def _wipe_notifications():
    """Wipe ALL notifications between tests, not just our two test users.
    Broadcast tests fan out to every active user (including the admin user 1
    seeded by init_db), so scoping the wipe by id leaks rows across tests."""
    db = SessionLocal()
    try:
        db.query(Notification).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _setup_each():
    _ensure_users()
    _wipe_notifications()
    app.dependency_overrides[get_current_active_user] = lambda: _user_a
    yield
    _wipe_notifications()


def _act_as(user):
    app.dependency_overrides[get_current_active_user] = lambda: user


# ── Dual-write: trade events persist a Notification row ──────────────

class TestDualWrite:
    @patch("backend.email_utils.requests.post")
    def test_trade_webhook_creates_notification(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        with patch("backend.email_utils.WEBHOOK_URL", ""):
            send_trade_webhook("AAPL", "BUY", 10, 150.0, "breakout", user_id=USER_A_ID)

        db = SessionLocal()
        try:
            rows = db.query(Notification).filter(Notification.user_id == USER_A_ID).all()
            assert len(rows) == 1
            n = rows[0]
            assert n.kind == "trade"
            assert n.title == "BUY: AAPL"
            assert n.priority == "high"
            assert n.read_at is None
            assert n.data["ticker"] == "AAPL"
            assert n.data["action"] == "BUY"
        finally:
            db.close()

    @patch("backend.email_utils.requests.post")
    def test_stop_loss_creates_urgent_notification(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        send_stop_loss_webhook("NVDA", 5, 100.0, "STOP LOSS", -7.0, user_id=USER_A_ID)

        db = SessionLocal()
        try:
            n = db.query(Notification).filter_by(user_id=USER_A_ID, kind="stop_loss").one()
            assert n.priority == "urgent"
            assert n.title == "STOP LOSS: NVDA"
            assert n.data["loss_pct"] == -7.0
        finally:
            db.close()

    @patch("backend.email_utils.requests.post")
    def test_score_crash_creates_notification(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        send_score_crash_warning_push("TSLA", 80.0, 60.0, 5.0, 2, 3, user_id=USER_A_ID)

        db = SessionLocal()
        try:
            n = db.query(Notification).filter_by(user_id=USER_A_ID, kind="score_crash").one()
            assert n.priority == "high"
            assert n.data["consecutive_low"] == 2
        finally:
            db.close()

    @patch("backend.email_utils.requests.post")
    def test_dual_write_when_user_has_no_webhook(self, mock_post):
        """Even when ntfy is silenced (no URL), the in-app row is still created."""
        # User has no webhook_url — get_user_webhook_url returns "" → ntfy skipped.
        # But the DB write must still happen.
        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            send_trade_webhook("MSFT", "BUY", 1, 200.0, "test", user_id=USER_A_ID)

        # ntfy was NOT called (user URL empty silences even with global set)
        assert mock_post.call_count == 0
        # But the notification row exists
        db = SessionLocal()
        try:
            assert db.query(Notification).filter_by(user_id=USER_A_ID).count() == 1
        finally:
            db.close()

    @patch("backend.email_utils.requests.post")
    def test_db_error_does_not_block_ntfy(self, mock_post):
        """create_notification is fail-soft: a DB error inside it must not
        prevent the ntfy POST from firing. The phone push is more important
        than the in-app surface in a degraded-DB scenario.

        Models a real DB outage by making SessionLocal raise — the realistic
        failure mode (connection pool exhausted, network partition, etc.).
        """
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        with patch("backend.database.SessionLocal",
                   side_effect=RuntimeError("simulated db down")), \
             patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            # user_id=None → ntfy via legacy global URL, no per-user lookup.
            ok = send_trade_webhook("AAPL", "BUY", 1, 100.0, "x", user_id=None)
        assert ok is True
        assert mock_post.call_count == 1

    def test_create_notification_returns_false_for_missing_user(self):
        assert create_notification(0, "trade", "T", "B") is False
        assert create_notification(None, "trade", "T", "B") is False


# ── Broadcast helper for system events ───────────────────────────────

class TestBroadcastNotification:
    """Broadcast fans out to all active users. The shared test DB may have
    other active users (e.g. the admin seed at id=1) so we assert by user_id
    rather than total row count, and compare the return value to the actual
    active count at test time."""

    def _active_count(self):
        db = SessionLocal()
        try:
            return db.query(User).filter(User.is_active == True).count()
        finally:
            db.close()

    def test_broadcast_writes_to_every_active_user(self):
        active = self._active_count()
        n = broadcast_notification(kind="breakout", title="X up", body="hi",
                                   priority="high", tags=["fire"],
                                   data={"ticker": "X"})
        assert n == active
        db = SessionLocal()
        try:
            for uid in (USER_A_ID, USER_B_ID):
                row = (db.query(Notification)
                       .filter_by(user_id=uid, kind="breakout").one())
                assert row.title == "X up"
                assert row.priority == "high"
                assert row.data["ticker"] == "X"
        finally:
            db.close()

    def test_broadcast_skips_inactive_users(self):
        # Add a third user marked inactive — they must NOT receive a row.
        db = SessionLocal()
        try:
            inactive_id = USER_B_ID + 1
            if not db.query(User).filter_by(id=inactive_id).first():
                db.add(User(id=inactive_id, email="off@test.com",
                            display_name="Off", is_active=False, is_admin=False,
                            hashed_password=""))
                db.commit()
        finally:
            db.close()
        active_before = self._active_count()
        n = broadcast_notification(kind="spy_gate_change", title="flip", body="b")
        assert n == active_before  # excludes the new inactive user
        db = SessionLocal()
        try:
            assert db.query(Notification).filter_by(user_id=inactive_id).count() == 0
            db.query(Notification).filter_by(user_id=inactive_id).delete()
            db.query(User).filter_by(id=inactive_id).delete()
            db.commit()
        finally:
            db.close()

    def test_broadcast_returns_zero_when_no_active_users(self):
        # Temporarily disable every user, then restore.
        db = SessionLocal()
        try:
            db.query(User).update({User.is_active: False}, synchronize_session=False)
            db.commit()
        finally:
            db.close()
        try:
            assert broadcast_notification(kind="risk_alert", title="t", body="b") == 0
        finally:
            db = SessionLocal()
            try:
                db.query(User).update({User.is_active: True}, synchronize_session=False)
                db.commit()
            finally:
                db.close()

    @patch("backend.email_utils.requests.post")
    def test_breakout_path_dual_writes(self, mock_post):
        """Same call shape as breakout_monitor.py — broadcast + ntfy together."""
        mock_post.return_value = MagicMock(status_code=200, text="ok")
        active = self._active_count()
        broadcast_notification(kind="breakout", title="AAPL ↑", body="x")
        from backend.email_utils import send_webhook_notification
        with patch("backend.email_utils.WEBHOOK_URL", "https://global.example/topic"):
            send_webhook_notification("AAPL ↑", "x", priority="high")
        assert mock_post.call_count == 1
        db = SessionLocal()
        try:
            assert db.query(Notification).filter_by(kind="breakout").count() == active
            # And specifically: both our test users got it
            for uid in (USER_A_ID, USER_B_ID):
                assert db.query(Notification).filter_by(
                    user_id=uid, kind="breakout").count() == 1
        finally:
            db.close()


# ── Cross-user isolation ─────────────────────────────────────────────

class TestIsolation:
    def _seed(self):
        db = SessionLocal()
        try:
            db.add(Notification(user_id=USER_A_ID, kind="trade", title="A1", body=""))
            db.add(Notification(user_id=USER_A_ID, kind="trade", title="A2", body=""))
            db.add(Notification(user_id=USER_B_ID, kind="trade", title="B1", body=""))
            db.commit()
        finally:
            db.close()

    def test_list_only_returns_own_notifications(self):
        self._seed()
        _act_as(_user_a)
        items = client.get("/api/notifications").json()["items"]
        titles = {i["title"] for i in items}
        assert titles == {"A1", "A2"}

        _act_as(_user_b)
        items = client.get("/api/notifications").json()["items"]
        assert {i["title"] for i in items} == {"B1"}

    def test_unread_count_is_per_user(self):
        self._seed()
        _act_as(_user_a)
        assert client.get("/api/notifications/unread-count").json()["count"] == 2
        _act_as(_user_b)
        assert client.get("/api/notifications/unread-count").json()["count"] == 1

    def test_user_cannot_read_other_users_notification(self):
        self._seed()
        # Find user B's notification id while acting as B
        _act_as(_user_b)
        b_id = client.get("/api/notifications").json()["items"][0]["id"]
        # Now act as A and try to mark B's as read → 404
        _act_as(_user_a)
        r = client.post(f"/api/notifications/{b_id}/read")
        assert r.status_code == 404

    def test_user_cannot_delete_other_users_notification(self):
        self._seed()
        _act_as(_user_b)
        b_id = client.get("/api/notifications").json()["items"][0]["id"]
        _act_as(_user_a)
        assert client.delete(f"/api/notifications/{b_id}").status_code == 404
        # B's row still exists
        _act_as(_user_b)
        assert client.get("/api/notifications/unread-count").json()["count"] == 1

    def test_read_all_only_affects_current_user(self):
        self._seed()
        _act_as(_user_a)
        client.post("/api/notifications/read-all")
        # B's unread is unchanged
        _act_as(_user_b)
        assert client.get("/api/notifications/unread-count").json()["count"] == 1


# ── List behavior: ordering, filtering, pagination ───────────────────

class TestListing:
    def _seed_n(self, n: int, user_id: int = USER_A_ID):
        db = SessionLocal()
        try:
            for i in range(n):
                db.add(Notification(user_id=user_id, kind="trade",
                                    title=f"#{i}", body=""))
                db.flush()  # ensure created_at differs by row
            db.commit()
        finally:
            db.close()

    def test_newest_first(self):
        self._seed_n(3)
        items = client.get("/api/notifications").json()["items"]
        # Inserted #0, #1, #2 in order — list returns newest first.
        assert [i["title"] for i in items] == ["#2", "#1", "#0"]

    def test_unread_only_filter(self):
        self._seed_n(2)
        # Mark one as read
        items = client.get("/api/notifications").json()["items"]
        client.post(f"/api/notifications/{items[0]['id']}/read")

        all_items = client.get("/api/notifications").json()
        assert all_items["total"] == 2
        unread = client.get("/api/notifications?unread_only=true").json()
        assert unread["total"] == 1
        assert unread["unread_count"] == 1  # always overall, not filter-scoped

    def test_pagination(self):
        self._seed_n(5)
        page1 = client.get("/api/notifications?limit=2&offset=0").json()
        page2 = client.get("/api/notifications?limit=2&offset=2").json()
        assert len(page1["items"]) == 2
        assert len(page2["items"]) == 2
        assert page1["total"] == page2["total"] == 5
        # No overlap
        ids1 = {i["id"] for i in page1["items"]}
        ids2 = {i["id"] for i in page2["items"]}
        assert ids1.isdisjoint(ids2)


# ── Mutation: read / read-all / delete ───────────────────────────────

class TestMutations:
    def _seed_one(self):
        db = SessionLocal()
        try:
            n = Notification(user_id=USER_A_ID, kind="trade", title="X", body="")
            db.add(n)
            db.commit()
            return n.id
        finally:
            db.close()

    def test_mark_read_sets_read_at(self):
        nid = self._seed_one()
        r = client.post(f"/api/notifications/{nid}/read")
        assert r.status_code == 200
        assert r.json()["read_at"] is not None
        # Idempotent: hitting again doesn't change it
        first_read_at = r.json()["read_at"]
        r2 = client.post(f"/api/notifications/{nid}/read")
        assert r2.json()["read_at"] == first_read_at

    def test_mark_all_read_returns_count(self):
        for _ in range(3):
            self._seed_one()
        r = client.post("/api/notifications/read-all")
        assert r.json()["updated"] == 3
        assert client.get("/api/notifications/unread-count").json()["count"] == 0

    def test_mark_all_read_when_none_unread(self):
        r = client.post("/api/notifications/read-all")
        assert r.json()["updated"] == 0

    def test_delete_removes_notification(self):
        nid = self._seed_one()
        r = client.delete(f"/api/notifications/{nid}")
        assert r.status_code == 200
        assert client.get("/api/notifications/unread-count").json()["count"] == 0

    def test_read_nonexistent_returns_404(self):
        r = client.post("/api/notifications/999999/read")
        assert r.status_code == 404
