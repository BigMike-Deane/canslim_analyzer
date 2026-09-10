"""
Broker mirror trade-update stream (2026-09-10): fills recorded the moment
Alpaca reports them instead of at the next minute poll.

What must hold:

  * PAPER ONLY -- the stream URL derives from the order client's constant.
  * ONLY OPEN ROWS MOVE -- a finished row (filled, lapsed, covered...) is
    never rewritten by a late or duplicate event; the poll stays authority.
  * THE HANDSHAKE IS CHECKED -- refused auth backs off for minutes instead
    of hammering; a dropped socket reconnects with growing backoff.
  * INERT UNTIL ACTIVATED -- no connection while there is nothing to mirror.
"""

import json
import os
import threading
from datetime import datetime, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend import broker_mirror as bm
from backend import broker_stream as bs
from backend.database import BrokerMirrorOrder, SessionLocal, SystemSetting, init_db

UID = 99080
init_db()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(bm, "mirror_config", lambda: {"enabled": True, "user_id": UID})
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    bs._state.update(connected=False, last_error=None, events=0, applied=0, reconnects=0)
    _wipe()
    yield
    _wipe()


def _wipe():
    db = SessionLocal()
    try:
        db.query(BrokerMirrorOrder).filter(BrokerMirrorOrder.user_id == UID).delete()
        db.query(SystemSetting).filter(SystemSetting.key == bm.ACTIVATION_KEY).delete()
        db.commit()
    finally:
        db.close()


def _row(cid, action="BUY", side="buy", status="submitted", **kw):
    db = SessionLocal()
    try:
        r = BrokerMirrorOrder(user_id=UID, ticker="AAA", action=action, side=side,
                              internal_qty=10, internal_price=kw.pop("internal_price", 100.0),
                              client_order_id=cid, status=status, submitted_qty=10, **kw)
        db.add(r)
        db.commit()
        return r.id
    finally:
        db.close()


def _get(rid):
    db = SessionLocal()
    try:
        return db.get(BrokerMirrorOrder, rid)
    finally:
        db.close()


def _event(cid, status="filled", price="100.50", event="fill"):
    return {"stream": "trade_updates", "data": {"event": event, "order": {
        "id": "b1", "client_order_id": cid, "status": status, "filled_qty": "10",
        "filled_avg_price": price, "filled_at": datetime.now(timezone.utc).isoformat()}}}


class TestApplying:

    def test_fill_moves_an_open_row(self):
        rid = _row("cs-u1-t1")
        bs.handle_message(json.dumps(_event("cs-u1-t1")).encode(), SessionLocal)  # binary frame
        r = _get(rid)
        assert r.status == "filled" and r.filled_avg_price == 100.5 and r.slippage_bps == 50.0
        assert bs._state["events"] == 1 and bs._state["applied"] == 1

    def test_stop_fill_is_measured_against_its_stop(self):
        rid = _row("cs-u1-xAAA-1", action="STOP", side="sell", stop_price=92.0,
                   cost_basis=100.0, internal_price=None)
        bs.handle_message(json.dumps(_event("cs-u1-xAAA-1", price="91.08")), SessionLocal)
        r = _get(rid)
        assert r.status == "filled" and r.slippage_bps == pytest.approx(100.0)

    @pytest.mark.parametrize("final", ["filled", "lapsed", "covered", "skipped", "error"])
    def test_finished_rows_are_never_rewritten(self, final):
        rid = _row("cs-u1-t2", status=final)
        bs.handle_message(json.dumps(_event("cs-u1-t2", status="canceled", event="canceled")),
                          SessionLocal)
        assert _get(rid).status == final and bs._state["applied"] == 0

    def test_orders_that_are_not_ours_are_ignored(self):
        bs.handle_message(json.dumps(_event("manual-order-123")), SessionLocal)
        assert bs._state["events"] == 1 and bs._state["applied"] == 0

    def test_other_streams_are_not_events(self):
        bs.handle_message(json.dumps({"stream": "listening", "data": {}}), SessionLocal)
        assert bs._state["events"] == 0


# ---------------------------------------------------------------- the connection

class FakeWS:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def send(self, msg):
        self.sent.append(json.loads(msg))

    def recv(self, timeout=None):
        if not self.frames:
            raise ConnectionError("socket closed")
        f = self.frames.pop(0)
        if isinstance(f, Exception):
            raise f
        return f


def _handshake(status="authorized"):
    return [json.dumps({"stream": "authorization", "data": {"action": "authenticate", "status": status}}),
            json.dumps({"stream": "listening", "data": {"streams": ["trade_updates"]}})]


class TestConnection:

    def test_stream_is_paper(self):
        assert bs.stream_url() == "wss://paper-api.alpaca.markets/stream"

    def test_session_authenticates_subscribes_and_applies(self, monkeypatch):
        monkeypatch.setattr(bs, "_ready", lambda: True)
        rid = _row("cs-u1-t3")
        ws = FakeWS(_handshake() + [TimeoutError(), json.dumps(_event("cs-u1-t3"))])
        urls = []

        def connect(url, **kw):
            urls.append(url)
            return ws
        with pytest.raises(ConnectionError):          # socket closes after the frames
            bs.run_session(connect, SessionLocal, threading.Event())
        assert urls == ["wss://paper-api.alpaca.markets/stream"]
        assert ws.sent[0] == {"action": "auth", "key": "key-id", "secret": "secret"}
        assert ws.sent[1] == {"action": "listen", "data": {"streams": ["trade_updates"]}}
        assert _get(rid).status == "filled"

    def test_refused_auth_is_an_auth_error(self):
        ws = FakeWS(_handshake(status="unauthorized"))
        with pytest.raises(bs.StreamAuthError):
            bs.run_session(lambda url, **kw: ws, SessionLocal, threading.Event())

    def _run_loop(self, monkeypatch, connect, ready=True, ticks=4):
        monkeypatch.setattr(bs, "_ready", lambda: ready)
        stop, sleeps = threading.Event(), []

        def sleep(s):
            sleeps.append(s)
            if len(sleeps) >= ticks:
                stop.set()
        bs._loop(connect, SessionLocal, sleep=sleep, stop=stop)
        return sleeps

    def test_inert_until_activated(self, monkeypatch):
        calls = []
        sleeps = self._run_loop(monkeypatch, lambda *a, **k: calls.append(1), ready=False, ticks=2)
        assert calls == [] and sleeps == [bs.IDLE_RECHECK_SECONDS] * 2

    def test_drops_reconnect_with_growing_backoff(self, monkeypatch):
        def connect(url, **kw):
            raise OSError("network down")
        sleeps = self._run_loop(monkeypatch, connect, ticks=5)
        assert sleeps == [1, 2, 4, 8, 16]
        assert "network down" in bs._state["last_error"]

    def test_refused_auth_backs_off_for_minutes(self, monkeypatch):
        sleeps = self._run_loop(monkeypatch, lambda url, **kw: FakeWS(_handshake("unauthorized")),
                                ticks=2)
        assert sleeps == [bs.UNAUTHORIZED_BACKOFF_SECONDS] * 2


class TestStatusEndpoint:

    def test_card_payload_carries_stream_status(self, monkeypatch):
        from fastapi.testclient import TestClient
        from backend.auth import get_admin_user, get_current_active_user
        from backend.database import User
        from backend.main import app
        from tests.conftest import override_dependency
        monkeypatch.delenv("ALPACA_API_KEY_ID")
        admin = User(id=99081, email="s@acct.example.org", display_name="A",
                     is_active=True, is_admin=True, hashed_password="")
        with override_dependency(get_current_active_user, admin), \
             override_dependency(get_admin_user, admin):
            body = TestClient(app).get("/api/admin/broker-mirror").json()
        assert set(body["stream"]) >= {"running", "connected", "last_event_at", "applied"}
