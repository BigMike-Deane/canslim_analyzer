"""
Broker mirror (2026-09-10): a share-for-share copy of one live book's
committed trades into an Alpaca PAPER account, to measure real fills against
booked prices.

The properties that matter, in the order they would hurt if broken:

  * PAPER ONLY -- the base URL cannot be redirected by configuration.
  * INERT UNTIL ACTIVATED -- no keys, or keys but no seed, means no orders.
  * EXACTLY ONCE -- re-running the job never submits a trade twice, and a
    retry after a timeout adopts the broker's existing order.
  * NEVER LEAVE DUST, NEVER OVERSELL -- full exits sell the broker's whole
    holding; partial sells are capped at what the broker holds.
  * FAILURES ARE LOUD -- a rejected order raises exactly one ops alert.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient

from backend import broker_mirror as bm
from backend.database import (
    init_db, SessionLocal, User, SystemSetting,
    AIPortfolioPosition, AIPortfolioTrade, BrokerMirrorOrder,
)
from backend.auth import get_current_active_user, get_admin_user
from backend.main import app
from tests.conftest import override_dependency

UID = 99050
init_db()


class FakeClient:
    """In-memory stand-in for AlpacaPaperClient."""

    def __init__(self, fill_prices=None, fractionable=True, positions=None):
        self.fill_prices = fill_prices or {}
        self.fractionable = fractionable
        self.held = dict(positions or {})
        self.orders = {}
        self.submits = []
        self.fail_next = None          # AlpacaError to raise on next submit
        self.fill_on_submit = True
        self.unknown_symbols = set()

    def account(self):
        return {"account_number": "PA1234567890", "status": "ACTIVE",
                "equity": "100000", "cash": "100000", "buying_power": "200000"}

    def clock(self):
        return {"is_open": True}

    def asset(self, symbol):
        if symbol in self.unknown_symbols:
            raise bm.AlpacaError(404, "asset not found")
        return {"symbol": symbol, "tradable": True, "fractionable": self.fractionable}

    def positions(self):
        return [{"symbol": s, "qty": str(q), "qty_available": str(q)}
                for s, q in self.held.items() if q > 0]

    def submit_order(self, symbol, qty, side, client_order_id):
        if self.fail_next:
            err, self.fail_next = self.fail_next, None
            raise err
        if client_order_id in self.orders:
            raise bm.AlpacaError(422, "client_order_id must be unique")
        self.submits.append((symbol, qty, side, client_order_id))
        order = {"id": f"o{len(self.orders)}", "client_order_id": client_order_id,
                 "status": "accepted", "filled_qty": "0", "filled_avg_price": None}
        self.orders[client_order_id] = order
        if self.fill_on_submit:
            self._fill(client_order_id, symbol, qty, side)
        return dict(order)

    def _fill(self, cid, symbol, qty, side):
        o = self.orders[cid]
        o.update(status="filled", filled_qty=str(qty),
                 filled_avg_price=str(self.fill_prices.get(symbol, 100.0)),
                 filled_at="2026-09-10T14:31:00Z")
        self.held[symbol] = self.held.get(symbol, 0.0) + (qty if side == "buy" else -qty)

    def order_by_client_id(self, cid):
        return dict(self.orders[cid])


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(bm, "mirror_config", lambda: {"enabled": True, "user_id": UID})
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")
    alerts = []
    import backend.email_utils as eu
    monkeypatch.setattr(eu, "send_ops_alert", lambda *a, **k: alerts.append((a, k)) or True)
    _wipe()
    yield alerts
    _wipe()


def _wipe():
    db = SessionLocal()
    try:
        db.query(BrokerMirrorOrder).filter(BrokerMirrorOrder.user_id == UID).delete()
        db.query(AIPortfolioTrade).filter(AIPortfolioTrade.user_id == UID).delete()
        db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == UID).delete()
        db.query(SystemSetting).filter(SystemSetting.key == bm.ACTIVATION_KEY).delete()
        if not db.query(User).filter_by(id=UID).first():
            db.add(User(id=UID, email=f"u{UID}@acct.example.org", display_name="mirror",
                        is_active=True, is_admin=False, hashed_password=""))
        db.commit()
    finally:
        db.close()


def _position(db, ticker, shares, price=100.0):
    db.add(AIPortfolioPosition(user_id=UID, ticker=ticker, shares=shares,
                               cost_basis=price, current_price=price))
    db.commit()


def _trade(db, ticker, action, shares, price=100.0, reason="test"):
    t = AIPortfolioTrade(user_id=UID, ticker=ticker, action=action, shares=shares,
                         price=price, total_value=shares * price, reason=reason,
                         executed_at=datetime.now(timezone.utc).replace(tzinfo=None))
    db.add(t)
    db.commit()
    return t


def _rows(db, **filters):
    q = db.query(BrokerMirrorOrder).filter(BrokerMirrorOrder.user_id == UID)
    for k, v in filters.items():
        q = q.filter(getattr(BrokerMirrorOrder, k) == v)
    return q.order_by(BrokerMirrorOrder.id).all()


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.close()


# ---------------------------------------------------------------- safety

class TestPaperOnly:

    def test_base_url_is_paper(self):
        assert bm.PAPER_BASE_URL == "https://paper-api.alpaca.markets"

    def test_environment_cannot_redirect_to_live(self, monkeypatch):
        for var in ("ALPACA_BASE_URL", "APCA_API_BASE_URL", "ALPACA_ENDPOINT"):
            monkeypatch.setenv(var, "https://api.alpaca.markets")
        assert bm.get_client().base == bm.PAPER_BASE_URL


class TestInertUntilActivated:

    def test_disabled(self, db, monkeypatch):
        monkeypatch.setattr(bm, "mirror_config", lambda: {"enabled": False, "user_id": UID})
        assert bm.run_mirror_cycle(db, FakeClient())["state"] == "disabled"

    def test_not_activated_submits_nothing(self, db):
        _trade(db, "AAA", "BUY", 5)
        client = FakeClient()
        assert bm.run_mirror_cycle(db, client)["state"] == "not_activated"
        assert client.submits == [] and _rows(db) == []

    def test_no_credentials(self, db, monkeypatch):
        c = FakeClient()
        bm.seed(db, c)
        monkeypatch.delenv("ALPACA_API_KEY_ID")
        assert bm.run_mirror_cycle(db)["state"] == "no_credentials"


# ---------------------------------------------------------------- activation

class TestSeed:

    def test_copies_the_held_book_and_sets_the_watermark(self, db):
        old = _trade(db, "OLD", "BUY", 3)
        _position(db, "AAA", 4.2)
        _position(db, "BBB", 10)
        client = FakeClient()
        res = bm.seed(db, client)
        assert res["ok"] and res["seeded"] == 2
        assert bm.get_activation()["watermark_trade_id"] == old.id
        seeds = _rows(db, action="SEED")
        assert [(r.ticker, r.internal_qty) for r in seeds] == [("AAA", 4.2), ("BBB", 10)]

    def test_refuses_twice(self, db):
        bm.seed(db, FakeClient())
        assert bm.seed(db, FakeClient()) == {"ok": False, "error": "already activated"}

    def test_seed_tops_up_only_what_the_broker_lacks(self, db):
        _position(db, "AAA", 10)
        bm.seed(db, FakeClient(positions={"AAA": 4}))
        assert _rows(db, action="SEED")[0].internal_qty == 6

    def test_trade_during_seeding_aborts(self, db):
        _position(db, "AAA", 10)
        client = FakeClient()

        def racing_positions():
            s = SessionLocal()
            try:
                _trade(s, "RACE", "BUY", 1)   # lands between watermark and write
            finally:
                s.close()
            return []
        client.positions = racing_positions
        res = bm.seed(db, client)
        assert res["ok"] is False and "retry" in res["error"]
        assert bm.get_activation() is None and _rows(db) == []

    def test_seed_orders_carry_no_slippage(self, db):
        _position(db, "AAA", 2, price=100.0)
        client = FakeClient(fill_prices={"AAA": 103.0})
        bm.seed(db, client)
        bm.run_mirror_cycle(db, client)
        r = _rows(db, action="SEED")[0]
        assert r.status == "filled" and r.slippage_bps is None


# ---------------------------------------------------------------- mirroring

class TestMirroring:

    def _activate(self, db, client=None):
        bm.seed(db, client or FakeClient())

    def test_only_trades_after_activation_are_mirrored(self, db):
        _trade(db, "BEFORE", "BUY", 1)
        client = FakeClient()
        self._activate(db, client)
        t = _trade(db, "AFTER", "BUY", 2.5, price=50.0)
        bm.run_mirror_cycle(db, client)
        assert [s[0] for s in client.submits] == ["AFTER"]
        r = _rows(db, trade_id=t.id)[0]
        assert r.client_order_id == f"cs-u{UID}-t{t.id}"
        assert r.status == "filled" and r.submitted_qty == 2.5

    def test_rerunning_never_submits_twice(self, db):
        client = FakeClient()
        self._activate(db, client)
        _trade(db, "AAA", "BUY", 1)
        for _ in range(3):
            bm.run_mirror_cycle(db, client)
        assert len(client.submits) == 1
        assert len(_rows(db, ticker="AAA")) == 1

    def test_timeout_then_retry_adopts_the_existing_order(self, db):
        client = FakeClient()
        self._activate(db, client)
        t = _trade(db, "AAA", "BUY", 1)
        # The broker accepted it but our side never saw the response.
        client.submit_order("AAA", 1, "buy", f"cs-u{UID}-t{t.id}")
        bm.run_mirror_cycle(db, client)
        r = _rows(db, trade_id=t.id)[0]
        assert r.status == "filled" and r.broker_order_id == "o0"
        assert len(client.submits) == 1

    def test_dry_run_trades_are_skipped(self, db):
        client = FakeClient()
        self._activate(db, client)
        _trade(db, "AAA", "BUY", 1, reason="PAPER: would buy")
        bm.run_mirror_cycle(db, client)
        assert client.submits == []
        assert _rows(db, ticker="AAA")[0].status == "skipped"

    def test_full_exit_sells_the_whole_broker_holding(self, db):
        # Broker holds a fractional sliver more than the book booked.
        client = FakeClient(positions={"AAA": 4.200000001})
        self._activate(db, client)
        _trade(db, "AAA", "SELL", 4.2)          # no internal position left
        bm.run_mirror_cycle(db, client)
        assert client.submits[-1][1] == pytest.approx(4.200000001)
        assert client.held["AAA"] == pytest.approx(0.0)

    def test_partial_sell_capped_at_broker_holding(self, db):
        client = FakeClient(positions={"AAA": 3})
        self._activate(db, client)
        _position(db, "AAA", 2)                 # still holding -> partial
        _trade(db, "AAA", "SELL", 5)
        bm.run_mirror_cycle(db, client)
        r = _rows(db, ticker="AAA", action="SELL")[0]
        assert r.submitted_qty == 3 and "capped" in r.note

    def test_non_fractionable_rounds_down_and_sub_share_skips(self, db):
        client = FakeClient(fractionable=False)
        self._activate(db, client)
        _trade(db, "AAA", "BUY", 7.6)
        _trade(db, "BBB", "PYRAMID", 0.4)
        bm.run_mirror_cycle(db, client)
        assert _rows(db, ticker="AAA")[0].submitted_qty == 7
        assert _rows(db, ticker="BBB")[0].status == "skipped"

    def test_sell_waits_for_an_unfilled_buy(self, db):
        client = FakeClient()
        client.fill_on_submit = False
        self._activate(db, client)
        _trade(db, "AAA", "BUY", 2)
        bm.run_mirror_cycle(db, client)          # buy accepted, not filled
        _trade(db, "AAA", "SELL", 2)
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, ticker="AAA", side="sell")[0]
        assert sell.status == "pending" and "waiting" in sell.note

    def test_class_share_symbol_resolved(self, db):
        client = FakeClient()
        client.unknown_symbols = {"BRK-B"}
        self._activate(db, client)
        _trade(db, "BRK-B", "BUY", 1)
        bm.run_mirror_cycle(db, client)
        assert client.submits[-1][0] == "BRK.B"


class TestFailures:

    def test_transient_error_stays_pending(self, db, _isolate):
        client = FakeClient()
        bm.seed(db, client)
        client.fail_next = bm.AlpacaError(503, "unavailable")
        _trade(db, "AAA", "BUY", 1)
        bm.run_mirror_cycle(db, client)
        assert _rows(db, ticker="AAA")[0].status == "pending"
        bm.run_mirror_cycle(db, client)          # recovers on the next tick
        assert _rows(db, ticker="AAA")[0].status == "filled"
        assert _isolate == []

    def test_rejection_is_an_error_with_exactly_one_alert(self, db, _isolate):
        client = FakeClient()
        bm.seed(db, client)
        client.fail_next = bm.AlpacaError(403, "insufficient buying power")
        _trade(db, "AAA", "BUY", 1)
        bm.run_mirror_cycle(db, client)
        bm.run_mirror_cycle(db, client)
        assert _rows(db, ticker="AAA")[0].status == "error"
        assert len(_isolate) == 1
        assert "AAA" in _isolate[0][0][1]


# ---------------------------------------------------------------- pure

class TestPure:

    def test_slippage_sign_is_worse_for_us_positive(self):
        assert bm.slippage_bps("buy", 100.0, 100.5) == 50.0     # paid more
        assert bm.slippage_bps("sell", 100.0, 99.0) == 100.0    # got less
        assert bm.slippage_bps("sell", 100.0, 101.0) == -100.0  # price improvement
        assert bm.slippage_bps("buy", None, 100.0) is None

    def test_qty_string_trims_float_noise(self):
        assert bm._qty_str(4.2) == "4.2"
        assert bm._qty_str(7.0) == "7"

    def test_reconcile_flags_mismatches(self):
        rec = {r["ticker"]: r for r in bm.reconcile({"A": 5.0, "B": 2.0}, {"A": 5.0, "C": 1.0})}
        assert rec["A"]["match"] and not rec["B"]["match"] and not rec["C"]["match"]
        assert rec["B"]["diff"] == -2.0

    def test_whole_share_rounding_is_a_match_with_a_note(self):
        # Live 2026-09-10: HWBK is not fractionable -> 108 at the broker vs
        # 108.33 booked. Expected, so it must not warn forever.
        rec = bm.reconcile({"HWBK": 108.325758}, {"HWBK": 108.0},
                           whole_share_only={"HWBK"})[0]
        assert rec["match"] is True and "whole shares" in rec["note"]

    def test_whole_share_exemption_does_not_hide_real_drift(self):
        # A full share short, or the broker holding MORE, is drift even on a
        # whole-share asset.
        short = bm.reconcile({"X": 110.3}, {"X": 108.0}, whole_share_only={"X"})[0]
        over = bm.reconcile({"X": 108.3}, {"X": 109.0}, whole_share_only={"X"})[0]
        assert short["match"] is False and over["match"] is False
        # And the exemption only applies to assets actually named.
        other = bm.reconcile({"Y": 10.5}, {"Y": 10.0})[0]
        assert other["match"] is False


# ---------------------------------------------------------------- endpoints

_admin = User(id=99051, email="mirror-admin@acct.example.org", display_name="Admin",
              is_active=True, is_admin=True, hashed_password="")


class TestEndpoints:

    def test_status_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY_ID")
        with override_dependency(get_current_active_user, _admin), \
             override_dependency(get_admin_user, _admin):
            r = TestClient(app).get("/api/admin/broker-mirror")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["configured"] is False and body["activation"] is None
        assert body["paper_base_url"] == bm.PAPER_BASE_URL

    def test_seed_without_credentials_is_refused(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY_ID")
        with override_dependency(get_current_active_user, _admin), \
             override_dependency(get_admin_user, _admin):
            r = TestClient(app).post("/api/admin/broker-mirror/seed")
        assert r.status_code == 409
        assert bm.get_activation() is None

    def test_status_reports_reconciliation_after_activation(self, db, monkeypatch):
        client = FakeClient()
        monkeypatch.setattr(bm, "get_client", lambda: client)
        _position(db, "AAA", 3)
        bm.seed(db, client)
        bm.run_mirror_cycle(db, client)
        with override_dependency(get_current_active_user, _admin), \
             override_dependency(get_admin_user, _admin):
            body = TestClient(app).get("/api/admin/broker-mirror").json()
        assert body["account"]["account_number"] == "…7890"   # masked
        rec = {r["ticker"]: r for r in body["reconciliation"]}
        assert rec["AAA"]["match"] is True
        # Naive-UTC timestamps must carry an offset, or the browser renders
        # them as local time (caught on screenshot: fills 5h off in Central).
        filled_at = body["orders"][0]["filled_at"]
        assert filled_at.endswith("+00:00"), filled_at
