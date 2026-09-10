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

Phase 2 (resting stops) adds:

  * ONE WORKING STOP PER HELD POSITION at the app's effective hard stop,
    re-placed only when the level or size really changes.
  * A SELL NEVER COLLIDES WITH ITS STOP -- the stop reserves the shares, so
    it is canceled first (the fake broker enforces the reservation).
  * A FIRED STOP IS PAIRED, NOT DUPLICATED -- the book's later sell is
    "covered"; while the book still holds, the broker stays flat.
  * ROUTINE ENDINGS ARE QUIET -- expiry at the close / our own cancels
    never alert.
"""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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

    def __init__(self, fill_prices=None, fractionable=True, positions=None, prices=None):
        self.fill_prices = fill_prices or {}
        self.fractionable = fractionable
        self.held = dict(positions or {})
        self.prices = dict(prices or {})   # market price per symbol (default 100)
        self.orders = {}
        self.submits = []
        self.stop_submits = []
        self.fail_next = None          # AlpacaError to raise on next submit
        self.fail_next_stop = None     # ... on next stop submit
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

    def _reserved(self, symbol):
        # Alpaca holds shares for open sell orders: qty_available excludes them.
        return sum(o["qty"] for o in self.orders.values()
                   if o["symbol"] == symbol and o["side"] == "sell"
                   and o["status"] in ("new", "accepted", "pending_cancel"))

    def positions(self):
        return [{"symbol": s, "qty": str(q), "qty_available": str(q - self._reserved(s)),
                 "current_price": str(self.prices.get(s, 100.0))}
                for s, q in self.held.items() if q > 0]

    def submit_order(self, symbol, qty, side, client_order_id):
        if self.fail_next:
            err, self.fail_next = self.fail_next, None
            raise err
        if client_order_id in self.orders:
            raise bm.AlpacaError(422, "client_order_id must be unique")
        if side == "sell" and qty > self.held.get(symbol, 0.0) - self._reserved(symbol) + 1e-9:
            raise bm.AlpacaError(403, "insufficient qty available for order")
        self.submits.append((symbol, qty, side, client_order_id))
        order = {"id": f"o{len(self.orders)}", "client_order_id": client_order_id,
                 "symbol": symbol, "side": side, "qty": qty, "type": "market",
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
        if cid not in self.orders:
            raise bm.AlpacaError(404, "order not found")
        return dict(self.orders[cid])

    # -- resting stops
    def submit_stop_order(self, symbol, qty, stop_price, client_order_id):
        if self.fail_next_stop:
            err, self.fail_next_stop = self.fail_next_stop, None
            raise err
        if client_order_id in self.orders:
            raise bm.AlpacaError(422, "client_order_id must be unique")
        if qty > self.held.get(symbol, 0.0) - self._reserved(symbol) + 1e-9:
            raise bm.AlpacaError(403, "insufficient qty available for order")
        self.stop_submits.append((symbol, qty, stop_price, client_order_id))
        order = {"id": f"o{len(self.orders)}", "client_order_id": client_order_id,
                 "symbol": symbol, "side": "sell", "qty": qty, "type": "stop",
                 "stop_price": stop_price, "status": "new",
                 "filled_qty": "0", "filled_avg_price": None}
        self.orders[client_order_id] = order
        return dict(order)

    def cancel_order(self, order_id):
        o = next(o for o in self.orders.values() if o["id"] == order_id)
        if o["status"] not in ("new", "accepted"):
            raise bm.AlpacaError(422, "order is not cancelable")
        o["status"] = "canceled"

    def open_stops(self, symbol=None):
        return [o for o in self.orders.values() if o["type"] == "stop"
                and o["status"] in ("new", "accepted", "pending_cancel")
                and (symbol is None or o["symbol"] == symbol)]

    def trigger(self, symbol, fill_price):
        """The tape trades through the stop: it fires as a market order."""
        for o in self.open_stops(symbol):
            o.update(status="filled", filled_qty=str(o["qty"]), filled_avg_price=str(fill_price),
                     filled_at=datetime.now(timezone.utc).isoformat())
            self.held[symbol] = self.held.get(symbol, 0.0) - o["qty"]

    def expire_all(self):
        for o in self.open_stops():
            o["status"] = "expired"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(bm, "mirror_config", lambda: {"enabled": True, "user_id": UID})
    # Phase-1 tests run without resting stops; TestRestingStops turns them on.
    monkeypatch.setattr(bm, "resting_stop_config",
                        lambda: {**bm._RESTING_STOP_DEFAULTS, "enabled": False})
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


def _trade(db, ticker, action, shares, price=100.0, reason="test", signal_factors=None):
    t = AIPortfolioTrade(user_id=UID, ticker=ticker, action=action, shares=shares,
                         price=price, total_value=shares * price, reason=reason,
                         signal_factors=signal_factors,
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


# ---------------------------------------------------------------- phase 2: resting stops

@pytest.fixture
def stops_on(monkeypatch):
    """Resting stops enabled, with the stop level injected: ``levels`` maps
    ticker -> stop % below cost (default 8). hard_stop_level's own chain is
    tested separately in TestHardStopLevel."""
    levels = {}
    monkeypatch.setattr(bm, "resting_stop_config",
                        lambda: {**bm._RESTING_STOP_DEFAULTS, "enabled": True})
    monkeypatch.setattr(bm, "stop_context", lambda db, uid: {})

    def level(pos, ctx, today=None):
        pct = levels.get(pos.ticker, 8.0)
        return {"stop_pct": pct, "cost_basis": pos.cost_basis,
                "stop_price": bm.round_stop_price(pos.cost_basis * (1 - pct / 100))}
    monkeypatch.setattr(bm, "hard_stop_level", level)
    return levels


def _held_and_mirrored(db, ticker="AAA", shares=10, cost=100.0, client=None):
    """A position the book holds and the broker mirrors, one cycle in."""
    client = client or FakeClient()
    _position(db, ticker, shares, price=cost)
    bm.seed(db, client)
    bm.run_mirror_cycle(db, client)
    return client


def _exit_book(db, ticker, shares, price, reason):
    """The book fully exits: its SELL is booked and the position row is gone."""
    t = _trade(db, ticker, "SELL", shares, price=price, reason=reason)
    db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == UID,
                                         AIPortfolioPosition.ticker == ticker).delete()
    db.commit()
    return t


def _open_stop_rows(db, ticker="AAA"):
    return [r for r in _rows(db, ticker=ticker, action="STOP") if r.status in bm.STOP_OPEN]


class TestRestingStops:

    def test_one_day_stop_at_the_effective_hard_stop_and_no_churn(self, db, stops_on):
        client = _held_and_mirrored(db)
        assert [(s[0], s[1], s[2]) for s in client.stop_submits] == [("AAA", 10, 92.0)]
        row = _open_stop_rows(db)[0]
        assert row.status == "submitted" and row.stop_pct == 8.0 and row.cost_basis == 100.0
        for _ in range(3):
            bm.run_mirror_cycle(db, client)
        assert len(client.stop_submits) == 1

    def test_stop_follows_the_level_but_not_its_wiggles(self, db, stops_on):
        client = _held_and_mirrored(db)
        stops_on["AAA"] = 8.1                     # 92.00 -> 91.90: 0.11%, inside tolerance
        bm.run_mirror_cycle(db, client)
        assert len(client.stop_submits) == 1
        stops_on["AAA"] = 10.0                    # 92.00 -> 90.00: a real move
        bm.run_mirror_cycle(db, client)
        assert [s[2] for s in client.stop_submits] == [92.0, 90.0]
        stops = _rows(db, ticker="AAA", action="STOP")
        assert stops[0].status == "lapsed" and "replaced" in stops[0].note
        assert [r.stop_price for r in _open_stop_rows(db)] == [90.0]
        assert len(client.open_stops("AAA")) == 1

    def test_book_sell_cancels_the_stop_then_sells(self, db, stops_on):
        # The fake broker reserves the stop's shares exactly like Alpaca:
        # a sell sent without canceling first would be a 403.
        client = _held_and_mirrored(db)
        stop = _open_stop_rows(db)[0]
        t = _exit_book(db, "AAA", 10, 104.0, "TRAILING STOP: Peak $110 -> $104")
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, trade_id=t.id)[0]
        assert sell.status == "filled" and sell.submitted_qty == 10
        assert sell.linked_order_id == stop.id
        db.refresh(stop)
        assert stop.status == "lapsed" and "selling" in stop.note
        assert client.open_stops() == [] and _open_stop_rows(db) == []

    def test_partial_sell_resizes_the_stop(self, db, stops_on):
        client = _held_and_mirrored(db)
        _trade(db, "AAA", "SELL", 4, price=120.0, reason="PARTIAL TRAILING STOP (50%)")
        db.query(AIPortfolioPosition).filter_by(user_id=UID, ticker="AAA").update({"shares": 6})
        db.commit()
        bm.run_mirror_cycle(db, client)
        assert [r.submitted_qty for r in _open_stop_rows(db)] == [pytest.approx(6)]
        assert client.held["AAA"] == pytest.approx(6)

    def test_no_resize_while_a_buy_is_in_flight(self, db, stops_on):
        client = _held_and_mirrored(db)
        client.fill_on_submit = False
        t = _trade(db, "AAA", "PYRAMID", 2)
        db.query(AIPortfolioPosition).filter_by(user_id=UID, ticker="AAA").update({"shares": 12})
        db.commit()
        bm.run_mirror_cycle(db, client)
        assert [r.submitted_qty for r in _open_stop_rows(db)] == [10]   # waits
        buy = _rows(db, trade_id=t.id)[0]
        client._fill(buy.client_order_id, "AAA", 2, "buy")
        bm.run_mirror_cycle(db, client)
        assert [r.submitted_qty for r in _open_stop_rows(db)] == [pytest.approx(12)]

    def test_gap_through_the_stop_pairs_with_the_apps_stop_loss(self, db, stops_on):
        client = _held_and_mirrored(db)
        client.trigger("AAA", 90.5)               # opens below the 92 stop
        # The 15-min checker books its own stop-loss later, lower.
        t = _trade(db, "AAA", "SELL", 10, price=89.0, reason="STOP LOSS: Down 11.0%",
                   signal_factors={"sell_reason": "STOP LOSS", "stop_pct": 8.0,
                                   "gain_pct": -11.0, "slippage_pp": 3.0})
        db.query(AIPortfolioPosition).filter_by(user_id=UID, ticker="AAA").delete()
        db.commit()
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, trade_id=t.id)[0]
        assert sell.status == "covered" and sell.pairing == "same_exit"
        assert sell.filled_avg_price == 90.5
        assert sell.slippage_bps == pytest.approx(-168.5)   # broker did BETTER than booked
        assert [s[2] for s in client.submits] == ["buy"]    # the seed only: no second sell
        stop = _rows(db, action="STOP")[0]
        assert stop.slippage_bps == pytest.approx(163.0)    # vs its own stop price
        s = bm.summarize(_rows(db), internal_slippage_pp={t.id: 3.0})["resting_stops"]
        assert s["fired"] == 1 and s["same_exit"]["n"] == 1
        assert s["same_exit"]["broker_slippage_pp"] == pytest.approx(1.5)
        assert s["same_exit"]["app_slippage_pp"] == pytest.approx(3.0)

    def test_whipsaw_leaves_the_broker_flat_until_the_book_exits(self, db, stops_on):
        client = _held_and_mirrored(db)
        client.trigger("AAA", 91.8)               # dips through; the checker never sees it
        bm.run_mirror_cycle(db, client)
        assert bm.open_stop_exit(db, UID, "AAA") is not None
        assert len(client.stop_submits) == 1      # nothing left to protect
        rec = bm.reconcile({"AAA": 10.0}, {}, stopped_out={"AAA"})[0]
        assert rec["divergence"] == "resting_stop" and rec["match"] is False

        add = _trade(db, "AAA", "PYRAMID", 2)     # the book adds; the broker stays out
        bm.run_mirror_cycle(db, client)
        assert _rows(db, trade_id=add.id)[0].status == "skipped"
        assert [s[2] for s in client.submits] == ["buy"]

        out = _exit_book(db, "AAA", 12, 97.0, "TRAILING STOP: Peak $104 -> $97")
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, trade_id=out.id)[0]
        assert sell.status == "covered" and sell.pairing == "whipsaw"
        assert sell.slippage_bps > 0              # stopped out below where the book left
        assert bm.open_stop_exit(db, UID, "AAA") is None
        s = bm.summarize(_rows(db))
        assert s["resting_stops"]["whipsaw"]["n"] == 1
        assert s["avg_slippage_bps"] is None      # a whipsaw is not a fill-quality number

        _position(db, "AAA", 5)                   # a fresh entry mirrors normally again
        _trade(db, "AAA", "BUY", 5)
        bm.run_mirror_cycle(db, client)
        assert client.submits[-1][2] == "buy" and client.held["AAA"] == pytest.approx(5)
        assert [r.submitted_qty for r in _open_stop_rows(db)] == [pytest.approx(5)]

    def test_stop_that_fires_while_being_canceled_covers_the_sell(self, db, stops_on):
        client = _held_and_mirrored(db)
        real_cancel = client.cancel_order

        def cancel_loses_the_race(order_id):
            client.trigger("AAA", 91.9)
            real_cancel(order_id)                 # -> 422, no longer cancelable
        client.cancel_order = cancel_loses_the_race
        t = _exit_book(db, "AAA", 10, 91.0, "STOP LOSS: Down 9.0%")
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, trade_id=t.id)[0]
        assert sell.status == "covered" and sell.pairing == "same_exit"
        assert [s[2] for s in client.submits] == ["buy"]

    def test_stop_that_fires_after_a_slow_cancel_still_covers_the_sell(self, db, stops_on):
        # Cancel accepted but not yet done at the end of one tick; the stop
        # trades before the next. The waiting sell must not count as the
        # episode's close -- it IS the sell the stop covers.
        client = _held_and_mirrored(db)

        def slow_cancel(order_id):
            next(o for o in client.orders.values() if o["id"] == order_id)["status"] = "pending_cancel"
        client.cancel_order = slow_cancel
        t = _exit_book(db, "AAA", 10, 91.0, "STOP LOSS: Down 9.0%")
        bm.run_mirror_cycle(db, client)
        assert "waiting" in (_rows(db, trade_id=t.id)[0].note or "")
        client.trigger("AAA", 91.7)
        bm.run_mirror_cycle(db, client)
        sell = _rows(db, trade_id=t.id)[0]
        assert sell.status == "covered" and sell.pairing == "same_exit"
        assert [s[2] for s in client.submits] == ["buy"]

    def test_no_new_stop_while_a_sell_is_working(self, db, stops_on, _isolate):
        # A sent-but-unfilled partial sell reserves shares; a stop placed on
        # the full holding meanwhile would be refused (and alert).
        client = _held_and_mirrored(db)
        client.fill_on_submit = False
        _trade(db, "AAA", "SELL", 4, price=120.0, reason="PARTIAL TRAILING STOP (50%)")
        db.query(AIPortfolioPosition).filter_by(user_id=UID, ticker="AAA").update({"shares": 6})
        db.commit()
        for _ in range(2):
            bm.run_mirror_cycle(db, client)
        assert _open_stop_rows(db) == [] and len(client.stop_submits) == 1
        assert _isolate == []

    def test_apps_stop_firing_first_counts_as_missed(self, db, stops_on):
        client = _held_and_mirrored(db)
        _exit_book(db, "AAA", 10, 91.5, "STOP LOSS: Down 8.5%")   # broker stop never traded
        bm.run_mirror_cycle(db, client)
        assert client.submits[-1][2] == "sell"
        assert bm.summarize(_rows(db))["resting_stops"]["missed"] == 1

    def test_no_stop_at_or_through_the_market(self, db, stops_on):
        client = FakeClient(prices={"AAA": 92.1})  # stop 92.00, inside the 0.2% margin
        _held_and_mirrored(db, client=client)
        assert client.stop_submits == [] and _rows(db, action="STOP") == []

    def test_refused_level_is_quiet_and_backs_off(self, db, stops_on, _isolate):
        client = FakeClient()
        client.fail_next_stop = bm.AlpacaError(
            422, "stop price must not be greater than base price / 1.001 (91.87)")
        _held_and_mirrored(db, client=client)
        for _ in range(2):
            bm.run_mirror_cycle(db, client)
        rows = _rows(db, action="STOP")
        assert len(rows) == 1 and rows[0].status == "skipped"
        assert _isolate == []

    def test_expiry_at_the_close_is_routine_and_replaced(self, db, stops_on, _isolate):
        client = _held_and_mirrored(db)
        client.expire_all()
        bm.run_mirror_cycle(db, client)
        stops = _rows(db, ticker="AAA", action="STOP")
        assert stops[0].status == "lapsed" and "expired" in stops[0].note
        assert len(_open_stop_rows(db)) == 1 and len(client.stop_submits) == 2
        assert _isolate == [] and bm.summarize(_rows(db))["n_failed"] == 0

    def test_rejected_stop_alerts_exactly_once(self, db, stops_on, _isolate):
        client = FakeClient()
        client.fail_next_stop = bm.AlpacaError(403, "account restricted")
        _held_and_mirrored(db, client=client)
        bm.run_mirror_cycle(db, client)
        assert _rows(db, action="STOP")[0].status == "error"
        assert len(_isolate) == 1

    def test_disabling_cancels_every_working_stop(self, db, stops_on, monkeypatch):
        client = _held_and_mirrored(db)
        monkeypatch.setattr(bm, "resting_stop_config",
                            lambda: {**bm._RESTING_STOP_DEFAULTS, "enabled": False})
        bm.run_mirror_cycle(db, client)
        assert client.open_stops() == [] and _open_stop_rows(db) == []

    def test_stop_rows_stay_out_of_the_trade_stats(self, db, stops_on):
        client = _held_and_mirrored(db)
        client.expire_all()
        bm.run_mirror_cycle(db, client)
        s = bm.summarize(_rows(db))
        assert s["n_mirrored"] == 0 and s["n_open"] == 0 and s["n_skipped"] == 0
        assert s["resting_stops"]["working"] == 1


class TestHardStopLevel:
    """The level itself: the same base -> ATR -> new-position-guard chain
    both live stop paths run."""

    GUARD = {"enabled": True, "guard_days": 21, "guard_stop_pct": 8.0, "skip_if_pyramided": True}

    def _pos(self, days_held, pyramids=0, ticker="ZZZ"):
        return SimpleNamespace(
            ticker=ticker, cost_basis=50.0, current_price=48.0, pyramid_count=pyramids,
            purchase_date=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_held))

    def test_atr_widened_stop_capped_by_the_new_position_guard(self):
        from backend.trading_engine import cache_atr_stop
        cache_atr_stop("ZZZ", 12.0)
        ctx = {"base_pct": 7.0, "use_atr": True, "guard_cfg": self.GUARD}
        assert bm.hard_stop_level(self._pos(3), ctx)["stop_price"] == 46.0     # guard: 8%
        assert bm.hard_stop_level(self._pos(30), ctx)["stop_price"] == 44.0    # ATR: 12%
        assert bm.hard_stop_level(self._pos(3, pyramids=1), ctx)["stop_pct"] == 12.0

    def test_atr_off_uses_the_base(self):
        ctx = {"base_pct": 7.0, "use_atr": False, "guard_cfg": self.GUARD}
        assert bm.hard_stop_level(self._pos(30, ticker="NOATR"), ctx)["stop_price"] == 46.5

    def test_cold_cache_fetches_once_then_leaves_the_position_unstopped(self, monkeypatch):
        import backend.ai_trader as at
        calls = []
        monkeypatch.setattr(at, "calculate_atr_stop", lambda t, p, b: calls.append(t) or b)
        monkeypatch.setattr(bm, "_atr_fetch_attempts", {})
        ctx = {"base_pct": 7.0, "use_atr": True, "guard_cfg": self.GUARD}
        pos = self._pos(30, ticker="COLD")
        # A failed fetch must NOT fall back to the bare base: a stop tighter
        # than the app's would fire where the app would not.
        assert bm.hard_stop_level(pos, ctx) is None
        assert bm.hard_stop_level(pos, ctx) is None
        assert calls == ["COLD"]                  # throttled, not every minute

    def test_stop_price_floors_to_the_broker_tick(self):
        assert bm.round_stop_price(16.689) == 16.68
        assert bm.round_stop_price(100 * (1 - 0.08)) == 92.0
        assert bm.round_stop_price(17.94) == 17.94
        assert bm.round_stop_price(0.123456) == 0.1234
        assert bm._price_str(92.0) == "92.00" and bm._price_str(0.1234) == "0.1234"

    def test_stop_order_payload(self):
        sent = {}

        class Resp:
            status_code, content = 200, b"{}"

            def json(self):
                return {}
        c = bm.AlpacaPaperClient("k", "s")
        c.session.request = lambda method, url, **kw: sent.update(method=method, url=url, **kw) or Resp()
        c.submit_stop_order("AAA", 4.2, 92.0, "cs-u1-xAAA-1")
        assert sent["url"] == bm.PAPER_BASE_URL + "/v2/orders"
        assert sent["json"] == {"symbol": "AAA", "qty": "4.2", "side": "sell", "type": "stop",
                                "stop_price": "92.00", "time_in_force": "day",
                                "client_order_id": "cs-u1-xAAA-1"}


class TestRestingStopEndpoint:

    def test_status_lists_working_stops_and_keeps_them_out_of_orders(self, db, stops_on, monkeypatch):
        client = FakeClient(prices={"AAA": 96.0})
        monkeypatch.setattr(bm, "get_client", lambda: client)
        _held_and_mirrored(db, client=client)
        with override_dependency(get_current_active_user, _admin), \
             override_dependency(get_admin_user, _admin):
            body = TestClient(app).get("/api/admin/broker-mirror").json()
        [stop] = body["resting_stops"]
        assert stop["ticker"] == "AAA" and stop["stop_price"] == 92.0
        assert stop["cushion_pct"] == pytest.approx(4.17)
        assert all(o["action"] != "STOP" for o in body["orders"])
        assert body["resting_stops_enabled"] is True
        assert body["summary"]["resting_stops"]["working"] == 1
