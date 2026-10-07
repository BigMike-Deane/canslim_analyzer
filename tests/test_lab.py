"""Lab tab (backend/lab.py + backend/routes/lab.py): A1 rule, order planning,
decide/record against a fake Alpaca client, and the read API."""
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import lab
from backend.auth import get_current_user
from backend.database import (LabDecision, LabEquityMark, LabOrder, LabStrategy, SessionLocal, User,
                              init_db)
from tests.conftest import TEST_USER_A_ID, override_dependency

init_db()

CFG = {"lab_t_a1": {"kind": "a1_trend_2x", "label": "A1 test", "enabled": True, "starting_value": 25000,
                    "index": "^GSPC", "sma_days": 200, "risk_on": "SSO", "risk_off": "SGOV",
                    "key_env": "LAB_TEST_KEY", "secret_env": "LAB_TEST_SECRET"}}


def closes(n=260, start=100.0, step=0.1, end=date(2026, 10, 7)):
    d0 = end - timedelta(days=n)
    return [(d0 + timedelta(days=i), start + step * i) for i in range(n + 1)]  # last = `end` itself


class FakeClient:
    def __init__(self, equity=25000.0, positions=None):
        self.equity, self.pos, self.orders = equity, positions or {}, {}

    def account(self):
        mv = sum(q * 50.0 for q in self.pos.values())
        return {"equity": str(self.equity), "cash": str(self.equity - mv), "long_market_value": str(mv)}

    def positions(self):
        return [{"symbol": s, "qty": str(q), "market_value": str(q * 50.0), "avg_entry_price": "48",
                 "unrealized_plpc": "0.04"} for s, q in self.pos.items()]

    def submit_moc_order(self, symbol, qty, side, cid):
        self.orders[cid] = {"id": f"b-{cid}", "status": "accepted", "symbol": symbol, "qty": qty, "side": side}
        return self.orders[cid]

    def order_by_client_id(self, cid):
        o = self.orders[cid]
        return {"status": "filled", "filled_qty": str(o["qty"]), "filled_avg_price": "50.0",
                "filled_at": "2026-10-07T20:00:00Z"}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(lab, "lab_config", lambda: CFG)
    db = SessionLocal()
    try:
        for m in (LabOrder, LabDecision, LabEquityMark):
            db.query(m).filter(m.strategy_id.in_(db.query(LabStrategy.id).filter(LabStrategy.name == "lab_t_a1"))).delete(
                synchronize_session=False)
        db.query(LabStrategy).filter(LabStrategy.name == "lab_t_a1").delete()
        db.commit()
    finally:
        db.close()
    yield


# ---------------------------------------------------------------- rule

def test_a1_risk_on_above_sma_uses_prior_close_only():
    target, inputs = lab.a1_target(closes(), date(2026, 10, 7), CFG["lab_t_a1"])
    assert target == {"SSO": 1.0} and inputs["above"] is True
    # today's (2026-10-07) bar is excluded: the latest close used is the prior session's
    assert inputs["index_close"] == round(closes()[-2][1], 2)


def test_a1_risk_off_below_sma():
    falling = [(d, 200 - 0.2 * i) for i, (d, _) in enumerate(closes())]
    target, inputs = lab.a1_target(falling, date(2026, 10, 7), CFG["lab_t_a1"])
    assert target == {"SGOV": 1.0} and inputs["above"] is False


def test_a1_needs_enough_history():
    with pytest.raises(ValueError):
        lab.a1_target(closes(n=50), date(2026, 10, 7), CFG["lab_t_a1"])


# ---------------------------------------------------------------- planning

def test_plan_switch_sells_old_buys_new_whole_shares():
    orders = lab.plan_orders({"SGOV": 1.0}, {"SSO": 200.0}, 25000.0, {"SGOV": 100.0})
    assert orders[0] == {"symbol": "SSO", "side": "sell", "qty": 200.0}
    assert orders[1] == {"symbol": "SGOV", "side": "buy", "qty": 245}  # floor(25000 * 0.98 / 100)


def test_plan_no_dust_trades_when_already_on_target():
    assert lab.plan_orders({"SSO": 1.0}, {"SSO": 243.0}, 25000.0, {"SSO": 100.0}) == []


# ---------------------------------------------------------------- engine

def _strategy(db):
    return [s for s in lab.sync_strategies(db) if s.name == "lab_t_a1"][0]


def test_decide_without_broker_records_signal():
    db = SessionLocal()
    try:
        d = lab.decide_and_submit(db, _strategy(db), client=None, today=date(2026, 10, 7), closes=closes())
        assert d.status == "no_broker" and d.target == {"SSO": 1.0}
        assert db.query(LabOrder).filter(LabOrder.decision_id == d.id).count() == 0
    finally:
        db.close()


def test_decide_submits_moc_once_per_session_then_record_marks():
    db = SessionLocal()
    try:
        s, fc = _strategy(db), FakeClient()
        d1 = lab.decide_and_submit(db, s, fc, today=date(2026, 10, 7), closes=closes(), price_fn=lambda sym: 100.0)
        d2 = lab.decide_and_submit(db, s, fc, today=date(2026, 10, 7), closes=closes(), price_fn=lambda sym: 100.0)
        assert d1.id == d2.id and d1.status == "submitted" and len(fc.orders) == 1
        o = db.query(LabOrder).filter(LabOrder.strategy_id == s.id).one()
        assert (o.symbol, o.side, o.qty, o.status) == ("SSO", "buy", 245, "accepted")
        assert s.activated_at is not None

        fc.pos = {"SSO": 245.0}
        m = lab.record_close(db, s, fc, today=date(2026, 10, 7),
                             spy_closes=[(date(2026, 10, 7), 600.0)], spy_adj=[(date(2026, 10, 7), 598.0)])
        db.refresh(o)
        assert o.status == "filled" and o.filled_avg_price == 50.0
        assert m.equity == 25000.0 and m.spy_adj_close == 598.0 and m.positions[0]["symbol"] == "SSO"
    finally:
        db.close()


# ---------------------------------------------------------------- API

def test_api_lists_strategy_and_serves_history_and_edge():
    from backend.main import app
    db = SessionLocal()
    try:
        s = _strategy(db)
        for i, (eq, spy) in enumerate(((25000, 600), (25250, 603), (25100, 601), (25600, 606))):
            db.add(LabEquityMark(strategy_id=s.id, date=date(2026, 10, 1) + timedelta(days=i), equity=eq,
                                 spy_close=spy, spy_adj_close=spy))
        db.commit()
    finally:
        db.close()
    client = TestClient(app)
    with override_dependency(get_current_user, lambda: User(id=TEST_USER_A_ID, is_admin=False)):
        rows = client.get("/api/lab/strategies").json()
        a1 = [r for r in rows if r["name"] == "lab_t_a1"][0]
        assert a1["broker_connected"] is False and a1["days"] == 4
        assert a1["total_return_pct"] == 2.4 and a1["spy_return_pct"] == 1.0 and a1["excess_return_pct"] == 1.4
        hist = client.get("/api/lab/strategies/lab_t_a1/history").json()
        assert hist[0]["spy_value"] == 25000.0 and hist[-1]["spy_value"] == 25250.0
        edge = client.get("/api/lab/strategies/lab_t_a1/edge").json()
        assert edge["status"] == "ok" and edge["total_return_pct"] == 2.4
        assert client.get("/api/lab/strategies/nope").status_code == 404


def test_safe_error_never_leaks_api_key():
    import requests as rq
    e = rq.ConnectionError("HTTPSConnectionPool: /stable/quote?symbol=SSO&apikey=SECRET123 failed")
    assert "SECRET123" not in lab.safe_error(e) and "apikey=***" in lab.safe_error(e)
    resp = rq.Response()
    resp.status_code = 429
    assert lab.safe_error(rq.HTTPError("429 for url ...apikey=SECRET123", response=resp)) == "market data error: HTTP 429"
