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
                    "key_env": "LAB_TEST_KEY", "secret_env": "LAB_TEST_SECRET",
                    "stop_rules": {"target_share_min": 0.90, "fill_cost_review_bps": 7, "fill_cost_stop_bps": 14,
                                   "min_fills": 6, "tracking_review_pct": -1.71, "tracking_breakeven_pct": -1.71,
                                   "max_dd_review_pct": 45.7,
                                   "excess_review_pct": {63: -21.4, 126: -29.2},
                                   "excess_stop_pct": {63: -40.8, 126: -45.0}}}}


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


A5 = {"kind": "a5_dual_momentum", "sma_days": 200, "mom_days": 252, "risk_on_2x": "SSO", "risk_on_1x": "SPY",
      "risk_off": "SGOV"}


def _series(n=400, start=100.0, step=0.1, end=date(2026, 10, 7)):
    return closes(n=n, start=start, step=step, end=end)


def test_a5_both_signals_2x():
    target, inputs = lab.a5_target(_series(), date(2026, 10, 7), A5, spy_adj=_series(step=0.2), cash_adj=_series(step=0.01))
    assert target == {"SSO": 1.0} and inputs["exposure"] == 2 and inputs["momentum_positive"] is True


def test_a5_one_signal_1x_and_none_cash():
    up, down = _series(), [(d, 300 - 0.3 * i) for i, (d, _) in enumerate(_series())]
    # trend up, momentum below cash -> 1x
    t1, i1 = lab.a5_target(up, date(2026, 10, 7), A5, spy_adj=down, cash_adj=_series(step=0.01))
    assert t1 == {"SPY": 1.0} and i1["exposure"] == 1
    # trend down, momentum below cash -> cash ETF
    t0, i0 = lab.a5_target(down, date(2026, 10, 7), A5, spy_adj=down, cash_adj=_series(step=0.01))
    assert t0 == {"SGOV": 1.0} and i0["exposure"] == 0


def test_a5_needs_twelve_months():
    with pytest.raises(ValueError):
        lab.a5_target(_series(), date(2026, 10, 7), A5, spy_adj=_series(n=100), cash_adj=_series(n=100))


def test_flip_notifies_only_when_target_changes(monkeypatch):
    import backend.email_utils as eu
    sent = []
    monkeypatch.setattr(eu, "create_notification", lambda **kw: sent.append(kw) or True)
    db = SessionLocal()
    try:
        s = _strategy(db)
        lab.decide_and_submit(db, s, None, today=date(2026, 10, 6), closes=closes(end=date(2026, 10, 6)))
        lab.decide_and_submit(db, s, None, today=date(2026, 10, 7), closes=closes())   # same target: no alert
        assert sent == []
        falling = [(d, 200 - 0.2 * i) for i, (d, _) in enumerate(closes(end=date(2026, 10, 8)))]
        lab.decide_and_submit(db, s, None, today=date(2026, 10, 8), closes=falling)    # SSO -> SGOV
        assert len(sent) == 1 and sent[0]["kind"] == "lab_signal" and "SGOV" in sent[0]["title"]
        assert "Was SSO" in sent[0]["body"] and sent[0]["user_id"] == 1
    finally:
        db.close()


# ---------------------------------------------------------------- stop rules (backend/lab_checks.py)

from backend import lab_checks as lc  # noqa: E402

RULES = CFG["lab_t_a1"]["stop_rules"]
MON = date(2026, 10, 5)   # Mon Oct-5 .. Fri Oct-9 2026: all NYSE sessions


def _dec(db, s, d, status="submitted", target=None):
    row = LabDecision(strategy_id=s.id, date=d, inputs={}, target=target or {"SSO": 1.0}, status=status)
    db.add(row)
    db.commit()
    return row


def _order(db, s, d, sym="SSO", side="buy", qty=100, status="filled", px=None):
    db.add(LabOrder(strategy_id=s.id, date=d, symbol=sym, side=side, qty=qty, status=status, filled_qty=qty,
                    filled_avg_price=px, client_order_id=f"t-{s.id}-{d}-{sym}-{side}-{px}-{status}-{qty}"))
    db.commit()


def _mark(db, s, d, equity=25000.0, positions=None, spy=600.0):
    db.add(LabEquityMark(strategy_id=s.id, date=d, equity=equity, spy_close=spy, spy_adj_close=spy,
                         positions=positions if positions is not None else []))
    db.commit()


def test_m1_flags_a_session_without_a_traded_decision():
    db = SessionLocal()
    try:
        s = _strategy(db)
        assert lc.check_m1(db, s, MON)["level"] == "pending"        # no broker-backed decision yet
        _dec(db, s, MON)
        _dec(db, s, MON + timedelta(days=1), status="error")
        _dec(db, s, MON + timedelta(days=2), status="unchanged")
        c = lc.check_m1(db, s, MON + timedelta(days=2))
        assert c["level"] == "breach" and c["value"] == 1 and "2026-10-06" in c["detail"]
        assert lc.check_m1(db, s, MON)["level"] == "ok"
    finally:
        db.close()


def test_m2_open_today_is_pending_but_a_dead_order_is_a_breach():
    db = SessionLocal()
    try:
        s = _strategy(db)
        _order(db, s, MON, status="accepted")
        assert lc.check_m2(db, s, MON)["level"] == "pending"
        assert lc.check_m2(db, s, MON + timedelta(days=1))["level"] == "breach"    # never filled
        _order(db, s, MON + timedelta(days=1), sym="SGOV", status="rejected")
        c = lc.check_m2(db, s, MON + timedelta(days=1))
        assert c["level"] == "breach" and c["value"] == 2 and "rejected" in c["detail"]
    finally:
        db.close()


def test_m2_breach_noted_in_the_breach_log_is_not_an_open_bug():
    db = SessionLocal()
    try:
        s = _strategy(db)
        _order(db, s, MON, status="expired", qty=288)                       # paper partial fill, rest expired
        noted = {**RULES, "noted_breaches": {MON.isoformat(): "breach log"}}
        assert lc.check_m2(db, s, MON + timedelta(days=1))["level"] == "breach"
        c = lc.check_m2(db, s, MON + timedelta(days=1), noted)
        assert c["level"] == "ok" and c["detail"] == "every order filled except 1 noted in the breach log (2026-10-05 buy SSO (expired))"
        _order(db, s, MON + timedelta(days=1), status="expired", qty=56)    # a new, un-noted one still breaches
        c = lc.check_m2(db, s, MON + timedelta(days=2), noted)
        assert c["level"] == "breach" and c["value"] == 1 and "2026-10-06" in c["detail"]
    finally:
        db.close()


def test_c1_counts_the_filled_part_of_an_expired_order():
    db = SessionLocal()
    try:
        s = _strategy(db)
        db.query(LabOrder).filter(LabOrder.strategy_id == s.id).delete()
        db.commit()
        _order(db, s, MON, status="expired", qty=288, px=100.1)           # 288 of 344 filled at +10 bps
        _order(db, s, MON + timedelta(days=1), status="expired", qty=0, px=None)   # nothing filled: not a fill
        closes = {MON + timedelta(days=k): 100.0 for k in range(5)}
        c = lc.check_c1(db, s, RULES, lambda sym, adj: list(closes.items()))
        assert c["detail"].startswith("1 fills measured") and abs(c["value"] - 10) < 0.01
    finally:
        db.close()


def test_m3_target_share_and_stray_holdings():
    db = SessionLocal()
    try:
        s = _strategy(db)
        _dec(db, s, MON)
        _mark(db, s, MON, positions=[{"symbol": "SSO", "market_value": 24400.0}])
        assert lc.check_m3(db, s, MON, RULES)["level"] == "ok"
        _dec(db, s, MON + timedelta(days=1), target={"SGOV": 1.0})
        _mark(db, s, MON + timedelta(days=1), positions=[{"symbol": "SGOV", "market_value": 22000.0},
                                                         {"symbol": "SSO", "market_value": 2500.0}])
        c = lc.check_m3(db, s, MON + timedelta(days=1), RULES)
        assert c["level"] == "breach" and "also holds SSO" in c["detail"] and c["value"] == 0.88
    finally:
        db.close()


def test_c1_fill_cost_vs_official_close_levels():
    def run(bps, n):
        db = SessionLocal()
        try:
            s = _strategy(db)
            for m in (LabOrder,):
                db.query(m).filter(m.strategy_id == s.id).delete()
            db.commit()
            for i in range(n):
                side = "buy" if i % 2 == 0 else "sell"
                fill = 100 * (1 + bps / 1e4) if side == "buy" else 100 * (1 - bps / 1e4)   # adverse both ways
                _order(db, s, MON + timedelta(days=i % 5), side=side, px=fill, qty=10 + i)
            closes = {MON + timedelta(days=k): 100.0 for k in range(5)}
            return lc.check_c1(db, s, RULES, lambda sym, adj: list(closes.items()))
        finally:
            db.close()
    assert run(10, 5)["level"] == "pending"                       # rule needs 6 fills
    c = run(10, 6)
    assert c["level"] == "review" and abs(c["value"] - 10) < 0.01
    assert run(15, 6)["level"] == "stop"
    assert run(3, 6)["level"] == "ok"


def _prices(n, daily_ret):
    d0, out, v = date(2025, 1, 2), [], 100.0
    for i in range(n):
        out.append((d0 + timedelta(days=i), v))
        v *= 1 + daily_ret
    return out


def test_tracking_gap_matches_the_backtest_cost_model():
    spy, bil = _prices(300, 0.0004), _prices(300, 0.04 / 252)
    model = 2 * 0.0004 - (0.04 / 252 + 0.005 / 252) - 0.018 / 252
    exact = {"SPY": spy, "BIL": bil, "SSO": _prices(300, model)}
    assert abs(lc.tracking_gap(lambda sym, adj: exact[sym], 126)) < 0.01
    lagging = dict(exact, SSO=_prices(300, model - 0.03 / 252))
    assert abs(lc.tracking_gap(lambda sym, adj: lagging[sym], 126) - (-3.0)) < 0.01
    assert lc.tracking_gap(lambda sym, adj: exact[sym], 400) is None
    c = lc.check_c2(RULES, {126: -3.0, 252: -2.5})
    assert c["level"] == "review" and "STOP candidate" in c["detail"]
    assert lc.check_c2(RULES, {126: 0.6, 252: 0.5})["level"] == "ok"


class _M:
    def __init__(self, equity, spy):
        self.equity, self.spy_adj_close, self.spy_close = equity, spy, spy


def test_p1_trailing_excess_levels_and_p2_drawdown():
    flat = [_M(25000, 600) for _ in range(40)]
    assert lc.check_p1(flat, RULES)["level"] == "pending"
    down25 = [_M(25000 * (1 - 0.25 * i / 63), 600) for i in range(64)]       # -25% vs flat SPY over 63 sessions
    c = lc.check_p1(down25, RULES)
    assert c["level"] == "review" and "63d -25.0%" in c["detail"]
    down45 = [_M(25000 * (1 - 0.45 * i / 63), 600) for i in range(64)]
    assert lc.check_p1(down45, RULES)["level"] == "stop"
    up = [_M(25000 * (1 + 0.1 * i / 63), 600 * (1 + 0.05 * i / 63)) for i in range(64)]
    assert lc.check_p1(up, RULES)["level"] == "ok"
    assert lc.check_p2(down25, RULES)["level"] == "ok"
    assert lc.check_p2([_M(25000, 1), _M(13000, 1), _M(14000, 1)], RULES)["level"] == "review"   # 48% > 45.7%


def test_run_checks_stores_and_pushes_only_when_a_rule_gets_worse(monkeypatch):
    import backend.email_utils as eu
    sent = []
    monkeypatch.setattr(eu, "create_notification", lambda **kw: sent.append(kw) or True)
    flat = {"SSO": _prices(300, 0.0), "SPY": _prices(300, 0.0), "BIL": _prices(300, 0.0)}
    daily = lambda sym, adj: flat.get(sym, [])  # noqa: E731
    db = SessionLocal()
    try:
        s = _strategy(db)
        _dec(db, s, MON)
        _mark(db, s, MON, positions=[{"symbol": "SSO", "market_value": 24500.0}])
        checks = lc.run_checks(db, s, MON, daily)
        assert {c["rule"] for c in checks} == {"M1", "M2", "M3", "C1", "C2", "P1", "P2"}
        assert lc.worst(checks) in ("ok", "pending") and sent == []
        stored = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == MON).one()
        assert stored.checks == checks
        # next day: no decision recorded -> M1 breach, pushed once
        _mark(db, s, MON + timedelta(days=1), positions=[{"symbol": "SSO", "market_value": 24500.0}])
        lc.run_checks(db, s, MON + timedelta(days=1), daily)
        assert len(sent) == 1 and sent[0]["kind"] == "lab_stop_rule" and "BREACH on M1" in sent[0]["title"]
        assert sent[0]["priority"] == "high"
        lc.run_checks(db, s, MON + timedelta(days=1), daily)            # 2nd evening pass, same state: quiet
        assert len(sent) == 1
    finally:
        db.close()


def test_api_serves_latest_checks():
    from backend.main import app
    client = TestClient(app)
    with override_dependency(get_current_user, lambda: User(id=TEST_USER_A_ID, is_admin=False)):
        db = SessionLocal()
        try:
            s = _strategy(db)
        finally:
            db.close()
        assert client.get("/api/lab/strategies/lab_t_a1/checks").json() == {"as_of": None, "level": "pending",
                                                                           "checks": []}
        db = SessionLocal()
        try:
            s = _strategy(db)
            db.add(LabEquityMark(strategy_id=s.id, date=MON, equity=25000.0,
                                 checks=[{"rule": "M1", "level": "breach", "value": 1, "threshold": 0, "detail": "x"},
                                         {"rule": "P1", "level": "pending", "value": None, "threshold": None,
                                          "detail": "y"}]))
            db.commit()
        finally:
            db.close()
        body = client.get("/api/lab/strategies/lab_t_a1/checks").json()
        assert body["as_of"] == "2026-10-05" and body["level"] == "breach" and len(body["checks"]) == 2
