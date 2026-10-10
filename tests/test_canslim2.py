"""CANSLIM 2.0 (backend/canslim2.py + backend/routes/canslim2.py): input parsers, the
research scoring rule, the simulated model portfolio, and the read API."""
import math
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from backend import canslim2 as c2
from backend import lab
from backend.auth import get_current_user
from backend.database import (AIPortfolioPosition, Canslim2Input, Canslim2Score, LabDecision, LabEquityMark, LabOrder,
                              LabStrategy, Notification, SessionLocal, Stock, StockDataCache, User, Watchlist, init_db)
from tests.conftest import TEST_USER_A_ID, override_dependency

init_db()

PFX = "ZC2"
TODAY = date(2026, 10, 9)
CFG = {"lab_t_c2": {"kind": "canslim2_tilt", "label": "C2 test", "enabled": True, "starting_value": 25000},
       "lab_t_c2p": {"kind": "canslim2_picks", "label": "C2 picks test", "enabled": True, "starting_value": 10000}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(lab, "lab_config", lambda: CFG)

    def wipe():
        db = SessionLocal()
        try:
            ids = db.query(LabStrategy.id).filter(LabStrategy.name.in_(["lab_t_c2", "lab_t_c2p"]))
            for m in (LabOrder, LabDecision, LabEquityMark):
                db.query(m).filter(m.strategy_id.in_(ids)).delete(synchronize_session=False)
            db.query(LabStrategy).filter(LabStrategy.name.in_(["lab_t_c2", "lab_t_c2p"])).delete(synchronize_session=False)
            for m in (Canslim2Score, Canslim2Input, StockDataCache, Stock, Watchlist, AIPortfolioPosition):
                db.query(m).filter(m.ticker.like(f"{PFX}%")).delete(synchronize_session=False)
            db.query(Notification).filter(Notification.kind == "canslim2_move",
                                          Notification.user_id == TEST_USER_A_ID).delete(synchronize_session=False)
            db.query(Notification).filter(Notification.kind.in_(["canslim2_alarm", "canslim2_status"]),
                                          Notification.user_id == c2.OWNER_ID).delete(synchronize_session=False)
            db.query(Canslim2Score).filter(Canslim2Score.date >= date(2099, 1, 1)).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()
    wipe()
    yield
    wipe()


# ---------------------------------------------------------------- input parsers

def test_s3_is_positive_for_buybacks_and_needs_a_fresh_year_ago_quarter():
    rows = [{"date": "2026-06-30", "weightedAverageShsOutDil": 90e6},
            {"date": "2026-03-31", "weightedAverageShsOutDil": 95e6},
            {"date": "2025-06-30", "weightedAverageShsOutDil": 100e6}]
    s = c2.s3_from_income(rows, TODAY)
    assert s["s3"] == pytest.approx(-math.log(0.9)) and s["s3"] > 0 and s["shares_asof"] == date(2026, 6, 30)
    assert c2.s3_from_income(rows[:2], TODAY) is None                                  # no quarter ~1y earlier
    assert c2.s3_from_income(rows, date(2027, 3, 1)) is None                           # latest quarter stale (>200d)
    issuer = [{"date": "2026-06-30", "weightedAverageShsOut": 120e6}, {"date": "2025-07-01", "weightedAverageShsOut": 100e6}]
    assert c2.s3_from_income(issuer, TODAY)["s3"] < 0                                  # dilution -> negative


def test_n_brokers_counts_distinct_brokers_in_the_prior_365_days_only():
    g = [{"date": "2026-10-08", "gradingCompany": "Morgan Stanley"}, {"date": "2026-05-01", "gradingCompany": " morgan stanley "},
         {"date": "2026-01-02", "gradingCompany": "Needham"}, {"date": "2026-10-09", "gradingCompany": "Today Co"},
         {"date": "2025-10-01", "gradingCompany": "Too Old"}, {"date": "bad", "gradingCompany": "X"}]
    assert c2.n_brokers_from_grades(g, TODAY) == 2
    assert c2.n_brokers_from_grades([], TODAY) == 0


def test_latest_dtc_uses_only_published_settlements_and_drops_no_volume_code():
    rows = [{"symbolCode": "AAA", "settlementDate": "2026-09-15", "daysToCoverQuantity": 2.5, "currentShortPositionQuantity": 1e6},
            {"symbolCode": "AAA", "settlementDate": "2026-09-30", "daysToCoverQuantity": 3.5},   # published Oct-12: not yet
            {"symbolCode": "BBB", "settlementDate": "2026-09-15", "daysToCoverQuantity": 999.99}]
    d = c2.latest_dtc(rows, TODAY)
    assert d["AAA"] == (2.5, date(2026, 9, 15), 1e6) and d["BBB"] == (None, date(2026, 9, 15), None)
    assert c2.latest_dtc(rows, date(2026, 10, 12))["AAA"] == (3.5, date(2026, 9, 30), None)


# ---------------------------------------------------------------- scoring rule

def _u(t, cap, **kw):
    base = {"ticker": t, "market_cap": cap, "beat_streak": 0, "surprise_pct": 0.0, "roe": None, "s3": None,
            "dtc": None, "n_brokers": 0}
    return {**base, **kw}


def test_score_signs_missing_is_median_and_letters(monkeypatch):
    monkeypatch.setattr(c2, "TILT_TOP", 2)
    rows = [_u("GOOD", 3e9, beat_streak=8, surprise_pct=20, roe=0.4, s3=0.05, dtc=0.5, n_brokers=30),
            _u("MID", 2e9, beat_streak=2, surprise_pct=2, roe=0.1, s3=0.0, dtc=3.0, n_brokers=10),
            _u("BAD", 5e9, beat_streak=0, surprise_pct=-30, roe=-0.2, s3=-0.2, dtc=12.0, n_brokers=1),
            _u("BLANK", 1e9)]          # no roe / s3 / dtc -> those ranks sit at the median
    out = {r["ticker"]: r for r in c2.score_universe(rows)}
    assert out["GOOD"]["rank"] == 1 and out["BAD"]["rank"] == 4
    assert out["GOOD"]["score_pct"] == 100.0 and out["GOOD"]["c_pct"] == 100.0 and out["GOOD"]["s_pct"] == 100.0
    assert out["BAD"]["s_pct"] < out["MID"]["s_pct"]                 # high days-to-cover counts AGAINST
    assert out["BLANK"]["score"] > out["BAD"]["score"]               # missing beats clearly bad
    assert sum(1 for r in out.values() if r["in_tilt"]) == 2         # the 2 largest: BAD (5e9), GOOD (3e9)
    assert out["GOOD"]["tilt_mult"] == 2.0 and out["BAD"]["tilt_mult"] == 1.0 and out["MID"]["tilt_mult"] is None


def _seed_universe(db, n=6):
    for i in range(n):
        t = f"{PFX}{i}"
        db.add(Stock(ticker=t, name=f"Test {i}", sector="Tech", current_price=50.0, market_cap=(i + 2) * 1e9))
        db.add(StockDataCache(ticker=t, earnings_beat_streak=i, latest_surprise_pct=float(i), roe=0.05 * i))   # i=0 -> roe 0.0 = missing
        db.add(Canslim2Input(ticker=t, s3=0.01 * i, dtc=10.0 - i, n_brokers=i * 3, dvol20=20e6))
    db.add(Stock(ticker=f"{PFX}THIN", current_price=50.0, market_cap=9e9))
    db.add(Canslim2Input(ticker=f"{PFX}THIN", dvol20=1e6))               # fails the $5M dollar-volume screen
    db.add(Stock(ticker=f"{PFX}PENNY", current_price=3.0, market_cap=9e9))
    db.add(Canslim2Input(ticker=f"{PFX}PENNY", dvol20=50e6))             # fails the $5 price screen
    db.commit()


def test_compute_scores_applies_the_universe_screen_and_replaces_the_day(monkeypatch):
    db = SessionLocal()
    try:
        _seed_universe(db)
        rows = c2.universe_rows(db)
        mine = {r["ticker"] for r in rows if r["ticker"].startswith(PFX)}
        assert f"{PFX}THIN" not in mine and f"{PFX}PENNY" not in mine and f"{PFX}5" in mine
        by = {r["ticker"]: r for r in rows}
        assert by[f"{PFX}0"]["roe"] is None and by[f"{PFX}1"]["roe"] == pytest.approx(0.05)   # FMP's 0 placeholder = missing
        monkeypatch.setattr(c2, "universe_rows", lambda db, segment="core": [r for r in rows if r["ticker"].startswith(PFX)] if segment == "core" else [])
        day = date(2099, 1, 2)
        assert c2.compute_scores(db, day) == 6
        assert c2.compute_scores(db, day) == 6                           # re-run replaces, no duplicates
        best = db.query(Canslim2Score).filter(Canslim2Score.date == day).order_by(Canslim2Score.rank).first()
        assert best.ticker == f"{PFX}5" and best.inputs["dtc"] == 5.0
        w = c2.tilt_weights(db, day)
        assert sum(w.values()) == pytest.approx(1.0) and w[f"{PFX}5"] == max(w.values())
    finally:
        db.close()


# ---------------------------------------------------------------- model portfolio

def _strategy(db):
    return [s for s in lab.sync_strategies(db) if s.name == "lab_t_c2"][0]


def test_model_starts_drifts_and_rebalances_every_20_sessions():
    db = SessionLocal()
    try:
        s = _strategy(db)
        weights = {"A": {"X": 0.5, "Y": 0.5}, "B": {"X": 1.0}}
        state = {"w": weights["A"]}
        wf = lambda db, day: state["w"]
        d0 = date(2026, 10, 1)
        spy = lambda d: ([(d, 600.0)], [(d - timedelta(days=1), 600.0), (d, 600.0)])
        m0 = c2.mark_model(db, s, d0, lambda t, st: {}, *spy(d0), weights_fn=wf)
        assert m0.equity == pytest.approx(25000 * (1 - 0.0019))           # opening buy pays full turnover
        assert db.query(LabDecision).filter(LabDecision.strategy_id == s.id).count() == 1
        assert c2.mark_model(db, s, d0, lambda t, st: {}, *spy(d0), weights_fn=wf) is None   # idempotent
        # day 1: X +10%, Y -10% -> equity flat, weights drift
        d1 = d0 + timedelta(days=1)
        px = lambda t, st: {"X": {d0: 10.0, d1: 11.0}, "Y": {d0: 20.0, d1: 18.0}}
        m1 = c2.mark_model(db, s, d1, px, *spy(d1), weights_fn=wf)
        assert m1.equity == pytest.approx(m0.equity)
        assert {p["symbol"]: p["weight"] for p in m1.positions} == pytest.approx({"X": 0.55, "Y": 0.45})
        # sessions 2..19 flat, no rebalance; session 20 moves to B and pays on the turnover
        d, prev = d1, d1
        for i in range(2, 21):
            d = d0 + timedelta(days=i)
            if i == 20:
                state["w"] = weights["B"]
            pxi = (lambda p, q: (lambda t, st: {"X": {p: 11.0, q: 11.0}, "Y": {p: 18.0, q: 18.0}}))(prev, d)
            m = c2.mark_model(db, s, d, pxi, *spy(d), weights_fn=wf)
            prev = d
        decs = db.query(LabDecision).filter(LabDecision.strategy_id == s.id).order_by(LabDecision.date).all()
        assert [x.date for x in decs] == [d0, d0 + timedelta(days=20)]
        assert decs[1].inputs["turnover"] == pytest.approx(0.45) and decs[1].inputs["sessions_since_last"] == 20
        assert m.equity == pytest.approx(m1.equity * (1 - 0.45 * 0.0019))
        assert [p["symbol"] for p in m.positions] == ["X"]
    finally:
        db.close()


def test_model_skips_the_mark_when_closes_are_missing():
    db = SessionLocal()
    try:
        s = _strategy(db)
        d0, d1 = date(2026, 10, 1), date(2026, 10, 2)
        wf = lambda db, day: {"X": 0.5, "Y": 0.5}
        c2.mark_model(db, s, d0, lambda t, st: {}, [(d0, 600.0)], [(d0, 600.0)], weights_fn=wf)
        assert c2.mark_model(db, s, d1, lambda t, st: {"X": {d0: 10.0, d1: 11.0}}, [(d1, 600.0)],
                             [(d0, 600.0), (d1, 600.0)], weights_fn=wf) is None     # 1 of 2 missing (> 20%)
        assert c2.mark_model(db, s, d1, lambda t, st: {}, [], [], weights_fn=lambda db, day: {}) is None
    finally:
        db.close()


def test_decision_job_never_runs_a_broker_rule_for_the_simulated_kind():
    assert "canslim2_tilt" not in lab.RULES


# ---------------------------------------------------------------- API

def test_api_meta_top_and_stock():
    from backend.main import app
    db = SessionLocal()
    try:
        _seed_universe(db)
        day = date(2099, 1, 3)
        rows = [r for r in c2.universe_rows(db) if r["ticker"].startswith(PFX)]
        for r in c2.score_universe(rows):
            db.add(Canslim2Score(date=day, ticker=r["ticker"], score=r["score"], score_pct=r["score_pct"],
                                 rank=r["rank"], c_pct=r["c_pct"], a_pct=r["a_pct"], s_pct=r["s_pct"],
                                 i_pct=r["i_pct"], market_cap=r["market_cap"], in_tilt=r["in_tilt"],
                                 tilt_mult=r["tilt_mult"], inputs={"dtc": r["dtc"]}))
        db.commit()
    finally:
        db.close()
    client = TestClient(app)
    with override_dependency(get_current_user, lambda: User(id=TEST_USER_A_ID, is_admin=False)):
        meta = client.get("/api/canslim2/meta").json()
        assert meta["as_of"] == "2099-01-03" and meta["universe"] == 6 and set(meta["letters"]) == {"C", "A", "S", "I"}
        assert {s["key"] for s in meta["signals"]} == {"beat_streak", "surprise_pct", "roe", "s3", "dtc", "n_brokers"}
        top = client.get("/api/canslim2/top", params={"limit": 3}).json()
        assert [s["ticker"] for s in top["stocks"]] == [f"{PFX}5", f"{PFX}4", f"{PFX}3"]
        assert top["stocks"][0]["name"] == "Test 5" and set(top["stocks"][0]["letters"]) == {"C", "A", "S", "I"}
        bottom = client.get("/api/canslim2/top", params={"limit": 1, "bottom": True}).json()
        assert bottom["stocks"][0]["ticker"] == f"{PFX}0"
        one = client.get(f"/api/canslim2/stock/{PFX.lower()}5").json()
        assert one["rank"] == 1 and one["universe"] == 6
        assert client.get(f"/api/canslim2/stock/{PFX}THIN").status_code == 404


# ---------------------------------------------------------------- 20-stock picks (pre-registered rules)

def _picks(db):
    return [s for s in lab.sync_strategies(db) if s.name == "lab_t_c2p"][0]


def test_picks_buy_top_decile_with_sector_cap_then_sell_on_score_stop_and_exit(monkeypatch):
    monkeypatch.setattr(c2, "PICKS_N", 3)
    monkeypatch.setattr(c2, "PICKS_SECTOR_MAX", 2)
    db = SessionLocal()
    try:
        s = _picks(db)
        d0, d1 = date(2026, 10, 12), date(2026, 10, 13)
        day0 = [{"ticker": "A", "score_pct": 99.0, "sector": "Tech"}, {"ticker": "B", "score_pct": 98.0, "sector": "Tech"},
                {"ticker": "C", "score_pct": 97.0, "sector": "Tech"},      # 3rd Tech: sector cap -> skipped
                {"ticker": "D", "score_pct": 95.0, "sector": "Energy"}, {"ticker": "E", "score_pct": 91.0, "sector": "Energy"},
                {"ticker": "F", "score_pct": 50.0, "sector": "Energy"}]
        raw = lambda t, st: {k: {d0: 10.0, d1: 10.0} for k in "ABCDEFG"}
        spy = ([(d0, 600.0)], [(d0, 600.0)])
        m0 = c2.mark_picks(db, s, d0, lambda t, st: {}, raw, *spy, scores_fn=lambda db, d: day0)
        assert sorted(p["symbol"] for p in m0.positions) == ["A", "B", "D"]
        assert m0.cash == pytest.approx(10000 - 3 * 10000 / 3) and m0.equity == pytest.approx(10000 * (1 - 0.001))
        buys = db.query(LabOrder).filter(LabOrder.strategy_id == s.id, LabOrder.side == "buy").all()
        assert len(buys) == 3 and all(o.status == "filled" and o.filled_avg_price == 10.0 for o in buys)
        assert buys[0].filled_qty == pytest.approx(10000 / 3 / 10.0)
        dec = db.query(LabDecision).filter(LabDecision.strategy_id == s.id).one()
        assert dec.inputs["kind"] == "trades" and len(dec.inputs["buys"]) == 3
        assert c2.mark_picks(db, s, d0, lambda t, st: {}, raw, *spy, scores_fn=lambda db, d: day0) is None   # idempotent
        # day 1: A drops to 65th pct (sell), B -20% (stop), D gone from the universe; C/E/G refill
        day1 = [{"ticker": "C", "score_pct": 99.0, "sector": "Tech"}, {"ticker": "B", "score_pct": 96.0, "sector": "Tech"},
                {"ticker": "G", "score_pct": 94.0, "sector": "Health"}, {"ticker": "E", "score_pct": 92.0, "sector": "Energy"},
                {"ticker": "A", "score_pct": 65.0, "sector": "Tech"}]
        adj = lambda t, st: {"A": {d0: 10.0, d1: 11.0}, "B": {d0: 10.0, d1: 8.0}, "D": {d0: 10.0, d1: 10.0}}
        m1 = c2.mark_picks(db, s, d1, adj, raw, [(d1, 600.0)], [(d0, 600.0), (d1, 600.0)], scores_fn=lambda db, d: day1)
        sells = {o.symbol: o.reason for o in db.query(LabOrder).filter(LabOrder.strategy_id == s.id, LabOrder.side == "sell")}
        assert set(sells) == {"A", "B", "D"}
        assert sells["A"].startswith("score fell to 65") and sells["B"].startswith("stop") and sells["D"].startswith("left")
        assert sorted(p["symbol"] for p in m1.positions) == ["C", "E", "G"]          # B (sold today) not re-bought
        slot = 10000 / 3 * 0.999
        proceeds = (slot * 1.1 + slot * 0.8 + slot) * 0.999
        assert m1.equity == pytest.approx(proceeds * (1 - 0.001 * 3 / 3), rel=1e-3)
    finally:
        db.close()


def test_picks_skip_the_mark_without_scores_or_closes():
    db = SessionLocal()
    try:
        s = _picks(db)
        d0, d1 = date(2026, 10, 12), date(2026, 10, 13)
        assert c2.mark_picks(db, s, d0, lambda t, st: {}, lambda t, st: {}, [], [], scores_fn=lambda db, d: []) is None
        top = [{"ticker": f"T{i}", "score_pct": 99.0 - i * 0.1, "sector": f"S{i}"} for i in range(20)]
        c2.mark_picks(db, s, d0, lambda t, st: {}, lambda t, st: {}, [(d0, 600.0)], [(d0, 600.0)], scores_fn=lambda db, d: top)
        assert c2.mark_picks(db, s, d1, lambda t, st: {}, lambda t, st: {}, [], [], scores_fn=lambda db, d: top) is None
    finally:
        db.close()


# ---------------------------------------------------------------- rescoring, movers, alerts, Screener / Watchlist

def _score_rows(db, day, pcts):
    for t, p in pcts.items():
        db.add(Canslim2Score(date=day, ticker=t, score=p / 100 - 0.5, score_pct=p, rank=1, c_pct=p, a_pct=p, s_pct=p,
                             i_pct=p, market_cap=2e9, in_tilt=False, inputs={"beat_streak": 3, "surprise_pct": 5.0}))
    db.commit()


def test_rescore_keeps_the_previous_days_percentile(monkeypatch):
    db = SessionLocal()
    try:
        _seed_universe(db)
        rows = [r for r in c2.universe_rows(db) if r["ticker"].startswith(PFX)]
        monkeypatch.setattr(c2, "universe_rows", lambda db, segment="core": [dict(r) for r in rows] if segment == "core" else [])
        d1, d2 = date(2099, 2, 1), date(2099, 2, 2)
        c2.compute_scores(db, d1)
        c2.compute_scores(db, d2)
        r = db.query(Canslim2Score).filter(Canslim2Score.date == d2, Canslim2Score.ticker == f"{PFX}5").one()
        assert r.prev_score_pct == r.score_pct == 100.0 and r.scored_at is not None
    finally:
        db.close()


def test_score_move_alerts_once_per_user_ticker_for_held_or_watched_only():
    db = SessionLocal()
    try:
        day = date(2099, 3, 1)
        _score_rows(db, day, {f"{PFX}HELD": 92.0, f"{PFX}WATCH": 30.0, f"{PFX}OTHER": 95.0, f"{PFX}SMALL": 55.0})
        db.add(AIPortfolioPosition(user_id=TEST_USER_A_ID, ticker=f"{PFX}HELD", shares=1, cost_basis=10))
        db.add(Watchlist(user_id=TEST_USER_A_ID, ticker=f"{PFX}WATCH"))
        db.add(Watchlist(user_id=TEST_USER_A_ID, ticker=f"{PFX}SMALL"))
        db.commit()
        before = {f"{PFX}HELD": 60.0, f"{PFX}WATCH": 70.0, f"{PFX}OTHER": 10.0, f"{PFX}SMALL": 50.0}
        sent = []

        def notify(**kw):
            sent.append(kw)
            db.add(Notification(user_id=kw["user_id"], kind=kw["kind"], title=kw["title"], body=kw["body"], data=kw["data"]))
            db.commit()
        assert c2.score_move_alerts(db, before, notify=notify) == 2       # OTHER not held/watched, SMALL moved 5
        assert {x["data"]["ticker"] for x in sent} == {f"{PFX}HELD", f"{PFX}WATCH"}
        assert "60 → 92" in [x for x in sent if x["data"]["ticker"] == f"{PFX}HELD"][0]["title"]
        assert c2.score_move_alerts(db, before, notify=notify) == 0       # once per ticker per day
    finally:
        db.close()


def test_screener_sorts_by_canslim2_and_watchlist_shows_it():
    from backend.main import app
    db = SessionLocal()
    try:
        for i, p in enumerate((40.0, 99.0, 70.0)):
            db.add(Stock(ticker=f"{PFX}S{i}", name=f"S{i}", current_price=20.0, market_cap=5e9, canslim_score=50 + i,
                         last_updated=__import__("datetime").datetime.now()))
        db.commit()
        _score_rows(db, date(2099, 4, 1), {f"{PFX}S0": 40.0, f"{PFX}S1": 99.0, f"{PFX}S2": 70.0})
        db.add(Watchlist(user_id=TEST_USER_A_ID, ticker=f"{PFX}S1"))
        db.commit()
    finally:
        db.close()
    from backend.auth import get_current_active_user
    client = TestClient(app)
    u = lambda: User(id=TEST_USER_A_ID, is_admin=False, is_active=True)
    with override_dependency(get_current_user, u), override_dependency(get_current_active_user, u):
        rows = client.get("/api/stocks", params={"sort_by": "canslim2", "limit": 200}).json()["stocks"]
        mine = [r["ticker"] for r in rows if r["ticker"].startswith(PFX)]
        assert mine == [f"{PFX}S1", f"{PFX}S2", f"{PFX}S0"]
        assert rows[[r["ticker"] for r in rows].index(f"{PFX}S1")]["canslim2"]["score_pct"] == 99.0
        wl = client.get("/api/watchlist").json()["items"]
        assert [w["canslim2"]["score_pct"] for w in wl if w["ticker"] == f"{PFX}S1"] == [99.0]
        mv = client.get("/api/canslim2/top", params={"movers": "up", "limit": 5}).json()
        assert "stocks" in mv


# ---------------------------------------------------------------- provisional small caps

def test_small_caps_are_ranked_among_themselves_labelled_and_never_traded(monkeypatch):
    from backend import canslim2_trader as ct
    db = SessionLocal()
    try:
        _seed_universe(db)                                   # 6 core names (+ THIN: $9B but $1M/day)
        for i, cap in enumerate((3e8, 6e8, 1.5e8)):          # two confirmed-band small caps, one micro
            t = f"{PFX}SM{i}"
            db.add(Stock(ticker=t, name=t, sector="Banks", current_price=30.0, market_cap=cap))
            db.add(StockDataCache(ticker=t, earnings_beat_streak=10 + i, latest_surprise_pct=50.0, roe=0.3))
            db.add(Canslim2Input(ticker=t, s3=0.05, dtc=1.0, n_brokers=1, dvol20=2e6 if i < 2 else 5e5))
        db.commit()
        core = {r["ticker"] for r in c2.universe_rows(db, "core") if r["ticker"].startswith(PFX)}
        small = {r["ticker"] for r in c2.universe_rows(db, "small") if r["ticker"].startswith(PFX)}
        assert core == {f"{PFX}{i}" for i in range(6)}
        assert small == {f"{PFX}SM0", f"{PFX}SM1", f"{PFX}SM2", f"{PFX}THIN"}   # PENNY: price < $5
        rows_c = [r for r in c2.universe_rows(db, "core") if r["ticker"].startswith(PFX)]
        rows_s = [r for r in c2.universe_rows(db, "small") if r["ticker"].startswith(PFX)]
        monkeypatch.setattr(c2, "universe_rows", lambda db, segment="core": [dict(r) for r in (rows_c if segment == "core" else rows_s)])
        day = date(2099, 5, 1)
        assert c2.compute_scores(db, day) == 10
        got = {r.ticker: r for r in db.query(Canslim2Score).filter(Canslim2Score.date == day)}
        assert {t for t, r in got.items() if r.segment == "small"} == small
        assert all(not r.in_tilt for r in got.values() if r.segment == "small")
        assert sorted(r.rank for r in got.values() if r.segment == "small") == [1, 2, 3, 4]   # ranked among themselves
        assert "confirmed for $250M-$1B" in c2.segment_label("small", 3e8, 2e6)
        assert "untested below $250M" in c2.segment_label("small", 1.5e8, 5e5) and "thinly traded" in c2.segment_label("small", 1.5e8, 5e5)
        assert c2.segment_label("core", 5e9, 1e8) is None
        # no trading path sees a small cap
        assert not small & {r["ticker"] for r in c2.picks_universe(db, day)}
        assert not small & set(c2.tilt_weights(db, day))
        assert not small & set(ct.c2_pcts(db))
        prof = {"engine": "canslim2_picks", "max_positions": 20, "canslim2": {"buy_pct": 0, "sector_max": 20}}
        assert not small & {b["stock"].ticker for b in ct.engine_buys(db, TEST_USER_A_ID, prof, {"total_value": 1e5}, [])}
    finally:
        db.close()


# ---------------------------------------------------------------- mechanics alarms (pre-registered)

def test_mechanics_alarms_missing_mark_first_trades_and_turnover(monkeypatch):
    db = SessionLocal()
    try:
        tilt, picks = _strategy(db), _picks(db)
        sent = []

        def notify(**kw):
            sent.append(kw)
            db.add(Notification(user_id=kw["user_id"], kind=kw["kind"], title=kw["title"], body=kw["body"]))
            db.commit()
        d0, d1 = date(2026, 10, 12), date(2026, 10, 13)
        # day 0: picks marked (first mark), tilt missed
        db.add(LabEquityMark(strategy_id=picks.id, date=d0, equity=10000.0, cash=0.0, positions=[{"symbol": "A"}]))
        db.add(LabOrder(strategy_id=picks.id, date=d0, symbol="A", side="buy", qty=1000, status="filled", filled_qty=1000,
                        filled_avg_price=10.0, client_order_id="t-open"))
        db.commit()
        got = c2.check_mechanics(db, d0, notify=notify)
        assert any("C2 test: no mark" in t for t in got) and any(t == "C2 picks test started" for t in got)
        assert [x["priority"] for x in sent if "started" in x["title"]] == ["default"]
        assert c2.check_mechanics(db, d0, notify=notify) == []          # deduped
        # day 1: picks churned 1.5x the book after the opening buys -> turnover alarm; no second "started"
        db.add(LabEquityMark(strategy_id=picks.id, date=d1, equity=10000.0, cash=0.0, positions=[]))
        db.add(LabEquityMark(strategy_id=tilt.id, date=d1, equity=25000.0, cash=0.0, positions=[{"symbol": "X"}]))
        for i, side in enumerate(("sell", "buy", "sell")):
            db.add(LabOrder(strategy_id=picks.id, date=d1, symbol=f"T{i}", side=side, qty=1000, status="filled",
                            filled_qty=1000, filled_avg_price=10.0, client_order_id=f"t-churn-{i}"))
        db.commit()
        got = c2.check_mechanics(db, d1, notify=notify)
        assert got == ["C2 test started", "C2 picks test: turnover 150% in 30 days"]
    finally:
        db.close()


def test_stock_page_short_interest_prefers_finra_and_labels_the_fallback():
    from types import SimpleNamespace
    db = SessionLocal()
    try:
        db.add(Canslim2Input(ticker=f"{PFX}SI", dtc=9.6, dtc_settle=date(2026, 9, 15), short_shares=2e5, shares_now=6.9e6))
        db.commit()
        old = SimpleNamespace(ticker=f"{PFX}SI", short_interest_pct=1.3, short_ratio=4.3,
                              short_updated_at=__import__("datetime").datetime(2026, 7, 22, 21, 31))
        d = c2.short_display(db, old)
        assert d["short_ratio"] == 9.6 and d["short_interest_pct"] == pytest.approx(2.9, abs=0.01)
        assert d["short_basis"] == "shares" and d["short_source"] == "FINRA, settled 2026-09-15"
        none = SimpleNamespace(ticker=f"{PFX}NOPE", short_interest_pct=1.3, short_ratio=4.3,
                               short_updated_at=__import__("datetime").datetime(2026, 7, 22, 21, 31))
        f = c2.short_display(db, none)
        assert f["short_ratio"] == 4.3 and f["short_basis"] == "float" and f["short_source"] == "Yahoo, 2026-07-22"
    finally:
        db.close()
