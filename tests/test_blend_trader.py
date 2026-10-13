"""Evidence-weighted model (backend/blend_trader.py, docs/blend-model.md), its dispatch points in
ai_trader / shadow_trader, and the Lab mirrors + weekly review (backend/lab_mirror.py).
Fresh in-memory SQLite per test."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import blend_trader as bt
from backend.database import (AIPortfolioPosition, AIPortfolioTrade, Base, Canslim2Score, LabDecision,
                              LabEquityMark, LabOrder, LabStrategy, ShadowEquityMark, ShadowStrategy, ShadowTrade,
                              Stock)

DAY = date(2026, 10, 13)
PROFILE = {"engine": "blend", "max_positions": 10,
           "blend": {"buy_pct": 90, "regime_positions": {2: 10, 1: 5, 0: 0}, "sector_max": 3, "small_max": 3,
                     "max_extension_pct": 15}}


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    bt._cache.update(key=None, val={})
    try:
        yield s
    finally:
        s.close()


def _row(t, **kw):
    base = {"beat_streak": 0, "surprise_pct": 0.0, "roe": None, "s3": None, "dtc": None, "n_brokers": 0}
    return {"ticker": t, **base, **kw}


def _c2(db, t, segment="core", cap=5e9, dvol=2e7, stamp=None, **inputs):
    db.add(Canslim2Score(date=DAY, ticker=t, segment=segment, score=0.0, score_pct=50.0, rank=1, c_pct=50, a_pct=50,
                         s_pct=50, i_pct=50, market_cap=cap, in_tilt=False, scored_at=stamp or datetime(2026, 10, 13, 21),
                         inputs={**_row(t, **inputs), "dvol20": dvol}))


def _stock(db, t, sector="Tech", price=50.0, ma50=48.0, cap=5e9):
    db.add(Stock(ticker=t, name=t, sector=sector, current_price=price, ma_50=ma50, market_cap=cap))


# ---------------------------------------------------------------- score

def test_score_weights_follow_the_evidence_and_dtc_is_inverted():
    # A: best earnings (C, 35%); B: best buybacks only (10%); C: lowest days-to-cover (20%, lower = better)
    # (every row reports dtc and s3: a missing value sits at the median, as CANSLIM 2.0 treats it)
    rows = [_row("A", beat_streak=8, surprise_pct=20.0, dtc=3.0, s3=0.0), _row("B", s3=0.05, dtc=3.0),
            _row("C", dtc=0.5, s3=0.0), _row("D", dtc=9.0, s3=0.0), _row("E", dtc=3.0, s3=0.0)]
    sc = bt.score_rows(rows)
    assert sc["A"][1] > sc["C"][1] > sc["B"][1]          # 35% > 20% > 10%
    assert sc["C"][1] > sc["D"][1]                         # low short pressure beats high
    assert set(bt.WEIGHTS) == {"beat_streak", "surprise_pct", "n_brokers", "dtc", "roe", "s3"}
    assert sum(bt.WEIGHTS.values()) == pytest.approx(1.0)


def test_blend_scores_rank_within_segment_and_recompute_only_on_a_new_run(db):
    for i in range(10):
        _c2(db, f"K{i}", beat_streak=i)
        _c2(db, f"S{i}", segment="small", cap=5e8, beat_streak=i * 10)
    db.commit()
    sc = bt.blend_scores(db)
    assert sc["K9"]["pct"] == 100.0 and sc["S9"]["pct"] == 100.0       # each segment ranked on its own
    assert sc["S0"]["segment"] == "small" and sc["K0"]["pct"] < 50
    assert bt.blend_scores(db) is sc                                      # cached while the run is unchanged
    db.query(Canslim2Score).filter(Canslim2Score.ticker == "K0").update({"scored_at": datetime(2026, 10, 13, 22)})
    db.commit()
    assert bt.blend_scores(db) is not sc


# ---------------------------------------------------------------- buys

def _universe(db):
    """20 core names ranked by beat streak (K19 best) + small caps; returns nothing."""
    for i in range(20):
        _c2(db, f"K{i:02d}", beat_streak=i)
        _stock(db, f"K{i:02d}", sector=("Tech" if i % 2 else "Energy"))
    db.commit()


def test_buys_best_first_with_room_from_the_market_switch(db):
    _universe(db)
    buys = bt.blend_buys(db, 1, PROFILE, {"total_value": 25000.0}, [], exposure=2, brake=None)
    # >= 90th percentile of 20 names = K19 (100), K18 (95), K17 (90); equal dollars = 25,000 / 10
    assert [b["stock"].ticker for b in buys] == ["K19", "K18", "K17"]
    assert buys[0]["value"] == pytest.approx(2500.0) and buys[0]["stock"].canslim_score == 100.0
    assert "MODEL top 1%" in buys[0]["reason"] and buys[0]["signal_factors"]["engine"] == "blend"
    held = [SimpleNamespace(ticker=f"H{i}") for i in range(5)]
    assert bt.blend_buys(db, 1, PROFILE, {"total_value": 25000.0}, held, exposure=1, brake=None) == []   # 5/5 when mixed
    assert bt.blend_buys(db, 1, PROFILE, {"total_value": 25000.0}, [], exposure=0, brake=None) == []      # downtrend


def test_buys_skip_stretched_recently_sold_and_full_sectors(db):
    _universe(db)
    db.query(Stock).filter(Stock.ticker == "K19").update({"current_price": 60.0, "ma_50": 50.0})      # +20% stretched
    db.add(AIPortfolioTrade(user_id=1, ticker="K18", action="SELL", shares=1, price=1, total_value=1,
                            executed_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)))
    db.commit()
    prof = {**PROFILE, "blend": {**PROFILE["blend"], "buy_pct": 70}}
    held = [SimpleNamespace(ticker=t) for t in ("HT1", "HT2", "HT3")]
    for t in ("HT1", "HT2", "HT3"):
        _stock(db, t, sector="Tech")
    db.commit()
    buys = [b["stock"].ticker for b in bt.blend_buys(db, 1, prof, {"total_value": 25000.0}, held, exposure=2, brake=None)]
    assert "K19" not in buys and "K18" not in buys                       # stretched / sold 2 days ago
    assert all(int(t[1:]) % 2 == 0 for t in buys)                         # Tech already holds 3 -> Energy only
    # a sale 6 days ago no longer blocks
    db.query(AIPortfolioTrade).update({"executed_at": datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=6)})
    db.commit()
    bt._cache.update(key=None, val={})
    assert "K18" in [b["stock"].ticker for b in bt.blend_buys(db, 1, prof, {"total_value": 25000.0}, [], exposure=2, brake=None)]


def test_small_caps_band_limit_and_brake(db):
    for i in range(10):
        _c2(db, f"S{i}", segment="small", cap=5e8, dvol=3e6, beat_streak=i)
        _stock(db, f"S{i}", sector=f"Sec{i}", cap=5e8)
    _c2(db, "TINY", segment="small", cap=1.5e8, beat_streak=99)            # under $250M: outside the tested band
    _c2(db, "THIN", segment="small", cap=6e8, dvol=2e5, beat_streak=98)     # under $1M/day
    _stock(db, "TINY", cap=1.5e8)
    _stock(db, "THIN", cap=6e8)
    db.commit()
    prof = {**PROFILE, "blend": {**PROFILE["blend"], "buy_pct": 0}}
    buys = [b["stock"].ticker for b in bt.blend_buys(db, 1, prof, {"total_value": 25000.0}, [], exposure=2, brake=None)]
    assert buys == ["S9", "S8", "S7"]                                       # max 3 small caps, best first
    assert bt.blend_buys(db, 1, prof, {"total_value": 25000.0}, [], exposure=2, brake="IWM < 50MA") == []


# ---------------------------------------------------------------- market switch

def test_a5_exposure_reads_the_lab_decision_then_falls_back_to_mixed(db, monkeypatch):
    s = LabStrategy(name="a5_dual_momentum", kind="a5_dual_momentum")
    db.add(s)
    db.commit()
    db.add(LabDecision(strategy_id=s.id, date=DAY, inputs={"exposure": 2}, target={"SSO": 1.0}, status="submitted"))
    db.commit()
    assert bt.a5_exposure(db, DAY + timedelta(days=1)) == (2, f"A5 decision {DAY}")
    # stale decision + computing fails -> mixed (1): never more risk than the data supports
    from backend import lab
    monkeypatch.setattr(lab, "fmp_daily", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    bt._a5_fallback.update(day=None, exposure=None)
    assert bt.a5_exposure(db, DAY + timedelta(days=20))[0] == 1


def test_spy_sweep_follows_the_a5_gate(db, monkeypatch):
    import backend.ai_trader as ai
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                        lambda *a, **k: {"success": True, "indexes": {"SPY": {"price": 500.0, "ma_50": 520.0}}})
    monkeypatch.setattr(ai, "fetch_live_price", lambda t: 500.0)
    monkeypatch.setattr(ai, "get_portfolio_value", lambda *a, **k: {"total_value": 25000.0})
    cfg = SimpleNamespace(current_cash=20000.0, spy_sweep_shares=0.0)
    monkeypatch.setattr(bt, "sweep_allowed", lambda db: True)              # A5 >= 1 although SPY < 50MA
    ai.handle_spy_sweep(db, cfg, {"spy_sweep": {"enabled": True, "gate": "a5", "min_idle_pct": 20}})
    assert cfg.spy_sweep_shares > 0
    monkeypatch.setattr(bt, "sweep_allowed", lambda db: False)             # A5 0 -> sell the sweep
    ai.handle_spy_sweep(db, cfg, {"spy_sweep": {"enabled": True, "gate": "a5", "min_idle_pct": 20}})
    assert cfg.spy_sweep_shares == 0


# ---------------------------------------------------------------- dispatch points

def test_positions_carry_the_model_score(db, monkeypatch):
    import backend.ai_trader as ai
    from backend.database import AIPortfolioConfig
    _universe(db)
    db.add(AIPortfolioConfig(user_id=1, current_cash=0.0, starting_cash=25000.0, strategy="blend_v1"))
    for t in ("K19", "GONE"):
        db.add(AIPortfolioPosition(user_id=1, ticker=t, shares=1, cost_basis=40.0, current_price=50.0, current_value=50.0,
                                   purchase_date=datetime.now(timezone.utc)))
    _stock(db, "GONE")
    db.commit()
    ai.update_position_prices(db, use_live_prices=False, user_id=1)
    sc = {p.ticker: p.current_score for p in db.query(AIPortfolioPosition).all()}
    assert sc["K19"] == 100.0 and sc["GONE"] == 1.0                      # left the universe -> score exit


def test_evaluate_buys_dispatches_to_the_model(db, monkeypatch):
    import backend.ai_trader as ai
    from backend.database import AIPortfolioConfig
    db.add(AIPortfolioConfig(user_id=1, current_cash=25000.0, starting_cash=25000.0, strategy="blend_v1"))
    db.commit()
    seen = {}
    monkeypatch.setattr(bt, "blend_buys", lambda *a, **k: seen.setdefault("called", True) and [])
    assert ai.evaluate_buys(db, user_id=1) == [] and seen["called"]


def test_shadow_book_sizes_from_an_engine_profile_only():
    from backend.shadow_trader import ShadowSession
    ss = ShadowSession.__new__(ShadowSession)
    ss.shadow_strategy = SimpleNamespace(id=1, starting_value=25000.0, parent_strategy="blend_v1")
    assert ss._build_synthetic_config().max_positions == 10
    ss.shadow_strategy = SimpleNamespace(id=2, starting_value=25000.0, parent_strategy="nostate_cs_bear")
    assert ss._build_synthetic_config().max_positions == 8                 # running arms unchanged


def test_model_exits_are_the_ai_portfolios_exits():
    from config_loader import config
    p = config.get("strategy_profiles")
    m, live = p["blend_v1"], p["nostate_cs_bear"]
    for k in ("stop_loss_pct", "bearish_stop_loss_pct", "take_profit_pct", "trailing_stops"):
        assert m[k] == live[k], k
    assert m["engine"] == "blend" and config.get("shadow_strategy_profiles")["shadow_model_v1"]["parent_strategy"] == "blend_v1"


# ---------------------------------------------------------------- Lab mirrors + weekly review

def test_mirror_shadow_writes_marks_holdings_trades_idempotently(db):
    from backend import lab_mirror as lm
    st = ShadowStrategy(name="shadow_model_v1", parent_strategy="blend_v1", config_snapshot={}, starting_value=25000.0,
                        activated_at=datetime(2026, 10, 13, 2))
    lab_s = LabStrategy(name="model_v1", kind="mirror_shadow", starting_value=25000.0, is_active=True)
    db.add_all([st, lab_s])
    db.commit()
    db.add_all([
        ShadowTrade(shadow_strategy_id=st.id, ticker="AAA", action="BUY", shares=10, price=100.0, total_value=1000.0,
                    reason="MODEL top 1%", canslim_score=99.0, executed_at=datetime(2026, 10, 13, 15)),
        ShadowTrade(shadow_strategy_id=st.id, ticker="SPY", action="BUY", shares=2, price=500.0, total_value=1000.0,
                    reason="SPY SWEEP BUY", executed_at=datetime(2026, 10, 13, 15, 5)),
        ShadowEquityMark(shadow_strategy_id=st.id, date=DAY, equity=25100.0, cash=23000.0, positions_value=1100.0,
                         sweep_value=1000.0, n_positions=1, unpriced_positions=0)])
    db.commit()
    spy_raw, spy_adj = {DAY: 778.0}, [(DAY, 778.0)]
    closes = lambda tickers, start: {"AAA": {DAY: 110.0}, "SPY": {DAY: 500.0}}
    for _ in range(2):
        assert lm.mirror_shadow(db, lab_s, {"source": "shadow_model_v1", "start": "2026-10-13"}, spy_raw, spy_adj, closes) == 1
    m = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == lab_s.id).one()
    assert m.equity == 25100.0 and m.spy_close == 778.0
    aaa = [p for p in m.positions if p["symbol"] == "AAA"][0]
    assert aaa["market_value"] == 1100.0 and aaa["unrealized_plpc"] == pytest.approx(0.10)
    assert db.query(LabOrder).filter(LabOrder.strategy_id == lab_s.id).count() == 1     # sweep not a trade; no dupes
    dec = db.query(LabDecision).filter(LabDecision.strategy_id == lab_s.id).one()
    assert dec.inputs["kind"] == "trades" and dec.inputs["buys"][0][0] == "AAA"


def test_weekly_review_flags_the_gaps(db, monkeypatch):
    from backend import lab, lab_mirror as lm
    model = LabStrategy(name="model_v1", kind="mirror_shadow", label="Model", is_active=True)
    ai = LabStrategy(name="ai_portfolio_ref", kind="mirror_ai_portfolio", label="AI Portfolio", is_active=True)
    db.add_all([model, ai])
    db.commit()
    fri = date(2026, 10, 16)
    for s, (e0, e1), pos, inv in ((model, (25000, 25750), [{"symbol": "AAA", "market_value": 20000}], 20000),
                                  (ai, (25000, 25000), [{"symbol": "BBB", "market_value": 5000}], 5000)):
        db.add(LabEquityMark(strategy_id=s.id, date=date(2026, 10, 9), equity=e0, spy_close=770.0, spy_adj_close=770.0))
        db.add(LabEquityMark(strategy_id=s.id, date=fri, equity=e1, positions_value=inv, positions=pos,
                             spy_close=777.7, spy_adj_close=777.7))
    for i in range(2):
        db.add(LabOrder(strategy_id=ai.id, date=fri, symbol=f"X{i}", side="sell", qty=1, client_order_id=f"t{i}",
                        status="filled", reason="STOP LOSS: Down 8%"))
    db.commit()
    monkeypatch.setattr(lab, "sync_strategies", lambda d: [model, ai])
    monkeypatch.setattr(bt, "a5_exposure", lambda d, today=None: (2, "test"))
    monkeypatch.setattr(bt, "small_cap_brake", lambda md=None: None)
    rv = lm.weekly_review(db, fri)
    assert rv["status"] == "ok" and rv["books"]["model"]["week_pct"] == 3.0 and rv["books"]["ai"]["week_pct"] == 0.0
    text = " | ".join(rv["flags"])
    assert "Return gap" in text and "Invested gap" in text and "No stocks in common" in text and "AI Portfolio hit 2" in text
    title, body = lm.review_text(rv)
    assert title.startswith("Weekly review: model +3.0% vs AI Portfolio +0.0%") and "Gaps:" in body
