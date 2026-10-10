"""AI Portfolio engine `canslim2_picks` (backend/canslim2_trader.py) and its dispatch points in
ai_trader / broker_mirror / exit_plan / backtester. Pre-registered: docs/canslim2-forward-plan.md
(Strategy 3). Fresh in-memory SQLite per test."""
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import canslim2_trader as ct
from backend.database import (AIPortfolioConfig, AIPortfolioPosition, AIPortfolioTrade, Base, Canslim2Score,
                              Stock)

ENGINE_PROFILE = {"engine": "canslim2_picks", "max_positions": 3,
                  "canslim2": {"buy_pct": 90, "sell_pct": 70, "stop_pct": 15, "sector_max": 2}}
DAY = date(2026, 10, 22)


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    try:
        yield s
    finally:
        s.close()


def _score(db, t, pct, sector="Tech", price=50.0, rank=1):
    db.add(Stock(ticker=t, name=t, sector=sector, current_price=price, market_cap=5e9, canslim_score=60.0))
    db.add(Canslim2Score(date=DAY, ticker=t, score=pct / 100 - 0.5, score_pct=pct, rank=rank, c_pct=pct, a_pct=pct,
                         s_pct=pct, i_pct=pct, market_cap=5e9, in_tilt=False))


def _pos(db, t, cost=50.0, price=50.0, user_id=1):
    p = AIPortfolioPosition(user_id=user_id, ticker=t, shares=10, cost_basis=cost, current_price=price,
                            current_value=10 * price, gain_loss=10 * (price - cost), purchase_date=datetime.now(timezone.utc))
    db.add(p)
    return p


# ---------------------------------------------------------------- decisions

def test_engine_sells_on_score_stop_and_leaving_the_universe(db):
    for t, pct in (("KEEP", 85.0), ("FADE", 65.0), ("DROP", 95.0)):
        _score(db, t, pct)
    keep, fade, drop, gone = _pos(db, "KEEP"), _pos(db, "FADE"), _pos(db, "DROP", cost=50, price=42.4), _pos(db, "GONE")
    db.commit()
    out = {s["position"].ticker: s["reason"] for s in ct.engine_sells(db, [keep, fade, drop, gone], ENGINE_PROFILE)}
    assert set(out) == {"FADE", "DROP", "GONE"}
    assert "score fell to 65" in out["FADE"] and "STOP" in out["DROP"] and "not scored" in out["GONE"]
    assert ct.engine_sells(db, [keep], ENGINE_PROFILE) == []
    # no scores at all (fresh install): hold everything rather than dump the book
    db.query(Canslim2Score).delete()
    db.commit()
    assert ct.engine_sells(db, [fade, gone], ENGINE_PROFILE) == []


def test_engine_buys_best_first_with_sector_cap_room_and_no_same_day_rebuy(db):
    for i, (t, pct, sec) in enumerate((("A", 99.0, "Tech"), ("B", 98.0, "Tech"), ("C", 97.0, "Tech"),
                                       ("D", 96.0, "Energy"), ("E", 92.0, "Health"), ("LOW", 80.0, "Health"))):
        _score(db, t, pct, sec, rank=i + 1)
    db.add(AIPortfolioTrade(user_id=1, ticker="D", action="SELL", shares=1, price=1, total_value=1,
                            executed_at=datetime.now(timezone.utc)))
    held = _pos(db, "H")
    db.add(Stock(ticker="H", sector="Tech", current_price=50.0))
    db.commit()
    buys = ct.engine_buys(db, 1, ENGINE_PROFILE, {"total_value": 30000.0}, [held])
    # room = 3 - 1 held = 2; H is Tech so only one more Tech fits (cap 2); D was sold today
    assert [b["stock"].ticker for b in buys] == ["A", "E"]
    assert buys[0]["value"] == pytest.approx(10000.0) and buys[0]["stock"].canslim_score == 99.0
    assert buys[0]["signal_factors"]["engine"] == "canslim2_picks" and "CANSLIM 2.0 top 1%" in buys[0]["reason"]


def test_engine_stop_job_sells_only_at_the_fixed_stop_in_market_hours(db, monkeypatch):
    import backend.ai_trader as ai
    monkeypatch.setattr(ai, "is_market_open", lambda: True)
    monkeypatch.setattr(ai, "take_portfolio_snapshot", lambda *a, **k: None)
    trades = []
    monkeypatch.setattr(ai, "execute_trade", lambda **kw: trades.append(kw))
    cfg = AIPortfolioConfig(user_id=1, current_cash=1000.0, starting_cash=25000.0, strategy="x")
    db.add(cfg)
    down14, down16 = _pos(db, "OK", cost=100, price=86.0), _pos(db, "HIT", cost=100, price=84.0)
    db.commit()
    res = ct.engine_stop_check(db, 1, ENGINE_PROFILE, cfg, [down14, down16])
    assert [s["ticker"] for s in res["sells_executed"]] == ["HIT"] and trades[0]["action"] == "SELL"
    assert cfg.current_cash == pytest.approx(1000.0 + 840.0)
    assert [p.ticker for p in db.query(AIPortfolioPosition).all()] == ["OK"]
    monkeypatch.setattr(ai, "is_market_open", lambda: False)
    assert ct.engine_stop_check(db, 1, ENGINE_PROFILE, cfg, [down14])["sells_executed"] == []


# ---------------------------------------------------------------- the dated switch

def test_pivot_flips_real_champion_portfolios_once_on_the_date(db, monkeypatch):
    monkeypatch.setattr(ct, "_pivot_cfg", lambda: {"activate_on": "2026-10-22", "strategy": "canslim2_picks_live",
                                                   "from_strategies": ["nostate_cs_bear"]})
    for uid, strat in ((1, "nostate_cs_bear"), (2, "nostate_cs_bear"), (4, "something_else"), (-1, "nostate_cs_bear")):
        db.add(AIPortfolioConfig(user_id=uid, current_cash=1.0, starting_cash=1.0, strategy=strat))
    db.commit()
    sent = []
    notify = lambda **kw: sent.append(kw)
    assert ct.apply_pivot_if_due(db, today=date(2026, 10, 21), notify=notify) == 0
    assert ct.apply_pivot_if_due(db, today=date(2026, 10, 22), notify=notify) == 2
    got = {c.user_id: c.strategy for c in db.query(AIPortfolioConfig).all()}
    assert got == {1: "canslim2_picks_live", 2: "canslim2_picks_live", 4: "something_else", -1: "nostate_cs_bear"}
    assert ct.apply_pivot_if_due(db, today=date(2026, 10, 23), notify=notify) == 0 and len(sent) == 1
    monkeypatch.setattr(ct, "pivot_active", lambda today=None: True)
    assert ct.pivot_default_strategy("nostate_cs_bear") == "canslim2_picks_live"


def test_pivot_is_off_in_the_test_suite_by_default():
    assert ct._pivot_cfg() == {} and ct.pivot_default_strategy("nostate_cs_bear") == "nostate_cs_bear"


# ---------------------------------------------------------------- stop surfaces agree

def test_exit_plan_and_broker_stop_use_the_fixed_stop(db, monkeypatch):
    from backend import exit_plan, broker_mirror
    import backend.trading_utils as tu
    real = tu.get_strategy_profile
    monkeypatch.setattr(exit_plan, "get_strategy_profile", lambda s: ENGINE_PROFILE if s == "c2" else real(s))
    plan = exit_plan.compute_exit_plan(cost_basis=100.0, current_price=110.0, peak_price=130.0, current_score=82.0,
                                       strategy="c2", sell_score_threshold=45, stop_loss_pct=7.0)
    kinds = {t["kind"]: t for t in plan["triggers"]}
    assert set(kinds) == {"stop_loss", "score_exit"} and kinds["stop_loss"]["price"] == 85.0
    assert kinds["score_exit"]["threshold"] == 70 and "now 82" in kinds["score_exit"]["note"]
    db.add(AIPortfolioConfig(user_id=1, current_cash=1.0, starting_cash=1.0, strategy="c2"))
    db.commit()
    monkeypatch.setattr(tu, "get_strategy_profile", lambda s: ENGINE_PROFILE if s == "c2" else real(s))
    ctx = broker_mirror.stop_context(db, 1)
    lvl = broker_mirror.hard_stop_level(_pos(db, "Z", cost=100.0, price=110.0), ctx)
    assert ctx["use_atr"] is False and lvl["stop_price"] == 85.0


def test_backtester_refuses_the_engine(db, monkeypatch):
    import backend.backtester as bt
    from backend.database import BacktestRun
    run = BacktestRun(strategy="c2", starting_cash=25000.0, start_date=date(2025, 1, 2), end_date=date(2025, 6, 30))
    db.add(run)
    db.commit()
    monkeypatch.setattr(bt, "get_strategy_profile", lambda s: ENGINE_PROFILE)
    with pytest.raises(ValueError, match="cannot replay"):
        bt.BacktestEngine(db, run.id)


# ---------------------------------------------------------------- one full live cycle on the engine

def test_full_trading_cycle_runs_the_engine_end_to_end(db, monkeypatch):
    import backend.ai_trader as ai
    import backend.email_utils as eu
    import data_fetcher
    real = ai.get_strategy_profile
    monkeypatch.setattr(ai, "get_strategy_profile", lambda s: ENGINE_PROFILE if s == "c2" else real(s))
    monkeypatch.setattr(ai, "fetch_live_price", lambda t: 40.0)
    monkeypatch.setattr(ai, "stale_quote_reason", lambda *a, **k: None)
    monkeypatch.setattr(ai, "HistoricalDataProvider", None)
    monkeypatch.setattr(ai, "take_portfolio_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(ai, "get_market_regime", lambda *a, **k: {"regime": "bearish"})   # classic reserve would be 40-60%
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                        lambda *a, **k: {"success": True, "weighted_signal": -2.0,
                                         "indexes": {"SPY": {"price": 400.0, "ma_50": 480.0}}})
    for name in ("send_trade_webhook", "send_risk_alert_webhook", "send_webhook_notification", "send_ops_alert",
                 "send_stop_loss_webhook", "send_spy_gate_change_push", "send_market_turn_ready_push"):
        if hasattr(eu, name):
            monkeypatch.setattr(eu, name, lambda *a, **k: None)
    pyramids_called = []
    real_pyr = ai.evaluate_pyramids
    monkeypatch.setattr(ai, "evaluate_pyramids", lambda *a, **k: pyramids_called.append(1) or real_pyr(*a, **k))
    monkeypatch.setattr(ai, "time", __import__("types").SimpleNamespace(sleep=lambda *a: None), raising=False)
    import time as _t
    monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

    db.add(AIPortfolioConfig(user_id=1, starting_cash=25000.0, current_cash=20000.0, max_positions=8,
                             is_active=True, strategy="c2", stop_loss_pct=7.0, sell_score_threshold=45,
                             peak_portfolio_value=21000.0))
    _score(db, "KEEP", 88.0, "Tech")
    _score(db, "OLD", 40.0, "Tech")                       # classic holding the engine no longer wants
    _score(db, "NEW1", 99.0, "Energy")
    _score(db, "NEW2", 95.0, "Health")
    _pos(db, "KEEP", cost=40.0, price=40.0)
    _pos(db, "OLD", cost=40.0, price=40.0)
    db.commit()

    res = ai.run_ai_trading_cycle(db, user_id=1)
    assert res.get("status") not in ("error", "inactive", "busy"), res
    assert [s["ticker"] for s in res["sells_executed"]] == ["OLD"]
    assert sorted(b["ticker"] for b in res["buys_executed"]) == ["NEW1", "NEW2"]
    held = {p.ticker: p for p in db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == 1).all()}
    assert set(held) == {"KEEP", "NEW1", "NEW2"}                    # max_positions 3
    assert held["NEW1"].purchase_score == 99.0 and held["KEEP"].current_score == 88.0   # CANSLIM 2.0 percentiles
    total = 20000.0 + 2 * 400.0                                      # cash + two 10-share positions at $40
    assert held["NEW1"].current_value == pytest.approx(total / 3, rel=0.02)   # equal dollars, no bearish reserve
    buys = db.query(AIPortfolioTrade).filter(AIPortfolioTrade.action == "BUY").all()
    assert all("CANSLIM 2.0 top" in t.reason for t in buys)
    assert pyramids_called and not any(t.action == "PYRAMID" for t in db.query(AIPortfolioTrade).all())
