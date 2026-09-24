"""Daily parity monitor (backend.parity_monitor), 2026-09-24: each check
fires on the exact shape of a Sep-24 finding and stays quiet on clean data."""
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from backend.database import (
    SessionLocal, init_db, AIPortfolioConfig, AIPortfolioTrade, ShadowStrategy,
    ShadowTrade, ShadowEquityMark, ShadowPositionPeak,
)
from backend import parity_monitor as pm

ET = ZoneInfo("America/New_York")
FRI = date(2026, 9, 25)


def _clean(db):
    for m in (ShadowPositionPeak, ShadowEquityMark, ShadowTrade, ShadowStrategy,
              AIPortfolioTrade, AIPortfolioConfig):
        db.query(m).delete()
    db.commit()


@pytest.fixture
def db():
    init_db()
    s = SessionLocal()
    _clean(s)
    try:
        yield s
    finally:
        _clean(s)
        s.close()


def _utc(d, h, m=0):
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET).astimezone(timezone.utc).replace(tzinfo=None)


def _arm(db, name="arm_x"):
    s = ShadowStrategy(name=name, parent_strategy="nostate_cs_bear", config_snapshot={},
                       scorer_overrides={}, starting_value=25000.0,
                       activated_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    db.add(s)
    db.commit()
    return s


def _st(db, s, ticker, action, value, when, shares=10.0, reason=None):
    db.add(ShadowTrade(shadow_strategy_id=s.id, ticker=ticker, action=action, shares=shares,
                       price=value / shares, total_value=value, reason=reason, executed_at=when))
    db.commit()


def test_off_hours_flags_after_close_and_weekend_but_not_session_or_grace(db):
    s = _arm(db)
    _st(db, s, "IN", "BUY", 2000, _utc(FRI, 10))
    _st(db, s, "GRACE", "BUY", 2000, _utc(FRI, 16, 5))
    _st(db, s, "LATE", "BUY", 2000, _utc(FRI, 17, 26))
    _st(db, s, "SPL", "SPLIT", 0.01, _utc(FRI, 20))
    db.add(AIPortfolioTrade(ticker="LIVELATE", action="BUY", shares=1, price=10, total_value=10,
                            user_id=1, executed_at=_utc(FRI, 20)))
    db.commit()
    got = pm.check_off_hours(db, FRI)
    flagged = sorted(t for t in ("IN", "GRACE", "LATE", "SPL", "LIVELATE")
                     if any(f" {t} at " in g for g in got))
    assert flagged == ["LATE", "LIVELATE"]
    sat = date(2026, 9, 26)
    _st(db, s, "WKND", "SELL", 2000, _utc(sat, 0, 55))
    assert len(pm.check_off_hours(db, sat)) == 1


def test_runt_buy_flags_below_live_floor_only(db):
    s = _arm(db)
    _st(db, s, "RUNT", "BUY", 9.46, _utc(FRI, 11))
    _st(db, s, "FULL", "BUY", 2500, _utc(FRI, 11))
    _st(db, s, "SPY", "BUY", 50, _utc(FRI, 11), reason="SPY SWEEP BUY: park")
    got = pm.check_runt_buys(db, FRI)
    assert len(got) == 1 and "RUNT $9.46" in got[0]


def test_unpriced_latest_mark_is_flagged(db):
    s = _arm(db)
    for d, unpriced in ((date(2026, 9, 23), 0), (date(2026, 9, 24), 2)):
        db.add(ShadowEquityMark(shadow_strategy_id=s.id, date=d, equity=25000, cash=0,
                                positions_value=0, sweep_value=0, n_positions=2,
                                unpriced_positions=unpriced))
    db.commit()
    assert pm.check_unpriced_marks(db) == ["unpriced_mark: arm_x 2026-09-24 has 2 holding(s) with no close"]


def test_negative_arm_cash_is_flagged(db):
    s = _arm(db)
    _st(db, s, "BIG", "BUY", 30000, _utc(FRI, 11), shares=1000)
    assert pm.check_negative_cash(db) == ["negative_cash: arm_x $-5,000.00"]


def test_live_cash_must_reconcile_to_the_ledger(db):
    db.add(AIPortfolioConfig(user_id=1, starting_cash=25000.0, current_cash=25200.0,
                             is_active=True, strategy="nostate_cs_bear"))
    for action, value in (("BUY", 1000.0), ("PYRAMID", 500.0), ("SELL", 1700.0)):
        db.add(AIPortfolioTrade(ticker="T", action=action, shares=1, price=value,
                                total_value=value, user_id=1, executed_at=_utc(FRI, 11)))
    db.commit()
    assert pm.check_live_cash(db) == []
    db.query(AIPortfolioConfig).update({"current_cash": 25000.0})
    db.commit()
    assert pm.check_live_cash(db) == ["live_cash: u1 cash $25,000.00 vs ledger $25,200.00"]


def test_days_before_the_fix_are_not_re_reported(db):
    s = _arm(db)
    thu = date(2026, 9, 24)
    _st(db, s, "OLD", "BUY", 5, _utc(thu, 21))
    assert pm.run_parity_checks(db, day=thu) == []
