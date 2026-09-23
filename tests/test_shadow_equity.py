"""Daily closing equity per shadow stack (backend/shadow_equity.py, 2026-09-23).

The Oct-21 readout's mechanism check needs a daily return series per arm;
shadow equity used to be marked only "now". These tests pin: the replay
ledger equals ShadowSession's own ledger, the session-close cutoff, the
fill job's idempotency / readiness / feed guards, and the regime split.
"""

import os
import sys
from datetime import date, datetime, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db, SessionLocal, ShadowStrategy, ShadowTrade, ShadowEquityMark,
    ShadowPositionPeak, MarketSnapshot,
)
from backend.shadow_trader import ShadowSession
from backend import shadow_equity as SE


@pytest.fixture
def db():
    init_db()
    s = SessionLocal()

    def _wipe():
        for m in (ShadowEquityMark, ShadowPositionPeak, ShadowTrade, ShadowStrategy, MarketSnapshot):
            s.query(m).delete()
        s.commit()
    _wipe()
    try:
        yield s
    finally:
        _wipe()
        s.close()


@pytest.fixture(autouse=True)
def no_history_backfill(monkeypatch):
    monkeypatch.setattr(ShadowSession, "_init_peak_from_history", lambda self, pos: None)


def _strategy(db, name="shadow_eq_test", activated=datetime(2026, 9, 21, 13, 5)):
    s = ShadowStrategy(name=name, parent_strategy="nostate_cs_bear",
                       config_snapshot={"strategy": "nostate_cs_bear"},
                       scorer_overrides={}, starting_value=25000.0, activated_at=activated)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _trade(db, s, tk, action, sh, px, at, reason="shadow", sf=None):
    db.add(ShadowTrade(shadow_strategy_id=s.id, ticker=tk, action=action, shares=sh,
                       price=px, total_value=sh * px, reason=reason,
                       signal_factors=sf, executed_at=at))
    db.commit()


def _trades(db, s):
    return db.query(ShadowTrade).filter(ShadowTrade.shadow_strategy_id == s.id) \
        .order_by(ShadowTrade.executed_at).all()


class TestReplayMatchesShadowSession:
    def test_end_of_log_ledger_is_identical(self, db):
        s = _strategy(db)
        t = lambda h: datetime(2026, 9, 21, h)
        _trade(db, s, "AAA", "BUY", 10, 100.0, t(14))
        _trade(db, s, "AAA", "PYRAMID", 5, 110.0, t(15))
        _trade(db, s, "AAA", "SELL", 8, 120.0, t(16), reason="PARTIAL TRAILING STOP (50%): x")
        _trade(db, s, "BBB", "BUY", 20, 50.0, t(17))
        _trade(db, s, "SPY", "BUY", 3, 500.0, t(18), reason="SPY SWEEP: park idle cash")
        _trade(db, s, "SPY", "SELL", 1, 505.0, t(19), reason="SPY SWEEP LIQUIDATION")
        _trade(db, s, "BBB", "SPLIT", 2, 0.0, t(20), reason="SPLIT ADJUST 2:1",
               sf={"split_factor": 2})
        session = ShadowSession(db, s, [])
        cfg = session._synthetic_config
        live_shares = {p.ticker: p.shares for p in session._synthetic_positions or []}
        rb = SE.replay_book(_trades(db, s), 25000.0)
        assert rb["cash"] == pytest.approx(cfg.current_cash)
        assert rb["sweep_shares"] == pytest.approx(cfg.spy_sweep_shares)
        assert rb["shares"] == pytest.approx(live_shares)
        assert rb["shares"]["BBB"] == pytest.approx(40)      # split applied


class TestMarkForDay:
    CLOSES = {"AAA": {date(2026, 9, 21): 110.0, date(2026, 9, 22): 121.0},
              "SPY": {date(2026, 9, 21): 500.0, date(2026, 9, 22): 510.0}}

    def test_after_hours_fill_counts_toward_the_next_session(self, db):
        s = _strategy(db)
        _trade(db, s, "AAA", "BUY", 10, 100.0, datetime(2026, 9, 21, 14))       # 10:00 ET
        _trade(db, s, "AAA", "BUY", 10, 111.0, datetime(2026, 9, 21, 23, 35))   # 19:35 ET
        tr = _trades(db, s)
        d1 = SE.mark_for_day(tr, 25000.0, date(2026, 9, 21), self.CLOSES)
        d2 = SE.mark_for_day(tr, 25000.0, date(2026, 9, 22), self.CLOSES)
        assert d1["positions_value"] == pytest.approx(10 * 110.0)
        assert d1["equity"] == pytest.approx(25000 - 1000 + 1100)
        assert d2["positions_value"] == pytest.approx(20 * 121.0)

    def test_missing_close_carries_and_counts_unpriced(self, db):
        s = _strategy(db)
        _trade(db, s, "ZZZ", "BUY", 10, 40.0, datetime(2026, 9, 21, 14))
        m = SE.mark_for_day(_trades(db, s), 25000.0, date(2026, 9, 22), self.CLOSES)
        assert m["positions_value"] == pytest.approx(400.0) and m["unpriced_positions"] == 1


class TestFill:
    # Mon Sep-21 .. Wed Sep-23 2026
    CLOSES = {"AAA": {date(2026, 9, d): 100.0 + d for d in (21, 22, 23)},
              "SPY": {date(2026, 9, d): 700.0 for d in (21, 22, 23)}}

    def _setup(self, db):
        s = _strategy(db)
        _trade(db, s, "AAA", "BUY", 10, 100.0, datetime(2026, 9, 21, 14))
        return s

    def _dates(self, db, s):
        return [r.date for r in db.query(ShadowEquityMark).filter(
            ShadowEquityMark.shadow_strategy_id == s.id).order_by(ShadowEquityMark.date)]

    def test_backfills_from_activation_then_is_idempotent(self, db):
        s = self._setup(db)
        after = datetime(2026, 9, 23, 20, 30, tzinfo=timezone.utc)    # 16:30 ET
        assert SE.fill_shadow_equity_marks(db, after, lambda t, st: self.CLOSES) == 3
        assert self._dates(db, s) == [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
        assert SE.fill_shadow_equity_marks(db, after, lambda t, st: self.CLOSES) == 0

    def test_today_waits_for_a_complete_bar(self, db):
        s = self._setup(db)
        early = datetime(2026, 9, 23, 20, 5, tzinfo=timezone.utc)     # 16:05 ET
        SE.fill_shadow_equity_marks(db, early, lambda t, st: self.CLOSES)
        assert self._dates(db, s)[-1] == date(2026, 9, 22)

    def test_session_missing_from_the_feed_is_deferred(self, db):
        s = self._setup(db)
        partial = {"AAA": self.CLOSES["AAA"],
                   "SPY": {date(2026, 9, 21): 700.0, date(2026, 9, 22): 700.0}}
        after = datetime(2026, 9, 23, 20, 30, tzinfo=timezone.utc)
        assert SE.fill_shadow_equity_marks(db, after, lambda t, st: partial) == 2
        assert SE.fill_shadow_equity_marks(db, after, lambda t, st: self.CLOSES) == 1

    def test_no_feed_writes_nothing(self, db):
        self._setup(db)
        after = datetime(2026, 9, 23, 20, 30, tzinfo=timezone.utc)
        assert SE.fill_shadow_equity_marks(db, after, lambda t, st: {}) == 0

    def test_weekends_are_not_sessions(self, db):
        s = _strategy(db, activated=datetime(2026, 9, 18, 13, 5))     # Friday
        after = datetime(2026, 9, 21, 20, 30, tzinfo=timezone.utc)    # Monday 16:30 ET
        closes = {"SPY": {date(2026, 9, 18): 700.0, date(2026, 9, 21): 701.0}}
        SE.fill_shadow_equity_marks(db, after, lambda t, st: closes)
        assert self._dates(db, s) == [date(2026, 9, 18), date(2026, 9, 21)]


class TestRegimeExcessVs:
    def test_splits_arm_minus_comparator_by_regime(self, db):
        arm, comp = _strategy(db, "arm"), _strategy(db, "comp")
        days = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
        for d, a_eq, c_eq in zip(days, (100.0, 101.0, 101.0), (100.0, 100.0, 102.02)):
            db.add(ShadowEquityMark(shadow_strategy_id=arm.id, date=d, equity=a_eq))
            db.add(ShadowEquityMark(shadow_strategy_id=comp.id, date=d, equity=c_eq))
        db.add(MarketSnapshot(date=days[1], spy_price=505.0, spy_50_ma=500.0))   # +1.0%: chop
        db.add(MarketSnapshot(date=days[2], spy_price=520.0, spy_50_ma=500.0))   # +4.0%: trend
        db.commit()
        out = SE.regime_excess_vs(db, arm.id, comp.id)
        assert out["chop"] == {"n_days": 1, "mean_excess_bps": 100.0}    # arm +1%, comp 0
        assert out["trend"] == {"n_days": 1, "mean_excess_bps": -202.0}  # arm 0, comp +2.02%
