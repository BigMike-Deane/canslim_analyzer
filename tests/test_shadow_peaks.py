"""Tests for shadow peak persistence + the intraday shadow stop pass.

Covers the 2026-07-25 fix for the peak-blind shadow harness (zero trailing
exits in 2.5 months because the FIFO rebuild collapsed peak to
max(cost_basis, current_price)):

  - _persist_peaks writes rows for open positions, deletes rows on close.
  - _apply_persisted_peaks restores a higher persisted peak across ticks
    (the ratchet), and ignores stale rows from a prior position generation.
  - reset_peak decisions (partial trailing) snap the in-memory peak so the
    persisted state stores the reset.
  - _init_peak_from_history is skipped for same-day positions (no network
    in tests / no pointless fetch for fresh buys).
  - is_stop_family_reason filter matches the lead intraday checker's scope.
  - run_shadow_stop_checks: emits only stop-family sells, skips when the
    scan-tick lock is held, short-circuits with no positions.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db,
    SessionLocal,
    ShadowPositionPeak,
    ShadowStrategy,
    ShadowTrade,
)
import backend.shadow_trader as shadow_trader
from backend.shadow_trader import (
    SHADOW_USER_ID,
    ShadowSession,
    _emit_sell_decisions,
    _persist_peaks,
    _run_one_stop_check,
    _same_generation,
    _shadow_run_lock,
    is_stop_family_reason,
    run_shadow_stop_checks,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowPositionPeak).delete()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowPositionPeak).delete()
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.commit()
        db.close()


@pytest.fixture(autouse=True)
def no_history_backfill(monkeypatch):
    """Keep tests off the network: history backfill no-ops unless a test
    explicitly restores/instruments it."""
    monkeypatch.setattr(ShadowSession, "_init_peak_from_history", lambda self, pos: None)


def _make_strategy(db, name="shadow_peak_test", parent="nostate_cs_bear"):
    s = ShadowStrategy(
        name=name,
        parent_strategy=parent,
        config_snapshot={"strategy": parent},
        scorer_overrides={},
        starting_value=25000.0,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _seed_buy(db, strategy, ticker="TEST", shares=10.0, price=100.0, days_ago=5):
    t = ShadowTrade(
        shadow_strategy_id=strategy.id,
        ticker=ticker,
        action="BUY",
        shares=shares,
        price=price,
        total_value=shares * price,
        reason="shadow buy",
        executed_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    db.add(t)
    db.commit()
    return t


class TestPeakPersistence:
    def test_persist_creates_rows_and_close_deletes_them(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0)

        session = ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 130.0}])
        persist = SessionLocal()
        try:
            _persist_peaks(persist, strategy.id, session)
            rows = persist.query(ShadowPositionPeak).filter(
                ShadowPositionPeak.shadow_strategy_id == strategy.id).all()
            assert len(rows) == 1
            assert rows[0].ticker == "TEST"
            assert rows[0].peak_price == pytest.approx(130.0)

            # Full close removes the synthetic position → row pruned.
            pos = session._synthetic_positions[0]
            session.emit_shadow_sell(position=pos, shares=pos.shares,
                                     price=110.0, reason="TRAILING STOP: test")
            _persist_peaks(persist, strategy.id, session)
            assert persist.query(ShadowPositionPeak).filter(
                ShadowPositionPeak.shadow_strategy_id == strategy.id).count() == 0
        finally:
            persist.close()

    def test_ratchet_restores_higher_peak_on_next_tick(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0)

        tick1 = ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 150.0}])
        persist = SessionLocal()
        try:
            _persist_peaks(persist, strategy.id, tick1)
        finally:
            persist.close()

        # Price fell to 110 — the FIFO rebuild alone would say peak=110
        # (max(cost, current)); the overlay must restore 150.
        tick2 = ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 110.0}])
        pos = tick2._synthetic_positions[0]
        assert pos.peak_price == pytest.approx(150.0)
        drop_from_peak = (pos.peak_price - pos.current_price) / pos.peak_price * 100
        assert drop_from_peak == pytest.approx(26.67, abs=0.1)

    def test_stale_generation_row_is_ignored(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0, days_ago=2)

        # Peak row from a PRIOR generation of the same ticker (opened_at far
        # from the current BUY's executed_at) must not leak its high in.
        db_session.add(ShadowPositionPeak(
            shadow_strategy_id=strategy.id,
            ticker="TEST",
            opened_at=datetime.now(timezone.utc) - timedelta(days=40),
            peak_price=500.0,
        ))
        db_session.commit()

        session = ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 120.0}])
        pos = session._synthetic_positions[0]
        assert pos.peak_price == pytest.approx(120.0)  # approximation, not 500

    def test_same_generation_tolerates_naive_aware_and_micro_drift(self):
        aware = datetime.now(timezone.utc)
        naive = aware.replace(tzinfo=None)
        assert _same_generation(aware, naive)
        assert _same_generation(aware, aware + timedelta(seconds=3))
        assert not _same_generation(aware, aware + timedelta(seconds=30))
        assert not _same_generation(None, aware)

    def test_reset_peak_decision_snaps_peak_before_persist(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0)

        session = ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 150.0}])
        pos = session._synthetic_positions[0]
        pos.pyramid_count = 2  # partial-eligible position shape
        n = _emit_sell_decisions(session, [{
            "position": pos,
            "reason": "PARTIAL TRAILING STOP: Peak $150.00 → $150.00 (-0.0%), keeping 50%",
            "is_partial": True,
            "sell_pct": 50,
            "reset_peak": True,
        }])
        assert n == 1
        assert pos.peak_price == pytest.approx(150.0)  # snapped to current

        persist = SessionLocal()
        try:
            _persist_peaks(persist, strategy.id, session)
            row = persist.query(ShadowPositionPeak).filter(
                ShadowPositionPeak.shadow_strategy_id == strategy.id).one()
            assert row.peak_price == pytest.approx(150.0)
        finally:
            persist.close()

    def test_history_backfill_skipped_for_same_day_position(self, db_session, monkeypatch):
        # Restore the real method, but trip a flag if it gets past the
        # same-day guard (i.e. tries to import the history provider).
        monkeypatch.undo()
        calls = []
        import backend.historical_data as hd
        monkeypatch.setattr(
            hd, "HistoricalDataProvider",
            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("no net")),
        )
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0, days_ago=0)
        ShadowSession(db_session, strategy, [{"ticker": "TEST", "current_price": 105.0}])
        assert calls == []


class TestStopFamilyFilter:
    def test_stop_family_reasons(self):
        assert is_stop_family_reason("STOP LOSS: Down 8.1%")
        assert is_stop_family_reason("TRAILING STOP: Peak $150.00 → $110.00 (-26.7%)")
        assert is_stop_family_reason("PARTIAL TRAILING STOP: Peak $150 → $120, keeping 50%")
        assert not is_stop_family_reason("PRE-EARNINGS STOP: 4d to earnings")
        assert not is_stop_family_reason("SCORE CRASH: 72 → 41")
        assert not is_stop_family_reason("TAKE PROFIT: Up 76.0%")
        assert not is_stop_family_reason(None)
        assert not is_stop_family_reason("")


class TestIntradayStopPass:
    def test_emits_only_stop_family_decisions(self, db_session, monkeypatch):
        strategy = _make_strategy(db_session)
        _seed_buy(db_session, strategy, price=100.0)

        import backend.ai_trader as ai_trader

        def fake_update_prices(db, use_live_prices=True, user_id=1):
            for p in db._synthetic_positions:
                p.current_price = 91.0
                p.gain_loss_pct = -9.0
            return len(db._synthetic_positions)

        def fake_evaluate_sells(db, user_id=1):
            pos = db._synthetic_positions[0]
            return [
                {"position": pos, "reason": "SCORE CRASH: 72 → 40", "priority": 4},
                {"position": pos, "reason": "STOP LOSS: Down 9.0%", "priority": 1},
            ]

        monkeypatch.setattr(ai_trader, "update_position_prices", fake_update_prices)
        monkeypatch.setattr(ai_trader, "evaluate_sells", fake_evaluate_sells)

        n = _run_one_stop_check(strategy.id)
        assert n == 1
        sells = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id,
            ShadowTrade.action == "SELL").all()
        assert len(sells) == 1
        assert sells[0].reason.startswith("STOP LOSS")

    def test_skips_when_scan_run_holds_lock(self, db_session):
        assert _shadow_run_lock.acquire(blocking=False)
        try:
            summary = run_shadow_stop_checks()
            assert summary["skipped"] is True
            assert summary["strategies_run"] == 0
        finally:
            _shadow_run_lock.release()

    def test_no_positions_short_circuits_without_price_fetch(self, db_session, monkeypatch):
        strategy = _make_strategy(db_session)  # no trades → no positions

        import backend.ai_trader as ai_trader

        def boom(*a, **k):
            raise AssertionError("update_position_prices must not run with no positions")

        monkeypatch.setattr(ai_trader, "update_position_prices", boom)
        assert _run_one_stop_check(strategy.id) == 0

    def test_archived_strategy_is_skipped(self, db_session):
        strategy = _make_strategy(db_session)
        strategy.archived_at = datetime.now(timezone.utc)
        db_session.commit()
        assert _run_one_stop_check(strategy.id) == 0
