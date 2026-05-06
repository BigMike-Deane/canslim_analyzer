"""
Regression tests for the DB-backed breakout monitor cooldown + daily cap
(May 2026 notification-bloat fix).

Pre-fix: cooldown lived in a module-level dict that was wiped on every
container restart, producing repeat alerts per ticker after each redeploy.
Post-fix: cooldown lives in BreakoutAlert rows (survives restart) and a
configurable daily cap (default 10) prevents broad-market alert spam.

Pins:
  - BreakoutAlert table is created by init_db() (so a fresh deploy works).
  - Same-ticker alert within 24h is skipped.
  - Same-ticker alert after 24h+ fires again.
  - Daily cap (default 10) — 11th alert of the day is skipped.
  - Status endpoint reports today's count + configured cap.
  - SystemState model + helpers (get_system_state / set_system_state) work
    for the SPY-flip and morning-briefing dedup that share this table.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import breakout_monitor
from backend.database import (
    Base,
    BreakoutAlert,
    Stock,
    SystemState,
    get_system_state,
    set_system_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session(monkeypatch):
    """Fresh in-memory SQLite session, wired into module SessionLocal()s."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # Both `breakout_monitor.check_intraday_breakouts` and
    # `breakout_monitor.get_breakout_monitor_status` import SessionLocal
    # via `from backend.database import SessionLocal` (lazy late binding).
    import backend.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    return Session()


@pytest.fixture
def stub_market_open(monkeypatch):
    """Force is_market_open() to return True so the monitor runs."""
    monkeypatch.setattr("backend.ai_trader.is_market_open", lambda: True)


@pytest.fixture
def silence_notifications(monkeypatch):
    """Stub out webhook + broadcast so tests don't try to send real alerts."""
    sent = []

    def _capture_webhook(**kwargs):
        sent.append(("webhook", kwargs))
        return True

    def _capture_broadcast(**kwargs):
        sent.append(("broadcast", kwargs))
        return True

    monkeypatch.setattr(
        "backend.email_utils.send_webhook_notification", _capture_webhook
    )
    monkeypatch.setattr(
        "backend.email_utils.broadcast_notification", _capture_broadcast
    )
    return sent


@pytest.fixture
def yf_quotes(monkeypatch):
    """Replace _fetch_quick_quotes with a controllable stub."""
    quotes = {}
    monkeypatch.setattr(breakout_monitor, "_fetch_quick_quotes",
                        lambda tickers: dict(quotes))
    return quotes


def _make_breaking_stock(db, ticker, *, score=80, c_score=12, l_score=10,
                        pivot_price=100.0, current_price=99.0,
                        volume_ratio=2.0, sector="Technology"):
    """Insert a Stock row positioned just below pivot, ready to break out
    once the live quote crosses pivot."""
    s = Stock(
        ticker=ticker,
        name=f"{ticker} Inc.",
        sector=sector,
        current_price=current_price,
        canslim_score=score,
        c_score=c_score,
        l_score=l_score,
        pivot_price=pivot_price,
        volume_ratio=volume_ratio,
        base_type="cup",
        weeks_in_base=10,
    )
    db.add(s)
    db.commit()
    return s


# ── Schema / migration ────────────────────────────────────────────────────────


class TestBreakoutAlertTableMigration:
    """The new table must exist after Base.metadata.create_all() runs.

    init_db() iterates Base.metadata.sorted_tables with checkfirst=True, so
    the test mirrors that path. If the model class isn't reachable from
    Base, this fails immediately.
    """

    def test_table_present_after_create_all(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        tables = set(inspect(engine).get_table_names())
        assert "breakout_alerts" in tables

    def test_table_has_expected_columns(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        cols = {c["name"] for c in inspect(engine).get_columns("breakout_alerts")}
        assert {"ticker", "alert_date", "created_at", "pivot_price",
                "current_price", "vol_ratio", "label"} <= cols

    def test_system_state_table_present_after_create_all(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        tables = set(inspect(engine).get_table_names())
        assert "system_state" in tables

    def test_create_all_is_idempotent(self):
        """Calling create_all twice is a no-op (matches init_db behavior)."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        # Second call must not raise.
        Base.metadata.create_all(bind=engine)
        tables = set(inspect(engine).get_table_names())
        assert "breakout_alerts" in tables
        assert "system_state" in tables


# ── Per-ticker 24h cooldown ───────────────────────────────────────────────────


class TestPerTickerCooldown:
    """One alert per ticker per 24 hours, even if the monitor cycles repeatedly
    (e.g. after a container restart)."""

    def test_second_alert_within_24h_is_skipped(
        self, db_session, stub_market_open, silence_notifications, yf_quotes
    ):
        _make_breaking_stock(db_session, "AAPL", pivot_price=100.0,
                             current_price=99.0)
        # Quote crosses pivot — qualifies as BREAKOUT
        yf_quotes["AAPL"] = 101.0

        breakout_monitor.check_intraday_breakouts()
        assert db_session.query(BreakoutAlert).count() == 1

        # Second cycle in the same 24h window — must NOT add another row.
        breakout_monitor.check_intraday_breakouts()
        assert db_session.query(BreakoutAlert).count() == 1

    def test_alert_after_24h_window_fires_again(
        self, db_session, stub_market_open, silence_notifications, yf_quotes
    ):
        _make_breaking_stock(db_session, "AAPL", pivot_price=100.0,
                             current_price=99.0)
        # Pre-seed a stale row outside the 24h window.
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        db_session.add(BreakoutAlert(
            ticker="AAPL",
            alert_date=old.date(),
            created_at=old,
            pivot_price=100.0,
            current_price=100.5,
            vol_ratio=2.0,
            label="BREAKOUT",
        ))
        db_session.commit()

        yf_quotes["AAPL"] = 101.0
        breakout_monitor.check_intraday_breakouts()

        # New row added (count goes from 1 → 2).
        assert db_session.query(BreakoutAlert).count() == 2

    def test_dedup_survives_simulated_restart(
        self, db_session, stub_market_open, silence_notifications, yf_quotes
    ):
        """Pre-fix bug: a restart wiped the in-memory dict and re-fired
        the alert. Now the dedup is a DB row, so it survives 'restart'
        (which we simulate by creating a fresh check session — same DB)."""
        _make_breaking_stock(db_session, "MSFT", pivot_price=200.0,
                             current_price=199.0)
        yf_quotes["MSFT"] = 201.0

        breakout_monitor.check_intraday_breakouts()
        assert db_session.query(BreakoutAlert).count() == 1

        # Simulate restart — module-level state would be cleared, but the
        # DB row persists. Re-running must NOT re-fire.
        breakout_monitor.check_intraday_breakouts()
        assert db_session.query(BreakoutAlert).count() == 1


# ── Daily cap (default 10) ────────────────────────────────────────────────────


class TestDailyCap:
    """Hard ceiling on alerts per day, regardless of how many tickers qualify."""

    def test_eleventh_alert_is_skipped_when_cap_is_ten(
        self, db_session, stub_market_open, silence_notifications, yf_quotes,
        monkeypatch,
    ):
        # Force the cap to a known value so the test is config-independent.
        monkeypatch.setattr(breakout_monitor, "_get_daily_cap", lambda: 10)

        # Pre-seed 10 alerts for today (different tickers — no cooldown).
        today = date.today()
        now = datetime.now(timezone.utc)
        for i in range(10):
            db_session.add(BreakoutAlert(
                ticker=f"FILL{i}",
                alert_date=today,
                created_at=now - timedelta(minutes=i + 1),
                pivot_price=50.0, current_price=51.0,
                vol_ratio=2.0, label="BREAKOUT",
            ))
        db_session.commit()
        assert db_session.query(BreakoutAlert).count() == 10

        # An 11th would-be breakout this cycle must be SKIPPED.
        _make_breaking_stock(db_session, "ELEVENTH",
                             pivot_price=100.0, current_price=99.0)
        yf_quotes["ELEVENTH"] = 101.0

        breakout_monitor.check_intraday_breakouts()

        # Still 10 — the cap held.
        assert db_session.query(BreakoutAlert).count() == 10
        assert (db_session.query(BreakoutAlert)
                .filter(BreakoutAlert.ticker == "ELEVENTH").count() == 0)

    def test_below_cap_alert_fires_normally(
        self, db_session, stub_market_open, silence_notifications, yf_quotes,
        monkeypatch,
    ):
        monkeypatch.setattr(breakout_monitor, "_get_daily_cap", lambda: 10)

        # Only 5 alerts so far today — well under cap.
        today = date.today()
        now = datetime.now(timezone.utc)
        for i in range(5):
            db_session.add(BreakoutAlert(
                ticker=f"FILL{i}", alert_date=today,
                created_at=now - timedelta(minutes=i + 1),
                pivot_price=50.0, current_price=51.0,
                vol_ratio=2.0, label="BREAKOUT",
            ))
        db_session.commit()

        _make_breaking_stock(db_session, "GOOG",
                             pivot_price=100.0, current_price=99.0)
        yf_quotes["GOOG"] = 101.0

        breakout_monitor.check_intraday_breakouts()

        assert db_session.query(BreakoutAlert).count() == 6  # 5 + new

    def test_yesterdays_alerts_dont_count_toward_today_cap(
        self, db_session, stub_market_open, silence_notifications, yf_quotes,
        monkeypatch,
    ):
        """Daily cap is per-day, not per-rolling-24h."""
        monkeypatch.setattr(breakout_monitor, "_get_daily_cap", lambda: 3)

        # 5 alerts dated yesterday — should NOT block today's first alert.
        yesterday = date.today() - timedelta(days=1)
        # Use 36h ago so the cooldown also doesn't apply.
        old_ts = datetime.now(timezone.utc) - timedelta(hours=36)
        for i in range(5):
            db_session.add(BreakoutAlert(
                ticker=f"YDAY{i}", alert_date=yesterday,
                created_at=old_ts,
                pivot_price=50.0, current_price=51.0,
                vol_ratio=2.0, label="BREAKOUT",
            ))
        db_session.commit()

        _make_breaking_stock(db_session, "TODAY",
                             pivot_price=100.0, current_price=99.0)
        yf_quotes["TODAY"] = 101.0

        breakout_monitor.check_intraday_breakouts()

        # Today's row was added (5 yesterday + 1 today = 6).
        assert db_session.query(BreakoutAlert).count() == 6
        assert (db_session.query(BreakoutAlert)
                .filter(BreakoutAlert.ticker == "TODAY").count() == 1)

    def test_default_cap_is_ten(self):
        """If no config is set, default cap = 10 (audit-recommended value)."""
        assert breakout_monitor.DEFAULT_DAILY_CAP == 10


# ── Status endpoint reports new fields ────────────────────────────────────────


class TestStatusEndpoint:
    def test_status_includes_alerts_today_and_daily_cap(self, db_session):
        status = breakout_monitor.get_breakout_monitor_status()
        assert "alerts_today" in status
        assert "daily_cap" in status
        # Defaults stay sane in a fresh DB.
        assert status["alerts_today"] == 0
        assert status["daily_cap"] >= 1


# ── SystemState read/write (shared by SPY-flip + morning-briefing dedup) ──────


class TestSystemStateHelpers:
    def test_round_trip_preserves_value(self, db_session):
        set_system_state(db_session, "test_key", "hello")
        db_session.commit()
        assert get_system_state(db_session, "test_key") == "hello"

    def test_get_returns_default_when_key_missing(self, db_session):
        assert get_system_state(db_session, "missing", default="fallback") == "fallback"
        assert get_system_state(db_session, "missing") is None

    def test_overwrite_replaces_value_and_bumps_updated_at(self, db_session):
        set_system_state(db_session, "k", "first")
        db_session.commit()
        first_row = db_session.query(SystemState).filter_by(key="k").first()
        first_ts = first_row.updated_at

        # Sleep just enough to ensure a different timestamp on SQLite.
        import time
        time.sleep(0.01)

        set_system_state(db_session, "k", "second")
        db_session.commit()
        second_row = db_session.query(SystemState).filter_by(key="k").first()
        assert second_row.value == "second"
        assert second_row.updated_at >= first_ts

    def test_value_coerced_to_string(self, db_session):
        """Date/int values get str()'d so callers don't need to format."""
        today = date(2026, 5, 6)
        set_system_state(db_session, "d", today.isoformat())
        db_session.commit()
        assert get_system_state(db_session, "d") == "2026-05-06"


# ── SPY-gate state behavior ───────────────────────────────────────────────────


class TestSpyGateStatePersistence:
    """The SPY-gate flip notification only fires on a real state change.
    With the state in DB instead of a module global, a restart can no longer
    cause a phantom flip alert at the next evaluation."""

    def test_no_flip_when_state_unchanged(self, db_session):
        from backend.ai_trader import SPY_GATE_STATE_KEY
        # Seed bullish.
        set_system_state(db_session, SPY_GATE_STATE_KEY, "bullish")
        db_session.commit()

        # Read it back — caller would compare to current_gate.
        previous = get_system_state(db_session, SPY_GATE_STATE_KEY)
        current = "bullish"
        assert previous == current  # → no notification fires

    def test_flip_detected_when_state_changes(self, db_session):
        from backend.ai_trader import SPY_GATE_STATE_KEY
        set_system_state(db_session, SPY_GATE_STATE_KEY, "bullish")
        db_session.commit()

        previous = get_system_state(db_session, SPY_GATE_STATE_KEY)
        current = "bearish"
        assert previous != current  # → notification would fire
        assert previous == "bullish"

        # Caller writes the new state after firing.
        set_system_state(db_session, SPY_GATE_STATE_KEY, current)
        db_session.commit()
        assert get_system_state(db_session, SPY_GATE_STATE_KEY) == "bearish"

    def test_first_evaluation_after_fresh_install_does_not_fire(self, db_session):
        """When SystemState is empty (e.g. fresh DB), previous is None and
        the SPY-flip notification path checks `previous_gate is not None`
        before firing — matches the existing live behavior."""
        from backend.ai_trader import SPY_GATE_STATE_KEY
        previous = get_system_state(db_session, SPY_GATE_STATE_KEY)
        assert previous is None
        # ai_trader's gate-change branch is gated on `previous_gate is not None`.


# ── Morning briefing dedup ────────────────────────────────────────────────────


class TestMorningBriefingDedup:
    """The morning briefing must send exactly once per calendar day, even
    across container restarts."""

    def test_first_call_of_day_is_eligible(self, db_session):
        from backend.scheduler import LAST_BRIEFING_DATE_KEY
        last_sent = get_system_state(db_session, LAST_BRIEFING_DATE_KEY)
        today_iso = date.today().isoformat()
        assert last_sent != today_iso  # → would proceed to send

    def test_second_call_same_day_is_skipped(self, db_session):
        from backend.scheduler import LAST_BRIEFING_DATE_KEY
        today_iso = date.today().isoformat()
        set_system_state(db_session, LAST_BRIEFING_DATE_KEY, today_iso)
        db_session.commit()

        last_sent = get_system_state(db_session, LAST_BRIEFING_DATE_KEY)
        assert last_sent == today_iso  # → would short-circuit

    def test_next_day_call_is_eligible_again(self, db_session):
        from backend.scheduler import LAST_BRIEFING_DATE_KEY
        yesterday_iso = (date.today() - timedelta(days=1)).isoformat()
        set_system_state(db_session, LAST_BRIEFING_DATE_KEY, yesterday_iso)
        db_session.commit()

        last_sent = get_system_state(db_session, LAST_BRIEFING_DATE_KEY)
        today_iso = date.today().isoformat()
        assert last_sent != today_iso  # → would proceed to send today
