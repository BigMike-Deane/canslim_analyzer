"""
Regression tests for backend/scheduler.update_coiled_spring_outcomes.

Pins three bug fixes plus the documented design intent:

  Fix #1 — outcome_updated_at must be written alongside outcome.
  Fix #3 — wait window must use stocks.next_earnings_date when available
           (handles earnings reschedules), with snapshotted DTE fallback.
  Fix #4 — bucketing must be alpha-relative when SPY data is available
           (subtract SPY same-window change), with absolute fallback when
           the alert-date SPY snapshot is missing.

Plus design-intent docstring asserts:
  #2 (first-eval-wins) and #5 (signal-quality, not trader P&L).

These tests exercise the function directly using an in-memory SQLite
database and a SessionLocal monkeypatch — no Redis, no network.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import scheduler as scheduler_mod
from backend.database import (
    Base,
    CoiledSpringAlert,
    MarketSnapshot,
    Stock,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session(monkeypatch):
    """Fresh in-memory SQLite session, wired into scheduler.SessionLocal."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # The scheduler imports SessionLocal lazily inside the function,
    # so we patch it at the source module.
    import backend.database as db_mod

    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    return Session()


@pytest.fixture
def cs_config(monkeypatch):
    """Force a known coiled_spring outcome_tracking config."""
    cfg = {
        "coiled_spring": {
            "outcome_tracking": {
                "enabled": True,
                "check_days_after_earnings": 3,
                "thresholds": {
                    "big_win_pct": 15,
                    "win_pct": 5,
                    "loss_pct": -5,
                },
            }
        }
    }

    class FakeConfig:
        def get(self, key, default=None):
            # Two-level dotted access not used here; mimic dict-style get.
            return cfg.get(key, default)

    import config_loader

    monkeypatch.setattr(config_loader, "config", FakeConfig())
    return cfg


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_alert(db, *, ticker="AAPL", alert_date=None, days_to_earnings=7,
                price_at_alert=100.0):
    alert = CoiledSpringAlert(
        ticker=ticker,
        alert_date=alert_date or (date.today() - timedelta(days=20)),
        days_to_earnings=days_to_earnings,
        weeks_in_base=20,
        beat_streak=4,
        c_score=13,
        total_score=75,
        cs_bonus=20,
        price_at_alert=price_at_alert,
        base_type="cup",
        institutional_pct=15,
        l_score=10,
        confidence=70,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def _make_stock(db, *, ticker="AAPL", current_price=110.0,
                next_earnings_date=None):
    stock = Stock(
        ticker=ticker,
        name=f"{ticker} Inc.",
        sector="Technology",
        current_price=current_price,
        next_earnings_date=next_earnings_date,
    )
    db.add(stock)
    db.commit()
    return stock


def _make_market_snapshot(db, *, snap_date, spy_price):
    snap = MarketSnapshot(date=snap_date, spy_price=spy_price)
    db.add(snap)
    db.commit()
    return snap


# ── Fix #1: outcome_updated_at is written ─────────────────────────────────────


class TestOutcomeUpdatedAtPinned:
    """When an outcome is written, outcome_updated_at MUST be set."""

    def test_outcome_updated_at_is_set_on_win(self, db_session, cs_config):
        alert_date = date.today() - timedelta(days=20)
        # 13% gain → "win" by default thresholds
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=113.0)

        before = datetime.now(timezone.utc)
        scheduler_mod.update_coiled_spring_outcomes()
        after = datetime.now(timezone.utc)

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "win"
        assert alert.outcome_updated_at is not None
        # Stored as naive in SQLite — compare normalized.
        ts = alert.outcome_updated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        assert before <= ts <= after

    def test_outcome_updated_at_is_set_on_loss(self, db_session, cs_config):
        alert_date = date.today() - timedelta(days=20)
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=90.0)  # -10% → "loss"

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "loss"
        assert alert.outcome_updated_at is not None

    def test_outcome_updated_at_skipped_when_outcome_skipped(self, db_session, cs_config):
        # Wait window not satisfied — should NOT write outcome_updated_at.
        alert_date = date.today() - timedelta(days=2)
        _make_alert(db_session, alert_date=alert_date, days_to_earnings=14,
                    price_at_alert=100.0)
        _make_stock(db_session, current_price=200.0)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome is None
        assert alert.outcome_updated_at is None


# ── Fix #3: actual next_earnings_date overrides snapshotted DTE ───────────────


class TestActualEarningsDateOverridesSnapshot:
    """If stocks.next_earnings_date is set, use it; else fall back to DTE."""

    def test_actual_earnings_date_used_when_present(self, db_session, cs_config):
        # Snapshotted DTE says 30 days (window = 33 days, would skip).
        # Actual earnings happened 10 days ago → wait satisfied at day 13+.
        alert_date = date.today() - timedelta(days=15)
        actual_earnings = date.today() - timedelta(days=10)
        _make_alert(db_session, alert_date=alert_date, days_to_earnings=30,
                    price_at_alert=100.0)
        _make_stock(db_session, current_price=110.0,
                    next_earnings_date=actual_earnings)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        # 10% raw → "win"
        assert alert.outcome == "win"

    def test_falls_back_to_snapshotted_dte_when_no_earnings_date(
        self, db_session, cs_config
    ):
        # No next_earnings_date on stock; uses snapshotted DTE = 7.
        alert_date = date.today() - timedelta(days=12)  # 7 + 3 = 10, satisfied
        _make_alert(db_session, alert_date=alert_date, days_to_earnings=7,
                    price_at_alert=100.0)
        _make_stock(db_session, current_price=120.0,
                    next_earnings_date=None)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        # 20% raw → "big_win" (>= 15)
        assert alert.outcome == "big_win"

    def test_actual_earnings_date_in_future_does_not_satisfy_window(
        self, db_session, cs_config
    ):
        # Earnings rescheduled to the future — wait NOT satisfied yet.
        alert_date = date.today() - timedelta(days=5)
        future_earnings = date.today() + timedelta(days=10)
        _make_alert(db_session, alert_date=alert_date, days_to_earnings=1,
                    price_at_alert=100.0)
        _make_stock(db_session, current_price=200.0,
                    next_earnings_date=future_earnings)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome is None  # window not yet satisfied


# ── Fix #4: SPY-relative bucketing with safe fallback ─────────────────────────


class TestAlphaRelativeBucketing:
    """When SPY data is available at alert_date AND now, subtract SPY's
    same-window % change from the stock's price change before bucketing."""

    def test_alpha_relative_lifts_bucket_when_spy_tanked(
        self, db_session, cs_config
    ):
        # Stock up 3% absolute (raw → "flat") but SPY down 3% over same
        # window → alpha-relative is +6% → "win".
        alert_date = date.today() - timedelta(days=20)
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=103.0)
        _make_stock(db_session, ticker="SPY", current_price=485.0)
        _make_market_snapshot(db_session, snap_date=alert_date, spy_price=500.0)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "win"
        # 3.0 - (-3.0) = 6.0
        assert alert.price_change_pct == pytest.approx(6.0, abs=0.05)

    def test_alpha_relative_lowers_bucket_when_spy_ripped(
        self, db_session, cs_config
    ):
        # Stock up 6% absolute (raw → "win") but SPY ripped 8% → alpha
        # is -2% → "flat".
        alert_date = date.today() - timedelta(days=20)
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=106.0)
        _make_stock(db_session, ticker="SPY", current_price=540.0)
        _make_market_snapshot(db_session, snap_date=alert_date, spy_price=500.0)

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "flat"
        # 6.0 - 8.0 = -2.0
        assert alert.price_change_pct == pytest.approx(-2.0, abs=0.05)

    def test_falls_back_to_absolute_when_spy_snapshot_missing(
        self, db_session, cs_config
    ):
        # SPY ticker exists in stocks but no MarketSnapshot for alert_date.
        # Should preserve absolute (pre-fix-#4) behavior.
        alert_date = date.today() - timedelta(days=20)
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=108.0)  # +8% absolute
        _make_stock(db_session, ticker="SPY", current_price=540.0)
        # Note: no MarketSnapshot row added.

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "win"  # 8% absolute → "win"
        assert alert.price_change_pct == pytest.approx(8.0, abs=0.05)

    def test_falls_back_to_absolute_when_spy_stock_row_missing(
        self, db_session, cs_config
    ):
        # No SPY row at all — must not blow up; falls back to absolute.
        alert_date = date.today() - timedelta(days=20)
        _make_alert(db_session, alert_date=alert_date, price_at_alert=100.0)
        _make_stock(db_session, current_price=120.0)  # +20% absolute

        scheduler_mod.update_coiled_spring_outcomes()

        alert = db_session.query(CoiledSpringAlert).first()
        assert alert.outcome == "big_win"
        assert alert.price_change_pct == pytest.approx(20.0, abs=0.05)


# ── Design intent: docstring comments are present ─────────────────────────────


class TestDesignIntentDocumented:
    """Comments #2 and #5 must remain in the function so future readers see
    the intent. If somebody strips them, this test fails to remind them."""

    def test_first_eval_wins_design_note_present(self):
        import inspect

        src = inspect.getsource(scheduler_mod.update_coiled_spring_outcomes)
        assert "#2" in src
        assert "first evaluation wins" in src.lower() or "first eval wins" in src.lower()

    def test_signal_quality_design_note_present(self):
        import inspect

        src = inspect.getsource(scheduler_mod.update_coiled_spring_outcomes)
        assert "#5" in src
        assert "signal" in src.lower() and "p&l" in src.lower()
