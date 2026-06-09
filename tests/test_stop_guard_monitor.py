"""Tests for backend/stop_guard_monitor.py — the freeze-safe breach alert.

The monitor mirrors the new_position_guard block in ai_trader.py:1304-1314
without modifying that file (June-18 freeze). These tests pin:
  1. the pure guard mirror (window edge, pyramid skip, loss threshold),
  2. the DB scan path (alert creation, dedup, NaN-price skip, config gate),
  3. that ai_trader.py's guard block still has the semantics we mirror —
     so if the post-freeze guard port changes them, this file fails loudly
     and the monitor (or its deletion) gets revisited.
"""

import math
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, AIPortfolioConfig, AIPortfolioPosition, Notification
from backend.stop_guard_monitor import (
    NOTIFICATION_KIND,
    check_stop_guard_breaches,
    guard_would_fire,
)

TODAY = date(2026, 6, 9)


def _fire(**overrides):
    kwargs = dict(
        gain_loss_pct=-12.0,
        purchase_date=TODAY - timedelta(days=5),
        pyramid_count=0,
        today=TODAY,
        guard_enabled=True,
        guard_days=21,
        guard_stop_pct=8.0,
        skip_if_pyramided=True,
    )
    kwargs.update(overrides)
    return guard_would_fire(**kwargs)


class TestGuardWouldFire:
    def test_new_position_past_guard_fires(self):
        assert _fire() is True

    def test_loss_inside_guard_does_not_fire(self):
        assert _fire(gain_loss_pct=-5.0) is False

    def test_loss_exactly_at_guard_fires(self):
        # ai_trader uses gain_pct <= -position_stop_pct (inclusive)
        assert _fire(gain_loss_pct=-8.0) is True

    def test_holding_days_at_window_edge_fires(self):
        # guard block uses holding_days <= guard_days (inclusive)
        assert _fire(purchase_date=TODAY - timedelta(days=21)) is True

    def test_holding_days_past_window_does_not_fire(self):
        assert _fire(purchase_date=TODAY - timedelta(days=22)) is False

    def test_pyramided_position_skipped(self):
        assert _fire(pyramid_count=1) is False

    def test_pyramided_fires_when_skip_disabled(self):
        assert _fire(pyramid_count=1, skip_if_pyramided=False) is True

    def test_pyramid_count_none_treated_as_zero(self):
        assert _fire(pyramid_count=None) is True

    def test_disabled_guard_never_fires(self):
        assert _fire(guard_enabled=False) is False

    def test_missing_purchase_date_never_fires(self):
        assert _fire(purchase_date=None) is False

    def test_datetime_purchase_date_accepted(self):
        # DB column is DateTime; mirror's .date() normalization must handle it
        assert _fire(purchase_date=datetime(2026, 6, 4, 14, 30)) is True

    def test_none_gain_treated_as_zero(self):
        assert _fire(gain_loss_pct=None) is False


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _position(user_id=1, ticker="FSLR", gain=-12.0, days_held=5,
              price=100.0, pyramids=0):
    # cost_basis must stay a real number even when price is NaN — SQLite
    # coerces NaN to NULL, tripping the NOT NULL constraint.
    cost_basis = price / (1 + gain / 100)
    if math.isnan(cost_basis):
        cost_basis = 100.0
    return AIPortfolioPosition(
        user_id=user_id,
        ticker=ticker,
        shares=10,
        cost_basis=cost_basis,
        purchase_date=datetime.now(timezone.utc) - timedelta(days=days_held),
        current_price=price,
        gain_loss_pct=gain,
        pyramid_count=pyramids,
    )


@pytest.fixture
def patched_io(monkeypatch):
    """Silence the two I/O seams: live-stop estimate and notification fan-out."""
    sent = []

    def fake_notify(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(
        "backend.stop_guard_monitor._estimate_live_stop_pct",
        lambda ticker, price, cfg: 19.2,
    )
    monkeypatch.setattr("backend.email_utils.create_notification", fake_notify)
    return sent


class TestCheckStopGuardBreaches:
    def test_breach_creates_urgent_notification(self, db, patched_io):
        db.add(_position(ticker="FSLR", gain=-12.0, days_held=5))
        db.add(AIPortfolioConfig(user_id=1, stop_loss_pct=7.0, strategy="balanced"))
        db.commit()

        breaches = check_stop_guard_breaches(db=db)

        assert len(breaches) == 1
        b = breaches[0]
        assert b["ticker"] == "FSLR"
        assert b["gain_loss_pct"] == -12.0
        assert b["live_stop_pct"] == 19.2
        assert len(patched_io) == 1
        note = patched_io[0]
        assert note["user_id"] == 1
        assert note["kind"] == NOTIFICATION_KIND
        assert note["priority"] == "urgent"
        assert "FSLR" in note["title"]
        assert "12.0%" in note["title"]
        assert "manual exit" in note["body"]

    def test_healthy_and_old_positions_ignored(self, db, patched_io):
        db.add(_position(ticker="WINNER", gain=12.0, days_held=5))
        db.add(_position(ticker="SMALLDIP", gain=-4.0, days_held=5))
        db.add(_position(ticker="OLDLOSER", gain=-15.0, days_held=40))
        db.add(_position(ticker="PYRAMIDED", gain=-12.0, days_held=5, pyramids=2))
        db.commit()

        assert check_stop_guard_breaches(db=db) == []
        assert patched_io == []

    def test_nan_and_none_price_skipped(self, db, patched_io):
        db.add(_position(ticker="NANPX", gain=-12.0, days_held=5, price=math.nan))
        p = _position(ticker="NOPX", gain=-12.0, days_held=5)
        p.current_price = None
        db.add(p)
        db.commit()

        assert check_stop_guard_breaches(db=db) == []

    def test_recent_alert_dedups(self, db, patched_io):
        db.add(_position(ticker="FSLR", gain=-12.0, days_held=5))
        db.add(Notification(
            user_id=1,
            kind=NOTIFICATION_KIND,
            title="x",
            body="x",
            data={"ticker": "FSLR"},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        ))
        db.commit()

        assert check_stop_guard_breaches(db=db) == []
        assert patched_io == []

    def test_stale_alert_does_not_dedup(self, db, patched_io):
        db.add(_position(ticker="FSLR", gain=-12.0, days_held=5))
        db.add(Notification(
            user_id=1,
            kind=NOTIFICATION_KIND,
            title="x",
            body="x",
            data={"ticker": "FSLR"},
            created_at=datetime.now(timezone.utc) - timedelta(hours=30),
        ))
        db.commit()

        assert len(check_stop_guard_breaches(db=db)) == 1

    def test_other_ticker_alert_does_not_dedup(self, db, patched_io):
        db.add(_position(ticker="AMD", gain=-13.7, days_held=5))
        db.add(Notification(
            user_id=1,
            kind=NOTIFICATION_KIND,
            title="x",
            body="x",
            data={"ticker": "FSLR"},
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ))
        db.commit()

        assert len(check_stop_guard_breaches(db=db)) == 1

    def test_per_user_isolation(self, db, patched_io):
        db.add(_position(user_id=1, ticker="FSLR", gain=-12.0, days_held=5))
        db.add(_position(user_id=2, ticker="AMD", gain=-13.7, days_held=5))
        db.commit()

        breaches = check_stop_guard_breaches(db=db)

        assert {(b["user_id"], b["ticker"]) for b in breaches} == {(1, "FSLR"), (2, "AMD")}
        assert {n["user_id"] for n in patched_io} == {1, 2}

    def test_guard_disabled_in_config_returns_empty(self, db, patched_io, monkeypatch):
        from config_loader import config as yaml_config
        db.add(_position(ticker="FSLR", gain=-12.0, days_held=5))
        db.commit()

        original_get = yaml_config.get

        def fake_get(key, default=None, **kw):
            if key == "ai_trader.new_position_guard":
                return {"enabled": False}
            return original_get(key, default, **kw)

        monkeypatch.setattr(yaml_config, "get", fake_get)
        assert check_stop_guard_breaches(db=db) == []

    def test_live_stop_estimate_failure_still_alerts(self, db, patched_io, monkeypatch):
        monkeypatch.setattr(
            "backend.stop_guard_monitor._estimate_live_stop_pct",
            lambda ticker, price, cfg: None,
        )
        db.add(_position(ticker="FSLR", gain=-12.0, days_held=5))
        db.commit()

        breaches = check_stop_guard_breaches(db=db)

        assert len(breaches) == 1
        assert breaches[0]["live_stop_pct"] is None
        assert "likely widened" in patched_io[0]["body"]


class TestGuardSemanticsStillMirrorAiTrader:
    """Pin the frozen guard block's semantics so a post-freeze edit to
    ai_trader.py forces this monitor to be revisited (likely: deleted)."""

    def test_ai_trader_guard_block_unchanged(self):
        import inspect
        import backend.ai_trader as ai_trader

        src = inspect.getsource(ai_trader)
        # The clamp exists in the manual stop path...
        assert "position_stop_pct = min(position_stop_pct, guard_stop_pct)" in src
        # ...with the exact window/skip semantics we mirror.
        assert "if holding_days <= guard_days:" in src
        assert "if not (skip_if_pyramided and pyramid_count > 0):" in src

    def test_evaluate_sells_still_missing_guard(self):
        """The bug this monitor covers: evaluate_sells has NO guard clamp.
        When the June-18 guard port lands, this assertion fails on purpose —
        delete backend/stop_guard_monitor.py and this test file with it."""
        import inspect
        from backend.ai_trader import evaluate_sells

        src = inspect.getsource(evaluate_sells)
        assert "new_position_guard" not in src, (
            "evaluate_sells now applies the new_position_guard — the breach "
            "monitor is obsolete. Remove backend/stop_guard_monitor.py, the "
            "scheduler job (start_stop_guard_monitor_job), and this test file."
        )
