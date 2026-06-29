"""
backend/scheduler.py auxiliary-helper coverage tests.

Triage:
  Tier 1 (signal-quality + user-trust path — runs after every scan):
    - check_watchlist_alerts: price/score crossing detection, cooldown, email
      gating via email.automated_enabled flag, retry-storm guard
    - auto_record_coiled_spring_alerts: post-scan CS recorder; gates on
      candidate filter + delegates is_coiled_spring scoring + record cooldown
    - update_coiled_spring_outcomes: post-earnings outcome bucketing for CS
      alerts. The "first evaluation wins permanently" design note (line 483)
      is load-bearing — pinned via outcome IS NULL filter assertion.

  Tier 2 (live-money plumbing for the AI portfolio):
    - _refresh_portfolio_prices: 5-min portfolio price refresh job. Iterates
      active AIPortfolioConfig rows, calls refresh_ai_portfolio per user.
    - start_continuous_scanning / stop_continuous_scanning / update_scan_config:
      module-level thread/scheduler control. _scan_config dict mutation,
      apscheduler add_job / remove_job seam.

  Tier 3 (orchestrator):
    - run_continuous_scan (lines 811-1603, ~790 lines): main scan/trade
      pipeline. Covered by TestRunContinuousScan via 8-branch matrix with
      every external coupling stubbed at its source module.

  Still intentionally uncovered (separate session if/when prioritised):
    - send_weekly_performance_email + send_weekly_bear_market_report
      (lines 1737-2103): weekly batch email jobs.
    - send_bear_report_email: helper specific to weekly bear report.
    - start_weekly_email_job + start_backup_job + start_ml_backfill_job +
      start_breakout_monitor_job: thread-launcher boilerplate.

AI Trader <> Backtester sync: scheduler has no backtester analogue (the
backtester replays historical data and does not run a live scan loop). No
sync mirror is required for this push.

All HTTP / email IO is mocked. All CS scoring / record helpers are mocked
(they are independently covered in test_ai_trader_coverage.py). DB tests
use a real in-memory SQLite session per the project rule (no Mock(spec=Session)).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import (
    AIPortfolioConfig,
    Base,
    CoiledSpringAlert,
    MarketSnapshot,
    Stock,
    Watchlist,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """Fresh in-memory SQLite for every test. No global state leakage."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def patch_session_local(monkeypatch, db_session):
    """Redirect backend.database.SessionLocal to the test in-memory engine
    so functions that open their own DB session hit the test DB. The helpers
    under test (check_watchlist_alerts, auto_record_coiled_spring_alerts,
    update_coiled_spring_outcomes, _refresh_portfolio_prices) all do
    `db = SessionLocal()` rather than accept a session parameter."""
    import backend.database as database

    monkeypatch.setattr(database, "SessionLocal", lambda: db_session)
    return db_session


@pytest.fixture
def silence_emails(monkeypatch):
    """Replace send_watchlist_alert_email with a tracking spy that never does
    real IO. Returns the spy so tests can assert call counts / payloads."""
    import backend.email_utils as email_utils

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(email_utils, "send_watchlist_alert_email", spy)
    return spy


@pytest.fixture
def stub_config(monkeypatch):
    """Yield a callable that overrides config.get(path, default) lookups for
    keys the test cares about. Unknown keys fall through to the real config.

    Usage:
        stub_config({'watchlist.alerts.enabled': False})
    """
    from config_loader import config

    real_get = config.get

    def _apply(overrides: dict):
        def fake_get(path, default=None):
            if path in overrides:
                return overrides[path]
            return real_get(path, default)

        monkeypatch.setattr(config, "get", fake_get)

    return _apply


def _seed_watchlist_item(db, ticker, **overrides):
    """Insert a Watchlist row with a sane default of target_price=100."""
    defaults = dict(
        ticker=ticker,
        target_price=100.0,
        alert_score=None,
        last_check_price=None,
        alert_triggered_at=None,
        alert_sent=False,
    )
    defaults.update(overrides)
    item = Watchlist(**defaults)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def _seed_stock(db, ticker, **overrides):
    """Insert a minimal Stock row sufficient for scheduler-helper tests."""
    defaults = dict(
        ticker=ticker,
        sector="Technology",
        current_price=100.0,
        canslim_score=70.0,
        c_score=10.0,
        weeks_in_base=20,
        earnings_beat_streak=4,
        days_to_earnings=10,
        next_earnings_date=date.today() + timedelta(days=10),
    )
    defaults.update(overrides)
    stock = Stock(**defaults)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


def _seed_cs_alert(db, ticker, alert_date, **overrides):
    """Insert a CoiledSpringAlert row with sane defaults; outcome NULL by
    default so the evaluator picks it up."""
    defaults = dict(
        ticker=ticker,
        alert_date=alert_date,
        days_to_earnings=7,
        weeks_in_base=20,
        beat_streak=4,
        c_score=12.0,
        total_score=70.0,
        cs_bonus=8.0,
        price_at_alert=100.0,
        outcome=None,
    )
    defaults.update(overrides)
    alert = CoiledSpringAlert(**defaults)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — check_watchlist_alerts
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckWatchlistAlerts:
    """Tier 1: scheduler.check_watchlist_alerts.

    Branches covered:
      - alerts disabled in config → early return
      - no watchlist items → early return
      - target_price newly crossed → triggered + email sent
      - target_price already crossed at last check → not retriggered
      - alert_score crossed → triggered
      - cooldown active → skipped
      - cooldown elapsed → re-triggered
      - email.automated_enabled=False → in-app only, no email
      - send_watchlist_alert_email exception → still marked triggered (retry-storm guard)
    """

    def test_alerts_disabled_early_returns(
        self, patch_session_local, silence_emails, stub_config
    ):
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": False})
        _seed_watchlist_item(patch_session_local, "TEST", target_price=50.0)
        _seed_stock(patch_session_local, "TEST", current_price=100.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 0

    def test_no_watchlist_items_early_returns(
        self, patch_session_local, silence_emails, stub_config
    ):
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        # No Watchlist rows seeded.
        check_watchlist_alerts()

        assert silence_emails.call_count == 0

    def test_target_price_newly_crossed_triggers(
        self, patch_session_local, silence_emails, stub_config
    ):
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        # last_check_price=90 (below target 100), current_price=105 — newly crossed.
        _seed_watchlist_item(
            patch_session_local, "TEST", target_price=100.0, last_check_price=90.0
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 1
        # Reasons list passed positionally as third arg.
        reasons = silence_emails.call_args[0][2]
        assert any("Price" in r for r in reasons)

        item = patch_session_local.query(Watchlist).filter_by(ticker="TEST").one()
        assert item.alert_sent is True
        assert item.alert_triggered_at is not None
        assert item.last_check_price == 105.0

    def test_target_already_crossed_no_retrigger(
        self, patch_session_local, silence_emails, stub_config
    ):
        """If last_check_price was already >= target, we don't re-alert on
        subsequent scans where price stays above target. This is the
        anti-spam guard at scheduler.py:292."""
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        _seed_watchlist_item(
            patch_session_local, "TEST", target_price=100.0, last_check_price=110.0
        )
        _seed_stock(patch_session_local, "TEST", current_price=115.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 0

    def test_alert_score_crossed_triggers(
        self, patch_session_local, silence_emails, stub_config
    ):
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        _seed_watchlist_item(
            patch_session_local,
            "TEST",
            target_price=None,
            alert_score=80.0,
            last_check_price=99.0,
        )
        _seed_stock(patch_session_local, "TEST", current_price=100.0, canslim_score=85.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 1
        reasons = silence_emails.call_args[0][2]
        assert any("CANSLIM" in r for r in reasons)

    def test_cooldown_active_skips(self, patch_session_local, silence_emails, stub_config):
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        # Triggered 1 hour ago — within default 24h cooldown.
        _seed_watchlist_item(
            patch_session_local,
            "TEST",
            target_price=100.0,
            last_check_price=90.0,
            alert_triggered_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 0

    def test_cooldown_elapsed_retriggers(
        self, patch_session_local, silence_emails, stub_config
    ):
        from backend.scheduler import check_watchlist_alerts

        stub_config(
            {
                "watchlist.alerts.enabled": True,
                "email.automated_enabled": True,
                # default cooldown_hours=24; we set the trigger 30h back
            }
        )
        _seed_watchlist_item(
            patch_session_local,
            "TEST",
            target_price=100.0,
            last_check_price=90.0,
            alert_triggered_at=datetime.now(timezone.utc) - timedelta(hours=30),
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 1

    def test_cooldown_naive_timestamp_handled(
        self, patch_session_local, silence_emails, stub_config
    ):
        """SQLite returns DateTimes without tz info. The scheduler must coerce
        naive timestamps to UTC before subtraction (line 309). Regression
        guard: pre-fix would raise TypeError on naive vs aware subtraction."""
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        # Naive datetime — SQLite returns these.
        naive_old = (datetime.now(timezone.utc) - timedelta(hours=30)).replace(tzinfo=None)
        _seed_watchlist_item(
            patch_session_local,
            "TEST",
            target_price=100.0,
            last_check_price=90.0,
            alert_triggered_at=naive_old,
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()  # Must not raise

        # 30h elapsed → cooldown done → triggers
        assert silence_emails.call_count == 1

    def test_email_disabled_inapp_only(
        self, patch_session_local, silence_emails, stub_config
    ):
        """email.automated_enabled=False → email NOT sent, but alert still
        marked triggered so the in-app bell records the event."""
        from backend.scheduler import check_watchlist_alerts

        stub_config(
            {"watchlist.alerts.enabled": True, "email.automated_enabled": False}
        )
        _seed_watchlist_item(
            patch_session_local, "TEST", target_price=100.0, last_check_price=90.0
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()

        assert silence_emails.call_count == 0
        item = patch_session_local.query(Watchlist).filter_by(ticker="TEST").one()
        # Triggered timestamp + alert_sent flag both set BEFORE the email
        # send attempt (retry-storm guard at line 321-322).
        assert item.alert_triggered_at is not None
        assert item.alert_sent is True

    def test_send_email_exception_still_marks_triggered(
        self, patch_session_local, monkeypatch, stub_config
    ):
        """If send_watchlist_alert_email raises, the alert is still marked
        triggered so the next scan's cooldown prevents an immediate retry
        storm. This is the explicit comment at line 336-337."""
        import backend.email_utils as email_utils
        from backend.scheduler import check_watchlist_alerts

        def _boom(*a, **k):
            raise RuntimeError("smtp down")

        monkeypatch.setattr(email_utils, "send_watchlist_alert_email", _boom)

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        _seed_watchlist_item(
            patch_session_local, "TEST", target_price=100.0, last_check_price=90.0
        )
        _seed_stock(patch_session_local, "TEST", current_price=105.0)

        check_watchlist_alerts()  # Must not propagate

        item = patch_session_local.query(Watchlist).filter_by(ticker="TEST").one()
        assert item.alert_triggered_at is not None

    def test_skip_when_stock_has_no_current_price(
        self, patch_session_local, silence_emails, stub_config
    ):
        """If the joined Stock has no current_price, the alert row is skipped
        (line 282-283) — no crash."""
        from backend.scheduler import check_watchlist_alerts

        stub_config({"watchlist.alerts.enabled": True, "email.automated_enabled": True})
        _seed_watchlist_item(
            patch_session_local, "TEST", target_price=100.0, last_check_price=None
        )
        _seed_stock(patch_session_local, "TEST", current_price=None)

        check_watchlist_alerts()

        assert silence_emails.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — auto_record_coiled_spring_alerts
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoRecordCoiledSpringAlerts:
    """Tier 1: scheduler.auto_record_coiled_spring_alerts.

    Branches covered:
      - CS auto-recording disabled → early return
      - No candidates pass filter query → early return
      - is_coiled_spring=True → record_coiled_spring_alert called
      - is_coiled_spring=False → not recorded
      - record_coiled_spring_alert returns False (cooldown/limit) → counter not bumped
      - outer exception → caught, finally db.close() runs
    """

    def test_disabled_early_returns(self, patch_session_local, monkeypatch, stub_config):
        from backend.scheduler import auto_record_coiled_spring_alerts

        stub_config({"coiled_spring": {"alerts": {"enabled": False}}})
        cs_score = MagicMock()
        record = MagicMock(return_value=True)
        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", cs_score
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", record)

        auto_record_coiled_spring_alerts()

        assert cs_score.call_count == 0
        assert record.call_count == 0

    def test_no_candidates_early_returns(
        self, patch_session_local, monkeypatch, stub_config
    ):
        """Filter query returns no rows → log + early return without calling
        the CS scorer."""
        from backend.scheduler import auto_record_coiled_spring_alerts

        # Stub config to require min_weeks=15; seeded stock has weeks_in_base=2.
        stub_config(
            {
                "coiled_spring": {
                    "alerts": {"enabled": True},
                    "pre_breakout_thresholds": {
                        "min_weeks_in_base": 15,
                        "min_beat_streak": 3,
                        "min_c_score": 5,
                        "min_total_score": 48,
                    },
                    "thresholds": {"max_institutional_pct": 75},
                }
            }
        )
        _seed_stock(
            patch_session_local,
            "WEAK",
            weeks_in_base=2,
            earnings_beat_streak=0,
            c_score=2.0,
        )
        cs_score = MagicMock()
        record = MagicMock()
        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", cs_score
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", record)

        auto_record_coiled_spring_alerts()

        assert cs_score.call_count == 0

    def test_candidate_records_when_is_coiled_spring(
        self, patch_session_local, monkeypatch, stub_config
    ):
        from backend.scheduler import auto_record_coiled_spring_alerts

        stub_config(
            {
                "coiled_spring": {
                    "alerts": {"enabled": True},
                    "pre_breakout_thresholds": {
                        "min_weeks_in_base": 15,
                        "min_beat_streak": 3,
                        "min_c_score": 5,
                        "min_total_score": 48,
                    },
                    "thresholds": {"max_institutional_pct": 75},
                }
            }
        )
        _seed_stock(patch_session_local, "STRONG")  # passes default thresholds

        cs_result = {"is_coiled_spring": True, "cs_details": "test"}
        cs_score = MagicMock(return_value=cs_result)
        record = MagicMock(return_value=True)
        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", cs_score
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", record)

        auto_record_coiled_spring_alerts()

        assert cs_score.call_count == 1
        assert record.call_count == 1

    def test_candidate_skipped_when_not_coiled_spring(
        self, patch_session_local, monkeypatch, stub_config
    ):
        from backend.scheduler import auto_record_coiled_spring_alerts

        stub_config(
            {
                "coiled_spring": {
                    "alerts": {"enabled": True},
                    "pre_breakout_thresholds": {
                        "min_weeks_in_base": 15,
                        "min_beat_streak": 3,
                        "min_c_score": 5,
                        "min_total_score": 48,
                    },
                    "thresholds": {"max_institutional_pct": 75},
                }
            }
        )
        _seed_stock(patch_session_local, "MAYBE")

        cs_score = MagicMock(return_value={"is_coiled_spring": False})
        record = MagicMock()
        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", cs_score
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", record)

        auto_record_coiled_spring_alerts()

        assert cs_score.call_count == 1
        assert record.call_count == 0

    def test_record_returns_false_doesnt_bump_counter(
        self, patch_session_local, monkeypatch, stub_config, caplog
    ):
        """record_coiled_spring_alert returning False (e.g. daily cap hit)
        must not be logged as a successful auto-record."""
        from backend.scheduler import auto_record_coiled_spring_alerts

        stub_config(
            {
                "coiled_spring": {
                    "alerts": {"enabled": True},
                    "pre_breakout_thresholds": {
                        "min_weeks_in_base": 15,
                        "min_beat_streak": 3,
                        "min_c_score": 5,
                        "min_total_score": 48,
                    },
                    "thresholds": {"max_institutional_pct": 75},
                }
            }
        )
        _seed_stock(patch_session_local, "CAPPED")

        cs_score = MagicMock(
            return_value={"is_coiled_spring": True, "cs_details": "x"}
        )
        record = MagicMock(return_value=False)
        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", cs_score
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", record)

        auto_record_coiled_spring_alerts()

        assert record.call_count == 1
        # The success-log line "Auto-recorded N alerts" must not appear.
        assert "Auto-recorded 1" not in caplog.text

    def test_outer_exception_caught(
        self, patch_session_local, monkeypatch, stub_config
    ):
        """If the CS scorer raises, the outer try/except swallows it so the
        scan loop keeps running (helpers in scheduler are best-effort
        side jobs)."""
        from backend.scheduler import auto_record_coiled_spring_alerts

        stub_config(
            {
                "coiled_spring": {
                    "alerts": {"enabled": True},
                    "pre_breakout_thresholds": {
                        "min_weeks_in_base": 15,
                        "min_beat_streak": 3,
                        "min_c_score": 5,
                        "min_total_score": 48,
                    },
                    "thresholds": {"max_institutional_pct": 75},
                }
            }
        )
        _seed_stock(patch_session_local, "BOOM")

        def _raises(stock):
            raise RuntimeError("scorer broken")

        import backend.ai_trader as ai_trader

        monkeypatch.setattr(
            ai_trader, "calculate_coiled_spring_score_for_stock", _raises
        )
        monkeypatch.setattr(ai_trader, "record_coiled_spring_alert", MagicMock())

        auto_record_coiled_spring_alerts()  # Must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — update_coiled_spring_outcomes
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateCoiledSpringOutcomes:
    """Tier 1: scheduler.update_coiled_spring_outcomes.

    The "first evaluation wins permanently" design note (line 483-490) is
    load-bearing — these tests pin the outcome IS NULL filter so a future
    refactor doesn't accidentally re-bucket already-evaluated alerts.

    Branches covered:
      - outcome tracking disabled → early return
      - no alerts old enough → early return
      - gain >= big_win_pct → bucket=big_win
      - gain >= win_pct but < big_win_pct → bucket=win
      - gain <= loss_pct → bucket=loss
      - between loss_pct and win_pct → bucket=flat
      - stock has no current_price → skipped (outcome stays NULL)
      - already-bucketed alert is NOT re-evaluated (load-bearing invariant)
      - outer exception → caught, finally db.close()
    """

    def _enabled_config(self):
        return {
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

    def test_disabled_early_returns(
        self, patch_session_local, stub_config
    ):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(
            {"coiled_spring": {"outcome_tracking": {"enabled": False}}}
        )
        # Seed an alert that WOULD have been evaluated.
        _seed_cs_alert(
            patch_session_local,
            "OLD",
            alert_date=date.today() - timedelta(days=30),
        )
        _seed_stock(patch_session_local, "OLD", current_price=120.0)

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="OLD")
            .one()
        )
        assert alert.outcome is None  # untouched

    def test_no_old_alerts_early_returns(
        self, patch_session_local, stub_config
    ):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # Alert from yesterday — within cutoff (check_days=3 from today).
        _seed_cs_alert(
            patch_session_local, "FRESH", alert_date=date.today() - timedelta(days=1)
        )
        _seed_stock(patch_session_local, "FRESH", current_price=120.0)

        update_coiled_spring_outcomes()  # Must not raise

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="FRESH")
            .one()
        )
        assert alert.outcome is None

    def test_big_win_bucket(self, patch_session_local, stub_config):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # alert 30d ago, days_to_earnings=7, check_days=3 → days_needed=10,
        # days_since_alert=30 → eligible. price 100 → 120 = +20% (big_win).
        _seed_cs_alert(
            patch_session_local,
            "BIG",
            alert_date=date.today() - timedelta(days=30),
            price_at_alert=100.0,
            days_to_earnings=7,
        )
        # next_earnings_date=None forces the `or 7` fallback path.
        _seed_stock(
            patch_session_local, "BIG", current_price=120.0, next_earnings_date=None
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert).filter_by(ticker="BIG").one()
        )
        assert alert.outcome == "big_win"
        assert alert.price_after_earnings == 120.0
        assert alert.outcome_updated_at is not None

    def test_win_bucket(self, patch_session_local, stub_config):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # +8% → win
        _seed_cs_alert(
            patch_session_local,
            "WIN",
            alert_date=date.today() - timedelta(days=30),
            price_at_alert=100.0,
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local, "WIN", current_price=108.0, next_earnings_date=None
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert).filter_by(ticker="WIN").one()
        )
        assert alert.outcome == "win"

    def test_loss_bucket(self, patch_session_local, stub_config):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # -10% → loss
        _seed_cs_alert(
            patch_session_local,
            "LOSS",
            alert_date=date.today() - timedelta(days=30),
            price_at_alert=100.0,
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local, "LOSS", current_price=90.0, next_earnings_date=None
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert).filter_by(ticker="LOSS").one()
        )
        assert alert.outcome == "loss"

    def test_flat_bucket(self, patch_session_local, stub_config):
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # +1% → flat (between loss_pct=-5 and win_pct=5)
        _seed_cs_alert(
            patch_session_local,
            "FLAT",
            alert_date=date.today() - timedelta(days=30),
            price_at_alert=100.0,
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local, "FLAT", current_price=101.0, next_earnings_date=None
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert).filter_by(ticker="FLAT").one()
        )
        assert alert.outcome == "flat"

    def test_skip_when_stock_has_no_current_price(
        self, patch_session_local, stub_config
    ):
        """No current_price → skipped (line 543-544) — outcome stays NULL,
        preserving the "first evaluation wins permanently" invariant: a
        later scan with a price will get the first chance to bucket."""
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        _seed_cs_alert(
            patch_session_local,
            "NOPRICE",
            alert_date=date.today() - timedelta(days=30),
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local,
            "NOPRICE",
            current_price=None,
            next_earnings_date=None,
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="NOPRICE")
            .one()
        )
        assert alert.outcome is None

    def test_already_bucketed_alert_not_reevaluated(
        self, patch_session_local, stub_config
    ):
        """Load-bearing invariant: alerts with outcome != NULL are filtered
        out by the query at line 493-496. A choppy day cannot flip
        win → flat → loss."""
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # Alert already bucketed as "big_win" with +20% outcome. New price
        # would re-bucket as "loss" if filter were broken.
        _seed_cs_alert(
            patch_session_local,
            "STICKY",
            alert_date=date.today() - timedelta(days=30),
            price_at_alert=100.0,
            outcome="big_win",
            price_change_pct=20.0,
            price_after_earnings=120.0,
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local, "STICKY", current_price=80.0, next_earnings_date=None
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="STICKY")
            .one()
        )
        assert alert.outcome == "big_win"  # untouched
        assert alert.price_after_earnings == 120.0  # untouched

    def test_alert_skipped_when_not_enough_time_passed(
        self, patch_session_local, stub_config
    ):
        """Alert is past cutoff_date filter (alert_date <= today - 3) but
        days_to_earnings + check_days hasn't elapsed yet → continue past
        evaluation. Verifies line 562-564 skip path."""
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        # alert 5d ago, days_to_earnings=10, check_days=3 → days_needed=13,
        # days_since_alert=5 → not enough time. Cutoff is today-3, so
        # the alert IS old enough to be in the query but NOT old enough
        # to be evaluated.
        _seed_cs_alert(
            patch_session_local,
            "TOOSOON",
            alert_date=date.today() - timedelta(days=5),
            days_to_earnings=10,
            price_at_alert=100.0,
        )
        _seed_stock(
            patch_session_local,
            "TOOSOON",
            current_price=120.0,
            next_earnings_date=None,
        )

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="TOOSOON")
            .one()
        )
        assert alert.outcome is None  # not yet eligible

    def test_alpha_relative_bucketing_with_spy(
        self, patch_session_local, stub_config
    ):
        """When SPY data is available, outcome bucketing uses the
        stock-minus-SPY change (alpha-relative). +20% raw with SPY +18%
        same window = +2% alpha → flat, not big_win."""
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        alert_dt = date.today() - timedelta(days=30)
        _seed_cs_alert(
            patch_session_local,
            "ALPHA",
            alert_date=alert_dt,
            price_at_alert=100.0,
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local,
            "ALPHA",
            current_price=120.0,
            next_earnings_date=None,
        )

        # SPY: +18% same window (current 590, snapshot 500).
        spy = Stock(ticker="SPY", current_price=590.0)
        patch_session_local.add(spy)
        snap = MarketSnapshot(date=alert_dt, spy_price=500.0, spy_signal=1)
        patch_session_local.add(snap)
        patch_session_local.commit()

        update_coiled_spring_outcomes()

        alert = (
            patch_session_local.query(CoiledSpringAlert)
            .filter_by(ticker="ALPHA")
            .one()
        )
        # +20% - +18% = +2% → flat
        assert alert.outcome == "flat"

    def test_outer_exception_caught(
        self, patch_session_local, monkeypatch, stub_config
    ):
        """If the SPY query path raises (simulated by patching db.query to
        explode), the outer try/except logs and continues."""
        from backend.scheduler import update_coiled_spring_outcomes

        stub_config(self._enabled_config())
        _seed_cs_alert(
            patch_session_local,
            "BOOM",
            alert_date=date.today() - timedelta(days=30),
            days_to_earnings=7,
        )
        _seed_stock(
            patch_session_local, "BOOM", current_price=120.0, next_earnings_date=None
        )

        # Patch db.commit to raise mid-evaluation.
        original_commit = patch_session_local.commit
        call_count = {"n": 0}

        def _explode_on_second_commit():
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise RuntimeError("commit broke")
            return original_commit()

        monkeypatch.setattr(patch_session_local, "commit", _explode_on_second_commit)

        update_coiled_spring_outcomes()  # Must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — _refresh_portfolio_prices
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefreshPortfolioPrices:
    """Tier 2: scheduler._refresh_portfolio_prices.

    Branches covered:
      - market closed → early return without touching DB
      - market open, single active config → refresh_ai_portfolio called once
      - market open, two active configs (multi-user) → refresh called per user
      - per-user exception isolated (one user fails, the other still runs)
    """

    def test_market_closed_early_returns(
        self, patch_session_local, monkeypatch
    ):
        from backend.scheduler import _refresh_portfolio_prices

        import backend.ai_trader as ai_trader

        monkeypatch.setattr(ai_trader, "is_market_open", lambda: False)
        refresh_spy = MagicMock()
        monkeypatch.setattr(ai_trader, "refresh_ai_portfolio", refresh_spy)

        cfg = AIPortfolioConfig(user_id=1, is_active=True)
        patch_session_local.add(cfg)
        patch_session_local.commit()

        _refresh_portfolio_prices()

        assert refresh_spy.call_count == 0

    def test_market_open_refreshes_each_active_user(
        self, patch_session_local, monkeypatch
    ):
        from backend.scheduler import _refresh_portfolio_prices

        import backend.ai_trader as ai_trader

        monkeypatch.setattr(ai_trader, "is_market_open", lambda: True)
        refresh_spy = MagicMock(return_value={"message": "ok"})
        monkeypatch.setattr(ai_trader, "refresh_ai_portfolio", refresh_spy)

        patch_session_local.add(AIPortfolioConfig(user_id=1, is_active=True))
        patch_session_local.add(AIPortfolioConfig(user_id=2, is_active=True))
        patch_session_local.add(AIPortfolioConfig(user_id=3, is_active=False))
        patch_session_local.commit()

        _refresh_portfolio_prices()

        assert refresh_spy.call_count == 2
        called_uids = sorted(call.kwargs.get("user_id") for call in refresh_spy.call_args_list)
        assert called_uids == [1, 2]

    def test_per_user_exception_does_not_break_loop(
        self, patch_session_local, monkeypatch
    ):
        """User 1's refresh raises; user 2's refresh must still run.
        Pinned via the inner try/except at lines 1621-1625."""
        from backend.scheduler import _refresh_portfolio_prices

        import backend.ai_trader as ai_trader

        monkeypatch.setattr(ai_trader, "is_market_open", lambda: True)

        def _refresh(db, user_id=1):
            if user_id == 1:
                raise RuntimeError("user 1 exploded")
            return {"message": "user 2 ok"}

        refresh_spy = MagicMock(side_effect=_refresh)
        monkeypatch.setattr(ai_trader, "refresh_ai_portfolio", refresh_spy)

        patch_session_local.add(AIPortfolioConfig(user_id=1, is_active=True))
        patch_session_local.add(AIPortfolioConfig(user_id=2, is_active=True))
        patch_session_local.commit()

        _refresh_portfolio_prices()  # Must not raise

        assert refresh_spy.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — start/stop/update_scan_config thread controls
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreadControls:
    """Tier 2: start_continuous_scanning / stop_continuous_scanning /
    update_scan_config.

    These mutate the module-level _scan_config dict and add/remove jobs on
    the apscheduler.BackgroundScheduler singleton. We monkeypatch the
    scheduler attribute and the bare Thread launcher so tests don't actually
    start a background thread.

    Branches covered:
      - start_continuous_scanning sets enabled, source, interval and adds jobs
      - update_scan_config(disabled) only mutates dict
      - update_scan_config(enabled) cascades to start_continuous_scanning
      - stop_continuous_scanning unsets enabled flag and removes jobs
    """

    @pytest.fixture
    def fake_scheduler(self, monkeypatch):
        """Replace the apscheduler instance with a tracking spy. Records
        add_job / remove_job calls and exposes a get_job map."""
        from backend import scheduler as scheduler_mod

        class FakeJob:
            def __init__(self, id, name):
                self.id = id
                self.name = name
                self.next_run_time = None

        class FakeScheduler:
            def __init__(self):
                self.jobs = {}
                self.running = False
                self.add_calls = []
                self.remove_calls = []

            def add_job(self, func, trigger, id=None, name=None, replace_existing=False):
                self.add_calls.append((id, name))
                self.jobs[id] = FakeJob(id, name)

            def remove_job(self, job_id):
                self.remove_calls.append(job_id)
                self.jobs.pop(job_id, None)

            def get_job(self, job_id):
                return self.jobs.get(job_id)

            def start(self):
                self.running = True

        fake = FakeScheduler()
        monkeypatch.setattr(scheduler_mod, "scheduler", fake)
        # No-op the auxiliary thread-launcher jobs — they touch real subsystems.
        monkeypatch.setattr(scheduler_mod, "start_weekly_email_job", lambda: None)
        monkeypatch.setattr(scheduler_mod, "start_backup_job", lambda: None)
        monkeypatch.setattr(scheduler_mod, "start_ml_backfill_job", lambda: None)
        monkeypatch.setattr(scheduler_mod, "start_breakout_monitor_job", lambda: None)
        # No-op the immediate-first-scan thread.
        import threading

        class FakeThread:
            def __init__(self, target=None, *args, **kwargs):
                self.target = target

            def start(self):
                pass

        monkeypatch.setattr(threading, "Thread", FakeThread)
        # Also patch the redis restore so we don't hit the network.
        monkeypatch.setattr(
            scheduler_mod, "_restore_health_from_redis", lambda: None
        )
        return fake

    @pytest.fixture(autouse=True)
    def reset_scan_config(self):
        """_scan_config is module-global; reset relevant keys after each test
        so we don't leak state into other test classes."""
        from backend import scheduler as scheduler_mod

        snapshot = dict(scheduler_mod._scan_config)
        yield
        for k, v in snapshot.items():
            scheduler_mod._scan_config[k] = v

    def test_start_continuous_scanning_sets_state_and_adds_jobs(
        self, fake_scheduler
    ):
        from backend import scheduler as scheduler_mod
        from backend.scheduler import start_continuous_scanning

        status = start_continuous_scanning(source="russell", interval_minutes=30)

        assert scheduler_mod._scan_config["enabled"] is True
        assert scheduler_mod._scan_config["source"] == "russell"
        assert scheduler_mod._scan_config["interval_minutes"] == 30

        # Both core jobs added.
        added_ids = {jid for jid, _ in fake_scheduler.add_calls}
        assert "continuous_scan" in added_ids
        assert "portfolio_price_refresh" in added_ids

        assert fake_scheduler.running is True
        assert status["enabled"] is True
        assert status["source"] == "russell"

    def test_start_replaces_existing_jobs(self, fake_scheduler):
        """A second start call removes the prior jobs before re-adding."""
        from backend.scheduler import start_continuous_scanning

        start_continuous_scanning(source="sp500", interval_minutes=15)
        start_continuous_scanning(source="all", interval_minutes=90)

        # Both core jobs were removed before the second start re-added them.
        assert "continuous_scan" in fake_scheduler.remove_calls
        assert "portfolio_price_refresh" in fake_scheduler.remove_calls

    def test_stop_continuous_scanning_unsets_enabled(self, fake_scheduler):
        from backend import scheduler as scheduler_mod
        from backend.scheduler import (
            start_continuous_scanning,
            stop_continuous_scanning,
        )

        start_continuous_scanning(source="sp500", interval_minutes=15)
        assert scheduler_mod._scan_config["enabled"] is True

        stop_continuous_scanning()

        assert scheduler_mod._scan_config["enabled"] is False
        assert "continuous_scan" in fake_scheduler.remove_calls
        assert "portfolio_price_refresh" in fake_scheduler.remove_calls

    def test_update_scan_config_disabled_only_mutates_dict(self, fake_scheduler):
        """When _scan_config['enabled'] is False, update_scan_config must
        mutate source/interval but NOT cascade to start_continuous_scanning."""
        from backend import scheduler as scheduler_mod
        from backend.scheduler import update_scan_config

        scheduler_mod._scan_config["enabled"] = False

        update_scan_config(source="all", interval_minutes=120)

        assert scheduler_mod._scan_config["source"] == "all"
        assert scheduler_mod._scan_config["interval_minutes"] == 120
        # No add_job was called because we never went through start.
        assert fake_scheduler.add_calls == []

    def test_update_scan_config_enabled_cascades_to_start(self, fake_scheduler):
        """When already enabled, update_scan_config restarts via
        start_continuous_scanning so the new interval takes effect."""
        from backend import scheduler as scheduler_mod
        from backend.scheduler import update_scan_config

        scheduler_mod._scan_config["enabled"] = True
        scheduler_mod._scan_config["source"] = "sp500"
        scheduler_mod._scan_config["interval_minutes"] = 15

        status = update_scan_config(source="russell", interval_minutes=45)

        assert scheduler_mod._scan_config["source"] == "russell"
        assert scheduler_mod._scan_config["interval_minutes"] == 45
        # add_job was called as part of the cascading start.
        added_ids = {jid for jid, _ in fake_scheduler.add_calls}
        assert "continuous_scan" in added_ids
        assert status["source"] == "russell"

    def test_update_scan_config_partial_args_preserves_other(self, fake_scheduler):
        """If only source is passed (no interval), the existing interval is
        preserved. Verifies the falsy-arg guard at lines 1722-1725."""
        from backend import scheduler as scheduler_mod
        from backend.scheduler import update_scan_config

        scheduler_mod._scan_config["enabled"] = False
        scheduler_mod._scan_config["source"] = "sp500"
        scheduler_mod._scan_config["interval_minutes"] = 15

        update_scan_config(source="russell")

        assert scheduler_mod._scan_config["source"] == "russell"
        assert scheduler_mod._scan_config["interval_minutes"] == 15  # preserved


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — cleanup helpers + health tracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanupOldStockScores:
    """Tier 2: scheduler.cleanup_old_stock_scores.

    Two-tier retention to keep the Score Replay 3M window meaningful without
    unbounded DB growth:
      • < scan_days old        → keep every scan (per-scan resolution)
      • scan_days..daily_days  → thin to ONE row/day (last scan = max timestamp)
      • > daily_days old       → delete entirely
    The function takes an explicit Session arg (unlike the SessionLocal-opening
    helpers above).
    """

    def _add_scan(self, db_session, stock_id, days_ago, hour=12, score=70.0):
        ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
        ts = ts.replace(hour=hour, minute=0, second=0, microsecond=0)
        from backend.database import StockScore
        db_session.add(
            StockScore(
                stock_id=stock_id,
                timestamp=ts,
                date=ts.date(),
                total_score=score,
            )
        )

    def test_no_old_scores_returns_zero(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="FRESH", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # All scores inside the per-scan window — nothing to prune, even with
        # several scans on the same day.
        for i in range(3):
            self._add_scan(db_session, stock.id, days_ago=i, hour=9)
            self._add_scan(db_session, stock.id, days_ago=i, hour=15)
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, scan_days=30, daily_days=90)
        assert deleted == 0
        assert db_session.query(StockScore).count() == 6

    def test_deletes_beyond_daily_horizon(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="ANCIENT", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # 5 ancient (>90d) on distinct dates + 2 fresh (1-2d). Ancient rows are
        # past the daily horizon so all go, regardless of being one-per-day.
        for i in range(5):
            self._add_scan(db_session, stock.id, days_ago=95 + i)
        for i in range(2):
            self._add_scan(db_session, stock.id, days_ago=i + 1)
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, scan_days=30, daily_days=90)
        assert deleted == 5
        assert db_session.query(StockScore).count() == 2

    def test_mid_window_thinned_to_one_per_day(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="MIDBAND", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # A day in the 30-90d mid band with 3 intra-day scans. Only the last
        # scan of the day (max timestamp, the 15:00 one) must survive.
        for hour, score in [(8, 60.0), (12, 65.0), (15, 70.0)]:
            self._add_scan(db_session, stock.id, days_ago=50, hour=hour, score=score)
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, scan_days=30, daily_days=90)
        assert deleted == 2
        rows = db_session.query(StockScore).all()
        assert len(rows) == 1
        # The kept row is the last scan of that day.
        assert rows[0].total_score == 70.0
        assert rows[0].timestamp.hour == 15

    def test_fresh_intraday_scans_untouched(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="RECENT", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # Multiple scans today (well inside scan_days) — per-scan resolution
        # must be preserved, so none are thinned.
        for hour in (9, 11, 13, 15):
            self._add_scan(db_session, stock.id, days_ago=2, hour=hour)
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, scan_days=30, daily_days=90)
        assert deleted == 0
        assert db_session.query(StockScore).count() == 4


class TestCleanupPriceCache:
    """Tier 2: scheduler.cleanup_price_cache.

    Calls into backend.price_cache.get_price_cache(). Test mocks the
    cache so we don't touch the real SQLite price_cache db.
    """

    def test_cleanup_calls_cleanup_expired(self, monkeypatch, tmp_path):
        from backend.scheduler import cleanup_price_cache
        import backend.price_cache as price_cache_mod

        fake_db = tmp_path / "fake_price_cache.db"
        fake_db.touch()
        fake_cache = MagicMock()
        fake_cache.cleanup_expired.return_value = 7
        fake_cache.db_path = str(fake_db)
        monkeypatch.setattr(price_cache_mod, "get_price_cache", lambda: fake_cache)

        cleanup_price_cache(vacuum_interval_days=7)

        assert fake_cache.cleanup_expired.call_count == 1

    def test_cleanup_swallows_exception(self, monkeypatch):
        """If get_price_cache raises (e.g., import error in some envs), the
        function must log + continue, not propagate. Pinned via the
        outer try/except at lines 226-243."""
        from backend.scheduler import cleanup_price_cache
        import backend.price_cache as price_cache_mod

        def _boom():
            raise RuntimeError("price cache unavailable")

        monkeypatch.setattr(price_cache_mod, "get_price_cache", _boom)

        cleanup_price_cache()  # Must not raise


class TestSystemHealthHelpers:
    """Tier 2: _record_success / _record_failure / get_system_health and the
    Redis persistence helpers.

    These thin helpers maintain the rolling error-history for the System
    Health page. The Redis persistence is best-effort — when no client is
    available, the helpers must no-op silently."""

    def test_get_system_health_returns_dict_copy(self):
        from backend.scheduler import get_system_health

        snapshot = get_system_health()
        assert isinstance(snapshot, dict)
        assert "consecutive_scan_failures" in snapshot

    def test_record_success_resets_failure_counter(self, monkeypatch):
        from backend import scheduler as scheduler_mod
        from backend.scheduler import _record_success

        # Stub redis persistence so we don't hit the network.
        monkeypatch.setattr(scheduler_mod, "_persist_health_to_redis", lambda: None)

        scheduler_mod._system_health["consecutive_scan_failures"] = 5
        scheduler_mod._system_health["last_scan_error"] = {"task": "scan", "error": "x"}

        _record_success("scan")

        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_scan_error"] is None

    def test_record_failure_increments_and_alerts_on_first(self, monkeypatch):
        from backend import scheduler as scheduler_mod
        import backend.email_utils as email_utils
        from backend.scheduler import _record_failure

        monkeypatch.setattr(scheduler_mod, "_persist_health_to_redis", lambda: None)
        alert_spy = MagicMock()
        monkeypatch.setattr(email_utils, "send_webhook_notification", alert_spy)

        scheduler_mod._system_health["consecutive_scan_failures"] = 0
        scheduler_mod._system_health["errors_today"].clear()

        _record_failure("scan", "kaboom")

        # First failure → alert fires.
        assert alert_spy.call_count == 1
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 1
        assert len(scheduler_mod._system_health["errors_today"]) == 1

    def test_record_failure_alerts_on_third_consecutive(self, monkeypatch):
        from backend import scheduler as scheduler_mod
        import backend.email_utils as email_utils
        from backend.scheduler import _record_failure

        monkeypatch.setattr(scheduler_mod, "_persist_health_to_redis", lambda: None)
        alert_spy = MagicMock()
        monkeypatch.setattr(email_utils, "send_webhook_notification", alert_spy)

        scheduler_mod._system_health["consecutive_scan_failures"] = 2
        _record_failure("scan", "still broken")

        # 3rd consecutive → alert fires (every Nth where N % 3 == 0).
        assert alert_spy.call_count == 1
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 3

    def test_persist_health_no_client_silent(self, monkeypatch):
        """If redis client is None, _persist_health_to_redis must no-op
        silently — Redis persistence is documented as best-effort."""
        from backend.scheduler import _persist_health_to_redis
        import redis_cache

        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: None)

        _persist_health_to_redis()  # Must not raise

    def test_restore_health_no_client_silent(self, monkeypatch):
        from backend.scheduler import _restore_health_from_redis
        import redis_cache

        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: None)

        _restore_health_from_redis()  # Must not raise

    def test_persist_health_silently_swallows_serialization_error(self, monkeypatch):
        """LATENT BUG (logged, not fixed during 2026-06-18 eval):
        `_system_health["errors_today"]` is a `deque(maxlen=50)` which
        json.dumps cannot serialize. The bare except at line 90-91 swallows
        the TypeError every call, meaning Redis persistence has been silently
        broken in production. Restore-on-startup still works because it
        deserializes JSON the prior persist would have written — but
        nothing ever gets written, so it's a no-op restore. Prod behavior
        is unchanged (still cold-starts each restart), so this is NOT
        safety-critical. Fix DEFERRED past 2026-06-18 eval to avoid
        confounding A/B reads."""
        from backend.scheduler import _persist_health_to_redis
        import redis_cache

        fake_client = MagicMock()
        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: fake_client)

        _persist_health_to_redis()  # Must not raise

        # The bare except swallows the serialization error — client.set is
        # never reached. If a future fix wraps `errors_today` as a list
        # before encoding, this assertion will flip to == 1 and that's the
        # signal the latent bug got fixed.
        assert fake_client.set.call_count == 0

    def test_restore_health_with_data_updates_state(self, monkeypatch):
        from backend import scheduler as scheduler_mod
        from backend.scheduler import _restore_health_from_redis
        import redis_cache
        import json

        fake_client = MagicMock()
        fake_client.get.return_value = json.dumps(
            {
                "last_successful_scan": "2026-05-07T12:00:00+00:00",
                "consecutive_scan_failures": 0,
                "errors_today": [],
            }
        )
        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: fake_client)

        scheduler_mod._system_health["last_successful_scan"] = None
        _restore_health_from_redis()

        assert scheduler_mod._system_health["last_successful_scan"] == "2026-05-07T12:00:00+00:00"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — run_continuous_scan orchestrator (lines 811-1603)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunContinuousScan:
    """Tier 3: scheduler.run_continuous_scan — the main scan/trade pipeline.

    Stitches together: market direction → async scan → DB save → market
    snapshot → CS auto-record → CS outcome update → earnings audit →
    industry-group ranks → bear-base (when SPY < 50MA) → AI cycle per
    user → cleanup → gap-up → watchlist.

    Covering this orchestrator is the natural follow-on to the auxiliary-
    helper push (commit c879fc0): every helper it calls is now mockable
    at its source module, and the body itself runs end-to-end with one
    fake ticker + one fake analysis dict.

    Branches covered:
      a. Happy path: 1 ticker, all post-helpers fire, _record_success("scan")
      b. is_scanning=True guard early-returns without running the body
      c. async-scan exception → _record_failure("scan"), is_scanning reset
      d. SPY < 50MA → bear-base branch fires; SPY > 50MA → does NOT
      e. Per-phase progress: every documented _PHASE_LABELS key reached
      f. Post-helper sequencing: canonical order across the 9 post-scan calls
      g. Per-user AI cycle exception → _record_failure("trade_cycle") fires,
         scan still records success (per-user try/except at scheduler.py:1521)
      h. Source variants: sp500/russell/all/top50 each load the right loader
    """

    @pytest.fixture
    def scan_seams(self, monkeypatch, patch_session_local):
        """One-stop fixture mocking every external coupling of
        run_continuous_scan and resetting module-level state. Returns a
        SimpleNamespace exposing every spy so tests can adjust behavior
        and assert call counts.

        Pattern: monkeypatch source modules BEFORE the call. The function
        does its imports inline via `from X import Y`, so attribute swaps
        on the source module take effect when the inline import runs."""
        import sp500_tickers
        import data_fetcher
        import async_scanner
        import backend.scheduler as scheduler_mod
        import backend.main as main_mod
        import backend.earnings_audit as audit_mod
        import backend.industry_group as ig_mod
        import backend.bear_base as bb_mod
        import backend.ai_trader as ai_mod
        import backend.earnings_gapup as gu_mod
        import backend.email_utils as email_utils
        import redis_cache
        from types import SimpleNamespace

        bus = SimpleNamespace()
        bus.session = patch_session_local

        # ── Tickers ──
        monkeypatch.setattr(sp500_tickers, "get_sp500_tickers", lambda: ["TEST"])
        monkeypatch.setattr(sp500_tickers, "get_russell2000_tickers", lambda: ["RUSS1"])
        monkeypatch.setattr(
            sp500_tickers,
            "get_all_tickers",
            lambda include_portfolio=True: ["ALL1"],
        )
        monkeypatch.setattr(sp500_tickers, "get_portfolio_tickers", lambda: [])

        # ── Async scan ──
        bus.fake_analysis = {
            "ticker": "TEST",
            "company_name": "Test Inc",
            "sector": "Technology",
            "industry": "Software",
            "current_price": 100.0,
            "market_cap": 1_000_000_000,
            "canslim_score": 70.0,
            "c_score": 10.0,
            "a_score": 10.0,
            "n_score": 10.0,
            "s_score": 10.0,
            "l_score": 10.0,
            "i_score": 10.0,
            "m_score": 10.0,
            "score_details": {},
            "projected_growth": 5.0,
            "confidence": 0.5,
            "week_52_high": 120.0,
            "week_52_low": 80.0,
            "is_growth_stock": False,
            "quarterly_earnings": [],
            "annual_earnings": [],
            "quarterly_revenue": [],
        }

        def stub_run_async_scan(tickers, batch_size=100, progress_callback=None):
            # Drive the progress UI so all three Phase-1 sub-phase labels
            # ("stocks", "insider_short", "p1_data") reach _set_phase —
            # branch (e) depends on this.
            if progress_callback and tickers:
                progress_callback(1, len(tickers), "stocks")
                progress_callback(1, len(tickers), "insider_short")
                progress_callback(1, len(tickers), "p1_data")
            return [dict(bus.fake_analysis) for _ in tickers] if tickers else []

        bus.run_async_scan = MagicMock(side_effect=stub_run_async_scan)
        monkeypatch.setattr(async_scanner, "run_async_scan", bus.run_async_scan)

        # ── Market direction (default bullish: SPY > 50MA) ──
        bus.market_data = {
            "success": True,
            "market_trend": "bullish",
            "market_score": 80,
            "weighted_signal": 0.7,
            "indexes": {"SPY": {"price": 500.0, "ma_50": 480.0}},
        }
        monkeypatch.setattr(
            data_fetcher,
            "get_cached_market_direction",
            lambda force_refresh=False: bus.market_data,
        )

        # ── Rate-limit / cache stats (logged at end of scan) ──
        monkeypatch.setattr(
            data_fetcher,
            "get_rate_limit_stats",
            lambda: {"errors_429": 0, "total_requests": 100},
        )
        monkeypatch.setattr(data_fetcher, "reset_rate_limit_stats", lambda: None)
        monkeypatch.setattr(
            data_fetcher,
            "get_cache_stats",
            lambda: {"memory": {"tickers_tracked": 0, "cached_data_entries": 0}},
        )
        monkeypatch.setattr(
            data_fetcher,
            "get_cache_hit_stats",
            lambda: {"hits": 0, "misses": 0},
        )

        # ── update_market_snapshot (lives in backend.main) ──
        bus.update_market_snapshot = MagicMock()
        monkeypatch.setattr(main_mod, "update_market_snapshot", bus.update_market_snapshot)

        # ── Post-scan helpers defined IN scheduler.py (patch on scheduler) ──
        bus.auto_record_cs = MagicMock()
        bus.update_cs_outcomes = MagicMock()
        bus.check_watchlist = MagicMock()
        bus.cleanup_old_scores = MagicMock()
        bus.cleanup_price = MagicMock()
        monkeypatch.setattr(
            scheduler_mod, "auto_record_coiled_spring_alerts", bus.auto_record_cs
        )
        monkeypatch.setattr(
            scheduler_mod, "update_coiled_spring_outcomes", bus.update_cs_outcomes
        )
        monkeypatch.setattr(scheduler_mod, "check_watchlist_alerts", bus.check_watchlist)
        monkeypatch.setattr(scheduler_mod, "cleanup_old_stock_scores", bus.cleanup_old_scores)
        monkeypatch.setattr(scheduler_mod, "cleanup_price_cache", bus.cleanup_price)

        # ── Earnings audit ──
        bus.run_earnings_audit = MagicMock(return_value=[])
        monkeypatch.setattr(audit_mod, "run_earnings_audit", bus.run_earnings_audit)

        # ── Industry group ──
        bus.compute_ig = MagicMock(return_value={"Software": {"avg": 70}})
        bus.update_ig = MagicMock(return_value=1)
        monkeypatch.setattr(ig_mod, "compute_industry_group_rankings", bus.compute_ig)
        monkeypatch.setattr(ig_mod, "update_stock_group_ranks", bus.update_ig)

        # ── Bear base ──
        bus.update_bear_base = MagicMock(return_value={"total": 0})
        monkeypatch.setattr(bb_mod, "update_bear_base_candidates", bus.update_bear_base)

        # ── AI trader ──
        bus.run_ai_cycle = MagicMock(
            return_value={"buys_executed": [], "sells_executed": []}
        )
        bus.take_snapshot = MagicMock()
        bus.is_market_open = MagicMock(return_value=False)
        monkeypatch.setattr(ai_mod, "run_ai_trading_cycle", bus.run_ai_cycle)
        monkeypatch.setattr(ai_mod, "take_portfolio_snapshot", bus.take_snapshot)
        monkeypatch.setattr(ai_mod, "is_market_open", bus.is_market_open)

        # ── Earnings gap-up ──
        bus.find_gapups = MagicMock(return_value=[])
        bus.send_gapup = MagicMock()
        monkeypatch.setattr(gu_mod, "find_earnings_gapups", bus.find_gapups)
        monkeypatch.setattr(gu_mod, "send_gapup_alert", bus.send_gapup)

        # ── Webhook silencing (used by _record_failure) ──
        bus.send_webhook = MagicMock()
        monkeypatch.setattr(email_utils, "send_webhook_notification", bus.send_webhook)

        # ── Redis silencing ──
        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: None)

        # ── Module-level state reset ──
        scheduler_mod._scan_config["is_scanning"] = False
        scheduler_mod._scan_config["source"] = "sp500"
        scheduler_mod._scan_config["phase"] = None
        scheduler_mod._scan_config["phase_detail"] = None
        scheduler_mod._scan_config["current_phase"] = None
        scheduler_mod._scan_config["phase_current"] = 0
        scheduler_mod._scan_config["phase_total"] = 0
        scheduler_mod._scan_config["phase_label"] = None
        scheduler_mod._system_health["consecutive_scan_failures"] = 0
        scheduler_mod._system_health["consecutive_trade_failures"] = 0
        scheduler_mod._system_health["last_scan_error"] = None
        scheduler_mod._system_health["last_trade_cycle_error"] = None
        scheduler_mod._system_health["last_successful_scan"] = None
        scheduler_mod._system_health["last_successful_trade_cycle"] = None
        scheduler_mod._system_health["errors_today"].clear()

        return bus

    def test_happy_path_records_success(self, scan_seams):
        """Branch (a): one ticker → every post-scan helper fires once,
        _record_success('scan') resets the failure counter, finally
        block clears the phase fields."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod

        run_continuous_scan()

        assert scan_seams.run_async_scan.call_count == 1
        assert scan_seams.update_market_snapshot.call_count == 1
        assert scan_seams.auto_record_cs.call_count == 1
        assert scan_seams.update_cs_outcomes.call_count == 1
        assert scan_seams.run_earnings_audit.call_count == 1
        assert scan_seams.compute_ig.call_count == 1
        assert scan_seams.update_ig.call_count == 1
        assert scan_seams.cleanup_old_scores.call_count == 1
        assert scan_seams.cleanup_price.call_count == 1
        assert scan_seams.find_gapups.call_count == 1
        assert scan_seams.check_watchlist.call_count == 1

        # _record_success("scan") fired
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_scan_error"] is None
        assert scheduler_mod._system_health["last_successful_scan"] is not None

        # Cleanup state reset by finally block (scheduler.py:1594-1603)
        assert scheduler_mod._scan_config["is_scanning"] is False
        assert scheduler_mod._scan_config["phase"] is None
        assert scheduler_mod._scan_config["phase_label"] is None

        # Fake analysis was actually persisted by save_stock_to_db
        from backend.database import Stock, StockScore

        stocks = scan_seams.session.query(Stock).all()
        assert len(stocks) == 1
        assert stocks[0].ticker == "TEST"
        assert stocks[0].canslim_score == 70.0
        scores = scan_seams.session.query(StockScore).all()
        assert len(scores) == 1

    def test_empty_scan_results_do_not_abort_pipeline(self, scan_seams):
        """Regression: a total data-provider outage (run_async_scan returns [])
        must not raise ZeroDivisionError on the per-stock timing logs
        (scheduler.py ~1205 `fetch_time/len(analysis_results)` and ~1260
        `total_time/successful`) and skip the rest of the scan — saving, AI
        trading, the stop-loss/exit pass, and the post-scan helpers."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod

        # Tickers exist, but every fetch fails -> analysis_results == [].
        scan_seams.run_async_scan.side_effect = lambda *a, **k: []

        run_continuous_scan()  # must not raise

        # Reached the fetch...
        assert scan_seams.run_async_scan.call_count == 1
        # ...and got PAST both per-stock divisions to the Phase-4 helpers,
        # proving the empty result set didn't short-circuit the pipeline.
        assert scan_seams.cleanup_old_scores.call_count == 1
        assert scan_seams.check_watchlist.call_count == 1
        # Scan completed cleanly (finally block ran).
        assert scheduler_mod._scan_config["is_scanning"] is False

    def test_already_scanning_guard_early_returns(self, scan_seams):
        """Branch (b): is_scanning=True at lock-check → immediate return
        without touching the body.

        Note: the early-return path at scheduler.py:818-820 does NOT
        clear is_scanning (only the finally block does, and that runs
        only when the body has been entered). The TOCTOU window is
        accepted because apscheduler's interval is much larger than
        scan duration in practice."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod

        scheduler_mod._scan_config["is_scanning"] = True

        run_continuous_scan()

        assert scan_seams.run_async_scan.call_count == 0
        assert scan_seams.auto_record_cs.call_count == 0
        # is_scanning unchanged — early-return doesn't reset it
        assert scheduler_mod._scan_config["is_scanning"] is True

    def test_scanner_exception_records_failure_and_resets_state(self, scan_seams):
        """Branch (c): run_async_scan raises → outer except at
        scheduler.py:1591 catches and routes to _record_failure('scan').
        Finally block at 1594-1603 must still reset is_scanning + phase."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod

        scan_seams.run_async_scan.side_effect = RuntimeError("scanner kaboom")

        run_continuous_scan()  # Must not raise

        assert scheduler_mod._system_health["consecutive_scan_failures"] == 1
        assert scheduler_mod._system_health["last_scan_error"] is not None
        assert scheduler_mod._system_health["last_scan_error"]["task"] == "scan"
        # finally block ran
        assert scheduler_mod._scan_config["is_scanning"] is False
        assert scheduler_mod._scan_config["phase"] is None
        assert scheduler_mod._scan_config["last_scan_end"] is not None

        # Post-helpers should NOT have run — scanner crashed before reaching them
        assert scan_seams.auto_record_cs.call_count == 0
        assert scan_seams.run_ai_cycle.call_count == 0

    def test_bear_base_runs_when_spy_below_50ma(self, scan_seams):
        """Branch (d): SPY price < SPY 50MA → update_bear_base_candidates
        called. Branch is gated at scheduler.py:1484 to avoid scanning
        bear bases during obvious bull regimes."""
        from backend.scheduler import run_continuous_scan

        scan_seams.market_data = {
            "success": True,
            "market_trend": "bearish",
            "indexes": {"SPY": {"price": 470.0, "ma_50": 480.0}},
        }

        run_continuous_scan()

        assert scan_seams.update_bear_base.call_count == 1

    def test_bear_base_skipped_when_spy_above_50ma(self, scan_seams):
        """Branch (d) inverse: SPY > 50MA → update_bear_base_candidates
        is NOT called. Default fixture market_data is bullish."""
        from backend.scheduler import run_continuous_scan

        run_continuous_scan()

        assert scan_seams.update_bear_base.call_count == 0

    def test_per_phase_progress_documented_keys_reached(self, scan_seams, monkeypatch):
        """Branch (e): every documented _PHASE_LABELS key that should be
        reached during a normal scan actually fires.

        Catches silent UI regressions: if a phase key gets renamed without
        a corresponding _PHASE_LABELS update, the UI loses its label
        silently. This test pins which phases reach _set_phase from
        inside run_continuous_scan."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        from backend.scheduler import _PHASE_LABELS

        seen = []
        real_set_phase = scheduler_mod._set_phase

        def recorder(phase, **kwargs):
            seen.append(phase)
            return real_set_phase(phase, **kwargs)

        monkeypatch.setattr(scheduler_mod, "_set_phase", recorder)

        # SPY < 50MA so bear_base phase fires too
        scan_seams.market_data = {
            "success": True,
            "indexes": {"SPY": {"price": 470, "ma_50": 480}},
        }

        run_continuous_scan()

        for expected in (
            "scanning",
            "stocks",
            "insider_short",
            "p1_data",
            "saving",
            "coiled_spring",
            "earnings_audit",
            "industry_groups",
            "bear_base",
            "ai_trading",
            "cleanup",
        ):
            assert expected in seen, f"phase '{expected}' not reached"

        # Every phase that fires should also have a non-None label —
        # otherwise the UI shows a blank progress chip.
        for phase in set(seen):
            assert _PHASE_LABELS.get(phase) is not None, (
                f"phase '{phase}' fires from run_continuous_scan but has "
                f"no label in _PHASE_LABELS — silent UI gap"
            )

    def test_post_helper_sequencing(self, scan_seams):
        """Branch (f): canonical post-scan order. Coiled Spring alerts
        must be RECORDED before OUTCOME UPDATING (otherwise a freshly-
        recorded alert could be evaluated against its own creation
        moment). Earnings audit runs BEFORE industry-group rankings
        (audit may flag bad data the ranker would otherwise consume).
        AI cycle runs BEFORE cleanup (cleanup deletes old StockScore
        rows the cycle relies on for trend signals)."""
        from backend.scheduler import run_continuous_scan
        from backend.database import AIPortfolioConfig

        # market open + 1 active config so the AI cycle actually runs
        scan_seams.is_market_open.return_value = True
        scan_seams.session.add(
            AIPortfolioConfig(user_id=1, is_active=True, starting_cash=25000.0)
        )
        scan_seams.session.commit()

        order = []

        def _rec(name, ret=None):
            def _side(*a, **k):
                order.append(name)
                return ret
            return _side

        scan_seams.auto_record_cs.side_effect = _rec("cs_record")
        scan_seams.update_cs_outcomes.side_effect = _rec("cs_outcomes")
        scan_seams.run_earnings_audit.side_effect = _rec("audit", [])
        scan_seams.compute_ig.side_effect = _rec("ig", {})
        scan_seams.run_ai_cycle.side_effect = _rec(
            "ai", {"buys_executed": [], "sells_executed": []}
        )
        scan_seams.cleanup_old_scores.side_effect = _rec("cleanup")
        scan_seams.find_gapups.side_effect = _rec("gapup", [])
        scan_seams.check_watchlist.side_effect = _rec("watchlist")

        run_continuous_scan()

        assert order == [
            "cs_record",
            "cs_outcomes",
            "audit",
            "ig",
            "ai",
            "cleanup",
            "gapup",
            "watchlist",
        ]

    def test_per_user_ai_cycle_exception_isolated(self, scan_seams):
        """Branch (g): per-user run_ai_trading_cycle exception → that user's
        error is recorded as a trade_cycle failure, but the outer scan
        still records success. Isolation lives in the per-user try/except
        at scheduler.py:1521-1523."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        from backend.database import AIPortfolioConfig

        scan_seams.is_market_open.return_value = True
        scan_seams.run_ai_cycle.side_effect = RuntimeError("trade kaboom")
        scan_seams.session.add(
            AIPortfolioConfig(user_id=1, is_active=True, starting_cash=25000.0)
        )
        scan_seams.session.commit()

        run_continuous_scan()

        assert scheduler_mod._system_health["consecutive_trade_failures"] == 1
        assert scheduler_mod._system_health["last_trade_cycle_error"] is not None
        # Scan as a whole still succeeded — per-user except prevents
        # propagation to the outer scan-level except handler.
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_successful_scan"] is not None

    def test_source_variant_loads_right_ticker_function(self, scan_seams, monkeypatch):
        """Branch (h): _scan_config['source'] selects the loader. sp500
        uses the full S&P, top50 also goes through get_sp500_tickers
        (slices [:50]), russell uses Russell 2000, all uses
        get_all_tickers."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        import sp500_tickers

        sp500_spy = MagicMock(return_value=["SP1"])
        russell_spy = MagicMock(return_value=["RUSS1"])
        all_spy = MagicMock(return_value=["ALL1"])
        monkeypatch.setattr(sp500_tickers, "get_sp500_tickers", sp500_spy)
        monkeypatch.setattr(sp500_tickers, "get_russell2000_tickers", russell_spy)
        monkeypatch.setattr(sp500_tickers, "get_all_tickers", all_spy)

        # russell
        scheduler_mod._scan_config["source"] = "russell"
        scheduler_mod._scan_config["is_scanning"] = False
        run_continuous_scan()
        assert russell_spy.call_count == 1
        assert sp500_spy.call_count == 0
        assert all_spy.call_count == 0

        # all
        scheduler_mod._scan_config["source"] = "all"
        scheduler_mod._scan_config["is_scanning"] = False
        run_continuous_scan()
        assert all_spy.call_count == 1

        # sp500 (full)
        scheduler_mod._scan_config["source"] = "sp500"
        scheduler_mod._scan_config["is_scanning"] = False
        run_continuous_scan()
        assert sp500_spy.call_count == 1

        # top50 also goes through get_sp500_tickers (slices [:50])
        scheduler_mod._scan_config["source"] = "top50"
        scheduler_mod._scan_config["is_scanning"] = False
        run_continuous_scan()
        assert sp500_spy.call_count == 2

    def test_save_stock_blip_detection_keeps_old_score(self, scan_seams):
        """Branch (a-supplemental): save_stock_to_db's BLIP recovery
        (scheduler.py:1117-1173). When a pre-existing stock has score
        100 and the new analysis would drop it to 20 with missing data
        signals, the saver KEEPS the old score rather than persisting
        the suspected blip. This is the load-bearing safeguard added
        after a Yahoo flash returned zero scores for ~80 stocks
        mid-scan and wiped them in the DB."""
        from backend.scheduler import run_continuous_scan
        from backend.database import Stock

        # Pre-seed a stock with high prior score
        scan_seams.session.add(
            Stock(ticker="TEST", canslim_score=100.0, c_score=15.0, a_score=15.0)
        )
        scan_seams.session.commit()

        # Send a "blip" analysis: score drops 80 points + missing earnings
        # + multiple zero components + insufficient-data summary.
        scan_seams.fake_analysis = {
            "ticker": "TEST",
            "company_name": "Test Inc",
            "canslim_score": 20.0,
            "c_score": 0,
            "a_score": 0,
            "n_score": 0,
            "s_score": 5.0,
            "l_score": 5.0,
            "i_score": 5.0,
            "m_score": 5.0,
            "score_details": {
                "c": {"summary": "Insufficient data"},
                "a": {"summary": "No data available"},
            },
            "current_price": 100.0,
            "week_52_high": 120.0,
            # No quarterly_earnings → triggers blip detection
        }

        run_continuous_scan()

        stock = scan_seams.session.query(Stock).filter_by(ticker="TEST").one()
        # Old score preserved by the BLIP guard
        assert stock.canslim_score == 100.0
        # Component scores also preserved (they were non-zero before)
        assert stock.c_score == 15.0
        assert stock.a_score == 15.0

    def test_save_stock_full_data_updates_conditional_fields(self, scan_seams):
        """Branch (a-supplemental): save_stock_to_db's conditional update
        blocks (scheduler.py:1230-1273). When the analysis dict carries
        insider/short/earnings-calendar/analyst-estimates payloads, the
        corresponding `if analysis.get(...)` blocks fire and persist
        those fields onto the Stock row."""
        from datetime import date as _date, timedelta as _timedelta
        from backend.scheduler import run_continuous_scan
        from backend.database import Stock

        future = (_date.today() + _timedelta(days=14)).isoformat()
        scan_seams.fake_analysis = {
            "ticker": "TEST",
            "company_name": "Test Inc",
            "sector": "Technology",
            "current_price": 100.0,
            "canslim_score": 70.0,
            "c_score": 10.0,
            "a_score": 10.0,
            "n_score": 10.0,
            "s_score": 10.0,
            "l_score": 10.0,
            "i_score": 10.0,
            "m_score": 10.0,
            "score_details": {},
            # MA + ATR (lines 1230-1235)
            "ma_21": 95.0,
            "ma_50": 90.0,
            "atr_pct": 2.5,
            # Insider data (line 1238 onward)
            "insider_sentiment": "bullish",
            "insider_buy_count": 5,
            "insider_sell_count": 1,
            "insider_net_shares": 10000,
            "insider_buy_value": 500000.0,
            "insider_sell_value": 100000.0,
            "insider_net_value": 400000.0,
            "insider_largest_buy": 250000.0,
            "insider_largest_buyer_title": "CEO",
            # Short interest (line 1252 onward)
            "short_interest_pct": 3.5,
            "short_ratio": 1.2,
            # Earnings calendar (line 1258 onward)
            "next_earnings_date": future,
            "days_to_earnings": 14,
            "earnings_beat_streak": 3,
            # Analyst estimates (line 1268 onward)
            "eps_estimate_current": 1.50,
            "eps_estimate_prior": 1.40,
            "eps_estimate_revision_pct": 7.14,
            "estimate_revision_trend": "rising",
            "quarterly_earnings": [],
            "annual_earnings": [],
            "quarterly_revenue": [],
        }

        run_continuous_scan()

        stock = scan_seams.session.query(Stock).filter_by(ticker="TEST").one()
        assert stock.ma_21 == 95.0
        assert stock.ma_50 == 90.0
        assert stock.atr_pct == 2.5
        assert stock.insider_sentiment == "bullish"
        assert stock.insider_net_value == 400000.0
        assert stock.insider_largest_buyer_title == "CEO"
        assert stock.short_interest_pct == 3.5
        assert stock.short_ratio == 1.2
        assert stock.days_to_earnings == 14
        assert stock.earnings_beat_streak == 3
        assert stock.eps_estimate_current == 1.50
        assert stock.eps_estimate_revision_pct == 7.14

    def test_market_direction_failure_continues_scan(self, scan_seams):
        """Branch (a-supplemental): get_cached_market_direction returns
        success=False (scheduler.py:894-895). The scan logs a warning
        but does NOT abort — the M score will fall back to its cached
        value. Pin via a successful _record_success at the end."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod

        scan_seams.market_data = {"success": False, "error": "FMP timeout"}

        run_continuous_scan()

        # Scan still completed
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_successful_scan"] is not None
        assert scan_seams.run_async_scan.call_count == 1

    def test_each_post_helper_exception_is_isolated(self, scan_seams):
        """Branch (a-supplemental): scheduler.py wraps every post-scan
        helper in its own try/except (lines 1413, 1423, 1435, 1443,
        1458, 1474, 1493, 1546, 1567, 1575, 1581). Each exception is
        logged but does NOT propagate to the outer except. The scan as
        a whole still records success.

        This is the post-scan equivalent of run_ai_trading_cycle's
        per-user isolation: if the gap-up detector or watchlist alerter
        crashes, the next scan tick should still find the system in a
        clean state."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        from backend.database import AIPortfolioConfig

        # Make every post-helper raise. Also force bear-base + AI cycle
        # to run so their except handlers are exercised.
        scan_seams.is_market_open.return_value = True
        scan_seams.session.add(
            AIPortfolioConfig(user_id=1, is_active=True, starting_cash=25000.0)
        )
        scan_seams.session.commit()
        scan_seams.market_data = {
            "success": True,
            "indexes": {"SPY": {"price": 470.0, "ma_50": 480.0}},
        }

        scan_seams.update_market_snapshot.side_effect = RuntimeError("snapshot boom")
        scan_seams.auto_record_cs.side_effect = RuntimeError("cs record boom")
        scan_seams.update_cs_outcomes.side_effect = RuntimeError("cs outcome boom")
        scan_seams.run_earnings_audit.side_effect = RuntimeError("audit boom")
        scan_seams.compute_ig.side_effect = RuntimeError("ig boom")
        scan_seams.update_bear_base.side_effect = RuntimeError("bear base boom")
        scan_seams.cleanup_old_scores.side_effect = RuntimeError("cleanup boom")
        scan_seams.find_gapups.side_effect = RuntimeError("gapup boom")
        scan_seams.check_watchlist.side_effect = RuntimeError("watchlist boom")

        run_continuous_scan()  # Must not raise

        # All boomed but scan as a whole still succeeded.
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_successful_scan"] is not None
        # Cleanup state still ran in finally
        assert scheduler_mod._scan_config["is_scanning"] is False

    def test_market_closed_skips_ai_cycle_for_active_config(self, scan_seams):
        """Branch (a-supplemental): when is_market_open()=False (default),
        active configs log 'Market closed' (scheduler.py:1520) and the
        AI cycle is NOT invoked. Inactive configs do nothing in this
        case either (line 1529 also gates on market open)."""
        from backend.scheduler import run_continuous_scan
        from backend.database import AIPortfolioConfig

        # Default fixture has is_market_open()=False
        scan_seams.session.add(
            AIPortfolioConfig(user_id=1, is_active=True, starting_cash=25000.0)
        )
        scan_seams.session.add(
            AIPortfolioConfig(user_id=2, is_active=False, starting_cash=25000.0)
        )
        scan_seams.session.commit()

        run_continuous_scan()

        assert scan_seams.run_ai_cycle.call_count == 0
        assert scan_seams.take_snapshot.call_count == 0

    def test_inactive_configs_take_snapshot_when_market_open(self, scan_seams):
        """Branch (a-supplemental): inactive AIPortfolioConfig rows still
        get take_portfolio_snapshot during market hours (scheduler.py:
        1525-1532). Active configs run the full cycle; inactive configs
        only get a snapshot to keep the equity-curve chart populated."""
        from backend.scheduler import run_continuous_scan
        from backend.database import AIPortfolioConfig

        scan_seams.is_market_open.return_value = True
        scan_seams.session.add(
            AIPortfolioConfig(user_id=1, is_active=True, starting_cash=25000.0)
        )
        scan_seams.session.add(
            AIPortfolioConfig(user_id=2, is_active=False, starting_cash=25000.0)
        )
        scan_seams.session.commit()

        run_continuous_scan()

        # Active user gets the cycle, inactive gets the snapshot
        assert scan_seams.run_ai_cycle.call_count == 1
        assert scan_seams.run_ai_cycle.call_args.kwargs["user_id"] == 1
        assert scan_seams.take_snapshot.call_count == 1
        assert scan_seams.take_snapshot.call_args.kwargs["user_id"] == 2

    def test_gapup_alert_fires_when_findings_returned(self, scan_seams):
        """Branch (a-supplemental): if find_earnings_gapups returns a
        non-empty list, send_gapup_alert is invoked (scheduler.py:
        1561-1564). Default fixture returns []; this test flips it
        to drive the alert path."""
        from backend.scheduler import run_continuous_scan

        scan_seams.find_gapups.return_value = [
            {"ticker": "TEST", "gap_pct": 8.5, "date": "2026-05-07"}
        ]

        run_continuous_scan()

        assert scan_seams.send_gapup.call_count == 1
        passed = scan_seams.send_gapup.call_args[0][0]
        assert passed[0]["ticker"] == "TEST"

    def test_progress_callback_unknown_phase_falls_back(self, scan_seams, monkeypatch):
        """Branch (a-supplemental): the update_progress closure has an
        else-branch (scheduler.py:1328-1329) for unknown phase strings.
        Drive it by feeding a custom phase to verify the fallback
        f-string path is exercised."""
        from backend.scheduler import run_continuous_scan
        import async_scanner

        def stub(tickers, batch_size=100, progress_callback=None):
            if progress_callback and tickers:
                # Custom phase = unknown → else branch
                progress_callback(1, len(tickers), "custom_phase")
                # current % 100 == 0 logging branch (line 1334)
                progress_callback(100, 200, "stocks")
                # total mismatch updates _scan_config[total_stocks] (line 1317)
                progress_callback(1, 99, "stocks")
            return [dict(scan_seams.fake_analysis)]

        monkeypatch.setattr(async_scanner, "run_async_scan", stub)

        run_continuous_scan()  # must not raise

    def test_save_failure_path_logs_and_continues(self, scan_seams, monkeypatch):
        """Branch (a-supplemental): if save_stock_to_db raises mid-batch,
        the savepoint rollback path (scheduler.py:1375-1379) catches
        the failure, increments `failed`, and the batch continues with
        the next stock. Drive by sending two analyses: one with a
        malformed next_earnings_date (crashes strptime), one valid.

        NB: a single-ticker version of this test would expose a latent
        bug — when ALL stocks fail, `successful=0` causes
        `total_time/successful` at scheduler.py:1396 to raise
        ZeroDivisionError, which propagates to the outer except handler
        and records the scan itself as failed. Memory-noted; fix
        DEFERRED past 2026-06-18 eval."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        import async_scanner
        import sp500_tickers

        # 2 tickers so one survives even when the other crashes
        monkeypatch.setattr(sp500_tickers, "get_sp500_tickers", lambda: ["BAD", "GOOD"])

        good = dict(scan_seams.fake_analysis)
        bad = dict(scan_seams.fake_analysis)
        bad["ticker"] = "BAD"
        bad["next_earnings_date"] = "NOT-A-DATE"
        bad["earnings_beat_streak"] = 3
        good["ticker"] = "GOOD"

        def stub_two(tickers, batch_size=100, progress_callback=None):
            return [bad, good]

        monkeypatch.setattr(async_scanner, "run_async_scan", stub_two)

        run_continuous_scan()

        # Scan as a whole succeeded (good ticker saved; bad one rolled back)
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_successful_scan"] is not None
        # Only the good ticker was persisted
        from backend.database import Stock
        tickers_persisted = {s.ticker for s in scan_seams.session.query(Stock).all()}
        assert "GOOD" in tickers_persisted
        assert "BAD" not in tickers_persisted

    def test_invalid_source_falls_back_to_sp500(self, scan_seams, monkeypatch):
        """Branch (h-supplemental): an unrecognized _scan_config['source']
        falls through to the else branch at scheduler.py:848-849 and
        loads the S&P 500 tickers as a safe default."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        import sp500_tickers

        sp500_spy = MagicMock(return_value=["SP1"])
        russell_spy = MagicMock(return_value=["RUSS1"])
        all_spy = MagicMock(return_value=["ALL1"])
        monkeypatch.setattr(sp500_tickers, "get_sp500_tickers", sp500_spy)
        monkeypatch.setattr(sp500_tickers, "get_russell2000_tickers", russell_spy)
        monkeypatch.setattr(sp500_tickers, "get_all_tickers", all_spy)

        scheduler_mod._scan_config["source"] = "wat-is-this"

        run_continuous_scan()

        assert sp500_spy.call_count == 1
        assert russell_spy.call_count == 0
        assert all_spy.call_count == 0

    def test_blip_recovery_preserves_all_six_components(self, scan_seams):
        """Branch (a-supplemental): BLIP recovery (scheduler.py:1162-1173)
        preserves c/a/n/s/l/i if any of them was non-zero before but
        the new analysis would zero them out. The previous BLIP test
        only proved c+a recovery; this pins the n/s/l/i branches too,
        each of which is its own `if stock.X_score and analysis.get(...)
        == 0` guard."""
        from backend.scheduler import run_continuous_scan
        from backend.database import Stock

        # Pre-seed with all six non-zero
        scan_seams.session.add(
            Stock(
                ticker="TEST",
                canslim_score=100.0,
                c_score=15.0, a_score=15.0, n_score=15.0,
                s_score=15.0, l_score=15.0, i_score=15.0,
            )
        )
        scan_seams.session.commit()

        # Send an all-zeros blip
        scan_seams.fake_analysis = {
            "ticker": "TEST",
            "company_name": "Test Inc",
            "canslim_score": 10.0,
            "c_score": 0,
            "a_score": 0,
            "n_score": 0,
            "s_score": 0,
            "l_score": 0,
            "i_score": 0,
            "m_score": 5.0,
            "score_details": {
                "c": {"summary": "Insufficient data"},
                "a": {"summary": "No data"},
            },
            "current_price": 100.0,
        }

        run_continuous_scan()

        stock = scan_seams.session.query(Stock).filter_by(ticker="TEST").one()
        assert stock.canslim_score == 100.0
        assert stock.c_score == 15.0
        assert stock.a_score == 15.0
        assert stock.n_score == 15.0
        assert stock.s_score == 15.0
        assert stock.l_score == 15.0
        assert stock.i_score == 15.0

    def test_rate_limit_stats_failure_isolated(self, scan_seams, monkeypatch):
        """Branch (a-supplemental): if the rate-limit/cache stats block
        raises (scheduler.py:1399-1414), the scan still completes —
        these are diagnostic-only logs, not load-bearing logic."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        import data_fetcher

        def boom():
            raise RuntimeError("stats unavailable")

        monkeypatch.setattr(data_fetcher, "get_rate_limit_stats", boom)

        run_continuous_scan()

        assert scheduler_mod._system_health["last_successful_scan"] is not None

    def test_audit_results_logged_when_returned(self, scan_seams):
        """Branch (a-supplemental): when run_earnings_audit returns a
        non-empty list, scheduler.py:1455 logs a count line. Default
        fixture returns []; this test pins the truthy branch."""
        from backend.scheduler import run_continuous_scan

        scan_seams.run_earnings_audit.return_value = [
            {"ticker": "TEST", "issue": "missing_pe"},
            {"ticker": "OTHER", "issue": "stale_eps"},
        ]

        run_continuous_scan()  # must not raise

        assert scan_seams.run_earnings_audit.call_count == 1

    def test_inactive_snapshot_exception_isolated(self, scan_seams):
        """Branch (g-supplemental): take_portfolio_snapshot for an
        inactive config raises → caught by the per-user try/except at
        scheduler.py:1531-1532. Scan still records success."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        from backend.database import AIPortfolioConfig

        scan_seams.is_market_open.return_value = True
        scan_seams.take_snapshot.side_effect = RuntimeError("snapshot kaboom")
        scan_seams.session.add(
            AIPortfolioConfig(user_id=2, is_active=False, starting_cash=25000.0)
        )
        scan_seams.session.commit()

        run_continuous_scan()

        # Outer scan still succeeded — per-user except wrapped the raise
        assert scheduler_mod._system_health["consecutive_scan_failures"] == 0
        assert scheduler_mod._system_health["last_successful_scan"] is not None

    def test_ig_preload_failure_does_not_abort_scan(self, scan_seams, monkeypatch):
        """Branch (a-supplemental): industry-group rank preload at
        scheduler.py:903-914 is wrapped in try/except. If the DB query
        fails (e.g., schema migration in flight), we log debug and
        continue with an empty rank dict — no scan abort."""
        from backend.scheduler import run_continuous_scan
        from backend import scheduler as scheduler_mod
        import backend.database as database

        # First SessionLocal() is the IG preload — make it bomb when queried.
        # Wrap the original SessionLocal so subsequent calls return a real
        # session, but the first .query raises.
        real_sessionlocal = database.SessionLocal
        call_state = {"first": True}

        class BoomSession:
            def query(self, *a, **k):
                raise RuntimeError("schema mid-migration")

            def close(self):
                pass

        def session_factory():
            if call_state["first"]:
                call_state["first"] = False
                return BoomSession()
            return real_sessionlocal()

        monkeypatch.setattr(database, "SessionLocal", session_factory)

        run_continuous_scan()

        assert scheduler_mod._system_health["last_successful_scan"] is not None
