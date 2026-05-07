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

  Tier 3 (intentionally NOT covered in this push):
    - run_continuous_scan (lines 811-1603, ~790 lines): heavy async + FMP rate
      limiter + redis_cache + sleep/retry coupling. Deserves its own session.
    - send_morning_briefing_if_due (lines 629-808): morning email logic;
      gated on time-of-day and SystemState dedup; tested via integration
      elsewhere.
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

    DB-bloat prevention. Deletes StockScore rows older than N days in
    batches; VACUUMs SQLite afterward. The function takes an explicit
    Session arg (unlike the SessionLocal-opening helpers above).
    """

    def test_no_old_scores_returns_zero(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="FRESH", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # All scores within window.
        for i in range(3):
            db_session.add(
                StockScore(
                    stock_id=stock.id,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=i),
                    date=date.today() - timedelta(days=i),
                    total_score=70.0,
                )
            )
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, days_to_keep=30)
        assert deleted == 0

    def test_deletes_old_scores(self, db_session):
        from backend.database import Stock, StockScore
        from backend.scheduler import cleanup_old_stock_scores

        stock = Stock(ticker="OLD", current_price=100.0)
        db_session.add(stock)
        db_session.commit()

        # 5 old (60d) + 2 fresh (1d).
        for i in range(5):
            db_session.add(
                StockScore(
                    stock_id=stock.id,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=60 + i),
                    date=date.today() - timedelta(days=60 + i),
                    total_score=70.0,
                )
            )
        for i in range(2):
            db_session.add(
                StockScore(
                    stock_id=stock.id,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=i + 1),
                    date=date.today() - timedelta(days=i + 1),
                    total_score=72.0,
                )
            )
        db_session.commit()

        deleted = cleanup_old_stock_scores(db_session, days_to_keep=30)
        assert deleted == 5
        remaining = db_session.query(StockScore).count()
        assert remaining == 2


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
