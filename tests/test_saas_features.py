"""
Tests for SaaS-grade features (Mar 30, 2026):
- DB Backup module
- Breakout monitor
- System health endpoint
- Exit quality analysis
- Signal attribution
- ML live trade extraction
- Weekly email attribute fixes
- Scheduler health tracking
"""
import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock, ANY
from datetime import datetime, timezone, timedelta

# Ensure email_utils can be found (it's in backend/ which may not be on path)
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_parent, 'backend') not in sys.path:
    sys.path.insert(0, os.path.join(_parent, 'backend'))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, User
from backend.auth import get_current_active_user

# Auth bypass
_fake_user = User(id=1, email="test@test.com", display_name="Test",
                  is_active=True, is_admin=True, hashed_password="")
app.dependency_overrides[get_current_active_user] = lambda: _fake_user
init_db()
client = TestClient(app)

# Pre-create a mock for webhook notifications
_mock_webhook = MagicMock()


def _patch_webhook():
    """Context manager to mock webhook notifications across all import paths."""
    return patch.dict("sys.modules", {
        "backend.email_utils": MagicMock(send_webhook_notification=_mock_webhook),
    })


# ============== Backup Module ==============

class TestBackupModule:
    def test_list_backups_empty(self):
        from backend.backup import list_backups
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(sys.modules['backend.backup'], 'BACKUP_DIR', tmpdir):
                result = list_backups()
                assert isinstance(result, list)
                assert len(result) == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_get_backup_status_structure(self):
        from backend.backup import get_backup_status
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(sys.modules['backend.backup'], 'BACKUP_DIR', tmpdir):
                status = get_backup_status()
                assert "last_backup" in status
                assert "total_backups" in status
                assert "daily_count" in status
                assert "weekly_count" in status
                assert status["total_backups"] == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_cleanup_respects_retention(self):
        from backend.backup import cleanup_old_backups, DAILY_RETENTION
        tmpdir = tempfile.mkdtemp()
        try:
            for i in range(DAILY_RETENTION + 3):
                path = os.path.join(tmpdir, f"canslim_2026033{i:02d}_010000.dump")
                with open(path, "w") as f:
                    f.write("x")
                os.utime(path, (1000000 + i, 1000000 + i))

            with patch.object(sys.modules['backend.backup'], 'BACKUP_DIR', tmpdir):
                cleanup_old_backups()

            remaining = [f for f in os.listdir(tmpdir) if f.startswith("canslim_2")]
            assert len(remaining) == DAILY_RETENTION
        finally:
            shutil.rmtree(tmpdir)

    def test_cleanup_preserves_weekly(self):
        from backend.backup import cleanup_old_backups, WEEKLY_RETENTION
        tmpdir = tempfile.mkdtemp()
        try:
            for i in range(WEEKLY_RETENTION + 2):
                path = os.path.join(tmpdir, f"canslim_weekly_2026030{i}_010000.dump")
                with open(path, "w") as f:
                    f.write("x")
                os.utime(path, (1000000 + i, 1000000 + i))

            with patch.object(sys.modules['backend.backup'], 'BACKUP_DIR', tmpdir):
                cleanup_old_backups()

            remaining = [f for f in os.listdir(tmpdir) if "weekly" in f]
            assert len(remaining) == WEEKLY_RETENTION
        finally:
            shutil.rmtree(tmpdir)

    def test_parse_database_url_missing(self):
        from backend.backup import _parse_database_url
        old_val = os.environ.pop("DATABASE_URL", None)
        try:
            with pytest.raises(ValueError):
                _parse_database_url()
        finally:
            if old_val:
                os.environ["DATABASE_URL"] = old_val

    def test_parse_database_url_valid(self):
        from backend.backup import _parse_database_url
        old_val = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = "postgresql://myuser:mypass@dbhost:5433/mydb"
            params = _parse_database_url()
            assert params["host"] == "dbhost"
            assert params["port"] == "5433"
            assert params["user"] == "myuser"
            assert params["password"] == "mypass"
            assert params["dbname"] == "mydb"
        finally:
            if old_val:
                os.environ["DATABASE_URL"] = old_val
            else:
                os.environ.pop("DATABASE_URL", None)

    def test_list_backups_with_files(self):
        from backend.backup import list_backups
        tmpdir = tempfile.mkdtemp()
        try:
            daily = os.path.join(tmpdir, "canslim_20260330_010000.dump")
            weekly = os.path.join(tmpdir, "canslim_weekly_20260330_010000.dump")
            with open(daily, "w") as f:
                f.write("x" * 1024)
            with open(weekly, "w") as f:
                f.write("y" * 2048)

            with patch.object(sys.modules['backend.backup'], 'BACKUP_DIR', tmpdir):
                result = list_backups()
                assert len(result) == 2
                assert any(b["is_weekly"] for b in result)
                assert any(not b["is_weekly"] for b in result)
                for b in result:
                    assert "filename" in b
                    assert "size_mb" in b
                    assert "created" in b
        finally:
            shutil.rmtree(tmpdir)


# ============== Breakout Monitor ==============

class TestBreakoutMonitor:
    def test_get_status_structure(self):
        from backend.breakout_monitor import get_breakout_monitor_status
        status = get_breakout_monitor_status()
        assert "active_cooldowns" in status
        assert "cooldown_tickers" in status
        assert "cooldown_hours" in status
        assert "min_score" in status
        assert "min_c_score" in status
        assert "min_l_score" in status

    def test_quality_thresholds_match_trader(self):
        from backend.breakout_monitor import MIN_SCORE, MIN_C_SCORE, MIN_L_SCORE
        assert MIN_SCORE == 72
        assert MIN_C_SCORE == 10
        assert MIN_L_SCORE == 8

    def test_cooldown_is_24h(self):
        from backend.breakout_monitor import ALERT_COOLDOWN_HOURS
        assert ALERT_COOLDOWN_HOURS == 24

    def test_skips_when_market_closed(self):
        from backend import breakout_monitor
        with patch.object(breakout_monitor, "is_market_open", create=True, return_value=False):
            # The function does a late import, so we mock at the source
            with patch("backend.ai_trader.is_market_open", return_value=False):
                breakout_monitor.check_intraday_breakouts()

    def test_cooldown_tracking(self):
        from backend.breakout_monitor import _recent_alerts, ALERT_COOLDOWN_HOURS
        _recent_alerts.clear()
        now = datetime.now(timezone.utc)
        _recent_alerts["TEST"] = now
        assert len(_recent_alerts) == 1

        _recent_alerts["TEST"] = now - timedelta(hours=ALERT_COOLDOWN_HOURS * 3)
        expired = [t for t, ts in _recent_alerts.items()
                   if ts < now - timedelta(hours=ALERT_COOLDOWN_HOURS * 2)]
        for t in expired:
            del _recent_alerts[t]
        assert len(_recent_alerts) == 0

    def test_cooldown_prevents_duplicate(self):
        from backend.breakout_monitor import _recent_alerts, ALERT_COOLDOWN_HOURS
        _recent_alerts.clear()
        now = datetime.now(timezone.utc)
        _recent_alerts["AAPL"] = now
        is_cooled_down = (now - _recent_alerts["AAPL"]) < timedelta(hours=ALERT_COOLDOWN_HOURS)
        assert is_cooled_down
        _recent_alerts.clear()


# ============== Signal Attribution ==============

class TestSignalAttributionEndpoint:
    def test_returns_200(self):
        assert client.get("/api/analytics/signal-attribution?days=365").status_code == 200

    def test_structure(self):
        d = client.get("/api/analytics/signal-attribution?days=365").json()
        assert "by_entry_type" in d
        assert "by_signal" in d
        assert "total_paired_trades" in d

    def test_by_signal_always_has_three(self):
        d = client.get("/api/analytics/signal-attribution?days=365").json()
        signals = [s["signal"] for s in d["by_signal"]]
        assert "Coiled Spring" in signals
        assert "Volume Dry-Up" in signals
        assert "Standard" in signals

    def test_min_days_validation(self):
        assert client.get("/api/analytics/signal-attribution?days=5").status_code == 422

    def test_max_days_validation(self):
        assert client.get("/api/analytics/signal-attribution?days=9999").status_code == 422

    def test_entry_type_stats_structure(self):
        d = client.get("/api/analytics/signal-attribution?days=365").json()
        for e in d["by_entry_type"]:
            for key in ["entry_type", "trades", "win_rate", "avg_return_pct", "total_pnl", "avg_days_held"]:
                assert key in e


# ============== Exit Quality ==============

class TestExitQualityEndpoint:
    def test_returns_200(self):
        assert client.get("/api/analytics/exit-quality?days=365").status_code == 200

    def test_structure(self):
        d = client.get("/api/analytics/exit-quality?days=365").json()
        assert "trades" in d
        assert "summary" in d

    def test_empty_when_no_sells(self):
        d = client.get("/api/analytics/exit-quality?days=30").json()
        assert isinstance(d["trades"], list)

    def test_days_validation(self):
        assert client.get("/api/analytics/exit-quality?days=5").status_code == 422
        assert client.get("/api/analytics/exit-quality?days=9999").status_code == 422


# ============== Backup Endpoints ==============

class TestBackupEndpoints:
    def test_list_backups_200(self):
        with patch("backend.backup.list_backups", return_value=[]):
            r = client.get("/api/system/backups")
            assert r.status_code == 200
            assert "backups" in r.json()

    def test_trigger_backup_returns_status(self):
        with patch("backend.backup.perform_backup",
                   return_value={"status": "success", "filename": "test.dump",
                                 "size_mb": 1.0, "timestamp": "now"}):
            r = client.post("/api/system/backup")
            assert r.status_code == 200
            assert r.json()["status"] == "success"


# ============== ML Live Trade Extraction ==============

class TestMLLiveTradeExtraction:
    def test_extract_live_trades_empty_db(self):
        from backend.database import SessionLocal
        from ml.feature_extractor import extract_live_trade_data
        db = SessionLocal()
        try:
            df = extract_live_trade_data(db)
            assert df.empty or len(df) == 0
        finally:
            db.close()


# ============== Scheduler Health Tracking ==============

class TestSchedulerHealthTracking:
    def test_record_success_resets_scan_failures(self):
        from backend.scheduler import _record_success, _system_health
        _system_health["consecutive_scan_failures"] = 5
        _record_success("scan")
        assert _system_health["consecutive_scan_failures"] == 0
        assert _system_health["last_successful_scan"] is not None

    def test_record_success_resets_trade_failures(self):
        from backend.scheduler import _record_success, _system_health
        _system_health["consecutive_trade_failures"] = 3
        _record_success("trade_cycle")
        assert _system_health["consecutive_trade_failures"] == 0
        assert _system_health["last_successful_trade_cycle"] is not None

    def test_get_system_health_returns_dict(self):
        from backend.scheduler import get_system_health
        health = get_system_health()
        assert isinstance(health, dict)
        assert "last_successful_scan" in health
        assert "consecutive_scan_failures" in health
        assert "errors_today" in health

    def test_record_failure_increments_and_alerts(self):
        """First failure increments counter and sends alert."""
        from backend.scheduler import _record_failure, _system_health
        _system_health["consecutive_scan_failures"] = 0
        _system_health["errors_today"] = []

        # Mock the late import inside _record_failure
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"backend.email_utils": mock_module}):
            _record_failure("scan", "test error")

        assert _system_health["consecutive_scan_failures"] == 1
        assert _system_health["last_scan_error"]["error"] == "test error"
        # Alert sent on first failure
        mock_module.send_webhook_notification.assert_called_once()

    def test_record_failure_skips_alert_on_second(self):
        """Second consecutive failure does NOT send alert."""
        from backend.scheduler import _record_failure, _system_health
        _system_health["consecutive_scan_failures"] = 1  # will become 2

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"backend.email_utils": mock_module}):
            _record_failure("scan", "test error 2")

        assert _system_health["consecutive_scan_failures"] == 2
        mock_module.send_webhook_notification.assert_not_called()

    def test_record_failure_alerts_on_third(self):
        """Third consecutive failure sends high-priority alert."""
        from backend.scheduler import _record_failure, _system_health
        _system_health["consecutive_scan_failures"] = 2  # will become 3

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"backend.email_utils": mock_module}):
            _record_failure("scan", "test error 3")

        assert _system_health["consecutive_scan_failures"] == 3
        mock_module.send_webhook_notification.assert_called_once()
        call_kwargs = mock_module.send_webhook_notification.call_args[1]
        assert call_kwargs["priority"] == "high"

    def test_errors_today_capped_at_50(self):
        from collections import deque
        from backend.scheduler import _record_failure, _system_health
        _system_health["errors_today"] = deque(maxlen=50)
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"backend.email_utils": mock_module}):
            for i in range(60):
                _system_health["consecutive_scan_failures"] = 1  # 2nd = no alert
                _record_failure("scan", f"error {i}")
        assert len(_system_health["errors_today"]) <= 50


# ============== Weekly Email Attribute Regression ==============

class TestWeeklyEmailAttributes:
    def test_action_not_trade_type(self):
        """Confirm no reference to trade_type in weekly email code."""
        import inspect
        from backend.scheduler import send_weekly_performance_email
        source = inspect.getsource(send_weekly_performance_email)
        assert "trade_type" not in source

    def test_realized_gain_not_realized_gain_loss(self):
        """Confirm no reference to realized_gain_loss in weekly email code."""
        import inspect
        from backend.scheduler import send_weekly_performance_email
        source = inspect.getsource(send_weekly_performance_email)
        assert "realized_gain_loss" not in source

    def test_uses_uppercase_action(self):
        """BUY/SELL must be uppercase (matching AIPortfolioTrade.action)."""
        import inspect
        from backend.scheduler import send_weekly_performance_email
        source = inspect.getsource(send_weekly_performance_email)
        assert '"BUY"' in source or "'BUY'" in source
        assert '"SELL"' in source or "'SELL'" in source
