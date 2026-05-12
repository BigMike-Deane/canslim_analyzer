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
from tests.conftest import override_dependency

# Auth bypass
_fake_user = User(id=1, email="test@test.com", display_name="Test",
                  is_active=True, is_admin=True, hashed_password="")


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    """Scoped auth bypass — see tests/conftest.py:override_dependency."""
    with override_dependency(get_current_active_user, _fake_user):
        yield


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

    # --- perform_backup() ------------------------------------------------
    # Covers backend/backup.py lines 38-120 — the entire pre-existing gap.
    # The function's externals are mocked at four boundaries: subprocess.run
    # (pg_dump), datetime.now (controls the Sunday→weekly rename branch),
    # send_webhook_notification (success + failure tags), and cleanup_old_backups
    # (avoids touching real filesystem retention). BACKUP_DIR is redirected
    # to a tempdir so the touch-then-getsize flow exercises the real
    # os.makedirs / os.path.getsize / os.rename code paths.

    @staticmethod
    def _fake_datetime_for(weekday_anchor):
        """Build a `datetime` class stand-in whose .now() always returns the
        anchor. weekday_anchor must already be tz-aware so isoformat() and
        strftime("%A") both behave like a real datetime.
        """

        class _FakeDateTime:
            @classmethod
            def now(cls, tz=None):
                return weekday_anchor

        return _FakeDateTime

    def _run_perform_backup(self, tmpdir, pg_dump_result, weekday_anchor):
        """Common scaffold for perform_backup() tests.

        - tmpdir is BACKUP_DIR (real os.makedirs runs against it)
        - pg_dump_result is the MagicMock returned by subprocess.run
          (must expose .returncode and .stderr)
        - weekday_anchor controls datetime.now() output

        Returns (result_dict, mock_webhook, mock_cleanup, mock_subprocess).
        """
        from backend import backup as backup_mod

        # subprocess.run is mocked to "create" the dump file if it would
        # succeed, so os.path.getsize has something real to measure.
        original_run = pg_dump_result

        def _fake_run(cmd, env=None, capture_output=None, text=None, timeout=None):
            if isinstance(original_run, Exception):
                raise original_run
            if original_run.returncode == 0:
                # Find the "-f filepath" arg and create the file
                if "-f" in cmd:
                    filepath = cmd[cmd.index("-f") + 1]
                    with open(filepath, "wb") as f:
                        f.write(b"FAKE_PG_DUMP_CONTENT")
            return original_run

        with patch.object(backup_mod, "BACKUP_DIR", tmpdir), \
             patch.object(backup_mod, "datetime",
                          self._fake_datetime_for(weekday_anchor)), \
             patch.object(backup_mod, "subprocess") as mock_sub, \
             patch.object(backup_mod, "cleanup_old_backups") as mock_cleanup, \
             patch("backend.email_utils.send_webhook_notification") as mock_webhook, \
             patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:p@h:5432/d"}):
            mock_sub.run.side_effect = _fake_run
            from backend.backup import perform_backup
            result = perform_backup()
        return result, mock_webhook, mock_cleanup, mock_sub

    def test_perform_backup_success_returns_status_dict(self):
        """Happy path on a weekday: success dict, cleanup called, low-priority webhook."""
        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        assert friday.strftime("%A") == "Friday"  # pin the anchor

        pg_ok = MagicMock(returncode=0, stderr="")
        tmpdir = tempfile.mkdtemp()
        try:
            result, mock_webhook, mock_cleanup, _ = self._run_perform_backup(
                tmpdir, pg_ok, friday,
            )
            assert result["status"] == "success"
            assert result["filename"].startswith("canslim_")
            assert result["filename"].endswith(".dump")
            assert "weekly" not in result["filename"]
            # size_mb is round(bytes / 1MB, 1); the fake payload is tiny
            # so it rounds to 0.0 — assert it's a real numeric reading, not
            # that the file was large.
            assert isinstance(result["size_mb"], (int, float))
            assert result["size_mb"] >= 0
            mock_cleanup.assert_called_once()
            # Low-priority on success
            mock_webhook.assert_called_once()
            assert mock_webhook.call_args.kwargs["priority"] == "low"
            assert mock_webhook.call_args.kwargs["title"] == "DB Backup Complete"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_renames_to_weekly_on_sunday(self):
        """Sunday runs are tagged weekly_* so cleanup keeps them on a separate retention."""
        sunday = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
        assert sunday.strftime("%A") == "Sunday"  # pin the anchor

        pg_ok = MagicMock(returncode=0, stderr="")
        tmpdir = tempfile.mkdtemp()
        try:
            result, _, _, _ = self._run_perform_backup(tmpdir, pg_ok, sunday)
            assert result["status"] == "success"
            assert result["filename"].startswith("canslim_weekly_")
            # The renamed file is what landed in BACKUP_DIR
            on_disk = os.listdir(tmpdir)
            assert any(f.startswith("canslim_weekly_") for f in on_disk)
            # The original non-weekly name was renamed away
            assert not any(
                f.startswith("canslim_") and "weekly" not in f
                for f in on_disk
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_pg_dump_nonzero_returncode_returns_failed(self):
        """Non-zero returncode raises RuntimeError → except path → failed dict."""
        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        pg_fail = MagicMock(returncode=1, stderr="pg_dump: connection refused")
        tmpdir = tempfile.mkdtemp()
        try:
            result, mock_webhook, _, _ = self._run_perform_backup(
                tmpdir, pg_fail, friday,
            )
            assert result["status"] == "failed"
            assert "pg_dump: connection refused" in result["error"]
            # Urgent webhook on failure
            mock_webhook.assert_called_once()
            assert mock_webhook.call_args.kwargs["priority"] == "urgent"
            assert mock_webhook.call_args.kwargs["title"] == "DB Backup FAILED"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_subprocess_raising_is_caught(self):
        """An OSError from subprocess.run (e.g. pg_dump binary missing)
        is caught by the broad except and returns failed status."""
        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        tmpdir = tempfile.mkdtemp()
        try:
            result, mock_webhook, _, _ = self._run_perform_backup(
                tmpdir, FileNotFoundError("pg_dump not found"), friday,
            )
            assert result["status"] == "failed"
            assert "pg_dump not found" in result["error"]
            assert mock_webhook.call_args.kwargs["priority"] == "urgent"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_removes_partial_file_on_failure(self):
        """If pg_dump partially wrote a file before failing, the except
        block deletes it so we don't leak a half-baked dump."""
        from backend import backup as backup_mod

        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        tmpdir = tempfile.mkdtemp()
        try:
            # subprocess.run "writes a partial file, then reports failure"
            def _partial_then_fail(cmd, **kwargs):
                if "-f" in cmd:
                    filepath = cmd[cmd.index("-f") + 1]
                    with open(filepath, "wb") as f:
                        f.write(b"PARTIAL")
                return MagicMock(returncode=2, stderr="aborted mid-dump")

            with patch.object(backup_mod, "BACKUP_DIR", tmpdir), \
                 patch.object(backup_mod, "datetime",
                              self._fake_datetime_for(friday)), \
                 patch.object(backup_mod, "subprocess") as mock_sub, \
                 patch.object(backup_mod, "cleanup_old_backups"), \
                 patch("backend.email_utils.send_webhook_notification"), \
                 patch.dict(os.environ, {"DATABASE_URL": "postgresql://u:p@h:5432/d"}):
                mock_sub.run.side_effect = _partial_then_fail
                from backend.backup import perform_backup
                result = perform_backup()

            assert result["status"] == "failed"
            # Partial file removed — nothing left behind in BACKUP_DIR
            assert os.listdir(tmpdir) == []
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_swallows_oserror_during_partial_cleanup(self):
        """The inner try/except around os.remove must swallow OSError so the
        original failure path still returns its failed-status dict.
        Without it, a remove failure would bubble out and mask the real error.
        """
        from backend import backup as backup_mod

        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        tmpdir = tempfile.mkdtemp()
        try:
            def _partial_then_fail(cmd, **kwargs):
                if "-f" in cmd:
                    filepath = cmd[cmd.index("-f") + 1]
                    with open(filepath, "wb") as f:
                        f.write(b"PARTIAL")
                return MagicMock(returncode=2, stderr="boom")

            with patch.object(backup_mod, "BACKUP_DIR", tmpdir), \
                 patch.object(backup_mod, "datetime",
                              self._fake_datetime_for(friday)), \
                 patch.object(backup_mod, "subprocess") as mock_sub, \
                 patch.object(backup_mod, "cleanup_old_backups"), \
                 patch.object(backup_mod, "os") as mock_os, \
                 patch("backend.email_utils.send_webhook_notification"):
                # Re-export the os attrs we still want to use
                mock_os.path = os.path
                mock_os.environ = {"DATABASE_URL": "postgresql://u:p@h:5432/d"}
                mock_os.makedirs = os.makedirs
                mock_os.rename = os.rename
                # The partial file is "detected" then os.remove raises
                mock_os.path.exists = MagicMock(return_value=True)
                mock_os.remove = MagicMock(side_effect=OSError("permission denied"))

                mock_sub.run.side_effect = _partial_then_fail
                from backend.backup import perform_backup
                result = perform_backup()  # Must NOT raise

            assert result["status"] == "failed"
            assert "boom" in result["error"]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_failure_skips_cleanup_call(self):
        """cleanup_old_backups runs only on the success path (line 91), not
        in the except block. Pin that contract so a future refactor doesn't
        accidentally invoke retention pruning after a failed dump.
        """
        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        pg_fail = MagicMock(returncode=1, stderr="failure")
        tmpdir = tempfile.mkdtemp()
        try:
            _, _, mock_cleanup, _ = self._run_perform_backup(
                tmpdir, pg_fail, friday,
            )
            mock_cleanup.assert_not_called()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_perform_backup_passes_pgpassword_to_subprocess(self):
        """The pg_dump invocation must inherit PGPASSWORD from _parse_database_url,
        not from the ambient environment. Guards against credential drift if
        someone refactors the env-copy line.
        """
        from backend import backup as backup_mod

        friday = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
        pg_ok = MagicMock(returncode=0, stderr="")
        tmpdir = tempfile.mkdtemp()
        captured_env = {}

        def _capture_env(cmd, env=None, **kwargs):
            captured_env.update(env or {})
            if "-f" in cmd:
                with open(cmd[cmd.index("-f") + 1], "wb") as f:
                    f.write(b"OK")
            return pg_ok

        try:
            with patch.object(backup_mod, "BACKUP_DIR", tmpdir), \
                 patch.object(backup_mod, "datetime",
                              self._fake_datetime_for(friday)), \
                 patch.object(backup_mod, "subprocess") as mock_sub, \
                 patch.object(backup_mod, "cleanup_old_backups"), \
                 patch("backend.email_utils.send_webhook_notification"), \
                 patch.dict(os.environ, {"DATABASE_URL": "postgresql://alice:s3cret@db:5432/canslim"}):
                mock_sub.run.side_effect = _capture_env
                from backend.backup import perform_backup
                result = perform_backup()
            assert result["status"] == "success"
            assert captured_env.get("PGPASSWORD") == "s3cret"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


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

    # Cooldown semantics are now DB-backed via BreakoutAlert rows; the
    # behavioral tests live in tests/test_breakout_monitor_dedup.py. The two
    # legacy module-level-dict tests have been removed (the dict no longer
    # exists post-fix).


# ============== Earnings Gap-Up Alert ==============

class TestGapupAlert:
    def _sample(self, ticker="ALV", score=72):
        return {
            "ticker": ticker, "gap_pct": 12.4, "volume_ratio": 3.2,
            "canslim_score": score, "is_actionable": True,
        }

    def test_skips_when_market_closed(self):
        from backend import earnings_gapup
        earnings_gapup._recent_gapup_alerts.clear()
        with patch("backend.ai_trader.is_market_open", return_value=False), \
             patch("backend.email_utils.send_webhook_notification") as mock_send:
            sent = earnings_gapup.send_gapup_alert([self._sample()])
            assert sent is False
            mock_send.assert_not_called()
            # Cooldown dict must NOT be mutated when market is closed
            assert len(earnings_gapup._recent_gapup_alerts) == 0

    def test_cooldown_prevents_duplicate_alert(self):
        from backend import earnings_gapup
        earnings_gapup._recent_gapup_alerts.clear()
        with patch("backend.ai_trader.is_market_open", return_value=True), \
             patch("backend.email_utils.send_webhook_notification", return_value=True) as mock_send:
            assert earnings_gapup.send_gapup_alert([self._sample("ALV")]) is True
            assert mock_send.call_count == 1
            # Second call with same ticker should be suppressed
            assert earnings_gapup.send_gapup_alert([self._sample("ALV")]) is False
            assert mock_send.call_count == 1
        earnings_gapup._recent_gapup_alerts.clear()

    def test_fresh_ticker_still_alerts_when_others_cooled(self):
        from backend import earnings_gapup
        earnings_gapup._recent_gapup_alerts.clear()
        earnings_gapup._recent_gapup_alerts["ALV"] = datetime.now(timezone.utc)
        with patch("backend.ai_trader.is_market_open", return_value=True), \
             patch("backend.email_utils.send_webhook_notification", return_value=True) as mock_send:
            sent = earnings_gapup.send_gapup_alert(
                [self._sample("ALV"), self._sample("NVDA")]
            )
            assert sent is True
            # Title should reflect 1 fresh, not 2
            args, kwargs = mock_send.call_args
            title = args[0] if args else kwargs.get("title", "")
            assert "1 actionable" in title
        earnings_gapup._recent_gapup_alerts.clear()


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
