"""Tests for backend/backup.py verify_latest_backup.

The weekly restore-verify job (Sun 3:30 UTC) restores the newest dump into a
scratch DB and checks row-count floors. These tests mock pg_restore and the
psycopg2 connections — the real end-to-end restore was proven manually on
2026-07-21 (4,401 stocks / 2.43M scores from that day's dump).
"""

import os
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("CANSLIM_ENV", "development")

from backend import backup

# DATABASE_URL must NOT be set at module level: pytest imports every test
# module at collection, and a module-level env write leaks into unrelated
# tests (same hazard class as the dependency-override collisions documented
# in CLAUDE.md — collection order decides who breaks).


@pytest.fixture
def db_url_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://canslim:pw@postgres:5432/canslim")


def _mock_admin_conn():
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


def _mock_verify_conn(counts):
    """Connection whose cursor returns `counts` values in VERIFY_MIN_ROWS
    table order, one per execute/fetchone pair."""
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.side_effect = [(c,) for c in counts]
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


class TestVerifyLatestBackup:
    def test_no_backups_fails(self, db_url_env):
        with patch.object(backup, "list_backups", return_value=[]):
            result = backup.verify_latest_backup()
        assert result["status"] == "failed"
        assert "no backups" in result["error"]

    def test_success_when_floors_met(self, db_url_env):
        counts = [4401, 197, 2432802]  # stocks, trades, scores
        with patch.object(backup, "list_backups",
                          return_value=[{"filename": "canslim_x.dump"}]), \
             patch.object(backup, "_admin_conn", side_effect=_mock_admin_conn), \
             patch("psycopg2.connect", return_value=_mock_verify_conn(counts)), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
            result = backup.verify_latest_backup()
        assert result["status"] == "success", result
        assert result["counts"]["stocks"] == 4401

    def test_empty_table_fails_floor(self, db_url_env):
        """A structurally-empty restore (0 stocks) must fail loudly even if
        pg_restore exited 0."""
        counts = [0, 197, 2432802]
        with patch.object(backup, "list_backups",
                          return_value=[{"filename": "canslim_x.dump"}]), \
             patch.object(backup, "_admin_conn", side_effect=_mock_admin_conn), \
             patch("psycopg2.connect", return_value=_mock_verify_conn(counts)), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")):
            result = backup.verify_latest_backup()
        assert result["status"] == "failed"
        assert "stocks" in result["error"]

    def test_restore_exception_fails_and_still_drops_scratch_db(self, db_url_env):
        admin_conns = []

        def track_admin():
            conn = _mock_admin_conn()
            admin_conns.append(conn)
            return conn

        with patch.object(backup, "list_backups",
                          return_value=[{"filename": "canslim_x.dump"}]), \
             patch.object(backup, "_admin_conn", side_effect=track_admin), \
             patch("subprocess.run", side_effect=OSError("pg_restore missing")):
            result = backup.verify_latest_backup()
        assert result["status"] == "failed"
        assert "pg_restore missing" in result["error"]
        # finally-block must have issued the DROP on the existing admin conn
        dropped = any(
            any("DROP DATABASE" in str(c) for c in conn.cursor.return_value.__enter__.return_value.execute.call_args_list)
            for conn in admin_conns
        )
        assert dropped, "scratch DB was not dropped on the failure path"


class TestPerformBackupKeepsDump:
    """Aug-6 audit #18: perform_backup's except deletes `filepath` (the dump).
    Post-success housekeeping (rename/webhook/cleanup) must not be able to
    reach that path — a cleanup hiccup used to destroy the fresh valid backup
    and alert FAILED."""

    def test_cleanup_raise_does_not_delete_fresh_dump(self, db_url_env, monkeypatch, tmp_path):
        from pathlib import Path
        monkeypatch.setattr(backup, "BACKUP_DIR", str(tmp_path))

        def fake_pg_dump(cmd, **kwargs):
            Path(cmd[cmd.index("-f") + 1]).write_bytes(b"x" * 1024)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_pg_dump), \
             patch.object(backup, "cleanup_old_backups",
                          side_effect=OSError("mtime race on rotated file")), \
             patch("backend.email_utils.send_webhook_notification", return_value=True):
            result = backup.perform_backup()

        assert result["status"] == "success"
        dumps = list(tmp_path.glob("canslim_*.dump"))
        assert len(dumps) == 1, "fresh dump was deleted by housekeeping failure"
        assert dumps[0].stat().st_size > 0

    def test_dump_failure_still_removes_partial_and_reports_failed(self, db_url_env, monkeypatch, tmp_path):
        from pathlib import Path
        monkeypatch.setattr(backup, "BACKUP_DIR", str(tmp_path))

        def fake_pg_dump_fail(cmd, **kwargs):
            Path(cmd[cmd.index("-f") + 1]).write_bytes(b"partial")
            return MagicMock(returncode=1, stderr="disk full")

        with patch("subprocess.run", side_effect=fake_pg_dump_fail), \
             patch("backend.email_utils.send_webhook_notification", return_value=True):
            result = backup.perform_backup()

        assert result["status"] == "failed"
        assert list(tmp_path.glob("canslim_*.dump")) == [], "partial dump not cleaned up"
