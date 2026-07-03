"""
Database Backup Module

Handles PostgreSQL backups with rotation and notifications.
Backups are stored in /app/data/backups/ (mounted volume persists on host).
"""

import os
import subprocess
import glob
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/app/data/backups")
DAILY_RETENTION = 7
WEEKLY_RETENTION = 4


def _parse_database_url():
    """Parse DATABASE_URL into pg_dump connection params."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL not set")
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "postgres",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "canslim",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "canslim",
    }


def perform_backup() -> dict:
    """Run pg_dump and save compressed backup. Returns status dict."""
    from backend.email_utils import send_webhook_notification

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    weekday = datetime.now(timezone.utc).strftime("%A").lower()
    filename = f"canslim_{timestamp}.dump"
    filepath = os.path.join(BACKUP_DIR, filename)

    try:
        params = _parse_database_url()
        env = os.environ.copy()
        env["PGPASSWORD"] = params["password"]

        cmd = [
            "pg_dump",
            "-h", params["host"],
            "-p", params["port"],
            "-U", params["user"],
            "-d", params["dbname"],
            "-Fc",  # Custom format (compressed)
            "-f", filepath,
        ]

        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {result.stderr.strip()}")

        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)

        # Tag Sunday backups as weekly (won't be rotated with daily)
        if weekday == "sunday":
            weekly_name = f"canslim_weekly_{timestamp}.dump"
            weekly_path = os.path.join(BACKUP_DIR, weekly_name)
            os.rename(filepath, weekly_path)
            filepath = weekly_path
            filename = weekly_name

        logger.info(f"Backup complete: {filename} ({size_mb:.1f} MB)")

        send_webhook_notification(
            title="DB Backup Complete",
            message=f"{filename} ({size_mb:.1f} MB)",
            priority="low",
            tags=["white_check_mark", "floppy_disk"],
        )

        # Clean up old backups
        cleanup_old_backups()

        return {
            "status": "success",
            "filename": filename,
            "size_mb": round(size_mb, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        # Remove partial file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

        send_webhook_notification(
            title="DB Backup FAILED",
            message=str(e)[:200],
            priority="urgent",
            tags=["rotating_light", "x"],
        )

        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def cleanup_old_backups():
    """Retain last N daily + M weekly backups."""
    daily_files = sorted(
        glob.glob(os.path.join(BACKUP_DIR, "canslim_2*.dump")),
        key=os.path.getmtime,
        reverse=True,
    )
    weekly_files = sorted(
        glob.glob(os.path.join(BACKUP_DIR, "canslim_weekly_*.dump")),
        key=os.path.getmtime,
        reverse=True,
    )

    # Guard each remove: cleanup runs inside create_backup's try block, so an
    # unhandled OSError here (bad perms / NFS hiccup on a STALE file) jumped to
    # the except path, which deletes the just-created VALID backup and alerts
    # "backup FAILED" — silently draining the retention window to zero.
    for f in daily_files[DAILY_RETENTION:]:
        try:
            logger.info(f"Removing old daily backup: {os.path.basename(f)}")
            os.remove(f)
        except OSError as e:
            logger.warning(f"Could not remove old daily backup {f}: {e}")

    for f in weekly_files[WEEKLY_RETENTION:]:
        try:
            logger.info(f"Removing old weekly backup: {os.path.basename(f)}")
            os.remove(f)
        except OSError as e:
            logger.warning(f"Could not remove old weekly backup {f}: {e}")


def list_backups() -> list:
    """Return list of available backups with metadata."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(BACKUP_DIR, "canslim_*.dump")),
        key=os.path.getmtime,
        reverse=True,
    )
    result = []
    for f in files:
        stat = os.stat(f)
        result.append({
            "filename": os.path.basename(f),
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "created": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "is_weekly": "weekly" in os.path.basename(f),
        })
    return result


def get_backup_status() -> dict:
    """Return current backup status for health dashboard."""
    backups = list_backups()
    last = backups[0] if backups else None
    return {
        "last_backup": last,
        "total_backups": len(backups),
        "daily_count": sum(1 for b in backups if not b["is_weekly"]),
        "weekly_count": sum(1 for b in backups if b["is_weekly"]),
        "backup_dir": BACKUP_DIR,
    }
