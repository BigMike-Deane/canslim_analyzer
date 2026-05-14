"""Tests for the global SystemSetting key/value store + scanner-config
persistence helpers.

Shipped 2026-05-13 alongside the Admin Scanner panel so admin-side cadence
changes survive container restarts.
"""
import os

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

import pytest

from backend.database import (
    init_db, SessionLocal, SystemSetting,
    get_system_setting, set_system_setting,
)
from backend import scheduler


@pytest.fixture
def clean_settings():
    """Delete any system_settings rows before/after each test so tests
    don't bleed state into one another. The in-memory SQLite (via
    conftest) persists for the session, so we clean explicitly.
    """
    init_db()
    db = SessionLocal()
    db.query(SystemSetting).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(SystemSetting).delete()
        db.commit()
        db.close()


class TestSystemSettingHelpers:
    def test_get_returns_default_when_missing(self, clean_settings):
        assert get_system_setting("nonexistent") is None
        assert get_system_setting("nonexistent", "fallback") == "fallback"
        assert get_system_setting("nonexistent", {"a": 1}) == {"a": 1}

    def test_set_then_get_roundtrips_dict(self, clean_settings):
        assert set_system_setting("scanner.config", {"source": "all", "interval_minutes": 35}) is True
        assert get_system_setting("scanner.config") == {"source": "all", "interval_minutes": 35}

    def test_set_then_get_roundtrips_int(self, clean_settings):
        assert set_system_setting("some_int", 42) is True
        assert get_system_setting("some_int") == 42

    def test_set_then_get_roundtrips_list(self, clean_settings):
        assert set_system_setting("some_list", ["a", "b", "c"]) is True
        assert get_system_setting("some_list") == ["a", "b", "c"]

    def test_set_upserts_existing_key(self, clean_settings):
        set_system_setting("k", "first")
        set_system_setting("k", "second")
        assert get_system_setting("k") == "second"
        # Single row, not two.
        assert clean_settings.query(SystemSetting).filter_by(key="k").count() == 1

    def test_get_returns_default_on_unparseable_json(self, clean_settings):
        """A row with malformed JSON in value column shouldn't crash callers;
        return default instead. Tests the fail-soft contract."""
        # Inject malformed JSON directly.
        clean_settings.add(SystemSetting(key="bad", value="{not valid json"))
        clean_settings.commit()
        assert get_system_setting("bad", "fallback") == "fallback"


class TestLoadPersistedScannerConfig:
    def test_returns_defaults_when_no_row(self, clean_settings):
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert src == "all"
        assert intv == 35

    def test_returns_persisted_values_when_present(self, clean_settings):
        set_system_setting("scanner.config", {"source": "sp500", "interval_minutes": 15})
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert src == "sp500"
        assert intv == 15

    def test_invalid_source_falls_back_to_default(self, clean_settings):
        set_system_setting("scanner.config", {"source": "bogus", "interval_minutes": 60})
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert src == "all"
        # Interval was valid → kept.
        assert intv == 60

    def test_interval_out_of_range_falls_back(self, clean_settings):
        set_system_setting("scanner.config", {"source": "russell", "interval_minutes": 999})
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert src == "russell"
        assert intv == 35

    def test_interval_below_range_falls_back(self, clean_settings):
        set_system_setting("scanner.config", {"source": "all", "interval_minutes": 2})
        _, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert intv == 35

    def test_interval_not_int_falls_back(self, clean_settings):
        set_system_setting("scanner.config", {"source": "all", "interval_minutes": "fifteen"})
        _, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert intv == 35

    def test_malformed_payload_falls_back(self, clean_settings):
        # Stored as a string, not a dict.
        set_system_setting("scanner.config", "not_a_dict")
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="top50", default_interval=20,
        )
        assert src == "top50"
        assert intv == 20

    def test_partial_payload_uses_defaults_for_missing(self, clean_settings):
        # Only source present → interval falls back.
        set_system_setting("scanner.config", {"source": "sp500"})
        src, intv = scheduler.load_persisted_scanner_config(
            default_source="all", default_interval=35,
        )
        assert src == "sp500"
        assert intv == 35


class TestPersistedConfigIntegration:
    def test_persist_helper_writes_to_db(self, clean_settings):
        scheduler._persist_scanner_config("russell", 60)
        cfg = get_system_setting("scanner.config")
        assert cfg == {"source": "russell", "interval_minutes": 60}

    def test_update_scan_config_persists_when_disabled(self, clean_settings):
        # Force disabled state so update_scan_config takes the
        # _persist_scanner_config branch instead of restart-with-persist.
        scheduler._scan_config["enabled"] = False
        scheduler.update_scan_config(source="top50", interval_minutes=20)
        cfg = get_system_setting("scanner.config")
        assert cfg == {"source": "top50", "interval_minutes": 20}


class TestSetSystemSettingFailSoft:
    """The fail-soft contract on set_system_setting: a DB error must return
    False rather than raise. Callers in boot paths rely on this so a
    degraded DB doesn't crash the runtime save."""

    def test_returns_false_when_session_raises(self):
        from unittest.mock import patch
        with patch(
            "backend.database.SessionLocal",
            side_effect=RuntimeError("DB down"),
        ):
            assert set_system_setting("any.key", {"x": 1}) is False
