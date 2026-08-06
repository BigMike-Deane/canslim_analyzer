"""Regression tests for two backend/scheduler.py audit fixes.

FIX 1 — is_scanning latch gap:
    run_continuous_scan() sets _scan_config["is_scanning"] = True under
    _state_lock, but the try/finally that clears the latch used to start
    ~485 lines later. Any exception in the gap (ticker assembly, scorer
    construction, market-direction fetch) escaped with the latch stuck
    True, so every subsequent tick exited at "Scan already in progress"
    forever and _record_failure("scan", ...) was never called. The try
    now begins immediately after the latch block; these tests raise at
    two points inside the formerly unprotected gap and assert the latch
    clears and the failure is recorded.

FIX 2 — health persistence to Redis:
    _system_health["errors_today"] is a deque, which json.dumps rejects,
    so _persist_health_to_redis raised TypeError on every call (swallowed
    by a bare except) and restore-across-restarts never worked. Persist
    now converts the deque to a list; restore rebuilds a bounded
    deque(maxlen=50).
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import scheduler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class FakeRedis:
    """Minimal get/set redis stand-in storing raw values."""

    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)


@pytest.fixture
def fake_redis(monkeypatch):
    """Route scheduler's health persistence at a fake redis client."""
    import redis_cache

    client = FakeRedis()
    monkeypatch.setattr(redis_cache, "get_redis_client", lambda: client)
    return client


@pytest.fixture
def health_snapshot():
    """Snapshot + restore _system_health so tests don't leak state."""
    saved = dict(scheduler._system_health)
    saved_errors = list(scheduler._system_health["errors_today"])
    yield
    scheduler._system_health.clear()
    scheduler._system_health.update(saved)
    scheduler._system_health["errors_today"] = deque(saved_errors, maxlen=50)


@pytest.fixture
def scan_config_snapshot():
    """Snapshot + restore _scan_config; force is_scanning False on exit."""
    saved = dict(scheduler._scan_config)
    yield
    scheduler._scan_config.clear()
    scheduler._scan_config.update(saved)
    scheduler._scan_config["is_scanning"] = False


# ---------------------------------------------------------------------------
# FIX 2 — health persistence round-trip
# ---------------------------------------------------------------------------

class TestHealthPersistence:
    def test_persist_serializes_deque_without_typeerror(
        self, fake_redis, health_snapshot
    ):
        scheduler._system_health["errors_today"].append(
            {"task": "scan", "error": "boom", "timestamp": "2026-08-06T00:00:00"}
        )
        scheduler._system_health["consecutive_scan_failures"] = 3

        scheduler._persist_health_to_redis()

        raw = fake_redis.store.get(scheduler._HEALTH_REDIS_KEY)
        assert raw is not None, "persist wrote nothing (TypeError regression?)"
        payload = json.loads(raw)
        assert payload["consecutive_scan_failures"] == 3
        assert payload["errors_today"] == [
            {"task": "scan", "error": "boom", "timestamp": "2026-08-06T00:00:00"}
        ]

    def test_round_trip_preserves_counters_and_bounded_deque(
        self, fake_redis, health_snapshot
    ):
        scheduler._system_health["consecutive_scan_failures"] = 2
        scheduler._system_health["last_successful_scan"] = "2026-08-06T01:00:00+00:00"
        scheduler._system_health["errors_today"].append(
            {"task": "scan", "error": "e1", "timestamp": "t1"}
        )
        scheduler._persist_health_to_redis()

        # Simulate a process restart: wipe in-memory state.
        scheduler._system_health["consecutive_scan_failures"] = 0
        scheduler._system_health["last_successful_scan"] = None
        scheduler._system_health["errors_today"] = deque(maxlen=50)

        scheduler._restore_health_from_redis()

        assert scheduler._system_health["consecutive_scan_failures"] == 2
        assert (
            scheduler._system_health["last_successful_scan"]
            == "2026-08-06T01:00:00+00:00"
        )
        errors = scheduler._system_health["errors_today"]
        assert isinstance(errors, deque), "restore must rebuild a deque, not a list"
        assert errors.maxlen == 50, "restored deque must stay bounded"
        assert list(errors) == [{"task": "scan", "error": "e1", "timestamp": "t1"}]

    def test_restore_bounds_oversized_error_list(self, fake_redis, health_snapshot):
        oversized = [{"task": "scan", "error": f"e{i}", "timestamp": str(i)}
                     for i in range(60)]
        fake_redis.store[scheduler._HEALTH_REDIS_KEY] = json.dumps(
            {**{k: v for k, v in scheduler._system_health.items()
                if k != "errors_today"},
             "errors_today": oversized}
        )

        scheduler._restore_health_from_redis()

        errors = scheduler._system_health["errors_today"]
        assert isinstance(errors, deque)
        assert errors.maxlen == 50
        assert len(errors) == 50
        # deque(iterable, maxlen=50) keeps the most recent (last) 50 entries
        assert errors[0]["error"] == "e10"
        assert errors[-1]["error"] == "e59"

    def test_persist_never_raises_on_redis_error(self, monkeypatch, health_snapshot):
        import redis_cache

        class ExplodingRedis:
            def set(self, key, value):
                raise ConnectionError("redis down")

        monkeypatch.setattr(redis_cache, "get_redis_client", lambda: ExplodingRedis())
        scheduler._persist_health_to_redis()  # must not raise

    def test_restore_missing_errors_key_keeps_existing_deque(
        self, fake_redis, health_snapshot
    ):
        """Old/partial payloads without errors_today must not clobber the deque."""
        fake_redis.store[scheduler._HEALTH_REDIS_KEY] = json.dumps(
            {"consecutive_scan_failures": 5}
        )
        scheduler._system_health["errors_today"] = deque(maxlen=50)

        scheduler._restore_health_from_redis()

        assert scheduler._system_health["consecutive_scan_failures"] == 5
        errors = scheduler._system_health["errors_today"]
        assert isinstance(errors, deque)
        assert errors.maxlen == 50


# ---------------------------------------------------------------------------
# FIX 1 — is_scanning latch cleared on exceptions in the former gap
# ---------------------------------------------------------------------------

@pytest.fixture
def silence_alarm(monkeypatch):
    """Capture _fire_system_alarm calls instead of hitting ntfy/email."""
    calls = []
    monkeypatch.setattr(
        scheduler, "_fire_system_alarm", lambda *a, **k: calls.append((a, k))
    )
    return calls


@pytest.fixture
def no_redis(monkeypatch):
    """_record_failure persists health; keep it off the network."""
    import redis_cache

    monkeypatch.setattr(redis_cache, "get_redis_client", lambda: None)


class TestScanLatchClearedOnEarlyException:
    def test_exception_in_ticker_assembly_clears_latch_and_records_failure(
        self, monkeypatch, scan_config_snapshot, health_snapshot,
        silence_alarm, no_redis,
    ):
        """get_portfolio_tickers() is the first call in the formerly
        unprotected gap between the latch set and the old try."""
        import sp500_tickers

        def boom():
            raise RuntimeError("boom in ticker assembly")

        monkeypatch.setattr(sp500_tickers, "get_portfolio_tickers", boom)
        scheduler._scan_config["is_scanning"] = False
        before_failures = scheduler._system_health["consecutive_scan_failures"]

        scheduler.run_continuous_scan()  # must not raise

        assert scheduler._scan_config["is_scanning"] is False, (
            "latch stuck True: every future scan tick would skip forever"
        )
        assert scheduler._scan_config["phase"] is None
        assert scheduler._scan_config["phase_detail"] is None
        assert scheduler._scan_config["current_phase"] is None
        assert scheduler._scan_config["phase_label"] is None
        assert scheduler._scan_config["last_scan_end"] is not None

        assert (
            scheduler._system_health["consecutive_scan_failures"]
            == before_failures + 1
        ), "_record_failure('scan', ...) did not fire for a gap exception"
        assert "boom in ticker assembly" in (
            scheduler._system_health["last_scan_error"]["error"]
        )
        assert len(silence_alarm) >= 1, "consecutive-failure alarm path never ran"

    def test_exception_in_market_direction_fetch_clears_latch(
        self, monkeypatch, scan_config_snapshot, health_snapshot,
        silence_alarm, no_redis,
    ):
        """get_cached_market_direction() sits deeper in the former gap
        (after ticker assembly, before scorer construction)."""
        import data_fetcher
        import sp500_tickers

        monkeypatch.setattr(sp500_tickers, "get_portfolio_tickers", lambda: [])
        monkeypatch.setattr(
            sp500_tickers, "get_sp500_tickers", lambda: ["AAPL", "MSFT"]
        )
        monkeypatch.setattr(scheduler, "_check_universe_shrink", lambda n: None)

        def boom(force_refresh=False):
            raise RuntimeError("boom in market direction")

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction", boom)

        scheduler._scan_config["is_scanning"] = False
        scheduler._scan_config["source"] = "sp500"
        before_failures = scheduler._system_health["consecutive_scan_failures"]

        scheduler.run_continuous_scan()  # must not raise

        assert scheduler._scan_config["is_scanning"] is False
        assert scheduler._scan_config["phase"] is None
        assert (
            scheduler._system_health["consecutive_scan_failures"]
            == before_failures + 1
        )
        assert "boom in market direction" in (
            scheduler._system_health["last_scan_error"]["error"]
        )

    def test_next_scan_is_not_skipped_after_gap_exception(
        self, monkeypatch, scan_config_snapshot, health_snapshot,
        silence_alarm, no_redis,
    ):
        """The original bug: after one gap exception, every later tick hit
        'Scan already in progress, skipping...'. Two consecutive failing
        runs must BOTH reach the error path."""
        import sp500_tickers

        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise RuntimeError(f"boom #{calls['n']}")

        monkeypatch.setattr(sp500_tickers, "get_portfolio_tickers", boom)
        scheduler._scan_config["is_scanning"] = False
        before_failures = scheduler._system_health["consecutive_scan_failures"]

        scheduler.run_continuous_scan()
        scheduler.run_continuous_scan()

        assert calls["n"] == 2, "second run was skipped — latch was left set"
        assert (
            scheduler._system_health["consecutive_scan_failures"]
            == before_failures + 2
        )
