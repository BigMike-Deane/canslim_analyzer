"""Mock-driven coverage tests for redis_cache.py.

Companion to tests/test_redis_cache.py — that file's integration
tests `pytest.skip()` when no Redis daemon is running, which leaves
the entire try-block surface (set/get/delete/flush/stats) uncovered
in dev + CI. This file injects a MagicMock client into a fresh
RedisCache instance so every happy path AND exception path runs
unconditionally.

Covers redis_cache.py lines 24, 28-30, 47-48, 51-52, 83-96, 114-135,
139-150, 154-168, 172-185, 189-198, 202-211, 215-238, 246-253.
"""

import sys
import json
import pytest
from unittest.mock import MagicMock, patch

import redis_cache
from redis_cache import RedisCache, get_redis_client, is_redis_available


@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Each test starts with a fresh lazy-client state.

    The module-level `_redis_client` and `_redis_available` are
    state-machine inputs to `get_redis_client()` and
    `is_redis_available()` — resetting between tests keeps the
    lifecycle suite deterministic.
    """
    redis_cache._redis_client = None
    redis_cache._redis_available = False
    yield
    redis_cache._redis_client = None
    redis_cache._redis_available = False


def _make_enabled_cache(client_mock=None) -> RedisCache:
    """Construct a RedisCache and force-enable it with a mock client.

    Bypasses the lazy-import + ping() handshake from get_redis_client
    so each test exercises the post-handshake code paths in isolation.
    """
    cache = RedisCache()
    cache.client = client_mock or MagicMock()
    cache.enabled = True
    # `freshness_intervals` is set from config in __init__; ensure it
    # is a dict for the set() path that reads it.
    if not isinstance(cache.freshness_intervals, dict):
        cache.freshness_intervals = {}
    return cache


# ── get_redis_client lifecycle ─────────────────────────────────────────


class TestGetRedisClient:
    """Covers redis_cache.py:19-59."""

    def test_singleton_returns_cached_client_on_second_call(self):
        """Line 23-24: if _redis_client is not None, short-circuit and reuse."""
        sentinel = MagicMock(name="cached-client")
        redis_cache._redis_client = sentinel
        assert get_redis_client() is sentinel

    def test_returns_none_when_config_disables_redis(self):
        """Lines 27-30: cache.redis.enabled=False short-circuits to None."""
        with patch.object(redis_cache.config, "get", return_value=False):
            result = get_redis_client()
        assert result is None
        assert redis_cache._redis_available is False

    def test_falls_back_on_import_error(self):
        """Lines 50-52: missing redis package → log warning + return None."""
        # Force `import redis` (inside the function) to raise ImportError
        # by injecting None at sys.modules['redis'].
        with patch.object(redis_cache.config, "get", return_value=True), \
             patch.dict(sys.modules, {"redis": None}):
            result = get_redis_client()
        assert result is None
        assert redis_cache._redis_available is False

    def test_successful_connection_sets_available_and_returns_client(self):
        """Lines 32-48: happy path through the redis.Redis constructor + ping()."""
        mock_redis_module = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_module.Redis.return_value = mock_client

        def _config_get(key, default=None):
            return True if key == "cache.redis.enabled" else default

        mock_section = {"redis": {"host": "myhost", "port": 6380, "db": 1, "decode_responses": True}}

        with patch.object(redis_cache.config, "get", side_effect=_config_get), \
             patch.object(redis_cache.config, "get_section", return_value=mock_section), \
             patch.dict(sys.modules, {"redis": mock_redis_module}):
            result = get_redis_client()

        assert result is mock_client
        assert redis_cache._redis_available is True
        mock_client.ping.assert_called_once()
        mock_redis_module.Redis.assert_called_once_with(
            host="myhost", port=6380, db=1,
            decode_responses=True,
            socket_connect_timeout=2, socket_timeout=2,
        )

    def test_ping_failure_falls_back_to_no_client(self):
        """Lines 54-57: ping() raising drops client to None, available=False."""
        mock_redis_module = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("nope")
        mock_redis_module.Redis.return_value = mock_client

        def _config_get(key, default=None):
            return True if key == "cache.redis.enabled" else default

        with patch.object(redis_cache.config, "get", side_effect=_config_get), \
             patch.object(redis_cache.config, "get_section", return_value={"redis": {}}), \
             patch.dict(sys.modules, {"redis": mock_redis_module}):
            result = get_redis_client()

        assert result is None
        assert redis_cache._redis_available is False


# ── is_redis_available ────────────────────────────────────────────────


class TestIsRedisAvailable:
    """Covers redis_cache.py:62-67."""

    def test_returns_false_when_disabled(self):
        with patch.object(redis_cache.config, "get", return_value=False):
            assert is_redis_available() is False

    def test_returns_true_after_successful_handshake(self):
        # Pre-seed the singleton so is_redis_available short-circuits
        # without re-running get_redis_client.
        redis_cache._redis_client = MagicMock()
        redis_cache._redis_available = True
        assert is_redis_available() is True

    def test_lazy_initialises_when_client_is_none(self):
        """If _redis_client is None, is_redis_available triggers get_redis_client."""
        with patch.object(redis_cache.config, "get", return_value=False):
            # Disabled config → get_redis_client returns None → _redis_available stays False
            assert is_redis_available() is False


# ── RedisCache.get ────────────────────────────────────────────────────


class TestGet:
    """Covers redis_cache.py:78-96."""

    def test_returns_none_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.get("AAPL", "earnings") is None

    def test_returns_deserialized_data_on_hit(self):
        client = MagicMock()
        client.get.return_value = json.dumps({"q": [1.0, 1.1, 1.2]})
        cache = _make_enabled_cache(client)
        result = cache.get("AAPL", "earnings")
        assert result == {"q": [1.0, 1.1, 1.2]}
        client.get.assert_called_once_with("canslim:AAPL:earnings")

    def test_returns_none_on_miss(self):
        client = MagicMock()
        client.get.return_value = None
        cache = _make_enabled_cache(client)
        assert cache.get("AAPL", "earnings") is None

    def test_swallows_redis_exception_returns_none(self):
        client = MagicMock()
        client.get.side_effect = RuntimeError("redis down")
        cache = _make_enabled_cache(client)
        assert cache.get("AAPL", "earnings") is None

    def test_swallows_json_decode_error_returns_none(self):
        """Malformed JSON in the cached value is treated as a soft miss."""
        client = MagicMock()
        client.get.return_value = "not-valid-json{"
        cache = _make_enabled_cache(client)
        assert cache.get("AAPL", "earnings") is None


# ── RedisCache.set ────────────────────────────────────────────────────


class TestSet:
    """Covers redis_cache.py:98-135."""

    def test_returns_false_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.set("AAPL", "earnings", {"x": 1}) is False

    def test_uses_explicit_ttl_with_setex(self):
        client = MagicMock()
        cache = _make_enabled_cache(client)
        ok = cache.set("AAPL", "earnings", {"x": 1}, ttl=120)
        assert ok is True
        client.setex.assert_called_once_with(
            "canslim:AAPL:earnings", 120, json.dumps({"x": 1}, default=str)
        )
        client.set.assert_not_called()

    def test_uses_freshness_intervals_when_ttl_is_none(self):
        """Defaults to config-driven freshness_intervals lookup."""
        client = MagicMock()
        cache = _make_enabled_cache(client)
        cache.freshness_intervals = {"earnings": 3600}
        cache.set("AAPL", "earnings", {"x": 1})
        client.setex.assert_called_once()
        ttl_used = client.setex.call_args.args[1]
        assert ttl_used == 3600

    def test_defaults_to_24h_when_data_type_missing_from_freshness(self):
        """Falls back to 86400 when data_type isn't in freshness_intervals."""
        client = MagicMock()
        cache = _make_enabled_cache(client)
        cache.freshness_intervals = {}  # no entry for "unknown"
        cache.set("AAPL", "unknown", {"x": 1})
        ttl_used = client.setex.call_args.args[1]
        assert ttl_used == 86400

    def test_ttl_zero_uses_plain_set_no_expiry(self):
        """TTL=0 → 'always fresh' branch using .set() instead of .setex()."""
        client = MagicMock()
        cache = _make_enabled_cache(client)
        ok = cache.set("AAPL", "price", {"close": 175.0}, ttl=0)
        assert ok is True
        client.set.assert_called_once_with(
            "canslim:AAPL:price", json.dumps({"close": 175.0}, default=str)
        )
        client.setex.assert_not_called()

    def test_swallows_redis_exception_returns_false(self):
        client = MagicMock()
        client.setex.side_effect = RuntimeError("connection reset")
        cache = _make_enabled_cache(client)
        assert cache.set("AAPL", "earnings", {"x": 1}, ttl=10) is False

    def test_json_serializer_uses_default_str_for_non_json_types(self):
        """The `default=str` kwarg lets datetime + Decimal etc. serialize without raising."""
        from datetime import datetime
        client = MagicMock()
        cache = _make_enabled_cache(client)
        payload = {"when": datetime(2026, 5, 10, 12, 0, 0)}
        ok = cache.set("AAPL", "earnings", payload, ttl=60)
        assert ok is True
        serialized = client.setex.call_args.args[2]
        # datetime got rendered through str(...) — exact ISO-ish form
        assert "2026-05-10" in serialized


# ── RedisCache.delete ──────────────────────────────────────────────────


class TestDelete:
    """Covers redis_cache.py:137-150."""

    def test_returns_false_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.delete("AAPL", "earnings") is False

    def test_delete_happy_path(self):
        client = MagicMock()
        cache = _make_enabled_cache(client)
        assert cache.delete("AAPL", "earnings") is True
        client.delete.assert_called_once_with("canslim:AAPL:earnings")

    def test_swallows_exception_returns_false(self):
        client = MagicMock()
        client.delete.side_effect = RuntimeError("eof")
        cache = _make_enabled_cache(client)
        assert cache.delete("AAPL", "earnings") is False


# ── RedisCache.flush_ticker ────────────────────────────────────────────


class TestFlushTicker:
    """Covers redis_cache.py:152-168."""

    def test_returns_zero_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.flush_ticker("AAPL") == 0

    def test_returns_count_when_keys_found(self):
        client = MagicMock()
        client.keys.return_value = ["canslim:AAPL:earnings", "canslim:AAPL:revenue", "canslim:AAPL:m"]
        client.delete.return_value = 3
        cache = _make_enabled_cache(client)
        assert cache.flush_ticker("AAPL") == 3
        client.keys.assert_called_once_with("canslim:AAPL:*")
        client.delete.assert_called_once_with(
            "canslim:AAPL:earnings", "canslim:AAPL:revenue", "canslim:AAPL:m"
        )

    def test_returns_zero_when_no_keys(self):
        client = MagicMock()
        client.keys.return_value = []
        cache = _make_enabled_cache(client)
        assert cache.flush_ticker("UNKNOWN") == 0
        client.delete.assert_not_called()

    def test_swallows_exception_returns_zero(self):
        client = MagicMock()
        client.keys.side_effect = RuntimeError("broken")
        cache = _make_enabled_cache(client)
        assert cache.flush_ticker("AAPL") == 0


# ── RedisCache.flush_all ───────────────────────────────────────────────


class TestFlushAll:
    """Covers redis_cache.py:170-185."""

    def test_returns_false_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.flush_all() is False

    def test_returns_true_after_deleting_canslim_keys(self):
        client = MagicMock()
        client.keys.return_value = ["canslim:AAPL:x", "canslim:MSFT:y"]
        client.delete.return_value = 2
        cache = _make_enabled_cache(client)
        assert cache.flush_all() is True
        client.keys.assert_called_once_with("canslim:*")
        client.delete.assert_called_once_with("canslim:AAPL:x", "canslim:MSFT:y")

    def test_returns_true_when_no_canslim_keys_to_delete(self):
        client = MagicMock()
        client.keys.return_value = []
        cache = _make_enabled_cache(client)
        assert cache.flush_all() is True
        client.delete.assert_not_called()

    def test_swallows_exception_returns_false(self):
        client = MagicMock()
        client.keys.side_effect = RuntimeError("nope")
        cache = _make_enabled_cache(client)
        assert cache.flush_all() is False


# ── RedisCache.get_ttl ─────────────────────────────────────────────────


class TestGetTtl:
    """Covers redis_cache.py:187-198."""

    def test_returns_neg_two_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.get_ttl("AAPL", "earnings") == -2

    def test_returns_ttl_from_redis(self):
        client = MagicMock()
        client.ttl.return_value = 42
        cache = _make_enabled_cache(client)
        assert cache.get_ttl("AAPL", "earnings") == 42
        client.ttl.assert_called_once_with("canslim:AAPL:earnings")

    def test_returns_neg_two_on_exception(self):
        client = MagicMock()
        client.ttl.side_effect = RuntimeError("conn lost")
        cache = _make_enabled_cache(client)
        assert cache.get_ttl("AAPL", "earnings") == -2


# ── RedisCache.exists ──────────────────────────────────────────────────


class TestExists:
    """Covers redis_cache.py:200-211."""

    def test_returns_false_when_disabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.exists("AAPL", "earnings") is False

    def test_returns_true_when_redis_reports_one(self):
        client = MagicMock()
        client.exists.return_value = 1  # redis-py returns int count
        cache = _make_enabled_cache(client)
        assert cache.exists("AAPL", "earnings") is True

    def test_returns_false_when_redis_reports_zero(self):
        client = MagicMock()
        client.exists.return_value = 0
        cache = _make_enabled_cache(client)
        assert cache.exists("AAPL", "earnings") is False

    def test_swallows_exception_returns_false(self):
        client = MagicMock()
        client.exists.side_effect = RuntimeError("boom")
        cache = _make_enabled_cache(client)
        assert cache.exists("AAPL", "earnings") is False


# ── RedisCache.get_stats ───────────────────────────────────────────────


class TestGetStats:
    """Covers redis_cache.py:213-238."""

    def test_returns_disabled_shape_when_not_enabled(self):
        cache = _make_enabled_cache()
        cache.enabled = False
        assert cache.get_stats() == {"enabled": False}

    def test_returns_full_stats_payload_on_success(self):
        client = MagicMock()

        def _info(section):
            return {
                "stats": {
                    "total_commands_processed": 1234,
                    "keyspace_hits": 80,
                    "keyspace_misses": 20,
                },
                "keyspace": {"db0": {"keys": 500}},
            }[section]

        client.info.side_effect = _info
        client.keys.return_value = ["canslim:A:x", "canslim:B:y", "canslim:C:z"]
        cache = _make_enabled_cache(client)
        stats = cache.get_stats()
        assert stats == {
            "enabled": True,
            "connected": True,
            "total_keys": 500,
            "canslim_keys": 3,
            "total_commands": 1234,
            "keyspace_hits": 80,
            "keyspace_misses": 20,
            "hit_rate": 80.0,
        }

    def test_returns_error_shape_when_redis_raises(self):
        client = MagicMock()
        client.info.side_effect = RuntimeError("info failed")
        cache = _make_enabled_cache(client)
        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["connected"] is False
        assert "info failed" in stats["error"]

    def test_handles_missing_db0_keyspace_section(self):
        """If redis.info('keyspace') has no db0 key, total_keys should default to 0."""
        client = MagicMock()
        client.info.side_effect = lambda section: {
            "stats": {"total_commands_processed": 1},
            "keyspace": {},  # no db0
        }[section]
        client.keys.return_value = []
        cache = _make_enabled_cache(client)
        stats = cache.get_stats()
        assert stats["total_keys"] == 0
        assert stats["canslim_keys"] == 0


# ── RedisCache._calculate_hit_rate ─────────────────────────────────────


class TestCalculateHitRate:
    """Covers redis_cache.py:244-253."""

    def test_zero_total_returns_zero(self):
        cache = _make_enabled_cache()
        assert cache._calculate_hit_rate({"keyspace_hits": 0, "keyspace_misses": 0}) == 0.0

    def test_mixed_hits_and_misses_returns_percentage(self):
        cache = _make_enabled_cache()
        assert cache._calculate_hit_rate({"keyspace_hits": 80, "keyspace_misses": 20}) == 80.0

    def test_rounds_to_two_decimal_places(self):
        cache = _make_enabled_cache()
        # 1/3 = 33.333...% → rounded to 33.33
        result = cache._calculate_hit_rate({"keyspace_hits": 1, "keyspace_misses": 2})
        assert result == 33.33

    def test_missing_keys_default_to_zero(self):
        cache = _make_enabled_cache()
        assert cache._calculate_hit_rate({}) == 0.0


# ── _make_key namespace ───────────────────────────────────────────────


class TestMakeKey:
    """Covers redis_cache.py:240-242. (also lightly exercised in existing
    test_redis_cache.py — additional shape assertions live here.)"""

    def test_key_format_is_canslim_ticker_datatype(self):
        cache = _make_enabled_cache()
        assert cache._make_key("AAPL", "earnings") == "canslim:AAPL:earnings"

    def test_key_preserves_uppercase_ticker_and_lowercase_data_type(self):
        cache = _make_enabled_cache()
        assert cache._make_key("MSFT", "revenue") == "canslim:MSFT:revenue"
