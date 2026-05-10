"""Tests for backend/price_cache.py — closes the 34.78% baseline
(90/138 lines uncovered as of 08a327c).

The disk-backed SQLite price cache sits between yfinance and the
backtester; every backtest hits it, including any catch-up runs near the
2026-06-18 Approach 2 verdict. Storage shape is JSON-records via
`df.to_json(orient='records')` + `json.loads(...)`. The `get` path
enforces TTL via `total_seconds() / 86400` (precision fix already pinned
in tests/test_bug_regressions.py::TestPriceCacheExpiry).

Eval-safe posture: every test isolates a fresh SQLite file via the
`tmp_path` fixture; no real yfinance calls; no scoring; no trading. The
module-level `_cache_instance` singleton is reset in setup/teardown so
no test bleeds an instance pinned to a tmp_path that's been cleaned up.
"""

import json
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

import backend.price_cache as pc_mod  # noqa: E402
from backend.price_cache import PriceHistoryCache, get_price_cache  # noqa: E402


class _SingletonResetMixin:
    """Reset the module-level _cache_instance around each test so a stale
    singleton pinned to a tmp_path that has been cleaned up doesn't bleed
    into the next test. Mirrors test_backtest_queue's _SingletonResetMixin."""

    def setup_method(self):
        self._original_instance = pc_mod._cache_instance
        pc_mod._cache_instance = None

    def teardown_method(self):
        pc_mod._cache_instance = self._original_instance


def _make_df(n_rows=5, start="2026-01-01"):
    """Build a small price DataFrame matching the schema the backtester
    writes: date + OHLCV columns. `date` is stored as `date` objects so
    the round-trip exercises the str/date conversion in get()/set()."""
    base = date.fromisoformat(start)
    rows = []
    for i in range(n_rows):
        d = base + timedelta(days=i)
        rows.append({
            "date": d,
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 1_000_000 + i * 1000,
        })
    return pd.DataFrame(rows)


# ============================================================================
# __init__ + _init_db
# ============================================================================
class TestInit(_SingletonResetMixin):
    def test_creates_db_file_at_provided_path(self, tmp_path):
        db_path = tmp_path / "subdir" / "cache.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache = PriceHistoryCache(db_path=str(db_path))
        assert db_path.exists()
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='price_cache'"
            ).fetchone()
        assert row is not None
        assert cache.expiry_days == 30  # default

    def test_indexes_exist_after_init(self, tmp_path):
        db_path = tmp_path / "cache.db"
        PriceHistoryCache(db_path=str(db_path))
        with sqlite3.connect(str(db_path)) as conn:
            indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_price_cache_created" in indexes
        assert "idx_price_cache_ticker" in indexes

    def test_re_init_is_idempotent(self, tmp_path):
        db_path = tmp_path / "cache.db"
        PriceHistoryCache(db_path=str(db_path))
        cache2 = PriceHistoryCache(db_path=str(db_path), expiry_days=7)
        assert cache2.expiry_days == 7

    def test_default_db_path_resolves_under_data_dir(self, tmp_path, monkeypatch):
        # Without a db_path arg, the cache resolves to <project_root>/data/price_cache.db.
        # Redirect the project root so we don't pollute the real ./data dir.
        fake_project = tmp_path / "fake_project"
        fake_project.mkdir()
        backend_dir = fake_project / "backend"
        backend_dir.mkdir()
        monkeypatch.setattr(pc_mod, "__file__", str(backend_dir / "price_cache.py"))
        cache = PriceHistoryCache()
        assert cache.db_path.endswith("price_cache.db")
        assert "data" in cache.db_path
        assert (fake_project / "data" / "price_cache.db").exists()


# ============================================================================
# _make_cache_key
# ============================================================================
class TestMakeCacheKey(_SingletonResetMixin):
    def test_key_includes_all_three_components(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        key = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 2, 1))
        assert "AAPL" in key
        assert "2026-01-01" in key
        assert "2026-02-01" in key

    def test_key_stable_for_same_inputs(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        a = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 2, 1))
        b = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 2, 1))
        assert a == b

    def test_different_tickers_yield_different_keys(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        a = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 2, 1))
        b = cache._make_cache_key("MSFT", date(2026, 1, 1), date(2026, 2, 1))
        assert a != b


# ============================================================================
# get / set round-trip
# ============================================================================
class TestGetSet(_SingletonResetMixin):
    def test_set_then_get_round_trip(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        df = _make_df(n_rows=3)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), df)
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is not None
        assert len(out) == 3
        assert all(isinstance(d, date) for d in out["date"])
        assert list(out["close"]) == [101.0, 102.0, 103.0]

    def test_get_miss_returns_none_and_increments_misses(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        out = cache.get("NOPE", date(2026, 1, 1), date(2026, 1, 2))
        assert out is None
        assert cache._misses == 1
        assert cache._hits == 0

    def test_set_no_op_on_empty_or_none_df(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 2), None)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 2), pd.DataFrame())
        assert cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 2)) is None
        with sqlite3.connect(cache.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 0

    def test_set_upsert_replaces_existing_row(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        df_a = _make_df(n_rows=2)
        df_b = _make_df(n_rows=5)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 6), df_a)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 6), df_b)  # overwrite
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 6))
        assert len(out) == 5
        with sqlite3.connect(cache.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 1

    def test_hit_increments_counter(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert cache._hits == 2
        assert cache._misses == 0


# ============================================================================
# TTL / expiry
# ============================================================================
class TestExpiry(_SingletonResetMixin):
    def _backdate_row(self, db_path, cache_key, days_old):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE price_cache SET created_at = ? WHERE cache_key = ?",
                (old_ts, cache_key),
            )
            conn.commit()

    def test_expired_row_returns_none_and_deletes_self(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"), expiry_days=30)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        key = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        self._backdate_row(cache.db_path, key, days_old=45)

        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is None
        # Expired row is deleted as a side effect of the read path.
        with sqlite3.connect(cache.db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM price_cache WHERE cache_key = ?", (key,)
            ).fetchone()[0]
        assert n == 0
        assert cache._misses == 1

    def test_fresh_row_under_ttl_returns_data(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"), expiry_days=30)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        key = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        self._backdate_row(cache.db_path, key, days_old=10)
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is not None
        assert len(out) == 3

    def test_naive_created_at_treated_as_utc(self, tmp_path):
        # Older rows from before the timezone-aware fix may carry naive
        # timestamps. The get() path normalizes them via .replace(tzinfo=utc).
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"), expiry_days=30)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        key = cache._make_cache_key("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        # Strip tzinfo to simulate the naive timestamp shape from older rows.
        naive = (datetime.now(timezone.utc) - timedelta(days=5)).replace(tzinfo=None).isoformat()
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE price_cache SET created_at = ? WHERE cache_key = ?",
                (naive, key),
            )
            conn.commit()
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is not None


# ============================================================================
# cleanup_expired
# ============================================================================
class TestCleanup(_SingletonResetMixin):
    def test_deletes_only_expired_rows_returns_count(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"), expiry_days=30)
        cache.set("OLD1", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        cache.set("OLD2", date(2026, 1, 2), date(2026, 1, 5), _make_df(2))
        cache.set("NEW1", date(2026, 1, 3), date(2026, 1, 6), _make_df(2))

        old_ts = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE price_cache SET created_at = ? WHERE ticker IN ('OLD1','OLD2')",
                (old_ts,),
            )
            conn.commit()

        deleted = cache.cleanup_expired()
        assert deleted == 2
        with sqlite3.connect(cache.db_path) as conn:
            tickers = {r[0] for r in conn.execute("SELECT ticker FROM price_cache").fetchall()}
        assert tickers == {"NEW1"}

    def test_cleanup_empty_db_returns_zero(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        assert cache.cleanup_expired() == 0


# ============================================================================
# clear_ticker / clear_all
# ============================================================================
class TestClear(_SingletonResetMixin):
    def test_clear_ticker_removes_only_that_tickers_rows(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        cache.set("AAPL", date(2026, 2, 1), date(2026, 2, 4), _make_df(2))
        cache.set("MSFT", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))

        deleted = cache.clear_ticker("AAPL")
        assert deleted == 2
        with sqlite3.connect(cache.db_path) as conn:
            survivors = {r[0] for r in conn.execute("SELECT ticker FROM price_cache").fetchall()}
        assert survivors == {"MSFT"}

    def test_clear_ticker_returns_zero_when_absent(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        assert cache.clear_ticker("NOPE") == 0

    def test_clear_all_empties_table(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        cache.set("MSFT", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        cache.clear_all()
        with sqlite3.connect(cache.db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM price_cache").fetchone()[0]
        assert n == 0


# ============================================================================
# get_stats
# ============================================================================
class TestStats(_SingletonResetMixin):
    def test_stats_shape_and_counts(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"), expiry_days=30)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        cache.set("AAPL", date(2026, 2, 1), date(2026, 2, 4), _make_df(5))
        cache.set("MSFT", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))  # hit
        cache.get("NOPE", date(2026, 1, 1), date(2026, 1, 4))  # miss

        stats = cache.get_stats()
        assert stats["entries"] == 3
        assert stats["total_rows"] == 3 + 5 + 2
        assert stats["unique_tickers"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50.0
        assert stats["expiry_days"] == 30
        assert isinstance(stats["db_size_mb"], float)
        assert stats["db_size_mb"] >= 0

    def test_stats_zero_state(self, tmp_path):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["total_rows"] == 0
        assert stats["unique_tickers"] == 0
        assert stats["hit_rate_pct"] == 0  # both hits and misses are zero

    def test_stats_returns_error_shape_when_db_unreachable(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        original_connect = sqlite3.connect

        def boom(path, *a, **k):
            if path == cache.db_path:
                raise sqlite3.OperationalError("simulated lock")
            return original_connect(path, *a, **k)

        monkeypatch.setattr(sqlite3, "connect", boom)
        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert "error" in stats


# ============================================================================
# Module singleton get_price_cache
# ============================================================================
class TestModuleSingleton(_SingletonResetMixin):
    def test_returns_same_instance_on_repeat_call(self, tmp_path):
        a = get_price_cache(db_path=str(tmp_path / "c.db"), expiry_days=15)
        b = get_price_cache(db_path=str(tmp_path / "DIFFERENT.db"), expiry_days=99)
        # Second call ignores args because the singleton already exists.
        assert a is b
        assert a.expiry_days == 15

    def test_first_call_uses_provided_db_path(self, tmp_path):
        target = tmp_path / "target.db"
        cache = get_price_cache(db_path=str(target), expiry_days=7)
        assert cache.db_path == str(target)
        assert cache.expiry_days == 7
        assert target.exists()

    def test_config_loader_consulted_when_args_omitted(self, tmp_path, monkeypatch):
        # When neither arg is supplied, the body imports config_loader and
        # asks for backtester.disk_cache.expiry_days + .database. Stub both
        # via a fake module so we don't depend on YAML state.
        fake_db = tmp_path / "from_config.db"

        class _FakeConfig:
            def get(self, key, default=None):
                if key == "backtester.disk_cache.expiry_days":
                    return 3
                if key == "backtester.disk_cache.database":
                    return str(fake_db)
                return default

        import sys
        fake_mod = type(sys)("config_loader")
        fake_mod.config = _FakeConfig()
        monkeypatch.setitem(sys.modules, "config_loader", fake_mod)

        cache = get_price_cache()
        assert cache.expiry_days == 3
        assert cache.db_path == str(fake_db)

    def test_falls_back_to_30_when_config_loader_missing(self, tmp_path, monkeypatch):
        # Force ImportError on the config_loader import inside get_price_cache.
        import builtins
        real_import = builtins.__import__

        def _no_config(name, *args, **kwargs):
            if name == "config_loader":
                raise ImportError("simulated missing config_loader")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_config)
        cache = get_price_cache(db_path=str(tmp_path / "c.db"))
        assert cache.expiry_days == 30


# ============================================================================
# Defensive paths — read/write/stats error swallowing
# ============================================================================
class TestDefensivePaths(_SingletonResetMixin):
    def test_get_swallows_malformed_json(self, tmp_path):
        # A corrupted data_json cell should NOT crash the caller.
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(3))
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE price_cache SET data_json = ? WHERE ticker = 'AAPL'",
                ("not-valid-json{{{",),
            )
            conn.commit()
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is None
        # Treated as a miss because the except branch increments _misses.
        assert cache._misses == 1

    def test_set_swallows_write_error(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        # Make sqlite3.connect raise after __init__ completes; the except in
        # set() should swallow without re-raising.
        def boom(path, *a, **k):
            raise sqlite3.OperationalError("disk full")
        monkeypatch.setattr(sqlite3, "connect", boom)
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))

    def test_get_swallows_read_error(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        cache.set("AAPL", date(2026, 1, 1), date(2026, 1, 4), _make_df(2))
        monkeypatch.setattr(sqlite3, "connect",
                            lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        out = cache.get("AAPL", date(2026, 1, 1), date(2026, 1, 4))
        assert out is None

    def test_cleanup_swallows_db_error(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        monkeypatch.setattr(sqlite3, "connect",
                            lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        assert cache.cleanup_expired() == 0

    def test_clear_ticker_swallows_db_error(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        monkeypatch.setattr(sqlite3, "connect",
                            lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        assert cache.clear_ticker("AAPL") == 0

    def test_clear_all_swallows_db_error(self, tmp_path, monkeypatch):
        cache = PriceHistoryCache(db_path=str(tmp_path / "c.db"))
        monkeypatch.setattr(sqlite3, "connect",
                            lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
        cache.clear_all()
