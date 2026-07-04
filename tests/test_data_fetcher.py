"""
Tests for data_fetcher.py — sync FMP/Yahoo fetcher + 3-tier cache.

Triage:
  Tier 1 (high blast radius — silent data poisoning into scoring):
    - load_cache_from_db / save_ticker_to_db_cache (P1 fields, key-format support)
    - fetch_with_cache + freshness primitives (memory cache layer)
    - _fmp_get (rate-limit + 429 backoff + circuit breaker integration)
    - fetch_fmp_earnings_surprise / _earnings_calendar (±200% clamp, beat-streak)
    - fetch_fmp_earnings / _revenue / _key_metrics / _balance_sheet
    - fetch_fmp_analyst_estimates (revision_pct + trend buckets)
    - fetch_fmp_insider_trading (sentiment buckets, $ aggregation)
    - calculate_index_signal / calculate_index_m_score (M-score math)

  Tier 2 (helpers consumed by Tier 1):
    - compute_data_hash, delisted-ticker lifecycle
    - fetch_fmp_profile / _quote / _price_target / _analyst
    - fetch_short_interest (decimal-to-percent normalization)
    - get_cache_hit_stats / get_cache_stats
    - get_cached_market_direction (cache-freshness gate)

  Tier 3 (intentionally NOT covered — see top of file):
    - fetch_finviz_institutional (HTML scraping)
    - fetch_weekly_price_history / fetch_price_from_chart_api (Yahoo wrappers,
      exercised indirectly via fetch_market_direction_data)
    - DataFetcher.get_stock_data (~330-line orchestration method that mocking
      end-to-end would require 15+ patches; it's exercised by scanner integration).

All HTTP is mocked at the requests / yfinance layer — no live network calls.
DB tests use a real in-memory SQLite session (per project rule: no Mock(spec=Session)).
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import time
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import data_fetcher
from backend.database import Base, DelistedTicker, StockDataCache


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session(monkeypatch):
    """Fresh in-memory SQLite, wired into both backend.database.SessionLocal and
    the lazy import inside data_fetcher._get_db_session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    import backend.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    return Session()


@pytest.fixture
def fmp_key(monkeypatch):
    """Set a non-empty FMP_API_KEY so fetchers don't early-return."""
    monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "test-key-1234")


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Reset all module-level caches between tests so freshness/hit counts
    don't leak between tests. autouse so every test gets a clean slate."""
    with data_fetcher._cached_data_lock:
        data_fetcher._cached_data.clear()
    with data_fetcher._freshness_lock:
        data_fetcher._data_freshness_cache.clear()
    with data_fetcher._stats_lock:
        data_fetcher._rate_limit_stats["errors_429"] = 0
        data_fetcher._rate_limit_stats["total_requests"] = 0
    data_fetcher._cache_hit_count = 0
    data_fetcher._cache_miss_count = 0
    data_fetcher._known_delisted_cache.clear()
    data_fetcher._delisted_cache_loaded = False
    data_fetcher._db_cache_loaded = False
    data_fetcher._cached_market_direction = None
    data_fetcher._market_direction_timestamp = None
    yield


def _mock_response(status_code=200, json_data=None, text=""):
    """Build a stub requests.Response-shaped object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else []
    resp.text = text
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Cache layer + freshness
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreshnessPrimitives:
    """Tier 1: is_data_fresh / mark_data_fetched / get_cached_data /
    set_cached_data — load-bearing for the scan-skip logic."""

    def test_is_data_fresh_returns_false_for_unknown_ticker(self):
        assert data_fetcher.is_data_fresh("UNKNOWN", "earnings") is False

    def test_is_data_fresh_returns_false_for_unknown_data_type(self):
        data_fetcher.mark_data_fetched("AAPL", "earnings")
        assert data_fetcher.is_data_fresh("AAPL", "revenue") is False

    def test_is_data_fresh_returns_true_within_interval(self):
        data_fetcher.mark_data_fetched("AAPL", "earnings")
        # earnings interval is 7 days, so just-marked must be fresh.
        assert data_fetcher.is_data_fresh("AAPL", "earnings") is True

    def test_is_data_fresh_returns_false_when_stale(self):
        # Force a stale timestamp (10 days old; interval is 7 days for earnings).
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache["AAPL"] = {
                "earnings": datetime.now() - timedelta(days=10)
            }
        assert data_fetcher.is_data_fresh("AAPL", "earnings") is False

    def test_price_data_type_always_stale(self):
        """price has interval=0, so is_data_fresh must always return False
        (drives real-time price re-fetch every cycle)."""
        data_fetcher.mark_data_fetched("AAPL", "price")
        assert data_fetcher.is_data_fresh("AAPL", "price") is False

    def test_set_get_cached_data_roundtrip(self):
        payload = {"quarterly_eps": [1.0, 0.9, 0.8]}
        data_fetcher.set_cached_data("AAPL", "earnings", payload, persist_to_db=False)
        assert data_fetcher.get_cached_data("AAPL", "earnings") == payload

    def test_get_cached_data_returns_none_when_missing(self):
        assert data_fetcher.get_cached_data("AAPL", "earnings") is None


class TestFetchWithCache:
    """Tier 1: fetch_with_cache — 4-tier hierarchy (Memory → Redis → DB → API)."""

    def test_returns_cached_when_fresh(self):
        cached = {"quarterly_eps": [1.0]}
        data_fetcher.set_cached_data("AAPL", "earnings", cached, persist_to_db=False)
        data_fetcher.mark_data_fetched("AAPL", "earnings")

        sentinel = {"never_called": True}
        called = []

        def fetch_func(*_a, **_k):
            called.append(1)
            return sentinel

        result = data_fetcher.fetch_with_cache("AAPL", "earnings", fetch_func)
        assert result == cached
        assert called == []  # fetch_func must NOT run when cache is fresh

    def test_invokes_fetch_when_stale(self, db_session, monkeypatch):
        """Use db_session fixture so load_cache_from_db hits a fresh in-memory
        DB, not the real data/canslim.db (which would pre-populate the cache
        with production values that mask the test signal)."""
        monkeypatch.setattr(data_fetcher, "REDIS_AVAILABLE", False)

        fresh = {"quarterly_eps": [2.0]}
        called = []

        def fetch_func(*_a, **_k):
            called.append(1)
            return fresh

        # Use a ticker name unlikely to exist in any DB.
        result = data_fetcher.fetch_with_cache("MSFTTEST", "earnings", fetch_func)
        assert result == fresh
        assert called == [1]
        # The cached entry should at minimum contain the fresh value.
        cached = data_fetcher.get_cached_data("MSFTTEST", "earnings")
        assert cached["quarterly_eps"] == [2.0]

    def test_caching_disabled_bypasses_cache(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "CACHING_ENABLED", False)

        # Pre-populate fresh cache; fetch_func MUST still run because caching is off.
        data_fetcher.set_cached_data("AAPL", "earnings", {"old": True}, persist_to_db=False)
        data_fetcher.mark_data_fetched("AAPL", "earnings")

        called = []

        def fetch_func(*_a, **_k):
            called.append(1)
            return {"new": True}

        result = data_fetcher.fetch_with_cache("AAPL", "earnings", fetch_func)
        assert called == [1]
        assert result == {"new": True}

    def test_falsy_fetch_result_not_cached(self, monkeypatch):
        """If the fetcher returns {} / None, we shouldn't pollute the cache —
        next call should retry the fetcher."""
        monkeypatch.setattr(data_fetcher, "REDIS_AVAILABLE", False)

        call_count = []

        def fetch_func(*_a, **_k):
            call_count.append(1)
            return {}  # falsy

        data_fetcher.fetch_with_cache("XYZ", "earnings", fetch_func)
        # Without a cached value, the next call should ALSO hit the fetcher.
        data_fetcher.fetch_with_cache("XYZ", "earnings", fetch_func)
        assert len(call_count) == 2


class TestRateLimitStats:
    """Tier 1: _track_request + get/reset_rate_limit_stats."""

    def test_track_request_counts_total(self):
        data_fetcher._track_request(200)
        data_fetcher._track_request(200)
        data_fetcher._track_request(429)
        stats = data_fetcher.get_rate_limit_stats()
        assert stats["total_requests"] == 3
        assert stats["errors_429"] == 1

    def test_reset_clears_counters(self):
        data_fetcher._track_request(429)
        data_fetcher.reset_rate_limit_stats()
        stats = data_fetcher.get_rate_limit_stats()
        assert stats["total_requests"] == 0
        assert stats["errors_429"] == 0

    def test_get_cache_hit_stats_shape(self):
        stats = data_fetcher.get_cache_hit_stats()
        assert stats == {"hits": 0, "misses": 0}

    def test_get_cache_stats_includes_memory_section(self):
        data_fetcher.set_cached_data("AAPL", "earnings", {"x": 1}, persist_to_db=False)
        data_fetcher.mark_data_fetched("AAPL", "earnings")

        stats = data_fetcher.get_cache_stats()
        assert "memory" in stats
        assert stats["memory"]["tickers_tracked"] >= 1
        assert stats["memory"]["cached_data_entries"] >= 1
        assert "hits" in stats and "misses" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _fmp_get rate-limited wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TestFmpGet:
    """Tier 1: _fmp_get — handles rate limiter, 429 backoff, circuit breaker.
    The rate-limiter integration is what kept us alive during FMP's tightening."""

    def test_returns_200_response_on_success(self, monkeypatch):
        import fmp_rate_limiter

        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", lambda: 0.0)
        monkeypatch.setattr(fmp_rate_limiter, "record_success", lambda *a, **k: None)

        fake_resp = _mock_response(status_code=200, json_data=[{"x": 1}])
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 200
        # Stats must reflect the request.
        assert data_fetcher.get_rate_limit_stats()["total_requests"] == 1

    def test_returns_fake_429_when_circuit_breaker_open(self, monkeypatch):
        """If acquire_sync raises CircuitBreakerOpen, _fmp_get must return a
        response object with status_code=429 (so callers handle gracefully
        without crashing) — not raise."""
        import fmp_rate_limiter

        def _raise(*_a, **_k):
            raise fmp_rate_limiter.CircuitBreakerOpen("circuit open")

        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", _raise)

        # Even if requests.get is patched to explode, we shouldn't reach it.
        def _explode(*_a, **_k):
            raise AssertionError("requests.get must not be called")

        monkeypatch.setattr(data_fetcher.requests, "get", _explode)

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 429

    def test_429_records_metric(self, monkeypatch):
        """A 429 response must call record_429 and bump errors_429 counter."""
        import fmp_rate_limiter

        record_429_called = []

        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", lambda: 0.0)
        monkeypatch.setattr(fmp_rate_limiter, "record_429",
                            lambda: record_429_called.append(1))
        # Force max_retries=1 so we don't actually backoff/sleep.
        monkeypatch.setitem(fmp_rate_limiter._config, "max_retries", 1)

        fake_resp = _mock_response(status_code=429)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 429
        assert data_fetcher.get_rate_limit_stats()["errors_429"] == 1
        assert record_429_called == [1]

    def test_5xx_records_error_metric(self, monkeypatch):
        """5xx responses bump fmp_rate_limiter.record_error (so circuit breaker
        eventually trips on persistent backend failures)."""
        import fmp_rate_limiter

        error_calls = []
        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", lambda: 0.0)
        monkeypatch.setattr(fmp_rate_limiter, "record_error",
                            lambda: error_calls.append(1))

        fake_resp = _mock_response(status_code=503)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 503
        assert error_calls == [1]


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Earnings fetchers (CANSLIM C-score critical path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpEarningsSurprise:
    """Tier 1: fetch_fmp_earnings_surprise — adjusted EPS, beat streak,
    surprise %. Output flows directly into C-score and ML features."""

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_earnings_surprise("AAPL") == {}

    def test_returns_empty_on_non_200(self, fmp_key, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(status_code=500))
        assert data_fetcher.fetch_fmp_earnings_surprise("AAPL") == {}

    def test_returns_empty_on_empty_array(self, fmp_key, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=[]))
        assert data_fetcher.fetch_fmp_earnings_surprise("AAPL") == {}

    def test_calculates_beat_streak_and_surprise(self, fmp_key, monkeypatch):
        # 3 consecutive beats, then a miss.
        records = [
            {"estimatedEarning": 1.00, "actualEarningResult": 1.20},  # +20% beat
            {"estimatedEarning": 1.10, "actualEarningResult": 1.25},  # beat
            {"estimatedEarning": 1.05, "actualEarningResult": 1.10},  # beat
            {"estimatedEarning": 1.00, "actualEarningResult": 0.95},  # MISS — stop
            {"estimatedEarning": 0.90, "actualEarningResult": 1.00},  # would-be beat
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))

        result = data_fetcher.fetch_fmp_earnings_surprise("AAPL")
        assert result["beat_streak"] == 3  # stops at 4th record (miss)
        # Latest surprise = (1.20 - 1.00) / |1.00| * 100 = 20.0
        assert result["latest_surprise_pct"] == pytest.approx(20.0)
        assert len(result["quarterly_adjusted_eps"]) == 5
        assert result["quarterly_adjusted_eps"][0] == pytest.approx(1.20)

    def test_zero_estimate_skips_surprise_calc(self, fmp_key, monkeypatch):
        """An estimated value of 0 must NOT cause a divide-by-zero. Latest
        surprise should be 0 (the safe default)."""
        records = [{"estimatedEarning": 0, "actualEarningResult": 1.0}]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_surprise("AAPL")
        assert result["latest_surprise_pct"] == 0


class TestFetchFmpEarningsRevenueFailureContract:
    """2026-07-04 (swallow-and-serve-stale sweep): fetch_fmp_earnings/revenue
    pre-initialize a non-empty (truthy) result dict, so on TOTAL failure the
    old `return result` passed fetch_with_cache's `if data:` guard → cached []
    as FRESH for 7d AND nulled prior good DB earnings. Must return {} on
    failure so nothing is cached and the next call retries. Sync mirror of the
    async has_financials fix."""

    def test_earnings_returns_empty_dict_on_total_failure(self, fmp_key, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(status_code=429))
        assert data_fetcher.fetch_fmp_earnings("AAPL") == {}

    def test_earnings_returns_empty_dict_on_exception(self, fmp_key, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("network")
        monkeypatch.setattr(data_fetcher, "_fmp_get", _boom)
        assert data_fetcher.fetch_fmp_earnings("AAPL") == {}

    def test_earnings_returns_data_on_success(self, fmp_key, monkeypatch):
        rows = [{"eps": 1.5, "netIncome": 1e9}, {"eps": 1.4, "netIncome": 9e8}]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=rows))
        result = data_fetcher.fetch_fmp_earnings("AAPL")
        assert result != {}
        assert result["quarterly_eps"][:2] == [1.5, 1.4]

    def test_revenue_returns_empty_dict_on_total_failure(self, fmp_key, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(status_code=500))
        assert data_fetcher.fetch_fmp_revenue("AAPL") == {}

    def test_revenue_returns_data_on_success(self, fmp_key, monkeypatch):
        rows = [{"revenue": 5e10}, {"revenue": 4.8e10}]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=rows))
        result = data_fetcher.fetch_fmp_revenue("AAPL")
        assert result != {}
        assert result["quarterly_revenue"][:2] == [5e10, 4.8e10]


class TestGetSp500HistoryFailureRetry:
    """2026-07-04: a total SPY fetch failure must NOT cache an empty frame and
    serve it for the process lifetime (SPY always has data, so empty = failure,
    not no-data). The cache must stay unset so the next call retries."""

    def test_failure_does_not_cache_empty_frame(self, monkeypatch):
        f = data_fetcher.DataFetcher()
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api", lambda t: {})
        class _BoomTicker:
            def history(self, *a, **k):
                raise RuntimeError("yf down")
        monkeypatch.setattr(data_fetcher.yf, "Ticker", lambda t: _BoomTicker())
        out = f.get_sp500_history()
        assert out.empty
        # Critical: the failure was NOT cached — a retry is still possible.
        assert f._sp500_history is None

    def test_empty_yfinance_result_not_cached(self, monkeypatch):
        import pandas as pd
        f = data_fetcher.DataFetcher()
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api", lambda t: {})
        class _EmptyTicker:
            def history(self, *a, **k):
                return pd.DataFrame()
        monkeypatch.setattr(data_fetcher.yf, "Ticker", lambda t: _EmptyTicker())
        f.get_sp500_history()
        assert f._sp500_history is None  # empty result treated as failure


class TestFetchFmpEarningsCalendar:
    """Tier 1: fetch_fmp_earnings_calendar — next earnings date, beat streak,
    latest_surprise_pct (Approach 2 input). The ±200% clamp at line 1078 is
    a load-bearing guardrail against near-zero-estimate explosions."""

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_earnings_calendar("AAPL") == {}

    def test_returns_empty_on_non_200(self, fmp_key, monkeypatch):
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(status_code=500))
        assert data_fetcher.fetch_fmp_earnings_calendar("AAPL") == {}

    def test_filters_to_requested_symbol(self, fmp_key, monkeypatch):
        """Should ignore records for other tickers."""
        records = [
            {"symbol": "OTHER", "date": "2026-06-01", "epsActual": None,
             "epsEstimated": 1.0},
            {"symbol": "MSFT", "date": "2026-04-01", "epsActual": 1.5,
             "epsEstimated": 1.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("MSFT")
        # Only MSFT survives the filter — beat streak 1, no future date.
        assert result.get("earnings_beat_streak") == 1

    def test_clamps_extreme_surprise_to_200_pct(self, fmp_key, monkeypatch):
        """Near-zero estimate (e.g. $0.001) with normal actual produces
        mathematically-valid but behaviorally-meaningless extremes (e.g.
        +69,000%). Line 1078 clamps to ±200% — verify the clamp fires."""
        today = date.today()
        past = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        records = [
            {"symbol": "AAPL", "date": past, "epsActual": 1.0,
             "epsEstimated": 0.001},  # raw = 99,900%; clamped to 200
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        assert result["latest_surprise_pct"] == pytest.approx(200.0)

    def test_clamps_extreme_negative_surprise_to_negative_200(self, fmp_key, monkeypatch):
        today = date.today()
        past = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        records = [
            {"symbol": "AAPL", "date": past, "epsActual": -1.0,
             "epsEstimated": 0.001},  # raw = -100,100%; clamped to -200
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        assert result["latest_surprise_pct"] == pytest.approx(-200.0)

    def test_finds_next_future_earnings(self, fmp_key, monkeypatch):
        today = date.today()
        future = (today + timedelta(days=21)).strftime("%Y-%m-%d")
        past = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        records = [
            {"symbol": "AAPL", "date": future, "epsActual": None,
             "epsEstimated": 1.0},
            {"symbol": "AAPL", "date": past, "epsActual": 1.5,
             "epsEstimated": 1.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        assert result["next_earnings_date"] == future
        assert result["days_to_earnings"] == 21
        assert result["earnings_beat_streak"] == 1

    def test_skips_record_with_bad_date_format(self, fmp_key, monkeypatch):
        """Bad date format must NOT crash — it's logged and skipped."""
        today = date.today()
        past = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        records = [
            {"symbol": "AAPL", "date": "not-a-date", "epsActual": 1.5,
             "epsEstimated": 1.0},
            {"symbol": "AAPL", "date": past, "epsActual": 1.5,
             "epsEstimated": 1.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        # Should not raise and should still process the second record.
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        assert result["earnings_beat_streak"] == 1


class TestFetchFmpEarnings:
    """Tier 1: fetch_fmp_earnings — quarterly + annual EPS + net income."""

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_earnings("AAPL") == {}

    def test_extracts_quarterly_and_annual_eps(self, fmp_key, monkeypatch):
        quarterly = [{"eps": 1.0, "netIncome": 1e9}, {"eps": 0.9, "netIncome": 0.9e9}]
        annual = [{"eps": 4.0, "netIncome": 4e9}]

        # _fmp_get is called twice — return quarterly first, annual second.
        responses = iter([
            _mock_response(json_data=quarterly),
            _mock_response(json_data=annual),
        ])
        monkeypatch.setattr(data_fetcher, "_fmp_get", lambda *a, **k: next(responses))

        result = data_fetcher.fetch_fmp_earnings("AAPL")
        assert result["quarterly_eps"] == [1.0, 0.9]
        assert result["annual_eps"] == [4.0]
        assert result["quarterly_net_income"] == [1e9, 0.9e9]

    def test_none_eps_normalized_to_zero(self, fmp_key, monkeypatch):
        """FMP can return null EPS; the `or 0` guard converts to 0."""
        quarterly = [{"eps": None, "netIncome": None}, {"eps": 1.0, "netIncome": 1e9}]
        responses = iter([
            _mock_response(json_data=quarterly),
            _mock_response(json_data=[]),
        ])
        monkeypatch.setattr(data_fetcher, "_fmp_get", lambda *a, **k: next(responses))
        result = data_fetcher.fetch_fmp_earnings("AAPL")
        assert result["quarterly_eps"] == [0, 1.0]


class TestFetchFmpRevenue:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_revenue("AAPL") == {}

    def test_extracts_quarterly_and_annual_revenue(self, fmp_key, monkeypatch):
        quarterly = [{"revenue": 100e9}, {"revenue": 90e9}]
        annual = [{"revenue": 380e9}]
        responses = iter([
            _mock_response(json_data=quarterly),
            _mock_response(json_data=annual),
        ])
        monkeypatch.setattr(data_fetcher, "_fmp_get", lambda *a, **k: next(responses))
        result = data_fetcher.fetch_fmp_revenue("AAPL")
        assert result["quarterly_revenue"] == [100e9, 90e9]
        assert result["annual_revenue"] == [380e9]


class TestFetchFmpKeyMetrics:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_key_metrics("AAPL") == {}

    def test_extracts_roe_and_other_metrics(self, fmp_key, monkeypatch):
        data = [{
            "returnOnEquity": 0.28,
            "returnOnAssets": 0.20,
            "returnOnInvestedCapital": 0.25,
            "currentRatio": 1.5,
            "earningsYield": 0.04,
            "freeCashFlowYield": 0.05,
        }]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_key_metrics("AAPL")
        assert result["roe"] == pytest.approx(0.28)
        assert result["roic"] == pytest.approx(0.25)

    def test_none_values_normalized_to_zero(self, fmp_key, monkeypatch):
        """All `metrics.get('x', 0) or 0` paths must convert None→0."""
        data = [{"returnOnEquity": None}]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_key_metrics("AAPL")
        assert result["roe"] == 0


class TestFetchFmpBalanceSheet:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_balance_sheet("AAPL") == {}

    def test_extracts_cash_debt_assets(self, fmp_key, monkeypatch):
        data = [{
            "cashAndCashEquivalents": 50e9,
            "totalDebt": 100e9,
            "totalAssets": 350e9,
            "totalLiabilities": 250e9,
        }]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_balance_sheet("AAPL")
        assert result["cash_and_equivalents"] == 50e9
        assert result["total_debt"] == 100e9


class TestFetchFmpAnalystEstimates:
    """Tier 1: P1 feature — drives eps_estimate_revision_pct in scoring."""

    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_analyst_estimates("AAPL") == {}

    def test_returns_empty_with_too_few_records(self, fmp_key, monkeypatch):
        """Need at least 2 records (current + prior year)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=[{"date": "2026-12-31"}]))
        assert data_fetcher.fetch_fmp_analyst_estimates("AAPL") == {}

    def test_calculates_revision_pct_and_up_trend(self, fmp_key, monkeypatch):
        current_year = date.today().year
        records = [
            {"date": f"{current_year}-12-31", "epsAvg": 11.0,
             "numberAnalystsEstimatedEps": 25},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 10.0,
             "numberAnalystsEstimatedEps": 22},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_analyst_estimates("AAPL")
        # (11 - 10) / 10 * 100 = 10.0 → trend "up" (>= 5)
        assert result["eps_estimate_revision_pct"] == pytest.approx(10.0)
        assert result["estimate_revision_trend"] == "up"
        assert result["num_analysts"] == 25

    def test_down_trend_when_revision_negative(self, fmp_key, monkeypatch):
        current_year = date.today().year
        records = [
            {"date": f"{current_year}-12-31", "epsAvg": 9.0},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 10.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_analyst_estimates("AAPL")
        # (9 - 10) / 10 * 100 = -10.0 → trend "down" (<= -5)
        assert result["estimate_revision_trend"] == "down"

    def test_stable_trend_within_5_pct(self, fmp_key, monkeypatch):
        current_year = date.today().year
        records = [
            {"date": f"{current_year}-12-31", "epsAvg": 10.2},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 10.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_analyst_estimates("AAPL")
        # 2.0% — within ±5% — stable
        assert result["estimate_revision_trend"] == "stable"


class TestFetchFmpInsiderTrading:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_insider_trading("AAPL") == {}

    def test_aggregates_buys_and_sells_with_sentiment(self, fmp_key, monkeypatch):
        recent = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        records = [
            {"transactionDate": recent, "transactionType": "P-Purchase",
             "securitiesTransacted": 1000, "price": 200,
             "typeOfOwner": "officer", "reportingName": "Smith CEO"},
            {"transactionDate": recent, "transactionType": "S-Sale",
             "securitiesTransacted": 500, "price": 150,
             "typeOfOwner": "officer", "reportingName": "Doe"},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["buy_count"] == 1
        assert result["sell_count"] == 1
        assert result["buy_value"] == 200_000  # 1000 * 200
        assert result["sell_value"] == 75_000  # 500 * 150
        # net_value = 200k - 75k = 125k → ≥100k → bullish
        assert result["sentiment"] == "bullish"
        # Largest buy was by "Smith CEO" → title detected as CEO
        assert result["largest_buyer_title"] == "CEO"

    def test_filters_old_trades_outside_3_month_window(self, fmp_key, monkeypatch):
        old = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        records = [
            {"transactionDate": old, "transactionType": "P-Purchase",
             "securitiesTransacted": 1000, "price": 100,
             "typeOfOwner": "officer", "reportingName": "Smith"},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # All trades >90 days old → filtered out → empty result.
        assert result.get("buy_count", 0) == 0

    def test_bearish_sentiment_when_net_value_below_negative_100k(self, fmp_key, monkeypatch):
        recent = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        records = [
            {"transactionDate": recent, "transactionType": "S-Sale",
             "securitiesTransacted": 5000, "price": 100,
             "typeOfOwner": "officer", "reportingName": "Doe"},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # net_value = -500_000 → bearish
        assert result["sentiment"] == "bearish"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — DB cache load + save (high blast radius — silent data poisoning)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveTickerToDbCache:
    """Tier 1: save_ticker_to_db_cache — every persistence path. The
    `num_analyst_opinions` latent bug (memory: d28f8de) was a similar
    column-name mismatch in this code path."""

    def test_creates_record_for_new_ticker_earnings(self, db_session):
        data_fetcher.save_ticker_to_db_cache("NEW", "earnings", {
            "quarterly_eps": [1.0, 0.9],
            "annual_eps": [4.0],
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="NEW").one()
        assert rec.quarterly_earnings == [1.0, 0.9]
        assert rec.annual_earnings == [4.0]
        assert rec.earnings_updated_at is not None

    def test_supports_legacy_quarterly_key_format(self, db_session):
        """Old format used "quarterly"/"annual" keys; new uses "quarterly_eps".
        Both must work — the `or` chain in save_ticker_to_db_cache provides
        backward compat (line 251)."""
        data_fetcher.save_ticker_to_db_cache("LEGACY", "earnings", {
            "quarterly": [2.0, 1.9],
            "annual": [8.0],
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="LEGACY").one()
        assert rec.quarterly_earnings == [2.0, 1.9]
        assert rec.annual_earnings == [8.0]

    def test_revenue_persistence(self, db_session):
        data_fetcher.save_ticker_to_db_cache("REV", "revenue", {
            "quarterly_revenue": [100e9, 90e9],
            "annual_revenue": [380e9],
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="REV").one()
        assert rec.quarterly_revenue == [100e9, 90e9]

    def test_balance_sheet_persistence(self, db_session):
        data_fetcher.save_ticker_to_db_cache("BS", "balance_sheet", {
            "total_cash": 50e9,
            "total_debt": 100e9,
            "shares_outstanding": 16e9,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="BS").one()
        assert rec.total_cash == 50e9
        assert rec.total_debt == 100e9

    def test_analyst_target_skipped_when_falsy(self, db_session):
        """Defensive: don't overwrite a real target_price with 0/None
        (line 269 — guards against API-flake clobbering valid cached data)."""
        # Pre-seed a real value.
        rec = StockDataCache(ticker="ANL", analyst_target_price=200.0,
                             analyst_count=12)
        db_session.add(rec)
        db_session.commit()

        # Save with falsy target_price — must NOT overwrite the 200.0.
        data_fetcher.save_ticker_to_db_cache("ANL", "analyst", {
            "target_price": 0,
            "count": 15,
        })
        db_session.expire_all()
        rec_after = db_session.query(StockDataCache).filter_by(ticker="ANL").one()
        assert rec_after.analyst_target_price == 200.0  # Preserved
        assert rec_after.analyst_count == 15  # `count` updated (different guard)

    def test_key_metrics_preserves_zero_roe(self, db_session):
        """Critical edge case: ROE of exactly 0 is valid data (not "missing").
        Line 283 uses `is not None` (NOT `or`) specifically to preserve 0.0."""
        data_fetcher.save_ticker_to_db_cache("METRICS", "key_metrics", {
            "roe": 0.0,  # Valid value!
            "trailing_pe": 25.0,
            "forward_pe": None,  # truly missing
            "peg_ratio": 1.5,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="METRICS").one()
        assert rec.roe == 0.0  # Preserved despite being falsy
        assert rec.trailing_pe == 25.0
        assert rec.forward_pe is None  # None correctly skipped

    def test_earnings_calendar_parses_string_date(self, db_session):
        """next_earnings_date arrives as a string from the FMP fetcher; must
        be parsed into a Python date object before storing."""
        data_fetcher.save_ticker_to_db_cache("CAL", "earnings_calendar", {
            "next_earnings_date": "2026-07-15",
            "days_to_earnings": 60,
            "earnings_beat_streak": 4,
            "latest_surprise_pct": 12.5,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="CAL").one()
        assert rec.next_earnings_date == date(2026, 7, 15)
        assert rec.earnings_beat_streak == 4
        assert rec.latest_surprise_pct == pytest.approx(12.5)

    def test_earnings_calendar_handles_bad_date(self, db_session):
        """Bad date string falls through to None — must NOT raise."""
        data_fetcher.save_ticker_to_db_cache("BAD", "earnings_calendar", {
            "next_earnings_date": "not-a-date",
            "days_to_earnings": None,
            "earnings_beat_streak": 0,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="BAD").one()
        assert rec.next_earnings_date is None

    def test_analyst_estimates_persistence(self, db_session):
        data_fetcher.save_ticker_to_db_cache("EST", "analyst_estimates", {
            "eps_estimate_current": 11.0,
            "eps_estimate_prior": 10.0,
            "eps_estimate_revision_pct": 10.0,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="EST").one()
        assert rec.eps_estimate_current == 11.0
        assert rec.eps_estimate_revision_pct == 10.0

    def test_short_interest_persistence(self, db_session):
        data_fetcher.save_ticker_to_db_cache("SHORT", "short_interest", {
            "short_interest_pct": 25.5,
            "short_ratio": 5.5,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="SHORT").one()
        assert rec.short_interest_pct == 25.5

    def test_yahoo_info_writes_num_analyst_opinions_to_analyst_count(self, db_session):
        """Regression guard: the field is named num_analyst_opinions in the
        yahoo_info dict but stored as analyst_count in the DB (line 303).
        This was the latent bug fixed in d28f8de."""
        data_fetcher.save_ticker_to_db_cache("YHIN", "yahoo_info", {
            "analyst_target_price": 250.0,
            "num_analyst_opinions": 42,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="YHIN").one()
        assert rec.analyst_target_price == 250.0
        assert rec.analyst_count == 42  # NOT num_analyst_opinions


class TestLoadCacheFromDb:
    """Tier 1: load_cache_from_db — populates in-memory cache from DB at
    scan startup. Bugs here mean the scan refetches everything from FMP
    (rate-limit pain) or worse, scores against stale data."""

    def test_loads_earnings_into_memory_cache(self, db_session):
        rec = StockDataCache(
            ticker="LOAD",
            quarterly_earnings=[1.0, 0.9],
            annual_earnings=[4.0],
            earnings_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.load_cache_from_db()
        assert data_fetcher._db_cache_loaded is True

        cached = data_fetcher.get_cached_data("LOAD", "earnings")
        assert cached is not None
        assert cached["quarterly_eps"] == [1.0, 0.9]
        assert cached["annual_eps"] == [4.0]
        assert data_fetcher.is_data_fresh("LOAD", "earnings") is True

    def test_loads_p1_earnings_calendar_with_latest_surprise(self, db_session):
        """latest_surprise_pct (added May 2026) must round-trip through
        load_cache_from_db — Approach 2 reads it for the surprise gate."""
        rec = StockDataCache(
            ticker="P1",
            next_earnings_date=date(2026, 7, 15),
            days_to_earnings=60,
            earnings_beat_streak=4,
            latest_surprise_pct=18.5,
            earnings_calendar_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("P1", "earnings_calendar")
        assert cached is not None
        assert cached["earnings_beat_streak"] == 4
        assert cached["latest_surprise_pct"] == pytest.approx(18.5)
        assert cached["days_to_earnings"] == 60

    def test_idempotent_after_first_load(self, db_session):
        """Second call must return immediately (line 95-96) — guarded by
        _db_cache_loaded flag."""
        data_fetcher.load_cache_from_db()
        assert data_fetcher._db_cache_loaded is True

        # Add a new record AFTER first load — second call must NOT pick it up.
        rec = StockDataCache(
            ticker="LATE",
            quarterly_earnings=[1.0],
            earnings_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.load_cache_from_db()
        # LATE was added after first load + flag is sticky → not loaded.
        assert data_fetcher.get_cached_data("LATE", "earnings") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Index M-score math (pure functions, easy + valuable)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateIndexSignal:
    """Tier 1: pure-math signal classifier. Drives market-direction display."""

    def test_returns_neutral_for_zero_data(self):
        assert data_fetcher.calculate_index_signal(0, 100, 100) == 0
        assert data_fetcher.calculate_index_signal(100, 100, 0) == 0

    def test_bullish_above_both_mas(self):
        assert data_fetcher.calculate_index_signal(110, 100, 90) == 2

    def test_cautious_above_200_below_50(self):
        assert data_fetcher.calculate_index_signal(95, 100, 90) == 1

    def test_neutral_below_200_above_50(self):
        assert data_fetcher.calculate_index_signal(95, 90, 100) == 0

    def test_bearish_below_both(self):
        assert data_fetcher.calculate_index_signal(80, 90, 100) == -1


class TestCalculateIndexMScore:
    """Tier 1: continuous M-score in [0, 1]. Used as M-component weight."""

    def test_returns_neutral_05_for_no_data(self):
        assert data_fetcher.calculate_index_m_score(0, 100, 100) == 0.5

    def test_at_both_mas_yields_05(self):
        # price == 50MA == 200MA → all components at midpoint.
        assert data_fetcher.calculate_index_m_score(100, 100, 100) == pytest.approx(0.5)

    def test_well_above_both_mas_approaches_1(self):
        # Price 30% above 50MA AND 30% above 200MA → near-max score.
        score = data_fetcher.calculate_index_m_score(130, 100, 100)
        assert 0.85 < score <= 1.0

    def test_well_below_both_mas_approaches_0(self):
        score = data_fetcher.calculate_index_m_score(70, 100, 100)
        assert 0.0 <= score < 0.15

    def test_score_is_clamped_to_unit_interval(self):
        # Extreme inputs must clamp to [0, 1].
        assert 0.0 <= data_fetcher.calculate_index_m_score(1000, 100, 100) <= 1.0
        assert 0.0 <= data_fetcher.calculate_index_m_score(1, 100, 100) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Helpers consumed by Tier 1
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDataHash:
    """Tier 2: delta-detection hash for cache invalidation."""

    def test_returns_consistent_16_char_hex(self):
        data_fetcher.set_cached_data("HASH", "earnings",
                                     {"quarterly": [1.0, 0.9]}, persist_to_db=False)
        h1 = data_fetcher.compute_data_hash("HASH")
        h2 = data_fetcher.compute_data_hash("HASH")
        assert h1 == h2
        assert len(h1) == 16

    def test_different_data_produces_different_hash(self):
        data_fetcher.set_cached_data("A", "earnings",
                                     {"quarterly": [1.0]}, persist_to_db=False)
        data_fetcher.set_cached_data("B", "earnings",
                                     {"quarterly": [2.0]}, persist_to_db=False)
        assert data_fetcher.compute_data_hash("A") != data_fetcher.compute_data_hash("B")


class TestDelistedTickerLifecycle:
    """Tier 2: mark_ticker_as_delisted / clear / get / refresh_delisted_cache."""

    def test_get_delisted_tickers_returns_empty_when_none(self, db_session):
        assert data_fetcher.get_delisted_tickers() == set()

    def test_get_delisted_only_returns_3_plus_failures_in_window(self, db_session):
        future = datetime.now() + timedelta(days=10)
        past = datetime.now() - timedelta(days=10)
        # 3+ failures, recheck in future → excluded
        db_session.add(DelistedTicker(ticker="DEAD", failure_count=5,
                                      recheck_after=future))
        # 3+ failures, recheck in past → re-checkable, NOT excluded
        db_session.add(DelistedTicker(ticker="MAYBE", failure_count=5,
                                      recheck_after=past))
        # < 3 failures → never excluded
        db_session.add(DelistedTicker(ticker="FLAKY", failure_count=2,
                                      recheck_after=future))
        db_session.commit()

        excluded = data_fetcher.get_delisted_tickers()
        assert "DEAD" in excluded
        assert "MAYBE" not in excluded
        assert "FLAKY" not in excluded

    def test_refresh_delisted_cache_populates_in_memory_set(self, db_session):
        db_session.add(DelistedTicker(ticker="GONE", failure_count=1))
        db_session.commit()

        data_fetcher.refresh_delisted_cache()
        assert "GONE" in data_fetcher._known_delisted_cache
        assert data_fetcher._delisted_cache_loaded is True

    def test_clear_short_circuits_when_not_in_cache(self, db_session, monkeypatch):
        """Fast path: if ticker isn't in the in-memory set, skip the DB."""
        # Mark cache as loaded but empty.
        data_fetcher._delisted_cache_loaded = True
        data_fetcher._known_delisted_cache = set()  # NEVER_DELISTED isn't here

        # Patch _get_db_session to detect any DB activity.
        called = []
        original = data_fetcher._get_db_session

        def _watcher():
            called.append(1)
            return original()

        monkeypatch.setattr(data_fetcher, "_get_db_session", _watcher)
        data_fetcher.clear_delisted_ticker("NEVER_DELISTED")
        assert called == []  # Short-circuited — no DB query.

    def test_clear_actually_deletes_when_present(self, db_session):
        db_session.add(DelistedTicker(ticker="REVIVED", failure_count=5))
        db_session.commit()
        data_fetcher.refresh_delisted_cache()  # populate cache
        assert "REVIVED" in data_fetcher._known_delisted_cache

        data_fetcher.clear_delisted_ticker("REVIVED")

        db_session.expire_all()
        assert db_session.query(DelistedTicker).filter_by(ticker="REVIVED").first() is None
        assert "REVIVED" not in data_fetcher._known_delisted_cache


class TestFmpConfirmsDelisted:
    """Tier 2: _fmp_confirms_delisted — verification gate before 30-day exclusion."""

    def test_no_api_key_assumes_delisted(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "")
        # Re-read inside the function via os.environ — override there too.
        monkeypatch.setattr(data_fetcher.os.environ, "get",
                            lambda k, default="": "")
        assert data_fetcher._fmp_confirms_delisted("DEAD") is True

    def test_returns_false_when_fmp_has_data(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        fake = _mock_response(status_code=200, json_data=[{"symbol": "ALIVE"}])
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake)
        assert data_fetcher._fmp_confirms_delisted("ALIVE") is False

    def test_returns_true_when_fmp_returns_empty(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        fake = _mock_response(status_code=200, json_data=[])
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake)
        assert data_fetcher._fmp_confirms_delisted("DEAD") is True

    def test_returns_true_when_profile_not_actively_trading(self, monkeypatch):
        # FMP retains profiles for acquired/delisted names (PXD/SGEN/etc.);
        # isActivelyTrading=False is the authoritative delisted signal.
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        fake = _mock_response(
            status_code=200,
            json_data=[{"symbol": "PXD", "isActivelyTrading": False}],
        )
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake)
        assert data_fetcher._fmp_confirms_delisted("PXD") is True

    def test_returns_false_when_profile_actively_trading(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        fake = _mock_response(
            status_code=200,
            json_data=[{"symbol": "AAPL", "isActivelyTrading": True}],
        )
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake)
        assert data_fetcher._fmp_confirms_delisted("AAPL") is False

    def test_missing_actively_trading_flag_fails_safe(self, monkeypatch):
        # No flag at all -> ambiguous -> must NOT confirm delisting (fail safe),
        # so a transient Yahoo outage can't wrongly exclude a live stock.
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        fake = _mock_response(status_code=200, json_data=[{"symbol": "MSFT"}])
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake)
        assert data_fetcher._fmp_confirms_delisted("MSFT") is False


class TestFetchFmpProfile:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_profile("AAPL") == {}

    def test_extracts_company_metadata(self, fmp_key, monkeypatch):
        data = [{
            "companyName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "mktCap": 3e12,
            "price": 200.0,
            "range": "150.00-250.00",
            "sharesOutstanding": 16e9,
        }]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_profile("AAPL")
        assert result["name"] == "Apple Inc."
        assert result["sector"] == "Technology"
        assert result["market_cap"] == 3e12
        assert result["high_52w"] == "250.00"  # last segment of range


class TestFetchFmpQuote:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_quote("AAPL") == {}

    def test_extracts_quote_fields(self, fmp_key, monkeypatch):
        data = [{
            "price": 200.0,
            "yearHigh": 250.0,
            "yearLow": 150.0,
            "volume": 50_000_000,
            "avgVolume": 60_000_000,
            "marketCap": 3e12,
            "pe": 28.5,
            "sharesOutstanding": 16e9,
        }]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_quote("AAPL")
        assert result["current_price"] == 200.0
        assert result["pe"] == 28.5


class TestFetchFmpPriceTarget:
    def test_returns_empty_without_api_key(self, monkeypatch):
        monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "")
        assert data_fetcher.fetch_fmp_price_target("AAPL") == {}

    def test_extracts_consensus_targets(self, fmp_key, monkeypatch):
        data = [{
            "targetHigh": 300.0,
            "targetLow": 180.0,
            "targetConsensus": 240.0,
            "targetMedian": 235.0,
        }]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_price_target("AAPL")
        assert result["target_consensus"] == 240.0
        assert result["target_high"] == 300.0


class TestSyncAnalystEstimatesShapeParity:
    """2026-07-03 audit: fetch_fmp_analyst (v3-era field names, deleted) was
    cached under the shared 'analyst_estimates' key — wrong shape suppressed
    P1 estimate revisions for 7 days and its DB persist NULLed the
    eps_estimate_* columns. The sync path must cache the P1 shape."""

    def test_sync_path_uses_p1_shaped_fetcher(self):
        import inspect
        src = inspect.getsource(data_fetcher.DataFetcher.get_stock_data)
        assert 'fetch_with_cache(ticker, "analyst_estimates", fetch_fmp_analyst_estimates' in src
        assert "fetch_fmp_analyst," not in src  # deleted fetcher must not return

    def test_fetch_fmp_analyst_is_gone(self):
        # The misfielded v3-era fetcher must not silently come back.
        assert not hasattr(data_fetcher, "fetch_fmp_analyst")


class TestFetchShortInterest:
    """Tier 2: fetch_short_interest — Yahoo decimal/percent normalization
    is the interesting edge case (line 1368)."""

    def test_decimal_short_pct_converted_to_percent(self, monkeypatch):
        """Yahoo returns 0.15 (= 15%), function should convert to 15.0."""
        fake_ticker = MagicMock()
        fake_ticker.info = {"shortPercentOfFloat": 0.15, "shortRatio": 5.0}
        monkeypatch.setattr(data_fetcher.yf, "Ticker", lambda t: fake_ticker)

        result = data_fetcher.fetch_short_interest("AAPL")
        assert result["short_interest_pct"] == pytest.approx(15.0)
        assert result["short_ratio"] == pytest.approx(5.0)

    def test_already_percent_value_not_doubled(self, monkeypatch):
        """If shortPercentOfFloat is already 15.0 (not decimal), don't multiply
        by 100 again. Guard at line 1368: only convert if 0 < x < 3.0."""
        fake_ticker = MagicMock()
        fake_ticker.info = {"shortPercentOfFloat": 15.0, "shortRatio": 5.0}
        monkeypatch.setattr(data_fetcher.yf, "Ticker", lambda t: fake_ticker)

        result = data_fetcher.fetch_short_interest("AAPL")
        assert result["short_interest_pct"] == pytest.approx(15.0)

    def test_returns_empty_on_exception(self, monkeypatch):
        def _explode(_):
            raise Exception("network down")

        monkeypatch.setattr(data_fetcher.yf, "Ticker", _explode)
        result = data_fetcher.fetch_short_interest("AAPL")
        assert result == {}


class TestGetCachedMarketDirection:
    """Tier 2: 4-hour cache wrapper around fetch_market_direction_data."""

    def test_force_refresh_calls_underlying_fetch(self, monkeypatch):
        called = []

        def fake_fetch():
            called.append(1)
            return {"success": True, "weighted_signal": 1.5, "market_score": 12.0}

        monkeypatch.setattr(data_fetcher, "fetch_market_direction_data", fake_fetch)

        # First call: must fetch.
        r1 = data_fetcher.get_cached_market_direction()
        assert called == [1]
        assert r1["weighted_signal"] == 1.5

        # Second call without force: must use cache.
        data_fetcher.get_cached_market_direction()
        assert called == [1]  # still only one fetch

        # Force refresh: must fetch again.
        data_fetcher.get_cached_market_direction(force_refresh=True)
        assert called == [1, 1]

    def test_cache_invalidates_when_stale(self, monkeypatch):
        called = []

        def fake_fetch():
            called.append(1)
            return {"success": True, "market_score": 7.5}

        monkeypatch.setattr(data_fetcher, "fetch_market_direction_data", fake_fetch)

        data_fetcher.get_cached_market_direction()
        assert called == [1]

        # Force timestamp to be 5 hours ago (interval is 4 hours).
        data_fetcher._market_direction_timestamp = (
            datetime.now() - timedelta(hours=5)
        )

        data_fetcher.get_cached_market_direction()
        assert called == [1, 1]  # Re-fetched due to stale cache


class TestStockDataInit:
    """Tier 2: StockData container — drives downstream type expectations."""

    def test_initializes_with_empty_defaults(self):
        s = data_fetcher.StockData("AAPL")
        assert s.ticker == "AAPL"
        assert s.is_valid is False  # Becomes True only after successful fetch
        assert s.quarterly_earnings == []
        assert s.weekly_price_history == []
        assert s.eps_beat_streak == 0
        # P1 fields default sanely for downstream scoring math.
        assert s.eps_estimate_revision_pct == 0.0
        assert s.days_to_earnings == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — mark_ticker_as_delisted (multi-branch lifecycle)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarkTickerAsDelisted:
    """Tier 2: failure-count increment, FMP confirmation gate, hourly dedup."""

    def test_creates_new_record_with_failure_count_1(self, db_session):
        data_fetcher.mark_ticker_as_delisted("NEW", reason="404", source="screener")
        rec = db_session.query(DelistedTicker).filter_by(ticker="NEW").one()
        assert rec.failure_count == 1
        assert rec.reason == "404"
        # In-memory cache should be updated too.
        assert "NEW" in data_fetcher._known_delisted_cache

    def test_increments_failure_count_after_hour_window(self, db_session):
        # Pre-seed with last_failed_at > 1 hour ago.
        rec = DelistedTicker(
            ticker="REPEAT",
            failure_count=1,
            last_failed_at=datetime.now() - timedelta(hours=2),
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.mark_ticker_as_delisted("REPEAT", reason="500")

        db_session.expire_all()
        rec_after = db_session.query(DelistedTicker).filter_by(ticker="REPEAT").one()
        assert rec_after.failure_count == 2
        assert rec_after.reason == "500"

    def test_skips_increment_within_hour_window(self, db_session):
        """Hourly dedup: a single scan cycle hitting the ticker via multiple
        code paths must not push it over the 3-strike threshold prematurely.

        Seed `last_failed_at` in the same TZ frame the runtime code uses
        (naive UTC via data_fetcher._db_now) so the 30-minute window holds
        on hosts whose clock isn't UTC."""
        rec = DelistedTicker(
            ticker="DEDUP",
            failure_count=1,
            last_failed_at=data_fetcher._db_now() - timedelta(minutes=30),  # < 1hr
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.mark_ticker_as_delisted("DEDUP", reason="404")

        db_session.expire_all()
        rec_after = db_session.query(DelistedTicker).filter_by(ticker="DEDUP").one()
        assert rec_after.failure_count == 1  # Did NOT increment

    def test_third_failure_with_fmp_confirmation_sets_30_day_exclusion(self, db_session, monkeypatch):
        """3rd failure + FMP confirms delisted → 30-day recheck_after."""
        rec = DelistedTicker(
            ticker="DEAD",
            failure_count=2,
            last_failed_at=datetime.now() - timedelta(hours=2),
        )
        db_session.add(rec)
        db_session.commit()

        # FMP confirms it's truly delisted (returns True).
        monkeypatch.setattr(data_fetcher, "_fmp_confirms_delisted", lambda t: True)
        before = datetime.now()
        data_fetcher.mark_ticker_as_delisted("DEAD")

        db_session.expire_all()
        rec_after = db_session.query(DelistedTicker).filter_by(ticker="DEAD").one()
        assert rec_after.failure_count == 3
        # recheck_after should be ~30 days from now.
        assert rec_after.recheck_after >= before + timedelta(days=29)

    def test_third_failure_with_fmp_disagreeing_resets_to_zero(self, db_session, monkeypatch):
        """3rd Yahoo failure but FMP still has the ticker → reset failure_count
        to 0 and short-circuit (line 645). Prevents transient Yahoo flakes from
        excluding valid stocks."""
        rec = DelistedTicker(
            ticker="FLAKY",
            failure_count=2,
            last_failed_at=datetime.now() - timedelta(hours=2),
        )
        db_session.add(rec)
        db_session.commit()

        monkeypatch.setattr(data_fetcher, "_fmp_confirms_delisted", lambda t: False)
        data_fetcher.mark_ticker_as_delisted("FLAKY")

        db_session.expire_all()
        rec_after = db_session.query(DelistedTicker).filter_by(ticker="FLAKY").one()
        assert rec_after.failure_count == 0  # Reset!


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_market_direction_data (drives M-score for SPY/QQQ/DIA)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_chart_data(close_prices, volumes=None):
    """Helper: build a fake fetch_price_from_chart_api response."""
    n = len(close_prices)
    return {
        "current_price": close_prices[-1],
        "high_52w": max(close_prices),
        "low_52w": min(close_prices),
        "market_cap": 1e12,
        "name": "Index",
        "close_prices": close_prices,
        "volumes": volumes if volumes is not None else [1_000_000] * n,
        "timestamps": list(range(n)),
    }


class TestFetchMarketDirectionData:
    """Tier 1: aggregates SPY+QQQ+DIA chart data into composite weighted M-score
    + signal. This runs every 4 hours and feeds into the M-score for every scan."""

    def test_full_data_returns_bullish_with_high_m_score(self, monkeypatch):
        # Build prices well above both 50MA and 200MA.
        # 250 prices, climbing — last 50 ~ 200, last 200 ~ 180, current 230.
        prices = ([100] * 100 + [200] * 150)  # 200MA ≈ 160, 50MA = 200, current = 200
        prices[-1] = 230  # Push current well above
        chart = _build_chart_data(prices)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)

        result = data_fetcher.fetch_market_direction_data()
        assert result["success"] is True
        assert "indexes" in result and len(result["indexes"]) == 3
        # All three indexes get the same chart → all signals == 2 (bullish).
        for ticker in ["SPY", "QQQ", "DIA"]:
            assert result["indexes"][ticker]["signal"] == 2
            assert result["indexes"][ticker]["status"] == "ok"
        assert result["weighted_signal"] == pytest.approx(2.0)
        assert result["market_score"] >= 12.0  # high M score
        assert result["market_trend"] in ("bullish", "cautious")

    def test_partial_data_50_to_200_uses_partial_status(self, monkeypatch):
        """50 ≤ len < 200 → status='partial', uses available data."""
        prices = [100 + i * 0.1 for i in range(100)]  # 100 days
        chart = _build_chart_data(prices)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)

        result = data_fetcher.fetch_market_direction_data()
        assert result["success"] is True
        for ticker in ["SPY", "QQQ", "DIA"]:
            assert result["indexes"][ticker]["status"] == "partial"

    def test_insufficient_data_marked_status(self, monkeypatch):
        """< 50 closes → insufficient_data and excluded from weighted average."""
        prices = [100 + i * 0.1 for i in range(40)]
        chart = _build_chart_data(prices)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)

        result = data_fetcher.fetch_market_direction_data()
        # All 3 indexes have insufficient data → no weighted average possible.
        assert result["success"] is False
        assert result["error"] is not None

    def test_chart_fetch_failure_marked(self, monkeypatch):
        """Empty chart_data → status='fetch_failed', excluded from weights."""
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda t: {})
        result = data_fetcher.fetch_market_direction_data()
        for ticker in ["SPY", "QQQ", "DIA"]:
            assert result["indexes"][ticker]["status"] == "fetch_failed"
        assert result["success"] is False

    def test_severe_bear_trend_when_well_below_mas(self, monkeypatch):
        """Long downtrend → composite_m below 0.20 → 'severe_bear'."""
        # Prices drop from 200 → 80, current well below both MAs.
        prices = [200 - i * 0.5 for i in range(250)]
        chart = _build_chart_data(prices)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)

        result = data_fetcher.fetch_market_direction_data()
        assert result["success"] is True
        assert result["market_trend"] in ("bearish", "severe_bear")


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — fetch_fmp_institutional (FMP-then-Finviz fallback)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpInstitutional:
    """Tier 2: FMP institutional path. The Finviz fallback is Tier 3 (HTML
    scraping) but we can still cover the FMP success path."""

    def test_returns_total_institutional_shares(self, fmp_key, monkeypatch):
        # FMP returns top 50 holders with shares each.
        data = [{"shares": 1000} for _ in range(10)]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=data))
        result = data_fetcher.fetch_fmp_institutional("AAPL")
        assert result == 10_000  # 10 holders * 1000 shares

    def test_falls_back_to_finviz_when_fmp_empty(self, fmp_key, monkeypatch):
        """FMP returns empty list → falls through to fetch_finviz_institutional."""
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=[]))
        # Stub Finviz to a sentinel value so we know the fallback ran.
        monkeypatch.setattr(data_fetcher, "fetch_finviz_institutional",
                            lambda t: 42.0)
        result = data_fetcher.fetch_fmp_institutional("AAPL")
        assert result == 42.0


# ═══════════════════════════════════════════════════════════════════════════════
# Additional cache-loader coverage (institutional + analyst + key_metrics paths)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchPriceFromChartApi:
    """Tier 1-misclassified-as-Tier-3: this is the Yahoo fallback path used
    by fetch_market_direction_data and DataFetcher.get_stock_data when FMP
    is rate-limited. Worth direct coverage."""

    def test_extracts_price_meta_and_arrays_from_chart_response(self, monkeypatch):
        chart_payload = {
            "chart": {
                "result": [{
                    "meta": {
                        "regularMarketPrice": 200.0,
                        "fiftyTwoWeekHigh": 250.0,
                        "fiftyTwoWeekLow": 150.0,
                        "marketCap": 3e12,
                        "longName": "Apple Inc.",
                    },
                    "indicators": {
                        "quote": [{
                            "close": [195.0, 198.0, 200.0],
                            "volume": [50_000_000, 55_000_000, 60_000_000],
                        }]
                    },
                    "timestamp": [1700000000, 1700086400, 1700172800],
                }]
            }
        }
        fake_resp = _mock_response(json_data=chart_payload)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        result = data_fetcher.fetch_price_from_chart_api("AAPL")
        assert result["current_price"] == 200.0
        assert result["high_52w"] == 250.0
        assert result["low_52w"] == 150.0
        assert result["market_cap"] == 3e12
        assert result["name"] == "Apple Inc."
        assert len(result["close_prices"]) == 3
        assert len(result["volumes"]) == 3
        assert len(result["timestamps"]) == 3

    def test_falls_back_to_previous_close_when_market_price_missing(self, monkeypatch):
        chart_payload = {
            "chart": {
                "result": [{
                    "meta": {
                        # regularMarketPrice missing; previousClose used.
                        "previousClose": 198.0,
                        "shortName": "Apple",
                    },
                    "indicators": {"quote": [{"close": [], "volume": []}]},
                    "timestamp": [],
                }]
            }
        }
        fake_resp = _mock_response(json_data=chart_payload)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        result = data_fetcher.fetch_price_from_chart_api("AAPL")
        assert result["current_price"] == 198.0
        assert result["name"] == "Apple"

    def test_returns_empty_on_non_200(self, monkeypatch):
        fake_resp = _mock_response(status_code=429)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)
        result = data_fetcher.fetch_price_from_chart_api("AAPL")
        assert result == {}

    def test_returns_empty_on_request_exception(self, monkeypatch):
        def _explode(*_a, **_k):
            raise data_fetcher.requests.RequestException("network down")

        monkeypatch.setattr(data_fetcher.requests, "get", _explode)
        result = data_fetcher.fetch_price_from_chart_api("AAPL")
        assert result == {}


class TestFetchWeeklyPriceHistory:
    """Tier 1-misclassified-as-Tier-3: weekly OHLC drives base-pattern
    detection (cup, flat-base, double-bottom). Bugs here mute breakout signals."""

    def test_extracts_weekly_ohlc_data(self, monkeypatch):
        chart_payload = {
            "chart": {
                "result": [{
                    "timestamp": [1700000000, 1700604800],  # 2 weeks
                    "indicators": {
                        "quote": [{
                            "open": [100.0, 102.0],
                            "high": [105.0, 107.0],
                            "low": [99.0, 101.0],
                            "close": [104.0, 106.0],
                            "volume": [50_000_000, 52_000_000],
                        }]
                    }
                }]
            }
        }
        fake_resp = _mock_response(json_data=chart_payload)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)

        result = data_fetcher.fetch_weekly_price_history("AAPL")
        assert len(result) == 2
        assert result[0]["close"] == 104.0
        assert result[1]["high"] == 107.0

    def test_returns_empty_list_on_non_200(self, monkeypatch):
        fake_resp = _mock_response(status_code=500)
        monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **k: fake_resp)
        assert data_fetcher.fetch_weekly_price_history("AAPL") == []

    def test_returns_empty_list_on_exception(self, monkeypatch):
        def _explode(*_a, **_k):
            raise Exception("network down")

        monkeypatch.setattr(data_fetcher.requests, "get", _explode)
        assert data_fetcher.fetch_weekly_price_history("AAPL") == []


class TestLoadCacheFromDbAdditionalFields:
    """Tier 1: ensure load_cache_from_db handles ALL field types — covers the
    institutional/analyst/key_metrics/short_interest branches."""

    def test_loads_full_p1_record_with_all_fields(self, db_session):
        rec = StockDataCache(
            ticker="FULL",
            quarterly_earnings=[1.0, 0.9],
            annual_earnings=[4.0],
            earnings_updated_at=datetime.now(),

            quarterly_revenue=[100e9],
            annual_revenue=[380e9],
            revenue_updated_at=datetime.now(),

            total_cash=50e9,
            total_debt=100e9,
            shares_outstanding=16e9,
            balance_updated_at=datetime.now(),

            analyst_target_price=240.0,
            analyst_count=25,
            analyst_updated_at=datetime.now(),

            institutional_holders_pct=65.0,
            institutional_updated_at=datetime.now(),

            roe=0.28,
            trailing_pe=28.0,
            forward_pe=24.0,
            peg_ratio=1.5,
            metrics_updated_at=datetime.now(),

            eps_estimate_current=11.0,
            eps_estimate_prior=10.0,
            eps_estimate_revision_pct=10.0,
            analyst_estimates_updated_at=datetime.now(),

            short_interest_pct=15.0,
            short_ratio=5.0,
            short_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()

        data_fetcher.load_cache_from_db()

        # Verify every data_type loaded into memory cache.
        assert data_fetcher.get_cached_data("FULL", "earnings") is not None
        assert data_fetcher.get_cached_data("FULL", "revenue") is not None
        assert data_fetcher.get_cached_data("FULL", "balance_sheet") is not None
        assert data_fetcher.get_cached_data("FULL", "analyst") is not None
        assert data_fetcher.get_cached_data("FULL", "institutional") == 65.0
        assert data_fetcher.get_cached_data("FULL", "key_metrics") is not None
        assert data_fetcher.get_cached_data("FULL", "analyst_estimates") is not None
        assert data_fetcher.get_cached_data("FULL", "short_interest") is not None

        # Spot-check key_metrics shape.
        km = data_fetcher.get_cached_data("FULL", "key_metrics")
        assert km["roe"] == pytest.approx(0.28)
        assert km["trailing_pe"] == pytest.approx(28.0)


class TestDataFetcherObjectCacheTTL:
    """2026-07-03 audit: main.py holds ONE DataFetcher for the process
    lifetime and get_stock_data early-returned cached StockData with no TTL —
    the refresh endpoint and stale-cache background refresh were no-ops by
    construction (frozen price re-stamped as fresh forever)."""

    class _Bypass(Exception):
        """Raised by the stubbed chart fetch — proves the cache was bypassed
        (fetch_price_from_chart_api is the first call after the cache check)."""

    def _fetcher_with_entry(self, age_seconds):
        f = data_fetcher.DataFetcher()
        sd = data_fetcher.StockData("AAPL")
        f._cache["AAPL"] = sd
        f._cache_fetched_at["AAPL"] = time.time() - age_seconds
        return f, sd

    def test_fresh_entry_served_from_cache(self, monkeypatch):
        f, sd = self._fetcher_with_entry(age_seconds=60)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            self._raise_bypass)
        assert f.get_stock_data("AAPL") is sd

    def test_expired_entry_bypasses_cache_and_refetches(self, monkeypatch):
        f, _ = self._fetcher_with_entry(
            age_seconds=data_fetcher.DataFetcher.CACHE_TTL_SECONDS + 1)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            self._raise_bypass)
        with pytest.raises(self._Bypass):
            f.get_stock_data("AAPL")
        # Expired entry was dropped, not re-served.
        assert "AAPL" not in f._cache

    def test_invalidate_forces_refetch(self, monkeypatch):
        f, _ = self._fetcher_with_entry(age_seconds=0)
        f.invalidate("AAPL")
        assert "AAPL" not in f._cache
        assert "AAPL" not in f._cache_fetched_at
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            self._raise_bypass)
        with pytest.raises(self._Bypass):
            f.get_stock_data("AAPL")

    @classmethod
    def _raise_bypass(cls, ticker):
        raise cls._Bypass(ticker)


class TestScanIntegrityGuardsSourcePins:
    """2026-07-03 audit — pin the inline guards in the async orchestration
    body (get_stock_data_async is too large to drive end-to-end; same
    source-pin convention as TestSyncAnalystEstimatesShapeParity)."""

    def _src(self):
        import inspect
        import async_data_fetcher
        return inspect.getsource(async_data_fetcher.get_stock_data_async)

    def test_financials_guard_is_not_bare_truthy(self):
        # fetch_fmp_financials_async ALWAYS returns a pre-populated dict, so
        # `if financials:` cached EMPTY earnings as fresh for 7 days on a
        # total FMP failure and overwrote good DB rows with [].
        src = self._src()
        assert "has_financials" in src
        assert "if financials:\n" not in src

    def test_cache_fallback_reuse_does_not_relaunder_age(self):
        # Fallback reuse must not re-cache/mark-fetched (age laundering).
        src = self._src()
        assert src.count('get("used_cache_fallback")') >= 2

    def test_institutional_fallback_guards_share_counts(self):
        # The "institutional" cache holds FMP share COUNTS or Finviz percents;
        # assigning a count as a percent inflates the I score to top tier.
        src = self._src()
        assert "cached_inst > 100" in src
