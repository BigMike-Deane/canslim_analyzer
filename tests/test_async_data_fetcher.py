"""
Tests for async_data_fetcher.py — async FMP/Yahoo fetcher used by the live scanner.

Triage:
  Tier 1 (high blast radius — silent data poisoning into live scoring):
    - fetch_json_async (THE shared HTTP entry point: 200/429/500-504/timeout retry,
      exponential backoff cap, circuit-breaker integration)
    - _check_rate_limit (legacy in-module rate limiter: minute reset, backoff_until,
      counter increment, per-minute cap)
    - _init_async_primitives (semaphore + rate-limiter binding to the running loop)
    - fetch_fmp_single_quote / single_profile (per-ticker fetchers; quote marks
      delisted on empty response — load-bearing for the universe-pruning loop)
    - fetch_fmp_financials_async (the consolidated earnings+revenue path; cache
      fallback gate at MAX_FALLBACK_CACHE_AGE_DAYS=7 — silently writing stale
      data into score inputs is the single highest-blast bug class here)
    - fetch_fmp_key_metrics_async / balance_sheet_async / analyst_async
      (NaN-guard via `or 0` — must NOT poison fields that flow into A/L scoring)
    - fetch_fmp_earnings_surprise_async (beat_streak, surprise_pct, "N/A"-safe
      quarterly_adjusted_eps — Approach 2 surprise_pct gap signal lives here)
    - fetch_fmp_analyst_estimates_async (revision_pct + trend bucket; year-match
      logic + malformed-date skip)

  Tier 2 (helpers consumed by Tier 1 / used in scanner observability):
    - get_cache_age_days, get_fallback_stats, reset_fallback_tracker
    - save_scan_progress / load_scan_progress / clear_scan_progress
      (checkpoint persistence — scan_id mismatch + 1-hour TTL gates)
    - fetch_fmp_batch_quotes / batch_profiles (parallel chunking + filter)
    - get_rate_limit_stats (centralized stats passthrough)

  Tier 3 (intentionally NOT covered — see top of file):
    - refresh_yfinance_session, get_yf_ticker_with_retry, yf_safe_call
      (mock yfinance module internals — would couple tests to yfinance internals)
    - fetch_yahoo_info_comprehensive_async, fetch_yahoo_supplement_async
      (orchestrators that read .info DataFrames + rotate semaphores)
    - fetch_insider_trading_async (pandas DataFrame walk over yf.insider_transactions
      with regex price extraction — high-fidelity test would need 8+ row fixtures)
    - get_stock_data_async (~270-line orchestrator — would need 15+ patches each;
      exercised by scanner integration / live scans)
    - fetch_stocks_batch_async (orchestration of get_stock_data_async + checkpoint
      + fallback stats; same 15+-patch tax)
    - get_price_data_only / fetch_stocks_async_wrapper (sync wrappers + ETF helper
      with no scoring blast radius)
    - __main__ block

All FMP HTTP is mocked at the aiohttp.ClientSession layer — no live network.
No DB tests here (async_data_fetcher delegates DB I/O to data_fetcher, which is
already covered by tests/test_data_fetcher.py).
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

import async_data_fetcher
import data_fetcher
import fmp_rate_limiter


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch, tmp_path):
    """Reset all cross-module state between tests so counters/sets/breakers
    don't leak. Also redirects the checkpoint file to a per-test tmp path so
    save/load/clear progress tests don't write to data/scan_checkpoint.json."""
    # async_data_fetcher rate-limiter dict
    async_data_fetcher._rate_limiter["calls_this_minute"] = 0
    async_data_fetcher._rate_limiter["minute_start"] = None
    async_data_fetcher._rate_limiter["consecutive_429s"] = 0
    async_data_fetcher._rate_limiter["backoff_until"] = None
    async_data_fetcher._rate_limiter["total_calls"] = 0
    async_data_fetcher._rate_limiter["total_429s"] = 0

    # async_data_fetcher fallback tracker
    async_data_fetcher._fallback_tracker["stocks_using_fallback"] = set()
    async_data_fetcher._fallback_tracker["stocks_no_data"] = set()
    async_data_fetcher._fallback_tracker["last_reset"] = None

    # data_fetcher caches (async fetchers read/write through these)
    with data_fetcher._cached_data_lock:
        data_fetcher._cached_data.clear()
    with data_fetcher._freshness_lock:
        data_fetcher._data_freshness_cache.clear()

    # fmp_rate_limiter singletons — reset so a tripped breaker / drained bucket
    # from a prior test doesn't make the next test hang in backoff.
    fmp_rate_limiter._breaker.state = fmp_rate_limiter.CircuitBreaker.CLOSED
    fmp_rate_limiter._breaker.consecutive_failures = 0
    fmp_rate_limiter._breaker._cooldown_until = None
    fmp_rate_limiter._breaker._trip_count = 0
    fmp_rate_limiter._governor._tokens = 1000.0  # plenty so tests don't sleep
    fmp_rate_limiter._metrics.reset()

    # Redirect checkpoint file to a tmp path for save/load/clear tests
    monkeypatch.setattr(
        async_data_fetcher, "CHECKPOINT_FILE", tmp_path / "scan_checkpoint.json"
    )

    yield


@pytest.fixture
def fmp_key(monkeypatch):
    """Set a non-empty FMP_API_KEY on both modules so async fetchers don't early-return."""
    monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "test-key-1234")
    monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "test-key-1234")


@pytest.fixture
def no_delist(monkeypatch):
    """Stub mark_ticker_as_delisted / clear_delisted_ticker so async fetchers
    don't write to the production SQLite delisted table during tests."""
    monkeypatch.setattr(async_data_fetcher, "mark_ticker_as_delisted", lambda *a, **k: None)
    monkeypatch.setattr(async_data_fetcher, "clear_delisted_ticker", lambda *a, **k: None)


@pytest.fixture
async def primitives():
    """Bind asyncio primitives + initialize fmp_rate_limiter to the running loop.
    Required for any test that calls fetch_json_async or _check_rate_limit."""
    await async_data_fetcher._init_async_primitives()
    # _init also re-tunes the governor; force a full bucket so tests don't sleep.
    fmp_rate_limiter._governor._tokens = 1000.0
    yield


# ── aiohttp session/response mocks ────────────────────────────────────────────


class MockResponse:
    """Mocks aiohttp's response object AND the async-context-manager that
    session.get(...) returns. session.get(url) is used as
        async with session.get(url) as response: ...
    so the returned object must implement __aenter__/__aexit__ AND expose
    .status and an async .json() method."""

    def __init__(self, status: int, json_data=None, raise_on_enter=None):
        self.status = status
        self._json_data = json_data
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self):
        return self._json_data


class MockSession:
    """Mocks aiohttp.ClientSession. Each call to .get() pops the next
    queued response (or callable). Records every URL it was asked for."""

    def __init__(self, responses):
        # responses can be: list of MockResponse OR list of (status, json) tuples
        # OR list of Exceptions OR list of callables(url)->MockResponse
        self._responses = list(responses)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if not self._responses:
            return MockResponse(200, [])
        item = self._responses.pop(0)
        if callable(item) and not isinstance(item, MockResponse):
            return item(url)
        if isinstance(item, BaseException):
            return MockResponse(500, raise_on_enter=item)
        if isinstance(item, MockResponse):
            return item
        # tuple form
        status, data = item
        return MockResponse(status, data)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _init_async_primitives + module configuration
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsyncPrimitives:
    """Tier 1: _init_async_primitives — semaphore binding to the running loop."""

    async def test_init_creates_semaphore_bound_to_current_loop(self):
        await async_data_fetcher._init_async_primitives()
        assert async_data_fetcher.api_semaphore is not None
        # Must be usable from this loop — would raise RuntimeError if mis-bound.
        async with async_data_fetcher.api_semaphore:
            pass

    async def test_init_resets_rate_limiter_state(self):
        async_data_fetcher._rate_limiter["calls_this_minute"] = 100
        async_data_fetcher._rate_limiter["consecutive_429s"] = 5
        async_data_fetcher._rate_limiter["total_429s"] = 7
        async_data_fetcher._rate_limiter["total_calls"] = 999

        await async_data_fetcher._init_async_primitives()

        assert async_data_fetcher._rate_limiter["calls_this_minute"] == 0
        assert async_data_fetcher._rate_limiter["consecutive_429s"] == 0
        assert async_data_fetcher._rate_limiter["total_429s"] == 0
        assert async_data_fetcher._rate_limiter["total_calls"] == 0
        assert async_data_fetcher._rate_limiter["minute_start"] is None
        assert async_data_fetcher._rate_limiter["backoff_until"] is None

    async def test_init_creates_yahoo_semaphore(self):
        await async_data_fetcher._init_async_primitives()
        assert async_data_fetcher._yahoo_semaphore is not None
        async with async_data_fetcher._yahoo_semaphore:
            pass


class TestModuleConfiguration:
    """Tier 1: load-bearing module constants."""

    def test_max_concurrent_requests_is_set(self):
        # 12 was tuned by bench_fetch.py — guard against silent reductions
        # that would re-introduce FMP wait time.
        assert async_data_fetcher.MAX_CONCURRENT_REQUESTS == 12

    def test_fmp_base_url_is_https_stable(self):
        # Project rule: /stable/ endpoints only — /api/v3/ requires legacy sub.
        assert async_data_fetcher.FMP_BASE_URL.startswith("https://")
        assert "/stable" in async_data_fetcher.FMP_BASE_URL

    def test_max_fallback_cache_age_is_seven_days(self):
        # The cache-fallback gate. If this drops to 0 we silently never fall back;
        # if it grows to 30 we silently feed month-old data into the scorer.
        assert async_data_fetcher.MAX_FALLBACK_CACHE_AGE_DAYS == 7

    def test_yahoo_semaphore_size_is_five(self):
        # bench_fetch.py picked 5; smaller values reintroduce wall-clock cost.
        # Note: the module-level _yahoo_semaphore is replaced per-scan by
        # _init_async_primitives; this guards the ad-hoc default.
        assert async_data_fetcher._yahoo_semaphore._value == 5

    def test_yahoo_max_retries_is_two(self):
        assert async_data_fetcher._yahoo_max_retries == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_json_async (THE shared HTTP path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchJsonAsync:
    """Tier 1: fetch_json_async — single point of HTTP failure for every FMP
    fetcher. Coverage here lights up the path used by ~10 downstream callers."""

    async def test_returns_json_on_200(self, primitives):
        session = MockSession([(200, [{"ticker": "AAPL", "price": 100.0}])])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y?z")
        assert result == [{"ticker": "AAPL", "price": 100.0}]

    async def test_429_then_200_succeeds_after_backoff(self, primitives, monkeypatch):
        # Avoid actually sleeping during the backoff
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        # Make backoff deterministic and short (calculate_backoff uses random.uniform)
        monkeypatch.setattr(fmp_rate_limiter, "calculate_backoff", lambda *_a, **_k: 0.0)

        session = MockSession([(429, None), (200, {"ok": True})])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result == {"ok": True}

    async def test_500_triggers_retry_with_exponential_backoff(self, primitives, monkeypatch):
        sleeps = []

        async def fake_sleep(n):
            sleeps.append(n)

        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", fake_sleep)

        session = MockSession([(500, None), (502, None), (200, {"ok": True})])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result == {"ok": True}
        # min(2 ** attempt, 10) — first retry waits 1s, second waits 2s
        assert sleeps[:2] == [1, 2]

    async def test_500_exhaustion_returns_none(self, primitives, monkeypatch):
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        # max_retries from fmp_rate_limiter._config is 3 → all three return 500
        session = MockSession([(500, None), (503, None), (504, None)])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result is None

    async def test_404_returns_none_immediately(self, primitives):
        session = MockSession([(404, None)])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result is None
        # Single call — no retry on non-retriable status
        assert len(session.calls) == 1

    async def test_circuit_breaker_open_returns_none(self, primitives, monkeypatch):
        async def raise_breaker():
            raise fmp_rate_limiter.CircuitBreakerOpen("open")

        monkeypatch.setattr(fmp_rate_limiter, "acquire_async", raise_breaker)
        session = MockSession([(200, {"never": "called"})])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result is None
        # session.get is NOT called when breaker is open
        assert session.calls == []

    async def test_timeout_retries_then_returns_none(self, primitives, monkeypatch):
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        # All attempts time out → returns None
        session = MockSession(
            [asyncio.TimeoutError(), asyncio.TimeoutError(), asyncio.TimeoutError()]
        )
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result is None

    async def test_generic_exception_returns_none_no_retry(self, primitives):
        # Generic (non-Timeout) exception inside the request: returns None
        # immediately without retrying.
        session = MockSession([RuntimeError("boom")])
        result = await async_data_fetcher.fetch_json_async(session, "https://x/y")
        assert result is None
        # The request was attempted exactly once
        assert len(session.calls) == 1

    async def test_records_success_metrics(self, primitives):
        session = MockSession([(200, [])])
        await async_data_fetcher.fetch_json_async(session, "https://x/y")
        stats = fmp_rate_limiter.get_rate_limit_stats()
        assert stats["successful_requests"] == 1
        assert stats["errors_429"] == 0

    async def test_records_429_metrics(self, primitives, monkeypatch):
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(fmp_rate_limiter, "calculate_backoff", lambda *_a, **_k: 0.0)
        session = MockSession([(429, None), (429, None), (429, None)])
        await async_data_fetcher.fetch_json_async(session, "https://x/y")
        stats = fmp_rate_limiter.get_rate_limit_stats()
        assert stats["errors_429"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _check_rate_limit (legacy in-module rate limiter)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckRateLimit:
    """Tier 1: _check_rate_limit — minute boundary, max-per-minute cap,
    backoff_until gate. Increments total_calls/calls_this_minute on every call."""

    async def test_increments_call_counters(self, primitives):
        await async_data_fetcher._check_rate_limit()
        assert async_data_fetcher._rate_limiter["calls_this_minute"] == 1
        assert async_data_fetcher._rate_limiter["total_calls"] == 1

        await async_data_fetcher._check_rate_limit()
        assert async_data_fetcher._rate_limiter["calls_this_minute"] == 2
        assert async_data_fetcher._rate_limiter["total_calls"] == 2

    async def test_minute_start_set_on_first_call(self, primitives):
        assert async_data_fetcher._rate_limiter["minute_start"] is None
        await async_data_fetcher._check_rate_limit()
        assert async_data_fetcher._rate_limiter["minute_start"] is not None

    async def test_minute_boundary_resets_counter(self, primitives):
        # Simulate minute_start as 90 seconds ago — must reset counter.
        async_data_fetcher._rate_limiter["minute_start"] = datetime.now() - timedelta(seconds=90)
        async_data_fetcher._rate_limiter["calls_this_minute"] = 200

        await async_data_fetcher._check_rate_limit()

        assert async_data_fetcher._rate_limiter["calls_this_minute"] == 1  # reset then ++

    # backoff_until branch tests removed 2026-05-21 alongside commit 6b9b368:
    # the field and its if-branch in _check_rate_limit were dead code (no
    # setter ever wrote a non-None value). 429 backoff is handled by the
    # centralized fmp_rate_limiter module.

    async def test_at_max_calls_waits_for_next_minute(self, primitives, monkeypatch):
        sleep_calls = []

        async def fake_sleep(n):
            sleep_calls.append(n)

        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", fake_sleep)

        # We are AT the max already; minute started 30s ago.
        async_data_fetcher._rate_limiter["minute_start"] = datetime.now() - timedelta(seconds=30)
        async_data_fetcher._rate_limiter["calls_this_minute"] = (
            async_data_fetcher._rate_limiter["max_calls_per_minute"]
        )

        await async_data_fetcher._check_rate_limit()

        # Should have slept ~30s (60 - 30 = 30s remaining in window)
        assert len(sleep_calls) == 1
        assert 28.0 <= sleep_calls[0] <= 31.0
        # After sleep, counter resets and increments to 1
        assert async_data_fetcher._rate_limiter["calls_this_minute"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_single_quote / single_profile
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpSingleQuote:
    """Tier 1: fetch_fmp_single_quote — and the empty-response → mark_delisted path."""

    async def test_no_api_key_returns_none(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_single_quote(session, "AAPL")
        assert result is None
        assert session.calls == []  # short-circuited before HTTP

    async def test_successful_response_parsed(self, fmp_key, primitives, no_delist):
        quote_data = [{
            "price": 150.0, "yearHigh": 200.0, "yearLow": 100.0,
            "volume": 1_000_000, "avgVolume": 800_000, "marketCap": 3e12,
            "pe": 30.0, "sharesOutstanding": 16e9, "name": "Apple Inc.",
        }]
        session = MockSession([(200, quote_data)])
        result = await async_data_fetcher.fetch_fmp_single_quote(session, "AAPL")
        assert result["current_price"] == 150.0
        assert result["high_52w"] == 200.0
        assert result["low_52w"] == 100.0
        assert result["market_cap"] == 3e12
        assert result["pe"] == 30.0
        assert result["name"] == "Apple Inc."

    async def test_empty_list_marks_delisted_and_returns_none(self, fmp_key, primitives, monkeypatch):
        delisted_calls = []
        monkeypatch.setattr(
            async_data_fetcher, "mark_ticker_as_delisted",
            lambda ticker, reason="", source="": delisted_calls.append((ticker, reason)),
        )
        monkeypatch.setattr(async_data_fetcher, "clear_delisted_ticker", lambda *a, **k: None)

        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_single_quote(session, "DELISTED")
        assert result is None
        assert delisted_calls == [("DELISTED", "fmp_quote_empty")]

    async def test_none_response_returns_none_no_delist(self, fmp_key, primitives, monkeypatch):
        delisted_calls = []
        monkeypatch.setattr(
            async_data_fetcher, "mark_ticker_as_delisted",
            lambda *a, **k: delisted_calls.append(a),
        )

        session = MockSession([(500, None), (500, None), (500, None)])
        # All retries 500 → fetch_json_async returns None → don't mark delisted
        # (None != empty list — the contract is: empty list IS authoritative
        # delisting, None is "API failure, try again later").
        with patch.object(async_data_fetcher.asyncio, "sleep", new_callable=AsyncMock):
            result = await async_data_fetcher.fetch_fmp_single_quote(session, "FOO")
        assert result is None
        assert delisted_calls == []


class TestFetchFmpSingleProfile:
    """Tier 1: fetch_fmp_single_profile — including the range-string parse."""

    async def test_no_api_key_returns_none(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "AAPL")
        assert result is None
        assert session.calls == []

    async def test_successful_response_parses_range(self, fmp_key, primitives):
        profile_data = [{
            "companyName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
            "mktCap": 3e12, "price": 150.0, "range": "100.50-200.75",
            "sharesOutstanding": 16e9,
            "description": "Apple designs consumer electronics.",
            "fullTimeEmployees": "164000",
        }]
        session = MockSession([(200, profile_data)])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "AAPL")
        assert result["sector"] == "Technology"
        assert result["industry"] == "Consumer Electronics"
        assert result["high_52w"] == 200.75
        assert result["low_52w"] == 100.50
        assert result["market_cap"] == 3e12
        # Security-type guard inputs must pass through
        assert result["description"] == "Apple designs consumer electronics."
        assert result["full_time_employees"] == "164000"

    async def test_profile_missing_guard_fields_defaults_safely(self, fmp_key, primitives):
        """Older/partial FMP payloads without description/employees must not
        break the guard inputs — empty description means 'not a CEF'."""
        profile_data = [{
            "companyName": "Bare Co", "sector": "X", "industry": "Y",
            "mktCap": 1e9, "price": 10.0, "sharesOutstanding": 1e8,
        }]
        session = MockSession([(200, profile_data)])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "BARE")
        assert result["description"] == ""
        assert result["full_time_employees"] is None

    async def test_malformed_range_gracefully_zeroed(self, fmp_key, primitives):
        profile_data = [{
            "companyName": "Foo", "sector": "X", "industry": "Y",
            "mktCap": 1e9, "price": 50.0, "range": "garbage",  # no hyphen → split returns single
            "sharesOutstanding": 1e9,
        }]
        session = MockSession([(200, profile_data)])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "FOO")
        # Default zero values when parse fails
        assert result["high_52w"] == 0
        assert result["low_52w"] == 0

    async def test_range_with_non_numeric_values_falls_back_to_zero(self, fmp_key, primitives):
        # Range like "N/A-N/A" — parse should fail silently, NOT raise
        profile_data = [{
            "companyName": "X", "sector": "", "industry": "",
            "mktCap": 0, "price": 0.0, "range": "N/A-N/A",
            "sharesOutstanding": 0,
        }]
        session = MockSession([(200, profile_data)])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "X")
        assert result["high_52w"] == 0
        assert result["low_52w"] == 0

    async def test_empty_list_returns_none(self, fmp_key, primitives):
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_single_profile(session, "EMPTY")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — fetch_fmp_batch_quotes / batch_profiles
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpBatchQuotes:
    """Tier 2: fetch_fmp_batch_quotes — parallel chunking + filter."""

    async def test_no_tickers_returns_empty(self, fmp_key, primitives):
        session = MockSession([])
        result = await async_data_fetcher.fetch_fmp_batch_quotes(session, [])
        assert result == {}

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        session = MockSession([])
        result = await async_data_fetcher.fetch_fmp_batch_quotes(session, ["AAPL", "MSFT"])
        assert result == {}

    async def test_aggregates_successful_results(self, fmp_key, primitives, no_delist):
        good_quote = [{"price": 150.0, "yearHigh": 200, "yearLow": 100,
                       "volume": 1, "avgVolume": 1, "marketCap": 1e9,
                       "pe": 0, "sharesOutstanding": 1e6, "name": "X"}]
        # Two tickers, both succeed (one response per ticker).
        session = MockSession([(200, good_quote), (200, good_quote)])
        result = await async_data_fetcher.fetch_fmp_batch_quotes(session, ["AAPL", "MSFT"])
        assert set(result.keys()) == {"AAPL", "MSFT"}

    async def test_filters_out_failures(self, fmp_key, primitives, no_delist):
        good_quote = [{"price": 1.0, "yearHigh": 1, "yearLow": 1, "volume": 1,
                       "avgVolume": 1, "marketCap": 1, "pe": 0,
                       "sharesOutstanding": 1, "name": "X"}]
        # AAPL succeeds, BAD returns empty list (None after parsing)
        session = MockSession([(200, good_quote), (200, [])])
        result = await async_data_fetcher.fetch_fmp_batch_quotes(session, ["AAPL", "BAD"])
        # BAD is filtered out
        assert "AAPL" in result
        assert "BAD" not in result


class TestFetchFmpBatchProfiles:
    """Tier 2: fetch_fmp_batch_profiles — same parallel-chunk shape."""

    async def test_no_tickers_returns_empty(self, fmp_key, primitives):
        session = MockSession([])
        result = await async_data_fetcher.fetch_fmp_batch_profiles(session, [])
        assert result == {}

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_batch_profiles(MockSession([]), ["AAPL"])
        assert result == {}

    async def test_aggregates_successful_results(self, fmp_key, primitives):
        good = [{"companyName": "X", "sector": "T", "industry": "I",
                 "mktCap": 1, "price": 1, "range": "1-2", "sharesOutstanding": 1}]
        session = MockSession([(200, good), (200, good)])
        result = await async_data_fetcher.fetch_fmp_batch_profiles(session, ["AAPL", "MSFT"])
        assert set(result.keys()) == {"AAPL", "MSFT"}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_financials_async + cache fallback gate
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpFinancialsAsync:
    """Tier 1: fetch_fmp_financials_async — earnings+revenue + cache fallback."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        session = MockSession([])
        result = await async_data_fetcher.fetch_fmp_financials_async(session, "AAPL")
        assert result == {}

    async def test_quarterly_and_annual_populated_on_success(self, fmp_key, primitives):
        quarterly = [
            {"eps": 1.2, "revenue": 100, "netIncome": 50},
            {"eps": 1.1, "revenue": 95,  "netIncome": 45},
        ]
        annual = [
            {"eps": 4.5, "revenue": 400, "netIncome": 200},
        ]
        session = MockSession([(200, quarterly), (200, annual)])
        result = await async_data_fetcher.fetch_fmp_financials_async(session, "AAPL")
        assert result["quarterly_eps"] == [1.2, 1.1]
        assert result["quarterly_revenue"] == [100, 95]
        assert result["quarterly_net_income"] == [50, 45]
        assert result["annual_eps"] == [4.5]
        assert result["annual_revenue"] == [400]
        assert result["used_cache_fallback"] is False

    async def test_nan_safe_zero_on_missing_eps(self, fmp_key, primitives):
        # Ensure `q.get("eps", 0) or 0` — None values become 0, not NaN.
        # Project rule: NaN poisoning has bitten this codebase twice.
        quarterly = [{"eps": None, "revenue": None, "netIncome": None}]
        session = MockSession([(200, quarterly), (200, [])])
        result = await async_data_fetcher.fetch_fmp_financials_async(session, "X")
        assert result["quarterly_eps"] == [0]
        assert result["quarterly_revenue"] == [0]
        assert result["quarterly_net_income"] == [0]

    async def test_both_fail_with_fresh_cache_uses_fallback(self, fmp_key, primitives, monkeypatch):
        # API both fail (None) but earnings cache is fresh (just written) → fallback.
        ticker = "FALLBACK"
        # Stale enough to have failed, fresh enough to fall back: write to cache,
        # then mark freshness 2 days ago (within 7-day MAX_FALLBACK_CACHE_AGE_DAYS).
        data_fetcher.set_cached_data(
            ticker, "earnings",
            {"quarterly_eps": [1.0, 0.9], "annual_eps": [3.5]},
            persist_to_db=False,
        )
        data_fetcher.set_cached_data(
            ticker, "revenue",
            {"quarterly_revenue": [100, 95], "annual_revenue": [400]},
            persist_to_db=False,
        )
        # Set explicit freshness timestamp 2 days ago (gives age within 7-day gate)
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache[ticker] = {
                "earnings": datetime.now() - timedelta(days=2),
            }

        # Both API calls return None
        session = MockSession([(500, None), (500, None), (500, None),
                               (500, None), (500, None), (500, None)])
        with patch.object(async_data_fetcher.asyncio, "sleep", new_callable=AsyncMock):
            result = await async_data_fetcher.fetch_fmp_financials_async(session, ticker)
        assert result["used_cache_fallback"] is True
        assert result["quarterly_eps"] == [1.0, 0.9]
        assert result["annual_eps"] == [3.5]
        assert result["quarterly_revenue"] == [100, 95]
        # Tracker recorded the fallback usage
        assert ticker in async_data_fetcher._fallback_tracker["stocks_using_fallback"]

    async def test_both_fail_with_stale_cache_skips_fallback(self, fmp_key, primitives):
        # Cache is too old (10 days > 7) → don't use it, mark as no-data.
        ticker = "STALECACHE"
        data_fetcher.set_cached_data(
            ticker, "earnings", {"quarterly_eps": [9.9]}, persist_to_db=False,
        )
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache[ticker] = {
                "earnings": datetime.now() - timedelta(days=10),
            }

        session = MockSession([(500, None), (500, None), (500, None),
                               (500, None), (500, None), (500, None)])
        with patch.object(async_data_fetcher.asyncio, "sleep", new_callable=AsyncMock):
            result = await async_data_fetcher.fetch_fmp_financials_async(session, ticker)
        assert result["used_cache_fallback"] is False
        # Stale cache must NOT be merged into result (would silently feed
        # 10-day-old EPS into the scorer)
        assert result["quarterly_eps"] == []
        assert ticker in async_data_fetcher._fallback_tracker["stocks_no_data"]

    async def test_both_fail_no_cache_marks_no_data(self, fmp_key, primitives):
        ticker = "NEVERSEEN"
        session = MockSession([(500, None), (500, None), (500, None),
                               (500, None), (500, None), (500, None)])
        with patch.object(async_data_fetcher.asyncio, "sleep", new_callable=AsyncMock):
            result = await async_data_fetcher.fetch_fmp_financials_async(session, ticker)
        assert result["used_cache_fallback"] is False
        assert result["quarterly_eps"] == []
        assert ticker in async_data_fetcher._fallback_tracker["stocks_no_data"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _apply_cached_financials (fresh-but-missing cache must force a fetch)
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyCachedFinancials:
    """Regression (Jul 2026 mass C-score wipe): _apply_cached_financials must
    return False — forcing a real FMP fetch — whenever the fresh-looking cache
    can't actually supply quarterly EPS. The original inline branch trusted
    is_data_fresh() alone; after memory-cache eviction desync that meant
    scoring 400+ stocks with empty earnings and persisting the wipe."""

    def test_fresh_cache_with_eps_applies_and_returns_true(self):
        data_fetcher.set_cached_data(
            "CACHED", "earnings",
            {"quarterly_eps": [1.2, 1.0], "annual_eps": [4.0]},
            persist_to_db=False,
        )
        data_fetcher.set_cached_data(
            "CACHED", "revenue",
            {"quarterly_revenue": [10, 9], "annual_revenue": [40]},
            persist_to_db=False,
        )
        data_fetcher.mark_data_fetched("CACHED", "earnings")

        sd = async_data_fetcher.StockData("CACHED")
        assert async_data_fetcher._apply_cached_financials("CACHED", sd) is True
        assert sd.quarterly_earnings == [1.2, 1.0]
        assert sd.annual_earnings == [4.0]
        assert sd.quarterly_revenue == [10, 9]
        assert sd.annual_revenue == [40]

    def test_fresh_stamp_with_missing_entry_returns_false(self):
        # The desync itself: freshness says fresh, data entry was evicted.
        data_fetcher.mark_data_fetched("GONE", "earnings")
        sd = async_data_fetcher.StockData("GONE")
        assert async_data_fetcher._apply_cached_financials("GONE", sd) is False
        assert sd.quarterly_earnings == []

    def test_fresh_entry_with_empty_eps_returns_false(self):
        # Poisoned entry: fresh stamp, payload present, but no EPS behind it.
        data_fetcher.set_cached_data(
            "EMPTY", "earnings", {"quarterly_eps": [], "annual_eps": []},
            persist_to_db=False,
        )
        data_fetcher.mark_data_fetched("EMPTY", "earnings")
        sd = async_data_fetcher.StockData("EMPTY")
        assert async_data_fetcher._apply_cached_financials("EMPTY", sd) is False

    def test_not_fresh_returns_false_even_with_data(self):
        data_fetcher.set_cached_data(
            "STALE", "earnings", {"quarterly_eps": [1.0]}, persist_to_db=False,
        )
        # No mark_data_fetched → not fresh → must fetch.
        sd = async_data_fetcher.StockData("STALE")
        assert async_data_fetcher._apply_cached_financials("STALE", sd) is False

    def test_legacy_quarterly_key_supported(self):
        # set_cached_data at fetch time writes {"quarterly": ..., "annual": ...};
        # DB hydration writes {"quarterly_eps": ..., "annual_eps": ...}. Both
        # shapes must count as usable.
        data_fetcher.set_cached_data(
            "LEGACY", "earnings", {"quarterly": [0.5, 0.4], "annual": [2.0]},
            persist_to_db=False,
        )
        data_fetcher.mark_data_fetched("LEGACY", "earnings")
        sd = async_data_fetcher.StockData("LEGACY")
        assert async_data_fetcher._apply_cached_financials("LEGACY", sd) is True
        assert sd.quarterly_earnings == [0.5, 0.4]
        assert sd.annual_earnings == [2.0]


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_key_metrics / balance_sheet / analyst (NaN guards)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpKeyMetricsAsync:
    """Tier 1: fetch_fmp_key_metrics_async — NaN/None → 0 for scoring inputs."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_key_metrics_async(MockSession([]), "AAPL")
        assert result == {}

    async def test_successful_response(self, fmp_key, primitives):
        data = [{"returnOnEquity": 0.25, "returnOnAssets": 0.15,
                 "returnOnInvestedCapital": 0.20, "currentRatio": 1.5,
                 "earningsYield": 0.05, "freeCashFlowYield": 0.04}]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_key_metrics_async(session, "AAPL")
        assert result["roe"] == 0.25
        assert result["roa"] == 0.15
        assert result["roic"] == 0.20

    async def test_none_values_become_zero(self, fmp_key, primitives):
        # None / missing fields must NOT propagate as NaN/None into scoring.
        data = [{"returnOnEquity": None, "returnOnAssets": None,
                 "returnOnInvestedCapital": None, "currentRatio": None,
                 "earningsYield": None, "freeCashFlowYield": None}]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_key_metrics_async(session, "X")
        assert result == {"roe": 0, "roa": 0, "roic": 0,
                          "current_ratio": 0, "earnings_yield": 0, "fcf_yield": 0}

    async def test_empty_response_returns_empty_dict(self, fmp_key, primitives):
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_key_metrics_async(session, "X")
        assert result == {}


class TestFetchFmpBalanceSheetAsync:
    """Tier 1: fetch_fmp_balance_sheet_async — NaN guard."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_balance_sheet_async(MockSession([]), "AAPL")
        assert result == {}

    async def test_successful_response(self, fmp_key, primitives):
        data = [{"cashAndCashEquivalents": 5e9, "totalDebt": 1e9,
                 "totalAssets": 100e9, "totalLiabilities": 50e9}]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_balance_sheet_async(session, "AAPL")
        assert result["cash_and_equivalents"] == 5e9
        assert result["total_debt"] == 1e9

    async def test_none_values_become_zero(self, fmp_key, primitives):
        data = [{"cashAndCashEquivalents": None, "totalDebt": None,
                 "totalAssets": None, "totalLiabilities": None}]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_balance_sheet_async(session, "X")
        assert result == {"cash_and_equivalents": 0, "total_debt": 0,
                          "total_assets": 0, "total_liabilities": 0}

    async def test_empty_response_returns_empty_dict(self, fmp_key, primitives):
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_balance_sheet_async(session, "X")
        assert result == {}


class TestFetchFmpAnalystAsync:
    """Tier 1: fetch_fmp_analyst_async — estimates + targets in parallel."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_analyst_async(MockSession([]), "AAPL")
        assert result == {}

    async def test_both_estimates_and_targets_parsed(self, fmp_key, primitives):
        estimates = [{"estimatedEpsAvg": 5.0, "numberAnalystsEstimatedEps": 25}]
        targets = [{"targetHigh": 200, "targetLow": 100,
                    "targetConsensus": 150, "targetMedian": 145}]
        session = MockSession([(200, estimates), (200, targets)])
        result = await async_data_fetcher.fetch_fmp_analyst_async(session, "AAPL")
        assert result["estimated_eps_avg"] == 5.0
        assert result["num_analysts"] == 25
        assert result["target_high"] == 200
        assert result["target_consensus"] == 150

    async def test_only_estimates_returns_partial(self, fmp_key, primitives):
        estimates = [{"estimatedEpsAvg": 5.0, "numberAnalystsEstimatedEps": 10}]
        session = MockSession([(200, estimates), (200, [])])
        result = await async_data_fetcher.fetch_fmp_analyst_async(session, "AAPL")
        assert result["estimated_eps_avg"] == 5.0
        assert "target_high" not in result

    async def test_only_targets_returns_partial(self, fmp_key, primitives):
        targets = [{"targetHigh": 200, "targetLow": 100,
                    "targetConsensus": 150, "targetMedian": 145}]
        session = MockSession([(200, []), (200, targets)])
        result = await async_data_fetcher.fetch_fmp_analyst_async(session, "AAPL")
        assert "estimated_eps_avg" not in result
        assert result["target_consensus"] == 150

    async def test_both_empty_returns_empty(self, fmp_key, primitives):
        session = MockSession([(200, []), (200, [])])
        result = await async_data_fetcher.fetch_fmp_analyst_async(session, "AAPL")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_earnings_surprise_async (beat-streak + N/A safety)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpEarningsSurpriseAsync:
    """Tier 1: surprise_pct + beat_streak (consumed by Approach 2 scoring) +
    quarterly_adjusted_eps with non-numeric "N/A" filtering."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(
            MockSession([]), "AAPL"
        )
        assert result == {}

    async def test_surprise_pct_calculation(self, fmp_key, primitives):
        # Estimated $1.00, actual $1.20 → 20% surprise
        data = [
            {"estimatedEarning": 1.00, "actualEarningResult": 1.20},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(session, "AAPL")
        assert result["latest_surprise_pct"] == pytest.approx(20.0)
        assert result["beat_streak"] == 1
        assert result["quarterly_adjusted_eps"] == [1.20]

    async def test_beat_streak_three_then_miss(self, fmp_key, primitives):
        # 3 beats then a miss → beat_streak == 3
        data = [
            {"estimatedEarning": 1.0, "actualEarningResult": 1.2},  # beat
            {"estimatedEarning": 1.0, "actualEarningResult": 1.1},  # beat
            {"estimatedEarning": 1.0, "actualEarningResult": 1.05}, # beat
            {"estimatedEarning": 1.0, "actualEarningResult": 0.9},  # MISS
            {"estimatedEarning": 1.0, "actualEarningResult": 1.5},  # beat (after miss, doesn't count)
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(session, "AAPL")
        assert result["beat_streak"] == 3

    async def test_quarterly_adjusted_eps_filters_non_numeric(self, fmp_key, primitives):
        # "N/A" must be filtered out by the quarterly_adjusted_eps loop, NOT crash.
        #
        # NOTE: the upstream beat_streak loop (line 697-703) does NOT guard
        # against non-numeric actualEarningResult — it raises TypeError on
        # `actual > estimated` when actual is a string. The sync sibling
        # (data_fetcher.py:956) has the same flaw but is wrapped in try/except
        # that silently returns {}. The async version has no such guard.
        # Real FMP data only emits "N/A" AFTER missed earnings (so the
        # beat_streak loop has already broken on the miss), which is why
        # this hasn't blown up in production. Test data below mirrors that
        # ordering: clean miss BEFORE the "N/A" — beat_streak breaks at the
        # miss, then quarterly_adjusted_eps loop runs over all 4 records.
        # See `canslim-async-fetcher-na-latent-bug.md` for fix recipe.
        data = [
            {"estimatedEarning": 1.0, "actualEarningResult": 1.5},   # beat
            {"estimatedEarning": 1.0, "actualEarningResult": 0.9},   # MISS (breaks beat_streak loop)
            {"estimatedEarning": 1.0, "actualEarningResult": "N/A"}, # adjusted_eps loop must skip
            {"estimatedEarning": 1.0, "actualEarningResult": 0.8},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(session, "X")
        # beat_streak hit one beat then a miss → 1
        assert result["beat_streak"] == 1
        # quarterly_adjusted_eps filters "N/A" without exploding
        assert result["quarterly_adjusted_eps"] == [1.5, 0.9, 0.8]

    async def test_zero_estimated_avoids_div_by_zero(self, fmp_key, primitives):
        data = [{"estimatedEarning": 0, "actualEarningResult": 1.0}]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(session, "X")
        # latest_surprise_pct stays 0 (no div-by-zero)
        assert result["latest_surprise_pct"] == 0

    async def test_empty_response_returns_empty(self, fmp_key, primitives):
        session = MockSession([(200, [])])
        result = await async_data_fetcher.fetch_fmp_earnings_surprise_async(session, "X")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_analyst_estimates_async (revision_pct + trend bucket)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpAnalystEstimatesAsync:
    """Tier 1: revision_pct + estimate_revision_trend bucket — feeds N/I scoring."""

    async def test_no_api_key_returns_empty(self, monkeypatch, primitives):
        monkeypatch.setattr(async_data_fetcher, "FMP_API_KEY", "")
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(
            MockSession([]), "AAPL"
        )
        assert result == {}

    async def test_revision_up_trend(self, fmp_key, primitives):
        from datetime import date
        current_year = date.today().year
        # Current year EPS estimate is 10% higher than prior year → "up"
        data = [
            {"date": f"{current_year}-12-31", "epsAvg": 5.5, "numAnalystsEps": 10},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 5.0, "numAnalystsEps": 9},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "AAPL")
        assert result["eps_estimate_current"] == 5.5
        assert result["eps_estimate_prior"] == 5.0
        assert result["eps_estimate_revision_pct"] == pytest.approx(10.0)
        assert result["estimate_revision_trend"] == "up"

    async def test_revision_down_trend(self, fmp_key, primitives):
        from datetime import date
        current_year = date.today().year
        data = [
            {"date": f"{current_year}-12-31", "epsAvg": 4.0, "numAnalystsEps": 10},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 5.0, "numAnalystsEps": 9},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "X")
        assert result["eps_estimate_revision_pct"] == pytest.approx(-20.0)
        assert result["estimate_revision_trend"] == "down"

    async def test_revision_stable_when_within_5pct(self, fmp_key, primitives):
        from datetime import date
        current_year = date.today().year
        data = [
            {"date": f"{current_year}-12-31", "epsAvg": 5.05, "numAnalystsEps": 10},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 5.0, "numAnalystsEps": 9},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "X")
        assert result["estimate_revision_trend"] == "stable"

    async def test_falls_back_to_first_two_when_year_match_fails(self, fmp_key, primitives):
        # Neither item is for the current/prior year → falls back to data[0], data[1]
        data = [
            {"date": "2099-12-31", "epsAvg": 6.0, "numAnalystsEps": 5},
            {"date": "2098-12-31", "epsAvg": 5.0, "numAnalystsEps": 5},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "X")
        assert result["eps_estimate_current"] == 6.0
        assert result["eps_estimate_prior"] == 5.0

    async def test_malformed_dates_are_skipped(self, fmp_key, primitives):
        # Bad date strings must NOT raise — just be skipped over.
        from datetime import date
        current_year = date.today().year
        data = [
            {"date": "not-a-date", "epsAvg": 99.0, "numAnalystsEps": 1},
            {"date": f"{current_year}-12-31", "epsAvg": 5.5, "numAnalystsEps": 10},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 5.0, "numAnalystsEps": 9},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "X")
        # The malformed date is skipped; current and prior come from valid rows.
        assert result["eps_estimate_current"] == 5.5
        assert result["eps_estimate_prior"] == 5.0

    async def test_zero_prior_avoids_div_by_zero(self, fmp_key, primitives):
        from datetime import date
        current_year = date.today().year
        data = [
            {"date": f"{current_year}-12-31", "epsAvg": 5.0, "numAnalystsEps": 10},
            {"date": f"{current_year - 1}-12-31", "epsAvg": 0, "numAnalystsEps": 9},
        ]
        session = MockSession([(200, data)])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(session, "X")
        assert result["eps_estimate_revision_pct"] == 0
        assert result["estimate_revision_trend"] == "stable"

    async def test_too_few_records_returns_empty(self, fmp_key, primitives):
        # Only 1 record → len(data) < 2 → returns {}
        session = MockSession([(200, [{"date": "2026-01-01", "epsAvg": 1}])])
        result = await async_data_fetcher.fetch_fmp_analyst_estimates_async(
            MockSession([(200, [{"date": "2026-01-01", "epsAvg": 1}])]), "X"
        )
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Cache age + fallback tracker helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheAgeAndFallbackTracker:
    """Tier 2: get_cache_age_days, get_fallback_stats, reset_fallback_tracker."""

    def test_unknown_ticker_returns_999(self):
        # Sentinel value so callers gate fallback safely
        assert async_data_fetcher.get_cache_age_days("NEVER", "earnings") == 999

    def test_known_ticker_returns_age_in_days(self):
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache["AAPL"] = {
                "earnings": datetime.now() - timedelta(days=3),
            }
        age = async_data_fetcher.get_cache_age_days("AAPL", "earnings")
        # ~3 days, with sub-second drift acceptable
        assert 2.99 <= age <= 3.01

    def test_known_ticker_unknown_data_type_returns_999(self):
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache["AAPL"] = {
                "earnings": datetime.now() - timedelta(days=1),
            }
        assert async_data_fetcher.get_cache_age_days("AAPL", "revenue") == 999

    def test_get_fallback_stats_initially_empty(self):
        stats = async_data_fetcher.get_fallback_stats()
        assert stats["fallback_count"] == 0
        assert stats["no_data_count"] == 0
        assert stats["stocks_using_fallback"] == []
        assert stats["stocks_no_data"] == []

    def test_get_fallback_stats_reflects_tracker(self):
        async_data_fetcher._fallback_tracker["stocks_using_fallback"].add("AAPL")
        async_data_fetcher._fallback_tracker["stocks_no_data"].add("BAD")
        stats = async_data_fetcher.get_fallback_stats()
        assert stats["fallback_count"] == 1
        assert stats["no_data_count"] == 1
        assert "AAPL" in stats["stocks_using_fallback"]
        assert "BAD" in stats["stocks_no_data"]

    def test_reset_clears_tracker(self):
        async_data_fetcher._fallback_tracker["stocks_using_fallback"].add("AAPL")
        async_data_fetcher._fallback_tracker["stocks_no_data"].add("BAD")
        async_data_fetcher.reset_fallback_tracker()
        assert async_data_fetcher._fallback_tracker["stocks_using_fallback"] == set()
        assert async_data_fetcher._fallback_tracker["stocks_no_data"] == set()
        assert async_data_fetcher._fallback_tracker["last_reset"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Scan progress checkpoint persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestScanProgress:
    """Tier 2: save_scan_progress / load_scan_progress / clear_scan_progress.
    Used by fetch_stocks_batch_async to resume interrupted scans."""

    def test_save_then_load_roundtrip(self):
        async_data_fetcher.save_scan_progress(["AAPL", "MSFT"], scan_id="scan_1")
        result = async_data_fetcher.load_scan_progress(scan_id="scan_1")
        assert result == ["AAPL", "MSFT"]

    def test_load_with_no_checkpoint_returns_empty(self):
        # CHECKPOINT_FILE redirected to tmp_path (autouse) so no file exists.
        assert async_data_fetcher.load_scan_progress("any") == []

    def test_load_with_mismatched_scan_id_returns_empty(self):
        async_data_fetcher.save_scan_progress(["AAPL"], scan_id="scan_1")
        # Different scan_id → ignored (the saved file is for a different scan)
        assert async_data_fetcher.load_scan_progress(scan_id="scan_2") == []

    def test_load_ignores_checkpoint_older_than_one_hour(self):
        # Write a checkpoint, then re-write with a stale timestamp.
        async_data_fetcher.save_scan_progress(["AAPL"], scan_id="scan_1")
        ckpt_path = async_data_fetcher.CHECKPOINT_FILE
        old_ts = (datetime.now() - timedelta(hours=2)).isoformat()
        ckpt_path.write_text(json.dumps({
            "scan_id": "scan_1",
            "completed": ["AAPL"],
            "timestamp": old_ts,
        }))
        # Stale → returns [] (project rule: don't resume across long gaps)
        assert async_data_fetcher.load_scan_progress(scan_id="scan_1") == []

    def test_clear_progress_removes_file(self):
        async_data_fetcher.save_scan_progress(["AAPL"], scan_id="scan_1")
        assert async_data_fetcher.CHECKPOINT_FILE.exists()
        async_data_fetcher.clear_scan_progress()
        assert not async_data_fetcher.CHECKPOINT_FILE.exists()

    def test_clear_progress_handles_missing_file(self):
        # Should NOT raise even if no checkpoint exists.
        assert not async_data_fetcher.CHECKPOINT_FILE.exists()
        async_data_fetcher.clear_scan_progress()  # silent

    def test_save_handles_unwritable_directory(self, monkeypatch, tmp_path):
        # If save fails (e.g. read-only volume) → log warning, don't raise.
        bad_path = tmp_path / "no" / "such" / "dir" / "ckpt.json"
        # Make the parent inaccessible: point to a path under a regular file
        marker = tmp_path / "blocker"
        marker.write_text("x")
        # CHECKPOINT_FILE.parent.mkdir(exist_ok=True) — if parent is a file,
        # this would raise. The function catches Exception and logs a warning.
        monkeypatch.setattr(async_data_fetcher, "CHECKPOINT_FILE", marker / "ckpt.json")
        async_data_fetcher.save_scan_progress(["AAPL"], scan_id="scan_x")  # silent


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — get_rate_limit_stats + StockData reuse
# ═══════════════════════════════════════════════════════════════════════════════


class TestRateLimitStats:
    """Tier 2: get_rate_limit_stats — passes through to fmp_rate_limiter."""

    def test_returns_centralized_dict_shape(self):
        stats = async_data_fetcher.get_rate_limit_stats()
        # Centralized stats include both metrics and circuit-breaker state
        assert "total_requests" in stats
        assert "errors_429" in stats
        assert "circuit_breaker" in stats
        assert "rate_429_pct" in stats

    def test_rate_limiter_legacy_keys_present(self):
        # Used as a backwards-compat anchor — code paths may still poke these.
        for key in ("calls_this_minute", "max_calls_per_minute",
                    "minute_start", "backoff_until", "consecutive_429s",
                    "total_429s", "total_calls"):
            assert key in async_data_fetcher._rate_limiter


class TestStockDataClass:
    """Tier 2: StockData re-exported from data_fetcher — must remain importable."""

    def test_stock_data_imports_from_async_module(self):
        from async_data_fetcher import StockData
        d = StockData(ticker="AAPL")
        assert d.ticker == "AAPL"
        assert d.is_valid is False

    def test_cache_key_format_consistency(self):
        # Project rule: cache keys are ticker:data_type — sync + async must agree.
        ticker, data_type = "AAPL", "earnings"
        assert f"{ticker}:{data_type}" == "AAPL:earnings"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_earnings_calendar_async (sync-wrapper passthrough)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpEarningsCalendarAsync:
    """Tier 1: fetch_fmp_earnings_calendar_async — async wrapper around sync
    `fetch_fmp_earnings_calendar`. Must passthrough the sync result, including
    None → {}."""

    async def test_passes_through_sync_result(self, monkeypatch):
        sync_result = {"next_earnings": "2026-04-25", "beat_streak": 3}
        monkeypatch.setattr(data_fetcher, "fetch_fmp_earnings_calendar", lambda t: sync_result)
        result = await async_data_fetcher.fetch_fmp_earnings_calendar_async(MockSession([]), "AAPL")
        assert result == sync_result

    async def test_none_result_becomes_empty_dict(self, monkeypatch):
        # The sync function returns None when no API key / no data — the
        # async wrapper must coerce that to {} so downstream callers can
        # safely .get() into the result.
        monkeypatch.setattr(data_fetcher, "fetch_fmp_earnings_calendar", lambda t: None)
        result = await async_data_fetcher.fetch_fmp_earnings_calendar_async(MockSession([]), "X")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — fetch_short_interest_async (Yahoo wrapper)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchShortInterestAsync:
    """Tier 2: fetch_short_interest_async — decimal-to-percent normalization
    (matches the same logic in data_fetcher.fetch_short_interest)."""

    async def test_decimal_normalized_to_percent(self, monkeypatch):
        # Yahoo returns 0.15 (decimal) → must be normalized to 15%
        monkeypatch.setattr(
            async_data_fetcher, "yf_safe_call",
            lambda *_a, **_k: {"shortPercentOfFloat": 0.15, "shortRatio": 2.5},
        )
        result = await async_data_fetcher.fetch_short_interest_async("AAPL")
        assert result["short_interest_pct"] == pytest.approx(15.0)
        assert result["short_ratio"] == 2.5

    async def test_already_percent_left_alone(self, monkeypatch):
        # Yahoo sometimes returns >3.0 for heavily-shorted (already pct) — leave it.
        monkeypatch.setattr(
            async_data_fetcher, "yf_safe_call",
            lambda *_a, **_k: {"shortPercentOfFloat": 25.0, "shortRatio": 5.0},
        )
        result = await async_data_fetcher.fetch_short_interest_async("HEAVYSHORT")
        assert result["short_interest_pct"] == 25.0

    async def test_zero_short_interest(self, monkeypatch):
        monkeypatch.setattr(
            async_data_fetcher, "yf_safe_call",
            lambda *_a, **_k: {"shortPercentOfFloat": 0, "shortRatio": 0},
        )
        result = await async_data_fetcher.fetch_short_interest_async("X")
        assert result["short_interest_pct"] == 0
        assert result["short_ratio"] == 0

    async def test_no_info_returns_empty(self, monkeypatch):
        monkeypatch.setattr(async_data_fetcher, "yf_safe_call", lambda *_a, **_k: None)
        result = await async_data_fetcher.fetch_short_interest_async("X")
        assert result == {}

    async def test_none_values_zeroed(self, monkeypatch):
        # NaN-safe: None values must become 0, not propagate
        monkeypatch.setattr(
            async_data_fetcher, "yf_safe_call",
            lambda *_a, **_k: {"shortPercentOfFloat": None, "shortRatio": None},
        )
        result = await async_data_fetcher.fetch_short_interest_async("X")
        assert result["short_interest_pct"] == 0
        assert result["short_ratio"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — get_price_data_only (sync ETF helper)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetPriceDataOnly:
    """Tier 2: get_price_data_only — synchronous price-only fetcher used by
    growth_projector for ETFs/indexes (no fundamentals)."""

    def test_no_chart_data_returns_invalid_stock(self, monkeypatch):
        monkeypatch.setattr(async_data_fetcher, "fetch_price_from_chart_api",
                            lambda t: {})
        result = async_data_fetcher.get_price_data_only("SPY")
        assert result.ticker == "SPY"
        assert result.is_valid is False

    def test_with_50plus_bars_marks_valid(self, monkeypatch):
        # 60 daily bars (>= 50 threshold required by the function) with
        # synthetic timestamps spanning 60 days back from now.
        now = int(datetime.now().timestamp())
        timestamps = [now - (60 - i) * 86400 for i in range(60)]
        chart = {
            "current_price": 500.0,
            "high_52w": 520.0,
            "low_52w": 400.0,
            "market_cap": 0,  # ETFs typically have no market_cap
            "name": "SPDR S&P 500",
            "close_prices": [500.0 + i for i in range(60)],
            "volumes": [1_000_000] * 60,
            "timestamps": timestamps,
        }
        monkeypatch.setattr(async_data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)
        result = async_data_fetcher.get_price_data_only("SPY")
        assert result.current_price == 500.0
        assert result.is_valid is True
        assert result.high_52w == 520.0
        assert result.avg_volume_50d == 1_000_000

    def test_short_history_does_not_populate_dataframe(self, monkeypatch):
        # Fewer than 50 bars → price_history stays empty, but current_price
        # is still set (the function still returns useful state).
        chart = {
            "current_price": 500.0, "high_52w": 0, "low_52w": 0, "market_cap": 0,
            "name": "X",
            "close_prices": [500.0] * 10,
            "volumes": [1] * 10,
            "timestamps": [int(datetime.now().timestamp())] * 10,
        }
        monkeypatch.setattr(async_data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)
        result = async_data_fetcher.get_price_data_only("X")
        assert result.current_price == 500.0
        # is_valid requires both price AND non-empty price_history
        assert result.is_valid is False

    def test_bad_timestamps_logged_but_not_raised(self, monkeypatch):
        # Garbage timestamps must trigger the ValueError/OverflowError except —
        # function returns the stock_data without a price_history but doesn't crash.
        chart = {
            "current_price": 500.0, "high_52w": 0, "low_52w": 0, "market_cap": 0,
            "name": "X",
            "close_prices": [500.0] * 60,
            "volumes": [1] * 60,
            "timestamps": [10**20] * 60,  # Out-of-range for to_datetime(unit='s')
        }
        monkeypatch.setattr(async_data_fetcher, "fetch_price_from_chart_api",
                            lambda t: chart)
        result = async_data_fetcher.get_price_data_only("X")  # must not raise
        assert result.current_price == 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_yahoo_info_comprehensive_async (high-volume Yahoo path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchYahooInfoComprehensiveAsync:
    """Tier 1: fetch_yahoo_info_comprehensive_async — the consolidated Yahoo
    info call. Replaces multiple FMP calls (key_metrics, balance_sheet,
    analyst, short_interest) for every stock in the live scan.

    Key invariants under test:
      - decimal → percent normalization for short_pct (< 3.0 only) and inst_pct
      - earnings_growth_estimate decimal-vs-pct heuristic (abs(growth) <= 1)
      - ROIC fallback (= ROE * 0.8 when ROE > 0)
      - early-return when info has no regularMarketPrice (treats as no data)
      - last-resort cached-data fallback after retries fail
    """

    async def test_successful_path_populates_all_fields(self, monkeypatch, primitives, no_delist):
        # Patch sleep so the per-attempt 0.3s delay doesn't pad test runtime
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())

        info = {
            "regularMarketPrice": 150.0,
            "returnOnEquity": 0.25, "returnOnAssets": 0.15,
            "trailingPE": 20.0, "freeCashflow": 1e9, "marketCap": 2e10,
            "totalCash": 5e9, "totalDebt": 1e9,
            "targetMeanPrice": 175.0, "targetHighPrice": 200, "targetLowPrice": 150,
            "numberOfAnalystOpinions": 25,
            "earningsQuarterlyGrowth": 0.30,  # decimal — must * 100
            "shortPercentOfFloat": 0.05,      # decimal — must * 100
            "shortRatio": 2.5,
            "heldPercentInstitutions": 0.65,  # decimal — must * 100
        }
        monkeypatch.setattr(async_data_fetcher, "yf_safe_call",
                            lambda *_a, **_k: info)

        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async("AAPL")
        assert result["success"] is True
        assert result["roe"] == 0.25
        assert result["roic"] == pytest.approx(0.20)  # ROE * 0.8
        assert result["earnings_yield"] == pytest.approx(1 / 20.0)
        assert result["fcf_yield"] == pytest.approx(1e9 / 2e10)
        assert result["cash_and_equivalents"] == 5e9
        assert result["analyst_target_price"] == 175.0
        assert result["num_analyst_opinions"] == 25
        # Decimal → percent for the three suspect fields
        assert result["earnings_growth_estimate"] == pytest.approx(30.0)
        assert result["short_interest_pct"] == pytest.approx(5.0)
        assert result["institutional_holders_pct"] == pytest.approx(65.0)

    async def test_string_typed_numerics_do_not_discard_payload(
        self, monkeypatch, primitives, no_delist
    ):
        # Live regression 2026-07-03: Yahoo intermittently returns numeric
        # info fields as STRINGS (BILL failed every scan with "'>' not
        # supported between instances of 'str' and 'int'") — one bad field
        # threw in the comparison chain and the broad except discarded the
        # ENTIRE payload. All numerics now coerce via _yf_num.
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        info = {
            "regularMarketPrice": 150.0,
            "returnOnEquity": "0.25",           # string decimal
            "returnOnAssets": "N/A",            # garbage → 0
            "trailingPE": "20.0",
            "freeCashflow": "1000000000",
            "marketCap": 2e10,
            "totalCash": "Infinity",            # non-finite → 0
            "totalDebt": None,
            "targetMeanPrice": "175.0",
            "targetHighPrice": 200, "targetLowPrice": 150,
            "numberOfAnalystOpinions": "25",
            "earningsQuarterlyGrowth": "0.30",
            "shortPercentOfFloat": "0.05",
            "shortRatio": "2.5",
            "heldPercentInstitutions": "0.65",
        }
        monkeypatch.setattr(async_data_fetcher, "yf_safe_call",
                            lambda *_a, **_k: info)

        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async("BILL")
        assert result["success"] is True          # payload NOT discarded
        assert result["roe"] == pytest.approx(0.25)
        assert result["roa"] == 0                 # garbage coerced to default
        assert result["earnings_yield"] == pytest.approx(1 / 20.0)
        assert result["fcf_yield"] == pytest.approx(1e9 / 2e10)
        assert result["cash_and_equivalents"] == 0  # Infinity → default
        assert result["analyst_target_price"] == 175.0
        assert result["num_analyst_opinions"] == 25
        assert result["earnings_growth_estimate"] == pytest.approx(30.0)
        assert result["short_interest_pct"] == pytest.approx(5.0)
        assert result["institutional_holders_pct"] == pytest.approx(65.0)

    async def test_short_pct_above_3_left_alone(self, monkeypatch, primitives, no_delist):
        # Yahoo can return >3.0 for heavily-shorted (already pct, not decimal).
        # The conditional `if 0 < short_pct < 3.0` MUST NOT touch these.
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        info = {
            "regularMarketPrice": 50.0,
            "shortPercentOfFloat": 25.0,  # already pct
            "shortRatio": 5.0,
            "heldPercentInstitutions": 1.5,  # 150% (some Yahoo data sources do this)
        }
        monkeypatch.setattr(async_data_fetcher, "yf_safe_call",
                            lambda *_a, **_k: info)
        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async("X")
        # short_interest_pct stays 25.0 (not 25 * 100 = 2500)
        assert result["short_interest_pct"] == 25.0
        # institutional 1.5 IS in the 0-3.0 window, so it normalizes to 150
        assert result["institutional_holders_pct"] == pytest.approx(150.0)

    async def test_no_regular_market_price_returns_no_success(self, monkeypatch, primitives, no_delist):
        # Yahoo returned data but no regularMarketPrice → treated as no data.
        # Function falls through retries and returns the default-zero result
        # (success=False) directly on the last attempt. Does NOT reach the
        # LAST-RESORT cached-fallback path (that's gated on an exception
        # escaping _fetch_yahoo_info, which the inner try/except suppresses
        # for non-rate-limit errors).
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        monkeypatch.setattr(async_data_fetcher, "yf_safe_call",
                            lambda *_a, **_k: {"someOtherField": 1})
        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async("Y")
        assert result.get("success") is False

    async def test_uses_cached_fallback_when_retries_fail(self, monkeypatch, primitives, no_delist):
        # The cached-fallback path is reached ONLY when an exception escapes
        # _fetch_yahoo_info (which only rate-limit errors do). So simulate a
        # rate-limit storm — every attempt raises 429 → outer for-loop breaks
        # → LAST RESORT cache check runs.
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        ticker = "CACHEDONLY"

        def raise_rate_limit(*_a, **_k):
            raise RuntimeError("429 too many requests")

        monkeypatch.setattr(async_data_fetcher, "yf_safe_call", raise_rate_limit)

        cached = {
            "success": True, "roe": 0.20, "roa": 0.10, "roic": 0.16,
            "cash_and_equivalents": 1e9, "total_debt": 5e8,
            "analyst_target_price": 100, "num_analyst_opinions": 8,
            "short_interest_pct": 5.0, "short_ratio": 2.0,
            "institutional_holders_pct": 70.0,
        }
        data_fetcher.set_cached_data(ticker, "yahoo_info", cached, persist_to_db=False)
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache[ticker] = {
                "yahoo_info": datetime.now() - timedelta(days=3),
            }

        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async(ticker)
        assert result["success"] is True
        assert result["roe"] == 0.20
        assert result["used_cache_fallback"] is True
        assert ticker in async_data_fetcher._fallback_tracker["stocks_using_fallback"]

    async def test_stale_cache_skipped_returns_no_success(self, monkeypatch, primitives, no_delist):
        # Same exception-escape preamble — but cache is too old (15d > 7d gate).
        # Function returns {"success": False, "cache_too_old": True, "cache_age_days": ...}
        # WITHOUT merging the stale field values into the result.
        monkeypatch.setattr(async_data_fetcher.asyncio, "sleep", AsyncMock())
        ticker = "STALEYAHOO"

        def raise_rate_limit(*_a, **_k):
            raise RuntimeError("429 too many requests")

        monkeypatch.setattr(async_data_fetcher, "yf_safe_call", raise_rate_limit)
        cached = {"success": True, "roe": 999.0}  # bad-but-success — must not surface
        data_fetcher.set_cached_data(ticker, "yahoo_info", cached, persist_to_db=False)
        with data_fetcher._freshness_lock:
            data_fetcher._data_freshness_cache[ticker] = {
                "yahoo_info": datetime.now() - timedelta(days=15),
            }
        result = await async_data_fetcher.fetch_yahoo_info_comprehensive_async(ticker)
        assert result.get("success") is False
        assert result.get("cache_too_old") is True
        # Stale fallback is NOT applied — the 999.0 must NOT leak into result
        assert result.get("roe") != 999.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — exception paths in checkpoint helpers (load/clear)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScanProgressExceptionPaths:
    """Tier 2: load_scan_progress / clear_scan_progress exception handling."""

    def test_load_with_corrupt_json_returns_empty(self):
        # If checkpoint file exists but is corrupt JSON, function logs & returns []
        async_data_fetcher.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        async_data_fetcher.CHECKPOINT_FILE.write_text("{not valid json")
        assert async_data_fetcher.load_scan_progress("any") == []

    def test_clear_progress_handles_unlink_error(self, monkeypatch):
        # If unlink fails (e.g. permission error) → caught + logged, no raise.
        async_data_fetcher.CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        async_data_fetcher.CHECKPOINT_FILE.write_text("{}")

        # Force unlink to raise OSError
        original_unlink = Path.unlink

        def bad_unlink(self, *a, **k):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "unlink", bad_unlink)
        async_data_fetcher.clear_scan_progress()  # silent — must not raise
        # Restore so the next test's autouse fixture can clean up
        monkeypatch.setattr(Path, "unlink", original_unlink)
