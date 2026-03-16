"""Tests for the centralized FMP rate limiter module."""

import asyncio
import time
import threading
import pytest
from unittest.mock import patch

import fmp_rate_limiter
from fmp_rate_limiter import (
    RateGovernor,
    CircuitBreaker,
    FMPMetrics,
    calculate_backoff,
    CircuitBreakerOpen,
)


class TestRateGovernor:
    """Tests for the token bucket rate governor."""

    def test_acquire_sync_succeeds_immediately_with_tokens(self):
        gov = RateGovernor(max_per_minute=600, burst_size=5)
        waited = gov.acquire_sync()
        assert waited == 0.0

    def test_burst_allows_multiple_immediate_acquires(self):
        gov = RateGovernor(max_per_minute=600, burst_size=3)
        for _ in range(3):
            waited = gov.acquire_sync()
            assert waited == 0.0

    def test_burst_exhaustion_causes_wait(self):
        gov = RateGovernor(max_per_minute=60, burst_size=1)
        gov.acquire_sync()  # Uses the one token
        start = time.monotonic()
        gov.acquire_sync()  # Must wait for refill
        elapsed = time.monotonic() - start
        assert elapsed >= 0.5  # At 60/min = 1/sec, should wait ~1s for token

    def test_reconfigure_updates_rate(self):
        gov = RateGovernor(max_per_minute=60, burst_size=5)
        gov.reconfigure(max_per_minute=120, burst_size=10)
        assert gov._refill_rate == 2.0  # 120/60
        assert gov._max_tokens == 10.0

    @pytest.mark.asyncio
    async def test_acquire_async_succeeds(self):
        gov = RateGovernor(max_per_minute=600, burst_size=5)
        waited = await gov.acquire_async()
        assert waited == 0.0

    @pytest.mark.asyncio
    async def test_acquire_async_burst(self):
        gov = RateGovernor(max_per_minute=600, burst_size=3)
        for _ in range(3):
            waited = await gov.acquire_async()
            assert waited == 0.0


class TestCircuitBreaker:
    """Tests for the circuit breaker state machine."""

    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_trips_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        tripped = cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_success_resets_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.consecutive_failures == 0
        assert cb.state == CircuitBreaker.CLOSED

    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, base_cooldown=0.05, max_cooldown=1.0)
        cb.record_failure()  # Trips immediately (cooldown = 0.05 * 2^1 = 0.1s)
        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.15)
        assert cb.can_proceed() is True
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, base_cooldown=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()  # Transitions to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, base_cooldown=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()  # HALF_OPEN
        tripped = cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreaker.OPEN

    def test_escalating_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, base_cooldown=10, max_cooldown=300)
        cb.record_failure()  # Trip #1
        state1 = cb.get_state()
        # Cooldown should be base * 2^1 = 20s (first trip)
        assert state1["trip_count"] == 1

    def test_get_state_returns_dict(self):
        cb = CircuitBreaker(failure_threshold=3)
        state = cb.get_state()
        assert "state" in state
        assert "consecutive_failures" in state
        assert "trip_count" in state
        assert "cooldown_remaining_s" in state

    def test_reconfigure(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.reconfigure(failure_threshold=10, base_cooldown=60, max_cooldown=600)
        assert cb.failure_threshold == 10
        assert cb.base_cooldown == 60


class TestBackoff:
    """Tests for exponential backoff with jitter."""

    def test_backoff_increases_with_attempt(self):
        # With enough samples, higher attempts should produce higher avg delays
        low_delays = [calculate_backoff(0, base_delay=2.0) for _ in range(100)]
        high_delays = [calculate_backoff(3, base_delay=2.0) for _ in range(100)]
        assert sum(low_delays) / len(low_delays) < sum(high_delays) / len(high_delays)

    def test_backoff_respects_max(self):
        for _ in range(50):
            delay = calculate_backoff(10, base_delay=2.0, max_delay=30.0)
            assert delay <= 30.0

    def test_backoff_is_non_negative(self):
        for attempt in range(10):
            delay = calculate_backoff(attempt)
            assert delay >= 0.0

    def test_backoff_has_jitter(self):
        # Two calls with same params should produce different values (probabilistically)
        delays = {calculate_backoff(2, base_delay=2.0) for _ in range(20)}
        assert len(delays) > 1  # Should have variance


class TestFMPMetrics:
    """Tests for the unified metrics tracker."""

    def test_initial_state(self):
        m = FMPMetrics()
        d = m.to_dict()
        assert d["total_requests"] == 0
        assert d["errors_429"] == 0
        assert d["rate_429_pct"] == 0.0

    def test_record_success(self):
        m = FMPMetrics()
        m.record_success(wait_time=0.5)
        d = m.to_dict()
        assert d["total_requests"] == 1
        assert d["successful_requests"] == 1
        assert d["total_wait_time_s"] == 0.5

    def test_record_429(self):
        m = FMPMetrics()
        m.record_429()
        d = m.to_dict()
        assert d["errors_429"] == 1
        assert d["last_429_at"] is not None

    def test_rate_calculation(self):
        m = FMPMetrics()
        m.record_success()
        m.record_success()
        m.record_429()
        d = m.to_dict()
        assert d["rate_429_pct"] == pytest.approx(33.3, abs=0.1)

    def test_reset(self):
        m = FMPMetrics()
        m.record_success()
        m.record_429()
        m.reset()
        d = m.to_dict()
        assert d["total_requests"] == 0
        assert d["errors_429"] == 0

    def test_thread_safety(self):
        m = FMPMetrics()
        def increment():
            for _ in range(100):
                m.record_success()
        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert m.to_dict()["total_requests"] == 500


class TestModuleLevelFunctions:
    """Tests for the module-level convenience functions."""

    def test_init_configures_singletons(self):
        fmp_rate_limiter.init({"target_rate": 200, "burst_size": 5, "failure_threshold": 10})
        gov = fmp_rate_limiter.get_governor()
        assert gov._refill_rate == pytest.approx(200 / 60.0, abs=0.01)
        # Reset
        fmp_rate_limiter.init({"target_rate": 250, "burst_size": 10, "failure_threshold": 5})

    def test_get_rate_limit_stats_includes_circuit_breaker(self):
        stats = fmp_rate_limiter.get_rate_limit_stats()
        assert "circuit_breaker" in stats
        assert stats["circuit_breaker"]["state"] == "closed"

    def test_record_success_updates_metrics(self):
        fmp_rate_limiter.reset_rate_limit_stats()
        fmp_rate_limiter.record_success(0.1)
        stats = fmp_rate_limiter.get_rate_limit_stats()
        assert stats["successful_requests"] >= 1

    def test_record_429_updates_metrics(self):
        fmp_rate_limiter.reset_rate_limit_stats()
        fmp_rate_limiter.record_429()
        stats = fmp_rate_limiter.get_rate_limit_stats()
        assert stats["errors_429"] >= 1

    def test_circuit_breaker_open_raises(self):
        # Trip the circuit breaker
        breaker = fmp_rate_limiter.get_circuit_breaker()
        original_threshold = breaker.failure_threshold
        breaker.reconfigure(failure_threshold=1, base_cooldown=60, max_cooldown=300)
        breaker.consecutive_failures = 0
        breaker.state = CircuitBreaker.CLOSED
        breaker.record_failure()  # Should trip

        with pytest.raises(CircuitBreakerOpen):
            fmp_rate_limiter.acquire_sync()

        # Reset
        breaker.record_success()
        breaker.reconfigure(failure_threshold=original_threshold, base_cooldown=30, max_cooldown=300)


class TestMLPredictionCache:
    """Tests for the ML prediction cache in ml/model.py."""

    def test_clear_prediction_cache(self):
        from ml.model import clear_prediction_cache, get_prediction_cache_stats
        clear_prediction_cache()
        stats = get_prediction_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_prediction_cache_deduplicates(self):
        """Same features should return cached result."""
        from ml.model import get_ml_prediction, clear_prediction_cache, get_prediction_cache_stats

        clear_prediction_cache()

        # Make two identical predictions (model may not be loaded, so result could be None)
        features = dict(
            total_score=80, composite_score=75, entry_type=2,
            market_regime=1, estimate_revision_bonus=0,
            coiled_spring=0, soft_zone=0, soft_zone_multiplier=1.0,
            deterministic_boost=0,
        )
        result1 = get_ml_prediction(**features)
        result2 = get_ml_prediction(**features)

        assert result1 == result2
        stats = get_prediction_cache_stats()
        # If model is loaded: 1 miss + 1 hit. If not loaded: both return None (no cache interaction)
        if result1 is not None:
            assert stats["hits"] >= 1
            assert stats["misses"] >= 1


class TestBacktesterProfileOverrides:
    """Tests for the BacktestEngine profile override mechanism."""

    def test_deep_merge_simple(self):
        from backend.backtester import BacktestEngine
        base = {"a": 1, "b": 2}
        BacktestEngine._deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        from backend.backtester import BacktestEngine
        base = {"ml_signal": {"enabled": True, "log_only": True, "weight": 20}}
        BacktestEngine._deep_merge(base, {"ml_signal": {"log_only": False, "min_confidence": 0.5}})
        assert base["ml_signal"]["enabled"] is True  # Unchanged
        assert base["ml_signal"]["log_only"] is False  # Overridden
        assert base["ml_signal"]["min_confidence"] == 0.5  # Added
        assert base["ml_signal"]["weight"] == 20  # Unchanged

    def test_deep_merge_none_override_is_noop(self):
        from backend.backtester import BacktestEngine
        base = {"a": 1}
        original = base.copy()
        # Passing None should not call _deep_merge at all (handled in __init__)
        # Just verify the method doesn't crash with empty dict
        BacktestEngine._deep_merge(base, {})
        assert base == original
