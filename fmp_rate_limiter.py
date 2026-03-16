"""
Centralized FMP API rate limiter with circuit breaker, exponential backoff, and metrics.

All FMP API calls (async scanner, sync data fetcher, MCP server) should flow through
this module for consistent rate limiting and failure handling.
"""

import asyncio
import logging
import random
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults (overridden by config_loader at init time)
# ---------------------------------------------------------------------------
_config = {
    "target_rate": 250,        # requests per minute target (FMP limit is 300)
    "burst_size": 10,          # max tokens in bucket
    "failure_threshold": 5,    # consecutive 429s to trip circuit breaker
    "base_cooldown": 30,       # seconds initial cooldown when tripped
    "max_cooldown": 300,       # seconds max cooldown (5 min)
    "backoff_base": 2.0,       # seconds base delay for exponential backoff
    "backoff_max": 60.0,       # seconds max delay
    "max_retries": 3,          # retries per request on 429
}


def init(config_overrides: Optional[dict] = None):
    """Initialize rate limiter with config from YAML.

    Call once at app startup. Safe to call multiple times.
    """
    if config_overrides:
        _config.update(config_overrides)
    _governor.reconfigure(_config["target_rate"], _config["burst_size"])
    _breaker.reconfigure(_config["failure_threshold"], _config["base_cooldown"], _config["max_cooldown"])
    logger.info(
        f"FMP rate limiter initialized: {_config['target_rate']}/min, "
        f"burst={_config['burst_size']}, circuit_threshold={_config['failure_threshold']}"
    )


# ---------------------------------------------------------------------------
# Token Bucket Rate Governor
# ---------------------------------------------------------------------------
class RateGovernor:
    """Token bucket rate limiter — controls request flow to stay under FMP limits."""

    def __init__(self, max_per_minute: int = 250, burst_size: int = 10):
        self._tokens: float = float(burst_size)
        self._max_tokens: float = float(burst_size)
        self._refill_rate: float = max_per_minute / 60.0  # tokens per second
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

    def reconfigure(self, max_per_minute: int, burst_size: int):
        with self._lock:
            self._refill_rate = max_per_minute / 60.0
            self._max_tokens = float(burst_size)
            # Don't reset tokens — let current state drain/refill naturally

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def acquire_sync(self) -> float:
        """Block until a token is available. Returns time waited (seconds)."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._refill_rate
            time.sleep(wait_time)
            waited += wait_time

    async def acquire_async(self) -> float:
        """Async version — awaits until a token is available."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        waited = 0.0
        while True:
            async with self._async_lock:
                with self._lock:
                    self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return waited
                    wait_time = (1.0 - self._tokens) / self._refill_rate
            await asyncio.sleep(wait_time)
            waited += wait_time


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Trips after N consecutive 429s, enters cooldown, then probes."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, base_cooldown: float = 30, max_cooldown: float = 300):
        self.state: str = self.CLOSED
        self.consecutive_failures: int = 0
        self.failure_threshold: int = failure_threshold
        self.base_cooldown: float = base_cooldown
        self.max_cooldown: float = max_cooldown
        self._cooldown_until: Optional[float] = None  # monotonic time
        self._trip_count: int = 0
        self._lock = threading.Lock()

    def reconfigure(self, failure_threshold: int, base_cooldown: float, max_cooldown: float):
        with self._lock:
            self.failure_threshold = failure_threshold
            self.base_cooldown = base_cooldown
            self.max_cooldown = max_cooldown

    def record_success(self):
        """Reset on success."""
        with self._lock:
            if self.state == self.HALF_OPEN:
                logger.info("Circuit breaker: HALF_OPEN → CLOSED (probe succeeded)")
            self.state = self.CLOSED
            self.consecutive_failures = 0
            self._cooldown_until = None

    def record_failure(self) -> bool:
        """Record a 429. Returns True if circuit just tripped to OPEN."""
        with self._lock:
            self.consecutive_failures += 1

            if self.state == self.HALF_OPEN:
                # Probe failed — re-trip with escalated cooldown
                self._trip_count += 1
                cooldown = min(self.base_cooldown * (2 ** self._trip_count), self.max_cooldown)
                self._cooldown_until = time.monotonic() + cooldown
                self.state = self.OPEN
                logger.warning(f"Circuit breaker: HALF_OPEN → OPEN (probe failed, cooldown={cooldown:.0f}s)")
                return True

            if self.consecutive_failures >= self.failure_threshold and self.state == self.CLOSED:
                self._trip_count += 1
                cooldown = min(self.base_cooldown * (2 ** min(self._trip_count, 5)), self.max_cooldown)
                self._cooldown_until = time.monotonic() + cooldown
                self.state = self.OPEN
                logger.warning(
                    f"Circuit breaker TRIPPED: {self.consecutive_failures} consecutive 429s, "
                    f"cooldown={cooldown:.0f}s (trip #{self._trip_count})"
                )
                return True
            return False

    def can_proceed(self) -> bool:
        """Check if requests are allowed. Transitions OPEN → HALF_OPEN when cooldown expires."""
        with self._lock:
            if self.state == self.CLOSED:
                return True

            if self.state == self.OPEN:
                if self._cooldown_until and time.monotonic() >= self._cooldown_until:
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker: OPEN → HALF_OPEN (cooldown expired, allowing probe)")
                    return True
                return False

            # HALF_OPEN: allow one probe
            return True

    def get_state(self) -> dict:
        """Get current circuit breaker state for monitoring."""
        with self._lock:
            remaining = 0.0
            if self._cooldown_until and self.state == self.OPEN:
                remaining = max(0, self._cooldown_until - time.monotonic())
            return {
                "state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "trip_count": self._trip_count,
                "cooldown_remaining_s": round(remaining, 1),
            }


# ---------------------------------------------------------------------------
# Backoff Calculator
# ---------------------------------------------------------------------------
def calculate_backoff(attempt: int, base_delay: float = 2.0, max_delay: float = 60.0) -> float:
    """Exponential backoff with full jitter to prevent thundering herd.

    Returns random delay in [0, min(max_delay, base * 2^attempt)].
    """
    delay = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0, delay)


# ---------------------------------------------------------------------------
# Unified Metrics
# ---------------------------------------------------------------------------
class FMPMetrics:
    """Thread-safe metrics for all FMP API usage across all paths."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.errors_429: int = 0
        self.errors_other: int = 0
        self.circuit_breaker_rejections: int = 0
        self.cache_fallbacks: int = 0
        self.total_wait_time: float = 0.0
        self.last_429_at: Optional[datetime] = None
        self.last_reset: datetime = datetime.now(timezone.utc)

    def record_success(self, wait_time: float = 0.0):
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.total_wait_time += wait_time

    def record_429(self):
        with self._lock:
            self.total_requests += 1
            self.errors_429 += 1
            self.last_429_at = datetime.now(timezone.utc)

    def record_other_error(self):
        with self._lock:
            self.total_requests += 1
            self.errors_other += 1

    def record_circuit_rejection(self):
        with self._lock:
            self.circuit_breaker_rejections += 1

    def record_cache_fallback(self):
        with self._lock:
            self.cache_fallbacks += 1

    def to_dict(self) -> dict:
        with self._lock:
            rate = 0.0
            if self.total_requests > 0:
                rate = round(self.errors_429 / self.total_requests * 100, 1)
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "errors_429": self.errors_429,
                "errors_other": self.errors_other,
                "rate_429_pct": rate,
                "circuit_breaker_rejections": self.circuit_breaker_rejections,
                "cache_fallbacks": self.cache_fallbacks,
                "total_wait_time_s": round(self.total_wait_time, 1),
                "last_429_at": self.last_429_at.isoformat() if self.last_429_at else None,
                "last_reset": self.last_reset.isoformat(),
            }

    def reset(self):
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.errors_429 = 0
            self.errors_other = 0
            self.circuit_breaker_rejections = 0
            self.cache_fallbacks = 0
            self.total_wait_time = 0.0
            self.last_429_at = None
            self.last_reset = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_governor = RateGovernor()
_breaker = CircuitBreaker()
_metrics = FMPMetrics()


def get_metrics() -> FMPMetrics:
    """Get the global metrics instance."""
    return _metrics


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    return _breaker


def get_governor() -> RateGovernor:
    """Get the global rate governor instance."""
    return _governor


# ---------------------------------------------------------------------------
# Convenience accessors for backward compatibility
# ---------------------------------------------------------------------------
def get_rate_limit_stats() -> dict:
    """Returns stats dict compatible with existing API endpoints."""
    stats = _metrics.to_dict()
    stats["circuit_breaker"] = _breaker.get_state()
    return stats


def reset_rate_limit_stats():
    """Reset metrics counters."""
    _metrics.reset()


# ---------------------------------------------------------------------------
# Pre-request gate (used by data fetchers)
# ---------------------------------------------------------------------------
async def acquire_async() -> float:
    """Acquire permission to make an async FMP request.

    Returns wait time in seconds. Raises CircuitBreakerOpen if circuit is open.
    """
    if not _breaker.can_proceed():
        _metrics.record_circuit_rejection()
        raise CircuitBreakerOpen("FMP circuit breaker is OPEN — too many 429s")
    return await _governor.acquire_async()


def acquire_sync() -> float:
    """Acquire permission to make a sync FMP request.

    Returns wait time in seconds. Raises CircuitBreakerOpen if circuit is open.
    """
    if not _breaker.can_proceed():
        _metrics.record_circuit_rejection()
        raise CircuitBreakerOpen("FMP circuit breaker is OPEN — too many 429s")
    return _governor.acquire_sync()


def record_success(wait_time: float = 0.0):
    """Record a successful FMP API response."""
    _breaker.record_success()
    _metrics.record_success(wait_time)


def record_429():
    """Record a 429 response. May trip the circuit breaker."""
    _breaker.record_failure()
    _metrics.record_429()


def record_error():
    """Record a non-429 error."""
    _metrics.record_other_error()


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""
    pass
