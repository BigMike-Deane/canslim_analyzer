"""In-memory audit-log ring buffer for security-relevant rejections.

Records three kinds of events:
  - "rate_limit":  slowapi RateLimitExceeded (CSO #4 limiter is firing).
  - "auth_fail":   Google login rejected (invalid token / unknown user / etc).
  - "token_reuse": a spent refresh-token jti was replayed outside the
                   concurrency grace window (token family revoked).

The buffer is a bounded deque so memory cannot grow unboundedly. Events are
PII-scrubbed by callers — record only the failure REASON, never the email
or token contents.

Single-process uvicorn is fine for v1. If we ever go multi-worker, swap the
deque for Redis (one-line change in record_event).
"""
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

_MAX_EVENTS = 500  # ~24h at 20 events/hr; deque drops oldest on overflow

EventKind = Literal["rate_limit", "auth_fail", "token_reuse"]

_events: "deque[dict]" = deque(maxlen=_MAX_EVENTS)
_lock = Lock()


def record_event(kind: EventKind, *, source_ip: str, route: str, detail: str = "") -> None:
    """Append an event to the ring buffer. Caller is responsible for PII scrub."""
    with _lock:
        _events.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "source_ip": source_ip or "unknown",
            "route": route,
            "detail": detail,
        })


def recent_events(limit: int = 100) -> list[dict]:
    """Return up to `limit` most-recent events (newest last)."""
    with _lock:
        snapshot = list(_events)
    if limit <= 0:
        return []
    return snapshot[-limit:]


def buffer_size() -> int:
    """Max capacity of the ring buffer (for /api/admin/audit-log payload)."""
    return _MAX_EVENTS


def _reset_for_tests() -> None:
    """Clear the buffer. Tests-only — keeps each test independent."""
    with _lock:
        _events.clear()
