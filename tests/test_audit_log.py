"""Tests for the in-memory audit-log ring buffer + GET /api/admin/audit-log.

Locks in:
- Module-level record/recent round-trip + ring-buffer overflow.
- Admin gate on the endpoint (non-admin gets 401, admin sees events).
- Query-param plumbing: `limit` and `kind` filter both work.
- The two real call sites: Google auth failure + slowapi RateLimitExceeded
  both produce an audit-log row.
- PII scrub: email submitted to /api/auth/google never appears in the buffer.
"""
import os
import pytest
from unittest.mock import patch

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db, SessionLocal, User
from backend.auth import get_current_active_user, get_admin_user
from backend.audit_log import record_event, recent_events, buffer_size, _reset_for_tests
from backend.rate_limiter import limiter


ADMIN_ID = 9001
NONADMIN_ID = 9002


def _admin_for_request(request: Request):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == ADMIN_ID).first()
        db.expunge(user)
        return user
    finally:
        db.close()


def _nonadmin_for_request(request: Request):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == NONADMIN_ID).first()
        db.expunge(user)
        return user
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def _setup_users_and_overrides():
    init_db()
    db = SessionLocal()
    try:
        for uid, email, is_admin in (
            (ADMIN_ID, "audit-admin@test.com", True),
            (NONADMIN_ID, "audit-user@test.com", False),
        ):
            if not db.query(User).filter(User.id == uid).first():
                db.add(User(
                    id=uid, email=email, display_name=email,
                    is_active=True, is_admin=is_admin, hashed_password="",
                ))
        db.commit()
    finally:
        db.close()

    prev_user = app.dependency_overrides.get(get_current_active_user)
    prev_admin = app.dependency_overrides.get(get_admin_user)
    app.dependency_overrides[get_current_active_user] = _admin_for_request
    app.dependency_overrides[get_admin_user] = _admin_for_request
    yield
    if prev_user is not None:
        app.dependency_overrides[get_current_active_user] = prev_user
    else:
        app.dependency_overrides.pop(get_current_active_user, None)
    if prev_admin is not None:
        app.dependency_overrides[get_admin_user] = prev_admin
    else:
        app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture(autouse=True)
def _reset_buffer():
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture
def client():
    return TestClient(app)


# ── Module-level (storage) ──────────────────────────────────────────────


def test_record_and_recent_round_trip():
    record_event("auth_fail", source_ip="1.2.3.4", route="/api/auth/google", detail="invalid_token")
    record_event("rate_limit", source_ip="5.6.7.8", route="/health", detail="30 per 1 minute")
    events = recent_events()
    assert len(events) == 2
    kinds = [e["kind"] for e in events]
    assert kinds == ["auth_fail", "rate_limit"]
    assert events[0]["source_ip"] == "1.2.3.4"
    assert events[0]["detail"] == "invalid_token"
    assert "ts" in events[0]


def test_ring_buffer_drops_oldest_beyond_capacity():
    cap = buffer_size()
    for i in range(cap + 25):
        record_event("auth_fail", source_ip="0.0.0.0", route="/x", detail=f"n{i}")
    events = recent_events(limit=cap + 25)
    # Oldest 25 dropped; we keep exactly `cap` and the last detail is n{cap+24}.
    assert len(events) == cap
    assert events[-1]["detail"] == f"n{cap + 24}"
    assert events[0]["detail"] == f"n{25}"


# ── HTTP endpoint ───────────────────────────────────────────────────────


def test_endpoint_requires_admin(client):
    """Swap the dependency override to a non-admin and expect a 401/403."""
    def _denying_admin(request: Request):
        # Mirror auth.get_admin_user behavior when current user is not admin.
        raise HTTPException(status_code=401, detail="Admin only")
    prev = app.dependency_overrides.get(get_admin_user)
    app.dependency_overrides[get_admin_user] = _denying_admin
    try:
        resp = client.get("/api/admin/audit-log")
        assert resp.status_code in (401, 403)
    finally:
        app.dependency_overrides[get_admin_user] = prev


def test_endpoint_returns_events_with_kind_filter_and_limit(client):
    record_event("auth_fail", source_ip="1.1.1.1", route="/api/auth/google", detail="invalid_token")
    record_event("rate_limit", source_ip="2.2.2.2", route="/health", detail="30 per 1 minute")
    record_event("auth_fail", source_ip="3.3.3.3", route="/api/auth/google", detail="unknown_user")

    # Default: all events, newest last.
    resp = client.get("/api/admin/audit-log")
    assert resp.status_code == 200
    body = resp.json()
    assert body["buffer_size"] == buffer_size()
    assert body["total"] == 3
    assert [e["kind"] for e in body["events"]] == ["auth_fail", "rate_limit", "auth_fail"]

    # kind filter narrows to two auth_fail rows.
    resp = client.get("/api/admin/audit-log?kind=auth_fail")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert all(e["kind"] == "auth_fail" for e in body["events"])

    # limit caps the read window (applied before kind filter, by design).
    resp = client.get("/api/admin/audit-log?limit=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["events"][0]["detail"] == "unknown_user"


# ── Real call sites ─────────────────────────────────────────────────────


def test_google_auth_invalid_token_records_pii_free_event(client):
    """A failed Google login produces ONE auth_fail row with no email payload."""
    leaky_email = "supersecret-do-not-leak@example.com"

    def _boom(_token):
        raise HTTPException(status_code=401, detail="Invalid Google token")

    with patch("backend.routes.auth.verify_google_token", side_effect=_boom):
        resp = client.post(
            "/api/auth/google",
            json={"credential": f"x-id-token-with-email-{leaky_email}"},
        )
    assert resp.status_code == 401

    events = recent_events()
    assert len(events) == 1
    e = events[0]
    assert e["kind"] == "auth_fail"
    assert e["route"] == "/api/auth/google"
    assert e["detail"] == "invalid_token"
    # PII scrub: the leaky email must not appear anywhere in the recorded row.
    serialized = repr(e)
    assert leaky_email not in serialized
    assert "credential" not in serialized


def test_rate_limit_handler_records_event(client):
    """Re-enable the limiter (conftest disables it) and overshoot /health.

    Exercises the actual exception_handler wiring so a refactor that
    forgets to call audit_log.record_event would fail this test.
    """
    prev_enabled = limiter.enabled
    limiter.enabled = True
    limiter.reset()
    try:
        # /health is capped at 30/min — the 31st should 429.
        codes = [client.get("/health").status_code for _ in range(31)]
        assert codes[-1] == 429
    finally:
        limiter.reset()
        limiter.enabled = prev_enabled

    rate_events = [e for e in recent_events() if e["kind"] == "rate_limit"]
    assert rate_events, "Rate-limit handler did not record an audit event"
    e = rate_events[-1]
    assert e["route"] == "/health"
    assert e["source_ip"]  # populated, not empty


def test_rate_limit_records_real_ip_via_xff(client):
    """Behind Caddy, the real client IP arrives in X-Forwarded-For.

    The TrustForwardedFor middleware rewrites request.scope["client"] from
    the first XFF value, so the rate-limit handler records the real client
    IP rather than the reverse-proxy loopback. Closes the smoke gap from
    the CSO #6 ship: prod smoke saw the Docker bridge IP because the test
    request bypassed Caddy.
    """
    real_ip = "203.0.113.42"  # TEST-NET-3 (RFC 5737), safe to hard-code

    prev_enabled = limiter.enabled
    limiter.enabled = True
    limiter.reset()
    try:
        codes = [
            client.get("/health", headers={"X-Forwarded-For": real_ip}).status_code
            for _ in range(31)
        ]
        assert codes[-1] == 429
    finally:
        limiter.reset()
        limiter.enabled = prev_enabled

    rate_events = [e for e in recent_events() if e["kind"] == "rate_limit"]
    assert rate_events, "Rate-limit handler did not record an audit event"
    assert rate_events[-1]["source_ip"] == real_ip
