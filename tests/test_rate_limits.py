"""Rate-limit regression suite for slowapi-decorated routes.

Locks in the per-route policies wired up in CSO #4:
- /health                       30/min
- POST /api/auth/google         10/min
- POST /api/auth/refresh        20/min
- /api/admin/strategy-ab-eval   30/min
- /api/admin/strategy-ab-eval-trades  30/min
- everything else               60/min default

The slowapi limiter is stateful and shared across the whole app.state, so
each test calls `limiter.reset()` before exercising endpoints. Without that
the in-memory counter bleeds between tests in the same session and any
non-trivial ordering produces false 429s.

We also lock in the TrustForwardedFor middleware: if the limiter were to
key on Caddy's loopback IP instead of the real client IP, every browser
on the internet would share one bucket and the limits would be useless.
"""

import os

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db, SessionLocal, User
from backend.auth import get_current_active_user, get_admin_user
from backend.rate_limiter import limiter


ADMIN_ID = 90


def _admin_for_request(request: Request):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == ADMIN_ID).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Test fixture: admin missing")
        db.expunge(user)
        return user
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def _setup_admin_and_overrides():
    init_db()
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.id == ADMIN_ID).first():
            db.add(User(
                id=ADMIN_ID, email="ratelimit-admin@test.com",
                display_name="ratelimit-admin", is_active=True,
                is_admin=True, hashed_password="",
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
def _reset_limiter():
    # conftest.py disables the limiter for the whole session so tests that
    # call decorated handlers directly (asyncio.run(handler(...))) don't
    # crash. Re-enable here so this file actually exercises the limiter.
    prev_enabled = limiter.enabled
    limiter.enabled = True
    limiter.reset()
    yield
    limiter.reset()
    limiter.enabled = prev_enabled


@pytest.fixture
def client():
    return TestClient(app)


class TestPerRouteLimits:
    """Each decorated route fires 429 at the documented threshold."""

    def test_health_endpoint_caps_at_thirty_per_minute(self, client):
        codes = [client.get("/health").status_code for _ in range(31)]
        assert codes.count(200) == 30
        assert codes[-1] == 429
        # Sanity: nothing else (e.g., 500) leaked through.
        assert sum(1 for c in codes if c not in (200, 429)) == 0

    def test_auth_google_caps_at_ten_per_minute(self, client):
        # We do not stub verify_google_token — the request body is invalid
        # for any real verifier, so these calls return 401/4xx well before
        # rate limiting kicks in for the first ten. The 11th must be 429
        # regardless of body validity, because slowapi runs before the
        # handler body. That's the whole defense-in-depth point.
        body = {"credential": "not-a-real-jwt"}
        codes = [
            client.post("/api/auth/google", json=body).status_code
            for _ in range(11)
        ]
        # First 10 hit the handler (and fail auth), 11th is rate-limited.
        assert codes[-1] == 429
        assert sum(1 for c in codes if c == 429) == 1

    def test_auth_refresh_caps_at_twenty_per_minute(self, client):
        body = {"refresh_token": "not-a-real-jwt"}
        codes = [
            client.post("/api/auth/refresh", json=body).status_code
            for _ in range(21)
        ]
        assert codes[-1] == 429
        assert sum(1 for c in codes if c == 429) == 1

    def test_strategy_ab_eval_caps_at_thirty_per_minute(self, client):
        # Use a strategy/cutoff that the endpoint will reject quickly
        # — we just want to count handler invocations.
        params = {"strategy": "nostate_optimized", "cutoff_date": "2099-01-01"}
        codes = [
            client.get("/api/admin/strategy-ab-eval", params=params).status_code
            for _ in range(31)
        ]
        assert codes[-1] == 429
        assert sum(1 for c in codes if c == 429) == 1


class TestDefaultLimit:
    """Routes without an explicit decorator inherit 60/minute."""

    def test_default_limit_is_sixty_per_minute(self, client):
        # Use an authenticated read route that has no per-route decorator.
        # /api/admin/users is admin-only and not decorated, so it inherits
        # the 60/min default from the limiter.
        codes = [
            client.get("/api/admin/users").status_code for _ in range(61)
        ]
        assert codes[-1] == 429
        assert sum(1 for c in codes if c == 429) == 1


class TestForwardedForKeying:
    """TrustForwardedFor middleware re-keys the limiter on the real client IP.

    Without this, every request behind Caddy looks like 127.0.0.1 and the
    bucket is shared across the entire internet. We prove keying works by
    saturating one XFF, then asserting a different XFF still gets through.
    """

    def test_distinct_xff_addresses_get_distinct_buckets(self, client):
        # Saturate the bucket for IP A (31 requests on /health → 30 ok + 1 429).
        ip_a = {"x-forwarded-for": "203.0.113.10"}
        codes_a = [
            client.get("/health", headers=ip_a).status_code for _ in range(31)
        ]
        assert codes_a.count(200) == 30
        assert codes_a[-1] == 429

        # IP B should still have its full quota.
        ip_b = {"x-forwarded-for": "203.0.113.99"}
        codes_b = [
            client.get("/health", headers=ip_b).status_code for _ in range(5)
        ]
        assert codes_b == [200] * 5

    def test_xff_chain_uses_first_address(self, client):
        # If XFF carries a chain (client, proxy1, proxy2), the limiter
        # must key on the first entry — the original client.
        chain = {"x-forwarded-for": "198.51.100.5, 10.0.0.1, 10.0.0.2"}
        codes = [
            client.get("/health", headers=chain).status_code for _ in range(31)
        ]
        assert codes[-1] == 429

        # A different first hop is still under quota.
        other_chain = {"x-forwarded-for": "198.51.100.99, 10.0.0.1"}
        assert client.get("/health", headers=other_chain).status_code == 200
