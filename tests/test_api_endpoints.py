"""
Integration tests for new API endpoints.

Tests response structure and parameter validation.
Uses FastAPI dependency_overrides to bypass auth.
"""
import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db, User
from backend.auth import get_current_active_user
from tests.conftest import override_dependency

# Create a fake user that bypasses all auth
_fake_user = User(id=1, email="test@test.com", display_name="Test",
                  is_active=True, is_admin=True, hashed_password="")


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    """Scoped auth bypass — see tests/conftest.py:override_dependency."""
    with override_dependency(get_current_active_user, _fake_user):
        yield


init_db()
client = TestClient(app)


class TestTradeJournalEndpoint:
    def test_returns_structure(self):
        r = client.get("/api/trade-journal?days=30")
        assert r.status_code == 200
        d = r.json()
        assert "entries" in d and "total" in d and "period_days" in d

    def test_filters_by_buy(self):
        assert client.get("/api/trade-journal?action=BUY").status_code == 200

    def test_rejects_invalid_action(self):
        assert client.get("/api/trade-journal?action=INVALID").status_code == 422

    def test_short_window(self):
        r = client.get("/api/trade-journal?days=7")
        assert r.status_code == 200
        d = r.json()
        assert "entries" in d
        assert "total" in d


class TestBearBaseEndpoint:
    def test_returns_structure(self):
        r = client.get("/api/bear-base?limit=10")
        assert r.status_code == 200
        d = r.json()
        assert "candidates" in d and "total" in d and "is_bear_market" in d

    def test_limit(self):
        assert len(client.get("/api/bear-base?limit=5").json()["candidates"]) <= 5


class TestIndustryGroupsEndpoint:
    def test_returns_structure(self):
        r = client.get("/api/industry-groups")
        assert r.status_code == 200
        d = r.json()
        assert "groups" in d and "total" in d and "rotation" in d


class TestCorrelationEndpoint:
    def test_empty_portfolio(self):
        r = client.get("/api/ai-portfolio/correlation")
        assert r.status_code == 200
        assert "matrix" in r.json() and "tickers" in r.json()


class TestEarningsGapupsEndpoint:
    def test_returns_structure(self):
        r = client.get("/api/earnings-gapups?days=7")
        assert r.status_code == 200
        d = r.json()
        assert "gapups" in d and "total" in d

    def test_validates_min_gap(self):
        assert client.get("/api/earnings-gapups?min_gap_pct=0.5").status_code == 422

    def test_defaults(self):
        assert client.get("/api/earnings-gapups").json()["days_lookback"] == 7
