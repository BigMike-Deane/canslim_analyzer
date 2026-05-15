"""
Coverage for /api/ai-portfolio/window-returns (Time-Range Returns slicer).

Endpoint computes portfolio + per-position return % for 1D/7D/30D/All
windows used by the AI Portfolio page slicer. Tests lock down:

  - Portfolio anchor selection (latest snapshot strictly before window start)
  - Per-position fallback to cost_basis when position opened mid-window
  - Per-position historical price path (mocked HistoricalDataProvider)
  - "all" window matches current lifetime cost_basis behavior
  - Graceful response with no positions / no snapshots
  - Query-param validation (regex on `window`)

User-scoped via TEST_USER_B_ID (99003) so we don't collide with the
spy_overlay test module (which uses TEST_USER_A_ID).
"""

import os
import pytest
from datetime import datetime, date, timedelta, timezone
from unittest.mock import patch

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db, SessionLocal, User,
    AIPortfolioConfig, AIPortfolioPosition, AIPortfolioSnapshot,
)
from backend.auth import get_current_active_user, get_admin_user
from tests.conftest import override_dependency, TEST_USER_B_ID


_fake_user = User(
    id=TEST_USER_B_ID, email="window-returns@test.com",
    display_name="Window Returns Test", is_active=True, is_admin=True,
    hashed_password="",
)


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    with override_dependency(get_current_active_user, _fake_user), \
         override_dependency(get_admin_user, _fake_user):
        yield


init_db()
client = TestClient(app)


def _db():
    return SessionLocal()


def _ensure_user_and_config(starting_cash=25000.0, current_cash=None):
    """Seed user + config. `current_cash` controls the *live* cash component
    of portfolio.total_value (= current_cash + positions_value), which is
    what `get_portfolio_value()` returns — independent of snapshot rows."""
    if current_cash is None:
        current_cash = starting_cash
    db = _db()
    try:
        if not db.query(User).filter_by(id=TEST_USER_B_ID).first():
            db.add(User(
                id=TEST_USER_B_ID, email="window-returns@test.com",
                display_name="Window Returns Test", is_active=True,
                is_admin=True, hashed_password="",
            ))
        # Replace config so tests are deterministic
        db.query(AIPortfolioConfig).filter_by(user_id=TEST_USER_B_ID).delete()
        db.add(AIPortfolioConfig(
            user_id=TEST_USER_B_ID,
            starting_cash=starting_cash,
            current_cash=current_cash,
            is_active=True,
        ))
        db.commit()
    finally:
        db.close()


def _wipe():
    db = _db()
    try:
        db.query(AIPortfolioSnapshot).filter_by(user_id=TEST_USER_B_ID).delete()
        db.query(AIPortfolioPosition).filter_by(user_id=TEST_USER_B_ID).delete()
        db.commit()
    finally:
        db.close()


def _seed_snapshot(day_offset, total_value):
    db = _db()
    try:
        d = date.today() - timedelta(days=day_offset)
        ts = datetime.now(timezone.utc) - timedelta(days=day_offset)
        db.add(AIPortfolioSnapshot(
            user_id=TEST_USER_B_ID, timestamp=ts, date=d,
            total_value=total_value, cash=1000.0,
            positions_value=total_value - 1000.0, positions_count=1,
            total_return=total_value - 25000.0,
            total_return_pct=((total_value - 25000.0) / 25000.0) * 100,
        ))
        db.commit()
    finally:
        db.close()


def _seed_position(ticker, shares, cost_basis, current_price,
                   purchase_day_offset=30):
    db = _db()
    try:
        purchase = datetime.now(timezone.utc) - timedelta(days=purchase_day_offset)
        gain_loss = (current_price - cost_basis) * shares
        gain_loss_pct = ((current_price - cost_basis) / cost_basis) * 100
        db.add(AIPortfolioPosition(
            user_id=TEST_USER_B_ID, ticker=ticker, shares=shares,
            cost_basis=cost_basis, current_price=current_price,
            current_value=current_price * shares,
            gain_loss=gain_loss, gain_loss_pct=gain_loss_pct,
            purchase_date=purchase,
        ))
        db.commit()
    finally:
        db.close()


# ── Tests ─────────────────────────────────────────────────────────────


class TestWindowReturnsAll:
    """`window=all` uses earliest snapshot for portfolio + cost_basis per
    position. This is the existing lifetime-return behavior."""

    def test_no_positions_no_snapshots(self):
        _ensure_user_and_config(starting_cash=10000.0)
        _wipe()
        r = client.get("/api/ai-portfolio/window-returns?window=all")
        assert r.status_code == 200
        body = r.json()
        assert body["window"] == "all"
        assert body["positions"] == []
        # No snapshots yet → falls back to starting cash; current = starting cash
        assert body["portfolio"]["start_value"] == 10000.0

    def test_uses_cost_basis_per_position(self):
        # current_cash=10400 + NVDA position_value=1100 → live total=$11500.
        _ensure_user_and_config(starting_cash=10000.0, current_cash=10400.0)
        _wipe()
        _seed_snapshot(day_offset=30, total_value=10000.0)
        _seed_snapshot(day_offset=0,  total_value=11500.0)
        # NVDA bought 30 days ago at $100, now $110
        _seed_position("NVDA", shares=10, cost_basis=100.0,
                       current_price=110.0, purchase_day_offset=30)

        r = client.get("/api/ai-portfolio/window-returns?window=all")
        assert r.status_code == 200
        body = r.json()
        assert body["window"] == "all"
        # Portfolio: live total $11500 vs earliest snap $10000 → 15%
        assert body["portfolio"]["return_pct"] == 15.0
        # Position: 10% — based on cost_basis, no historical fetch
        pos = body["positions"][0]
        assert pos["ticker"] == "NVDA"
        assert pos["source"] == "cost_basis"
        assert pos["return_pct"] == 10.0
        assert pos["start_price"] == 100.0


class TestWindowReturnsWithHistorical:
    """1D/7D/30D windows use HistoricalDataProvider.get_price_on_date."""

    def test_position_held_through_window_uses_historical(self):
        # current_cash=9900 + AAPL position_value=1100 → live total=$11000.
        _ensure_user_and_config(starting_cash=10000.0, current_cash=9900.0)
        _wipe()
        # Snapshots: 30d ago = $10k, 8d ago = $10500 (anchor for 7D),
        # today = $11k
        _seed_snapshot(day_offset=30, total_value=10000.0)
        _seed_snapshot(day_offset=8,  total_value=10500.0)
        _seed_snapshot(day_offset=0,  total_value=11000.0)
        # AAPL bought 30 days ago at $200, now $220
        _seed_position("AAPL", shares=5, cost_basis=200.0,
                       current_price=220.0, purchase_day_offset=30)

        # Mock historical: 7 days ago AAPL closed at $210
        with patch(
            "backend.historical_data.HistoricalDataProvider.preload_data",
            return_value=True,
        ), patch(
            "backend.historical_data.HistoricalDataProvider.get_price_on_date",
            return_value=210.0,
        ):
            r = client.get("/api/ai-portfolio/window-returns?window=7d")

        assert r.status_code == 200
        body = r.json()
        # Portfolio: anchor is the snapshot 8d ago ($10500) — strictly
        # before window_start (7d ago). (11000-10500)/10500 ≈ 4.76%
        assert body["window"] == "7d"
        assert body["portfolio"]["start_value"] == 10500.0
        assert body["portfolio"]["return_pct"] == pytest.approx(4.76, abs=0.01)
        # Position: 220 vs 210 = 4.76%
        pos = body["positions"][0]
        assert pos["source"] == "historical"
        assert pos["start_price"] == 210.0
        assert pos["return_pct"] == pytest.approx(4.76, abs=0.01)

    def test_position_opened_mid_window_falls_back_to_cost_basis(self):
        _ensure_user_and_config(starting_cash=10000.0)
        _wipe()
        _seed_snapshot(day_offset=10, total_value=10000.0)
        _seed_snapshot(day_offset=0,  total_value=10300.0)
        # TSLA bought 3 days ago — INSIDE the 7-day window
        _seed_position("TSLA", shares=4, cost_basis=300.0,
                       current_price=315.0, purchase_day_offset=3)

        # Even though historical returns a price, we should ignore it for
        # mid-window positions (you didn't own it 7 days ago).
        with patch(
            "backend.historical_data.HistoricalDataProvider.preload_data",
            return_value=True,
        ), patch(
            "backend.historical_data.HistoricalDataProvider.get_price_on_date",
            return_value=999.0,  # should NOT be used
        ):
            r = client.get("/api/ai-portfolio/window-returns?window=7d")

        assert r.status_code == 200
        pos = r.json()["positions"][0]
        assert pos["ticker"] == "TSLA"
        assert pos["source"] == "cost_basis"
        assert pos["start_price"] == 300.0  # cost basis, not 999
        assert pos["notes"] == ["opened mid-window"]
        assert pos["return_pct"] == pytest.approx(5.0, abs=0.001)

    def test_no_historical_data_falls_back_to_cost_basis(self):
        _ensure_user_and_config(starting_cash=10000.0)
        _wipe()
        _seed_snapshot(day_offset=10, total_value=10000.0)
        _seed_snapshot(day_offset=0,  total_value=10500.0)
        _seed_position("ZZZZ", shares=2, cost_basis=50.0,
                       current_price=55.0, purchase_day_offset=30)

        with patch(
            "backend.historical_data.HistoricalDataProvider.preload_data",
            return_value=True,
        ), patch(
            "backend.historical_data.HistoricalDataProvider.get_price_on_date",
            return_value=None,  # historical data missing for this ticker
        ):
            r = client.get("/api/ai-portfolio/window-returns?window=7d")

        assert r.status_code == 200
        pos = r.json()["positions"][0]
        assert pos["source"] == "cost_basis"
        assert pos["start_price"] == 50.0
        assert pos["notes"] == ["no historical data"]
        assert pos["return_pct"] == pytest.approx(10.0, abs=0.001)


class TestWindowReturnsValidation:
    """The Query regex rejects unknown windows."""

    def test_invalid_window_rejected(self):
        _ensure_user_and_config()
        r = client.get("/api/ai-portfolio/window-returns?window=42d")
        assert r.status_code == 422

    def test_1d_window_accepted(self):
        _ensure_user_and_config()
        _wipe()
        r = client.get("/api/ai-portfolio/window-returns?window=1d")
        assert r.status_code == 200
        assert r.json()["window"] == "1d"


class TestWindowReturnsAnchorFallback:
    """When window predates portfolio inception, fall back to earliest
    snapshot rather than returning null (preserves user-visible UX)."""

    def test_window_before_inception_uses_earliest(self):
        # current_cash=11000, no positions → live total=$11000.
        _ensure_user_and_config(starting_cash=10000.0, current_cash=11000.0)
        _wipe()
        # Only one snapshot, 2 days old. 30D window predates it.
        _seed_snapshot(day_offset=2, total_value=10500.0)
        _seed_snapshot(day_offset=0, total_value=11000.0)

        r = client.get("/api/ai-portfolio/window-returns?window=30d")
        assert r.status_code == 200
        body = r.json()
        # No pre-window snapshot exists → falls back to earliest snap
        assert body["portfolio"]["start_value"] == 10500.0
        assert body["portfolio"]["return_pct"] == pytest.approx(4.76, abs=0.01)
