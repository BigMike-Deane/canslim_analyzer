"""
Coverage for /api/ai-portfolio/history SPY benchmark overlay (commit 6205293).

Locks down spy_value behavior the frontend depends on:
  - null when no MarketSnapshot rows exist in the window
  - anchored to first snapshot's day; later days scale by spy_price ratio
  - carry-forward on weekend / scanner-gap days
  - drops MarketSnapshot rows where spy_price IS NULL
  - regression: first snapshot day == start_date_only edge case
    (the MarketSnapshot.date >= start_date_only filter must look back
    far enough to find an anchor even when day-zero has no row)

User-scoped via TEST_USER_A_ID (99002) to avoid collision with
test_main_coverage's user_id=1 seed. Each test cleans up the
MarketSnapshot date range it created — that table has unique(date)
and is shared across test modules.
"""

import os
import pytest
from datetime import datetime, date, timedelta, timezone

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db, SessionLocal, User, AIPortfolioSnapshot, MarketSnapshot,
)
from backend.auth import get_current_active_user, get_admin_user
from tests.conftest import override_dependency, TEST_USER_A_ID


_fake_user = User(
    id=TEST_USER_A_ID, email="spy-overlay@test.com",
    display_name="SPY Overlay Test", is_active=True, is_admin=True,
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


def _ensure_user():
    db = _db()
    try:
        if not db.query(User).filter_by(id=TEST_USER_A_ID).first():
            db.add(User(
                id=TEST_USER_A_ID, email="spy-overlay@test.com",
                display_name="SPY Overlay Test", is_active=True,
                is_admin=True, hashed_password="",
            ))
            db.commit()
    finally:
        db.close()


def _wipe_window(days_back=40):
    """Wipe AIPortfolioSnapshot (for our user) and ALL MarketSnapshot rows
    inside today-days_back .. today, so other test files' seeds can't
    leak SPY data into our window. MarketSnapshot is shared app-wide;
    wiping it here is fine because tests in this file always re-seed
    what they need and clean up after themselves.
    """
    db = _db()
    try:
        cutoff = date.today() - timedelta(days=days_back)
        db.query(AIPortfolioSnapshot).filter(
            AIPortfolioSnapshot.user_id == TEST_USER_A_ID,
            AIPortfolioSnapshot.date >= cutoff,
        ).delete(synchronize_session=False)
        db.query(MarketSnapshot).filter(
            MarketSnapshot.date >= cutoff,
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _seed_snapshot(day_offset, total_value):
    db = _db()
    try:
        d = date.today() - timedelta(days=day_offset)
        ts = datetime.now(timezone.utc) - timedelta(days=day_offset)
        db.add(AIPortfolioSnapshot(
            user_id=TEST_USER_A_ID, timestamp=ts, date=d,
            total_value=total_value, cash=1000.0,
            positions_value=total_value - 1000.0, positions_count=1,
            total_return=0.0, total_return_pct=0.0,
        ))
        db.commit()
    finally:
        db.close()


def _seed_market(day_offset, spy_price):
    """Idempotent: replaces any prior row on the same date."""
    db = _db()
    try:
        d = date.today() - timedelta(days=day_offset)
        existing = db.query(MarketSnapshot).filter_by(date=d).first()
        if existing:
            existing.spy_price = spy_price
        else:
            db.add(MarketSnapshot(date=d, spy_price=spy_price))
        db.commit()
    finally:
        db.close()


_ensure_user()


class TestSpyOverlay:

    def test_returns_empty_list_when_no_snapshots(self):
        _wipe_window()
        r = client.get("/api/ai-portfolio/history?days=30")
        assert r.status_code == 200
        assert r.json() == []

    def test_spy_value_is_none_when_no_market_snapshots(self):
        offsets = [10, 5, 1]
        _wipe_window()
        for o, v in zip(offsets, [20000.0, 21000.0, 22000.0]):
            _seed_snapshot(o, v)
        # Intentionally no MarketSnapshot rows in the window
        try:
            r = client.get("/api/ai-portfolio/history?days=30")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) == 3
            for row in rows:
                assert row["spy_value"] is None
        finally:
            _wipe_window()

    def test_spy_value_anchors_to_first_snapshot_day(self):
        # Day-10 anchor; days-10/5/1 each have MarketSnapshot
        offsets = [10, 5, 1]
        _wipe_window()
        for o, v in zip(offsets, [25000.0, 26000.0, 27000.0]):
            _seed_snapshot(o, v)
        for o, p in zip(offsets, [100.0, 110.0, 120.0]):
            _seed_market(o, p)
        try:
            r = client.get("/api/ai-portfolio/history?days=30")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) == 3
            # Sorted ascending by timestamp → day_offset=10 is rows[0]
            assert rows[0]["spy_value"] == round(25000.0 * (100.0 / 100.0), 2)
            assert rows[1]["spy_value"] == round(25000.0 * (110.0 / 100.0), 2)
            assert rows[2]["spy_value"] == round(25000.0 * (120.0 / 100.0), 2)
        finally:
            _wipe_window()

    def test_spy_value_carries_forward_on_gap_day(self):
        # Snapshots on days 10/9/8/7; MarketSnapshot only on 10 and 8.
        # Day 9 must carry-forward day 10's price; day 7 carries day 8's.
        snap_offsets = [10, 9, 8, 7]
        mkt_offsets = [10, 8]
        _wipe_window()
        for o, v in zip(snap_offsets, [30000.0, 30100.0, 30200.0, 30300.0]):
            _seed_snapshot(o, v)
        for o, p in zip(mkt_offsets, [100.0, 105.0]):
            _seed_market(o, p)
        try:
            r = client.get("/api/ai-portfolio/history?days=30")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) == 4
            # Anchor = 100 at day 10; base_value = 30000
            assert rows[0]["spy_value"] == 30000.0  # day 10 (exact)
            assert rows[1]["spy_value"] == 30000.0  # day 9 (carry-forward day 10)
            assert rows[2]["spy_value"] == round(30000.0 * (105.0 / 100.0), 2)  # day 8 exact
            assert rows[3]["spy_value"] == round(30000.0 * (105.0 / 100.0), 2)  # day 7 carry
        finally:
            _wipe_window()

    def test_spy_value_ignores_null_spy_price_rows(self):
        # MarketSnapshot exists on day 10 but with spy_price=None.
        # Should be treated as if the row didn't exist → spy_anchor=None
        # → all spy_values None.
        offsets = [10, 5]
        _wipe_window()
        for o, v in zip(offsets, [40000.0, 41000.0]):
            _seed_snapshot(o, v)
        _seed_market(10, None)
        _seed_market(5, None)
        try:
            r = client.get("/api/ai-portfolio/history?days=30")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) == 2
            for row in rows:
                assert row["spy_value"] is None
        finally:
            _wipe_window()

    def test_spy_anchor_falls_back_to_pre_window_market_snapshot(self):
        # Regression: first snapshot day == start_date_only and that exact
        # day has no MarketSnapshot. Pre-fix the MarketSnapshot query loads
        # only date >= start_date_only, so the "nearest earlier" fallback
        # finds nothing and spy_anchor=None silently kills the overlay.
        # Post-fix the query widens its lookback so a slightly earlier
        # MarketSnapshot can serve as the anchor.
        first_offset = 30  # equals start_date_only when days=30
        offsets = [first_offset, 20, 10]
        _wipe_window()
        for o, v in zip(offsets, [50000.0, 51000.0, 52000.0]):
            _seed_snapshot(o, v)
        # Seed MarketSnapshot a few days BEFORE start_date_only (offset=33)
        # and on the later snapshot days.
        _seed_market(33, 200.0)  # anchor candidate
        _seed_market(20, 210.0)
        _seed_market(10, 220.0)
        try:
            r = client.get("/api/ai-portfolio/history?days=30")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) == 3
            # After fix: anchor = 200 (day 33), base = 50000
            assert rows[0]["spy_value"] == round(50000.0 * (200.0 / 200.0), 2)
            assert rows[1]["spy_value"] == round(50000.0 * (210.0 / 200.0), 2)
            assert rows[2]["spy_value"] == round(50000.0 * (220.0 / 200.0), 2)
        finally:
            _wipe_window()
