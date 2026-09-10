"""
Live-account windows start at INCEPTION, not at the first snapshot (2026-09-10).

A paper book sits at exactly starting_cash from Initialize until its first
trade. The scoreboard and the vintage clock both measured from the first
snapshot, so that idle cash stretch counted as the account's own window:
SPY moved, the book could not, and alpha was taken over a span the strategy
never traded.

How it was found: the owner (first snapshot Mar-09) and Karstell (Mar-04)
read as a pair started 5 days apart and 8.4pp of alpha apart. Both actually
sat in cash until 2026-04-08 and made their first trades that same day --
same tickers, same reasons. From inception they are a same-day TWIN pair,
~7.2pp apart, and that spread is sizing/fill path noise, not launch vintage.

Same bug class as d7a5bfd (the go-live gate not trimming leading flat days).
The rule: any surface measuring a live book's window reuses the /edge
construction, trim included.
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import (
    init_db, SessionLocal, User,
    AIPortfolioConfig, AIPortfolioSnapshot, AIPortfolioTrade, MarketSnapshot,
)
from backend.auth import get_current_active_user, get_admin_user
from backend.routes.admin import _account_inception, compute_experiment_gates
from tests.conftest import override_dependency

OWNER = 1
TWIN = 99030            # same config, created 5 days before the owner

# Owner created 400d ago, twin 405d ago; BOTH first trade 360d ago.
OWNER_CREATED, TWIN_CREATED, FIRST_TRADE = 400, 405, 360
LAST_FLAT = FIRST_TRADE + 1

# SPY drifts DOWN through the cash-only stretch, then up: measuring from
# account creation flatters nothing and penalises nothing consistently --
# it just measures the wrong span. Prices chosen so the two windows differ
# by a lot, which makes the bug impossible to pass by rounding.
SPY = {TWIN_CREATED: 120.0, OWNER_CREATED: 118.0, LAST_FLAT: 100.0, 0: 130.0}

_admin = User(
    id=OWNER, email="inception@acct.example.org", display_name="Owner",
    is_active=True, is_admin=True, hashed_password="",
)

init_db()


def _snap(db, uid, days_ago, value):
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    db.add(AIPortfolioSnapshot(
        user_id=uid, timestamp=when, date=when.date(),
        total_value=value, cash=value, positions_value=0.0,
        positions_count=0, total_return=value - 25000.0,
        total_return_pct=(value / 25000.0 - 1) * 100,
    ))


def _seed():
    db = SessionLocal()
    try:
        for uid in (OWNER, TWIN):
            db.query(AIPortfolioConfig).filter_by(user_id=uid).delete()
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
            db.query(AIPortfolioTrade).filter_by(user_id=uid).delete()
            if not db.query(User).filter_by(id=uid).first():
                db.add(User(id=uid, email=f"u{uid}@acct.example.org",
                            display_name=f"acct {uid}", is_active=True,
                            is_admin=uid == OWNER, hashed_password=""))
        db.commit()

        for uid, created, end_cash in ((OWNER, OWNER_CREATED, 32500.0),
                                       (TWIN, TWIN_CREATED, 34000.0)):
            db.add(AIPortfolioConfig(
                user_id=uid, starting_cash=25000.0, current_cash=end_cash,
                is_active=True, strategy="nostate_cs_bear",
                min_score_to_buy=72, stop_loss_pct=8.0, max_positions=8,
            ))
            # Flat, cash-only from creation up to the day before first trade.
            for d in range(created, FIRST_TRADE, -1):
                _snap(db, uid, d, 25000.0)
            # Deployed from the first-trade day on.
            _snap(db, uid, FIRST_TRADE, 25180.0)
            _snap(db, uid, 1, end_cash)

        for days_ago, price in SPY.items():
            d = date.today() - timedelta(days=days_ago)
            row = db.query(MarketSnapshot).filter_by(date=d).first()
            if row:
                row.spy_price = price
            else:
                db.add(MarketSnapshot(date=d, spy_price=price))
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def seeded():
    _seed()
    yield


class TestInceptionHelper:

    def test_starts_on_the_last_flat_day_not_the_first_snapshot(self, seeded):
        db = SessionLocal()
        try:
            inc = _account_inception(db, OWNER)
        finally:
            db.close()
        assert inc["start_day"] == date.today() - timedelta(days=LAST_FLAT)
        assert inc["created_day"] == date.today() - timedelta(days=OWNER_CREATED)
        assert inc["start_value"] == 25000.0

    def test_a_book_that_never_traded_is_not_trimmed(self):
        db = SessionLocal()
        try:
            uid = 99031
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
            for d in (30, 20, 10):
                _snap(db, uid, d, 25000.0)
            db.commit()
            inc = _account_inception(db, uid)
            assert inc["start_day"] == date.today() - timedelta(days=30)
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
            db.commit()
        finally:
            db.close()

    def test_no_snapshots_returns_nones(self):
        db = SessionLocal()
        try:
            assert _account_inception(db, 99039) == {
                "start_day": None, "start_value": None, "created_day": None}
        finally:
            db.close()


class TestScoreboardMeasuresFromInception:

    @pytest.fixture
    def row(self, seeded):
        client = TestClient(app)
        with override_dependency(get_current_active_user, _admin), \
             override_dependency(get_admin_user, _admin):
            r = client.get("/api/admin/user-portfolios?include_test=true")
        assert r.status_code == 200, r.text
        return next(u for u in r.json()["users"] if u["user_id"] == OWNER)

    def test_spy_window_starts_at_first_trade(self, row):
        # SPY 100 (last flat day) -> 130 now = +30%. From creation (118) it
        # would read +10.17% -- the bug.
        assert row["spy_return_pct"] == pytest.approx(30.0, abs=0.01)

    def test_started_on_is_inception_and_creation_is_kept_visible(self, row):
        assert row["started_on"] == (date.today() - timedelta(days=LAST_FLAT)).isoformat()
        assert row["created_on"] == (date.today() - timedelta(days=OWNER_CREATED)).isoformat()
        assert row["days_active"] == LAST_FLAT


class TestTwinAccountsAreNamedAsPathNoise:

    @pytest.fixture
    def vintage(self, seeded):
        db = SessionLocal()
        try:
            return compute_experiment_gates(db)["program_clocks"]["vintage_spread"]
        finally:
            db.close()

    def test_twins_share_a_start_despite_different_creation_dates(self, vintage):
        live = {s["label"]: s for s in vintage["stacks"] if s.get("kind") == "live_account"}
        assert live[f"live u{OWNER}"]["activated_at"] == live[f"live u{TWIN}"]["activated_at"]
        assert live[f"live u{OWNER}"]["days"] == live[f"live u{TWIN}"]["days"] == LAST_FLAT

    def test_cohort_is_labelled_path_noise(self, vintage):
        c = vintage["cohort"]
        assert set(c["labels"]) == {f"live u{OWNER}", f"live u{TWIN}"}
        assert c["same_start"] is True
        assert c["measures"].startswith("path noise")
