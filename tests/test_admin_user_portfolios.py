"""
Admin per-user portfolio scoreboard.

Two things this endpoint must get right, both learned the hard way:

1. TEST ACCOUNTS MUST NOT POLLUTE AGGREGATES. Real and test books live in the
   same tables, and unfiltered aggregates over them have leaked twice -- the
   gate-card u2 leak (fixed 67f96f3) and a stop-loss count that read 7 rows
   when only 2 belonged to the owner. Production currently has a live account
   literally named "Mike (TEST)".

2. RANK BY ALPHA, NEVER RAW RETURN. Accounts start on different dates and
   launch vintage is worth roughly 7pp/month of sigma on this strategy -- the
   confound behind three separate false cohort reads. Each book is measured
   against SPY over exactly its own span.
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
    AIPortfolioConfig, AIPortfolioSnapshot, MarketSnapshot,
)
from backend.auth import get_current_active_user, get_admin_user
from tests.conftest import override_dependency

REAL_ID = 99010
TEST_ID = 99011

_admin = User(
    id=REAL_ID, email="scoreboard@acct.example.org", display_name="Real Account",
    is_active=True, is_admin=True, hashed_password="",
)


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    with override_dependency(get_current_active_user, _admin), \
         override_dependency(get_admin_user, _admin):
        yield


init_db()
client = TestClient(app)

START_DAYS_AGO = 120
SPY_START = 100.0
SPY_NOW = 110.0          # SPY +10% over the window


def _seed():
    db = SessionLocal()
    try:
        for uid, name, cash in (
            (REAL_ID, "Real Account", 12500.0),   # +25% from 10000
            (TEST_ID, "Someone (TEST)", 9000.0),
        ):
            if not db.query(User).filter_by(id=uid).first():
                db.add(User(id=uid, email=f"u{uid}@acct.example.org", display_name=name,
                            is_active=True, is_admin=False, hashed_password=""))
            db.query(AIPortfolioConfig).filter_by(user_id=uid).delete()
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
        db.commit()

        for uid, cash in ((REAL_ID, 12500.0), (TEST_ID, 9000.0)):
            db.add(AIPortfolioConfig(
                user_id=uid, starting_cash=10000.0, current_cash=cash,
                is_active=True,
            ))
            db.add(AIPortfolioSnapshot(
                user_id=uid,
                timestamp=datetime.now(timezone.utc) - timedelta(days=START_DAYS_AGO),
                date=date.today() - timedelta(days=START_DAYS_AGO),
                total_value=10000.0, cash=10000.0, positions_value=0.0,
                positions_count=0, total_return=0.0, total_return_pct=0.0,
            ))

        for days_ago, price in ((START_DAYS_AGO, SPY_START), (0, SPY_NOW)):
            d = date.today() - timedelta(days=days_ago)
            row = db.query(MarketSnapshot).filter_by(date=d).first()
            if row:
                row.spy_price = price
            else:
                db.add(MarketSnapshot(date=d, spy_price=price))
        db.commit()
    finally:
        db.close()


def _fetch(include_test=False):
    url = "/api/admin/user-portfolios"
    if include_test:
        url += "?include_test=true"
    r = client.get(url)
    assert r.status_code == 200, r.text
    return r.json()


def _row(payload, uid):
    for u in payload["users"]:
        if u["user_id"] == uid:
            return u
    return None


class TestTestAccountsAreExcluded:

    def test_hidden_by_default(self):
        _seed()
        payload = _fetch()
        assert _row(payload, REAL_ID) is not None
        assert _row(payload, TEST_ID) is None, (
            "a (TEST) account leaked into the default scoreboard"
        )
        assert payload["excluded_test_accounts"] >= 1

    def test_visible_when_explicitly_requested(self):
        _seed()
        payload = _fetch(include_test=True)
        row = _row(payload, TEST_ID)
        assert row is not None
        assert row["is_test"] is True

    def test_real_account_not_flagged(self):
        _seed()
        assert _row(_fetch(), REAL_ID)["is_test"] is False


class TestAlphaIsTheRankingNumber:

    def test_alpha_is_return_minus_spy_over_the_same_window(self):
        _seed()
        row = _row(_fetch(), REAL_ID)
        # Book: 10000 -> 12500 = +25%. SPY: 100 -> 110 = +10%.
        assert row["return_pct"] == pytest.approx(25.0, abs=0.01)
        assert row["spy_return_pct"] == pytest.approx(10.0, abs=0.01)
        assert row["alpha_pp"] == pytest.approx(15.0, abs=0.02)

    def test_sorted_by_alpha_descending(self):
        _seed()
        payload = _fetch(include_test=True)
        alphas = [u["alpha_pp"] for u in payload["users"] if u["alpha_pp"] is not None]
        assert alphas == sorted(alphas, reverse=True)


class TestSmallSampleHonesty:
    """A three-week account is measuring its launch window, not the strategy."""

    def test_long_window_no_trades_is_still_low_sample(self):
        _seed()
        # 120 days but zero closed trades -> not comparable.
        assert _row(_fetch(), REAL_ID)["low_sample"] is True

    def test_short_window_is_low_sample(self):
        db = SessionLocal()
        try:
            db.query(AIPortfolioSnapshot).filter_by(user_id=REAL_ID).delete()
            db.add(AIPortfolioSnapshot(
                user_id=REAL_ID,
                timestamp=datetime.now(timezone.utc) - timedelta(days=5),
                date=date.today() - timedelta(days=5),
                total_value=10000.0, cash=10000.0, positions_value=0.0,
                positions_count=0, total_return=0.0, total_return_pct=0.0,
            ))
            db.commit()
        finally:
            db.close()
        row = _row(_fetch(), REAL_ID)
        assert row["days_active"] == 5
        assert row["low_sample"] is True


class TestTheTestHeuristicIsNotOverEager:
    """Hiding a REAL account is worse than showing a test one.

    The first version substring-matched "test", which also matched the
    surname "Testa" and every address at a *test*.com domain -- a real
    account would have silently vanished from the scoreboard.
    """

    @pytest.mark.parametrize("name", ["Testa", "Protester", "Contested Ltd"])
    def test_real_names_containing_test_are_not_flagged(self, name):
        uid = 99012
        db = SessionLocal()
        try:
            db.query(AIPortfolioConfig).filter_by(user_id=uid).delete()
            db.query(AIPortfolioSnapshot).filter_by(user_id=uid).delete()
            existing = db.query(User).filter_by(id=uid).first()
            if existing:
                existing.display_name = name
            else:
                db.add(User(id=uid, email="real@acct.example.org",
                            display_name=name, is_active=True,
                            is_admin=False, hashed_password=""))
            db.add(AIPortfolioConfig(
                user_id=uid, starting_cash=10000.0, current_cash=10000.0,
                is_active=True,
            ))
            db.commit()
        finally:
            db.close()

        row = _row(_fetch(), uid)
        assert row is not None, f"a real account named {name!r} was hidden"
        assert row["is_test"] is False

    def test_parenthesised_marker_is_flagged(self):
        # Production's actual convention: "Mike (TEST)".
        uid = 99013
        db = SessionLocal()
        try:
            db.query(AIPortfolioConfig).filter_by(user_id=uid).delete()
            if not db.query(User).filter_by(id=uid).first():
                db.add(User(id=uid, email="m@acct.example.org",
                            display_name="Mike (TEST)", is_active=True,
                            is_admin=False, hashed_password=""))
            db.add(AIPortfolioConfig(
                user_id=uid, starting_cash=10000.0, current_cash=10000.0,
                is_active=True,
            ))
            db.commit()
        finally:
            db.close()
        assert _row(_fetch(), uid) is None
        assert _row(_fetch(include_test=True), uid)["is_test"] is True
