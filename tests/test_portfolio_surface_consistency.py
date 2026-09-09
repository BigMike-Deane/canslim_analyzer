"""
One invariant, every portfolio surface: they must agree at BOTH ends.

Origin (2026-09-09). Three bugs shipped the same defect in three places,
because two different notions of "now" exist in this app:

  * the LIVE book   -- positions re-priced by every scan, round the clock
  * the SNAPSHOT series -- AIPortfolioSnapshot rows, written ONLY during
    market hours

Any surface that ends its window on the snapshot series disagrees with any
surface that ends on the live book, and the gap opens overnight and through
pre-market. Measured live that morning: the AI Portfolio 30D slicer read
-1.05% while the Command Center read -0.15% on the same book and the same
anchor, 0.90pp apart.

Every pre-existing test missed all three for the SAME reason: they seeded
`current_cash` equal to the last snapshot's `total_value`, so the two ends
were identical by construction and the bug was invisible. This module
deliberately seeds them APART, and asserts the surfaces agree anyway.

If you add a new surface that reports a portfolio value or a windowed
return, add it to `_collect_surfaces` -- that is the point of this file.
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
    AIPortfolioConfig, AIPortfolioPosition, AIPortfolioSnapshot,
)
from backend.auth import get_current_active_user, get_admin_user
from tests.conftest import override_dependency

# Own user id so this module cannot collide with the window-returns or
# spy-overlay suites, which share the same tables.
TEST_USER_C_ID = 99004

_fake_user = User(
    id=TEST_USER_C_ID, email="surface-consistency@test.com",
    display_name="Surface Consistency Test", is_active=True, is_admin=True,
    hashed_password="",
)


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    with override_dependency(get_current_active_user, _fake_user), \
         override_dependency(get_admin_user, _fake_user):
        yield


init_db()
client = TestClient(app)

STARTING_CASH = 10000.0
# The whole point: the live book must NOT equal the last snapshot.
LIVE_CASH = 10500.0
LAST_SNAPSHOT_VALUE = 11000.0
ANCHOR_VALUE = 10000.0


def _db():
    return SessionLocal()


def _seed(with_snapshot_today=False):
    """A book where the live value and the last snapshot deliberately differ.

    No positions, so `get_portfolio_value()` == current_cash exactly and the
    live number is pinned without touching price providers.
    """
    db = _db()
    try:
        if not db.query(User).filter_by(id=TEST_USER_C_ID).first():
            db.add(User(
                id=TEST_USER_C_ID, email="surface-consistency@test.com",
                display_name="Surface Consistency Test", is_active=True,
                is_admin=True, hashed_password="",
            ))
        db.query(AIPortfolioConfig).filter_by(user_id=TEST_USER_C_ID).delete()
        db.query(AIPortfolioSnapshot).filter_by(user_id=TEST_USER_C_ID).delete()
        db.query(AIPortfolioPosition).filter_by(user_id=TEST_USER_C_ID).delete()
        # Commit the user BEFORE the config. On a first run against a fresh
        # DB the endpoints call get_or_create_config(), which CREATES a
        # default $25k config when it finds none -- so a half-committed seed
        # made this module pass or fail depending on whether the test DB had
        # been used before. Seed deterministically instead.
        db.commit()
        db.add(AIPortfolioConfig(
            user_id=TEST_USER_C_ID, starting_cash=STARTING_CASH,
            current_cash=LIVE_CASH, is_active=True,
        ))

        def snap(day_offset, value):
            d = date.today() - timedelta(days=day_offset)
            ts = datetime.now(timezone.utc) - timedelta(days=day_offset)
            db.add(AIPortfolioSnapshot(
                user_id=TEST_USER_C_ID, timestamp=ts, date=d,
                total_value=value, cash=value, positions_value=0.0,
                positions_count=0, total_return=value - STARTING_CASH,
                total_return_pct=((value - STARTING_CASH) / STARTING_CASH) * 100,
            ))

        snap(30, ANCHOR_VALUE)          # the 30D window anchor
        snap(1, LAST_SNAPSHOT_VALUE)    # last written snapshot, stale
        if with_snapshot_today:
            snap(0, 10900.0)            # today, still a scan-cycle behind
        db.commit()

        # Fail loudly rather than measure the wrong book: exactly one config,
        # holding the live cash we just seeded.
        configs = db.query(AIPortfolioConfig).filter_by(
            user_id=TEST_USER_C_ID).all()
        assert len(configs) == 1, f"expected 1 config, found {len(configs)}"
        assert configs[0].current_cash == LIVE_CASH
    finally:
        db.close()


def _collect_surfaces():
    """Every surface that reports the book, plus how it names each end.

    Returns {surface_name: (window_start_value, window_end_value)}.
    """
    wr = client.get("/api/ai-portfolio/window-returns?window=30d").json()["portfolio"]
    cc = client.get("/api/command-center").json()
    hist = [r for r in client.get(
        "/api/ai-portfolio/history?days=30&resolution=auto").json()
        if r.get("total_value") is not None]
    hist.sort(key=lambda r: r.get("timestamp") or r.get("date"))
    summary = client.get("/api/ai-portfolio").json()["summary"]

    spark = cc["sparkline"]
    return {
        "window-returns": (wr["start_value"], wr["current_value"]),
        "command-center": (spark[0]["value"], spark[-1]["value"]),
        # /history is not windowed server-side; the chart slices client-side,
        # so only its END is comparable here.
        "history": (None, hist[-1]["total_value"]),
        "ai-portfolio summary": (None, summary["total_value"]),
    }


class TestEverySurfaceAgrees:

    @pytest.mark.parametrize("with_snapshot_today", [False, True],
                             ids=["pre-market", "mid-session"])
    def test_all_surfaces_end_on_the_live_book(self, with_snapshot_today):
        """The end of the window is the LIVE book on every surface.

        Parametrized over both conditions because the bug was invisible
        mid-session (live value and newest snapshot minutes apart) and only
        opened up pre-market, when no snapshot exists for today yet.
        """
        _seed(with_snapshot_today=with_snapshot_today)
        surfaces = _collect_surfaces()

        ends = {name: round(v[1], 2) for name, v in surfaces.items()}
        assert len(set(ends.values())) == 1, (
            f"surfaces disagree on the window END: {ends}"
        )
        # ...and it is the live book, not the stale snapshot.
        assert set(ends.values()) == {round(LIVE_CASH, 2)}, (
            f"expected the live book {LIVE_CASH}, got {ends}"
        )

    def test_windowed_surfaces_share_the_anchor(self):
        _seed()
        surfaces = _collect_surfaces()
        starts = {n: round(v[0], 2) for n, v in surfaces.items() if v[0] is not None}
        assert len(set(starts.values())) == 1, (
            f"surfaces disagree on the window START: {starts}"
        )
        assert set(starts.values()) == {round(ANCHOR_VALUE, 2)}

    def test_reported_percentages_agree(self):
        """The number the user actually reads, not just the endpoints."""
        _seed()
        wr = client.get(
            "/api/ai-portfolio/window-returns?window=30d").json()["portfolio"]
        spark = client.get("/api/command-center").json()["sparkline"]

        cc_pct = ((spark[-1]["value"] - spark[0]["value"])
                  / spark[0]["value"]) * 100
        assert round(cc_pct, 2) == round(wr["return_pct"], 2), (
            f"Command Center reads {cc_pct:.2f}% but the slicer reads "
            f"{wr['return_pct']:.2f}% on the same book"
        )

    def test_the_fixture_would_expose_the_bug(self):
        """Guards the guard.

        If someone 'tidies' the fixture so the live book equals the last
        snapshot, every assertion above passes vacuously -- which is exactly
        how the original tests missed three shipped bugs.
        """
        assert LIVE_CASH != LAST_SNAPSHOT_VALUE, (
            "fixture no longer distinguishes the live book from the last "
            "snapshot; these tests would pass vacuously"
        )
