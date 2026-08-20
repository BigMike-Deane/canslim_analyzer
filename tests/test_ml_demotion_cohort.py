"""Tests for /api/admin/ml-demotion-cohort.

The 2026-07-04 ML demotion (283b33c) kept ml_confidence logging on every
BUY. The planned entry-rate re-check turned out to be doubly confounded
(portfolios pinned at max_positions; pre-period had the veto active), so
this endpoint gives the direct read: split post-demotion BUYs into
would-have-been-vetoed (conf < old 0.30 threshold) vs passed cohorts and
compare outcomes.

Direct-invocation pattern mirrors tests/test_cap_delta_diagnostics.py —
bypass FastAPI's resolver with current_user=None and explicit db.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, AIPortfolioTrade, AIPortfolioPosition,
)

CUTOFF = "2026-07-04"
AFTER = datetime(2026, 7, 6, tzinfo=timezone.utc)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioPosition).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioPosition).delete()
        db.commit()
        db.close()


def _buy(db, ticker, conf, *, user_id=1, price=100.0, shares=10.0,
         executed_at=AFTER, signal_factors=None):
    sf = signal_factors if signal_factors is not None else {"ml_confidence": conf}
    t = AIPortfolioTrade(
        ticker=ticker, action="BUY", shares=shares, price=price,
        total_value=shares * price, executed_at=executed_at,
        signal_factors=sf, user_id=user_id,
    )
    db.add(t)
    return t


def _sell(db, ticker, *, realized_gain, user_id=1, price=100.0, shares=10.0,
          executed_at=None):
    t = AIPortfolioTrade(
        ticker=ticker, action="SELL", shares=shares, price=price,
        total_value=shares * price, realized_gain=realized_gain,
        executed_at=executed_at or (AFTER + timedelta(days=3)),
        user_id=user_id,
    )
    db.add(t)
    return t


def _position(db, ticker, *, gain_loss_pct, user_id=1):
    p = AIPortfolioPosition(
        ticker=ticker, user_id=user_id, shares=10.0, cost_basis=100.0,
        current_price=100.0 * (1 + gain_loss_pct / 100),
        current_value=1000.0 * (1 + gain_loss_pct / 100),
        gain_loss=1000.0 * gain_loss_pct / 100, gain_loss_pct=gain_loss_pct,
        purchase_date=AFTER,
    )
    db.add(p)
    return p


def _call(db, **kwargs):
    from backend.routes.admin import ml_demotion_cohort
    defaults = dict(
        cutoff_date=CUTOFF, threshold=0.30, current_user=None, db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(ml_demotion_cohort(**defaults))


class TestCohortSplit:
    def test_threshold_splits_cohorts_boundary_passes(self, db_session):
        """conf < threshold → would_veto; conf == threshold passes (the old
        veto skipped only strictly-below rows)."""
        _buy(db_session, "LOWC", 0.16)
        _buy(db_session, "EDGE", 0.30)
        _buy(db_session, "HIGH", 0.42)
        db_session.commit()
        out = _call(db_session)
        by_ticker = {r["ticker"]: r["cohort"] for r in out["trades"]}
        assert by_ticker == {"LOWC": "would_veto", "EDGE": "passed", "HIGH": "passed"}
        assert out["would_veto"]["n"] == 1
        assert out["passed"]["n"] == 2

    def test_buys_before_cutoff_and_without_confidence_excluded(self, db_session):
        _buy(db_session, "OLD", 0.10,
             executed_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
        _buy(db_session, "NOCONF", None, signal_factors={"entry_type": "breakout"})
        _buy(db_session, "KEPT", 0.20)
        db_session.commit()
        out = _call(db_session)
        assert [r["ticker"] for r in out["trades"]] == ["KEPT"]

    def test_legacy_string_signal_factors_decoded(self, db_session):
        """Legacy rows store signal_factors as a JSON string — must decode,
        not crash (same handling as _summarize_trades)."""
        _buy(db_session, "LEGACY", None,
             signal_factors=json.dumps({"ml_confidence": 0.22}))
        db_session.commit()
        out = _call(db_session)
        assert out["trades"][0]["ticker"] == "LEGACY"
        assert out["trades"][0]["cohort"] == "would_veto"


class TestOutcomes:
    def test_open_position_marks_to_market(self, db_session):
        _buy(db_session, "OPEN", 0.20)
        _position(db_session, "OPEN", gain_loss_pct=7.5)
        db_session.commit()
        out = _call(db_session)
        row = out["trades"][0]
        assert row["status"] == "open"
        assert row["gain_pct"] == 7.5

    def test_closed_position_realizes_against_sells(self, db_session):
        """No position row + the lot fully sold → the LOT's own FIFO
        outcome from sell prices (partial sells accumulate)."""
        _buy(db_session, "GONE", 0.20, price=100.0, shares=10.0)
        _sell(db_session, "GONE", realized_gain=50.0, price=110.0, shares=5.0,
              executed_at=AFTER + timedelta(days=2))
        _sell(db_session, "GONE", realized_gain=30.0, price=106.0, shares=5.0,
              executed_at=AFTER + timedelta(days=4))
        db_session.commit()
        out = _call(db_session)
        row = out["trades"][0]
        assert row["status"] == "closed"
        assert row["gain_pct"] == 8.0  # (5*10 + 5*6) / 1000 * 100

    def test_regenerated_ticker_does_not_cross_contaminate_lots(self, db_session):
        """Live 2026-08-20 bug (ANET-u3): buy A closed at a small gain, the
        ticker was re-bought MUCH larger and stopped out. The old code
        summed ALL later sells over buy A's cost (-59.1% for a +4.1% lot)
        and, had the second position still been open, would have marked the
        closed buy A 'open' with the new position's gain. Each buy must
        report its OWN lot's FIFO outcome."""
        # Generation 1: 2 shares @ 100, fully sold @ 104 (+4%)
        _buy(db_session, "REGEN", 0.20, price=100.0, shares=2.0,
             executed_at=AFTER)
        _sell(db_session, "REGEN", realized_gain=8.0, price=104.0, shares=2.0,
              executed_at=AFTER + timedelta(days=1))
        # Generation 2: 20 shares @ 105, stopped out @ 96 (-8.6%)
        _buy(db_session, "REGEN", 0.35, price=105.0, shares=20.0,
             executed_at=AFTER + timedelta(days=2))
        _sell(db_session, "REGEN", realized_gain=-180.0, price=96.0, shares=20.0,
              executed_at=AFTER + timedelta(days=10))
        db_session.commit()
        out = _call(db_session)
        by_gen = {r["cohort"]: r for r in out["trades"]}
        gen1 = by_gen["would_veto"]   # conf 0.20
        gen2 = by_gen["passed"]       # conf 0.35
        assert gen1["status"] == "closed" and gen1["gain_pct"] == 4.0
        assert gen2["status"] == "closed" and gen2["gain_pct"] == pytest.approx(-8.57, abs=0.01)

    def test_closed_generation_stays_closed_when_ticker_rebought_open(self, db_session):
        """A closed lot must not flip to 'open' just because a NEWER
        position for the same ticker exists."""
        _buy(db_session, "REOPEN", 0.20, price=100.0, shares=10.0,
             executed_at=AFTER)
        _sell(db_session, "REOPEN", realized_gain=100.0, price=110.0, shares=10.0,
              executed_at=AFTER + timedelta(days=1))
        _buy(db_session, "REOPEN", 0.40, price=120.0, shares=10.0,
             executed_at=AFTER + timedelta(days=2))
        _position(db_session, "REOPEN", gain_loss_pct=5.0)
        db_session.commit()
        out = _call(db_session)
        by_conf = {r["ml_confidence"]: r for r in out["trades"]}
        assert by_conf[0.20]["status"] == "closed"
        assert by_conf[0.20]["gain_pct"] == 10.0
        assert by_conf[0.40]["status"] == "open"
        assert by_conf[0.40]["gain_pct"] == 5.0

    def test_position_scoped_by_user(self, db_session):
        """User 2's open position must not resolve user 1's buy — user 1's
        buy with no position and no sells is 'unknown'."""
        _buy(db_session, "SCOPE", 0.20, user_id=1)
        _position(db_session, "SCOPE", gain_loss_pct=99.0, user_id=2)
        db_session.commit()
        out = _call(db_session)
        assert out["trades"][0]["status"] == "unknown"
        assert out["trades"][0]["gain_pct"] is None

    def test_aggregates_and_win_rate(self, db_session):
        _buy(db_session, "W1", 0.10)
        _position(db_session, "W1", gain_loss_pct=10.0)
        _buy(db_session, "L1", 0.15)
        _position(db_session, "L1", gain_loss_pct=-5.0)
        _buy(db_session, "P1", 0.50)
        _position(db_session, "P1", gain_loss_pct=3.0)
        db_session.commit()
        out = _call(db_session)
        assert out["would_veto"]["n"] == 2
        assert out["would_veto"]["open_n"] == 2
        assert out["would_veto"]["avg_gain_pct"] == 2.5
        assert out["would_veto"]["win_rate"] == 50.0
        assert out["passed"]["avg_gain_pct"] == 3.0
        assert out["passed"]["win_rate"] == 100.0


class TestCohortAge:
    def test_row_age_days_from_executed_at(self, db_session):
        _buy(db_session, "AGED", 0.20,
             executed_at=datetime.now(timezone.utc) - timedelta(days=14))
        _position(db_session, "AGED", gain_loss_pct=5.0)
        db_session.commit()
        out = _call(db_session)
        assert out["trades"][0]["age_days"] == 14

    def test_naive_executed_at_does_not_crash(self, db_session):
        """Live DB rows store naive-UTC executed_at; age math must not raise
        on the naive/aware mix."""
        _buy(db_session, "NAIVE", 0.20,
             executed_at=datetime.utcnow() - timedelta(days=3))
        db_session.commit()
        out = _call(db_session)
        assert out["trades"][0]["age_days"] == 3

    def test_avg_age_days_only_over_gain_scored_rows(self, db_session):
        """avg_age_days must qualify avg_gain_pct, so 'unknown' rows (no
        position, no sells → gain_pct None) are excluded from the average."""
        _buy(db_session, "SCORED", 0.20,
             executed_at=datetime.now(timezone.utc) - timedelta(days=10))
        _position(db_session, "SCORED", gain_loss_pct=5.0)
        _buy(db_session, "UNKNOWN", 0.20,
             executed_at=datetime.now(timezone.utc) - timedelta(days=2))
        db_session.commit()
        out = _call(db_session)
        assert out["would_veto"]["avg_age_days"] == 10.0
        assert out["would_veto"]["n"] == 2


class TestValidation:
    def test_bad_cutoff_422(self, db_session):
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date="not-a-date")
        assert exc.value.status_code == 422
