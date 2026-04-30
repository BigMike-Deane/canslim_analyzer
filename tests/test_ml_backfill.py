"""
Tests for ml_predictions outcome backfill.

The backfill matches each ml_predictions row to its eventual buy/sell
trade pair and records the realized outcome. These tests pin the matching
behavior so it stays correct as schema and trade flows evolve.
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db,
    SessionLocal,
    AIPortfolioTrade,
    MLModel,
    MLPrediction,
)
from backend.ml_backfill import backfill_actual_outcomes


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(MLPrediction).delete()
    db.query(MLModel).delete()
    db.query(AIPortfolioTrade).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(MLPrediction).delete()
        db.query(MLModel).delete()
        db.query(AIPortfolioTrade).delete()
        db.commit()
        db.close()


def _make_model(db):
    m = MLModel(
        version=1, strategy="nostate_optimized", status="active",
        feature_count=24, roc_auc=0.6,
        activated_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _make_prediction(db, model_id, ticker, prediction_date, ml_confidence=0.55):
    p = MLPrediction(
        model_id=model_id, ticker=ticker, prediction_date=prediction_date,
        ml_confidence=ml_confidence, features={"c_score": 12.0},
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _make_trade(db, ticker, action, executed_at, *, shares=100.0, price=10.0,
                cost_basis=None, realized_gain=None, signal_factors=None,
                reason=None):
    t = AIPortfolioTrade(
        ticker=ticker, action=action, shares=shares, price=price,
        total_value=shares * price, executed_at=executed_at,
        cost_basis=cost_basis, realized_gain=realized_gain,
        signal_factors=signal_factors, reason=reason, user_id=1,
    )
    db.add(t)
    db.commit()
    return t


class TestBackfillCore:
    def test_winning_trade_marks_outcome_one(self, db_session):
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "AAPL", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        sell_at = buy_at + timedelta(days=3)
        _make_trade(db_session, "AAPL", "BUY", buy_at, price=100.0, shares=10)
        _make_trade(
            db_session, "AAPL", "SELL", sell_at,
            price=110.0, shares=10, cost_basis=100.0, realized_gain=100.0,
            signal_factors={"sell_reason": "TAKE_PROFIT", "gain_pct": 10.0},
            reason="TAKE PROFIT",
        )

        result = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert result["updated"] == 1
        assert p.actual_outcome == 1
        assert p.actual_gain_pct == pytest.approx(10.0)

    def test_losing_trade_marks_outcome_zero(self, db_session):
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "XYZ", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        sell_at = buy_at + timedelta(days=2)
        _make_trade(db_session, "XYZ", "BUY", buy_at, price=50.0, shares=10)
        _make_trade(
            db_session, "XYZ", "SELL", sell_at,
            price=46.5, shares=10, cost_basis=50.0, realized_gain=-35.0,
            signal_factors={"sell_reason": "STOP LOSS", "gain_pct": -7.0},
            reason="STOP LOSS",
        )

        backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert p.actual_outcome == 0
        assert p.actual_gain_pct == pytest.approx(-7.0)

    def test_open_position_left_pending(self, db_session):
        # Buy with no matching sell = position still open. Outcome stays NULL.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "OPEN", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        _make_trade(db_session, "OPEN", "BUY", buy_at)

        result = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert result["still_open"] == 1
        assert p.actual_outcome is None

    def test_no_buy_means_veto_left_pending(self, db_session):
        # Prediction with no corresponding buy is presumed vetoed. Stays NULL —
        # we don't try to fabricate a counterfactual outcome.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "VETOED", pred_date, ml_confidence=0.18)

        result = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert result["no_buy"] == 1
        assert p.actual_outcome is None

    def test_partial_sell_does_not_close_position(self, db_session):
        # PARTIAL sells must be skipped — they don't represent the trade's
        # final outcome. Position should still be considered open.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "PART", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        partial_at = buy_at + timedelta(days=2)
        _make_trade(db_session, "PART", "BUY", buy_at, shares=100, price=10)
        _make_trade(
            db_session, "PART", "SELL", partial_at,
            shares=25, price=11, cost_basis=10.0, realized_gain=25.0,
            signal_factors={"sell_reason": "PRE-EARNINGS", "gain_pct": 10.0, "sell_pct": 25},
            reason="PRE-EARNINGS PARTIAL: protecting gains",
        )

        result = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert result["still_open"] == 1
        assert p.actual_outcome is None

    def test_partial_then_full_sell_uses_full(self, db_session):
        # Common case: protect gains with a partial, then full close later.
        # We should pair to the full close, not the partial.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=10)
        p = _make_prediction(db_session, m.id, "PFULL", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        partial_at = buy_at + timedelta(days=2)
        full_at = buy_at + timedelta(days=5)

        _make_trade(db_session, "PFULL", "BUY", buy_at, shares=100, price=10)
        _make_trade(
            db_session, "PFULL", "SELL", partial_at,
            shares=25, price=11, cost_basis=10.0, realized_gain=25.0,
            signal_factors={"gain_pct": 10.0, "sell_pct": 25},
            reason="PRE-EARNINGS PARTIAL",
        )
        _make_trade(
            db_session, "PFULL", "SELL", full_at,
            shares=75, price=14, cost_basis=10.0, realized_gain=300.0,
            signal_factors={"gain_pct": 40.0},
            reason="TRAILING STOP",
        )

        backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert p.actual_outcome == 1
        assert p.actual_gain_pct == pytest.approx(40.0)

    def test_idempotent_rerun_skips_resolved_rows(self, db_session):
        # Second run must NOT re-touch already-resolved rows. Cheap to re-run.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "IDEM", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        sell_at = buy_at + timedelta(days=2)
        _make_trade(db_session, "IDEM", "BUY", buy_at, price=20, shares=10)
        _make_trade(
            db_session, "IDEM", "SELL", sell_at,
            price=22, shares=10, cost_basis=20.0, realized_gain=20.0,
            signal_factors={"gain_pct": 10.0}, reason="TAKE PROFIT",
        )

        first = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()  # In production the scheduler runs a fresh session per call.
        second = backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert first["updated"] == 1
        assert second["checked"] == 0
        assert second["updated"] == 0

    def test_lookback_window_bounds_workload(self, db_session):
        # Old predictions are out of scope — keeps the job O(window) not O(table).
        m = _make_model(db_session)
        old_date = date.today() - timedelta(days=200)
        _make_prediction(db_session, m.id, "OLD", old_date)

        result = backfill_actual_outcomes(db_session, lookback_days=90)
        assert result["checked"] == 0

    def test_falls_back_to_computed_gain_when_signal_factors_missing(self, db_session):
        # Old trades have NULL signal_factors; outcome can still be derived
        # from cost_basis × shares against realized_gain.
        m = _make_model(db_session)
        pred_date = date.today() - timedelta(days=5)
        p = _make_prediction(db_session, m.id, "OLDFMT", pred_date)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        sell_at = buy_at + timedelta(days=2)
        _make_trade(db_session, "OLDFMT", "BUY", buy_at, price=10, shares=100)
        _make_trade(
            db_session, "OLDFMT", "SELL", sell_at,
            price=11, shares=100, cost_basis=10.0, realized_gain=100.0,
            signal_factors=None, reason="TRAILING STOP",
        )

        backfill_actual_outcomes(db_session, lookback_days=30)
        db_session.commit()

        assert p.actual_outcome == 1
        assert p.actual_gain_pct == pytest.approx(10.0)
