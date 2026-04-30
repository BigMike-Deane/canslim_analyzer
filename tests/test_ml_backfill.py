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
    Stock,
    StockScore,
)
from backend.ml_backfill import backfill_actual_outcomes, counterfactual_backfill


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(MLPrediction).delete()
    db.query(MLModel).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(StockScore).delete()
    db.query(Stock).filter(Stock.ticker.like('CF_%')).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(MLPrediction).delete()
        db.query(MLModel).delete()
        db.query(AIPortfolioTrade).delete()
        db.query(StockScore).delete()
        db.query(Stock).filter(Stock.ticker.like('CF_%')).delete()
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


# ── Counterfactual backfill tests ────────────────────────────────────


def _make_stock(db, ticker="CF_FOO"):
    s = Stock(ticker=ticker, name=f"{ticker} Inc.", sector="Tech", current_price=100.0)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _make_score(db, stock_id, on_date, price):
    db.add(StockScore(
        stock_id=stock_id, date=on_date,
        timestamp=datetime.combine(on_date, datetime.min.time()),
        total_score=70.0, c_score=12, a_score=11, n_score=10,
        s_score=12, l_score=11, i_score=8, m_score=11,
        current_price=price,
    ))
    db.commit()


class TestCounterfactualBackfill:
    def test_synthesizes_winner_outcome(self, db_session):
        m = _make_model(db_session)
        s = _make_stock(db_session, "CF_WIN")
        # Prediction 30 days ago — old enough that 21 days of post-data exists.
        pred_date = date.today() - timedelta(days=30)
        p = _make_prediction(db_session, m.id, "CF_WIN", pred_date, ml_confidence=0.18)

        # Price doubled in 21 days — clear winner under min_gain_pct=10.0.
        _make_score(db_session, s.id, pred_date, price=50.0)
        _make_score(db_session, s.id, pred_date + timedelta(days=21), price=100.0)

        result = counterfactual_backfill(db_session, lookback_days=60, hold_days=21, min_gain_pct=10.0)
        db_session.commit()

        assert result["updated"] == 1
        assert p.actual_outcome == 1
        assert p.actual_gain_pct == pytest.approx(100.0)

    def test_synthesizes_loser_outcome(self, db_session):
        m = _make_model(db_session)
        s = _make_stock(db_session, "CF_LOSE")
        pred_date = date.today() - timedelta(days=30)
        p = _make_prediction(db_session, m.id, "CF_LOSE", pred_date, ml_confidence=0.10)

        # Price drifted +3% in 21 days — under 10% threshold, labeled loser.
        _make_score(db_session, s.id, pred_date, price=100.0)
        _make_score(db_session, s.id, pred_date + timedelta(days=21), price=103.0)

        counterfactual_backfill(db_session, lookback_days=60, hold_days=21, min_gain_pct=10.0)
        db_session.commit()

        assert p.actual_outcome == 0
        assert p.actual_gain_pct == pytest.approx(3.0)

    def test_skips_predictions_with_real_buy(self, db_session):
        # Realized backfill owns these — counterfactual must not double-touch.
        m = _make_model(db_session)
        s = _make_stock(db_session, "CF_REAL")
        pred_date = date.today() - timedelta(days=30)
        p = _make_prediction(db_session, m.id, "CF_REAL", pred_date, ml_confidence=0.55)

        buy_at = datetime.combine(pred_date, datetime.min.time()) + timedelta(hours=10)
        _make_trade(db_session, "CF_REAL", "BUY", buy_at, price=100, shares=10)
        _make_score(db_session, s.id, pred_date, price=100.0)
        _make_score(db_session, s.id, pred_date + timedelta(days=21), price=130.0)

        result = counterfactual_backfill(db_session, lookback_days=60)
        db_session.commit()

        assert result["skipped_has_buy"] == 1
        assert p.actual_outcome is None  # Left for realized backfill.

    def test_skips_predictions_too_recent_to_evaluate(self, db_session):
        # We need hold_days of post-prediction price. A prediction from yesterday
        # can't be backfilled yet — there's no future data to look at.
        m = _make_model(db_session)
        _make_stock(db_session, "CF_NEW")
        pred_date = date.today() - timedelta(days=2)
        p = _make_prediction(db_session, m.id, "CF_NEW", pred_date, ml_confidence=0.20)

        result = counterfactual_backfill(db_session, lookback_days=60, hold_days=21)
        db_session.commit()

        assert result["checked"] == 0
        assert p.actual_outcome is None

    def test_no_price_data_skips_gracefully(self, db_session):
        # Stock exists but has no StockScore rows — can't compute outcome.
        m = _make_model(db_session)
        _make_stock(db_session, "CF_NOPRICE")
        pred_date = date.today() - timedelta(days=30)
        p = _make_prediction(db_session, m.id, "CF_NOPRICE", pred_date, ml_confidence=0.20)

        result = counterfactual_backfill(db_session, lookback_days=60)
        db_session.commit()

        assert result["no_price_data"] == 1
        assert p.actual_outcome is None

    def test_idempotent_rerun(self, db_session):
        m = _make_model(db_session)
        s = _make_stock(db_session, "CF_IDEM")
        pred_date = date.today() - timedelta(days=30)
        _make_prediction(db_session, m.id, "CF_IDEM", pred_date, ml_confidence=0.15)
        _make_score(db_session, s.id, pred_date, price=50.0)
        _make_score(db_session, s.id, pred_date + timedelta(days=21), price=80.0)

        first = counterfactual_backfill(db_session, lookback_days=60)
        db_session.commit()
        second = counterfactual_backfill(db_session, lookback_days=60)
        db_session.commit()

        assert first["updated"] == 1
        assert second["checked"] == 0
        assert second["updated"] == 0

    def test_lookback_window_excludes_old_predictions(self, db_session):
        m = _make_model(db_session)
        _make_stock(db_session, "CF_OLD")
        pred_date = date.today() - timedelta(days=200)
        _make_prediction(db_session, m.id, "CF_OLD", pred_date, ml_confidence=0.15)

        result = counterfactual_backfill(db_session, lookback_days=90)
        assert result["checked"] == 0
