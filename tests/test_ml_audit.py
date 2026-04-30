"""
Tests for ML prediction audit logging in ai_trader.

The `_get_active_ml_model_id` and `_record_ml_prediction` helpers exist so
that every ML evaluation in evaluate_buys leaves a row in `ml_predictions`,
making the veto path observable in production. These tests pin that contract.
"""

import os
from datetime import date, datetime, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import init_db, SessionLocal, MLModel, MLPrediction
from backend.ai_trader import _get_active_ml_model_id, _record_ml_prediction


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(MLPrediction).delete()
    db.query(MLModel).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(MLPrediction).delete()
        db.query(MLModel).delete()
        db.commit()
        db.close()


def _make_model(db, *, status="active", version=1):
    m = MLModel(
        version=version,
        strategy="nostate_optimized",
        status=status,
        feature_count=24,
        roc_auc=0.6,
        model_path="/tmp/fake.joblib",
        activated_at=datetime.now(timezone.utc) if status == "active" else None,
        created_at=datetime.now(timezone.utc),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


class TestGetActiveMLModelId:
    def test_returns_id_when_active_model_exists(self, db_session):
        m = _make_model(db_session, status="active", version=12)
        assert _get_active_ml_model_id(db_session) == m.id

    def test_returns_none_when_no_active_model(self, db_session):
        _make_model(db_session, status="failed", version=11)
        assert _get_active_ml_model_id(db_session) is None

    def test_picks_latest_when_multiple_active(self, db_session):
        # Defensive: schema doesn't enforce single-active, so we should pick the newest.
        _make_model(db_session, status="active", version=11)
        newer = _make_model(db_session, status="active", version=12)
        assert _get_active_ml_model_id(db_session) == newer.id

    def test_swallows_db_errors(self):
        # Pass a sentinel that will blow up on .query() — helper must not raise.
        class Boom:
            def query(self, *_a, **_kw):
                raise RuntimeError("db down")
        assert _get_active_ml_model_id(Boom()) is None


class TestRecordMLPrediction:
    def test_writes_row_with_features(self, db_session):
        m = _make_model(db_session)
        features = {
            "ml_confidence": 0.72,
            "c_score": 15.0,
            "a_score": 12.1,
            "entry_type": 2,
            "market_regime": 1,
        }
        _record_ml_prediction(db_session, m.id, "AAPL", 0.72, features)
        db_session.commit()

        row = db_session.query(MLPrediction).one()
        assert row.model_id == m.id
        assert row.ticker == "AAPL"
        assert row.ml_confidence == pytest.approx(0.72)
        assert row.prediction_date == date.today()
        assert row.features["c_score"] == 15.0
        assert row.actual_outcome is None  # backfilled later

    def test_low_confidence_veto_is_recorded(self, db_session):
        # The whole point of the audit is to make vetoes observable.
        m = _make_model(db_session)
        _record_ml_prediction(db_session, m.id, "XYZ", 0.18, {"c_score": 5.0})
        db_session.commit()

        row = db_session.query(MLPrediction).filter_by(ticker="XYZ").one()
        assert row.ml_confidence == pytest.approx(0.18)

    def test_no_op_when_model_id_is_none(self, db_session):
        # Guards against running before any model is trained.
        _record_ml_prediction(db_session, None, "AAPL", 0.5, {})
        db_session.commit()
        assert db_session.query(MLPrediction).count() == 0

    def test_sanitizes_nan_features(self, db_session):
        # signal_factors can carry NaN from upstream; sanitize_signal_factors
        # converts them to 0 so the JSON column stays valid.
        m = _make_model(db_session)
        features = {"c_score": float("nan"), "a_score": 12.0}
        _record_ml_prediction(db_session, m.id, "AAPL", 0.5, features)
        db_session.commit()

        row = db_session.query(MLPrediction).one()
        assert row.features["c_score"] == 0
        assert row.features["a_score"] == 12.0

    def test_swallows_write_errors(self, db_session):
        # An audit failure must never block trading. Pass an obviously bad
        # model_id so the FK fails on flush — we should still return cleanly.
        # We don't commit here; the helper just stages the row.
        try:
            _record_ml_prediction(db_session, 999_999_999, "AAPL", 0.5, {"c_score": 10.0})
        except Exception as e:
            pytest.fail(f"_record_ml_prediction must not raise: {e}")
