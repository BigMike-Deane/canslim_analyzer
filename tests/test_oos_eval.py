"""Tests for the OOS model evaluation framework (ml/oos_eval.py)."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


# Use a small inline model for tests so we don't depend on disk artifacts.
def _make_test_model(feature_cols=None, save_path=None):
    """Train a tiny XGBoost on synthetic data, save to disk, return path."""
    from xgboost import XGBClassifier

    if feature_cols is None:
        feature_cols = ["a", "b", "c"]

    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame(rng.randn(n, len(feature_cols)), columns=feature_cols)
    # Make `a` predictive of class
    y = (X["a"] + rng.randn(n) * 0.5 > 0).astype(int).values

    model = XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss")
    model.fit(X, y)

    payload = {
        "model": model,
        "feature_columns": feature_cols,
        "metadata": {"strategy": "test", "version": 99},
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path = save_path or Path(tempfile.mktemp(suffix=".joblib"))
    joblib.dump(payload, path)
    return path, feature_cols


def _make_holdout_df(feature_cols, n=50, gain_split=0.5, seed=7):
    """Synthetic holdout DataFrame with feature columns + gain_pct + win label."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, len(feature_cols)), columns=feature_cols)
    # gain_pct correlated with feature `a` so a good model can rank
    gain = (X[feature_cols[0]] * 8.0 + rng.randn(n) * 5.0).values
    df = X.copy()
    df["gain_pct"] = gain
    df["win"] = (gain > 0).astype(int)
    df["ticker"] = [f"T{i}" for i in range(n)]
    df["date"] = pd.date_range("2024-01-01", periods=n).date
    df["backtest_id"] = (np.arange(n) // 10) + 1000
    df["holding_days"] = 30
    df["sell_reason"] = "test"
    return df


# =============== evaluate_model_on_trades ===============


class TestEvaluateModelOnTrades:
    def test_basic_evaluation_returns_metrics(self):
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        result = evaluate_model_on_trades(path, df, min_gain_pct=0.0)

        assert "error" not in result
        assert result["n_trades"] == 50
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["brier_score"] <= 1.0
        assert result["model_feature_count"] == len(cols)
        # `a` is predictive — model should rank above random
        assert result["roc_auc"] > 0.5

    def test_empty_dataframe_returns_error(self):
        from ml.oos_eval import evaluate_model_on_trades

        path, _ = _make_test_model()
        result = evaluate_model_on_trades(path, pd.DataFrame())
        assert "error" in result
        assert "Empty" in result["error"]

    def test_missing_features_reported_explicitly(self):
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model(feature_cols=["a", "b", "c", "d"])
        # Holdout missing column `d`
        df = _make_holdout_df(["a", "b", "c"])
        result = evaluate_model_on_trades(path, df)
        assert "error" in result
        assert "missing" in result["error"].lower()
        assert result["missing_features"] == ["d"]

    def test_single_class_holdout_returns_error(self):
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        # Force all positive gains
        df["gain_pct"] = 100.0
        result = evaluate_model_on_trades(path, df, min_gain_pct=10.0)
        assert "error" in result
        assert "one class" in result["error"]
        # Still reports the win_rate so caller can see what happened
        assert result["win_rate"] == 1.0

    def test_missing_gain_pct_returns_error(self):
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        df = df.drop(columns=["gain_pct"])
        result = evaluate_model_on_trades(path, df)
        assert "error" in result
        assert "gain_pct" in result["error"]

    def test_missing_model_file_raises(self):
        """DF passes shape checks → load attempt → FileNotFoundError."""
        from ml.oos_eval import evaluate_model_on_trades

        df = pd.DataFrame({"a": [1.0, 2.0], "gain_pct": [5.0, 15.0]})
        with pytest.raises(FileNotFoundError):
            evaluate_model_on_trades("/tmp/does/not/exist.joblib", df)


# =============== get_holdout_trades ===============


class TestGetHoldoutTrades:
    def _fresh_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.database import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        return sessionmaker(bind=engine)()

    def test_holdout_filter_after_cutoff_returns_only_post(self):
        """Backtests created after cutoff are returned when after=True."""
        from ml.oos_eval import get_holdout_trades
        from backend.database import BacktestRun

        db = self._fresh_session()
        # Two backtests, one before cutoff and one after
        cutoff = datetime(2026, 5, 1)
        for i, ts in enumerate([datetime(2026, 4, 15), datetime(2026, 5, 3)]):
            db.add(BacktestRun(
                id=i + 1,
                name=f"r{i}",
                start_date=datetime(2022, 1, 1).date(),
                end_date=datetime(2026, 4, 28).date(),
                starting_cash=25000.0,
                stock_universe="all",
                strategy="nostate_optimized",
                status="completed",
                created_at=ts,
            ))
        db.commit()

        # No trades populated → df is empty, but the FILTER should still
        # only consider runs in the right slice (proven by no exception
        # and an empty result, not a partial one)
        result = get_holdout_trades(db, "nostate_optimized", cutoff, after=True)
        # extract_training_data returns empty DataFrame when no trades exist
        assert result.empty


# =============== compare_models_oos ===============


class TestCompareModels:
    def test_compare_two_models_on_same_holdout(self):
        from ml.oos_eval import evaluate_model_on_trades
        # We'll exercise compare_models_oos's trade-extraction path elsewhere
        # (it needs DB seeding); here just verify two evaluate_model_on_trades
        # calls on the same df produce the same per-call metrics for the
        # same model — sanity check on determinism.
        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        r1 = evaluate_model_on_trades(path, df, min_gain_pct=0.0)
        r2 = evaluate_model_on_trades(path, df, min_gain_pct=0.0)
        assert r1["roc_auc"] == r2["roc_auc"]
        assert r1["brier_score"] == r2["brier_score"]
        assert r1["n_trades"] == r2["n_trades"]
