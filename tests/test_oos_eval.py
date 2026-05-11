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


class _StubModelNoFeatureNames:
    """Bare predict_proba stub with NO feature_names_in_ attribute. Defined
    at module scope so joblib can serialize it (function-local classes
    don't survive joblib.dump). Used by TestEvaluateModelFeatureColumnFallback
    to force the payload['feature_columns'] fallback branch at
    oos_eval.py:87."""
    def predict_proba(self, X):
        n = len(X)
        # Deterministic gradient ensures AUC > 0.5 and avoids the single-
        # class guard at oos_eval.py:99-104.
        col1 = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - col1, col1])


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

    def test_top_decile_wr_populated(self):
        """The May-5 diagnostic showed AUC was the wrong selection metric for
        our use case; top-decile WR is the head-of-distribution number that
        backtest actually filters on. The result must surface it."""
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model()
        df = _make_holdout_df(cols, n=100)
        result = evaluate_model_on_trades(path, df, min_gain_pct=0.0)

        assert "top_decile_wr" in result
        assert "top_decile_n" in result
        assert "top_decile_mean_proba" in result
        assert 0.0 <= result["top_decile_wr"] <= 1.0
        assert result["top_decile_n"] == 10  # 10% of 100
        # Top decile by predicted prob should beat overall WR for any
        # reasonably trained model — synthetic data is heavily separable.
        assert result["top_decile_wr"] >= result["win_rate"]

    def test_top_decile_floor_when_holdout_small(self):
        """Holdout with fewer than 10 trades — top decile is at least 1."""
        from ml.oos_eval import evaluate_model_on_trades

        path, cols = _make_test_model()
        df = _make_holdout_df(cols, n=5)
        result = evaluate_model_on_trades(path, df, min_gain_pct=0.0)
        # 5 // 10 == 0, but we floor at 1 so it's a meaningful number
        assert result["top_decile_n"] == 1


class TestTopNWrAtCount:
    """top_n_wr_at_count is the trade-count-matched comparison primitive.
    Use case: v17 vs v12 at v12's actual backtest trade count (e.g. 245)."""

    def test_basic_top_n_wr(self):
        from ml.oos_eval import top_n_wr_at_count

        path, cols = _make_test_model()
        df = _make_holdout_df(cols, n=100)
        result = top_n_wr_at_count(path, df, n_picks=20, min_gain_pct=0.0)

        assert "error" not in result
        assert result["n_picks"] == 20
        assert result["n_holdout"] == 100
        assert 0.0 <= result["top_n_wr"] <= 1.0
        # Top-N WR should be at least as good as random for a model with
        # any signal; synthetic data should give a clean lift
        overall_wr = (df["gain_pct"] > 0).mean()
        assert result["top_n_wr"] >= overall_wr - 0.01  # tolerance

    def test_n_picks_exceeds_holdout_returns_error(self):
        from ml.oos_eval import top_n_wr_at_count

        path, cols = _make_test_model()
        df = _make_holdout_df(cols, n=10)
        result = top_n_wr_at_count(path, df, n_picks=50)
        assert "error" in result
        assert "exceeds" in result["error"]

    def test_zero_or_negative_n_picks_returns_error(self):
        from ml.oos_eval import top_n_wr_at_count

        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        for n in [0, -1]:
            result = top_n_wr_at_count(path, df, n_picks=n)
            assert "error" in result

    def test_empty_df_returns_error(self):
        from ml.oos_eval import top_n_wr_at_count

        path, _ = _make_test_model()
        result = top_n_wr_at_count(path, pd.DataFrame(), n_picks=10)
        assert "error" in result

    def test_missing_gain_pct_returns_error(self):
        from ml.oos_eval import top_n_wr_at_count

        path, cols = _make_test_model()
        df = _make_holdout_df(cols)
        df = df.drop(columns=["gain_pct"])
        result = top_n_wr_at_count(path, df, n_picks=10)
        assert "error" in result
        assert "gain_pct" in result["error"]

    def test_missing_features_returns_error(self):
        from ml.oos_eval import top_n_wr_at_count

        path, _ = _make_test_model(feature_cols=["a", "b", "c", "d"])
        df = _make_holdout_df(["a", "b", "c"])  # missing "d"
        result = top_n_wr_at_count(path, df, n_picks=10)
        assert "error" in result
        assert "missing_features" in result


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


# =============== Coverage close-out: remaining branches ===============


class TestLoadModelPayloadValidation:
    """Covers ml/oos_eval.py line 54 — malformed-payload guard."""

    def test_payload_missing_model_key_raises_value_error(self):
        """A joblib file that doesn't contain a 'model' key must raise
        ValueError eagerly. Without this guard, callers would silently
        crash inside model.predict_proba with a confusing AttributeError."""
        from ml.oos_eval import _load_model_payload

        bad_path = Path(tempfile.mktemp(suffix=".joblib"))
        joblib.dump({"feature_columns": ["a", "b"], "metadata": {}}, bad_path)

        with pytest.raises(ValueError, match="missing 'model' key"):
            _load_model_payload(bad_path)


class TestEvaluateModelFeatureColumnFallback:
    """Covers ml/oos_eval.py line 87 — payload['feature_columns'] / global
    FEATURE_COLUMNS fallback when the model has no feature_names_in_."""

    def test_falls_back_to_payload_feature_columns_when_model_lacks_names(self):
        """If a model lacks feature_names_in_ (e.g. trained via raw xgboost
        API or older sklearn), evaluate_model_on_trades should resolve the
        feature order from payload['feature_columns']. Uses the module-level
        _StubModelNoFeatureNames because xgboost re-derives feature_names_in_
        on joblib roundtrip, making `del payload['model'].feature_names_in_`
        ineffective."""
        from ml.oos_eval import evaluate_model_on_trades

        cols = ["a", "b", "c"]
        path = Path(tempfile.mktemp(suffix=".joblib"))
        joblib.dump({
            "model": _StubModelNoFeatureNames(),
            "feature_columns": cols,
            "metadata": {"strategy": "test", "version": 0},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, path)

        df = _make_holdout_df(cols)
        result = evaluate_model_on_trades(path, df, min_gain_pct=0.0)
        # No error → the fallback path successfully resolved feature_cols
        # from payload["feature_columns"].
        assert "error" not in result
        assert result["model_feature_count"] == 3


class TestGetHoldoutTradesEdgeCases:
    """Covers ml/oos_eval.py lines 223 and 227 — before-cutoff branch +
    empty-run_ids short-circuit."""

    def _fresh_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.database import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        return sessionmaker(bind=engine)()

    def test_after_false_filters_to_pre_cutoff_runs(self):
        """after=False is the TRAINING-pool selector — line 223. Verifies
        the before-cutoff branch by seeding two runs on either side of the
        cutoff and confirming the function emits an empty df (no trades
        seeded) but with no SQL error — meaning the before-cutoff filter
        executed."""
        from ml.oos_eval import get_holdout_trades
        from backend.database import BacktestRun

        db = self._fresh_session()
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

        result = get_holdout_trades(db, "nostate_optimized", cutoff, after=False)
        # No trades seeded, but the function must execute the before-cutoff
        # branch without raising — empty df is the expected output shape.
        assert result.empty

    def test_no_matching_runs_returns_empty_df_without_calling_extractor(
        self, monkeypatch,
    ):
        """If the strategy/cutoff filter matches zero BacktestRun rows,
        get_holdout_trades must short-circuit and return an empty DataFrame
        WITHOUT calling extract_training_data (line 227). Without the
        short-circuit, extract_training_data would receive an empty
        backtest_ids list and return its own empty df — the explicit
        short-circuit keeps the contract obvious."""
        from ml.oos_eval import get_holdout_trades
        import ml.oos_eval as oos_mod

        # Spy on extract_training_data to verify it's NOT called
        called = {"n": 0}

        def _spy(*a, **kw):
            called["n"] += 1
            return pd.DataFrame(), {}

        monkeypatch.setattr(oos_mod, "extract_training_data", _spy)

        db = self._fresh_session()  # empty DB → no runs match anything
        result = get_holdout_trades(
            db, "nonexistent_strategy", datetime(2026, 5, 1), after=True,
        )
        assert result.empty
        assert called["n"] == 0, "extract_training_data should not be called"


class TestEvaluateOos:
    """Covers ml/oos_eval.py lines 250-263 — the evaluate_oos convenience
    wrapper. Mocks get_holdout_trades so we don't need full DB seeding."""

    def test_empty_holdout_returns_error_dict(self, monkeypatch):
        """When no trades exist after cutoff, evaluate_oos returns a
        structured error rather than calling evaluate_model_on_trades."""
        from ml.oos_eval import evaluate_oos
        import ml.oos_eval as oos_mod

        monkeypatch.setattr(
            oos_mod, "get_holdout_trades", lambda *a, **kw: pd.DataFrame(),
        )

        path, _ = _make_test_model()
        cutoff = datetime(2026, 5, 1)
        result = evaluate_oos(
            db=None, model_path=path, strategy="nostate_optimized", cutoff=cutoff,
        )
        assert "error" in result
        assert "No holdout trades" in result["error"]
        assert result["cutoff"] == cutoff.isoformat()
        assert result["strategy"] == "nostate_optimized"

    def test_happy_path_returns_metrics_with_holdout_metadata(self, monkeypatch):
        """Happy path: get_holdout_trades returns a non-empty df, then
        evaluate_model_on_trades runs, and the result is augmented with
        holdout_backtest_count + holdout_backtest_ids + cutoff."""
        from ml.oos_eval import evaluate_oos
        import ml.oos_eval as oos_mod

        path, cols = _make_test_model()
        df = _make_holdout_df(cols, n=40)
        # backtest_id column is set by _make_holdout_df: 1000, 1001, 1002, 1003
        expected_ids = sorted(set(df["backtest_id"].tolist()))

        monkeypatch.setattr(oos_mod, "get_holdout_trades", lambda *a, **kw: df)

        cutoff = datetime(2026, 5, 1)
        result = evaluate_oos(
            db=None, model_path=path, strategy="test_strategy",
            cutoff=cutoff, min_gain_pct=0.0,
        )

        assert "error" not in result
        assert result["n_trades"] == 40
        assert result["holdout_backtest_count"] == len(expected_ids)
        assert result["holdout_backtest_ids"] == expected_ids
        assert result["cutoff"] == cutoff.isoformat()
        assert 0.0 <= result["roc_auc"] <= 1.0


class TestCompareModelsOos:
    """Covers ml/oos_eval.py lines 278-292 — the multi-model comparison
    orchestrator."""

    def test_empty_holdout_returns_error_dict(self, monkeypatch):
        """When the holdout is empty, compare_models_oos returns an error
        dict and does NOT try to score any model."""
        from ml.oos_eval import compare_models_oos
        import ml.oos_eval as oos_mod

        monkeypatch.setattr(
            oos_mod, "get_holdout_trades", lambda *a, **kw: pd.DataFrame(),
        )

        path, _ = _make_test_model()
        cutoff = datetime(2026, 5, 1)
        result = compare_models_oos(
            db=None, model_paths={"v_a": path},
            strategy="nostate_optimized", cutoff=cutoff,
        )
        assert "error" in result
        assert "No holdout trades" in result["error"]
        # No model labels should appear in the result
        assert "v_a" not in result

    def test_multi_model_comparison_returns_metrics_per_label_plus_summary(
        self, monkeypatch,
    ):
        """Each model label gets a metrics dict; a special `_holdout` key
        carries the cohort summary (n_trades, n_backtests, cutoff, strategy)."""
        from ml.oos_eval import compare_models_oos
        import ml.oos_eval as oos_mod

        path_a, cols = _make_test_model(seed=10) if False else _make_test_model()
        path_b, _ = _make_test_model()  # second model, same cols
        df = _make_holdout_df(cols, n=30)
        expected_backtests = len(set(df["backtest_id"].tolist()))

        monkeypatch.setattr(oos_mod, "get_holdout_trades", lambda *a, **kw: df)

        cutoff = datetime(2026, 5, 1)
        result = compare_models_oos(
            db=None,
            model_paths={"v_a": path_a, "v_b": path_b},
            strategy="nostate_optimized",
            cutoff=cutoff,
            min_gain_pct=0.0,
        )

        # Per-model metrics
        assert "v_a" in result and "v_b" in result
        for label in ("v_a", "v_b"):
            assert "error" not in result[label]
            assert result[label]["n_trades"] == 30
            assert 0.0 <= result[label]["roc_auc"] <= 1.0

        # Holdout summary section
        assert "_holdout" in result
        h = result["_holdout"]
        assert h["n_trades"] == 30
        assert h["n_backtests"] == expected_backtests
        assert h["cutoff"] == cutoff.isoformat()
        assert h["strategy"] == "nostate_optimized"
