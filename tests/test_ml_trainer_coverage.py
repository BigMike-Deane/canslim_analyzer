"""Targeted coverage for ml/trainer.py branches missed by the scenario suite
in test_ml_pipeline.py.

23 lines closed across 12 tests:
- train_model data-accumulation log path (line 111)
- train_model label-collapse early return (lines 120-123)
- train_model_regression no-valid-CV-folds early return (lines 234-238)
- save_model feature-name recovery via calibrated_classifiers_ (lines 310-313)
- load_model "model not in payload" None return (line 343)
- load_model exception None return (lines 345-347)
- _walk_forward_cv fold-skip (lines 393-394)
- _per_regime_auc single-class regime returns None (lines 442-443)
- _train_baseline test-too-small (line 459) + exception (lines 469-470)
- _train_baseline_regression test-too-small (596) + exception (605-606)

Tests mock at the helper boundary (`_walk_forward_cv`, `_train_baseline`)
when running the full train_model is wasteful for the branch under test.
For pure helper coverage, the helpers are called directly with synthetic
numpy/pandas inputs.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from ml.trainer import (
    train_model,
    train_model_regression,
    save_model,
    load_model,
    _walk_forward_cv,
    _walk_forward_cv_regression,
    _per_regime_auc,
    _train_baseline,
    _train_baseline_regression,
)
from ml.feature_extractor import FEATURE_COLUMNS

# Import the test data fixture from the existing pipeline tests.
from tests.test_ml_pipeline import _make_labeled_df


class TestTrainModelEarlyReturns:
    def test_data_accumulation_mode_logs_when_under_min_total(self, caplog):
        # MIN_TRAINING_SAMPLES (80) <= len(X) < MIN_TOTAL_SAMPLES (200) -> log
        # the data-accumulation notice. Mock _walk_forward_cv + _train_baseline
        # to make the call fast and force the gate to fail (so we exit before
        # the expensive final-model fit).
        df = _make_labeled_df(n=150, win_rate=0.5)
        fake_cv = [{"roc_auc": 0.51, "accuracy": 0.5, "fold": 0,
                    "train_size": 100, "test_size": 50,
                    "per_regime_auc": {"bearish": None, "neutral": 0.51, "bullish": None}}]
        with caplog.at_level("INFO", logger="ml.trainer"), \
             patch("ml.trainer._walk_forward_cv", return_value=fake_cv), \
             patch("ml.trainer._train_baseline", return_value={"roc_auc": 0.50}):
            result = train_model(df, min_roc_auc=0.99)
        assert result["passed_gate"] is False
        # The accumulation message must have been logged
        assert any("Data accumulation mode" in r.message for r in caplog.records)

    def test_label_collapse_returns_error(self):
        # min_gain_pct=100 reclassifies every row as a loss (no synthetic gain
        # exceeds 50%), so y_win is all zeros -> only one class.
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_gain_pct=100.0)
        assert result["passed_gate"] is False
        assert "Label collapse" in result["error"]


class TestTrainRegressionEarlyReturn:
    def test_no_valid_cv_folds_returns_error(self):
        # n=99: MIN_TRAINING_SAMPLES (80) <= 99 passes the front gate, but
        # every fold's train slice (0.4/0.6/0.8 of 99 = 39/59/79) falls
        # below 80, so _walk_forward_cv_regression returns []. ALSO covers
        # lines 541-542 (per-fold warning + continue).
        df = _make_labeled_df(n=99, win_rate=0.5)
        result = train_model_regression(df)
        assert result["passed_gate"] is False
        assert result["model_type"] == "regression"
        assert "No valid CV folds" in result["error"]


class TestSaveModelCalibratedRecovery:
    def test_feature_names_recovered_from_calibrated_classifiers(self, tmp_path):
        # save_model accepts a CalibratedClassifierCV-shaped object: no
        # feature_names_in_ at the top level, but the inner .estimator on
        # each item in calibrated_classifiers_ has one. The fallback loop
        # at lines 310-313 walks them until it finds one. Patch joblib.dump
        # so the un-picklable MagicMock doesn't have to round-trip — we only
        # care that save_model assembled the right payload before dumping.
        inner = MagicMock()
        inner.estimator.feature_names_in_ = np.array(["a", "b", "c"])
        outer = MagicMock(spec=["calibrated_classifiers_", "method"])
        outer.calibrated_classifiers_ = [inner]
        outer.method = "isotonic"

        out_path = tmp_path / "model.joblib"
        with patch("ml.trainer.joblib.dump") as mock_dump:
            save_model(outer, metadata={}, path=out_path)
        payload = mock_dump.call_args.args[0]
        assert payload["feature_columns"] == ["a", "b", "c"]
        assert payload["metadata"]["calibration_method"] == "isotonic"


class TestLoadModelErrorPaths:
    def test_payload_without_model_key_returns_none(self, tmp_path):
        # File exists, loads fine, but missing the "model" key -> None
        import joblib
        bad_path = tmp_path / "bad.joblib"
        joblib.dump({"metadata": {"x": 1}}, bad_path)
        assert load_model(bad_path) is None

    def test_load_exception_returns_none(self, tmp_path):
        # joblib.load raises -> outer except returns None
        garbage = tmp_path / "garbage.joblib"
        garbage.write_text("not a real joblib file")
        assert load_model(garbage) is None


class TestWalkForwardCvFoldSkip:
    def test_skips_fold_when_train_too_small(self):
        # n=99 -> every fold's train slice (39/59/79) below MIN_TRAINING_SAMPLES.
        # All folds skip, results list is empty.
        n = 99
        X = pd.DataFrame(
            np.random.RandomState(0).rand(n, len(FEATURE_COLUMNS)),
            columns=FEATURE_COLUMNS,
        )
        y = np.random.RandomState(0).randint(0, 2, n)
        results = _walk_forward_cv(X, y)
        assert results == []


class TestPerRegimeAucSingleClass:
    def test_single_class_regime_returns_none(self):
        # 30 samples, all bullish (code=2), but all wins -> single class,
        # AUC undefined, function returns None for that regime.
        regime = np.full(30, 2)
        y_true = np.ones(30, dtype=int)
        y_prob = np.random.RandomState(0).rand(30)
        out = _per_regime_auc(y_true, y_prob, regime)
        assert out["bullish"] is None
        # Bearish + neutral have zero samples -> also None (covers the
        # n < 10 branch separately)
        assert out["bearish"] is None
        assert out["neutral"] is None


class TestTrainBaselineErrorPaths:
    def test_returns_error_when_test_size_under_10(self):
        # train_end = int(n * 0.8). n=40 -> test_size = 8 (< 10) -> error.
        n = 40
        X = pd.DataFrame(
            np.random.RandomState(0).rand(n, 3),
            columns=["f1", "f2", "f3"],
        )
        y = np.random.RandomState(0).randint(0, 2, n)
        result = _train_baseline(X, y)
        assert "Insufficient test data" in result["error"]

    def test_returns_error_on_logistic_regression_exception(self):
        # Force LogisticRegression.fit to raise -> outer except returns
        # {"error": str(e)}.
        n = 50
        X = pd.DataFrame(
            np.random.RandomState(0).rand(n, 3),
            columns=["f1", "f2", "f3"],
        )
        y = np.random.RandomState(0).randint(0, 2, n)
        with patch("ml.trainer.LogisticRegression") as MockLR:
            MockLR.return_value.fit.side_effect = ValueError("singular matrix")
            result = _train_baseline(X, y)
        assert result["error"] == "singular matrix"


class TestTrainBaselineRegressionErrorPaths:
    def test_returns_error_when_test_size_under_10(self):
        # n=40 -> test_size = 8 -> error.
        n = 40
        X = pd.DataFrame(
            np.random.RandomState(0).rand(n, 3),
            columns=["f1", "f2", "f3"],
        )
        y = np.random.RandomState(0).rand(n)
        result = _train_baseline_regression(X, y)
        assert "Insufficient test data" in result["error"]

    def test_returns_error_on_ridge_exception(self):
        n = 50
        X = pd.DataFrame(
            np.random.RandomState(0).rand(n, 3),
            columns=["f1", "f2", "f3"],
        )
        y = np.random.RandomState(0).rand(n)
        with patch("ml.trainer.Ridge") as MockRidge:
            MockRidge.return_value.fit.side_effect = RuntimeError("ridge boom")
            result = _train_baseline_regression(X, y)
        assert result["error"] == "ridge boom"
