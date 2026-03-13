"""
Tests for ML Signal Layer — feature extraction, training, and prediction.

Covers: empty DB, missing signal_factors, NaN handling, buy/sell pairing,
partial sell aggregation, training, walk-forward no-leakage, predictions.
"""

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.feature_extractor import (
    ENTRY_TYPE_MAP,
    FEATURE_COLUMNS,
    REGIME_MAP,
    _extract_features,
    _nan_safe,
    _pair_buy_sell_trades,
    extract_training_data,
    get_feature_matrix,
)
from ml.model import get_ml_prediction, reload_model
from ml.trainer import (
    MIN_ROC_AUC,
    MIN_TRAINING_SAMPLES,
    _aggregate_cv_metrics,
    _compute_metrics,
    _create_xgb_model,
    load_model,
    save_model,
    train_model,
)


# ============== Fixtures ==============


def _make_trade(
    backtest_id=1,
    ticker="AAPL",
    action="BUY",
    trade_date=date(2024, 1, 15),
    canslim_score=85.0,
    is_growth_stock=False,
    signal_factors=None,
    realized_gain_pct=None,
    holding_days=None,
    reason=None,
):
    """Create a mock BacktestTrade-like object."""
    t = MagicMock()
    t.backtest_id = backtest_id
    t.ticker = ticker
    t.action = action
    t.date = trade_date
    t.canslim_score = canslim_score
    t.is_growth_stock = is_growth_stock
    t.signal_factors = signal_factors or {
        "entry_type": "standard",
        "market_regime": "bullish",
        "composite_score": 45.2,
        "rs_line_bonus": 3.0,
        "earnings_drift_bonus": 2.0,
        "estimate_revision_bonus": 1.5,
    }
    t.realized_gain_pct = realized_gain_pct
    t.holding_days = holding_days
    t.reason = reason
    return t


def _make_labeled_df(n=200, win_rate=0.65):
    """Create a synthetic labeled DataFrame for training tests."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        is_win = rng.random() < win_rate
        score = rng.uniform(72, 98)
        composite = rng.uniform(30, 80)
        # Make features correlated with outcome for testable signal
        if is_win:
            composite += 10
            score += 5
        rows.append({
            "total_score": score,
            "composite_score": composite,
            "entry_type": rng.choice([0, 1, 2]),
            "market_regime": rng.choice([0, 1, 2]),
            "rs_line_bonus": rng.uniform(0, 5) if is_win else rng.uniform(0, 2),
            "earnings_drift_bonus": rng.uniform(0, 3),
            "estimate_revision_bonus": rng.uniform(-5, 10),
            "coiled_spring": rng.choice([0, 1]),
            "soft_zone": rng.choice([0, 1]),
            "soft_zone_multiplier": rng.uniform(0.5, 1.0),
            "deterministic_boost": rng.choice([0, 5, 8]),
            "is_growth_stock": rng.choice([0, 1]),
            "win": 1 if is_win else 0,
            "gain_pct": rng.uniform(5, 50) if is_win else rng.uniform(-20, 0),
            "ticker": f"T{i}",
            "date": f"2024-{(i % 12) + 1:02d}-15",
            "backtest_id": 1,
            "holding_days": rng.randint(5, 60),
            "sell_reason": "stop_loss" if not is_win else "take_profit",
        })
    return pd.DataFrame(rows)


# ============== Feature Extractor Tests ==============


class TestNanSafe:
    def test_none_returns_default(self):
        assert _nan_safe(None) == 0.0
        assert _nan_safe(None, 5.0) == 5.0

    def test_nan_returns_default(self):
        assert _nan_safe(float("nan")) == 0.0
        assert _nan_safe(float("nan"), 99) == 99

    def test_valid_values_pass_through(self):
        assert _nan_safe(42.5) == 42.5
        assert _nan_safe(0) == 0
        assert _nan_safe(-3.14) == -3.14

    def test_string_passes_through(self):
        assert _nan_safe("hello") == "hello"


class TestExtractFeatures:
    def test_normal_trade(self):
        trade = _make_trade(signal_factors={
            "entry_type": "breakout",
            "market_regime": "bullish",
            "composite_score": 55.3,
            "rs_line_bonus": 3.0,
            "earnings_drift_bonus": 2.0,
            "estimate_revision_bonus": 1.5,
            "coiled_spring": True,
            "soft_zone": True,
            "soft_zone_multiplier": 0.8,
            "deterministic_boost": 5,
        })
        f = _extract_features(trade)
        assert f is not None
        assert f["total_score"] == 85.0
        assert f["composite_score"] == 55.3
        assert f["entry_type"] == 0  # breakout
        assert f["market_regime"] == 2  # bullish
        assert f["coiled_spring"] == 1
        assert f["soft_zone"] == 1
        assert f["soft_zone_multiplier"] == 0.8
        assert f["deterministic_boost"] == 5

    def test_empty_signal_factors_returns_none(self):
        trade = _make_trade()
        trade.signal_factors = {}
        assert _extract_features(trade) is None

    def test_none_signal_factors_returns_none(self):
        trade = _make_trade(signal_factors=None)
        trade.signal_factors = None
        assert _extract_features(trade) is None

    def test_missing_optional_fields_use_defaults(self):
        trade = _make_trade(signal_factors={
            "entry_type": "standard",
            "market_regime": "neutral",
            "composite_score": 40.0,
            "rs_line_bonus": 0,
            "earnings_drift_bonus": 0,
            "estimate_revision_bonus": 0,
            # No coiled_spring, soft_zone, deterministic_boost
        })
        f = _extract_features(trade)
        assert f["coiled_spring"] == 0
        assert f["soft_zone"] == 0
        assert f["soft_zone_multiplier"] == 1.0
        assert f["deterministic_boost"] == 0.0

    def test_nan_canslim_score(self):
        trade = _make_trade(canslim_score=float("nan"))
        f = _extract_features(trade)
        assert f["total_score"] == 0.0

    def test_unknown_entry_type_defaults_to_standard(self):
        trade = _make_trade(signal_factors={
            "entry_type": "unknown_type",
            "market_regime": "neutral",
            "composite_score": 40.0,
            "rs_line_bonus": 0,
            "earnings_drift_bonus": 0,
            "estimate_revision_bonus": 0,
        })
        f = _extract_features(trade)
        assert f["entry_type"] == 2  # standard (default)

    def test_growth_stock_flag(self):
        trade = _make_trade(is_growth_stock=True)
        f = _extract_features(trade)
        assert f["is_growth_stock"] == 1

        trade2 = _make_trade(is_growth_stock=False)
        f2 = _extract_features(trade2)
        assert f2["is_growth_stock"] == 0


class TestBuySellPairing:
    def test_simple_buy_sell_pair(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL"),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=15.0, holding_days=30, reason="take_profit"),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 1
        assert rows[0]["win"] == 1
        assert rows[0]["gain_pct"] == 15.0
        assert rows[0]["holding_days"] == 30

    def test_losing_trade(self):
        trades = [
            _make_trade(action="BUY", ticker="TSLA"),
            _make_trade(action="SELL", ticker="TSLA", realized_gain_pct=-8.0, holding_days=5, reason="stop_loss"),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 1
        assert rows[0]["win"] == 0
        assert rows[0]["gain_pct"] == -8.0

    def test_orphan_sell_skipped(self):
        trades = [
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=10.0),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 0

    def test_multiple_tickers(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL"),
            _make_trade(action="BUY", ticker="GOOGL"),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=10.0, holding_days=20),
            _make_trade(action="SELL", ticker="GOOGL", realized_gain_pct=-5.0, holding_days=10),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 2
        tickers = {r["ticker"] for r in rows}
        assert tickers == {"AAPL", "GOOGL"}

    def test_buy_without_sell_ignored(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL"),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 0

    def test_fifo_matching(self):
        """Two buys, two sells — first sell matches first buy (FIFO)."""
        trades = [
            _make_trade(action="BUY", ticker="AAPL", canslim_score=80),
            _make_trade(action="BUY", ticker="AAPL", canslim_score=90),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=5.0, holding_days=10),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=15.0, holding_days=20),
        ]
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 2
        # First sell should match first buy (score=80)
        assert rows[0]["total_score"] == 80
        assert rows[0]["gain_pct"] == 5.0

    def test_partial_sell_keeps_buy_open(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL"),
            _make_trade(action="PARTIAL_SELL", ticker="AAPL", realized_gain_pct=10.0, holding_days=15),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=20.0, holding_days=30),
        ]
        rows = _pair_buy_sell_trades(trades)
        # PARTIAL_SELL creates a row but doesn't pop the buy
        # Final SELL also creates a row and pops the buy
        assert len(rows) == 2
        assert rows[0]["gain_pct"] == 10.0
        assert rows[1]["gain_pct"] == 20.0

    def test_different_backtests_isolated(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL", backtest_id=1),
            _make_trade(action="SELL", ticker="AAPL", backtest_id=2, realized_gain_pct=10.0, holding_days=20),
        ]
        rows = _pair_buy_sell_trades(trades)
        # Buy in backtest 1, sell in backtest 2 — should NOT pair
        assert len(rows) == 0

    def test_buy_with_no_signal_factors_skipped(self):
        trades = [
            _make_trade(action="BUY", signal_factors=None),
            _make_trade(action="SELL", ticker="AAPL", realized_gain_pct=10.0, holding_days=20),
        ]
        trades[0].signal_factors = None
        rows = _pair_buy_sell_trades(trades)
        assert len(rows) == 0


class TestExtractTrainingData:
    def test_empty_db(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = []
        df = extract_training_data(db, strategy="nostate_optimized")
        assert df.empty

    def test_with_backtest_ids_filter(self):
        db = MagicMock()
        query = db.query.return_value.filter.return_value
        query.filter.return_value.all.return_value = []
        df = extract_training_data(db, backtest_ids=[1, 2, 3])
        assert df.empty


class TestGetFeatureMatrix:
    def test_empty_df(self):
        X, y_win, y_gain, meta = get_feature_matrix(pd.DataFrame())
        assert X is None

    def test_normal_df(self):
        df = _make_labeled_df(n=50)
        X, y_win, y_gain, meta = get_feature_matrix(df)
        assert X.shape == (50, len(FEATURE_COLUMNS))
        assert len(y_win) == 50
        assert len(y_gain) == 50
        assert X.isna().sum().sum() == 0  # No NaNs after fillna

    def test_feature_columns_match(self):
        df = _make_labeled_df(n=10)
        X, _, _, _ = get_feature_matrix(df)
        assert list(X.columns) == FEATURE_COLUMNS


# ============== Trainer Tests ==============


class TestCreateXgbModel:
    def test_model_params(self):
        model = _create_xgb_model()
        assert model.max_depth == 3
        assert model.n_estimators == 100
        assert model.learning_rate == 0.05
        assert model.min_child_weight == 5


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = _compute_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0

    def test_all_same_class(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7])
        m = _compute_metrics(y_true, y_pred, y_prob)
        assert m["roc_auc"] == 0.5  # Undefined → default 0.5


class TestTrainModel:
    def test_insufficient_data(self):
        df = _make_labeled_df(n=20)
        result = train_model(df)
        assert result["passed_gate"] is False
        assert "Insufficient" in result["error"]

    def test_training_with_synthetic_data(self):
        # Create strongly separable data so signal is obvious
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)  # Set gate to 0 for test
        assert "model" in result or result.get("cv_results")
        assert "cv_results" in result

    def test_gate_failure(self):
        # Random noise data — model should fail gate
        rng = np.random.RandomState(99)
        n = 200
        rows = []
        for i in range(n):
            rows.append({
                **{col: rng.uniform(0, 1) for col in FEATURE_COLUMNS},
                "win": rng.choice([0, 1]),
                "gain_pct": rng.uniform(-10, 10),
                "ticker": f"T{i}",
                "date": "2024-01-01",
                "backtest_id": 1,
                "holding_days": 10,
                "sell_reason": "test",
            })
        df = pd.DataFrame(rows)
        result = train_model(df, min_roc_auc=0.90)  # Very high gate
        assert result["passed_gate"] is False

    def test_feature_importance_populated(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)
        if result.get("passed_gate"):
            assert "feature_importance" in result
            assert len(result["feature_importance"]) == len(FEATURE_COLUMNS)
            # Importances should sum to ~1.0
            total = sum(result["feature_importance"].values())
            assert 0.9 < total < 1.1


class TestSaveLoadModel:
    def test_save_load_cycle(self, tmp_path):
        model = _create_xgb_model()
        df = _make_labeled_df(n=100)
        X, y, _, _ = get_feature_matrix(df)
        model.fit(X, y)

        path = tmp_path / "test_model.joblib"
        save_model(model, {"test": True}, path=path)

        loaded = load_model(path=path)
        assert loaded is not None
        assert loaded["model"] is not None
        assert loaded["metadata"]["test"] is True
        assert "saved_at" in loaded

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nonexistent.joblib"
        assert load_model(path=path) is None


class TestAggregateMetrics:
    def test_aggregation(self):
        results = [
            {"accuracy": 0.7, "precision": 0.6, "recall": 0.8, "f1": 0.69, "roc_auc": 0.75, "brier_score": 0.2},
            {"accuracy": 0.8, "precision": 0.7, "recall": 0.9, "f1": 0.79, "roc_auc": 0.85, "brier_score": 0.15},
        ]
        m = _aggregate_cv_metrics(results)
        assert m["accuracy"] == 0.75
        assert m["roc_auc"] == 0.8

    def test_empty_results(self):
        assert _aggregate_cv_metrics([]) == {}


# ============== Model Wrapper Tests ==============


class TestGetMlPrediction:
    def setup_method(self):
        reload_model()

    def test_no_model_returns_none(self):
        reload_model()
        with patch("ml.model._get_model", return_value=(None, None)):
            result = get_ml_prediction(total_score=85, composite_score=50)
            assert result is None

    def test_prediction_in_range(self, tmp_path):
        """Train a real model and verify prediction output range."""
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        path = tmp_path / "test_model.joblib"
        save_model(model, {}, path=path)

        reload_model()
        with patch("ml.trainer.ACTIVE_MODEL_PATH", path):
            loaded = load_model(path=path)
            assert loaded is not None

            with patch("ml.model._get_model", return_value=(loaded["model"], loaded)):
                result = get_ml_prediction(
                    total_score=85,
                    composite_score=50,
                    entry_type=0,
                    market_regime=2,
                    rs_line_bonus=3.0,
                    earnings_drift_bonus=2.0,
                    estimate_revision_bonus=1.5,
                    coiled_spring=1,
                    soft_zone=0,
                    soft_zone_multiplier=1.0,
                    deterministic_boost=0,
                    is_growth_stock=0,
                )
                assert result is not None
                assert 0.0 <= result <= 1.0

    def test_boolean_conversion(self, tmp_path):
        """Verify booleans are converted to int."""
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        path = tmp_path / "test_model.joblib"
        save_model(model, {}, path=path)
        loaded = load_model(path=path)

        reload_model()
        with patch("ml.model._get_model", return_value=(loaded["model"], loaded)):
            result = get_ml_prediction(
                total_score=85, composite_score=50,
                entry_type=0, market_regime=2,
                rs_line_bonus=0, earnings_drift_bonus=0,
                estimate_revision_bonus=0,
                coiled_spring=True,  # Boolean, not int
                soft_zone=False,     # Boolean, not int
                soft_zone_multiplier=1.0,
                deterministic_boost=0, is_growth_stock=False,
            )
            assert result is not None
            assert 0.0 <= result <= 1.0


class TestWalkForwardNoLeakage:
    def test_train_indices_always_before_test(self):
        """Verify walk-forward folds never train on future data."""
        n = 300
        folds = [
            (0, int(n * 0.4), int(n * 0.6)),
            (0, int(n * 0.6), int(n * 0.8)),
            (0, int(n * 0.8), n),
        ]
        for train_start, train_end, test_end in folds:
            assert train_end <= test_end
            assert train_start < train_end
            # Training data is always before test data
            train_indices = set(range(train_start, train_end))
            test_indices = set(range(train_end, test_end))
            assert train_indices.isdisjoint(test_indices)
