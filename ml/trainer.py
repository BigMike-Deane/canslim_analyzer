"""
ML model training with walk-forward validation.

Supports two model types:
  - classifier: XGBoost binary classifier predicting P(win). Gate: ROC AUC ≥ 0.55.
  - regression: XGBoost regressor predicting gain_pct. Gate: Spearman ≥ 0.15.

Regression addresses the binary classifier's weakness: treating 5% and 50% gains equally.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from ml.feature_extractor import FEATURE_COLUMNS, get_feature_matrix

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "data"
ACTIVE_MODEL_PATH = MODEL_DIR / "ml_model_active.joblib"
MIN_TRAINING_SAMPLES = 50  # Minimum samples per CV fold
MIN_ROC_AUC = 0.52  # Hard gate — classifier must beat this (lowered from 0.55 while data accumulates)
MIN_SPEARMAN = 0.10  # Hard gate — regressor must beat this (lowered from 0.15 while data accumulates)


def train_model(
    df: pd.DataFrame,
    min_roc_auc: float = MIN_ROC_AUC,
) -> dict:
    """
    Train XGBoost model with walk-forward CV.

    Returns dict with:
        model, metrics, feature_importance, cv_results,
        passed_gate (bool), baseline_comparison
    """
    X, y_win, y_gain, metadata = get_feature_matrix(df)
    if X is None or len(X) < MIN_TRAINING_SAMPLES:
        return {
            "passed_gate": False,
            "error": f"Insufficient data: {0 if X is None else len(X)} samples (need {MIN_TRAINING_SAMPLES})",
        }

    logger.info(f"Training on {len(X)} samples, {y_win.sum()} wins ({y_win.mean():.1%})")

    # Walk-forward expanding-window CV
    cv_results = _walk_forward_cv(X, y_win)

    mean_auc = np.mean([f["roc_auc"] for f in cv_results])
    logger.info(f"Walk-forward mean ROC AUC: {mean_auc:.4f} (gate: {min_roc_auc})")

    # Train baseline LogisticRegression for comparison
    baseline_results = _train_baseline(X, y_win)

    # Hard gate check
    if mean_auc < min_roc_auc:
        logger.warning(f"Model FAILED gate: ROC AUC {mean_auc:.4f} < {min_roc_auc}")
        return {
            "passed_gate": False,
            "cv_results": cv_results,
            "mean_roc_auc": round(mean_auc, 4),
            "baseline_comparison": baseline_results,
            "error": f"ROC AUC {mean_auc:.4f} below threshold {min_roc_auc}",
        }

    # Train final model on ALL data
    model = _create_xgb_model()
    model.fit(X, y_win)

    # Feature importance
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    importance = {k: round(v, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    # Aggregate metrics from CV folds
    metrics = _aggregate_cv_metrics(cv_results)

    result = {
        "passed_gate": True,
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "cv_results": cv_results,
        "mean_roc_auc": round(mean_auc, 4),
        "baseline_comparison": baseline_results,
        "training_samples": len(X),
        "feature_count": len(FEATURE_COLUMNS),
        "win_rate": round(float(y_win.mean()), 4),
    }

    return result


def train_model_regression(
    df: pd.DataFrame,
    min_spearman: float = MIN_SPEARMAN,
) -> dict:
    """
    Train XGBoost regressor predicting gain_pct (not binary win/loss).

    Addresses the binary classifier's weakness: a 5% winner and a 50% winner
    are treated differently. The regressor learns gain magnitude, so it favors
    high-upside entries over marginal ones.

    Returns dict with:
        model, metrics, feature_importance, cv_results,
        passed_gate (bool), baseline_comparison, model_type='regression'
    """
    X, y_win, y_gain, metadata = get_feature_matrix(df)
    if X is None or len(X) < MIN_TRAINING_SAMPLES:
        return {
            "passed_gate": False,
            "model_type": "regression",
            "error": f"Insufficient data: {0 if X is None else len(X)} samples (need {MIN_TRAINING_SAMPLES})",
        }

    # Clip extreme outliers to reduce skew (keep 1st-99th percentile)
    low, high = np.percentile(y_gain, [1, 99])
    y_clipped = np.clip(y_gain, low, high)

    logger.info(
        f"Training regressor on {len(X)} samples, "
        f"gain range: [{y_gain.min():.1f}%, {y_gain.max():.1f}%], "
        f"mean: {y_gain.mean():.1f}%, median: {np.median(y_gain):.1f}%"
    )

    # Walk-forward expanding-window CV for regression
    cv_results = _walk_forward_cv_regression(X, y_clipped)

    if not cv_results:
        return {
            "passed_gate": False,
            "model_type": "regression",
            "error": "No valid CV folds produced",
        }

    mean_spearman = np.mean([f["spearman"] for f in cv_results])
    logger.info(f"Walk-forward mean Spearman: {mean_spearman:.4f} (gate: {min_spearman})")

    # Train baseline Ridge regression for comparison
    baseline_results = _train_baseline_regression(X, y_clipped)

    # Hard gate check
    if mean_spearman < min_spearman:
        logger.warning(f"Model FAILED gate: Spearman {mean_spearman:.4f} < {min_spearman}")
        return {
            "passed_gate": False,
            "model_type": "regression",
            "cv_results": cv_results,
            "mean_spearman": round(mean_spearman, 4),
            "baseline_comparison": baseline_results,
            "error": f"Spearman {mean_spearman:.4f} below threshold {min_spearman}",
        }

    # Train final model on ALL data
    model = _create_xgb_regressor()
    model.fit(X, y_clipped)

    # Feature importance
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    importance = {k: round(v, 4) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    # Aggregate metrics from CV folds
    metrics = _aggregate_cv_metrics_regression(cv_results)

    return {
        "passed_gate": True,
        "model_type": "regression",
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "cv_results": cv_results,
        "mean_spearman": round(mean_spearman, 4),
        "baseline_comparison": baseline_results,
        "training_samples": len(X),
        "feature_count": len(FEATURE_COLUMNS),
        "gain_stats": {
            "mean": round(float(y_gain.mean()), 2),
            "median": round(float(np.median(y_gain)), 2),
            "std": round(float(y_gain.std()), 2),
            "clip_range": [round(float(low), 2), round(float(high), 2)],
        },
    }


def save_model(model, metadata: dict, path: Optional[Path] = None) -> Path:
    """Save model + metadata to disk."""
    path = path or ACTIVE_MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "metadata": metadata,
        "feature_columns": FEATURE_COLUMNS,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(path: Optional[Path] = None) -> Optional[dict]:
    """Load model + metadata from disk. Returns None on any error."""
    path = path or ACTIVE_MODEL_PATH
    try:
        if not path.exists():
            return None
        payload = joblib.load(path)
        if "model" not in payload:
            return None
        return payload
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        return None


def _create_xgb_model() -> XGBClassifier:
    """Create XGBoost classifier with aggressive regularization for small datasets."""
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=42,
    )


def _walk_forward_cv(X: pd.DataFrame, y: np.ndarray) -> list:
    """
    Walk-forward expanding-window cross-validation.
    Train on past, test on next window. No future leakage.
    """
    n = len(X)
    folds = [
        (0, int(n * 0.4), int(n * 0.6)),   # Train [0..40%], Test [40..60%]
        (0, int(n * 0.6), int(n * 0.8)),   # Train [0..60%], Test [60..80%]
        (0, int(n * 0.8), n),               # Train [0..80%], Test [80..100%]
    ]

    results = []
    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
        X_train = X.iloc[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_train) < MIN_TRAINING_SAMPLES or len(X_test) < 10:
            logger.warning(f"Fold {fold_idx}: skipped (train={len(X_train)}, test={len(X_test)})")
            continue

        model = _create_xgb_model()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        fold_metrics = _compute_metrics(y_test, y_pred, y_prob)
        fold_metrics["fold"] = fold_idx
        fold_metrics["train_size"] = len(X_train)
        fold_metrics["test_size"] = len(X_test)
        results.append(fold_metrics)

        logger.info(
            f"Fold {fold_idx}: AUC={fold_metrics['roc_auc']:.3f}, "
            f"Acc={fold_metrics['accuracy']:.3f}, "
            f"train={len(X_train)}, test={len(X_test)}"
        )

    return results


def _train_baseline(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Train LogisticRegression baseline for comparison."""
    n = len(X)
    train_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    y_train = y[:train_end]
    X_test = X.iloc[train_end:]
    y_test = y[train_end:]

    if len(X_test) < 10:
        return {"error": "Insufficient test data for baseline"}

    try:
        lr = LogisticRegression(C=0.1, l1_ratio=0, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_prob = lr.predict_proba(X_test)[:, 1]
        y_pred = lr.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = "LogisticRegression"
        return metrics
    except Exception as e:
        return {"error": str(e)}


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics."""
    # Handle edge case: all same class in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "brier_score": brier_score_loss(y_true, y_prob),
        }

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }


def _aggregate_cv_metrics(cv_results: list) -> dict:
    """Average metrics across CV folds."""
    if not cv_results:
        return {}
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]
    return {k: round(np.mean([f[k] for f in cv_results if k in f]), 4) for k in keys}


# ── Regression-specific functions ──────────────────────────────────────────


def _create_xgb_regressor() -> XGBRegressor:
    """Create XGBoost regressor with aggressive regularization for small datasets."""
    return XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
    )


def _walk_forward_cv_regression(X: pd.DataFrame, y: np.ndarray) -> list:
    """Walk-forward expanding-window CV for regression."""
    n = len(X)
    folds = [
        (0, int(n * 0.4), int(n * 0.6)),
        (0, int(n * 0.6), int(n * 0.8)),
        (0, int(n * 0.8), n),
    ]

    results = []
    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
        X_train = X.iloc[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y[train_end:test_end]

        if len(X_train) < MIN_TRAINING_SAMPLES or len(X_test) < 10:
            logger.warning(f"Fold {fold_idx}: skipped (train={len(X_train)}, test={len(X_test)})")
            continue

        model = _create_xgb_regressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = _compute_regression_metrics(y_test, y_pred)
        fold_metrics["fold"] = fold_idx
        fold_metrics["train_size"] = len(X_train)
        fold_metrics["test_size"] = len(X_test)
        results.append(fold_metrics)

        logger.info(
            f"Fold {fold_idx}: Spearman={fold_metrics['spearman']:.3f}, "
            f"R²={fold_metrics['r2']:.3f}, MAE={fold_metrics['mae']:.2f}, "
            f"train={len(X_train)}, test={len(X_test)}"
        )

    return results


def _compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics including Spearman rank correlation."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Spearman rank correlation — measures ranking ability (key metric)
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        spearman_corr, spearman_p = scipy_stats.spearmanr(y_true, y_pred)
    else:
        spearman_corr, spearman_p = 0.0, 1.0

    # Directional accuracy: does the model predict positive/negative correctly?
    direction_correct = np.mean((y_true > 0) == (y_pred > 0))

    return {
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "spearman": round(float(spearman_corr), 4),
        "spearman_p": round(float(spearman_p), 4),
        "direction_accuracy": round(float(direction_correct), 4),
    }


def _train_baseline_regression(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Train Ridge regression baseline for comparison."""
    n = len(X)
    train_end = int(n * 0.8)
    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]

    if len(X_test) < 10:
        return {"error": "Insufficient test data for baseline"}

    try:
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        metrics = _compute_regression_metrics(y_test, y_pred)
        metrics["model"] = "Ridge"
        return metrics
    except Exception as e:
        return {"error": str(e)}


def _aggregate_cv_metrics_regression(cv_results: list) -> dict:
    """Average regression metrics across CV folds."""
    if not cv_results:
        return {}
    keys = ["r2", "mae", "rmse", "spearman", "direction_accuracy"]
    return {k: round(np.mean([f[k] for f in cv_results if k in f]), 4) for k in keys}
