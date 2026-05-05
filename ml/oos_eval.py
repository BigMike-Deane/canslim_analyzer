"""Out-of-sample (OOS) ML model evaluation.

Distinct from walk-forward CV (which evaluates within the training set):
this module evaluates a trained model file against trades the model has
never seen — the only honest way to compare candidate models.

Workflow:
    1. Pick a cutoff datetime T.
    2. Train candidate models using only backtests created before T
       (the trainer's contamination-free dedup pool already enforces this
       when given an explicit backtest_ids list).
    3. Use evaluate_oos(db, model_path, strategy, cutoff=T) to score
       each candidate against trades from backtests created after T.
    4. Compare AUC / Brier / win-rate across candidates.

Why not just walk-forward CV? Walk-forward CV is in-sample: every fold's
training data has the same provenance bias as the full set. OOS evaluates
on trades that genuinely couldn't have leaked into training, including
the dedup pool composition. v12 vs v17 backtest comparison (May 5)
showed v12 winning by 20pp — but both were trained on overlapping pools.
A clean OOS comparison is the only way to break that tie.

Loading models: any saved model file path works (active joblib,
experimental v17 file, or a future versioned file).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sqlalchemy.orm import Session

from ml.feature_extractor import (
    FEATURE_COLUMNS,
    extract_training_data,
)

logger = logging.getLogger(__name__)


def _load_model_payload(model_path: Union[str, Path]) -> dict:
    """Load a model joblib payload. Raises on any error."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    if "model" not in payload:
        raise ValueError(f"Invalid model payload at {path}: missing 'model' key")
    return payload


def evaluate_model_on_trades(
    model_path: Union[str, Path],
    df: pd.DataFrame,
    min_gain_pct: float = 10.0,
) -> dict:
    """Evaluate a saved model on a trades DataFrame.

    The DataFrame must contain the model's feature_columns plus a
    `gain_pct` column. Labels are derived as `gain_pct > min_gain_pct`
    to match the v12+ training convention (positive class = "big winner").

    Returns dict with metrics, or {"error": ...} on validation failure.
    Never raises on data shape issues — explicit error in the result.
    """
    if df is None or df.empty:
        return {"error": "Empty trades DataFrame"}
    if "gain_pct" not in df.columns:
        return {"error": "DataFrame missing 'gain_pct' column"}

    payload = _load_model_payload(model_path)
    model = payload["model"]
    feature_cols = payload.get("feature_columns") or FEATURE_COLUMNS

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {
            "error": f"DataFrame missing {len(missing)} model features",
            "missing_features": missing,
        }

    X = df[feature_cols].copy().fillna(0)
    y = (df["gain_pct"] > min_gain_pct).astype(int).values

    if len(np.unique(y)) < 2:
        return {
            "error": "Holdout has only one class — can't compute AUC",
            "n_trades": len(df),
            "win_rate": float(y.mean()),
        }

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "n_trades": int(len(df)),
        "n_wins": int(y.sum()),
        "win_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "brier_score": round(float(brier_score_loss(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "min_gain_pct": min_gain_pct,
        "model_path": str(model_path),
        "model_saved_at": payload.get("saved_at"),
        "model_feature_count": len(feature_cols),
    }


def get_holdout_trades(
    db: Session,
    strategy: str,
    cutoff: datetime,
    after: bool = True,
) -> pd.DataFrame:
    """Pull trades from completed backtests on one side of the cutoff.

    after=True  → backtests created on/after cutoff (use as OOS holdout)
    after=False → backtests created strictly before cutoff (training pool)

    Uses extract_training_data with explicit backtest_ids to bypass the
    dedup-and-exclude logic — we want ALL completed runs in the slice
    regardless of contamination status. Caller is responsible for picking
    a cutoff that places the right runs on each side.
    """
    from backend.database import BacktestRun

    q = db.query(BacktestRun.id).filter(
        BacktestRun.status == "completed",
        BacktestRun.strategy == strategy,
    )
    if after:
        q = q.filter(BacktestRun.created_at >= cutoff)
    else:
        q = q.filter(BacktestRun.created_at < cutoff)

    run_ids = [r.id for r in q.all()]
    if not run_ids:
        return pd.DataFrame()

    df, _ = extract_training_data(db, strategy=strategy, backtest_ids=run_ids)
    return df


def evaluate_oos(
    db: Session,
    model_path: Union[str, Path],
    strategy: str,
    cutoff: datetime,
    min_gain_pct: float = 10.0,
) -> dict:
    """Evaluate a model against backtests created on/after cutoff.

    Convenience wrapper that pulls the holdout slice and calls
    evaluate_model_on_trades. Caller is responsible for ensuring the
    model was actually trained on data before cutoff — there's no
    automatic check (would require storing train-time backtest_ids
    on the MLModel row).

    Returns dict with metrics + holdout backtest count for context.
    """
    holdout = get_holdout_trades(db, strategy, cutoff, after=True)
    if holdout.empty:
        return {
            "error": f"No holdout trades for {strategy} after {cutoff.isoformat()}",
            "cutoff": cutoff.isoformat(),
            "strategy": strategy,
        }

    holdout_backtest_ids = sorted(set(holdout["backtest_id"].tolist()))
    result = evaluate_model_on_trades(model_path, holdout, min_gain_pct=min_gain_pct)
    result["holdout_backtest_count"] = len(holdout_backtest_ids)
    result["holdout_backtest_ids"] = holdout_backtest_ids
    result["cutoff"] = cutoff.isoformat()
    return result


def compare_models_oos(
    db: Session,
    model_paths: dict,
    strategy: str,
    cutoff: datetime,
    min_gain_pct: float = 10.0,
) -> dict:
    """Evaluate multiple models on the same holdout slice.

    model_paths: {label: path} mapping (e.g. {"v12": "...", "v17": "..."}).
    Returns {label: metrics_dict} so callers can rank candidates side-by-side.
    """
    holdout = get_holdout_trades(db, strategy, cutoff, after=True)
    if holdout.empty:
        return {"error": f"No holdout trades for {strategy} after {cutoff.isoformat()}"}

    results = {}
    for label, path in model_paths.items():
        result = evaluate_model_on_trades(path, holdout, min_gain_pct=min_gain_pct)
        results[label] = result
    results["_holdout"] = {
        "n_trades": len(holdout),
        "n_backtests": len(set(holdout["backtest_id"].tolist())),
        "cutoff": cutoff.isoformat(),
        "strategy": strategy,
    }
    return results
