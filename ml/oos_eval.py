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
    # Prefer the model's own feature_names_in_ (the actual training order)
    # over payload['feature_columns'] (which is importance-sorted in some
    # legacy save paths — bug in trainer's save_model call site, not yet
    # tracked separately). Fall back to the payload list, then global default.
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and len(model_feature_names) > 0:
        feature_cols = list(model_feature_names)
    else:
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

    # Top-decile WR: WR among the top 10% by predicted probability. This is
    # the right model-selection metric for our use case — production strategies
    # filter to top-tier candidates (score >= 72 + max_positions cap), so what
    # matters is "how good are this model's top picks?", not "how well does it
    # rank the entire distribution?". May 5 diagnostic showed v17's OOS AUC
    # advantage was concentrated in middle-rank discrimination that backtest
    # never reaches; v12's top-decile WR was actually slightly higher than v17's
    # despite v12's lower AUC. Don't conflate the two — AUC integrates over
    # all ranks, top-decile WR isolates the head.
    n = len(y)
    top_n = max(1, n // 10)
    top_idx = np.argsort(proba)[-top_n:]
    top_decile_wr = float(y[top_idx].mean())
    top_decile_mean_proba = float(proba[top_idx].mean())

    return {
        "n_trades": int(len(df)),
        "n_wins": int(y.sum()),
        "win_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "brier_score": round(float(brier_score_loss(y, proba)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "top_decile_wr": round(top_decile_wr, 4),
        "top_decile_n": int(top_n),
        "top_decile_mean_proba": round(top_decile_mean_proba, 4),
        "min_gain_pct": min_gain_pct,
        "model_path": str(model_path),
        "model_saved_at": payload.get("saved_at"),
        "model_feature_count": len(feature_cols),
    }


def top_n_wr_at_count(
    model_path: Union[str, Path],
    df: pd.DataFrame,
    n_picks: int,
    min_gain_pct: float = 10.0,
) -> dict:
    """Score a model on a holdout, return WR of its top-N picks.

    Useful for trade-count-matched comparison: when comparing v17 (or any
    candidate) against an incumbent, we want to know "if both models took
    the same NUMBER of trades, who had the higher WR?". Threshold-based
    comparison conflates discrimination with operating-point selection.

    Returns dict with at minimum {n_picks, top_n_wr, mean_proba}, plus
    {error: ...} on validation failure.
    """
    if df is None or df.empty:
        return {"error": "Empty trades DataFrame", "n_picks": n_picks}
    if "gain_pct" not in df.columns:
        return {"error": "DataFrame missing 'gain_pct' column", "n_picks": n_picks}
    if n_picks <= 0:
        return {"error": "n_picks must be positive", "n_picks": n_picks}
    if n_picks > len(df):
        return {
            "error": f"n_picks ({n_picks}) exceeds holdout size ({len(df)})",
            "n_picks": n_picks,
        }

    payload = _load_model_payload(model_path)
    model = payload["model"]
    fn = getattr(model, "feature_names_in_", None)
    feature_cols = list(fn) if fn is not None and len(fn) > 0 else (
        payload.get("feature_columns") or FEATURE_COLUMNS
    )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return {
            "error": f"DataFrame missing {len(missing)} model features",
            "missing_features": missing,
        }

    X = df[feature_cols].copy().fillna(0)
    y = (df["gain_pct"] > min_gain_pct).astype(int).values
    proba = model.predict_proba(X)[:, 1]

    top_idx = np.argsort(proba)[-n_picks:]
    return {
        "n_picks": int(n_picks),
        "n_holdout": int(len(df)),
        "top_n_wr": round(float(y[top_idx].mean()), 4),
        "mean_proba": round(float(proba[top_idx].mean()), 4),
        "min_gain_pct": min_gain_pct,
        "model_path": str(model_path),
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
