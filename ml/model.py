"""
Thread-safe ML model wrapper for production use.

Lazy-loads model from disk on first call. Returns None on any error
for graceful fallback — never crashes the trading pipeline.

Supports two model types:
  - classifier: returns P(win) via predict_proba
  - regression: returns predicted gain_pct mapped to [0, 1] confidence via sigmoid
"""

import logging
import threading
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_cached_model = None
_cached_metadata = None
_load_attempted = False

# Sigmoid scale for mapping predicted gain_pct to [0, 1] confidence.
# At scale=10: predicted +10% → 0.73, 0% → 0.50, -10% → 0.27
_GAIN_SIGMOID_SCALE = 10.0

# Prediction cache — keyed by feature values, reset each scan cycle
_prediction_cache: dict = {}
_cache_hits: int = 0
_cache_misses: int = 0


def _predict_from_payload(model, metadata: dict, features: dict) -> Optional[float]:
    """Core prediction logic shared by global and per-payload predict paths.

    Returns confidence in [0.0, 1.0], or None if features can't be prepared
    or the model produces NaN. No caching — caller manages cache if desired.
    """
    # Prefer the model's own feature_names_in_ (training order) over
    # metadata.feature_columns (importance-sorted in some legacy save paths,
    # fixed in trainer commit 561e33c — fallback retained for backward compat).
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and len(model_feature_names) > 0:
        feature_columns = list(model_feature_names)
    else:
        feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        return None

    row = {col: features.get(col, 0) for col in feature_columns}
    for k, v in row.items():
        if isinstance(v, bool):
            row[k] = int(v)

    X = pd.DataFrame([row], columns=feature_columns).fillna(0)
    model_type = metadata.get("metadata", {}).get("model_type", "classifier")

    if model_type == "regression":
        predicted_gain = float(model.predict(X)[0])
        confidence = 1.0 / (1.0 + np.exp(-predicted_gain / _GAIN_SIGMOID_SCALE))
    else:
        proba = model.predict_proba(X)[:, 1]
        confidence = float(proba[0])

    confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
    if confidence != confidence:  # NaN != NaN per IEEE 754
        return None
    return confidence


def get_ml_prediction_with_model(payload: dict, **features) -> Optional[float]:
    """Predict using an explicitly provided model payload — used by the eval
    backtest gate to score candidate models without polluting the global cache.

    payload: a dict matching ml.trainer.load_model output (must contain
    'model' and either model.feature_names_in_ or 'feature_columns').
    Bypasses the global model cache and prediction cache. Never raises.
    """
    if not payload or "model" not in payload:
        return None
    try:
        return _predict_from_payload(payload["model"], payload, features)
    except Exception as e:
        logger.error(f"ML prediction (per-payload) error: {e}")
        return None


def get_ml_prediction(**features) -> Optional[float]:
    """
    Returns confidence in [0.0, 1.0] or None if model unavailable.

    For classifier models: returns P(win) via predict_proba.
    For regression models: returns sigmoid(predicted_gain / scale).

    Thread-safe, lazy-loaded. Never raises — returns None on any error.
    Caches predictions per feature set within a scan cycle.
    """
    global _cache_hits, _cache_misses

    try:
        model, metadata = _get_model()
        if model is None:
            return None

        model_feature_names = getattr(model, "feature_names_in_", None)
        if model_feature_names is not None and len(model_feature_names) > 0:
            feature_columns = list(model_feature_names)
        else:
            feature_columns = metadata.get("feature_columns")
        if not feature_columns:
            return None

        row = {col: features.get(col, 0) for col in feature_columns}
        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)

        cache_key = tuple(sorted((k, round(v, 6) if isinstance(v, float) else v) for k, v in row.items()))
        if cache_key in _prediction_cache:
            _cache_hits += 1
            return _prediction_cache[cache_key]
        _cache_misses += 1

        confidence = _predict_from_payload(model, metadata, features)
        if confidence is None:
            logger.warning("ML model produced NaN confidence or missing features — returning None")
            return None

        _prediction_cache[cache_key] = confidence
        return confidence

    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return None


def clear_prediction_cache():
    """Reset prediction cache. Call at the start of each scan/evaluation cycle."""
    global _prediction_cache, _cache_hits, _cache_misses
    _prediction_cache = {}
    _cache_hits = 0
    _cache_misses = 0


def get_prediction_cache_stats() -> dict:
    """Get prediction cache statistics."""
    return {
        "size": len(_prediction_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
    }


def reload_model():
    """Force reload model from disk on next prediction."""
    global _cached_model, _cached_metadata, _load_attempted
    with _model_lock:
        _cached_model = None
        _cached_metadata = None
        _load_attempted = False


def is_model_loaded() -> bool:
    """Check if a model is currently loaded."""
    return _cached_model is not None


def _get_model():
    """Lazy-load model with thread safety."""
    global _cached_model, _cached_metadata, _load_attempted

    if _cached_model is not None:
        return _cached_model, _cached_metadata

    with _model_lock:
        # Double-check after acquiring lock
        if _cached_model is not None:
            return _cached_model, _cached_metadata

        if _load_attempted:
            return None, None

        _load_attempted = True

        try:
            from ml.trainer import load_model
            payload = load_model()
            if payload is None:
                logger.info("No ML model found on disk — predictions disabled")
                return None, None

            _cached_model = payload["model"]
            _cached_metadata = payload
            logger.info(f"ML model loaded (saved: {payload.get('saved_at', 'unknown')})")
            return _cached_model, _cached_metadata

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None, None
