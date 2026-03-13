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


def get_ml_prediction(**features) -> Optional[float]:
    """
    Returns confidence in [0.0, 1.0] or None if model unavailable.

    For classifier models: returns P(win) via predict_proba.
    For regression models: returns sigmoid(predicted_gain / scale).

    Thread-safe, lazy-loaded. Never raises — returns None on any error.
    """
    try:
        model, metadata = _get_model()
        if model is None:
            return None

        feature_columns = metadata.get("feature_columns")
        if not feature_columns:
            return None

        # Build feature vector in correct column order
        row = {col: features.get(col, 0) for col in feature_columns}

        # Convert booleans to int
        for k, v in row.items():
            if isinstance(v, bool):
                row[k] = int(v)

        X = pd.DataFrame([row], columns=feature_columns)
        X = X.fillna(0)

        model_type = metadata.get("metadata", {}).get("model_type", "classifier")

        if model_type == "regression":
            # Regression: predicted gain_pct → sigmoid → [0, 1]
            predicted_gain = float(model.predict(X)[0])
            confidence = 1.0 / (1.0 + np.exp(-predicted_gain / _GAIN_SIGMOID_SCALE))
        else:
            # Classifier: P(win) directly
            proba = model.predict_proba(X)[:, 1]
            confidence = float(proba[0])

        return round(float(np.clip(confidence, 0.0, 1.0)), 4)

    except Exception as e:
        logger.error(f"ML prediction error: {e}")
        return None


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
