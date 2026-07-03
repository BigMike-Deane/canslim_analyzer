"""Per-trade ML model diagnostics: pairwise comparison on a holdout slice.

Sibling to ml/oos_eval.py. Where oos_eval reports headline metrics
(AUC, Brier, top-decile WR) for each model individually, this module
zooms into the *per-trade* disagreement structure between two or more
models on the same OOS holdout:

  - Probability distribution per model (binned counts + summary stats)
  - Pairwise Spearman rank correlation
  - Threshold disagreement at 0.30: only-A picks, only-B picks, both,
    neither, with WR per set
  - Top-decile WR per model (echoed from oos_eval for completeness)
  - Bottom-decile WR per model

Origin: built to investigate v12 vs v17 (May 5). v17 had a higher OOS
AUC but v12 still matched it in backtest. The disagreement breakdown
showed v12's *top-decile* picks slightly out-performed v17's despite
v12's lower AUC — i.e. the AUC delta was concentrated in middle-rank
discrimination that backtest never reaches (it filters to top-tier
candidates by score+max_positions). The original script lived only on
the VPS at /app/data/oos_disagreement.py; this module is the durable
in-repo replacement.

Loading models: reuses ml/oos_eval.py's model-payload pattern. Prefer
model.feature_names_in_ over payload['feature_columns']: trainer's
save_model emits importance-sorted feature lists in some legacy paths,
which xgboost rejects on predict() with feature_names_mismatch
(commits 1bbf4c9 + 61d55fb worked around this in live + OOS).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.feature_extractor import FEATURE_COLUMNS
from ml.oos_eval import _load_model_payload, positive_class_scores

logger = logging.getLogger(__name__)


# Default probability bin edges. 0.30 is the production veto threshold,
# so it has its own dedicated edge — letting callers see at-a-glance how
# many trades each model places in the gated/non-gated buckets.
_DEFAULT_BIN_EDGES: List[float] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]


def _resolve_feature_columns(payload: dict) -> List[str]:
    """Same feature-order resolution as evaluate_model_on_trades.

    Prefer model.feature_names_in_, fall back to payload['feature_columns'],
    then global FEATURE_COLUMNS. Kept here as a private mirror so callers
    of score_holdout_with_models don't need to know the mechanics.
    """
    model = payload["model"]
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    return payload.get("feature_columns") or list(FEATURE_COLUMNS)


def score_holdout_with_models(
    model_paths: Dict[str, Union[str, Path]],
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Score a holdout DataFrame with each model, return {label: probs}.

    Each model is loaded independently and scored using its own training
    feature order — different model versions can have different feature
    sets and the caller doesn't have to align them.

    Raises FileNotFoundError on missing model file (consistent with
    evaluate_model_on_trades). For per-model column mismatches, the
    label is OMITTED from the output (caller can detect the gap by
    comparing label sets) — partial results are more useful than failing
    the whole batch.
    """
    if df is None or df.empty:
        return {}

    out: Dict[str, np.ndarray] = {}
    for label, path in model_paths.items():
        payload = _load_model_payload(path)
        feature_cols = _resolve_feature_columns(payload)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(
                "score_holdout_with_models: skipping label=%s — holdout missing %d features: %s",
                label, len(missing), missing[:5],
            )
            continue
        X = df[feature_cols].copy().fillna(0)
        # Regressors (default /train mode) have no predict_proba — the old
        # direct call raised AttributeError → swallowed → decile-WR gate hard
        # fail, making regression candidates structurally un-promotable.
        proba = positive_class_scores(payload["model"], X)
        out[label] = np.asarray(proba, dtype=float)
    return out


def probability_distribution(
    probs: np.ndarray,
    bin_edges: Optional[List[float]] = None,
) -> dict:
    """Bin a probability vector and return counts + summary stats.

    bin_edges defaults to deciles + the 0.30 veto threshold. Use a custom
    edge list if you want to align with a different gate.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None,
            "min": None, "max": None, "bins": [], "counts": [],
        }
    edges = list(bin_edges) if bin_edges is not None else list(_DEFAULT_BIN_EDGES)
    counts, edges_out = np.histogram(arr, bins=edges)
    bins_labeled = [
        f"[{edges_out[i]:.2f}, {edges_out[i+1]:.2f}{')' if i < len(edges_out) - 2 else ']'}"
        for i in range(len(edges_out) - 1)
    ]
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "bins": bins_labeled,
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges_out],
    }


def pairwise_rank_correlation(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> float:
    """Spearman rank correlation between two probability vectors.

    Returns NaN-safe float. When either vector is empty/constant, scipy
    returns NaN — we surface that as None so JSON serialization doesn't
    fail downstream. Length mismatch raises ValueError (caller bug, not
    a data condition).
    """
    a = np.asarray(probs_a, dtype=float)
    b = np.asarray(probs_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: {a.shape} vs {b.shape}")
    if a.size < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    if rho is None or np.isnan(rho):
        return float("nan")
    return float(rho)


def disagreement_at_threshold(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.30,
) -> dict:
    """Decompose two models' picks at a threshold into 4 sets with WRs.

    only_a: A picks, B doesn't
    only_b: B picks, A doesn't
    both:   both pick
    neither: neither picks

    For each set, returns count and win rate (mean of y in that set).
    Win rate is None when the set is empty.

    y must be a binary 0/1 label aligned with probs_a/probs_b.
    """
    a = np.asarray(probs_a, dtype=float)
    b = np.asarray(probs_b, dtype=float)
    yarr = np.asarray(y).astype(int)
    if not (a.shape == b.shape == yarr.shape):
        raise ValueError(
            f"Length mismatch: probs_a={a.shape}, probs_b={b.shape}, y={yarr.shape}"
        )

    pick_a = a >= threshold
    pick_b = b >= threshold

    masks = {
        "only_a": pick_a & ~pick_b,
        "only_b": pick_b & ~pick_a,
        "both": pick_a & pick_b,
        "neither": ~pick_a & ~pick_b,
    }

    out: Dict[str, dict] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            wr = None
        else:
            wr = round(float(yarr[mask].mean()), 4)
        out[name] = {"n": n, "wr": wr}

    out["threshold"] = threshold
    out["n_total"] = int(yarr.size)
    out["pick_a_total"] = int(pick_a.sum())
    out["pick_b_total"] = int(pick_b.sum())
    return out


def decile_wr(
    probs: np.ndarray,
    y: np.ndarray,
    top: bool = True,
    decile: float = 0.10,
) -> dict:
    """WR among the top (or bottom) `decile` of trades ranked by `probs`.

    decile=0.10, top=True → top 10% by predicted probability
    decile=0.10, top=False → bottom 10%

    Useful for quantifying how good (or how bad) a model's tail picks
    are. The May 5 v12-vs-v17 diagnostic showed v12's top decile WR was
    actually marginally higher despite a lower AUC.
    """
    a = np.asarray(probs, dtype=float)
    yarr = np.asarray(y).astype(int)
    if a.shape != yarr.shape:
        raise ValueError(f"Length mismatch: probs={a.shape}, y={yarr.shape}")
    if a.size == 0:
        return {"n": 0, "mean_proba": None, "wr": None, "decile": decile, "top": top}

    n = a.size
    k = max(1, int(round(n * decile)))
    order = np.argsort(a)
    idx = order[-k:] if top else order[:k]
    return {
        "n": int(k),
        "mean_proba": round(float(a[idx].mean()), 4),
        "wr": round(float(yarr[idx].mean()), 4),
        "decile": decile,
        "top": bool(top),
    }


def run_full_diagnostic(
    model_paths: Dict[str, Union[str, Path]],
    holdout_df: pd.DataFrame,
    min_gain_pct: float = 10.0,
    threshold: float = 0.30,
) -> dict:
    """Orchestrate the full diagnostic pipeline.

    Returns a dict with keys:
      - holdout: {n_trades, win_rate, min_gain_pct}
      - models: list of {label, model_path}
      - distributions: {label: probability_distribution(...)}
      - rank_correlations: {f"{a}_vs_{b}": spearman_rho}
      - disagreement: {f"{a}_vs_{b}": disagreement_at_threshold(...)}
      - top_decile: {label: decile_wr(top=True)}
      - bottom_decile: {label: decile_wr(top=False)}
      - error: present if holdout is empty/invalid (other keys may be absent)

    Pairwise sections enumerate every unordered pair (label_a < label_b
    by sort order). Self-pairs are skipped.
    """
    if holdout_df is None or holdout_df.empty:
        return {"error": "Empty holdout DataFrame", "models": list(model_paths.keys())}
    if "gain_pct" not in holdout_df.columns:
        return {"error": "Holdout missing 'gain_pct' column", "models": list(model_paths.keys())}

    y = (holdout_df["gain_pct"] > min_gain_pct).astype(int).values

    scored = score_holdout_with_models(model_paths, holdout_df)
    if not scored:
        return {
            "error": "No models could be scored on this holdout (feature mismatches)",
            "models": list(model_paths.keys()),
            "holdout": {
                "n_trades": int(len(holdout_df)),
                "win_rate": round(float(y.mean()), 4),
                "min_gain_pct": min_gain_pct,
            },
        }

    distributions = {label: probability_distribution(probs) for label, probs in scored.items()}

    labels = sorted(scored.keys())
    rank_correlations: Dict[str, float] = {}
    disagreement: Dict[str, dict] = {}
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            key = f"{la}_vs_{lb}"
            rho = pairwise_rank_correlation(scored[la], scored[lb])
            # Surface NaN as None so the dict round-trips through JSON
            rank_correlations[key] = None if (rho is None or np.isnan(rho)) else rho
            disagreement[key] = disagreement_at_threshold(
                scored[la], scored[lb], y, threshold=threshold
            )

    top_decile = {label: decile_wr(probs, y, top=True) for label, probs in scored.items()}
    bottom_decile = {label: decile_wr(probs, y, top=False) for label, probs in scored.items()}

    return {
        "holdout": {
            "n_trades": int(len(holdout_df)),
            "win_rate": round(float(y.mean()), 4),
            "min_gain_pct": min_gain_pct,
        },
        "models": [{"label": l, "model_path": str(model_paths[l])} for l in labels],
        "skipped": [l for l in model_paths if l not in scored],
        "threshold": threshold,
        "distributions": distributions,
        "rank_correlations": rank_correlations,
        "disagreement": disagreement,
        "top_decile": top_decile,
        "bottom_decile": bottom_decile,
    }
