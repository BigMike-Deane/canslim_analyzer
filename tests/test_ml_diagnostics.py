"""Tests for ml/diagnostics.py — per-trade ML model diagnostics.

Companion to test_oos_eval.py. The diagnostics module is the durable
re-implementation of the VPS-only /app/data/oos_disagreement.py script
used to investigate model-vs-model disagreement on an OOS holdout
(distribution, rank correlation, threshold disagreement, deciles).
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def _make_test_model(feature_cols=None, save_path=None, seed=42):
    """Train a tiny XGBoost on synthetic data, save to disk, return path.

    Mirrors the helper in test_oos_eval.py — kept independent so changes
    here don't leak into the OOS-eval suite.
    """
    from xgboost import XGBClassifier

    if feature_cols is None:
        feature_cols = ["a", "b", "c"]

    rng = np.random.RandomState(seed)
    n = 200
    X = pd.DataFrame(rng.randn(n, len(feature_cols)), columns=feature_cols)
    # `a` is the predictive feature; `b` and `c` are noise
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


def _make_holdout_df(feature_cols, n=50, seed=7):
    """Synthetic holdout DataFrame with feature columns + gain_pct."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, len(feature_cols)), columns=feature_cols)
    # gain_pct correlated with `a` so a good model can rank
    gain = (X[feature_cols[0]] * 8.0 + rng.randn(n) * 5.0).values
    df = X.copy()
    df["gain_pct"] = gain
    df["ticker"] = [f"T{i}" for i in range(n)]
    df["backtest_id"] = (np.arange(n) // 10) + 1000
    return df


# =============== probability_distribution ===============


class TestProbabilityDistribution:
    def test_basic_summary_stats(self):
        from ml.diagnostics import probability_distribution

        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        dist = probability_distribution(probs)
        assert dist["n"] == 10
        assert abs(dist["mean"] - 0.545) < 1e-6
        assert dist["min"] == 0.1
        assert dist["max"] == 0.95
        assert len(dist["bins"]) == len(dist["counts"])
        # Sum of bin counts must equal n
        assert sum(dist["counts"]) == 10

    def test_empty_input_returns_zero_n(self):
        from ml.diagnostics import probability_distribution

        dist = probability_distribution(np.array([]))
        assert dist["n"] == 0
        assert dist["mean"] is None
        assert dist["counts"] == []

    def test_custom_bin_edges(self):
        from ml.diagnostics import probability_distribution

        probs = np.array([0.05, 0.15, 0.25, 0.35, 0.85])
        dist = probability_distribution(probs, bin_edges=[0.0, 0.5, 1.0])
        # 4 in [0, 0.5), 1 in [0.5, 1.0]
        assert dist["counts"] == [4, 1]


# =============== pairwise_rank_correlation ===============


class TestPairwiseRankCorrelation:
    def test_identical_vectors_rho_one(self):
        from ml.diagnostics import pairwise_rank_correlation

        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        rho = pairwise_rank_correlation(a, a.copy())
        assert abs(rho - 1.0) < 1e-9

    def test_reversed_vectors_rho_negative_one(self):
        from ml.diagnostics import pairwise_rank_correlation

        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        b = a[::-1].copy()
        rho = pairwise_rank_correlation(a, b)
        assert abs(rho - (-1.0)) < 1e-9

    def test_length_mismatch_raises(self):
        from ml.diagnostics import pairwise_rank_correlation

        with pytest.raises(ValueError):
            pairwise_rank_correlation(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))

    def test_too_short_returns_nan(self):
        """1-element vectors can't have rank correlation — surface as NaN."""
        from ml.diagnostics import pairwise_rank_correlation

        rho = pairwise_rank_correlation(np.array([0.5]), np.array([0.5]))
        assert np.isnan(rho)


# =============== disagreement_at_threshold ===============


class TestDisagreementAtThreshold:
    def test_partition_sums_to_total(self):
        """only_a + only_b + both + neither must equal n_total."""
        from ml.diagnostics import disagreement_at_threshold

        rng = np.random.RandomState(1)
        a = rng.uniform(size=100)
        b = rng.uniform(size=100)
        y = rng.randint(0, 2, size=100)

        result = disagreement_at_threshold(a, b, y, threshold=0.5)
        partition_n = sum(result[k]["n"] for k in ("only_a", "only_b", "both", "neither"))
        assert partition_n == result["n_total"] == 100

    def test_known_partition(self):
        """Hand-crafted vectors verify the membership logic."""
        from ml.diagnostics import disagreement_at_threshold

        # Threshold 0.30 — index by index:
        # 0: a=0.5 b=0.5 → both
        # 1: a=0.5 b=0.1 → only_a
        # 2: a=0.1 b=0.5 → only_b
        # 3: a=0.1 b=0.1 → neither
        a = np.array([0.5, 0.5, 0.1, 0.1])
        b = np.array([0.5, 0.1, 0.5, 0.1])
        y = np.array([1, 0, 1, 0])
        r = disagreement_at_threshold(a, b, y, threshold=0.30)
        assert r["both"]["n"] == 1
        assert r["only_a"]["n"] == 1
        assert r["only_b"]["n"] == 1
        assert r["neither"]["n"] == 1
        # WR for each set: both has 1 win, only_a has 0 wins, etc.
        assert r["both"]["wr"] == 1.0
        assert r["only_a"]["wr"] == 0.0
        assert r["only_b"]["wr"] == 1.0
        assert r["neither"]["wr"] == 0.0

    def test_empty_set_returns_none_wr(self):
        from ml.diagnostics import disagreement_at_threshold

        # Both models pick everything — only_a, only_b, neither all empty
        a = np.array([0.9, 0.9, 0.9])
        b = np.array([0.9, 0.9, 0.9])
        y = np.array([1, 0, 1])
        r = disagreement_at_threshold(a, b, y, threshold=0.30)
        assert r["both"]["n"] == 3
        assert r["only_a"]["wr"] is None
        assert r["only_b"]["wr"] is None
        assert r["neither"]["wr"] is None


# =============== decile_wr ===============


class TestDecileWr:
    def test_top_decile_with_perfect_signal(self):
        """A vector that perfectly ranks wins should give top-decile WR ~1.0."""
        from ml.diagnostics import decile_wr

        # 100 trades; probability rank perfectly equals win/loss assignment
        probs = np.linspace(0.01, 0.99, 100)
        # Top 10 (highest probs) are all wins
        y = np.zeros(100, dtype=int)
        y[-10:] = 1  # top 10 wins
        r = decile_wr(probs, y, top=True, decile=0.10)
        assert r["n"] == 10
        assert r["wr"] == 1.0
        assert r["top"] is True

    def test_bottom_decile_picks_lowest_proba(self):
        from ml.diagnostics import decile_wr

        probs = np.linspace(0.01, 0.99, 100)
        y = np.zeros(100, dtype=int)
        y[:10] = 1  # bottom 10 (by prob) are all wins → bottom decile WR=1
        r = decile_wr(probs, y, top=False, decile=0.10)
        assert r["n"] == 10
        assert r["wr"] == 1.0
        assert r["top"] is False

    def test_small_holdout_floors_at_one(self):
        """5 trades * 10% = 0.5 → must floor at 1, not error out."""
        from ml.diagnostics import decile_wr

        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = np.array([0, 0, 1, 1, 1])
        r = decile_wr(probs, y, top=True, decile=0.10)
        assert r["n"] == 1
        # Top 1 by prob is the last entry, which is a win
        assert r["wr"] == 1.0


# =============== score_holdout_with_models ===============


class TestScoreHoldoutWithModels:
    def test_two_models_score_independently(self):
        from ml.diagnostics import score_holdout_with_models

        path_a, cols_a = _make_test_model(seed=42)
        path_b, cols_b = _make_test_model(seed=99)
        df = _make_holdout_df(cols_a, n=50)

        scored = score_holdout_with_models({"a": path_a, "b": path_b}, df)
        assert "a" in scored
        assert "b" in scored
        assert len(scored["a"]) == 50
        assert len(scored["b"]) == 50
        # All probabilities must be in [0, 1]
        for probs in scored.values():
            assert (probs >= 0).all() and (probs <= 1).all()

    def test_missing_features_label_skipped(self):
        """Model trained on cols [a,b,c,d] should be skipped when df has only [a,b,c]."""
        from ml.diagnostics import score_holdout_with_models

        path_full, _ = _make_test_model(feature_cols=["a", "b", "c", "d"])
        path_part, cols_part = _make_test_model(feature_cols=["a", "b", "c"])
        df = _make_holdout_df(cols_part, n=30)
        scored = score_holdout_with_models(
            {"full": path_full, "part": path_part}, df,
        )
        assert "full" not in scored  # missing feature `d`
        assert "part" in scored
        assert len(scored["part"]) == 30

    def test_empty_df_returns_empty_dict(self):
        from ml.diagnostics import score_holdout_with_models

        path, _ = _make_test_model()
        scored = score_holdout_with_models({"a": path}, pd.DataFrame())
        assert scored == {}


# =============== run_full_diagnostic ===============


class TestRunFullDiagnostic:
    def test_full_diagnostic_returns_all_keys(self):
        from ml.diagnostics import run_full_diagnostic

        path_a, cols = _make_test_model(seed=42)
        path_b, _ = _make_test_model(feature_cols=cols, seed=99)
        df = _make_holdout_df(cols, n=80)

        result = run_full_diagnostic(
            {"v_a": path_a, "v_b": path_b}, df, min_gain_pct=0.0, threshold=0.30,
        )

        # Top-level keys
        for key in (
            "holdout", "models", "distributions", "rank_correlations",
            "disagreement", "top_decile", "bottom_decile",
        ):
            assert key in result, f"missing key {key}"

        # Holdout summary
        assert result["holdout"]["n_trades"] == 80
        assert 0.0 <= result["holdout"]["win_rate"] <= 1.0

        # Per-model sections
        assert set(result["distributions"].keys()) == {"v_a", "v_b"}
        assert set(result["top_decile"].keys()) == {"v_a", "v_b"}
        assert set(result["bottom_decile"].keys()) == {"v_a", "v_b"}

        # Pairwise sections — exactly one pair for two models
        assert len(result["rank_correlations"]) == 1
        assert len(result["disagreement"]) == 1
        pair_key = next(iter(result["rank_correlations"]))
        assert "_vs_" in pair_key

        # Disagreement: 4 mutually exclusive sets summing to n_total
        d = result["disagreement"][pair_key]
        partition_n = sum(d[k]["n"] for k in ("only_a", "only_b", "both", "neither"))
        assert partition_n == d["n_total"] == 80

    def test_empty_holdout_returns_error(self):
        from ml.diagnostics import run_full_diagnostic

        path_a, _ = _make_test_model()
        result = run_full_diagnostic({"a": path_a}, pd.DataFrame())
        assert "error" in result
        assert "Empty" in result["error"]

    def test_missing_gain_pct_returns_error(self):
        from ml.diagnostics import run_full_diagnostic

        path_a, cols = _make_test_model()
        df = _make_holdout_df(cols, n=20)
        df = df.drop(columns=["gain_pct"])
        result = run_full_diagnostic({"a": path_a}, df)
        assert "error" in result
        assert "gain_pct" in result["error"]

    def test_three_models_yields_three_pairs(self):
        """With 3 models, pairwise sections must enumerate C(3,2)=3 pairs."""
        from ml.diagnostics import run_full_diagnostic

        path_a, cols = _make_test_model(seed=1)
        path_b, _ = _make_test_model(feature_cols=cols, seed=2)
        path_c, _ = _make_test_model(feature_cols=cols, seed=3)
        df = _make_holdout_df(cols, n=60)

        result = run_full_diagnostic(
            {"a": path_a, "b": path_b, "c": path_c}, df, min_gain_pct=0.0,
        )
        assert len(result["rank_correlations"]) == 3
        assert len(result["disagreement"]) == 3
