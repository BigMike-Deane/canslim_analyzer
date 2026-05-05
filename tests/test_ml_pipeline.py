"""
Tests for ML Signal Layer — feature extraction, training, and prediction.

Covers: empty DB, missing signal_factors, NaN handling, buy/sell pairing,
partial sell aggregation, training (classifier + regression), walk-forward
no-leakage, predictions, deduplication, improvement gate, health checks.
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
    _deduplicate_trades,
    _extract_features,
    _is_ml_contaminated,
    _nan_safe,
    _pair_buy_sell_trades,
    extract_training_data,
    get_feature_matrix,
)

# CS-specific signal_factors keys retained for capture but pruned from FEATURE_COLUMNS in v13.
# Keep this list so we can verify the prune in test_feature_count_is_24.
PRUNED_FEATURE_COLUMNS = [
    "coiled_spring",
    "cs_weeks_in_base",
    "cs_beat_streak",
    "cs_days_to_earnings",
    "days_since_spy_pullback",
]

# Price action feature columns (in FEATURE_COLUMNS)
PRICE_ACTION_COLUMNS = [
    "relative_volume", "pct_from_21ma", "pct_from_50ma",
    "atr_pct", "sector_rs_rank",
]
from ml.model import get_ml_prediction, reload_model
from ml.trainer import (
    MIN_ROC_AUC,
    MIN_SPEARMAN,
    MIN_TRAINING_SAMPLES,
    _aggregate_cv_metrics,
    _aggregate_cv_metrics_regression,
    _compute_metrics,
    _compute_regression_metrics,
    _create_xgb_model,
    _create_xgb_regressor,
    load_model,
    save_model,
    train_model,
    train_model_regression,
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
        "estimate_revision_bonus": 1.5,
    }
    t.realized_gain_pct = realized_gain_pct
    t.holding_days = holding_days
    t.reason = reason
    return t


def _make_backtest_run(id, strategy="nostate_optimized", start_date="2022-01-01",
                       end_date="2026-03-01", status="completed", profile_overrides=None,
                       overlay_stats=None, created_at=None):
    """Create a mock BacktestRun-like object.

    overlay_stats and created_at default to None so existing tests fall
    through cleanly. Set them when exercising the post-fix
    overlay_stats.ml_was_active path or the graduation-date heuristic.
    """
    run = MagicMock()
    run.id = id
    run.strategy = strategy
    run.start_date = start_date
    run.end_date = end_date
    run.status = status
    run.profile_overrides = profile_overrides
    run.overlay_stats = overlay_stats
    run.created_at = created_at
    return run


def _make_labeled_df(n=200, win_rate=0.65):
    """Create a synthetic labeled DataFrame for training tests."""
    rng = np.random.RandomState(42)
    rows = []
    for i in range(n):
        is_win = rng.random() < win_rate
        score = rng.uniform(72, 98)
        composite = rng.uniform(30, 80)
        is_cs = rng.random() < 0.15  # ~15% are CS trades
        # Make features correlated with outcome for testable signal
        if is_win:
            composite += 10
            score += 5
        rows.append({
            "total_score": score,
            "composite_score": composite,
            "entry_type": rng.choice([0, 1, 2]),
            "market_regime": rng.choice([0, 1, 2]),
            "estimate_revision_bonus": rng.uniform(-5, 10),
            "coiled_spring": 1 if is_cs else 0,
            "soft_zone": rng.choice([0, 1]),
            "soft_zone_multiplier": rng.uniform(0.5, 1.0),
            "deterministic_boost": rng.choice([0, 5, 8]),
            # CS-specific features (0 for non-CS, actual values for CS)
            "cs_weeks_in_base": rng.choice([15, 20, 26]) if is_cs else 0,
            "cs_beat_streak": rng.randint(3, 10) if is_cs else 0,
            "cs_days_to_earnings": rng.randint(2, 14) if is_cs else 0,
            "cs_bonus": rng.choice([20, 25, 30, 35]) if is_cs else 0,
            "cs_c_score": rng.uniform(12, 15) if is_cs else 0,
            "cs_institutional_pct": rng.uniform(0, 75) if is_cs else 0,
            "cs_quality_rank": rng.uniform(30, 80) if is_cs else 0,
            # Price action features
            "relative_volume": rng.uniform(0.5, 3.0),
            "pct_from_21ma": rng.uniform(-10, 10),
            "pct_from_50ma": rng.uniform(-15, 15),
            "atr_pct": rng.uniform(1, 5),
            "sector_rs_rank": rng.uniform(0, 100),
            "days_since_spy_pullback": rng.randint(1, 60),
            # v11: restored + new features
            "rs_line_bonus": rng.choice([0, 8, -5]),
            "earnings_drift_bonus": rng.choice([0, 0, 5, 10]),
            "volume_dry_up": rng.choice([0, 0, 1]),
            "volume_dry_up_score": rng.uniform(0, 100),
            "c_score": rng.uniform(0, 15),
            "a_score": rng.uniform(0, 15),
            "n_score": rng.uniform(0, 15),
            "s_score": rng.uniform(0, 15),
            "l_score": rng.uniform(0, 15),
            "i_score": rng.uniform(0, 10),
            "spy_pct_above_50ma": rng.uniform(-5, 8),
            "industry_group_rank": rng.uniform(1, 100),
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
        assert f["soft_zone"] == 1
        # Removed features should not be in output
        assert "soft_zone_multiplier" not in f
        assert "coiled_spring" not in f  # pruned in v13
        # v11: deterministic_boost is restored
        assert "deterministic_boost" in f

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
            "estimate_revision_bonus": 0,
        })
        f = _extract_features(trade)
        assert f["soft_zone"] == 0
        # Pruned in v13 — must not appear regardless of signal_factors
        assert "coiled_spring" not in f

    def test_nan_canslim_score(self):
        trade = _make_trade(canslim_score=float("nan"))
        f = _extract_features(trade)
        assert f["total_score"] == 0.0

    def test_unknown_entry_type_defaults_to_standard(self):
        trade = _make_trade(signal_factors={
            "entry_type": "unknown_type",
            "market_regime": "neutral",
            "composite_score": 40.0,
            "estimate_revision_bonus": 0,
        })
        f = _extract_features(trade)
        assert f["entry_type"] == 2  # standard (default)

    def test_feature_count_is_24(self):
        """Verify v13: 24 features (pruned 5 zero-importance features from v12)."""
        assert len(FEATURE_COLUMNS) == 24
        # Pruned in v13 (zero importance in active v9 model)
        for col in PRUNED_FEATURE_COLUMNS:
            assert col not in FEATURE_COLUMNS, f"{col} should have been pruned in v13"
        # Earlier removals still excluded
        assert "soft_zone_multiplier" not in FEATURE_COLUMNS
        assert "cs_c_score" not in FEATURE_COLUMNS
        # v13: retained signal-carrying features
        assert "deterministic_boost" in FEATURE_COLUMNS
        assert "rs_line_bonus" in FEATURE_COLUMNS
        assert "earnings_drift_bonus" in FEATURE_COLUMNS
        # v12: continuous dry-up retained
        assert "volume_dry_up_score" in FEATURE_COLUMNS
        assert "volume_dry_up" in FEATURE_COLUMNS
        # v11: CANSLIM components retained
        assert "c_score" in FEATURE_COLUMNS
        assert "a_score" in FEATURE_COLUMNS
        for col in PRICE_ACTION_COLUMNS:
            assert col in FEATURE_COLUMNS


class TestPrunedFeatures:
    def test_pruned_features_excluded_from_matrix(self):
        """v13: pruned features must not appear in the extracted feature matrix."""
        trade = _make_trade(signal_factors={
            "entry_type": "standard",
            "market_regime": "bullish",
            "composite_score": 55.0,
            "estimate_revision_bonus": 5,
            "coiled_spring": True,
            "cs_weeks_in_base": 20,
            "cs_beat_streak": 5,
            "cs_days_to_earnings": 7,
            "days_since_spy_pullback": 10,
        })
        f = _extract_features(trade)
        for col in PRUNED_FEATURE_COLUMNS:
            assert col not in f, f"{col} was pruned in v13 but still appears in extracted features"


class TestPriceActionFeatures:
    def test_price_action_features_extracted(self):
        """Price action features should be extracted from signal_factors."""
        trade = _make_trade(signal_factors={
            "entry_type": "pre-breakout",
            "market_regime": "bullish",
            "composite_score": 65.0,
            "estimate_revision_bonus": 5,
            "relative_volume": 1.85,
            "pct_from_21ma": -2.5,
            "pct_from_50ma": 3.1,
            "atr_pct": 2.8,
            "sector_rs_rank": 75.0,
            "days_since_spy_pullback": 12,
        })
        f = _extract_features(trade)
        assert f["relative_volume"] == 1.85
        assert f["pct_from_21ma"] == -2.5
        assert f["pct_from_50ma"] == 3.1
        assert f["atr_pct"] == 2.8
        assert f["sector_rs_rank"] == 75.0
        # days_since_spy_pullback pruned in v13
        assert "days_since_spy_pullback" not in f

    def test_missing_price_action_defaults(self):
        """Missing price action features should default safely."""
        trade = _make_trade(signal_factors={
            "entry_type": "standard",
            "market_regime": "neutral",
            "composite_score": 40.0,
            "estimate_revision_bonus": 0,
        })
        f = _extract_features(trade)
        assert f["relative_volume"] == 1.0  # neutral default
        assert f["pct_from_21ma"] == 0.0
        assert f["pct_from_50ma"] == 0.0
        assert f["atr_pct"] == 0.0
        assert f["sector_rs_rank"] == 50.0  # median default


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
        assert len(rows) == 2
        assert rows[0]["gain_pct"] == 10.0
        assert rows[1]["gain_pct"] == 20.0

    def test_different_backtests_isolated(self):
        trades = [
            _make_trade(action="BUY", ticker="AAPL", backtest_id=1),
            _make_trade(action="SELL", ticker="AAPL", backtest_id=2, realized_gain_pct=10.0, holding_days=20),
        ]
        rows = _pair_buy_sell_trades(trades)
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
        df, stats = extract_training_data(db, strategy="nostate_optimized")
        assert df.empty

    def test_with_backtest_ids_filter(self):
        db = MagicMock()
        query = db.query.return_value.filter.return_value
        query.filter.return_value.all.return_value = []
        df, stats = extract_training_data(db, backtest_ids=[1, 2, 3])
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


# ============== Deduplication Tests ==============


class TestIsMlContaminated:
    def test_no_overrides(self):
        run = _make_backtest_run(1, profile_overrides=None)
        assert _is_ml_contaminated(run) is False

    def test_empty_overrides(self):
        run = _make_backtest_run(1, profile_overrides={})
        assert _is_ml_contaminated(run) is False

    def test_log_only_not_contaminated(self):
        run = _make_backtest_run(1, profile_overrides={
            "ml_signal": {"enabled": True, "log_only": True}
        })
        assert _is_ml_contaminated(run) is False

    def test_ml_active_is_contaminated(self):
        run = _make_backtest_run(1, profile_overrides={
            "ml_signal": {"enabled": True, "log_only": False}
        })
        assert _is_ml_contaminated(run) is True

    def test_ml_disabled_not_contaminated(self):
        run = _make_backtest_run(1, profile_overrides={
            "ml_signal": {"enabled": False, "log_only": False}
        })
        assert _is_ml_contaminated(run) is False

    def test_non_ml_overrides_not_contaminated(self):
        run = _make_backtest_run(1, profile_overrides={
            "stop_loss_pct": 5.0
        })
        assert _is_ml_contaminated(run) is False

    # --- Post-fix: overlay_stats.ml_was_active is authoritative ---

    def test_overlay_stats_ml_was_active_true(self):
        """overlay_stats.ml_was_active=True → contaminated regardless of overrides."""
        run = _make_backtest_run(1, profile_overrides=None,
                                 overlay_stats={"ml_was_active": True})
        assert _is_ml_contaminated(run) is True

    def test_overlay_stats_ml_was_active_false(self):
        """overlay_stats.ml_was_active=False → clean even if heuristic would flag."""
        from datetime import datetime
        run = _make_backtest_run(1, profile_overrides=None,
                                 overlay_stats={"ml_was_active": False},
                                 created_at=datetime(2026, 5, 4))
        assert _is_ml_contaminated(run) is False

    def test_overlay_stats_as_json_string(self):
        """overlay_stats stored as TEXT in production Postgres — must be decoded."""
        run = _make_backtest_run(1, profile_overrides=None,
                                 overlay_stats='{"ml_was_active": true}')
        assert _is_ml_contaminated(run) is True

    # --- Heuristic for pre-fix runs without overlay_stats or overrides ---

    def test_post_graduation_no_signals_assumed_contaminated(self):
        """Regression for the silent-contamination bug: nostate_optimized
        runs after Apr 29 with neither overlay_stats nor profile_overrides
        previously slipped into training as 'clean' even though the YAML
        default sets ml_signal.log_only=false."""
        from datetime import datetime
        run = _make_backtest_run(1, strategy="nostate_optimized",
                                 profile_overrides=None, overlay_stats=None,
                                 created_at=datetime(2026, 5, 4))
        assert _is_ml_contaminated(run) is True

    def test_pre_graduation_no_signals_clean(self):
        """Pre-graduation runs lack the contamination since YAML default
        was log_only=true at that point."""
        from datetime import datetime
        run = _make_backtest_run(1, strategy="nostate_optimized",
                                 profile_overrides=None, overlay_stats=None,
                                 created_at=datetime(2026, 4, 15))
        assert _is_ml_contaminated(run) is False

    def test_cs_bear_post_graduation_not_auto_flagged(self):
        """cs_bear's YAML default stays log_only=true even after Apr 29,
        so it should NOT be in the auto-flag table."""
        from datetime import datetime
        run = _make_backtest_run(1, strategy="nostate_cs_bear",
                                 profile_overrides=None, overlay_stats=None,
                                 created_at=datetime(2026, 5, 4))
        assert _is_ml_contaminated(run) is False


class TestDeduplicateTrades:
    def test_no_duplicates(self):
        rows = [
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 1, "sell_reason": "stop_loss", "gain_pct": -5.0},
            {"ticker": "GOOGL", "date": "2024-02-10", "backtest_id": 1, "sell_reason": "take_profit", "gain_pct": 15.0},
        ]
        result = _deduplicate_trades(rows)
        assert len(result) == 2

    def test_same_ticker_date_keeps_newest_backtest(self):
        rows = [
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 100, "sell_reason": "stop_loss", "gain_pct": -5.0},
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 200, "sell_reason": "stop_loss", "gain_pct": -3.0},
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 150, "sell_reason": "stop_loss", "gain_pct": -4.0},
        ]
        result = _deduplicate_trades(rows)
        assert len(result) == 1
        assert result[0]["backtest_id"] == 200
        assert result[0]["gain_pct"] == -3.0

    def test_different_dates_not_deduped(self):
        rows = [
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 1, "sell_reason": "stop_loss", "gain_pct": -5.0},
            {"ticker": "AAPL", "date": "2024-03-20", "backtest_id": 1, "sell_reason": "take_profit", "gain_pct": 10.0},
        ]
        result = _deduplicate_trades(rows)
        assert len(result) == 2

    def test_partial_and_full_sell_preserved(self):
        """PARTIAL_SELL and SELL for same ticker/date kept as separate entries."""
        rows = [
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 1, "sell_reason": "PARTIAL PROFIT", "gain_pct": 10.0},
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": 1, "sell_reason": "TRAILING STOP", "gain_pct": 20.0},
        ]
        result = _deduplicate_trades(rows)
        assert len(result) == 2

    def test_large_dedup_set(self):
        """10 overlapping backtests produce 10 copies of same trade — dedup to 1."""
        rows = [
            {"ticker": "AAPL", "date": "2024-01-15", "backtest_id": i, "sell_reason": "stop_loss", "gain_pct": -5.0}
            for i in range(10)
        ]
        result = _deduplicate_trades(rows)
        assert len(result) == 1
        assert result[0]["backtest_id"] == 9  # Newest


class TestBacktestDeduplication:
    def test_latest_per_date_range_selected(self):
        """When multiple backtests cover the same date range, only the latest is used."""
        from ml.feature_extractor import _select_deduplicated_backtests

        runs = [
            _make_backtest_run(100, start_date="2022-01-01", end_date="2026-03-01"),
            _make_backtest_run(200, start_date="2022-01-01", end_date="2026-03-01"),
            _make_backtest_run(150, start_date="2022-01-01", end_date="2026-03-01"),
            _make_backtest_run(50, start_date="2023-01-01", end_date="2024-01-01"),
            _make_backtest_run(75, start_date="2023-01-01", end_date="2024-01-01"),
        ]

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = runs

        stats = {"backtests_before": 0, "excluded_ml_contaminated": 0}
        result = _select_deduplicated_backtests(db, "nostate_optimized", stats)

        ids = {r.id for r in result}
        assert ids == {200, 75}  # Latest per date range
        assert stats["backtests_before"] == 5

    def test_ml_contaminated_excluded(self):
        from ml.feature_extractor import _select_deduplicated_backtests

        runs = [
            _make_backtest_run(100, start_date="2022-01-01", end_date="2026-03-01"),
            _make_backtest_run(200, start_date="2022-01-01", end_date="2026-03-01",
                              profile_overrides={"ml_signal": {"enabled": True, "log_only": False}}),
        ]

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = runs

        stats = {"backtests_before": 0, "excluded_ml_contaminated": 0}
        result = _select_deduplicated_backtests(db, "nostate_optimized", stats)

        ids = {r.id for r in result}
        assert ids == {100}  # 200 is ML-contaminated
        assert stats["excluded_ml_contaminated"] == 1


# ============== Classifier Trainer Tests ==============


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
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)  # Set gate to 0 for test
        assert "model" in result or result.get("cv_results")
        assert "cv_results" in result

    def test_gate_failure(self):
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
            total = sum(result["feature_importance"].values())
            assert 0.9 < total < 1.1

    def test_min_gain_pct_default_preserves_behavior(self):
        """min_gain_pct=0.0 (default) reproduces historical win definition."""
        df = _make_labeled_df(n=300, win_rate=0.65)
        result_default = train_model(df, min_roc_auc=0.0)
        result_zero = train_model(df, min_roc_auc=0.0, min_gain_pct=0.0)
        # Same win_rate either way (label unchanged)
        assert result_default.get("win_rate") == result_zero.get("win_rate")
        assert result_zero.get("min_gain_pct") == 0.0

    def test_min_gain_pct_tightens_label(self):
        """Raising min_gain_pct shrinks the positive class."""
        df = _make_labeled_df(n=300, win_rate=0.65)
        result_loose = train_model(df, min_roc_auc=0.0, min_gain_pct=0.0)
        result_strict = train_model(df, min_roc_auc=0.0, min_gain_pct=10.0)
        # Strict label must produce no more wins than loose label
        assert result_strict.get("win_rate", 1.0) <= result_loose.get("win_rate", 0.0) + 1e-9
        assert result_strict.get("min_gain_pct") == 10.0

    def test_result_feature_columns_in_training_order(self):
        """Regression: train_model must surface feature_columns in training
        order on the result dict. The buggy call site read
        result['feature_importance'].keys() — that's importance-sorted, not
        training order, and broke v17 in two prediction code paths."""
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)
        if result.get("passed_gate"):
            assert "feature_columns" in result
            # Must equal the model's own training-order record
            assert result["feature_columns"] == list(result["model"].feature_names_in_)
            # And must equal FEATURE_COLUMNS when no exclusion is set
            assert result["feature_columns"] == list(FEATURE_COLUMNS)

    def test_result_feature_columns_with_exclusion(self):
        """When excluded_features drops columns, result['feature_columns'] must
        reflect the surviving subset in training order."""
        excluded = ["total_score", "composite_score"]
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0, excluded_features=excluded)
        if result.get("passed_gate"):
            assert result["feature_columns"] == list(result["model"].feature_names_in_)
            for name in excluded:
                assert name not in result["feature_columns"]
            assert len(result["feature_columns"]) == len(FEATURE_COLUMNS) - len(excluded)

    def test_per_regime_auc_computed_in_folds(self):
        """Walk-forward CV folds should include per_regime_auc dict when regime feature is present."""
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)
        cv_results = result.get("cv_results", [])
        assert len(cv_results) > 0
        for fold in cv_results:
            assert "per_regime_auc" in fold
            pra = fold["per_regime_auc"]
            # Keys are regime names; values are float AUC or None when too few samples
            assert set(pra.keys()) <= {"bearish", "neutral", "bullish"}
            for v in pra.values():
                assert v is None or 0.0 <= v <= 1.0


class TestExcludedFeatures:
    """excluded_features lets us train ablation/leakage-audit models. The
    saved feature list MUST shrink to match — predict-time will pass features
    in that exact list, and an off-by-one would either crash or silently
    feed the model garbage."""

    def test_drop_columns_before_training(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(
            df, min_roc_auc=0.0,
            excluded_features=["total_score", "composite_score"],
        )
        if result.get("passed_gate"):
            assert result["feature_count"] == len(FEATURE_COLUMNS) - 2
            importance = result["feature_importance"]
            assert "total_score" not in importance
            assert "composite_score" not in importance
            # Excluded list is round-tripped on the result
            assert set(result["excluded_features"]) == {"total_score", "composite_score"}

    def test_unknown_excluded_names_silently_ignored(self):
        """Forward-compat: caller may pass column names that don't exist yet."""
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(
            df, min_roc_auc=0.0,
            excluded_features=["does_not_exist", "total_score"],
        )
        if result.get("passed_gate"):
            # Only total_score got dropped (the other was ignored)
            assert result["feature_count"] == len(FEATURE_COLUMNS) - 1


class TestCalibration:
    """Calibration wraps the XGBoost classifier in CalibratedClassifierCV so
    predict_proba returns calibrated probabilities. Without it, "0.30" is an
    arbitrary threshold; with isotonic calibration, "0.30" means "the model
    estimates ~30% empirical win rate."""

    def test_calibrate_wraps_model(self):
        from sklearn.calibration import CalibratedClassifierCV
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0, calibrate=True)
        if result.get("passed_gate"):
            assert isinstance(result["model"], CalibratedClassifierCV)
            assert result["calibrated"] is True

    def test_calibrated_model_predict_proba_works(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0, calibrate=True)
        if not result.get("passed_gate"):
            pytest.skip("training skipped on this seed")
        model = result["model"]
        # Build a single-row feature matrix matching what got trained
        from ml.feature_extractor import get_feature_matrix
        X, _, _, _ = get_feature_matrix(df)
        proba = model.predict_proba(X.iloc[[0]])
        assert proba.shape == (1, 2)
        # Probabilities sum to 1
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_uncalibrated_default(self):
        """Default behavior is uncalibrated XGBoost — preserves prior outputs."""
        from xgboost import XGBClassifier
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model(df, min_roc_auc=0.0)
        if result.get("passed_gate"):
            assert isinstance(result["model"], XGBClassifier)
            assert result.get("calibrated") in (False, None)


# ============== Regression Trainer Tests ==============


class TestCreateXgbRegressor:
    def test_regressor_params(self):
        model = _create_xgb_regressor()
        assert model.max_depth == 3
        assert model.n_estimators == 100
        assert model.learning_rate == 0.05


class TestComputeRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([5.0, -3.0, 10.0, -7.0])
        y_pred = np.array([5.0, -3.0, 10.0, -7.0])
        m = _compute_regression_metrics(y_true, y_pred)
        assert m["r2"] == 1.0
        assert m["mae"] == 0.0
        assert m["spearman"] == 1.0
        assert m["direction_accuracy"] == 1.0

    def test_inverse_predictions(self):
        y_true = np.array([5.0, -3.0, 10.0, -7.0])
        y_pred = np.array([-5.0, 3.0, -10.0, 7.0])
        m = _compute_regression_metrics(y_true, y_pred)
        assert m["spearman"] == -1.0
        assert m["direction_accuracy"] == 0.0

    def test_constant_predictions(self):
        y_true = np.array([5.0, -3.0, 10.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        m = _compute_regression_metrics(y_true, y_pred)
        # Constant predictions → Spearman = 0 (no variance in pred)
        assert m["spearman"] == 0.0


class TestTrainModelRegression:
    def test_insufficient_data(self):
        df = _make_labeled_df(n=20)
        result = train_model_regression(df)
        assert result["passed_gate"] is False
        assert result["model_type"] == "regression"
        assert "Insufficient" in result["error"]

    def test_training_with_synthetic_data(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model_regression(df, min_spearman=0.0)
        assert "cv_results" in result
        assert result["model_type"] == "regression"

    def test_gate_failure_regression(self):
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
        result = train_model_regression(df, min_spearman=0.90)
        assert result["passed_gate"] is False

    def test_feature_importance_regression(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model_regression(df, min_spearman=0.0)
        if result.get("passed_gate"):
            assert "feature_importance" in result
            assert len(result["feature_importance"]) == len(FEATURE_COLUMNS)

    def test_gain_stats_populated(self):
        df = _make_labeled_df(n=300, win_rate=0.65)
        result = train_model_regression(df, min_spearman=0.0)
        if result.get("passed_gate"):
            assert "gain_stats" in result
            gs = result["gain_stats"]
            assert "mean" in gs
            assert "median" in gs
            assert "clip_range" in gs


class TestAggregateRegressionMetrics:
    def test_aggregation(self):
        results = [
            {"r2": 0.3, "mae": 5.0, "rmse": 7.0, "spearman": 0.25, "direction_accuracy": 0.65},
            {"r2": 0.4, "mae": 4.0, "rmse": 6.0, "spearman": 0.35, "direction_accuracy": 0.70},
        ]
        m = _aggregate_cv_metrics_regression(results)
        assert m["r2"] == 0.35
        assert m["spearman"] == 0.3
        assert m["direction_accuracy"] == 0.675

    def test_empty_results(self):
        assert _aggregate_cv_metrics_regression([]) == {}


# ============== Save/Load Tests ==============


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

    def test_save_load_regressor(self, tmp_path):
        model = _create_xgb_regressor()
        df = _make_labeled_df(n=100)
        X, _, y_gain, _ = get_feature_matrix(df)
        model.fit(X, y_gain)

        path = tmp_path / "test_reg_model.joblib"
        save_model(model, {"model_type": "regression"}, path=path)

        loaded = load_model(path=path)
        assert loaded is not None
        assert loaded["metadata"]["model_type"] == "regression"

    def test_save_model_preserves_training_order_no_explicit_columns(self, tmp_path):
        """Regression: with no feature_columns arg, save_model must derive from
        model.feature_names_in_ (training order), NOT FEATURE_COLUMNS or any
        importance-sorted source. v17 hit this in two paths (commits 1bbf4c9, 61d55fb).
        """
        df = _make_labeled_df(n=100)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        path = tmp_path / "no_explicit_cols.joblib"
        save_model(model, {}, path=path)
        loaded = load_model(path=path)

        assert loaded["feature_columns"] == list(model.feature_names_in_)

    def test_save_model_preserves_training_order_explicit_columns(self, tmp_path):
        """When the caller passes feature_columns explicitly (the production path
        from backend/routes/ml.py), the saved list must equal what was passed in
        and equal model.feature_names_in_."""
        df = _make_labeled_df(n=100)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        training_order = list(X.columns)
        path = tmp_path / "explicit_cols.joblib"
        save_model(model, {}, path=path, feature_columns=training_order)
        loaded = load_model(path=path)

        assert loaded["feature_columns"] == training_order
        assert loaded["feature_columns"] == list(model.feature_names_in_)

    def test_save_model_rejects_importance_sorted_via_default(self, tmp_path):
        """End-to-end: a real classifier training run where one feature dominates
        importance. Saved feature_columns must NOT be importance-sorted (the v17 bug).
        """
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        # Build the importance-sorted list the way the buggy call site did.
        importances = dict(zip(X.columns, model.feature_importances_.tolist()))
        importance_sorted = [k for k, _ in sorted(importances.items(), key=lambda kv: -kv[1])]

        path = tmp_path / "default_save.joblib"
        save_model(model, {}, path=path)
        loaded = load_model(path=path)

        # Only meaningful if the two orders actually differ — guard against the
        # rare case where xgboost happens to produce features sorted in training order.
        if importance_sorted != list(X.columns):
            assert loaded["feature_columns"] != importance_sorted, (
                "save_model must not save feature_columns in importance-sorted order"
            )
        assert loaded["feature_columns"] == list(model.feature_names_in_)


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

    def test_classifier_prediction_in_range(self, tmp_path):
        """Train a real classifier and verify prediction output range."""
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        path = tmp_path / "test_model.joblib"
        save_model(model, {"model_type": "classifier"}, path=path)
        loaded = load_model(path=path)

        reload_model()
        with patch("ml.model._get_model", return_value=(loaded["model"], loaded)):
            result = get_ml_prediction(
                total_score=85, composite_score=50,
                entry_type=0, market_regime=2,
                estimate_revision_bonus=1.5,
                coiled_spring=1, soft_zone=0,
                soft_zone_multiplier=1.0,
                deterministic_boost=0,
            )
            assert result is not None
            assert 0.0 <= result <= 1.0

    def test_regressor_prediction_in_range(self, tmp_path):
        """Train a real regressor and verify sigmoid-mapped output range."""
        df = _make_labeled_df(n=200)
        X, _, y_gain, _ = get_feature_matrix(df)
        model = _create_xgb_regressor()
        model.fit(X, y_gain)

        path = tmp_path / "test_model.joblib"
        save_model(model, {"model_type": "regression"}, path=path)
        loaded = load_model(path=path)

        reload_model()
        with patch("ml.model._get_model", return_value=(loaded["model"], loaded)):
            result = get_ml_prediction(
                total_score=85, composite_score=50,
                entry_type=0, market_regime=2,
                estimate_revision_bonus=1.5,
                coiled_spring=1, soft_zone=0,
                soft_zone_multiplier=1.0,
                deterministic_boost=0,
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
                estimate_revision_bonus=0,
                coiled_spring=True,  # Boolean, not int
                soft_zone=False,     # Boolean, not int
                soft_zone_multiplier=1.0,
                deterministic_boost=0,
            )
            assert result is not None
            assert 0.0 <= result <= 1.0


class TestGetMlPredictionWithModel:
    """get_ml_prediction_with_model bypasses the global cache so the eval
    backtest gate can score a candidate model file without disturbing
    production. These tests verify the override path is functional and
    isolated from the global cache."""

    def test_returns_none_for_empty_payload(self):
        from ml.model import get_ml_prediction_with_model
        assert get_ml_prediction_with_model({}, total_score=85) is None
        assert get_ml_prediction_with_model(None, total_score=85) is None

    def test_predicts_from_explicit_payload(self, tmp_path):
        """Pass a freshly-loaded payload; result must be in [0,1]."""
        from ml.model import get_ml_prediction_with_model
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        path = tmp_path / "candidate.joblib"
        save_model(model, {"model_type": "classifier"}, path=path)
        payload = load_model(path=path)

        result = get_ml_prediction_with_model(
            payload,
            total_score=85, composite_score=50, entry_type=0, market_regime=2,
            estimate_revision_bonus=1.5, coiled_spring=1, soft_zone=0,
            soft_zone_multiplier=1.0, deterministic_boost=0,
        )
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_global_cache_not_polluted_by_per_payload_predict(self, tmp_path):
        """Critical for production: the eval gate must not push into the
        global prediction cache or set the cached active-model state."""
        from ml.model import (
            get_ml_prediction_with_model,
            get_prediction_cache_stats,
            clear_prediction_cache,
            is_model_loaded,
        )
        df = _make_labeled_df(n=200)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)
        path = tmp_path / "candidate2.joblib"
        save_model(model, {"model_type": "classifier"}, path=path)
        payload = load_model(path=path)

        reload_model()
        clear_prediction_cache()
        stats_before = get_prediction_cache_stats()
        loaded_before = is_model_loaded()

        # Two predictions through the override path
        for _ in range(2):
            get_ml_prediction_with_model(
                payload,
                total_score=85, composite_score=50, entry_type=0, market_regime=2,
                estimate_revision_bonus=1.5, coiled_spring=1, soft_zone=0,
                soft_zone_multiplier=1.0, deterministic_boost=0,
            )

        stats_after = get_prediction_cache_stats()
        # Cache size and hit/miss counts must be unchanged
        assert stats_after == stats_before
        # No global model should be loaded as a side effect
        assert is_model_loaded() == loaded_before

    def test_returns_none_when_model_predict_raises(self, tmp_path):
        """Robustness: a malformed payload that raises during predict must
        return None instead of bubbling up — the eval gate must never crash
        the training pipeline."""
        from ml.model import get_ml_prediction_with_model

        class BrokenModel:
            feature_names_in_ = ["a", "b"]
            def predict_proba(self, X):
                raise RuntimeError("simulated predict failure")

        payload = {"model": BrokenModel(), "feature_columns": ["a", "b"], "metadata": {"model_type": "classifier"}}
        assert get_ml_prediction_with_model(payload, a=1.0, b=2.0) is None


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
            train_indices = set(range(train_start, train_end))
            test_indices = set(range(train_end, test_end))
            assert train_indices.isdisjoint(test_indices)


class TestNaNSafety:
    def test_nan_prediction_returns_none(self):
        """If model.predict() returns NaN, get_ml_prediction() should return None."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([float("nan")])
        mock_metadata = {
            "feature_columns": ["total_score", "composite_score"],
            "metadata": {"model_type": "regression"},
        }
        reload_model()
        with patch("ml.model._get_model", return_value=(mock_model, mock_metadata)):
            result = get_ml_prediction(total_score=80, composite_score=90)
        assert result is None


# ============== Improvement Gate Tests ==============


class TestImprovementGate:
    """Test that _get_active_model_metric works correctly for gating."""

    def test_no_active_model_returns_none(self):
        from backend.routes.ml import _get_active_model_metric
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        metric, version = _get_active_model_metric(db, "nostate_optimized", "classifier")
        assert metric is None
        assert version is None

    def test_classifier_returns_roc_auc(self):
        from backend.routes.ml import _get_active_model_metric
        db = MagicMock()
        active = MagicMock()
        active.roc_auc = 0.65
        active.version = 1
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = active
        metric, version = _get_active_model_metric(db, "nostate_optimized", "classifier")
        assert metric == 0.65
        assert version == 1

    def test_regression_returns_spearman(self):
        from backend.routes.ml import _get_active_model_metric
        db = MagicMock()
        active = MagicMock()
        active.spearman = 0.22
        active.version = 2
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = active
        metric, version = _get_active_model_metric(db, "nostate_optimized", "regression")
        assert metric == 0.22
        assert version == 2


class TestEvalGateDecision:
    """The model-graduation eval-backtest gate replaces (well, supplements)
    the AUC-only gate. Pure decision function: deterministic, easy to lock
    down with regression tests so threshold tweaks are explicit."""

    def test_clear_winner_passes(self):
        from backend.routes.ml import _eval_gate_decision
        # Candidate beats incumbent on both axes
        passes, reason = _eval_gate_decision(200.0, 2.10, 192.0, 2.04)
        assert passes is True
        assert "+8.00pp" in reason
        assert "+0.060" in reason

    def test_below_return_threshold_blocked(self):
        from backend.routes.ml import _eval_gate_decision, MIN_RETURN_DELTA_PP
        # Beats incumbent by 4pp — below the 5pp default
        passes, reason = _eval_gate_decision(196.0, 2.10, 192.0, 2.04)
        assert passes is False
        assert "return Δ=+4.00pp" in reason
        assert f"+{MIN_RETURN_DELTA_PP}" in reason

    def test_sharpe_regression_blocked_even_if_return_passes(self):
        from backend.routes.ml import _eval_gate_decision, MAX_SHARPE_REGRESSION
        # Big return win but Sharpe drops too far
        passes, reason = _eval_gate_decision(220.0, 1.80, 192.0, 2.04)
        assert passes is False
        assert "sharpe Δ=-0.240" in reason
        assert f"{-MAX_SHARPE_REGRESSION}" in reason

    def test_small_sharpe_dip_within_tolerance_passes(self):
        """Sharpe 0.05 below incumbent is within MAX_SHARPE_REGRESSION=0.10 — pass."""
        from backend.routes.ml import _eval_gate_decision
        passes, _ = _eval_gate_decision(200.0, 1.99, 192.0, 2.04)
        assert passes is True

    def test_first_model_no_incumbent_auto_passes(self):
        """When no incumbent has eval baseline, candidate auto-passes —
        otherwise we'd never bootstrap the system."""
        from backend.routes.ml import _eval_gate_decision
        passes, reason = _eval_gate_decision(150.0, 1.5, None, None)
        assert passes is True
        assert "no incumbent" in reason.lower()

    def test_partial_incumbent_baseline_treated_as_missing(self):
        """If incumbent has return but missing sharpe (or vice versa),
        we don't pretend to compare — auto-pass and store new baseline."""
        from backend.routes.ml import _eval_gate_decision
        passes, _ = _eval_gate_decision(150.0, 1.5, 192.0, None)
        assert passes is True
        passes, _ = _eval_gate_decision(150.0, 1.5, None, 2.04)
        assert passes is True

    def test_missing_candidate_metrics_blocks(self):
        """If the eval backtest didn't produce return or sharpe, we can't
        decide — default to don't-activate. This is the safe direction."""
        from backend.routes.ml import _eval_gate_decision
        passes, reason = _eval_gate_decision(None, 2.10, 192.0, 2.04)
        assert passes is False
        assert "no metrics" in reason.lower()
        passes, reason = _eval_gate_decision(200.0, None, 192.0, 2.04)
        assert passes is False

    def test_exact_threshold_passes(self):
        """Boundary: return delta exactly at the threshold counts as a pass.
        Locks in the >= semantics so a future tweak to > would break this test."""
        from backend.routes.ml import _eval_gate_decision, MIN_RETURN_DELTA_PP
        passes, _ = _eval_gate_decision(192.0 + MIN_RETURN_DELTA_PP, 2.04, 192.0, 2.04)
        assert passes is True

    def test_exact_sharpe_floor_passes(self):
        """Sharpe regression equal to MAX_SHARPE_REGRESSION counts as passing."""
        from backend.routes.ml import _eval_gate_decision, MAX_SHARPE_REGRESSION
        passes, _ = _eval_gate_decision(200.0, 2.04 - MAX_SHARPE_REGRESSION, 192.0, 2.04)
        assert passes is True


class TestSaveModelPathing:
    """Regression: in the original eval-gate (1e11fc6) candidates were saved
    DIRECTLY to ACTIVE_MODEL_PATH, overwriting the incumbent's joblib BEFORE
    the gate ran. The dual-run (4c965a1) then compared candidate-vs-candidate
    instead of candidate-vs-incumbent. Lock down the new pathing semantics
    so this can't recur."""

    def test_save_model_with_explicit_path_writes_there_not_active(self, tmp_path):
        """save_model(path=...) must write to the explicit path, leaving
        ACTIVE_MODEL_PATH untouched. The fixed _run_training relies on this
        invariant to keep the incumbent's file intact during gating."""
        from ml.trainer import save_model, load_model, ACTIVE_MODEL_PATH

        # Capture current active.joblib state (whatever bytes happen to be there).
        # If it doesn't exist that's also fine — we just confirm save_model
        # doesn't *create* it when given an explicit path.
        active_existed_before = ACTIVE_MODEL_PATH.exists()
        active_bytes_before = ACTIVE_MODEL_PATH.read_bytes() if active_existed_before else None

        df = _make_labeled_df(n=100)
        X, y, _, _ = get_feature_matrix(df)
        model = _create_xgb_model()
        model.fit(X, y)

        candidate_path = tmp_path / "ml_model_v999.joblib"
        save_model(model, {"version": 999}, path=candidate_path)

        # Candidate file landed at the explicit path
        assert candidate_path.exists()
        loaded = load_model(path=candidate_path)
        assert loaded["metadata"]["version"] == 999

        # ACTIVE_MODEL_PATH was not modified by this save
        if active_existed_before:
            assert ACTIVE_MODEL_PATH.read_bytes() == active_bytes_before, (
                "save_model with explicit path must not modify ACTIVE_MODEL_PATH — "
                "doing so would overwrite the incumbent's file mid-gating."
            )
        else:
            # Either still doesn't exist, or if it does, must not be the candidate's bytes
            if ACTIVE_MODEL_PATH.exists():
                assert ACTIVE_MODEL_PATH.read_bytes() != candidate_path.read_bytes()
