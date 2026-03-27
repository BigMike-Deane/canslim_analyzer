"""
Extract training features from backtest trades.

Pairs BUY trades with their subsequent SELL(s) to build labeled training data.
Features come from signal_factors JSON stored on each BacktestTrade.

Includes deduplication to prevent overlapping backtests from inflating
training data, and excludes ML-contaminated backtests (where ML actively
influenced trade decisions, creating circular training data).
"""

import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Feature columns — v10: 15 high-signal features (simplified from v9's 22)
#
# Removed 7 low-signal/redundant features:
#   - soft_zone_multiplier: derivative of soft_zone binary
#   - deterministic_boost: correlated with total_score
#   - cs_c_score: correlated with total_score (C component already baked in)
#   - cs_institutional_pct: weak predictor, correlated with I score
#   - cs_quality_rank: derivative of other CS features
#   - cs_bonus: highly correlated with cs_weeks_in_base + cs_beat_streak
#   - days_since_spy_pullback: noisy signal, hard to capture in 200 samples
#
# Keeping 15 features that have structural/theoretical edge:
FEATURE_COLUMNS = [
    # Core quality (3) — the "what" of the stock
    "total_score",
    "composite_score",
    "estimate_revision_bonus",
    # Entry context (3) — the "when" and "how" of the entry
    "entry_type",           # ordinal: breakout=0, pre-breakout=1, standard=2
    "market_regime",        # ordinal: bearish=0, neutral=1, bullish=2
    "coiled_spring",        # binary 0/1 (earnings catalyst)
    # Entry quality flags (1)
    "soft_zone",            # binary 0/1 (below threshold = lower conviction)
    # Coiled Spring detail (3) — the 3 most predictive CS features
    "cs_weeks_in_base",     # consolidation length (structural)
    "cs_beat_streak",       # earnings consistency (fundamental)
    "cs_days_to_earnings",  # timing (catalyst proximity)
    # Price action (5) — entry timing quality
    "relative_volume",       # volume / 50-day avg at entry (accumulation)
    "pct_from_21ma",         # % distance from 21-day MA (support proximity)
    "pct_from_50ma",         # % distance from 50-day MA (trend health)
    "atr_pct",               # volatility context (tighter = more coiled)
    "sector_rs_rank",        # sector momentum (group leadership)
]

ENTRY_TYPE_MAP = {"breakout": 0, "pre-breakout": 1, "standard": 2}
REGIME_MAP = {"bearish": 0, "neutral": 1, "bullish": 2}


def _nan_safe(val, default=0.0):
    """Convert None/NaN to a safe default."""
    if val is None:
        return default
    try:
        if val != val:  # NaN != NaN per IEEE 754
            return default
    except (TypeError, ValueError):
        return default
    return val


def extract_training_data(
    db: Session,
    strategy: str = "nostate_optimized",
    backtest_ids: Optional[list] = None,
) -> tuple:
    """
    Extract labeled training data from completed backtest trades.

    Returns (DataFrame, dedup_stats dict).
    DataFrame has FEATURE_COLUMNS + label columns:
        win (binary), gain_pct (float),
        ticker, date, backtest_id, holding_days, sell_reason (metadata)

    dedup_stats contains:
        backtests_before, backtests_after, excluded_ml_contaminated,
        trades_before_dedup, trades_after_dedup
    """
    from backend.database import BacktestRun, BacktestTrade

    dedup_stats = {
        "backtests_before": 0,
        "backtests_after": 0,
        "excluded_ml_contaminated": 0,
        "trades_before_dedup": 0,
        "trades_after_dedup": 0,
    }

    if backtest_ids:
        # Explicit backtest IDs — use as-is, no dedup needed
        runs = db.query(BacktestRun).filter(
            BacktestRun.status == "completed",
            BacktestRun.strategy == strategy,
            BacktestRun.id.in_(backtest_ids),
        ).all()
    else:
        # Auto-select: latest backtest per unique (start_date, end_date),
        # excluding ML-contaminated runs
        runs = _select_deduplicated_backtests(db, strategy, dedup_stats)

    if not runs:
        logger.warning(f"No completed backtests found for strategy '{strategy}'")
        return pd.DataFrame(), dedup_stats

    run_ids = [r.id for r in runs]
    dedup_stats["backtests_after"] = len(run_ids)
    logger.info(
        f"Extracting from {len(run_ids)} deduplicated backtest runs: {run_ids} "
        f"(from {dedup_stats['backtests_before']} total, "
        f"{dedup_stats['excluded_ml_contaminated']} ML-contaminated excluded)"
    )

    # Pull all trades for these runs
    trades = (
        db.query(BacktestTrade)
        .filter(BacktestTrade.backtest_id.in_(run_ids))
        .order_by(BacktestTrade.backtest_id, BacktestTrade.date, BacktestTrade.id)
        .all()
    )

    if not trades:
        logger.warning("No trades found in selected backtests")
        return pd.DataFrame(), dedup_stats

    # Pair BUYs with their subsequent SELLs
    rows = _pair_buy_sell_trades(trades)

    if not rows:
        logger.warning("No buy-sell pairs found")
        return pd.DataFrame(), dedup_stats

    dedup_stats["trades_before_dedup"] = len(rows)

    # Trade-level dedup: if overlapping date ranges produced duplicate
    # (ticker, buy_date) pairs, keep only the one from the newest backtest
    rows = _deduplicate_trades(rows)
    dedup_stats["trades_after_dedup"] = len(rows)

    df = pd.DataFrame(rows)

    # Sort by date for correct walk-forward CV chronological ordering
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        f"Extracted {len(df)} labeled trades "
        f"(dedup removed {dedup_stats['trades_before_dedup'] - len(df)} duplicates)"
    )
    logger.info(f"Win rate: {df['win'].mean():.1%}, Mean gain: {df['gain_pct'].mean():.1f}%")

    return df, dedup_stats


def _select_deduplicated_backtests(db: Session, strategy: str, stats: dict) -> list:
    """
    Select the latest completed backtest per unique (start_date, end_date),
    excluding ML-contaminated runs (where ML actively influenced trades).
    """
    from backend.database import BacktestRun

    # Get all completed backtests for the strategy
    all_runs = db.query(BacktestRun).filter(
        BacktestRun.status == "completed",
        BacktestRun.strategy == strategy,
    ).all()

    stats["backtests_before"] = len(all_runs)

    # Exclude ML-contaminated backtests (ML ACTIVE profile_overrides)
    clean_runs = []
    for run in all_runs:
        if _is_ml_contaminated(run):
            stats["excluded_ml_contaminated"] += 1
            continue
        clean_runs.append(run)

    if not clean_runs:
        return []

    # Keep only the latest backtest per (start_date, end_date)
    best_per_range = {}
    for run in clean_runs:
        key = (str(run.start_date), str(run.end_date))
        if key not in best_per_range or run.id > best_per_range[key].id:
            best_per_range[key] = run

    return list(best_per_range.values())


def _is_ml_contaminated(run) -> bool:
    """Check if a backtest had ML actively influencing trade decisions."""
    overrides = run.profile_overrides
    if not overrides:
        return False

    ml_override = overrides.get("ml_signal", {})
    if not ml_override:
        return False

    # ML-contaminated if ML was enabled AND not in log-only mode
    enabled = ml_override.get("enabled", False)
    log_only = ml_override.get("log_only", True)

    return enabled and not log_only


def _deduplicate_trades(rows: list) -> list:
    """
    Deduplicate paired trades by (ticker, buy_date), keeping the record
    from the newest backtest (highest backtest_id).

    Only deduplicates SELL-based rows (full position exit).
    PARTIAL_SELL rows from the same buy are kept alongside the SELL row
    since they represent different exit events.
    """
    # Group by (ticker, date, sell_reason_type) to handle partial vs full sells
    best = {}
    for row in rows:
        # Use ticker + buy_date + whether it's a partial as the dedup key
        is_partial = "PARTIAL" in row.get("sell_reason", "")
        key = (row["ticker"], row["date"], is_partial)
        if key not in best or row["backtest_id"] > best[key]["backtest_id"]:
            best[key] = row

    return list(best.values())


def _pair_buy_sell_trades(trades) -> list:
    """
    Match each BUY to its subsequent SELL(s) for the same ticker within the same backtest.
    Aggregates partial sells by shares-weighted realized_gain_pct.
    """
    rows = []

    # Group by (backtest_id, ticker) for pairing
    buy_queue = {}  # (backtest_id, ticker) -> list of pending buys

    for trade in trades:
        key = (trade.backtest_id, trade.ticker)

        if trade.action == "BUY":
            if key not in buy_queue:
                buy_queue[key] = []
            buy_queue[key].append(trade)

        elif trade.action in ("SELL", "PARTIAL_SELL"):
            if key not in buy_queue or not buy_queue[key]:
                continue  # Orphan sell — skip

            # Match to oldest pending buy (FIFO)
            buy_trade = buy_queue[key][0]
            features = _extract_features(buy_trade)
            if features is None:
                buy_queue[key].pop(0)
                continue

            # Compute outcome from sell trade
            gain_pct = _nan_safe(trade.realized_gain_pct, 0.0)
            holding_days = trade.holding_days or 0
            sell_reason = trade.reason or ""

            # If this sell closes the position, pop the buy
            # For partial sells, keep the buy open until a full SELL comes
            if trade.action == "SELL":
                buy_queue[key].pop(0)

            rows.append({
                **features,
                "win": 1 if gain_pct > 0 else 0,
                "gain_pct": round(gain_pct, 2),
                "ticker": trade.ticker,
                "date": str(buy_trade.date),
                "backtest_id": trade.backtest_id,
                "holding_days": holding_days,
                "sell_reason": sell_reason,
            })

    return rows


def _extract_features(buy_trade) -> dict:
    """Extract the 15 high-signal features from a BUY trade's signal_factors."""
    sf = buy_trade.signal_factors or {}

    # Skip trades with no signal_factors (very old backtests)
    if not sf:
        return None

    # Feature defaults — used when signal_factors doesn't have the key
    _DEFAULTS = {
        "relative_volume": 1.0,
        "sector_rs_rank": 50.0,
    }

    # Build all possible features, then filter to FEATURE_COLUMNS
    all_features = {
        # Core quality
        "total_score": _nan_safe(buy_trade.canslim_score, 0.0),
        "composite_score": _nan_safe(sf.get("composite_score"), 0.0),
        "estimate_revision_bonus": _nan_safe(sf.get("estimate_revision_bonus"), 0.0),
        # Entry context
        "entry_type": ENTRY_TYPE_MAP.get(sf.get("entry_type", "standard"), 2),
        "market_regime": REGIME_MAP.get(sf.get("market_regime", "neutral"), 1),
        "coiled_spring": 1 if sf.get("coiled_spring", False) else 0,
        # Entry quality
        "soft_zone": 1 if sf.get("soft_zone", False) else 0,
        # CS detail (3 kept)
        "cs_weeks_in_base": _nan_safe(sf.get("cs_weeks_in_base"), 0.0),
        "cs_beat_streak": _nan_safe(sf.get("cs_beat_streak"), 0.0),
        "cs_days_to_earnings": _nan_safe(sf.get("cs_days_to_earnings"), 0.0),
        # Price action (5 kept)
        "relative_volume": _nan_safe(sf.get("relative_volume"), 1.0),
        "pct_from_21ma": _nan_safe(sf.get("pct_from_21ma"), 0.0),
        "pct_from_50ma": _nan_safe(sf.get("pct_from_50ma"), 0.0),
        "atr_pct": _nan_safe(sf.get("atr_pct"), 0.0),
        "sector_rs_rank": _nan_safe(sf.get("sector_rs_rank"), 50.0),
    }

    # Only return features in FEATURE_COLUMNS (v10: 15 features)
    return {k: v for k, v in all_features.items() if k in FEATURE_COLUMNS}


def get_feature_matrix(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X and labels y.
    Returns (X, y_win, y_gain, metadata_df)
    """
    if df.empty:
        return None, None, None, None

    X = df[FEATURE_COLUMNS].copy()

    # Fill any remaining NaNs with 0
    X = X.fillna(0)

    y_win = df["win"].values
    y_gain = df["gain_pct"].values

    metadata = df[["ticker", "date", "backtest_id", "holding_days", "sell_reason"]].copy()

    return X, y_win, y_gain, metadata
