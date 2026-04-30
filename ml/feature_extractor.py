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

# Feature columns — v13: 24 features (pruned from v12's 29)
#
# v12→v13 changes (Apr 29, 2026): pruned 5 zero-importance features per v9 importance dump.
#   Removed: coiled_spring, cs_weeks_in_base, cs_beat_streak, cs_days_to_earnings,
#            days_since_spy_pullback. All had importance == 0 in active v9 model.
#   The CS signal still reaches the model via composite_score and entry_type.
#   signal_factors STILL captures these fields for future use; only the matrix is pruned.
#
FEATURE_COLUMNS = [
    # Core quality (3) — the "what" of the stock
    "total_score",
    "composite_score",
    "estimate_revision_bonus",
    # Entry context (2) — the "when" and "how" of the entry
    "entry_type",           # ordinal: breakout=0, pre-breakout=1, standard=2
    "market_regime",        # ordinal: bearish=0, neutral=1, bullish=2
    # Entry quality flags (1)
    "soft_zone",            # binary 0/1 (below threshold = lower conviction)
    # Price action (5) — entry timing quality
    "relative_volume",       # volume / 50-day avg at entry (accumulation)
    "pct_from_21ma",         # % distance from 21-day MA (support proximity)
    "pct_from_50ma",         # % distance from 50-day MA (trend health)
    "atr_pct",               # volatility context (tighter = more coiled)
    "sector_rs_rank",        # sector momentum (group leadership)
    # Bonuses (4)
    "rs_line_bonus",         # RS trend confirmation (+8 rising, -5 diverging)
    "earnings_drift_bonus",  # post-earnings momentum for big beats
    "volume_dry_up",         # binary: tight volume in base (bullish consolidation)
    "volume_dry_up_score",   # continuous 0-100 (Apr 2026): how compressed recent vol vs baseline
    "deterministic_boost",   # boost for strong deterministic (N+S+L+I+M) scores
    # Individual CANSLIM components (6) — decompose total_score for finer signal
    "c_score",               # current quarterly earnings (0-15)
    "a_score",               # annual earnings growth (0-15)
    "n_score",               # new highs proximity (0-15)
    "s_score",               # supply/demand (0-15)
    "l_score",               # relative strength leader (0-15)
    "i_score",               # institutional ownership (0-10)
    # Market + sector context (2) — continuous gradient instead of binary
    "spy_pct_above_50ma",    # SPY % distance from 50MA (continuous market strength)
    "industry_group_rank",   # sector rank percentile (1-100)
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

    # profile_overrides is JSON in the ORM but stored as TEXT in production
    # Postgres — older rows come back as strings. Match the defensive pattern
    # used in backtest_queue.py (commit f17960c).
    if isinstance(overrides, str):
        try:
            import json
            overrides = json.loads(overrides)
        except (ValueError, TypeError):
            return False
    if not isinstance(overrides, dict):
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
    """Extract v13 features from a BUY trade's signal_factors."""
    sf = buy_trade.signal_factors or {}

    # Skip trades with no signal_factors (very old backtests)
    if not sf:
        return None

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
        # CS detail
        "cs_weeks_in_base": _nan_safe(sf.get("cs_weeks_in_base"), 0.0),
        "cs_beat_streak": _nan_safe(sf.get("cs_beat_streak"), 0.0),
        "cs_days_to_earnings": _nan_safe(sf.get("cs_days_to_earnings"), 0.0),
        # Price action
        "relative_volume": _nan_safe(sf.get("relative_volume"), 1.0),
        "pct_from_21ma": _nan_safe(sf.get("pct_from_21ma"), 0.0),
        "pct_from_50ma": _nan_safe(sf.get("pct_from_50ma"), 0.0),
        "atr_pct": _nan_safe(sf.get("atr_pct"), 0.0),
        "sector_rs_rank": _nan_safe(sf.get("sector_rs_rank"), 50.0),
        # Restored from v9
        "rs_line_bonus": _nan_safe(sf.get("rs_line_bonus"), 0.0),
        "earnings_drift_bonus": _nan_safe(sf.get("earnings_drift_bonus"), 0.0),
        "days_since_spy_pullback": _nan_safe(sf.get("days_since_spy_pullback"), 30.0),
        "volume_dry_up": 1 if sf.get("volume_dry_up", False) else 0,
        "volume_dry_up_score": _nan_safe(sf.get("volume_dry_up_score"), 0.0),
        "deterministic_boost": _nan_safe(sf.get("deterministic_boost"), 0.0),
        # Individual CANSLIM components
        "c_score": _nan_safe(sf.get("c_score"), 0.0),
        "a_score": _nan_safe(sf.get("a_score"), 0.0),
        "n_score": _nan_safe(sf.get("n_score"), 0.0),
        "s_score": _nan_safe(sf.get("s_score"), 0.0),
        "l_score": _nan_safe(sf.get("l_score"), 0.0),
        "i_score": _nan_safe(sf.get("i_score"), 0.0),
        # Market + sector context
        "spy_pct_above_50ma": _nan_safe(sf.get("spy_pct_above_50ma"), 0.0),
        "industry_group_rank": _nan_safe(sf.get("industry_group_rank"), 50.0),
    }

    # Only return features in FEATURE_COLUMNS (v13: 24 features)
    return {k: v for k, v in all_features.items() if k in FEATURE_COLUMNS}


def extract_live_trade_data(db: Session) -> pd.DataFrame:
    """
    Extract labeled training data from live AI portfolio trades.
    Pairs BUY trades with subsequent SELLs, same as backtester extraction.
    This supplements backtester data to bootstrap past cold-start.
    """
    from backend.database import AIPortfolioTrade

    trades = (
        db.query(AIPortfolioTrade)
        .order_by(AIPortfolioTrade.executed_at)
        .all()
    )

    if not trades:
        return pd.DataFrame()

    # Pair BUYs with SELLs by ticker (FIFO)
    buy_queue = {}  # ticker -> [buy_trade]
    rows = []

    for trade in trades:
        if trade.action == "BUY":
            buy_queue.setdefault(trade.ticker, []).append(trade)
        elif trade.action == "SELL" and trade.ticker in buy_queue and buy_queue[trade.ticker]:
            buy = buy_queue[trade.ticker].pop(0)
            sf = buy.signal_factors if isinstance(buy.signal_factors, dict) else {}
            if not sf:
                continue

            # Build the same feature dict shape as _extract_features (backtest path)
            # then filter to FEATURE_COLUMNS — keeps live and backtest extractors symmetric.
            features = {
                "total_score": _nan_safe(buy.canslim_score, 0.0),
                "composite_score": _nan_safe(sf.get("composite_score"), 0.0),
                "estimate_revision_bonus": _nan_safe(sf.get("estimate_revision_bonus"), 0.0),
                "entry_type": ENTRY_TYPE_MAP.get(sf.get("entry_type", "standard"), 2),
                "market_regime": REGIME_MAP.get(sf.get("market_regime", "neutral"), 1),
                "soft_zone": 1 if sf.get("soft_zone", False) else 0,
                "relative_volume": _nan_safe(sf.get("relative_volume"), 1.0),
                "pct_from_21ma": _nan_safe(sf.get("pct_from_21ma"), 0.0),
                "pct_from_50ma": _nan_safe(sf.get("pct_from_50ma"), 0.0),
                "atr_pct": _nan_safe(sf.get("atr_pct"), 0.0),
                "sector_rs_rank": _nan_safe(sf.get("sector_rs_rank"), 50.0),
                "rs_line_bonus": _nan_safe(sf.get("rs_line_bonus"), 0.0),
                "earnings_drift_bonus": _nan_safe(sf.get("earnings_drift_bonus"), 0.0),
                "volume_dry_up": 1 if sf.get("volume_dry_up", False) else 0,
                "volume_dry_up_score": _nan_safe(sf.get("volume_dry_up_score"), 0.0),
                "deterministic_boost": _nan_safe(sf.get("deterministic_boost"), 0.0),
                "c_score": _nan_safe(sf.get("c_score"), 0.0),
                "a_score": _nan_safe(sf.get("a_score"), 0.0),
                "n_score": _nan_safe(sf.get("n_score"), 0.0),
                "s_score": _nan_safe(sf.get("s_score"), 0.0),
                "l_score": _nan_safe(sf.get("l_score"), 0.0),
                "i_score": _nan_safe(sf.get("i_score"), 0.0),
                "spy_pct_above_50ma": _nan_safe(sf.get("spy_pct_above_50ma"), 0.0),
                "industry_group_rank": _nan_safe(sf.get("industry_group_rank"), 50.0),
            }

            features = {k: v for k, v in features.items() if k in FEATURE_COLUMNS}
            if len(features) < len(FEATURE_COLUMNS) * 0.5:
                continue  # Skip trades with too few features

            # Compute outcome
            gain_pct = ((trade.price / buy.price) - 1) * 100 if buy.price and buy.price > 0 else 0
            days_held = (trade.executed_at - buy.executed_at).days if trade.executed_at and buy.executed_at else 0
            sell_factors = trade.signal_factors if isinstance(trade.signal_factors, dict) else {}

            rows.append({
                **features,
                "win": 1 if gain_pct > 0 else 0,
                "gain_pct": round(gain_pct, 2),
                "ticker": trade.ticker,
                "date": str(buy.executed_at.date()) if buy.executed_at else "",
                "backtest_id": -1,  # Sentinel for live trades
                "holding_days": days_held,
                "sell_reason": sell_factors.get("sell_reason", ""),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    logger.info(f"Extracted {len(df)} live trades for ML training "
                f"(win rate: {df['win'].mean():.1%})")
    return df


def extract_combined_training_data(
    db: Session,
    strategy: str = "nostate_optimized",
    include_live: bool = True,
) -> tuple:
    """
    Extract combined training data from backtests + live trades.
    Returns (DataFrame, stats dict).
    """
    # Get backtest data
    bt_df, bt_stats = extract_training_data(db, strategy)

    if not include_live:
        return bt_df, bt_stats

    # Get live data
    live_df = extract_live_trade_data(db)

    stats = {
        **bt_stats,
        "live_trades": len(live_df),
    }

    if bt_df.empty and live_df.empty:
        return pd.DataFrame(), stats
    elif bt_df.empty:
        return live_df, stats
    elif live_df.empty:
        return bt_df, stats
    else:
        # Combine, then deduplicate by (ticker, date) keeping live trades preference
        combined = pd.concat([bt_df, live_df], ignore_index=True)
        # Live trades (backtest_id=-1) take priority over backtester trades
        combined = combined.sort_values("backtest_id", ascending=False)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="first")
        combined = combined.sort_values("date").reset_index(drop=True)
        stats["combined_total"] = len(combined)
        logger.info(f"Combined training data: {len(bt_df)} backtest + {len(live_df)} live "
                    f"= {len(combined)} total (after dedup)")
        return combined, stats


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
