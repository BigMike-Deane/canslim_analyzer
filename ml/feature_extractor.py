"""
Extract training features from backtest trades.

Pairs BUY trades with their subsequent SELL(s) to build labeled training data.
Features come from signal_factors JSON stored on each BacktestTrade.
"""

import logging
from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Feature columns (12 features — small dataset demands low dimensionality)
FEATURE_COLUMNS = [
    "total_score",
    "composite_score",
    "entry_type",           # ordinal: breakout=0, pre-breakout=1, standard=2
    "market_regime",        # ordinal: bearish=0, neutral=1, bullish=2
    "rs_line_bonus",
    "earnings_drift_bonus",
    "estimate_revision_bonus",
    "coiled_spring",        # binary 0/1
    "soft_zone",            # binary 0/1
    "soft_zone_multiplier",
    "deterministic_boost",
    "is_growth_stock",      # binary 0/1
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
) -> pd.DataFrame:
    """
    Extract labeled training data from completed backtest trades.

    Returns DataFrame with FEATURE_COLUMNS + label columns:
        win (binary), gain_pct (float),
        ticker, date, backtest_id, holding_days, sell_reason (metadata)
    """
    from backend.database import BacktestRun, BacktestTrade

    # Find completed backtests for the strategy
    query = db.query(BacktestRun).filter(
        BacktestRun.status == "completed",
        BacktestRun.strategy == strategy,
    )
    if backtest_ids:
        query = query.filter(BacktestRun.id.in_(backtest_ids))

    runs = query.all()
    if not runs:
        logger.warning(f"No completed backtests found for strategy '{strategy}'")
        return pd.DataFrame()

    run_ids = [r.id for r in runs]
    logger.info(f"Extracting from {len(run_ids)} backtest runs: {run_ids}")

    # Pull all trades for these runs
    trades = (
        db.query(BacktestTrade)
        .filter(BacktestTrade.backtest_id.in_(run_ids))
        .order_by(BacktestTrade.backtest_id, BacktestTrade.date, BacktestTrade.id)
        .all()
    )

    if not trades:
        logger.warning("No trades found in selected backtests")
        return pd.DataFrame()

    # Pair BUYs with their subsequent SELLs
    rows = _pair_buy_sell_trades(trades)

    if not rows:
        logger.warning("No buy-sell pairs found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} labeled trades from {len(run_ids)} backtests")
    logger.info(f"Win rate: {df['win'].mean():.1%}, Mean gain: {df['gain_pct'].mean():.1f}%")

    return df


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
    """Extract the 12 features from a BUY trade's signal_factors."""
    sf = buy_trade.signal_factors or {}

    # Skip trades with no signal_factors (very old backtests)
    if not sf:
        return None

    return {
        "total_score": _nan_safe(buy_trade.canslim_score, 0.0),
        "composite_score": _nan_safe(sf.get("composite_score"), 0.0),
        "entry_type": ENTRY_TYPE_MAP.get(sf.get("entry_type", "standard"), 2),
        "market_regime": REGIME_MAP.get(sf.get("market_regime", "neutral"), 1),
        "rs_line_bonus": _nan_safe(sf.get("rs_line_bonus"), 0.0),
        "earnings_drift_bonus": _nan_safe(sf.get("earnings_drift_bonus"), 0.0),
        "estimate_revision_bonus": _nan_safe(sf.get("estimate_revision_bonus"), 0.0),
        "coiled_spring": 1 if sf.get("coiled_spring", False) else 0,
        "soft_zone": 1 if sf.get("soft_zone", False) else 0,
        "soft_zone_multiplier": _nan_safe(sf.get("soft_zone_multiplier", 1.0), 1.0),
        "deterministic_boost": _nan_safe(sf.get("deterministic_boost", 0), 0.0),
        "is_growth_stock": 1 if buy_trade.is_growth_stock else 0,
    }


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
