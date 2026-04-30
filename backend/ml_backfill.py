"""
Backfill `actual_outcome` and `actual_gain_pct` on ml_predictions rows.

Each ml_predictions row records what the model said about a candidate at
evaluation time. Once that candidate's eventual buy/sell pair has closed,
we know what actually happened — backfilling the row turns the audit log
into ground truth that a trainer or AUC monitor can consume.

Vetoes (predictions that did NOT result in a buy) stay NULL forever — we
have no realized outcome to attribute. They are still useful for studying
the confidence distribution at the gate.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from backend.database import AIPortfolioTrade, MLPrediction, SessionLocal

logger = logging.getLogger(__name__)


def _gain_pct_from_sell(sell: AIPortfolioTrade) -> Optional[float]:
    """Extract or compute gain_pct from a SELL trade.

    Prefer signal_factors.gain_pct (what the trader logged at sell time).
    Fall back to (realized_gain / total_cost) * 100 when signal_factors is missing.
    """
    sf = sell.signal_factors or {}
    sf_gain = sf.get("gain_pct")
    if isinstance(sf_gain, (int, float)) and sf_gain == sf_gain:
        return float(sf_gain)
    if sell.cost_basis and sell.shares and sell.realized_gain is not None:
        cost = sell.cost_basis * sell.shares
        if cost > 0:
            return (sell.realized_gain / cost) * 100.0
    return None


def _find_closing_sell(
    db: Session, ticker: str, after: datetime
) -> Optional[AIPortfolioTrade]:
    """Find the next full-position SELL for `ticker` after `after`.

    Skips PARTIAL sells — they don't close the position, so they don't
    represent the trade's final outcome.
    """
    sells = (
        db.query(AIPortfolioTrade)
        .filter(
            AIPortfolioTrade.ticker == ticker,
            AIPortfolioTrade.action == "SELL",
            AIPortfolioTrade.executed_at > after,
            AIPortfolioTrade.realized_gain.isnot(None),
        )
        .order_by(AIPortfolioTrade.executed_at.asc())
        .all()
    )
    for s in sells:
        if s.reason and "PARTIAL" in s.reason.upper():
            continue
        return s
    return None


def backfill_actual_outcomes(
    db: Optional[Session] = None,
    lookback_days: int = 90,
) -> dict:
    """Backfill outcomes on pending ml_predictions rows.

    Args:
        db: SQLAlchemy session. Caller owns the transaction. If omitted,
            a fresh SessionLocal() is used and committed/closed here.
        lookback_days: only scan predictions whose prediction_date falls
            within this window. Bounds the workload as the table grows.

    Returns:
        {"checked": N, "updated": M, "still_open": K, "no_buy": L}
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()

    cutoff = date.today() - timedelta(days=lookback_days)
    pending = (
        db.query(MLPrediction)
        .filter(
            MLPrediction.actual_outcome.is_(None),
            MLPrediction.prediction_date >= cutoff,
        )
        .all()
    )

    counts = {"checked": len(pending), "updated": 0, "still_open": 0, "no_buy": 0}

    for pred in pending:
        # Predictions are made during a scan tick; the buy lands the same day
        # or the following morning. A 2-day window catches both. Boundaries
        # are naive UTC to match how AIPortfolioTrade.executed_at is stored.
        day_start = datetime.combine(pred.prediction_date, datetime.min.time())
        day_end = day_start + timedelta(days=2)

        buy = (
            db.query(AIPortfolioTrade)
            .filter(
                AIPortfolioTrade.ticker == pred.ticker,
                AIPortfolioTrade.action == "BUY",
                AIPortfolioTrade.executed_at >= day_start,
                AIPortfolioTrade.executed_at < day_end,
            )
            .order_by(AIPortfolioTrade.executed_at.asc())
            .first()
        )
        if buy is None:
            counts["no_buy"] += 1
            continue

        sell = _find_closing_sell(db, pred.ticker, buy.executed_at)
        if sell is None:
            counts["still_open"] += 1
            continue

        gain_pct = _gain_pct_from_sell(sell)
        if gain_pct is None:
            # SELL exists but is missing gain data — leave NULL and move on.
            continue

        pred.actual_gain_pct = round(gain_pct, 2)
        pred.actual_outcome = 1 if gain_pct > 0 else 0
        counts["updated"] += 1

    if own_session:
        try:
            db.commit()
        finally:
            db.close()

    logger.info(
        f"ml_predictions backfill: checked={counts['checked']} "
        f"updated={counts['updated']} still_open={counts['still_open']} "
        f"no_buy={counts['no_buy']}"
    )
    return counts
