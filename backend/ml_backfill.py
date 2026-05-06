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

from backend.database import (
    AIPortfolioTrade, MLPrediction, SessionLocal,
    Stock, StockScore,
)

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

        # Pyramids are not ML-gated and must NOT anchor an ml_prediction
        # outcome. As of 2026-05-05 live pyramids carry action="PYRAMID";
        # legacy rows (pre-fix) carry action="BUY" + reason="PYRAMID:..."
        # so we exclude both shapes defensively.
        from sqlalchemy import or_
        buy = (
            db.query(AIPortfolioTrade)
            .filter(
                AIPortfolioTrade.ticker == pred.ticker,
                AIPortfolioTrade.action == "BUY",
                AIPortfolioTrade.executed_at >= day_start,
                AIPortfolioTrade.executed_at < day_end,
                or_(
                    AIPortfolioTrade.reason.is_(None),
                    ~AIPortfolioTrade.reason.like("PYRAMID:%"),
                ),
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


def _price_at(db: Session, stock_id: int, target_date: date, *, after: bool) -> Optional[float]:
    """Find the StockScore.current_price closest to target_date.

    after=False: latest score on or before target_date (for the "buy day" price).
    after=True:  earliest score on or after target_date (for the "exit day" price).
    """
    q = db.query(StockScore).filter(StockScore.stock_id == stock_id)
    if after:
        q = q.filter(StockScore.date >= target_date).order_by(StockScore.date.asc())
    else:
        q = q.filter(StockScore.date <= target_date).order_by(StockScore.date.desc())
    score = q.first()
    if score is None or not score.current_price or score.current_price <= 0:
        return None
    return score.current_price


def counterfactual_backfill(
    db: Optional[Session] = None,
    lookback_days: int = 90,
    hold_days: int = 21,
    min_gain_pct: float = 10.0,
) -> dict:
    """Synthesize outcomes for ml_predictions that didn't lead to a real trade.

    The realized backfill (`backfill_actual_outcomes`) only fills predictions
    that resulted in actual buy/sell pairs — typically <20% of evaluated
    candidates because the live ML gate vetoes most. The remaining ~80% sit
    with NULL outcomes forever, which:
      1) loses 5x potential training data, and
      2) creates a feedback loop where future training reflects only the
         current model's preferences (concept drift).

    For each pending prediction old enough that we have hold_days of price
    data after it, we look up the stock's price on prediction_date (start)
    and prediction_date+hold_days (end), compute the gain percent, and label
    it according to min_gain_pct. Default 10.0 matches v12's training task.

    Skips predictions that have a matching real BUY — those are owned by
    the realized backfill (open or closed). Idempotent: only touches NULL.
    """
    own_session = db is None
    if own_session:
        db = SessionLocal()

    today = date.today()
    earliest = today - timedelta(days=lookback_days)
    # Need hold_days of post-prediction price data, so prediction_date can be
    # at most today - hold_days.
    latest = today - timedelta(days=hold_days)

    pending = (
        db.query(MLPrediction)
        .filter(
            MLPrediction.actual_outcome.is_(None),
            MLPrediction.prediction_date >= earliest,
            MLPrediction.prediction_date <= latest,
        )
        .all()
    )

    counts = {
        "checked": len(pending),
        "updated": 0,
        "skipped_has_buy": 0,
        "no_price_data": 0,
        "no_stock": 0,
    }

    # Cache stock lookups: many predictions hit the same ticker.
    stock_id_cache: dict = {}

    for pred in pending:
        # Skip if a real buy exists for this ticker on prediction_date —
        # realized backfill owns those rows.
        day_start = datetime.combine(pred.prediction_date, datetime.min.time())
        day_end = day_start + timedelta(days=2)
        # Same pyramid-exclusion as backfill_actual_outcomes — only a real
        # ML-gated BUY counts as "owned by the realized backfill".
        from sqlalchemy import or_
        has_buy = (
            db.query(AIPortfolioTrade.id)
            .filter(
                AIPortfolioTrade.ticker == pred.ticker,
                AIPortfolioTrade.action == "BUY",
                AIPortfolioTrade.executed_at >= day_start,
                AIPortfolioTrade.executed_at < day_end,
                or_(
                    AIPortfolioTrade.reason.is_(None),
                    ~AIPortfolioTrade.reason.like("PYRAMID:%"),
                ),
            )
            .first()
        )
        if has_buy is not None:
            counts["skipped_has_buy"] += 1
            continue

        if pred.ticker not in stock_id_cache:
            stock = db.query(Stock.id).filter(Stock.ticker == pred.ticker).first()
            stock_id_cache[pred.ticker] = stock[0] if stock else None
        stock_id = stock_id_cache[pred.ticker]
        if stock_id is None:
            counts["no_stock"] += 1
            continue

        start_price = _price_at(db, stock_id, pred.prediction_date, after=False)
        end_price = _price_at(
            db, stock_id, pred.prediction_date + timedelta(days=hold_days), after=True
        )
        if start_price is None or end_price is None:
            counts["no_price_data"] += 1
            continue

        gain_pct = (end_price / start_price - 1) * 100.0
        pred.actual_gain_pct = round(gain_pct, 2)
        pred.actual_outcome = 1 if gain_pct > min_gain_pct else 0
        counts["updated"] += 1

    if own_session:
        try:
            db.commit()
        finally:
            db.close()

    logger.info(
        f"ml_predictions counterfactual backfill (hold={hold_days}d, "
        f"min_gain={min_gain_pct}%): checked={counts['checked']} "
        f"updated={counts['updated']} skipped_has_buy={counts['skipped_has_buy']} "
        f"no_price_data={counts['no_price_data']} no_stock={counts['no_stock']}"
    )
    return counts
