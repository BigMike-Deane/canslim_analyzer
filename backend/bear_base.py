"""
Bear Market Base-Building Watchlist

Identifies stocks forming quality bases during bear markets (SPY < 50MA).
When the market turns bullish, this provides a ranked ready-to-buy list.

Key signals for bear base quality:
1. Tight consolidation (low ATR% in base)
2. Rising RS despite weak market (relative outperformance)
3. Institutional accumulation / volume dry-up in base
4. Strong fundamental scores (C+A score)
5. Proximity to pivot/breakout point
6. Industry group in top ranks (sector rotation leadership)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


def compute_readiness_score(stock) -> tuple:
    """
    Compute a 0-100 readiness score for a bear base candidate.

    Higher score = more ready to buy when market turns.

    Returns:
        (readiness_score, factors_dict)
    """
    factors = {}
    score = 0

    # 1. Fundamental quality: C + A scores (max 25 pts)
    c = getattr(stock, 'c_score', 0) or 0
    a = getattr(stock, 'a_score', 0) or 0
    ca_score = min(25, (c + a) / 30 * 25)  # C+A max is 30, scale to 25
    factors['fundamentals'] = round(ca_score, 1)
    score += ca_score

    # 2. Relative strength in bear market (max 25 pts)
    # RS > 1.0 during a bear = outperforming the falling market
    rs_3m = getattr(stock, 'rs_3m', None)
    rs_12m = getattr(stock, 'rs_12m', None)
    rs_score = 0
    if rs_3m is not None and rs_12m is not None:
        # Weighted toward recent RS (3m matters more in bear context)
        weighted_rs = rs_3m * 0.6 + rs_12m * 0.4
        if weighted_rs >= 1.3:
            rs_score = 25
        elif weighted_rs >= 1.15:
            rs_score = 20
        elif weighted_rs >= 1.0:
            rs_score = 15
        elif weighted_rs >= 0.85:
            rs_score = 8
        # Bonus if RS is improving (3m > 12m = momentum turning)
        if rs_3m > rs_12m:
            rs_score = min(25, rs_score + 3)
    factors['relative_strength'] = round(rs_score, 1)
    score += rs_score

    # 3. Base quality (max 20 pts)
    base_type = getattr(stock, 'base_type', 'none') or 'none'
    weeks = getattr(stock, 'weeks_in_base', 0) or 0
    atr_pct = getattr(stock, 'atr_pct', None)

    base_score = 0
    # Valid base pattern gets points
    quality_bases = {'cup_with_handle': 10, 'cup': 8, 'double_bottom': 8, 'flat_base': 7}
    base_score += quality_bases.get(base_type, 0)

    # Longer bases (6-20 weeks) are more powerful
    if 6 <= weeks <= 20:
        base_score += 5
    elif weeks > 20:
        base_score += 3  # Too long might indicate weakness
    elif 3 <= weeks < 6:
        base_score += 2

    # Tight base (low ATR) — the tighter the base, the more coiled
    if atr_pct is not None and atr_pct > 0:
        if atr_pct <= 2.0:
            base_score += 5  # Very tight
        elif atr_pct <= 3.5:
            base_score += 3
        elif atr_pct <= 5.0:
            base_score += 1

    base_score = min(20, base_score)
    factors['base_quality'] = round(base_score, 1)
    score += base_score

    # 4. Accumulation signals (max 15 pts)
    acc_score = 0
    if getattr(stock, 'volume_dry_up', False):
        acc_score += 7  # Volume drying up in base = healthy consolidation
    if getattr(stock, 'institutional_accumulation', False):
        acc_score += 5  # Institutions quietly accumulating
    insider = getattr(stock, 'insider_sentiment', '')
    if insider == 'bullish':
        acc_score += 3
    acc_score = min(15, acc_score)
    factors['accumulation'] = round(acc_score, 1)
    score += acc_score

    # 5. Industry group strength (max 10 pts)
    ig_rank = getattr(stock, 'industry_group_rank', None)
    ig_score = 0
    if ig_rank is not None:
        if ig_rank >= 80:
            ig_score = 10
        elif ig_rank >= 60:
            ig_score = 7
        elif ig_rank >= 40:
            ig_score = 4
    factors['industry_group'] = round(ig_score, 1)
    score += ig_score

    # 6. Pivot proximity bonus (max 5 pts)
    pivot = getattr(stock, 'pivot_price', None)
    price = getattr(stock, 'current_price', None)
    pivot_score = 0
    if pivot and price and pivot > 0:
        pct_from_pivot = ((pivot - price) / pivot) * 100
        if 0 <= pct_from_pivot <= 5:
            pivot_score = 5  # Within 5% of pivot — ready to break out
        elif 5 < pct_from_pivot <= 10:
            pivot_score = 3
        elif 10 < pct_from_pivot <= 15:
            pivot_score = 1
    factors['pivot_proximity'] = round(pivot_score, 1)
    score += pivot_score

    return round(min(100, score), 1), factors


def update_bear_base_candidates(db: Session) -> dict:
    """
    Scan for stocks building quality bases during bear market.
    Called after each scan cycle when SPY is below 50MA.

    Returns:
        Summary dict with counts and top candidates
    """
    from backend.database import Stock, BearBaseCandidate

    # Find stocks with valid bases and decent fundamentals
    # Minimum criteria: has a base pattern, some RS data, price > 0
    candidates = db.query(Stock).filter(
        Stock.current_price > 0,
        Stock.canslim_score != None,
        Stock.rs_3m != None,
        Stock.base_type != None,
        Stock.base_type != 'none',
        Stock.base_type != '',
        Stock.weeks_in_base >= 3,
    ).all()

    # Also include stocks with volume dry-up even without a formal base pattern
    dryup_candidates = db.query(Stock).filter(
        Stock.current_price > 0,
        Stock.canslim_score != None,
        Stock.rs_3m != None,
        Stock.volume_dry_up == True,
        Stock.canslim_score >= 50,  # At least moderate quality
    ).all()

    # Combine, deduplicate
    seen = set()
    all_candidates = []
    for stock in candidates + dryup_candidates:
        if stock.ticker not in seen:
            seen.add(stock.ticker)
            all_candidates.append(stock)

    # Compute readiness scores
    scored = []
    for stock in all_candidates:
        readiness, factors = compute_readiness_score(stock)
        if readiness >= 20:  # Minimum threshold to be on the list
            scored.append((stock, readiness, factors))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Upsert into BearBaseCandidate table
    now = datetime.now(timezone.utc)
    existing = {c.ticker: c for c in db.query(BearBaseCandidate).all()}
    updated_count = 0
    new_count = 0

    for stock, readiness, factors in scored:
        pivot = getattr(stock, 'pivot_price', None)
        price = stock.current_price
        pct_from_pivot = None
        if pivot and price and pivot > 0:
            pct_from_pivot = round(((pivot - price) / pivot) * 100, 1)

        if stock.ticker in existing:
            # Update existing
            rec = existing[stock.ticker]
            rec.canslim_score = stock.canslim_score
            rec.c_score = stock.c_score
            rec.a_score = stock.a_score
            rec.l_score = stock.l_score
            rec.rs_12m = stock.rs_12m
            rec.rs_3m = stock.rs_3m
            rec.industry_group_rank = stock.industry_group_rank
            rec.base_type = stock.base_type
            rec.weeks_in_base = stock.weeks_in_base
            rec.atr_pct = stock.atr_pct
            rec.pivot_price = pivot
            rec.current_price = price
            rec.pct_from_pivot = pct_from_pivot
            rec.volume_dry_up = stock.volume_dry_up
            rec.institutional_accumulation = stock.institutional_accumulation
            rec.insider_sentiment = stock.insider_sentiment
            rec.readiness_score = readiness
            rec.readiness_factors = factors
            rec.last_updated = now
            if rec.first_seen:
                rec.days_on_list = (now - rec.first_seen).days
            updated_count += 1
            del existing[stock.ticker]  # Mark as still active
        else:
            # New candidate
            rec = BearBaseCandidate(
                ticker=stock.ticker,
                name=stock.name,
                sector=stock.sector,
                industry=stock.industry,
                canslim_score=stock.canslim_score,
                c_score=stock.c_score,
                a_score=stock.a_score,
                l_score=stock.l_score,
                rs_12m=stock.rs_12m,
                rs_3m=stock.rs_3m,
                industry_group_rank=stock.industry_group_rank,
                base_type=stock.base_type,
                weeks_in_base=stock.weeks_in_base,
                atr_pct=stock.atr_pct,
                pivot_price=pivot,
                current_price=price,
                pct_from_pivot=pct_from_pivot,
                volume_dry_up=stock.volume_dry_up,
                institutional_accumulation=stock.institutional_accumulation,
                insider_sentiment=stock.insider_sentiment,
                readiness_score=readiness,
                readiness_factors=factors,
                first_seen=now,
                last_updated=now,
                days_on_list=0,
            )
            db.add(rec)
            new_count += 1

    # Remove candidates that no longer qualify
    removed = 0
    for ticker, stale_rec in existing.items():
        db.delete(stale_rec)
        removed += 1

    db.commit()

    top_5 = [(s.ticker, r) for s, r, _ in scored[:5]]
    logger.info(f"Bear base watchlist: {new_count} new, {updated_count} updated, "
                f"{removed} removed. Top: {top_5}")

    return {
        "total": len(scored),
        "new": new_count,
        "updated": updated_count,
        "removed": removed,
        "top_candidates": [
            {
                "ticker": s.ticker,
                "name": s.name,
                "readiness_score": r,
                "canslim_score": s.canslim_score,
                "base_type": s.base_type,
                "weeks_in_base": s.weeks_in_base,
                "rs_3m": s.rs_3m,
            }
            for s, r, _ in scored[:10]
        ],
    }


def _safe(val):
    """Sanitize NaN/Inf values for JSON serialization."""
    if val is None:
        return None
    try:
        if val != val:  # NaN check
            return None
        if val == float('inf') or val == float('-inf'):
            return None
    except (TypeError, ValueError):
        pass
    return val


def get_bear_base_list(db: Session, limit: int = 50) -> List[dict]:
    """Get the current bear base watchlist, sorted by readiness score."""
    from backend.database import BearBaseCandidate

    candidates = db.query(BearBaseCandidate).order_by(
        BearBaseCandidate.readiness_score.desc()
    ).limit(limit).all()

    return [
        {
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "industry": c.industry,
            "canslim_score": _safe(c.canslim_score),
            "c_score": _safe(c.c_score),
            "a_score": _safe(c.a_score),
            "l_score": _safe(c.l_score),
            "rs_12m": _safe(c.rs_12m),
            "rs_3m": _safe(c.rs_3m),
            "industry_group_rank": c.industry_group_rank,
            "base_type": c.base_type,
            "weeks_in_base": c.weeks_in_base,
            "atr_pct": round(c.atr_pct, 2) if c.atr_pct and c.atr_pct == c.atr_pct else None,
            "pivot_price": _safe(c.pivot_price),
            "current_price": _safe(c.current_price),
            "pct_from_pivot": _safe(c.pct_from_pivot),
            "volume_dry_up": c.volume_dry_up,
            "institutional_accumulation": c.institutional_accumulation,
            "insider_sentiment": c.insider_sentiment,
            "readiness_score": _safe(c.readiness_score),
            "readiness_factors": c.readiness_factors,
            "days_on_list": c.days_on_list,
            "first_seen": c.first_seen.isoformat() if c.first_seen else None,
        }
        for c in candidates
    ]
