"""
Industry Group Strength Module

Computes industry-group-level relative strength rankings.
IBD research shows ~37% of a stock's price move comes from its industry group.

Key concepts:
- Groups all stocks by their `industry` field (more granular than sector)
- Computes group-level RS by averaging member stocks' RS values
- Ranks groups 1-100 (percentile) — top 20 get bonus, bottom 40 get penalty
- Updates are batched and run after each scan cycle
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


def compute_industry_group_rankings(db: Session) -> Dict[str, dict]:
    """
    Compute industry group strength rankings from current stock RS data.

    Groups stocks by `industry` field, averages their rs_12m and rs_3m values,
    then ranks groups by a weighted composite (40% rs_12m + 60% rs_3m to favor
    recent momentum — matching IBD's emphasis on recent group rotation).

    Returns:
        Dict mapping industry_group -> {
            'rank': 1-100 percentile (100 = strongest),
            'composite_rs': weighted average RS,
            'stock_count': number of stocks in group,
            'avg_rs_12m': average 12-month RS,
            'avg_rs_3m': average 3-month RS,
        }
    """
    from backend.database import Stock

    # Query all stocks with valid RS data grouped by industry
    rows = db.query(
        Stock.industry,
        func.avg(Stock.rs_12m).label('avg_rs_12m'),
        func.avg(Stock.rs_3m).label('avg_rs_3m'),
        func.count(Stock.id).label('stock_count'),
    ).filter(
        Stock.industry != None,
        Stock.industry != '',
        Stock.rs_12m != None,
        Stock.rs_3m != None,
        Stock.current_price > 0,
    ).group_by(Stock.industry).having(
        func.count(Stock.id) >= 2  # Need at least 2 stocks for meaningful group RS
    ).all()

    if not rows:
        logger.warning("No industry groups with sufficient RS data")
        return {}

    # Compute composite RS per group (60% recent 3m, 40% longer-term 12m)
    groups = []
    for row in rows:
        avg_12m = row.avg_rs_12m or 0
        avg_3m = row.avg_rs_3m or 0
        composite = avg_12m * 0.40 + avg_3m * 0.60
        groups.append({
            'industry': row.industry,
            'composite_rs': composite,
            'avg_rs_12m': round(avg_12m, 4),
            'avg_rs_3m': round(avg_3m, 4),
            'stock_count': row.stock_count,
        })

    # Sort by composite RS and assign percentile ranks (1-100)
    groups.sort(key=lambda g: g['composite_rs'])
    total = len(groups)

    rankings = {}
    for i, group in enumerate(groups):
        # Percentile rank: 1 = weakest, 100 = strongest
        rank = round(((i + 1) / total) * 100)
        rank = max(1, min(100, rank))
        rankings[group['industry']] = {
            'rank': rank,
            'composite_rs': round(group['composite_rs'], 4),
            'stock_count': group['stock_count'],
            'avg_rs_12m': group['avg_rs_12m'],
            'avg_rs_3m': group['avg_rs_3m'],
        }

    logger.info(f"Computed industry group rankings for {total} groups "
                f"(top: {groups[-1]['industry']} RS={groups[-1]['composite_rs']:.3f}, "
                f"bottom: {groups[0]['industry']} RS={groups[0]['composite_rs']:.3f})")

    return rankings


def get_industry_group_bonus(rank: int) -> float:
    """
    Calculate scoring bonus/penalty based on industry group rank.

    IBD research: stocks in top industry groups significantly outperform.
    We apply a graduated bonus/penalty to the L (Leader) score component.

    Args:
        rank: Percentile rank 1-100 (100 = strongest group)

    Returns:
        Score adjustment (-3 to +3 points, applied to L score)
    """
    if rank >= 80:
        # Top 20% groups: +2 to +3 bonus
        return 2.0 + (rank - 80) / 20.0  # 80→+2.0, 100→+3.0
    elif rank >= 60:
        # 60-80%: +0.5 to +2 bonus
        return 0.5 + (rank - 60) / 20.0 * 1.5
    elif rank >= 40:
        # 40-60%: neutral (no adjustment)
        return 0
    elif rank >= 20:
        # 20-40%: -0.5 to -1.5 penalty
        return -0.5 - (40 - rank) / 20.0
    else:
        # Bottom 20%: -1.5 to -3 penalty
        return -1.5 - (20 - rank) / 20.0 * 1.5


def update_stock_group_ranks(db: Session, rankings: Dict[str, dict]) -> int:
    """
    Batch-update industry_group_rank on all Stock records.

    Args:
        db: Database session
        rankings: Output from compute_industry_group_rankings()

    Returns:
        Number of stocks updated
    """
    from backend.database import Stock

    if not rankings:
        return 0

    updated = 0
    # Batch fetch all stocks with an industry field
    stocks = db.query(Stock).filter(
        Stock.industry != None,
        Stock.industry != '',
    ).all()

    for stock in stocks:
        group_data = rankings.get(stock.industry)
        if group_data:
            new_rank = group_data['rank']
            if stock.industry_group_rank != new_rank:
                stock.industry_group_rank = new_rank
                updated += 1
        else:
            # Industry not in rankings (too few members) — neutral
            if stock.industry_group_rank is not None:
                stock.industry_group_rank = None
                updated += 1

    if updated > 0:
        db.commit()
        logger.info(f"Updated industry_group_rank for {updated} stocks")

    return updated


def get_top_groups(db: Session, limit: int = 20) -> List[dict]:
    """Get the top N industry groups by strength for API/frontend display."""
    rankings = compute_industry_group_rankings(db)
    sorted_groups = sorted(rankings.items(), key=lambda x: x[1]['rank'], reverse=True)
    return [
        {'industry': name, **data}
        for name, data in sorted_groups[:limit]
    ]


def get_bottom_groups(db: Session, limit: int = 20) -> List[dict]:
    """Get the bottom N industry groups by strength."""
    rankings = compute_industry_group_rankings(db)
    sorted_groups = sorted(rankings.items(), key=lambda x: x[1]['rank'])
    return [
        {'industry': name, **data}
        for name, data in sorted_groups[:limit]
    ]


def get_group_rotation_summary(db: Session) -> dict:
    """
    Get a summary of sector/group rotation for the bear market report.
    Compares 3m RS vs 12m RS to identify groups gaining/losing momentum.
    """
    rankings = compute_industry_group_rankings(db)
    if not rankings:
        return {'improving': [], 'deteriorating': [], 'total_groups': 0}

    improving = []
    deteriorating = []

    for name, data in rankings.items():
        rs_diff = data['avg_rs_3m'] - data['avg_rs_12m']
        entry = {
            'industry': name,
            'rank': data['rank'],
            'rs_diff': round(rs_diff, 4),
            'avg_rs_3m': data['avg_rs_3m'],
            'avg_rs_12m': data['avg_rs_12m'],
            'stock_count': data['stock_count'],
        }
        if rs_diff > 0.05:  # 3m RS meaningfully above 12m — gaining momentum
            improving.append(entry)
        elif rs_diff < -0.05:  # Losing momentum
            deteriorating.append(entry)

    improving.sort(key=lambda x: x['rs_diff'], reverse=True)
    deteriorating.sort(key=lambda x: x['rs_diff'])

    return {
        'improving': improving[:15],
        'deteriorating': deteriorating[:15],
        'total_groups': len(rankings),
    }
