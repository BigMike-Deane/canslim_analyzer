"""Admin routes: user management (invite-only, Google Sign-In) +
operational diagnostics that are too sensitive for the public API."""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db, User, AIPortfolioTrade, AIPortfolioConfig
from backend.auth import get_admin_user, UserCreate, UserResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users")
async def list_users(current_user: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """List all users (admin only)."""
    users = db.query(User).order_by(User.id).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "display_name": u.display_name,
            "is_active": u.is_active,
            "is_admin": u.is_admin,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]


@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Create a new user by email (admin only). User signs in via Google."""
    existing = db.query(User).filter(User.email == user_data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        email=user_data.email.lower(),
        hashed_password="",  # No password needed — Google Sign-In only
        display_name=user_data.display_name or user_data.email.split("@")[0],
        is_active=True,
        is_admin=False,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        display_name=new_user.display_name,
        is_admin=new_user.is_admin,
        is_active=new_user.is_active,
    )


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Update a user (admin only). Can toggle active/admin status, change name."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent admin from disabling themselves
    if user.id == current_user.id and data.is_active is False:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")
    if user.id == current_user.id and data.is_admin is False:
        raise HTTPException(status_code=400, detail="Cannot remove your own admin privileges")

    if data.display_name is not None:
        user.display_name = data.display_name
    if data.is_active is not None:
        user.is_active = data.is_active
    if data.is_admin is not None:
        user.is_admin = data.is_admin

    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin,
        is_active=user.is_active,
    )


def _summarize_trades(trades: list, days: int) -> dict:
    """Compute the same stats the May 5 cs_bear health-check script computed,
    but server-side and reusable. Returns a JSON-serializable dict."""
    by_action = {}
    ml_confs = []
    realized_pcts = []
    exit_reasons = {}

    for t in trades:
        by_action[t.action] = by_action.get(t.action, 0) + 1

        sf = t.signal_factors
        if sf:
            try:
                if isinstance(sf, str):
                    sf = json.loads(sf)
                if isinstance(sf, dict) and sf.get('ml_confidence') is not None:
                    ml_confs.append(float(sf['ml_confidence']))
            except Exception:
                pass

        if t.action == 'SELL':
            # Realized-pct math requires both cost_basis and shares; skip
            # silently when either is missing (legacy rows can lack them).
            if t.realized_gain is not None and t.cost_basis and t.shares:
                try:
                    cost_total = float(t.cost_basis) * float(t.shares)
                    if cost_total > 0:
                        realized_pcts.append(float(t.realized_gain) / cost_total * 100)
                except Exception:
                    pass

            # Exit-reason categorization runs on every SELL regardless of
            # whether cost_basis is set — it only needs the reason string.
            r = (t.reason or '').upper()
            if 'STOP LOSS' in r:
                key = 'STOP_LOSS'
            elif 'PARTIAL PROFIT' in r:
                key = 'PARTIAL_PROFIT'
            elif 'TRAILING' in r:
                key = 'TRAILING_STOP'
            elif 'PRE-EARNINGS' in r:
                key = 'PRE_EARNINGS'
            elif 'TAKE PROFIT' in r:
                key = 'TAKE_PROFIT'
            elif 'SCORE CRASH' in r:
                key = 'SCORE_CRASH'
            else:
                key = 'OTHER'
            exit_reasons[key] = exit_reasons.get(key, 0) + 1

    buys = by_action.get('BUY', 0)
    sells = by_action.get('SELL', 0)
    pyramids = by_action.get('PYRAMID', 0)
    safe_days = max(days, 1)

    summary = {
        'days': days,
        'total_trades': len(trades),
        'by_action': by_action,
        'entry_rate_per_day': round(buys / safe_days, 3),
        'exit_rate_per_day': round(sells / safe_days, 3),
        'pyramid_count': pyramids,
        'exit_reasons': exit_reasons,
        'ml_confidence': {
            'n_with': len(ml_confs),
            'min': round(min(ml_confs), 4) if ml_confs else None,
            'max': round(max(ml_confs), 4) if ml_confs else None,
            'mean': round(sum(ml_confs) / len(ml_confs), 4) if ml_confs else None,
        },
    }
    if realized_pcts:
        wins = sum(1 for p in realized_pcts if p > 0)
        summary['realized_sell_pct'] = {
            'n': len(realized_pcts),
            'win_rate': round(wins / len(realized_pcts), 4),
            'mean': round(sum(realized_pcts) / len(realized_pcts), 2),
            'min': round(min(realized_pcts), 2),
            'max': round(max(realized_pcts), 2),
        }
    else:
        summary['realized_sell_pct'] = None
    return summary


@router.get("/strategy-health")
async def strategy_health_audit(
    strategy: str = Query(default="nostate_cs_bear"),
    cutoff_iso: str = Query(default="2026-04-29T00:00:00", description="UTC datetime; trades on/after = post-graduation"),
    window_start_iso: str = Query(default="2026-04-08T00:00:00", description="Earliest trade considered (pre-window start)"),
    exclude_pyramid_reason: bool = Query(default=True, description="Filter out reason LIKE 'PYRAMID:%' rows per May 5 audit"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Pre vs post-graduation health audit for a strategy. Replicates the
    May 5 cs_bear manual audit so a remote routine can re-run it on a
    schedule without needing Tailscale access to the VPS.

    Default cutoff is 2026-04-29 (cs_bear ML graduation date).
    Default window starts 2026-04-08 (live portfolio reset date).
    PYRAMID-reason rows are excluded by default — they aren't ML-gated by
    design, so including them would inflate the "missing ml_confidence"
    count and confuse the WR/entry-rate stats.
    """
    cutoff = datetime.fromisoformat(cutoff_iso).replace(tzinfo=timezone.utc)
    window_start = datetime.fromisoformat(window_start_iso).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    # Scope to users currently on the requested strategy. Note: AIPortfolioTrade
    # has no historical strategy column — if a user switches strategies later,
    # their old trades would no longer match. For audits at fixed cutoffs this
    # is fine; for long-running historical comparisons, a future schema change
    # would store strategy on the trade row.
    strategy_user_ids = [
        row.user_id for row in
        db.query(AIPortfolioConfig.user_id).filter(AIPortfolioConfig.strategy == strategy).all()
    ]
    if not strategy_user_ids:
        return {
            'strategy': strategy,
            'cutoff': cutoff_iso,
            'window_start': window_start_iso,
            'now': now.isoformat(),
            'error': f"No users currently on strategy '{strategy}'",
        }

    base = db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.executed_at >= window_start,
        AIPortfolioTrade.user_id.in_(strategy_user_ids),
    )
    if exclude_pyramid_reason:
        # Pyramid rows are tagged either action="PYRAMID" or reason="PYRAMID:..."
        # depending on which file recorded them (sync drift documented in the
        # May 5 audit). Exclude both shapes.
        # CRITICAL: NULL reasons must be PRESERVED (~reason.like(...) is NULL
        # for NULL reasons, which SQL treats as FALSE → row excluded). Wrap in
        # an OR with IS NULL to keep them.
        from sqlalchemy import or_
        base = base.filter(
            ~AIPortfolioTrade.action.in_(['PYRAMID']),
            or_(
                AIPortfolioTrade.reason.is_(None),
                ~AIPortfolioTrade.reason.like('PYRAMID:%'),
            ),
        )
    pre = base.filter(AIPortfolioTrade.executed_at < cutoff).order_by(AIPortfolioTrade.executed_at).all()
    post = base.filter(AIPortfolioTrade.executed_at >= cutoff).order_by(AIPortfolioTrade.executed_at).all()

    pre_days = max((cutoff - window_start).days, 1)
    post_days = max((now - cutoff).days, 1)

    return {
        'strategy': strategy,
        'cutoff': cutoff_iso,
        'window_start': window_start_iso,
        'now': now.isoformat(),
        'exclude_pyramid_reason': exclude_pyramid_reason,
        'pre_graduation': _summarize_trades(pre, pre_days),
        'post_graduation': _summarize_trades(post, post_days),
        'baseline_may5': {
            'pre_entry_rate_per_day': 2.43,
            'post_entry_rate_per_day': 0.83,
            'post_realized_sell_wr': 0.7143,
            'post_realized_sell_n': 7,
            'note': 'May 5 read had n=7 SELLs post-grad — too small for confidence. Re-run at 2-3 weeks for n=25-40.',
        },
    }
