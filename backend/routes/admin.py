"""Admin routes: user management (invite-only, Google Sign-In) +
operational diagnostics that are too sensitive for the public API."""

import json
from collections import deque
from datetime import date as date_cls, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel

from backend.database import (
    get_db, User, AIPortfolioTrade, AIPortfolioConfig,
    ShadowStrategy, ShadowTrade, Stock, StockScore, BacktestRun,
)
from backend.auth import get_admin_user, UserCreate, UserResponse
from backend.rate_limiter import limiter
from backend.trading_utils import get_strategy_profile

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


# ---------------------------------------------------------------------------
# Live A/B comparison framework (cutoff-based pre/post windows)
#
# Built for evaluating scoring-rule experiments shipped to live trading. The
# backtest replay path cannot honestly evaluate scoring changes (snapshots
# freeze today's point-in-time scalars), so this is the authoritative harness
# for any live A/B. First consumer: Approach 2 (C-score excellence-tier cap)
# shipped 2026-05-07 as commit ec73f83.
# ---------------------------------------------------------------------------


def _per_trade_returns(trades: list) -> list:
    """Extract realized per-trade pct returns from SELL rows.
    Mirrors the cost_total math in _summarize_trades."""
    out = []
    for t in trades:
        if t.action != 'SELL':
            continue
        if t.realized_gain is None or not t.cost_basis or not t.shares:
            continue
        try:
            cost_total = float(t.cost_basis) * float(t.shares)
            if cost_total > 0:
                out.append(float(t.realized_gain) / cost_total * 100)
        except Exception:
            pass
    return out


def _realized_max_drawdown_pct(trades: list, starting_value: float) -> Optional[float]:
    """Walk SELL realized_gains chronologically, accumulate P&L, find the
    largest peak-to-trough decline as % of (starting_value + peak).

    A 'realized drawdown' — it doesn't see unrealized open-position swings,
    but it's reproducible from the trade table alone and consistent across
    pre/post windows. Returns None if fewer than 2 SELLs (no curve)."""
    sells = [t for t in trades if t.action == 'SELL' and t.realized_gain is not None]
    if len(sells) < 2:
        return None
    sells_sorted = sorted(sells, key=lambda t: t.executed_at)
    cum = 0.0
    peak = 0.0
    max_dd_pct = 0.0
    for t in sells_sorted:
        try:
            cum += float(t.realized_gain)
        except Exception:
            continue
        if cum > peak:
            peak = cum
        denom = starting_value + peak
        if denom > 0:
            dd_pct = ((peak - cum) / denom) * 100
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    return round(max_dd_pct, 2)


def _summarize_window(trades: list, days: int, starting_value: float) -> dict:
    """Window-level stats: trade-level summary (via _summarize_trades) plus
    portfolio-level metrics (total_return_pct, sharpe_per_trade, drawdown).

    starting_value is used to express realized return + drawdown as
    percentages of a reference capital base — typically the strategy's
    starting_cash summed across users. Both windows must use the same
    starting_value for comparison validity."""
    base = _summarize_trades(trades, days)

    per_trade = _per_trade_returns(trades)
    total_realized_gain = 0.0
    total_cost = 0.0
    for t in trades:
        if t.action != 'SELL':
            continue
        if t.realized_gain is None or not t.cost_basis or not t.shares:
            continue
        try:
            cost_total = float(t.cost_basis) * float(t.shares)
            if cost_total > 0:
                total_realized_gain += float(t.realized_gain)
                total_cost += cost_total
        except Exception:
            pass

    # Capital-efficiency return: realized $ / cost basis $ across closed trades.
    # NOT the same as portfolio total return — it ignores cash drag and
    # unrealized P&L — but it's a clean, reproducible signal that responds to
    # scoring-rule changes (which affect the trade-quality distribution).
    capital_efficiency_pct = None
    if total_cost > 0:
        capital_efficiency_pct = round((total_realized_gain / total_cost) * 100, 2)

    # Total return relative to the strategy's reference capital. An EMPTY
    # window must stay None, not 0.0 — a phantom 0% baseline makes every
    # delta look meaningful when the real story is "no trades attributed"
    # (live incident: strategy reassignment emptied the pre window and the
    # UI compared post returns against a fabricated flat baseline).
    total_return_pct = None
    if starting_value > 0 and trades:
        total_return_pct = round((total_realized_gain / starting_value) * 100, 2)

    # Per-trade Sharpe-ish: mean / std of per-trade pct returns. Not
    # annualized — the trades aren't time-uniform, so an annualization factor
    # would be misleading. As a within-framework comparison metric it's fine:
    # both windows compute it the same way, and the delta is what drives the
    # decision.
    sharpe_per_trade = None
    if len(per_trade) >= 2:
        import statistics
        mean_r = statistics.fmean(per_trade)
        std_r = statistics.pstdev(per_trade)
        if std_r > 0:
            sharpe_per_trade = round(mean_r / std_r, 4)

    base['total_realized_gain'] = round(total_realized_gain, 2)
    base['total_cost_basis'] = round(total_cost, 2)
    base['capital_efficiency_pct'] = capital_efficiency_pct
    base['total_return_pct'] = total_return_pct
    base['sharpe_per_trade'] = sharpe_per_trade
    base['realized_max_drawdown_pct'] = _realized_max_drawdown_pct(trades, starting_value)
    base['starting_value'] = round(starting_value, 2)
    return base


def _compute_delta(pre: dict, post: dict) -> dict:
    """Post minus pre on the numeric metrics that drive decisions. Fields
    that are None on either side become None in the delta — the comparator
    will refuse to decide when a required metric is missing."""
    def sub(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 4)

    return {
        'total_return_pct_delta': sub(pre.get('total_return_pct'), post.get('total_return_pct')),
        'capital_efficiency_pct_delta': sub(pre.get('capital_efficiency_pct'), post.get('capital_efficiency_pct')),
        'sharpe_per_trade_delta': sub(pre.get('sharpe_per_trade'), post.get('sharpe_per_trade')),
        'realized_max_drawdown_pct_delta': sub(pre.get('realized_max_drawdown_pct'), post.get('realized_max_drawdown_pct')),
        'entry_rate_per_day_delta': sub(pre.get('entry_rate_per_day'), post.get('entry_rate_per_day')),
        'exit_rate_per_day_delta': sub(pre.get('exit_rate_per_day'), post.get('exit_rate_per_day')),
    }


def _decide(pre: dict, post: dict, delta: dict, criteria: dict) -> dict:
    """Apply the brief's decision rule:
      - keep if return_delta >= min_return_delta_pp AND sharpe_delta >= min_sharpe_delta
      - revert if return_delta < min_return_delta_pp AND sharpe_delta < min_sharpe_delta
      - marginal otherwise (one regressed, the other compensated)
      - insufficient_data if either window has too few SELLs to compute Sharpe
    """
    min_return = criteria.get('min_return_delta_pp', -5.0)
    min_sharpe = criteria.get('min_sharpe_delta', 0.0)
    min_post_sells = criteria.get('min_post_sells', 5)

    pre_sells = (pre.get('realized_sell_pct') or {}).get('n', 0)
    post_sells = (post.get('realized_sell_pct') or {}).get('n', 0)
    if post_sells < min_post_sells:
        return {
            'decision': 'insufficient_data',
            'decision_reason': (
                f'Post window has only {post_sells} closed SELLs; '
                f'minimum {min_post_sells} required for a confident call. '
                f'Re-run when more trades have exited.'
            ),
            'decision_criteria': {
                'min_return_delta_pp': min_return,
                'min_sharpe_delta': min_sharpe,
                'min_post_sells': min_post_sells,
            },
        }

    return_delta = delta.get('total_return_pct_delta')
    sharpe_delta = delta.get('sharpe_per_trade_delta')
    if return_delta is None or sharpe_delta is None:
        return {
            'decision': 'insufficient_data',
            'decision_reason': (
                f'Cannot compute decision — return_delta={return_delta}, '
                f'sharpe_delta={sharpe_delta}. Pre window has '
                f'{pre_sells} SELLs, post has {post_sells}.'
            ),
            'decision_criteria': {
                'min_return_delta_pp': min_return,
                'min_sharpe_delta': min_sharpe,
                'min_post_sells': min_post_sells,
            },
        }

    return_pass = return_delta >= min_return
    sharpe_pass = sharpe_delta >= min_sharpe

    if return_pass and sharpe_pass:
        decision = 'keep'
        reason = (
            f'Return delta {return_delta:+.2f}pp >= {min_return}pp threshold AND '
            f'Sharpe delta {sharpe_delta:+.4f} >= {min_sharpe} threshold. '
            f'Experiment meets both bars — keep the change.'
        )
    elif not return_pass and not sharpe_pass:
        decision = 'revert'
        reason = (
            f'Return delta {return_delta:+.2f}pp < {min_return}pp threshold AND '
            f'Sharpe delta {sharpe_delta:+.4f} < {min_sharpe} threshold. '
            f'Both metrics regressed — revert the change.'
        )
    else:
        decision = 'marginal'
        which_failed = 'return' if not return_pass else 'sharpe'
        which_passed = 'sharpe' if not return_pass else 'return'
        reason = (
            f'{which_failed.capitalize()} regressed but {which_passed} compensated '
            f'(return delta {return_delta:+.2f}pp, sharpe delta {sharpe_delta:+.4f}). '
            f'Default to keep but re-run with a longer post window.'
        )

    return {
        'decision': decision,
        'decision_reason': reason,
        'decision_criteria': {
            'min_return_delta_pp': min_return,
            'min_sharpe_delta': min_sharpe,
            'min_post_sells': min_post_sells,
        },
    }


def _build_warnings(pre_window: dict, post_window: dict, pre: dict, post: dict) -> list:
    """Surface conditions that complicate interpretation. Doesn't fail
    anything — just makes the operator aware."""
    warnings = []
    if post_window['days'] < 21:
        warnings.append(
            f"Post window has only {post_window['days']} days — minimum recommended "
            f"is 21 for a confident read."
        )
    if pre_window['days'] < 21:
        warnings.append(
            f"Pre window has only {pre_window['days']} days — baseline may be noisy."
        )
    pre_sells = (pre.get('realized_sell_pct') or {}).get('n', 0)
    post_sells = (post.get('realized_sell_pct') or {}).get('n', 0)
    if pre_sells < 10:
        warnings.append(f"Pre window has only {pre_sells} closed SELLs — baseline statistics unstable.")
    if post_sells < 10:
        warnings.append(f"Post window has only {post_sells} closed SELLs — post statistics unstable.")
    if abs(pre_window['days'] - post_window['days']) > pre_window['days'] * 0.5:
        warnings.append(
            f"Pre window ({pre_window['days']}d) and post window ({post_window['days']}d) "
            f"differ by >50% — comparison less direct."
        )
    if (pre.get('total_trades') or 0) == 0:
        warnings.append(
            "Pre window has ZERO attributed trades — the baseline is not "
            "comparable. Likely cause: strategy membership changed after the "
            "cutoff (attribution for legacy trades follows the users' CURRENT "
            "strategy). Trades stamped after Jul 2026 carry their "
            "execution-time strategy and are immune to this."
        )
    return warnings


def _build_trade_query(db: Session, source: str, strategy: str):
    """Resolve the trade-source for an A/B comparison.

    Returns (base_query, starting_value, user_ids). For source='live' the
    query targets AIPortfolioTrade scoped to users on `strategy`; for
    source='shadow' it targets ShadowTrade scoped to the named ShadowStrategy.
    Raises HTTPException(404) if the requested source is empty.

    Both branches return a query that the rest of _resolve_ab_window can
    extend with date filters and the optional pyramid exclusion. Trade rows
    on either side share the duck-type contract used by _summarize_window
    and _serialize_trade_row (.action, .realized_gain, .executed_at,
    .cost_basis, .shares, etc.) — see ShadowTrade in backend/database.py."""
    if source == "shadow":
        ss = db.query(ShadowStrategy).filter(ShadowStrategy.name == strategy).first()
        if ss is None:
            raise HTTPException(
                status_code=404,
                detail=f"Shadow strategy '{strategy}' not found",
            )
        if ss.archived_at is not None:
            raise HTTPException(
                status_code=404,
                detail=f"Shadow strategy '{strategy}' is archived",
            )
        base = db.query(ShadowTrade).filter(ShadowTrade.shadow_strategy_id == ss.id)
        return base, float(ss.starting_value or 25000.0), [ss.id]

    # source == 'live' (default)
    user_rows = db.query(AIPortfolioConfig.user_id, AIPortfolioConfig.starting_cash).filter(
        AIPortfolioConfig.strategy == strategy
    ).all()
    if not user_rows:
        raise HTTPException(status_code=404, detail=f"No users currently on strategy '{strategy}'")

    user_ids = [row.user_id for row in user_rows]
    starting_value = sum((row.starting_cash or 25000.0) for row in user_rows)
    # Attribution: prefer the strategy STAMPED AT EXECUTION (Jul 2026 —
    # immune to later strategy switches); legacy NULL-stamped rows fall back
    # to current-membership scoping, which is the old (drifting) behavior.
    from sqlalchemy import or_, and_
    base = db.query(AIPortfolioTrade).filter(or_(
        AIPortfolioTrade.strategy == strategy,
        and_(AIPortfolioTrade.strategy.is_(None),
             AIPortfolioTrade.user_id.in_(user_ids)),
    ))
    return base, starting_value, user_ids


def _resolve_ab_window(
    strategy: str,
    cutoff_date: str,
    pre_window_days: int,
    post_window_days: Optional[int],
    exclude_pyramids: bool,
    db: Session,
    source: str = "live",
) -> dict:
    """Shared window/user/trade resolution for the live A/B endpoints.

    Returns a dict with the parsed cutoff, pre/post trade lists, window
    descriptors, starting_value, and user_ids (or [shadow_strategy_id] when
    source='shadow'). Both /strategy-ab-eval and /strategy-ab-eval-trades
    call this; keeping it shared prevents drift between the aggregate
    dashboard and the per-trade attribution view.

    The trade-source is selected by `source`: 'live' (default) reads
    AIPortfolioTrade rows for users on the strategy; 'shadow' reads
    ShadowTrade rows for the named ShadowStrategy. The downstream
    summarizer/serializer code paths are unchanged — both row types
    expose the same duck-typed columns.

    Raises HTTPException(400/404) on invalid inputs — both endpoints rely
    on identical error wording, which is locked by the existing tests."""
    try:
        cutoff = datetime.fromisoformat(cutoff_date).replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"cutoff_date '{cutoff_date}' is not a valid ISO date")

    now = datetime.now(timezone.utc)
    if cutoff > now:
        raise HTTPException(status_code=400, detail=f"cutoff_date {cutoff_date} is in the future")

    days_since_cutoff = max((now - cutoff).days, 1)
    if post_window_days is None:
        post_window_days = min(days_since_cutoff, 90)
    elif post_window_days > days_since_cutoff:
        raise HTTPException(
            status_code=400,
            detail=f"post_window_days={post_window_days} exceeds days-since-cutoff={days_since_cutoff}",
        )

    pre_start = cutoff - timedelta(days=pre_window_days)
    post_end = cutoff + timedelta(days=post_window_days)

    base, starting_value, user_ids = _build_trade_query(db, source, strategy)

    # Pick the trade model (AIPortfolioTrade or ShadowTrade) the helper
    # already filtered against — both expose .executed_at, .action, .reason.
    trade_model = ShadowTrade if source == "shadow" else AIPortfolioTrade

    base = base.filter(
        trade_model.executed_at >= pre_start,
        trade_model.executed_at < post_end,
    )
    if exclude_pyramids:
        # Same shape as strategy-health: filter both action='PYRAMID' rows
        # AND action='BUY' + reason LIKE 'PYRAMID:%' rows. NULL reasons MUST
        # be preserved — (~col.like(...)) is NULL for NULL → SQL false, so
        # wrap in OR with IS NULL.
        base = base.filter(
            ~trade_model.action.in_(['PYRAMID']),
            or_(
                trade_model.reason.is_(None),
                ~trade_model.reason.like('PYRAMID:%'),
            ),
        )

    pre_trades = base.filter(trade_model.executed_at < cutoff).order_by(trade_model.executed_at).all()
    post_trades = base.filter(trade_model.executed_at >= cutoff).order_by(trade_model.executed_at).all()

    pre_window = {
        'start': pre_start.date().isoformat(),
        'end': cutoff.date().isoformat(),
        'days': pre_window_days,
    }
    post_window = {
        'start': cutoff.date().isoformat(),
        'end': post_end.date().isoformat(),
        'days': post_window_days,
    }

    return {
        'cutoff': cutoff,
        'pre_start': pre_start,
        'post_end': post_end,
        'pre_window': pre_window,
        'post_window': post_window,
        'pre_window_days': pre_window_days,
        'post_window_days': post_window_days,
        'pre_trades': pre_trades,
        'post_trades': post_trades,
        'starting_value': starting_value,
        'user_ids': user_ids,
        'source': source,
    }


@router.get("/strategy-ab-eval")
@limiter.limit("30/minute")
async def run_strategy_ab_comparison(
    strategy: str = Query(..., description="Strategy profile name, e.g. 'nostate_optimized'"),
    cutoff_date: str = Query(..., description="ISO date marking experiment start, e.g. '2026-05-07'"),
    pre_window_days: int = Query(default=30, ge=1, le=365, description="Days BEFORE cutoff for baseline"),
    post_window_days: Optional[int] = Query(default=None, description="Days AFTER cutoff to evaluate; defaults to days-since-cutoff capped at 90"),
    exclude_pyramids: bool = Query(default=True, description="Filter out pyramid rows (action='PYRAMID' OR reason LIKE 'PYRAMID:%')"),
    min_return_delta_pp: float = Query(default=-5.0, description="Decision threshold: keep if return delta >= this"),
    min_sharpe_delta: float = Query(default=0.0, description="Decision threshold: keep if sharpe delta >= this"),
    min_post_sells: int = Query(default=5, ge=0, description="Minimum SELL count in post window for a confident decision"),
    source: str = Query(default="live", description="Trade source: 'live' (AIPortfolioTrade scoped to users on the strategy) or 'shadow' (ShadowTrade scoped to the named ShadowStrategy)"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Live A/B comparison: pre vs post-cutoff trade summary + decision.

    Built for assessing scoring-rule experiments shipped to live trading.
    The backtest replay path cannot honestly assess scoring changes
    (snapshot scalars are frozen at today's values), so this endpoint is
    the authoritative harness for any live A/B.

    First consumer: Approach 2 (C-score excellence-tier cap, commit ec73f83
    deployed 2026-05-07).

    Example invocation:
        GET /api/admin/strategy-ab-eval?strategy=nostate_optimized
            &cutoff_date=2026-05-07&pre_window_days=30&post_window_days=14
    """
    # When tests bypass FastAPI's dependency resolution, Query() defaults
    # arrive as Query objects rather than their underlying values — same
    # pattern documented in test_strategy_ab_eval._call. Coerce here so the
    # endpoint behaves identically whether called via FastAPI or directly.
    if not isinstance(source, str):
        source = "live"

    win = _resolve_ab_window(
        strategy, cutoff_date, pre_window_days, post_window_days,
        exclude_pyramids, db, source=source,
    )
    pre_trades = win['pre_trades']
    post_trades = win['post_trades']
    starting_value = win['starting_value']
    pre_window = win['pre_window']
    post_window = win['post_window']

    pre_summary = _summarize_window(pre_trades, win['pre_window_days'], starting_value)
    post_summary = _summarize_window(post_trades, win['post_window_days'], starting_value)
    delta = _compute_delta(pre_summary, post_summary)

    criteria = {
        'min_return_delta_pp': min_return_delta_pp,
        'min_sharpe_delta': min_sharpe_delta,
        'min_post_sells': min_post_sells,
    }
    decision = _decide(pre_summary, post_summary, delta, criteria)

    experiment = {
        'strategy': strategy,
        'cutoff_date': cutoff_date,
        'pre_window': pre_window,
        'post_window': post_window,
        'starting_value_reference': round(starting_value, 2),
        'user_ids': win['user_ids'],
        'exclude_pyramids': exclude_pyramids,
    }
    if source != "live":
        experiment['source'] = source

    return {
        'experiment': experiment,
        'summary': decision,
        'pre': pre_summary,
        'post': post_summary,
        'delta': delta,
        'warnings': _build_warnings(pre_window, post_window, pre_summary, post_summary),
    }


def _trade_scope_id(t):
    """Scope ID that disambiguates "the same ticker held by a different
    account" in prior-BUY maps: user_id for AIPortfolioTrade rows,
    shadow_strategy_id for ShadowTrade rows (which have no user_id — shadow
    trades belong to a strategy, not a user). Shared by the dashboard
    endpoint and ab_eval_email so the two paths can't drift."""
    sid = getattr(t, 'user_id', None)
    return sid if sid is not None else getattr(t, 'shadow_strategy_id', None)


def _serialize_trade_row(t, prior_buys_by_ticker_user: dict) -> dict:
    """Shape one AIPortfolioTrade row for the per-trade endpoint.

    realized_pct uses the same cost_total = cost_basis * shares formula as
    _per_trade_returns; null when cost_basis is missing.

    hold_days for SELLs prefers the persisted holding_days column (set by
    ai_trader on close), falling back to a (executed_at - prior_buy.executed_at)
    diff if a prior BUY for the same (ticker, user) is in our window.
    None on BUY/PYRAMID rows by design — there's nothing to hold yet."""
    realized_pct = None
    if t.action == 'SELL' and t.realized_gain is not None and t.cost_basis and t.shares:
        # Defensive try/except, practically unreachable in isolation: the
        # return dict at lines 795-799 calls float() unguarded on the
        # same fields (shares/price/cost_basis/realized_gain), so any
        # trade row crafted to trigger an exception here also crashes
        # the unguarded float()s downstream. The except branch covers
        # us if a future refactor guards those return-dict float() calls.
        # See canslim-may10-routes-admin-coverage-shipped.md.
        try:
            cost_total = float(t.cost_basis) * float(t.shares)
            if cost_total > 0:
                realized_pct = round(float(t.realized_gain) / cost_total * 100, 2)
        except Exception:
            pass

    scope_id = _trade_scope_id(t)

    hold_days = None
    if t.action == 'SELL':
        if t.holding_days is not None:
            hold_days = int(t.holding_days)
        else:
            key = (t.ticker, scope_id)
            prior = prior_buys_by_ticker_user.get(key)
            if prior is not None and t.executed_at and prior.executed_at:
                hold_days = max((t.executed_at - prior.executed_at).days, 0)

    return {
        'id': t.id,
        'ticker': t.ticker,
        'action': t.action,
        'executed_at': t.executed_at.isoformat() if t.executed_at else None,
        'shares': float(t.shares) if t.shares is not None else None,
        'price': float(t.price) if t.price is not None else None,
        'cost_basis': float(t.cost_basis) if t.cost_basis is not None else None,
        'realized_gain': float(t.realized_gain) if t.realized_gain is not None else None,
        'realized_pct': realized_pct,
        'reason': t.reason,
        'hold_days': hold_days,
        'user_id': scope_id,
    }


def _build_cumulative_curve(trades: list, anchor_date, end_date, starting_value: float) -> list:
    """Build a stepped cumulative-return curve over [anchor_date, end_date].

    Walks SELL rows chronologically and accumulates realized_gain /
    starting_value * 100. Anchored at 0% on anchor_date and emits a row
    each time cum_pct changes plus one terminal row at end_date so the
    chart line extends across the full window even after the final SELL.

    BUY/PYRAMID rows do not affect the curve — only realized P&L moves it."""
    out = [{'date': anchor_date.date().isoformat(), 'cum_pct': 0.0}]
    if starting_value <= 0:
        out.append({'date': end_date.date().isoformat(), 'cum_pct': 0.0})
        return out

    sells = sorted(
        [t for t in trades if t.action == 'SELL' and t.realized_gain is not None],
        key=lambda x: x.executed_at,
    )
    cum_gain = 0.0
    for t in sells:
        try:
            cum_gain += float(t.realized_gain)
        except Exception:
            continue
        cum_pct = round((cum_gain / starting_value) * 100, 4)
        date_iso = t.executed_at.date().isoformat() if t.executed_at else anchor_date.date().isoformat()
        # If multiple SELLs land on the same day, overwrite the prior same-day
        # entry rather than emitting duplicates — chart libs render either
        # interpretation but a single row per day is cleaner.
        if out and out[-1]['date'] == date_iso:
            out[-1]['cum_pct'] = cum_pct
        else:
            out.append({'date': date_iso, 'cum_pct': cum_pct})

    terminal_iso = end_date.date().isoformat()
    if out[-1]['date'] != terminal_iso:
        out.append({'date': terminal_iso, 'cum_pct': out[-1]['cum_pct']})
    return out


@router.get("/strategy-ab-eval-trades")
@limiter.limit("30/minute")
async def run_strategy_ab_trades(
    strategy: str = Query(..., description="Strategy profile name, e.g. 'nostate_optimized'"),
    cutoff_date: str = Query(..., description="ISO date marking experiment start"),
    pre_window_days: int = Query(default=30, ge=1, le=365),
    post_window_days: Optional[int] = Query(default=None),
    exclude_pyramids: bool = Query(default=True),
    source: str = Query(default="live", description="Trade source: 'live' or 'shadow'"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Per-trade attribution sibling of /strategy-ab-eval.

    Same window math, same user scope, same pyramid filter — but instead of
    aggregate stats it returns the row list and a cumulative-return curve
    per window. Built for the ABEval dashboard chart + collapsible trade
    table: when the dashboard verdict is 'marginal', per-trade visibility
    turns "the average drifted" into "this specific trade moved it"."""
    # See coercion note in run_strategy_ab_comparison — same Query() default
    # passthrough behavior when tests bypass FastAPI dependency resolution.
    if not isinstance(source, str):
        source = "live"

    win = _resolve_ab_window(
        strategy, cutoff_date, pre_window_days, post_window_days,
        exclude_pyramids, db, source=source,
    )
    pre_trades = win['pre_trades']
    post_trades = win['post_trades']
    starting_value = win['starting_value']
    cutoff = win['cutoff']
    pre_start = win['pre_start']
    post_end = win['post_end']

    # Build a (ticker, scope_id) → most-recent-prior-BUY map for hold_days
    # fallback. Walk ALL trades chronologically (pre + post) so a SELL in
    # post can match a BUY that landed in pre.
    _scope_id = _trade_scope_id

    prior_buy_map: dict = {}
    serialized_pre: list = []
    serialized_post: list = []
    for t in pre_trades:
        serialized_pre.append(_serialize_trade_row(t, prior_buy_map))
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _scope_id(t))] = t
    for t in post_trades:
        serialized_post.append(_serialize_trade_row(t, prior_buy_map))
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _scope_id(t))] = t

    pre_curve = _build_cumulative_curve(pre_trades, pre_start, cutoff, starting_value)
    post_curve = _build_cumulative_curve(post_trades, cutoff, post_end, starting_value)

    experiment = {
        'strategy': strategy,
        'cutoff_date': cutoff_date,
        'pre_window': win['pre_window'],
        'post_window': win['post_window'],
        'starting_value_reference': round(starting_value, 2),
        'user_ids': win['user_ids'],
        'exclude_pyramids': exclude_pyramids,
    }
    if source != "live":
        experiment['source'] = source

    return {
        'experiment': experiment,
        'pre_trades': serialized_pre,
        'post_trades': serialized_post,
        'cumulative_returns': {
            'pre': pre_curve,
            'post': post_curve,
        },
    }


class ABEvalEmailTestRequest(BaseModel):
    strategy: str
    cutoff_date: str
    recipient_override: Optional[str] = None
    pre_window_days: int = 30
    post_window_days: Optional[int] = None
    exclude_pyramids: bool = True
    source: str = "live"


@router.post("/strategy-ab-eval/email-test")
async def trigger_ab_eval_snapshot_email(
    payload: ABEvalEmailTestRequest,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Trigger an immediate A/B-eval snapshot email — same body as the weekly
    cron, sent on demand. Useful for verifying the wiring before the first
    Monday run, and for the 2026-06-18 real-eval routine which now triggers
    this endpoint instead of logging the JSON.

    `source` selects live vs shadow delivery. The shadow path is the same
    fan-out the Mon 9 AM UTC cron uses for active ShadowStrategy rows —
    operators can preview a single shadow snapshot before the next Monday.

    Window-validation errors (HTTPException 400/404) propagate from
    _resolve_ab_window verbatim — same wording as /strategy-ab-eval, so the
    UI can surface a single error message regardless of which path failed."""
    from backend.ab_eval_email import send_ab_eval_snapshot
    result = send_ab_eval_snapshot(
        payload.strategy,
        payload.cutoff_date,
        db,
        recipient=payload.recipient_override,
        pre_window_days=payload.pre_window_days,
        post_window_days=payload.post_window_days,
        exclude_pyramids=payload.exclude_pyramids,
        source=payload.source,
    )
    return {
        'sent': result['sent'],
        'recipient': result['recipient'],
        'subject': result['subject'],
        'decision': result['decision'],
        'post_sell_count': result['post_sell_count'],
        'source': result['source'],
    }


# ---------------------------------------------------------------------------
# Cap-Delta Diagnostics
#
# Population-level read of c_score vs c_score_uncapped over a rolling window.
# Closes the Scenario C.3 blind spot from the Shadow Step 7 verification:
# a shadow_no_excellence_cap arm that emits byte-identical trades to the
# baseline because today's buy-eligible universe happens to be all 4-of-4
# excellence stocks (cap doesn't bite) is invisible from trade counts alone.
# This endpoint surfaces the population view so verdict-day reads (Approach 2
# real eval ending 2026-06-18, and any future shadow stack) are
# pre-instrumented.
#
# 100% read-only: queries stock_scores + stocks, no scorer invocation, no
# scheduler hooks, no writes.
# ---------------------------------------------------------------------------

# Iteration buckets for the divergent population (delta >= min_delta). The
# "0" bucket is filled via the early-continue paths in the helper — it
# represents "row was in the window but was not divergent" (uncapped null,
# or delta below min_delta). Keeping it out of this tuple prevents the
# matching loop from accidentally claiming divergent rows for "0".
_DIVERGENT_BUCKETS = (
    ("0.5-1.5", 0.0, 1.5),
    ("1.5-3.0", 1.5, 3.0),
    ("3.0+",    3.0, None),
)
_BUCKET_LABELS = ("0", "0.5-1.5", "1.5-3.0", "3.0+")
_TOP_DIVERGENT_LIMIT = 25


def compute_cap_delta_diagnostics(
    db: Session,
    days: int,
    min_delta: float,
    buy_threshold: float,
) -> dict:
    """Population-level cap-delta diagnostics over a rolling stock_scores window.

    Pure function over a SQLAlchemy Session — endpoint is a thin wrapper.
    Reads StockScore.timestamp + Stock.ticker via a single join; no scorer
    code paths invoked.

    `delta` is `c_score_uncapped - c_score`. Because c_score_uncapped is the
    pre-excellence-cap C subscore and the rest of the canslim breakdown is
    untouched, delta on the C subscore equals delta on the total score.
    Threshold-crossing math therefore compares total_score vs total_score+delta
    against the buy_threshold.

    Returns the response shape consumed by GET /api/admin/cap-delta-diagnostics.
    Returns 200-shape with zeroed fields on empty windows; the endpoint never
    404s on no data.
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days)

    rows = (
        db.query(
            StockScore.id,
            StockScore.stock_id,
            StockScore.timestamp,
            StockScore.total_score,
            StockScore.c_score,
            StockScore.c_score_uncapped,
            Stock.ticker,
        )
        .join(Stock, Stock.id == StockScore.stock_id)
        .filter(
            StockScore.timestamp >= start_utc,
            StockScore.timestamp <= now_utc,
        )
        .all()
    )

    rows_total = len(rows)
    rows_with_uncapped = sum(1 for r in rows if r.c_score_uncapped is not None)
    distinct_stocks = len({r.stock_id for r in rows})

    bucket_counts = {label: 0 for label in _BUCKET_LABELS}
    deltas_present = []
    rows_with_delta = 0

    cross_uncapped_above = 0
    cross_capped_above = 0

    per_ticker = {}

    for r in rows:
        if r.c_score is None or r.c_score_uncapped is None:
            bucket_counts["0"] += 1
            continue
        delta = r.c_score_uncapped - r.c_score
        # Floating noise / under-min_delta rows count toward the zero-bucket.
        if delta < min_delta:
            bucket_counts["0"] += 1
            continue

        rows_with_delta += 1
        deltas_present.append(delta)
        # Half-open buckets [lo, hi); "3.0+" is unbounded above.
        for label, lo, hi in _DIVERGENT_BUCKETS:
            if hi is None:
                if delta >= lo:
                    bucket_counts[label] += 1
                    break
            elif lo <= delta < hi:
                bucket_counts[label] += 1
                break

        # Threshold-crossing: total_score is the persisted total. uncapped
        # total = total_score + delta (other subscores untouched by the cap).
        total = r.total_score
        if total is not None:
            uncapped_total = total + delta
            if total < buy_threshold <= uncapped_total:
                cross_uncapped_above += 1
            # Defensive — should never happen since uncapped >= capped, but
            # surface it if a data anomaly creates one.
            if uncapped_total < buy_threshold <= total:
                cross_capped_above += 1

        slot = per_ticker.setdefault(
            r.ticker,
            {
                "n_rows": 0,
                "max_delta": 0.0,
                "max_uncapped_score": None,
                "max_capped_score": None,
                "crossed_buy_threshold_at_least_once": False,
            },
        )
        slot["n_rows"] += 1
        if delta > slot["max_delta"]:
            slot["max_delta"] = delta
            if total is not None:
                slot["max_capped_score"] = round(total, 2)
                slot["max_uncapped_score"] = round(total + delta, 2)
        if total is not None and total < buy_threshold <= total + delta:
            slot["crossed_buy_threshold_at_least_once"] = True

    avg_delta = round(sum(deltas_present) / len(deltas_present), 2) if deltas_present else 0.0
    max_delta = round(max(deltas_present), 2) if deltas_present else 0.0

    delta_buckets = []
    if rows_total > 0:
        for label in _BUCKET_LABELS:
            n = bucket_counts[label]
            delta_buckets.append({
                "bucket": label,
                "n": n,
                "pct": round(n / rows_total * 100, 1),
            })

    rows_with_delta_pct = round(rows_with_delta / rows_total * 100, 1) if rows_total else 0.0

    # Stable ordering: max_delta desc, then ticker asc (idempotency for callers).
    top_tickers = sorted(
        (
            {"ticker": tkr, **stats, "max_delta": round(stats["max_delta"], 2)}
            for tkr, stats in per_ticker.items()
        ),
        key=lambda x: (-x["max_delta"], x["ticker"]),
    )[:_TOP_DIVERGENT_LIMIT]

    return {
        "window": {
            "start_utc": start_utc.isoformat().replace("+00:00", "Z"),
            "end_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "days": days,
            "n_score_writes": rows_total,
            "n_distinct_stocks": distinct_stocks,
        },
        "population": {
            "rows_total": rows_total,
            "rows_with_uncapped": rows_with_uncapped,
            "rows_with_delta": rows_with_delta,
            "rows_with_delta_pct": rows_with_delta_pct,
            "delta_buckets": delta_buckets,
            "avg_delta_when_present": avg_delta,
            "max_delta": max_delta,
        },
        "threshold_crossings": {
            "buy_threshold": round(buy_threshold, 2),
            "rows_baseline_below_threshold_uncapped_above": cross_uncapped_above,
            "rows_baseline_above_threshold_uncapped_below": cross_capped_above,
        },
        "top_divergent_tickers": top_tickers,
    }


@router.get("/cap-delta-diagnostics")
@limiter.limit("30/minute")
async def cap_delta_diagnostics_endpoint(
    days: int = Query(default=1, ge=1, le=14, description="Rolling window size in days, from now-UTC"),
    min_delta: float = Query(default=0.5, ge=0.0, le=15.0, description="Rows with c_score_uncapped - c_score below this don't count as divergent"),
    strategy: str = Query(default="nostate_cs_bear", description="Strategy profile to source the buy threshold from (default = current live profile)"),
    buy_threshold: Optional[float] = Query(default=None, ge=0.0, le=100.0, description="Override the threshold. If None, sourced from strategy_profiles.<strategy>.min_score"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Cap-Delta Diagnostics — population view of c_score vs c_score_uncapped.

    Built to close the Scenario C.3 blind spot identified in the Shadow Step 7
    verification: a shadow_no_excellence_cap arm that silently emits trades
    byte-identical to the baseline because today's universe happens to be
    all 4-of-4 excellence stocks. Trade counts can't tell you that; this can.

    Eval-safe: 100% read-only against stock_scores. No scorer invocation, no
    scheduler hooks, no writes. Same admin auth + rate limits as
    /strategy-ab-eval.

    Example:
        GET /api/admin/cap-delta-diagnostics?days=7&min_delta=0.5
    """
    # Bypass for direct (non-FastAPI) invocation — see _call pattern in
    # tests/test_strategy_ab_eval.py. Query() defaults arrive as Query objects
    # outside FastAPI's resolver, which breaks the numeric comparisons below.
    if not isinstance(days, int):
        days = 1
    if not isinstance(min_delta, (int, float)):
        min_delta = 0.5
    if not isinstance(strategy, str):
        strategy = "nostate_cs_bear"
    if buy_threshold is not None and not isinstance(buy_threshold, (int, float)):
        buy_threshold = None

    if buy_threshold is None:
        profile = get_strategy_profile(strategy)
        buy_threshold = float(profile.get("min_score", 72))

    return compute_cap_delta_diagnostics(
        db=db,
        days=int(days),
        min_delta=float(min_delta),
        buy_threshold=float(buy_threshold),
    )


# ---------------------------------------------------------------------------
# Shadow paper-trading: listing route
#
# Step 3 of the design in docs/shadow-paper-trading-design.md. Lets the
# operator (and the future ABEval frontend dropdown in Step 5) discover
# which candidate scoring stacks are running alongside the live scanner.
# ---------------------------------------------------------------------------


@router.get("/shadow-strategies")
async def list_shadow_strategies(
    archived: bool = Query(default=False, description="Include archived shadow strategies"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """List shadow strategies.

    By default returns only active (non-archived) rows. Pass ?archived=true to
    include archived stacks — useful for backfilling the ABEval dashboard
    when re-running historical reads.

    Powers the ABEval frontend dropdown shipping in Step 5; until then this
    is the operator's primary surface for confirming a registered stack is
    still active before invoking /strategy-ab-eval?source=shadow."""
    q = db.query(ShadowStrategy)
    if not archived:
        q = q.filter(ShadowStrategy.archived_at.is_(None))
    rows = q.order_by(ShadowStrategy.id).all()
    return [
        {
            'id': r.id,
            'name': r.name,
            'parent_strategy': r.parent_strategy,
            'scorer_overrides': r.scorer_overrides,
            'starting_value': float(r.starting_value) if r.starting_value is not None else None,
            'description': r.description,
            'created_at': r.created_at.isoformat() if r.created_at else None,
            'activated_at': r.activated_at.isoformat() if r.activated_at else None,
            'archived_at': r.archived_at.isoformat() if r.archived_at else None,
        }
        for r in rows
    ]


@router.get("/backtest-pairs")
async def get_backtest_pairs(
    strategy: Optional[str] = Query(None, description="Filter to one strategy. Omit for all."),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Return the set of (start_date, end_date) pairs in the completed-backtest
    pool, optionally filtered by strategy.

    Used by scripts/grow_training_pool.py to skip date ranges already in the
    pool — the trainer dedups by (start_date, end_date), so re-running a
    duplicate yields zero new training samples.
    """
    q = db.query(BacktestRun.start_date, BacktestRun.end_date, BacktestRun.strategy)\
          .filter(BacktestRun.status == "completed")
    if strategy:
        q = q.filter(BacktestRun.strategy == strategy)
    rows = q.distinct().all()
    return {
        "pairs": [
            {
                "start_date": r.start_date.isoformat() if r.start_date else None,
                "end_date": r.end_date.isoformat() if r.end_date else None,
                "strategy": r.strategy,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(100, ge=1, le=500),
    kind: Optional[str] = Query(None, pattern="^(rate_limit|auth_fail)$"),
    current_user: User = Depends(get_admin_user),
):
    """Recent security-relevant rejections (rate-limit hits + auth failures).

    In-memory ring buffer; resets on container restart. Use to verify that
    CSO #4 (slowapi limits) and Google auth are catching real attempts.
    No PII is recorded — only failure reasons (e.g. "invalid_token").
    """
    from backend.audit_log import recent_events, buffer_size
    events = recent_events(limit=limit)
    if kind:
        events = [e for e in events if e["kind"] == kind]
    return {
        "events": events,
        "total": len(events),
        "buffer_size": buffer_size(),
    }


# ---------------------------------------------------------------------------
# ML-demotion cohort report
#
# The 2026-07-04 demotion (283b33c) turned the ml_confidence veto off but
# kept logging the confidence on every BUY. The planned "entry-rate rises
# if the veto was filtering" re-check turned out to be doubly confounded
# (all users pinned at max_positions with cash under one position size, and
# the pre-period had the veto active) — measured live 2026-07-13, where
# 9 of the 10 post-demotion buys carried confidence under the old 0.30
# threshold. This endpoint gives the direct causal read instead: split
# post-demotion buys into would-have-been-vetoed vs would-have-passed
# cohorts and compare their outcomes. If the vetoed cohort performs no
# worse, the demotion is vindicated; if it clearly underperforms, the model
# had edge and re-graduation (via LIVE A/B only) is worth revisiting.
# ---------------------------------------------------------------------------


@router.get("/ml-demotion-cohort")
@limiter.limit("30/minute")
async def ml_demotion_cohort(
    cutoff_date: str = Query(default="2026-07-04", description="Demotion date — BUYs after this are cohorted"),
    threshold: float = Query(default=0.30, ge=0.0, le=1.0, description="The old veto threshold: conf below this = would-have-been-vetoed"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Would-the-veto-have-blocked-it cohort comparison. 100% read-only."""
    # Bypass for direct (non-FastAPI) invocation — Query() defaults arrive
    # as Query objects outside the resolver (see cap-delta-diagnostics).
    if not isinstance(cutoff_date, str):
        cutoff_date = "2026-07-04"
    if not isinstance(threshold, (int, float)):
        threshold = 0.30

    try:
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail="cutoff_date must be YYYY-MM-DD")

    from backend.database import AIPortfolioPosition

    buys = db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.action == 'BUY',
        AIPortfolioTrade.executed_at > cutoff,
    ).order_by(AIPortfolioTrade.executed_at).all()

    now_utc = datetime.now(timezone.utc)

    # Per-lot FIFO outcomes, built lazily per (ticker, user). The old logic
    # summed ALL later sells' realized_gain over ONE buy's cost, and called
    # any buy 'open' whenever ANY position row existed — both wrong once a
    # ticker is re-bought. Live 2026-08-20: ANET-u3's $453 first buy wore
    # the $268 loss of the later 18-share generation and reported -59.1%
    # for a lot that actually closed +4.1%.
    _lot_cache: dict = {}

    def _lot_outcome(buy_trade):
        key = (buy_trade.ticker, buy_trade.user_id)
        if key not in _lot_cache:
            history = db.query(AIPortfolioTrade).filter(
                AIPortfolioTrade.ticker == buy_trade.ticker,
                AIPortfolioTrade.user_id == buy_trade.user_id,
                AIPortfolioTrade.action.in_(('BUY', 'PYRAMID', 'SELL')),
            ).order_by(AIPortfolioTrade.executed_at, AIPortfolioTrade.id).all()
            lots = {}
            open_lots = deque()
            for h in history:
                if h.action in ('BUY', 'PYRAMID'):
                    lot = {'shares': h.shares or 0.0,
                           'shares_left': h.shares or 0.0,
                           'price': h.price or 0.0,
                           'realized': 0.0}
                    lots[h.id] = lot
                    if lot['shares_left'] > 1e-9:
                        open_lots.append(lot)
                else:
                    to_sell = h.shares or 0.0
                    while to_sell > 1e-9 and open_lots:
                        head = open_lots[0]
                        take = min(head['shares_left'], to_sell)
                        head['realized'] += ((h.price or 0.0) - head['price']) * take
                        head['shares_left'] -= take
                        to_sell -= take
                        if head['shares_left'] <= 1e-9:
                            open_lots.popleft()
            _lot_cache[key] = lots
        return _lot_cache[key].get(buy_trade.id)

    rows = []
    for t in buys:
        # Legacy rows store signal_factors as a JSON string — decode like
        # _summarize_trades does.
        sf = t.signal_factors
        if isinstance(sf, str):
            try:
                sf = json.loads(sf)
            except Exception:
                sf = None
        conf = sf.get('ml_confidence') if isinstance(sf, dict) else None
        if conf is None:
            continue
        lot = _lot_outcome(t)
        cost = (lot['price'] * lot['shares']) if lot else 0.0
        if lot is not None and lot['shares_left'] <= 1e-9 and cost > 0:
            # This buy's lot is fully consumed — its OWN realized outcome.
            status = 'closed'
            gain_pct = round(lot['realized'] / cost * 100, 2)
        else:
            # Lot (partially) outstanding: mark to market via the live
            # position row; no row = data gap, don't guess.
            pos = db.query(AIPortfolioPosition).filter(
                AIPortfolioPosition.ticker == t.ticker,
                AIPortfolioPosition.user_id == t.user_id,
            ).first()
            if pos is not None:
                status, gain_pct = 'open', pos.gain_loss_pct
            else:
                status, gain_pct = 'unknown', None
        age_days = None
        if t.executed_at is not None:
            exec_at = t.executed_at
            if exec_at.tzinfo is None:
                exec_at = exec_at.replace(tzinfo=timezone.utc)
            age_days = max((now_utc - exec_at).days, 0)
        rows.append({
            'ticker': t.ticker,
            'user_id': t.user_id,
            'executed_at': t.executed_at.isoformat() if t.executed_at else None,
            'age_days': age_days,
            'ml_confidence': conf,
            'cohort': 'would_veto' if conf < threshold else 'passed',
            'status': status,
            'gain_pct': gain_pct,
        })

    def _agg(cohort_rows):
        # avg_age_days is computed over the SAME rows that feed avg_gain_pct:
        # a mark-to-market gain grows with holding time, so cohorts of
        # different vintages (Jul-6 fill vs Jul-16 buys) aren't comparable
        # without knowing each side's age.
        scored = [r for r in cohort_rows if r['gain_pct'] is not None]
        gains = [r['gain_pct'] for r in scored]
        ages = [r['age_days'] for r in scored if r['age_days'] is not None]
        return {
            'n': len(cohort_rows),
            'open_n': sum(1 for r in cohort_rows if r['status'] == 'open'),
            'closed_n': sum(1 for r in cohort_rows if r['status'] == 'closed'),
            'avg_gain_pct': round(sum(gains) / len(gains), 2) if gains else None,
            'avg_age_days': round(sum(ages) / len(ages), 1) if ages else None,
            'win_rate': round(
                sum(1 for g in gains if g > 0) / len(gains) * 100, 1) if gains else None,
        }

    vetoed = [r for r in rows if r['cohort'] == 'would_veto']
    passed = [r for r in rows if r['cohort'] == 'passed']
    return {
        'cutoff_date': cutoff_date,
        'threshold': threshold,
        'would_veto': _agg(vetoed),
        'passed': _agg(passed),
        'trades': rows,
        'note': (
            'Gains blend open (mark-to-market) and closed (realized) rows — '
            'read avg_gain_pct as an early directional signal, not a verdict. '
            'Compare avg_age_days between cohorts first: mark-to-market gains '
            'grow with holding time, so a large age gap biases the comparison '
            'toward the older cohort in a trending tape. The demotion is '
            'vindicated if would_veto performs no worse than passed once '
            'closed_n accumulates.'
        ),
    }


def compute_experiment_gates(db: Session) -> dict:
    """Live progress of every pre-registered promotion gate. 100% read-only.

    Pure function over a Session (same seam as compute_cap_delta_diagnostics)
    so the weekly shadow A/B email renders the IDENTICAL clocks the
    /experiment-gates card shows — one computation, two surfaces. The gates
    themselves are pre-registered in config/default.yaml comments (per-arm);
    this just measures accrual, plus the mechanical PASS/FAIL for clocks
    whose pre-registered rule is fully arithmetic (stop-loss re-check).
    Adding an arm: give it an entry in ARM_GATES below.
    """
    from backend.database import MarketSnapshot

    now_utc = datetime.now(timezone.utc)

    def _chop_days_since(start_dt, band_pct=1.5):
        """Trading days where SPY sat 0..band_pct% above its 50MA."""
        if start_dt is None:
            return 0
        snaps = db.query(MarketSnapshot).filter(
            MarketSnapshot.date >= start_dt.date(),
            MarketSnapshot.spy_price.isnot(None),
            MarketSnapshot.spy_50_ma.isnot(None),
        ).all()
        n = 0
        for s in snaps:
            if s.spy_50_ma and s.spy_50_ma > 0:
                gap = (s.spy_price - s.spy_50_ma) / s.spy_50_ma * 100
                if 0 <= gap <= band_pct:
                    n += 1
        return n

    def _sf(t):
        sf = t.signal_factors
        if isinstance(sf, str):
            try:
                sf = json.loads(sf)
            except Exception:
                sf = None
        return sf if isinstance(sf, dict) else {}

    def _naive(dt):
        return dt.replace(tzinfo=None) if (dt is not None and dt.tzinfo) else dt

    arms = db.query(ShadowStrategy).filter(
        ShadowStrategy.archived_at.is_(None)).order_by(ShadowStrategy.id).all()
    trades_by_arm = {}
    for a in arms:
        trades_by_arm[a.id] = db.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == a.id,
        ).order_by(ShadowTrade.executed_at).all()

    def _rows(arm_id, action=None, include_sweeps=False):
        out = []
        for t in trades_by_arm.get(arm_id, []):
            if not include_sweeps and (t.reason or "").startswith("SPY SWEEP"):
                continue
            if action and t.action != action:
                continue
            out.append(t)
        return out

    baseline = next((a for a in arms if a.name == "shadow_baseline"), None)

    def _unmatched_vs_baseline(arm_id, action, reason_prefix=None):
        """Rows in an arm with no same-ticker same-day counterpart in the
        baseline stack — i.e. behavior the arm's lever caused."""
        if baseline is None:
            return 0
        def _key(t):
            return (t.ticker, t.executed_at.date() if t.executed_at else None)
        base_keys = {
            _key(t) for t in _rows(baseline.id, action)
            if reason_prefix is None or (t.reason or "").startswith(reason_prefix)
        }
        n = 0
        for t in _rows(arm_id, action):
            if reason_prefix is not None and not (t.reason or "").startswith(reason_prefix):
                continue
            if _key(t) not in base_keys:
                n += 1
        return n

    def _suppressed_vs_baseline(arm_id, action, reason_prefix):
        """Baseline rows with no counterpart in the arm — behavior the
        arm's lever SUPPRESSED (e.g. pre-earnings exits it skipped).
        Only baseline rows inside the arm's own active window count —
        the baseline stack can predate a later-activated arm."""
        if baseline is None:
            return 0
        arm_obj = next((x for x in arms if x.id == arm_id), None)
        activated = _naive(arm_obj.activated_at) if arm_obj else None
        def _key(t):
            return (t.ticker, t.executed_at.date() if t.executed_at else None)
        arm_keys = {
            _key(t) for t in _rows(arm_id, action)
            if (t.reason or "").startswith(reason_prefix)
        }
        n = 0
        for t in _rows(baseline.id, action):
            if (activated is not None and t.executed_at is not None
                    and _naive(t.executed_at) < activated):
                continue
            if not (t.reason or "").startswith(reason_prefix):
                continue
            if _key(t) not in arm_keys:
                n += 1
        return n

    # Per-arm gate metric builders. Each returns a list of
    # {label, n, target} dicts; the weekly-email >=5-closed-sells gate is
    # appended to every arm automatically.
    ARM_GATES = {
        "shadow_chop_damper": lambda a: [
            {"label": "chop days", "n": _chop_days_since(a.activated_at), "target": 15},
        ],
        "shadow_wide_trail": lambda a: [
            {"label": "exits", "n": len(_rows(a.id, "SELL")), "target": 15},
        ],
        "shadow_cap50": lambda a: [
            {"label": "exits", "n": len(_rows(a.id, "SELL")), "target": 15},
            {"label": "cap-tier fires",
             "n": sum(1 for t in _rows(a.id, "SELL")
                      if "PARTIAL PROFIT (100" in (t.reason or "")
                      or "+50" in (t.reason or "")), "target": 3},
        ],
        "shadow_chop_spy": lambda a: [
            {"label": "chop days", "n": _chop_days_since(a.activated_at), "target": 15},
        ],
        "shadow_sector_relief": lambda a: [
            {"label": "count-exempt pyramids",
             "n": _unmatched_vs_baseline(a.id, "PYRAMID"), "target": 5},
        ],
        "shadow_cs_window14": lambda a: [
            {"label": "CS buys in 8-14d band",
             "n": sum(1 for t in _rows(a.id, "BUY")
                      if 8 <= (_sf(t).get("cs_days_to_earnings") or 0) <= 14),
             "target": 5},
        ],
        "shadow_cs_exempt": lambda a: [
            {"label": "CS-cohort buys",
             "n": sum(1 for t in _rows(a.id, "BUY")
                      if _sf(t).get("coiled_spring")), "target": None},
            {"label": "pre-earnings exits suppressed",
             "n": _suppressed_vs_baseline(a.id, "SELL", "PRE-EARNINGS"),
             "target": 5},
        ],
        "shadow_ml_veto_off": lambda a: [
            # Buys the live veto would have blocked (conf < 0.30) — the
            # arm's whole reason to exist. signal_factors.ml_confidence is
            # stamped on every buy since the Jul-4 demotion kept logging on.
            {"label": "sub-0.30-confidence buys taken",
             "n": sum(1 for t in _rows(a.id, "BUY")
                      if (_sf(t).get("ml_confidence") or 1.0) < 0.30),
             "target": 5},
        ],
        "shadow_chop_entry_bar": lambda a: [
            {"label": "chop days", "n": _chop_days_since(a.activated_at), "target": 15},
            {"label": "baseline buys not taken (suppression proxy)",
             "n": _suppressed_vs_baseline(a.id, "BUY", ""), "target": 5},
        ],
        "shadow_chop_trim": lambda a: [
            {"label": "chop days", "n": _chop_days_since(a.activated_at), "target": 15},
            {"label": "chop trims fired",
             "n": sum(1 for t in _rows(a.id, "SELL")
                      if (t.reason or "").startswith("CHOP TRIM")), "target": 5},
        ],
    }

    arm_payload = []
    for a in arms:
        activated = a.activated_at
        if activated is not None and activated.tzinfo is None:
            activated = activated.replace(tzinfo=timezone.utc)
        sells = _rows(a.id, "SELL")
        metrics = ARM_GATES.get(a.name, lambda _a: [])(a)
        arm_payload.append({
            "id": a.id,
            "name": a.name,
            "description": a.description,
            "activated_at": a.activated_at.isoformat() if a.activated_at else None,
            "days_accrued": max((now_utc - activated).days, 0) if activated else None,
            "buys": len(_rows(a.id, "BUY")),
            "pyramids": len(_rows(a.id, "PYRAMID")),
            "sells": len(sells),
            "gate_metrics": metrics
            + [{"label": "closed sells (weekly-email gate)", "n": len(sells), "target": 5}],
        })

    # Program-level clocks that live outside the shadow fleet.
    STOP_RECHECK_CUTOFF = datetime(2026, 6, 24, tzinfo=timezone.utc)
    stop_sells = db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.action == "SELL",
        AIPortfolioTrade.user_id == 1,
        AIPortfolioTrade.executed_at > STOP_RECHECK_CUTOFF,
        AIPortfolioTrade.reason.like("STOP LOSS%"),
    ).all()
    # Split artifacts (e.g. SFBS 2:1 2026-08-21) are not genuine stops — same
    # exclusion as the reconciliation endpoint, so both cohort reads agree.
    stop_sells = [s for s in stop_sells
                  if not ((s.signal_factors or {}).get("split_artifact")
                          if isinstance(s.signal_factors, dict) else False)]
    stop_pcts = []
    for s in stop_sells:
        cost = (s.cost_basis or 0) * (s.shares or 0)
        if cost > 0 and s.realized_gain is not None:
            stop_pcts.append(s.realized_gain / cost * 100)
    stop_avg = round(sum(stop_pcts) / len(stop_pcts), 2) if stop_pcts else None
    # Pre-registered rule (Jun-24/Aug-3 program): at n>=5 clean stops, the fix
    # holds if the average stop lands at or above the -10% bar. Mechanical, so
    # the verdict fires itself instead of waiting on a manual curl.
    stop_verdict = None
    if stop_avg is not None and len(stop_pcts) >= 5:
        stop_verdict = "PASS" if stop_avg >= -10.0 else "FAIL"

    # Date-based program clocks (PM program 2026-08-25): the pre-registered
    # re-check calendar lives in the product instead of session notes, so
    # each date self-reports on the Gate Progress card and the Monday email.
    # Reschedule by editing here; provenance in the memory/backlog notes.
    CALENDAR_CLOCKS = [
        {"label": "Improving Radar out-of-sample re-validation",
         "due_date": "2026-09-15"},
        {"label": "ML demotion-cohort re-read (post per-lot FIFO fix)",
         "due_date": "2026-09-20"},
        {"label": "CS confidence v2 recalibration (~200 outcomes, post-Aug-19 rows)",
         "due_date": "2026-11-15"},
        {"label": "CANSLIM_API_TOKEN re-mint (Monday routine JWT dies 2027-03-01)",
         "due_date": "2027-02-15"},
    ]
    _today = datetime.now(timezone.utc).date()
    calendar = []
    for c in CALENDAR_CLOCKS:
        _due = date_cls.fromisoformat(c["due_date"])
        calendar.append({
            "label": c["label"],
            "due_date": c["due_date"],
            "days_until": (_due - _today).days,
            "due": _today >= _due,
        })

    return {
        "program_clocks": {
            "calendar": calendar,
            "stop_loss_recheck": {
                "label": "Exit-fix re-check (owner stops since Jun-24)",
                "n": len(stop_pcts), "target": 5,
                "avg_loss_pct": stop_avg,
                "bar_pct": -10.0,
                "verdict": stop_verdict,
            },
        },
        "arms": arm_payload,
        "note": (
            "Gate definitions are pre-registered in config/default.yaml "
            "comments; this endpoint only measures accrual. Verdicts come "
            "from the weekly A/B email decision rule, never from this card."
        ),
    }


@router.get("/experiment-gates")
@limiter.limit("30/minute")
async def experiment_gates(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Thin wrapper over compute_experiment_gates — see its docstring."""
    return compute_experiment_gates(db)

# ── Program milestone ledger ──────────────────────────────────────────────
# Owner-facing event history for the beat-SPY program. Rows come from
# backend/milestones.py (auto gate-diff writer + seeded history) and from
# the POST endpoint below (manual owner entries).

class MilestoneCreate(BaseModel):
    title: str
    detail: Optional[str] = None
    category: str = "decision"
    occurred_at: Optional[datetime] = None


def _milestone_out(r):
    return {
        "id": r.id,
        "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
        "category": r.category,
        "title": r.title,
        "detail": r.detail,
        "source": r.source,
    }


@router.get("/program-milestones")
@limiter.limit("60/minute")
async def list_program_milestones(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
    category: Optional[str] = Query(None),
    limit: int = Query(200, le=500),
):
    """Program Ledger: milestones newest-first, optional category filter."""
    from backend.database import ProgramMilestone
    q = db.query(ProgramMilestone)
    if category:
        q = q.filter(ProgramMilestone.category == category)
    rows = q.order_by(ProgramMilestone.occurred_at.desc(),
                      ProgramMilestone.id.desc()).limit(limit).all()
    return [_milestone_out(r) for r in rows]


@router.post("/program-milestones")
@limiter.limit("30/minute")
async def create_program_milestone(
    body: MilestoneCreate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Manual owner entry into the Program Ledger."""
    from backend.milestones import add_milestone
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title required")
    row = add_milestone(
        db,
        title=title,
        detail=(body.detail or "").strip() or None,
        category=body.category,
        occurred_at=body.occurred_at,
        source="owner",
    )
    return _milestone_out(row)


@router.delete("/program-milestones/{milestone_id}")
@limiter.limit("30/minute")
async def delete_program_milestone(
    milestone_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Remove a ledger row (typo cleanup). Auto rows re-fire only if their
    dedupe_key row is gone AND the condition still holds — deleting an
    auto row while its threshold remains crossed will restamp it with the
    next pass's date, so prefer deleting only manual entries."""
    from backend.database import ProgramMilestone
    row = db.query(ProgramMilestone).filter(
        ProgramMilestone.id == milestone_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Milestone not found")
    db.delete(row)
    db.commit()
    return {"deleted": milestone_id}

