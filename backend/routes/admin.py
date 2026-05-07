"""Admin routes: user management (invite-only, Google Sign-In) +
operational diagnostics that are too sensitive for the public API."""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
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

    # Total return relative to the strategy's reference capital.
    total_return_pct = None
    if starting_value > 0:
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
    return warnings


@router.get("/strategy-ab-eval")
async def run_strategy_ab_comparison(
    strategy: str = Query(..., description="Strategy profile name, e.g. 'nostate_optimized'"),
    cutoff_date: str = Query(..., description="ISO date marking experiment start, e.g. '2026-05-07'"),
    pre_window_days: int = Query(default=30, ge=1, le=365, description="Days BEFORE cutoff for baseline"),
    post_window_days: Optional[int] = Query(default=None, description="Days AFTER cutoff to evaluate; defaults to days-since-cutoff capped at 90"),
    exclude_pyramids: bool = Query(default=True, description="Filter out pyramid rows (action='PYRAMID' OR reason LIKE 'PYRAMID:%')"),
    min_return_delta_pp: float = Query(default=-5.0, description="Decision threshold: keep if return delta >= this"),
    min_sharpe_delta: float = Query(default=0.0, description="Decision threshold: keep if sharpe delta >= this"),
    min_post_sells: int = Query(default=5, ge=0, description="Minimum SELL count in post window for a confident decision"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
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
    # --- Validation ----------------------------------------------------------
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

    # --- User scoping --------------------------------------------------------
    user_rows = db.query(AIPortfolioConfig.user_id, AIPortfolioConfig.starting_cash).filter(
        AIPortfolioConfig.strategy == strategy
    ).all()
    if not user_rows:
        raise HTTPException(status_code=404, detail=f"No users currently on strategy '{strategy}'")

    user_ids = [row.user_id for row in user_rows]
    starting_value = sum((row.starting_cash or 25000.0) for row in user_rows)

    # --- Trade query ---------------------------------------------------------
    base = db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.user_id.in_(user_ids),
        AIPortfolioTrade.executed_at >= pre_start,
        AIPortfolioTrade.executed_at < post_end,
    )
    if exclude_pyramids:
        # Same shape as strategy-health: filter both action='PYRAMID' rows
        # AND action='BUY' + reason LIKE 'PYRAMID:%' rows (live-trader sync
        # drift fixed in commit 5269dbf, but legacy rows linger).
        # NULL reasons must be PRESERVED — wrap reason filter in an OR with
        # IS NULL because (~col.like(...)) is NULL for NULL → SQL false.
        base = base.filter(
            ~AIPortfolioTrade.action.in_(['PYRAMID']),
            or_(
                AIPortfolioTrade.reason.is_(None),
                ~AIPortfolioTrade.reason.like('PYRAMID:%'),
            ),
        )

    pre_trades = base.filter(AIPortfolioTrade.executed_at < cutoff).order_by(AIPortfolioTrade.executed_at).all()
    post_trades = base.filter(AIPortfolioTrade.executed_at >= cutoff).order_by(AIPortfolioTrade.executed_at).all()

    # --- Summarize -----------------------------------------------------------
    pre_summary = _summarize_window(pre_trades, pre_window_days, starting_value)
    post_summary = _summarize_window(post_trades, post_window_days, starting_value)
    delta = _compute_delta(pre_summary, post_summary)

    criteria = {
        'min_return_delta_pp': min_return_delta_pp,
        'min_sharpe_delta': min_sharpe_delta,
        'min_post_sells': min_post_sells,
    }
    decision = _decide(pre_summary, post_summary, delta, criteria)

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
        'experiment': {
            'strategy': strategy,
            'cutoff_date': cutoff_date,
            'pre_window': pre_window,
            'post_window': post_window,
            'starting_value_reference': round(starting_value, 2),
            'user_ids': user_ids,
            'exclude_pyramids': exclude_pyramids,
        },
        'summary': decision,
        'pre': pre_summary,
        'post': post_summary,
        'delta': delta,
        'warnings': _build_warnings(pre_window, post_window, pre_summary, post_summary),
    }
