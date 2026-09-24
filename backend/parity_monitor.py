"""Daily parity monitor (2026-09-24): the Sep-24 audit's findings as standing
checks, so a future change that lets the arms drift from live gets caught the
same evening instead of at the next hand audit.

Owner rule: shadow arms and any test logic trade exactly as the live book
would. Each check below is a way that rule was found broken:

  - off_hours: a fill outside the NYSE session (arms traded after every
    overnight/weekend scan until a69a2d9; 175 fills).
  - runt_buy: an arm buy under live's min_position_value (arms bought
    $0.34-$274 positions that held a slot for weeks).
  - unpriced_mark: an arm's latest daily equity mark carries a holding with
    no close on that day (bought-out ATAI/FBRX sat in ml_veto_off for weeks).
  - negative_cash: an arm's FIFO-derived cash below zero.
  - live_cash: a live book's cash differs from starting − buys − pyramids +
    sells (books without the SPY sweep only: the sweep moves cash without
    trade rows, so the identity doesn't hold there).

Read-only. Runs weekdays after the close; alerts the owner only when a check
fails. Days before PARITY_SINCE predate the fixes and are not re-reported.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
PARITY_SINCE = date(2026, 9, 25)
# A cycle that starts inside the session can stamp its last fills a few
# minutes after the bell (live's own 16:00 cycle lands ~16:00:25).
CLOSE_GRACE = timedelta(minutes=10)
RUNT_TOLERANCE = 0.9      # equity moves between marks; flag only clear misses
CASH_TOLERANCE = 0.05


def _et(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(ET)


def _session(day: date):
    """(open, close) ET for `day`, or None when the market was shut."""
    from backend.ai_trader import is_trading_day, _nyse_session_bounds
    if not is_trading_day(datetime(day.year, day.month, day.day, 12, tzinfo=ET)):
        return None
    bounds = _nyse_session_bounds(day)
    if bounds:
        return bounds[0].astimezone(ET), bounds[1].astimezone(ET)
    return (datetime.combine(day, time(9, 30), ET), datetime.combine(day, time(16, 0), ET))


def _day_window_utc(day: date):
    start = datetime.combine(day, time(0, 0), ET).astimezone(timezone.utc).replace(tzinfo=None)
    return start, start + timedelta(days=1)


def check_off_hours(db, day: date) -> list:
    from backend.database import AIPortfolioTrade, ShadowTrade, ShadowStrategy
    lo, hi = _day_window_utc(day)
    session = _session(day)
    out = []
    rows = [("live u%s" % t.user_id, t) for t in db.query(AIPortfolioTrade).filter(
        AIPortfolioTrade.executed_at >= lo, AIPortfolioTrade.executed_at < hi)]
    names = {s.id: s.name for s in db.query(ShadowStrategy)}
    rows += [(names.get(t.shadow_strategy_id, t.shadow_strategy_id), t) for t in db.query(ShadowTrade).filter(
        ShadowTrade.executed_at >= lo, ShadowTrade.executed_at < hi)]
    for who, t in rows:
        if t.action == "SPLIT":
            continue
        et = _et(t.executed_at)
        if session is None or not (session[0] <= et <= session[1] + CLOSE_GRACE):
            out.append(f"off_hours: {who} {t.action} {t.ticker} at {et:%a %m-%d %H:%M} ET")
    return out


def check_runt_buys(db, day: date) -> list:
    from backend.database import ShadowTrade, ShadowStrategy, ShadowEquityMark
    from backend.trading_engine import min_position_value
    lo, hi = _day_window_utc(day)
    out = []
    for s in db.query(ShadowStrategy).filter(ShadowStrategy.archived_at.is_(None)):
        mark = db.query(ShadowEquityMark).filter(
            ShadowEquityMark.shadow_strategy_id == s.id, ShadowEquityMark.date < day,
        ).order_by(ShadowEquityMark.date.desc()).first()
        equity = float(mark.equity if mark else (s.starting_value or 25000.0))
        floor = min_position_value(equity) * RUNT_TOLERANCE
        for t in db.query(ShadowTrade).filter(
                ShadowTrade.shadow_strategy_id == s.id, ShadowTrade.action == "BUY",
                ShadowTrade.executed_at >= lo, ShadowTrade.executed_at < hi):
            if t.ticker == "SPY" or (t.reason or "").startswith("SPY SWEEP"):
                continue
            if (t.total_value or 0) < floor:
                out.append(f"runt_buy: {s.name} {t.ticker} ${t.total_value:.2f} (floor ~${floor / RUNT_TOLERANCE:.0f})")
    return out


def check_unpriced_marks(db) -> list:
    from backend.database import ShadowStrategy, ShadowEquityMark
    out = []
    for s in db.query(ShadowStrategy).filter(ShadowStrategy.archived_at.is_(None)):
        m = db.query(ShadowEquityMark).filter(ShadowEquityMark.shadow_strategy_id == s.id) \
            .order_by(ShadowEquityMark.date.desc()).first()
        if m and (m.unpriced_positions or 0) > 0:
            out.append(f"unpriced_mark: {s.name} {m.date} has {m.unpriced_positions} holding(s) with no close")
    return out


def check_negative_cash(db) -> list:
    from backend.database import ShadowStrategy
    from backend.shadow_trader import ShadowSession
    out = []
    for s in db.query(ShadowStrategy).filter(ShadowStrategy.archived_at.is_(None)):
        cash = float(getattr(ShadowSession(db, s, [])._synthetic_config, "current_cash", 0) or 0)
        if cash < -1.0:
            out.append(f"negative_cash: {s.name} ${cash:,.2f}")
    return out


def check_live_cash(db) -> list:
    from sqlalchemy import func
    from backend.database import AIPortfolioConfig, AIPortfolioTrade
    from backend.trading_utils import get_strategy_profile
    out = []
    for cfg in db.query(AIPortfolioConfig).all():
        profile = get_strategy_profile(cfg.strategy or "balanced") or {}
        if (profile.get("spy_sweep") or {}).get("enabled") or (cfg.spy_sweep_shares or 0) > 0:
            continue    # sweep cash moves have no trade rows

        def total(actions):
            return float(db.query(func.coalesce(func.sum(AIPortfolioTrade.total_value), 0.0)).filter(
                AIPortfolioTrade.user_id == cfg.user_id,
                AIPortfolioTrade.action.in_(actions)).scalar() or 0)
        expected = float(cfg.starting_cash or 0) - total(["BUY", "PYRAMID"]) + total(["SELL"])
        if abs(expected - float(cfg.current_cash or 0)) > CASH_TOLERANCE:
            out.append(f"live_cash: u{cfg.user_id} cash ${cfg.current_cash:,.2f} vs ledger ${expected:,.2f}")
    return out


def run_parity_checks(db, day: date | None = None) -> list:
    day = day or datetime.now(ET).date()
    findings = []
    if day >= PARITY_SINCE:
        findings += check_off_hours(db, day)
        findings += check_runt_buys(db, day)
    findings += check_unpriced_marks(db)
    findings += check_negative_cash(db)
    findings += check_live_cash(db)
    return findings


def run_parity_monitor() -> list:
    """Scheduler entry: never raises; alerts only when a check fails."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        findings = run_parity_checks(db)
    except Exception as e:
        logger.error(f"Parity monitor failed: {e}", exc_info=True)
        findings = [f"monitor error: {e}"]
    finally:
        db.close()
    logger.info(f"Parity monitor: {len(findings)} finding(s)")
    if findings:
        try:
            from backend.email_utils import send_ops_alert
            shown = findings[:25] + ([f"... and {len(findings) - 25} more"] if len(findings) > 25 else [])
            send_ops_alert(title="Parity monitor", message="\n".join(shown),
                           priority="high", tags=["mag"])
        except Exception as e:
            logger.warning(f"Parity monitor alert failed: {e}")
    return findings
