"""Lab stop rules (docs/exposure-plan.md "Lab stop rules", pre-registered 2026-10-08).

Evaluated after each close mark (backend.lab.run_close_job), stored on the day's
LabEquityMark.checks, and pushed to the owner when a rule's level gets worse.

Levels, mildest first: ok, pending (not enough data yet), review, breach (mechanics
bug: pause and fix), stop (pre-registered end of the strategy).

Thresholds live in config ``lab_strategies.<name>.stop_rules`` (calibrated in
research/pit/lab_stop_calibration.py on history the Lab never sees).
"""
import logging
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional

from backend.database import LabDecision, LabEquityMark, LabOrder, LabStrategy

logger = logging.getLogger(__name__)

LEVELS = ("ok", "pending", "review", "breach", "stop")
_RANK = {lv: i for i, lv in enumerate(LEVELS)}
LEV_SPREAD, LEV_FEE = 0.005, 0.018     # backtest cost model for the 2x leg (cash + 0.5%, 2 x 0.9% fee)
_TERMINAL_BAD = ("canceled", "cancelled", "expired", "rejected", "error", "done_for_day", "stopped", "suspended")


def _rank(level: str) -> int:
    return _RANK[level]


def worst(checks: list) -> str:
    return max((c["level"] for c in checks), key=_rank, default="ok")


def _check(rule, level, value, threshold, detail):
    return {"rule": rule, "level": level, "value": value, "threshold": threshold, "detail": detail}


# ----------------------------------------------------------------- M: mechanics

def _start_date(db, s) -> Optional[date]:
    """First session the strategy traded with a broker (its forward test starts here)."""
    d = (db.query(LabDecision.date).filter(LabDecision.strategy_id == s.id,
                                           LabDecision.status.in_(("submitted", "unchanged")))
         .order_by(LabDecision.date).first())
    return d[0] if d else None


def _sessions(start: date, end: date) -> list:
    from backend.ai_trader import EASTERN_TZ, is_trading_day
    out, d = [], start
    while d <= end:
        if is_trading_day(datetime.combine(d, time(12), tzinfo=EASTERN_TZ)):
            out.append(d)
        d += timedelta(days=1)
    return out


def check_m1(db, s, today) -> dict:
    start = _start_date(db, s)
    if start is None:
        return _check("M1", "pending", None, "a decision every session", "no broker-backed decision yet")
    good = {d for (d,) in db.query(LabDecision.date).filter(
        LabDecision.strategy_id == s.id, LabDecision.date >= start,
        LabDecision.status.in_(("submitted", "unchanged")))}
    missed = [d.isoformat() for d in _sessions(start, today) if d not in good]
    return _check("M1", "breach" if missed else "ok", len(missed), 0,
                  f"sessions without a traded decision: {', '.join(missed[-5:])}" if missed
                  else f"decision recorded every session since {start.isoformat()}")


def check_m2(db, s, today, rules=None) -> dict:
    # Breaches already written up in the exposure-plan breach log ({order date: note}) stop
    # counting as open bugs; anything not listed still breaches.
    noted_dates = {str(d) for d in ((rules or {}).get("noted_breaches") or {})}
    bad, noted, open_today = [], [], 0
    for o in db.query(LabOrder).filter(LabOrder.strategy_id == s.id, LabOrder.date <= today).all():
        st = (o.status or "").lower()
        if st == "filled":
            continue
        if o.date == today and st not in _TERMINAL_BAD:
            open_today += 1          # a closing-auction fill can report late; 17:35 ET pass decides
            continue
        desc = f"{o.date.isoformat()} {o.side} {o.symbol} ({st or 'unknown'})"
        (noted if o.date.isoformat() in noted_dates else bad).append(desc)
    tail = f"; {len(noted)} noted in the breach log ({'; '.join(noted[-3:])})" if noted else ""
    if bad:
        # an order the paper broker let expire is its known random short fill, not our code: say so
        # plainly in the push (still a breach until written up; the next session repairs it)
        hint = (" -- Alpaca paper short-filled the order (known simulator quirk, see breach log); "
                "the next session tops up with a regular market order, no action needed"
                if all(b.endswith("(expired)") for b in bad) else "")
        return _check("M2", "breach", len(bad), 0, "orders not filled: " + "; ".join(bad[-5:]) + tail + hint)
    if open_today:
        return _check("M2", "pending", open_today, 0,
                      f"{open_today} of today's orders not yet reported filled" + tail)
    if noted:
        return _check("M2", "ok", 0, 0, f"every order filled except {len(noted)} noted in the breach log "
                                        f"({'; '.join(noted[-3:])})")
    return _check("M2", "ok", 0, 0, "every order filled")


def check_m3(db, s, today, rules) -> dict:
    floor = float(rules.get("target_share_min", 0.90))
    mark = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date <= today)
            .order_by(LabEquityMark.date.desc()).first())
    dec = (db.query(LabDecision).filter(LabDecision.strategy_id == s.id, LabDecision.date == mark.date)
           .first()) if mark else None
    if mark is None or dec is None or dec.status not in ("submitted", "unchanged") or not dec.target:
        return _check("M3", "pending", None, floor, "no traded decision with a closing mark yet")
    sym = next(iter(dec.target))
    eq = mark.equity or 0.0
    mv = {p["symbol"]: float(p.get("market_value") or 0) for p in (mark.positions or [])}
    share = mv.get(sym, 0.0) / eq if eq > 0 else 0.0
    stray = [k for k, v in mv.items() if k != sym and eq > 0 and v / eq > 0.01]
    ok = share >= floor and not stray
    detail = f"{share:.1%} of equity in {sym} on {mark.date.isoformat()}"
    if stray:
        detail += f"; also holds {', '.join(stray)}"
    return _check("M3", "ok" if ok else "breach", round(share, 4), floor, detail)


# ----------------------------------------------------------------- C: costs

def check_c1(db, s, rules, daily: Callable) -> dict:
    review, stop, n_min = (float(rules.get("fill_cost_review_bps", 7)), float(rules.get("fill_cost_stop_bps", 14)),
                           int(rules.get("min_fills", 6)))
    # every execution counts, including the filled part of an order that later expired or was canceled
    fills = [o for o in db.query(LabOrder).filter(LabOrder.strategy_id == s.id,
                                                  LabOrder.filled_avg_price.isnot(None)).all()
             if (o.filled_qty or 0) > 0 or (o.status or "").lower() == "filled"]
    closes, num, den, n = {}, 0.0, 0.0, 0
    for o in fills:
        if o.symbol not in closes:
            try:
                closes[o.symbol] = dict(daily(o.symbol, False))
            except Exception as e:
                logger.warning(f"lab checks: close for {o.symbol} unavailable ({type(e).__name__})")
                closes[o.symbol] = {}
        official = closes[o.symbol].get(o.date)
        if not official:
            continue
        sign = 1 if o.side == "buy" else -1                     # paying above / selling below the close = cost
        bps = sign * (o.filled_avg_price - official) / official * 1e4
        w = (o.filled_qty or o.qty) * o.filled_avg_price          # shares actually filled
        num, den, n = num + bps * w, den + w, n + 1
    avg = round(num / den, 2) if den else None
    detail = f"{n} fills measured vs the official close" + (f", average {avg:+.1f} bps" if avg is not None else "")
    if n < n_min or avg is None:
        return _check("C1", "pending", avg, review, detail + f" (rule needs {n_min})")
    level = "stop" if avg >= stop else "review" if avg > review else "ok"
    return _check("C1", level, avg, {"review": review, "stop": stop}, detail)


def tracking_gap(daily: Callable, sessions: int) -> Optional[float]:
    """SSO's realized return minus the backtest's modelled 2x leg over the trailing
    ``sessions``, annualized (%/yr; negative = SSO worse than the backtest assumed)."""
    sso, spy, bil = (dict(daily(sym, True)) for sym in ("SSO", "SPY", "BIL"))
    days = sorted(set(sso) & set(spy) & set(bil))
    if len(days) < sessions + 1:
        return None
    days = days[-(sessions + 1):]
    gaps = []
    for a, b in zip(days, days[1:]):
        r = {k: v[b] / v[a] - 1 for k, v in (("sso", sso), ("spy", spy), ("bil", bil))}
        model = 2 * r["spy"] - (r["bil"] + LEV_SPREAD / 252) - LEV_FEE / 252
        gaps.append(r["sso"] - model)
    return round(sum(gaps) / len(gaps) * 252 * 100, 2)


def check_c2(rules, gaps: dict) -> dict:
    review = float(rules.get("tracking_review_pct", -1.71))
    breakeven = float(rules.get("tracking_breakeven_pct", -1.71))
    g126, g252 = gaps.get(126), gaps.get(252)
    if g126 is None:
        return _check("C2", "pending", None, review, "SSO / SPY / BIL history unavailable")
    detail = f"SSO vs modelled 2x leg: {g126:+.2f}%/yr (126 sessions)" + (
        f", {g252:+.2f}%/yr (252)" if g252 is not None else "")
    if g252 is not None and g252 < breakeven:
        return _check("C2", "review", g126, {"review": review, "stop_candidate": breakeven},
                      detail + " -- STOP candidate: re-run the 1994-2026 backtest with this gap "
                               "(research/pit/lab_stop_calibration.py); STOP if the edge is <= 0")
    return _check("C2", "review" if g126 < review else "ok", g126, review, detail)


# ----------------------------------------------------------------- P: performance

def _bench(m):
    return m.spy_adj_close or m.spy_close


def check_p1(marks, rules) -> dict:
    rev, stp = rules.get("excess_review_pct", {}) or {}, rules.get("excess_stop_pct", {}) or {}
    marks = [m for m in marks if m.equity and _bench(m)]
    out, level, have = [], "ok", False
    for h in sorted(int(k) for k in rev):
        if len(marks) <= h:
            continue
        have = True
        a, b = marks[-1 - h], marks[-1]
        ex = ((b.equity / a.equity) - (_bench(b) / _bench(a))) * 100
        r, st = float(rev[h] if h in rev else rev[str(h)]), float(stp[h] if h in stp else stp[str(h)])
        lv = "stop" if ex < st else "review" if ex < r else "ok"
        level = max(level, lv, key=_rank)
        out.append(f"{h}d {ex:+.1f}% (review < {r:+.1f}, stop < {st:+.1f})")
    if not have:
        first = min((int(k) for k in rev), default=63)
        return _check("P1", "pending", None, None, f"{len(marks)} marks; first window at {first} sessions")
    return _check("P1", level, None, None, "; ".join(out))


def check_p2(marks, rules) -> dict:
    lim = float(rules.get("max_dd_review_pct", 45.7))
    eq = [m.equity for m in marks if m.equity]
    if len(eq) < 2:
        return _check("P2", "pending", None, lim, "not enough marks")
    peak, dd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, (1 - v / peak) * 100)
    cur = (1 - eq[-1] / max(eq)) * 100
    return _check("P2", "review" if dd > lim else "ok", round(dd, 2), lim,
                  f"max drawdown {dd:.1f}% (now {cur:.1f}%); backtest 1994-2026 max {lim:.1f}%")


# ----------------------------------------------------------------- entry points

def evaluate(db, s: LabStrategy, rules: dict, today: date, daily: Callable, gaps: Optional[dict] = None) -> list:
    """All stop-rule checks for one strategy as of ``today`` (after its close mark)."""
    marks = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date <= today)
             .order_by(LabEquityMark.date).all())
    start = _start_date(db, s)
    if start is not None:
        marks = [m for m in marks if m.date >= start]
    if gaps is None:
        gaps = {n: _safe_gap(daily, n) for n in (126, 252)}
    return [check_m1(db, s, today), check_m2(db, s, today, rules), check_m3(db, s, today, rules),
            check_c1(db, s, rules, daily), check_c2(rules, gaps), check_p1(marks, rules), check_p2(marks, rules)]


def _safe_gap(daily, n):
    try:
        return tracking_gap(daily, n)
    except Exception as e:
        logger.warning(f"lab checks: tracking gap unavailable ({type(e).__name__})")
        return None


def worsened(prev: Optional[list], cur: list) -> list:
    """Checks whose level is review-or-worse AND worse than in ``prev``."""
    before = {c["rule"]: c["level"] for c in (prev or [])}
    return [c for c in cur if _RANK[c["level"]] >= _RANK["review"]
            and _RANK[c["level"]] > _RANK.get(before.get(c["rule"], "ok"), 0)]


def run_checks(db, s: LabStrategy, today: date, daily: Callable, gaps: Optional[dict] = None,
               notify: bool = True) -> Optional[list]:
    """Evaluate, store on today's mark, and push the owner when a rule got worse."""
    from backend.lab import lab_config, safe_error
    cfg = lab_config().get(s.name, {}) or {}
    rules = cfg.get("stop_rules") or {}
    mark = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == today).first()
    if mark is None:
        return None
    prev = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date < today,
                                           LabEquityMark.checks.isnot(None))
            .order_by(LabEquityMark.date.desc()).first())
    prev_checks = mark.checks or (prev.checks if prev else None)   # 2nd pass the same evening compares to the 1st
    checks = evaluate(db, s, rules, today, daily, gaps)
    mark.checks = checks
    db.commit()
    new = worsened(prev_checks, checks)
    if new and notify:
        try:
            from backend.email_utils import create_notification
            lv = worst(new)
            create_notification(
                user_id=int(cfg.get("notify_user_id", 1)), kind="lab_stop_rule",
                priority="high" if lv in ("stop", "breach") else "default",
                title=f"Lab {s.label}: {lv.upper()} on {', '.join(c['rule'] for c in new)}",
                body=" | ".join(f"{c['rule']}: {c['detail']}" for c in new)[:900],
                data={"strategy": s.name, "date": today.isoformat(), "url": "/lab"})
        except Exception as e:
            logger.warning(f"lab[{s.name}] stop-rule notification failed: {safe_error(e)}")
    return checks
