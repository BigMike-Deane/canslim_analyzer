"""Corporate actions the price feed never reports (2026-09-24 audit).

FMP's quote endpoint keeps serving a bought-out or renamed stock's LAST trade
(volume 0, an old `timestamp`), and the scanner rewrites Stock.last_updated
every scan, so a dead name looks fresh. Found live on Sep-24:

  - ATAI: cash merger effective 2026-09-11, $6.75 + 1 CVR per share. User 3
    held 318 sh marked at the frozen $7.35 -- the stop could never fire and
    one of its 8 slots was stuck for good. shadow_ml_veto_off held it too.
  - FBRX: cash merger effective 2026-08-27 at $77 (shadow_ml_veto_off).
  - HLX: renamed HOS 2026-09-02; the scanner kept scoring the frozen HLX.

Splits are handled elsewhere (SPLIT rows). Once a day before the open this:

  1. Closes cash buyouts of held names -- live books and active shadow arms --
     at the deal cash, stamped at 09:30 ET on the effective date: what a real
     broker account shows. A non-tradable kicker (CVR) is carried at $0.
     Live snapshots from that moment on are restated to hold the cash
     instead of the frozen mark; shadow equity marks from that day are
     dropped so the next fill rebuilds them.
  2. Alerts on stock-for-stock deals and renames of held names (manual).
  3. Alerts on held names whose quote has not traded for 2+ sessions with
     no action on file (a halt, or an action Alpaca has not published).
  4. Sends scanned names Alpaca no longer lists AND whose FMP quote is
     frozen to delisted_tickers, so the scanner stops rescoring them.

Every network call is injectable for tests; the sweep never raises.
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LOOKBACK_DAYS = 90
FROZEN_QUOTE_DAYS = 10        # universe names: no trade in this long = dead
STALE_HELD_SESSIONS = 2       # held names: alert after this many silent sessions
DEAD_RECHECK_DAYS = 30
# Alpaca lists no preferreds; their app tickers end -P / -PA.. -PZ.
_PREFERRED = re.compile(r"-P[A-Z]?$")


# ── Feeds ──────────────────────────────────────────────────────────────────
def _alpaca_get(url: str, params: dict):
    from backend import alpaca_data as ad
    headers = ad._headers()
    if headers is None:
        return None
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        logger.warning(f"Alpaca {url}: HTTP {r.status_code} {r.text[:120]}")
        return None
    return r.json()


def fetch_active_symbols() -> set | None:
    """App-spelled tickers Alpaca lists as active US equities; None on failure."""
    body = _alpaca_get(f"{PAPER_BASE_URL}/v2/assets",
                       {"status": "active", "asset_class": "us_equity"})
    if not isinstance(body, list):
        return None
    return {a["symbol"].replace(".", "-") for a in body if a.get("symbol")}


def _d(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def normalize_actions(body: dict) -> list:
    """Alpaca v1beta1 corporate-actions payload -> flat action dicts."""
    ca = (body or {}).get("corporate_actions") or {}
    out = []
    for a in ca.get("cash_mergers") or []:
        out.append({"kind": "cash_merger", "symbol": a.get("acquiree_symbol"),
                    "cash": float(a.get("rate") or 0), "effective": _d(a.get("effective_date"))})
    for a in ca.get("stock_and_cash_mergers") or []:
        out.append({"kind": "stock_and_cash_merger", "symbol": a.get("acquiree_symbol"),
                    "cash": float(a.get("cash_rate") or 0), "effective": _d(a.get("effective_date")),
                    "acquirer": a.get("acquirer_symbol"), "acquirer_rate": a.get("acquirer_rate")})
    for a in ca.get("stock_mergers") or []:
        out.append({"kind": "stock_merger", "symbol": a.get("acquiree_symbol"),
                    "effective": _d(a.get("effective_date")),
                    "acquirer": a.get("acquirer_symbol"), "acquirer_rate": a.get("acquirer_rate")})
    for a in ca.get("name_changes") or []:
        out.append({"kind": "name_change", "symbol": a.get("old_symbol"),
                    "new_symbol": a.get("new_symbol"), "effective": _d(a.get("process_date"))})
    for a in out:
        a["symbol"] = (a.get("symbol") or "").replace(".", "-")
    return [a for a in out if a["symbol"] and a["effective"]]


def fetch_actions(tickers, start: date, end: date) -> list:
    from backend import alpaca_data as ad
    out, names = [], sorted({t for t in tickers if t})
    for i in range(0, len(names), 50):
        params = {"symbols": ",".join(ad.alpaca_symbol(t) for t in names[i:i + 50]),
                  "start": start.isoformat(), "end": end.isoformat(), "limit": 1000}
        for _ in range(20):                                   # page guard
            body = _alpaca_get(f"{ad.DATA_BASE_URL}/v1beta1/corporate-actions", params)
            if body is None:
                break
            out.extend(normalize_actions(body))
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    return out


def fmp_quote(ticker: str) -> dict | None:
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        return None
    for attempt in range(3):
        try:
            r = requests.get("https://financialmodelingprep.com/stable/quote",
                             params={"symbol": ticker, "apikey": key}, timeout=10)
            if r.status_code == 429:
                time.sleep(1 + attempt)
                continue
            if r.status_code != 200:
                return None
            data = r.json()
            return data[0] if isinstance(data, list) and data else {}
        except Exception:
            time.sleep(1)
    return None


def quote_trade_date(q: dict | None) -> date | None:
    """ET date of the quote's last trade, from FMP's epoch `timestamp`."""
    try:
        return datetime.fromtimestamp(int(q["timestamp"]), timezone.utc).astimezone(ET).date()
    except (KeyError, TypeError, ValueError):
        return None


# ── Classification ─────────────────────────────────────────────────────────
def cash_per_share(action: dict, active: set) -> float | None:
    """Deal cash per share when the whole consideration is cash (a
    non-tradable kicker such as a CVR counts as $0), else None."""
    if action["kind"] == "cash_merger" and action["cash"] > 0:
        return action["cash"]
    if action["kind"] == "stock_and_cash_merger" and action["cash"] > 0:
        acq = (action.get("acquirer") or "").replace(".", "-")
        if acq not in active:
            return action["cash"]
    return None


def effective_open_utc(d: date) -> datetime:
    """09:30 ET on the effective date, as naive UTC (the trades' convention)."""
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).astimezone(timezone.utc).replace(tzinfo=None)


# ── Live books ─────────────────────────────────────────────────────────────
def restate_snapshots(db, user_id: int, since_utc: datetime, shares: float,
                      frozen_px: float, cash_px: float, starting_cash: float) -> int:
    """Snapshots from `since_utc` on valued the dead position at the frozen
    mark; restate them to hold the deal cash instead."""
    from backend.database import AIPortfolioSnapshot
    delta = shares * (cash_px - frozen_px)
    snaps = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.user_id == user_id,
        AIPortfolioSnapshot.timestamp >= since_utc,
    ).order_by(AIPortfolioSnapshot.timestamp).all()
    for i, s in enumerate(snaps):
        s.cash = (s.cash or 0) + shares * cash_px
        s.positions_value = (s.positions_value or 0) - shares * frozen_px
        s.positions_count = max(0, (s.positions_count or 0) - 1)
        s.total_value = (s.total_value or 0) + delta
        if s.total_return is not None:
            s.total_return += delta
            if starting_cash:
                s.total_return_pct = s.total_return / starting_cash * 100
        if i > 0 and s.prev_value is not None:
            s.prev_value += delta
        if s.prev_value:
            s.value_change = s.total_value - s.prev_value
            s.value_change_pct = s.value_change / s.prev_value * 100
    return len(snaps)


def close_live_position(db, pos, cfg, action: dict, cash_px: float) -> dict:
    from backend.database import AIPortfolioTrade
    eff = effective_open_utc(action["effective"])
    shares = float(pos.shares or 0)
    frozen = float(pos.current_price or cash_px)
    cost = float(pos.cost_basis or 0)
    held = (action["effective"] - pos.purchase_date.date()).days if pos.purchase_date else None
    db.add(AIPortfolioTrade(
        ticker=pos.ticker, action="SELL", shares=shares, price=cash_px,
        total_value=shares * cash_px,
        reason=(f"CASH MERGER: ${cash_px:.2f}/sh effective {action['effective']} "
                f"({action['kind']}; booked by corporate-action sweep {date.today()})"),
        canslim_score=pos.current_score, growth_mode_score=pos.current_growth_score,
        is_growth_stock=pos.is_growth_stock or False, cost_basis=cost,
        realized_gain=(cash_px - cost) * shares,
        signal_factors={"sell_reason": "CASH MERGER", "corporate_action": action["kind"],
                        "gain_pct": round((cash_px / cost - 1) * 100, 1) if cost else None,
                        "frozen_mark": frozen},
        executed_at=eff, user_id=pos.user_id, holding_days=held,
        strategy=getattr(cfg, "strategy", None),
    ))
    cfg.current_cash = (cfg.current_cash or 0) + shares * cash_px
    n = restate_snapshots(db, pos.user_id, eff, shares, frozen, cash_px,
                          float(cfg.starting_cash or 0))
    db.delete(pos)
    return {"book": f"u{pos.user_id}", "ticker": pos.ticker, "shares": round(shares, 4),
            "cash_px": cash_px, "frozen_px": frozen, "snapshots_restated": n}


# ── Shadow arms ────────────────────────────────────────────────────────────
def close_shadow_holdings(db, strategy, closable: dict) -> list:
    """Close a stack's holdings in cash-bought-out names. `closable` maps
    ticker -> (action, cash_px). Caller holds shadow_trader._shadow_run_lock."""
    from backend.database import ShadowEquityMark
    from backend.shadow_trader import ShadowSession
    session = ShadowSession(db, strategy, [])
    done = []
    for pos in list(session._synthetic_positions or []):
        if pos.ticker not in closable or (pos.shares or 0) <= 0:
            continue
        action, cash_px = closable[pos.ticker]
        shares = float(pos.shares)
        st = session.emit_shadow_sell(
            position=pos, shares=shares, price=cash_px,
            reason=f"CASH MERGER: ${cash_px:.2f}/sh effective {action['effective']} ({action['kind']})")
        st.executed_at = effective_open_utc(action["effective"])
        st.signal_factors = {"sell_reason": "CASH MERGER", "corporate_action": action["kind"]}
        if pos.purchase_date:
            st.holding_days = (action["effective"] - pos.purchase_date.date()).days
        done.append({"book": strategy.name, "ticker": pos.ticker,
                     "shares": round(shares, 4), "cash_px": cash_px})
    pending = session.drain_pending_shadow_trades()
    for st in pending:
        db.add(st)
    if pending:
        first = min(a["effective"] for a, _ in closable.values())
        db.query(ShadowEquityMark).filter(
            ShadowEquityMark.shadow_strategy_id == strategy.id,
            ShadowEquityMark.date >= first,
        ).delete(synchronize_session=False)
    return done


def shadow_holdings(db) -> dict:
    """{strategy: {ticker: shares}} for active stacks (FIFO-derived)."""
    from backend.database import ShadowStrategy
    from backend.shadow_trader import ShadowSession
    out = {}
    for s in db.query(ShadowStrategy).filter(ShadowStrategy.archived_at.is_(None)).all():
        sess = ShadowSession(db, s, [])
        out[s] = {p.ticker: p.shares for p in (sess._synthetic_positions or [])
                  if (p.shares or 0) > 0}
    return out


# ── Universe hygiene ───────────────────────────────────────────────────────
def mark_dead_universe(db, active: set, held: set, quote_fn=fmp_quote,
                       today: date | None = None, pause_s: float = 0.25) -> list:
    """Universe tickers Alpaca does not list whose FMP quote has not traded
    for FROZEN_QUOTE_DAYS -> delisted_tickers (scanner exclusion). Held names
    are left to the alert path; rows marked here are re-checked monthly."""
    from backend.database import Stock, DelistedTicker
    today = today or datetime.now(ET).date()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    excluded = {r.ticker for r in db.query(DelistedTicker.ticker).filter(
        DelistedTicker.failure_count >= 3, DelistedTicker.recheck_after > now)}
    marked = []
    for (tk,) in db.query(Stock.ticker).all():
        if not tk or tk in active or tk in held or tk in excluded or _PREFERRED.search(tk):
            continue
        q = quote_fn(tk)
        if pause_s:
            time.sleep(pause_s)
        if q is None:                         # transport failure: decide tomorrow
            continue
        last = quote_trade_date(q)
        if q and last and (today - last).days < FROZEN_QUOTE_DAYS:
            continue                          # still trades somewhere Alpaca doesn't list
        why = f"frozen quote (last trade {last or 'none'}), not listed by Alpaca"
        row = db.query(DelistedTicker).filter(DelistedTicker.ticker == tk).first()
        if row is None:
            row = DelistedTicker(ticker=tk, source="corporate_actions")
            db.add(row)
        row.reason = why
        row.failure_count = max(3, row.failure_count or 0)
        row.last_failed_at = now
        row.recheck_after = now + timedelta(days=DEAD_RECHECK_DAYS)
        marked.append(tk)
    return marked


# ── Sweep ──────────────────────────────────────────────────────────────────
def run_sweep(db, *, active=None, actions=None, quote_fn=fmp_quote,
              today: date | None = None, universe: bool = True) -> dict:
    """One pass. `active` / `actions` / `quote_fn` are injectable for tests.
    Commits on success; returns a summary for logs and the ops alert."""
    from backend.database import AIPortfolioConfig, AIPortfolioPosition
    from backend.shadow_trader import _shadow_run_lock
    from backend.shadow_equity import session_days

    today = today or datetime.now(ET).date()
    summary = {"closed": [], "manual": [], "stale_held": [], "dead_marked": [], "error": None}
    if active is None:
        active = fetch_active_symbols()
    if not active:
        summary["error"] = "Alpaca asset list unavailable"
        return summary

    with _shadow_run_lock:
        live = db.query(AIPortfolioPosition).all()
        shadow = shadow_holdings(db)
        held = {p.ticker for p in live} | {t for h in shadow.values() for t in h}
        if actions is None:
            actions = fetch_actions(held, today - timedelta(days=LOOKBACK_DAYS), today)

        closable, seen = {}, set()
        for a in sorted(actions, key=lambda a: a["effective"]):
            if a["symbol"] not in held or a["effective"] > today:
                continue
            seen.add(a["symbol"])
            px = cash_per_share(a, active)
            if px is not None:
                closable.setdefault(a["symbol"], (a, px))
            else:
                summary["manual"].append(
                    f"{a['symbol']}: {a['kind']} eff {a['effective']}"
                    + (f" -> {a.get('new_symbol') or a.get('acquirer')}" if (a.get('new_symbol') or a.get('acquirer')) else ""))

        cfgs = {c.user_id: c for c in db.query(AIPortfolioConfig).all()}
        for pos in live:
            if pos.ticker in closable and pos.user_id in cfgs:
                a, px = closable[pos.ticker]
                if pos.purchase_date and pos.purchase_date.date() > a["effective"]:
                    continue                  # bought after the deal closed: not this lot
                summary["closed"].append(close_live_position(db, pos, cfgs[pos.user_id], a, px))
        for strat, holdings in shadow.items():
            mine = {t: closable[t] for t in holdings if t in closable}
            if mine:
                summary["closed"].extend(close_shadow_holdings(db, strat, mine))
        db.commit()

    # Held names that stopped trading with nothing on file.
    for tk in sorted(held - set(closable) - seen):
        last = quote_trade_date(quote_fn(tk) or {})
        if last is None:
            continue
        silent = [d for d in session_days(last + timedelta(days=1), today) if d < today]
        if len(silent) >= STALE_HELD_SESSIONS:
            summary["stale_held"].append(f"{tk}: last trade {last} ({len(silent)} sessions)")

    if universe:
        summary["dead_marked"] = mark_dead_universe(db, active, held, quote_fn=quote_fn, today=today)
        db.commit()
    return summary


def run_corporate_actions_sweep() -> dict:
    """Scheduler entry: never raises; alerts the owner when anything needs
    a human or a position was closed."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        s = run_sweep(db)
    except Exception as e:
        db.rollback()
        logger.error(f"Corporate-action sweep failed: {e}", exc_info=True)
        s = {"closed": [], "manual": [], "stale_held": [], "dead_marked": [], "error": str(e)}
    finally:
        db.close()
    logger.info(f"Corporate-action sweep: {len(s['closed'])} closed, {len(s['manual'])} manual, "
                f"{len(s['stale_held'])} stale held, {len(s['dead_marked'])} dead tickers marked"
                + (f", error: {s['error']}" if s.get("error") else ""))
    lines = [f"Closed at deal cash: {c['book']} {c['ticker']} {c['shares']} sh @ ${c['cash_px']:.2f}"
             for c in s["closed"]]
    lines += [f"Needs manual handling: {m}" for m in s["manual"]]
    lines += [f"Held but not trading: {m}" for m in s["stale_held"]]
    if s.get("error"):
        lines.append(f"Sweep error: {s['error']}")
    if lines:
        try:
            from backend.email_utils import send_ops_alert
            send_ops_alert(title="Corporate actions", message="\n".join(lines),
                           priority="high", tags=["briefcase"])
        except Exception as e:
            logger.warning(f"Corporate-action alert failed: {e}")
    return s
