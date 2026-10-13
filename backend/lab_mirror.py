"""Lab mirrors (owner Oct-12): show books that trade elsewhere as Lab strategies, side by side.

  mirror_shadow        a shadow stack (e.g. shadow_model_v1, the evidence-weighted model): daily marks
                       from ShadowEquityMark, holdings replayed from its trade log, its trades
  mirror_ai_portfolio  the live AI Portfolio of one user, from the same start date: daily marks from
                       its last snapshot of each session, its trades

Both write ordinary LabEquityMark / LabOrder / LabDecision rows, so the Lab's chart, scorecard,
positions and trade log work unchanged. Idempotent: re-running rewrites the same rows.
The weekly review (owner: "a weekly routine to review the two and identify any noticeable gaps")
compares the pair; /api/lab/weekly-review serves it any time, the Friday job pushes it.
"""
import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backend.database import LabDecision, LabEquityMark, LabOrder, LabStrategy

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MIRROR_KINDS = ("mirror_shadow", "mirror_ai_portfolio")
OWNER_ID = 1
FLAG_RETURN_GAP_PP = 2.0      # weekly return gap worth calling out
FLAG_INVESTED_GAP_PP = 25.0   # one book much more invested than the other


def _cfg(name: str) -> dict:
    from backend.lab import lab_config
    return lab_config().get(name) or {}


def _start(cfg: dict) -> Optional[date]:
    s = cfg.get("start")
    return date.fromisoformat(str(s)) if s else None


def _et_date(dt: datetime) -> date:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).date()


def _upsert_mark(db, s: LabStrategy, d: date, equity: float, cash: float, positions: list, spy_raw: dict, spy_adj: list):
    from backend.lab import chain_spy_adj
    m = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == d).first() \
        or LabEquityMark(strategy_id=s.id, date=d, equity=0.0)
    m.equity, m.cash = float(equity), float(cash or 0)
    m.positions_value = float(sum(p.get("market_value") or 0 for p in positions))
    m.positions = positions
    m.spy_close = spy_raw.get(d)
    m.spy_adj_close = chain_spy_adj(db, s.id, d, spy_adj)
    db.add(m)
    db.commit()


def _upsert_order(db, s: LabStrategy, key: str, at: datetime, symbol: str, action: str, shares: float, price: float,
                  reason: Optional[str]):
    cid = f"mirror-{s.id}-{key}"
    o = db.query(LabOrder).filter(LabOrder.client_order_id == cid).first() or LabOrder(
        strategy_id=s.id, client_order_id=cid, date=_et_date(at), symbol=symbol, qty=0.0, side="buy")
    o.side = "sell" if action == "SELL" else "buy"
    o.qty = o.filled_qty = float(shares)
    o.filled_avg_price = float(price)
    o.status = "filled"
    o.reason = (reason or "")[:500]
    o.submitted_at = o.filled_at = at.replace(tzinfo=None) if at.tzinfo else at
    db.add(o)


def _day_decision(db, s: LabStrategy, d: date, trades: list, holdings: int, cash: float, note: Optional[str] = None):
    """One LabDecision per session in the Lab's 'trades' shape (buys [sym, value, score], sells [sym, why, pnl])."""
    buys, sells = [], []
    for t in trades:
        if t["action"] in ("BUY", "PYRAMID"):
            buys.append([t["symbol"] + (" +" if t["action"] == "PYRAMID" else ""), round(t["value"], 2), t.get("score") or 0])
        elif t["action"] == "SELL":
            why = (t.get("reason") or "").split(":")[0][:40]
            sells.append([t["symbol"], why, t.get("pnl") or 0.0])
    dec = db.query(LabDecision).filter(LabDecision.strategy_id == s.id, LabDecision.date == d).first() \
        or LabDecision(strategy_id=s.id, date=d)
    dec.inputs = {"kind": "trades", "holdings": holdings, "cash": round(cash or 0, 2), "buys": buys, "sells": sells}
    dec.target = {}
    dec.status = "mirrored"
    dec.note = note
    db.add(dec)
    db.commit()


def _cost_book(trades: list) -> dict:
    """Average cost per open ticker, replaying BUY/PYRAMID/SELL (sweeps and splits handled by caller)."""
    sh, cost = defaultdict(float), defaultdict(float)
    for t in trades:
        q, px = float(t.shares or 0), float(t.price or 0)
        if t.action in ("BUY", "PYRAMID"):
            cost[t.ticker] += q * px
            sh[t.ticker] += q
        elif t.action == "SELL" and sh[t.ticker] > 0:
            avg = cost[t.ticker] / sh[t.ticker]
            sh[t.ticker] = max(0.0, sh[t.ticker] - q)
            cost[t.ticker] = avg * sh[t.ticker]
    return {k: cost[k] / sh[k] for k in sh if sh[k] > 1e-9}


# ----------------------------------------------------------------- mirrors

def mirror_shadow(db, s: LabStrategy, cfg: dict, spy_raw: dict, spy_adj: list, closes_fn=None) -> int:
    from backend.database import ShadowEquityMark, ShadowStrategy, ShadowTrade
    from backend.shadow_equity import replay_book, session_close_utc
    from backend.shadow_trader import is_sweep_reason
    st = db.query(ShadowStrategy).filter(ShadowStrategy.name == cfg.get("source")).first()
    if st is None:
        logger.warning(f"lab mirror {s.name}: shadow stack {cfg.get('source')!r} not found")
        return 0
    start = _start(cfg) or date.min
    marks = db.query(ShadowEquityMark).filter(ShadowEquityMark.shadow_strategy_id == st.id,
                                              ShadowEquityMark.date >= start).order_by(ShadowEquityMark.date).all()
    trades = db.query(ShadowTrade).filter(ShadowTrade.shadow_strategy_id == st.id).order_by(ShadowTrade.executed_at).all()
    if not marks:
        return 0
    if closes_fn is None:
        from backend.alpaca_data import daily_closes_raw as closes_fn
    held_ever = sorted({t.ticker for t in trades if t.ticker and not is_sweep_reason(t.reason)} | {"SPY"})
    closes = closes_fn(held_ever, marks[0].date - timedelta(days=7)) or {}
    by_day = defaultdict(list)
    for t in trades:
        if is_sweep_reason(t.reason) or _et_date(t.executed_at) < start:
            continue
        _upsert_order(db, s, f"s{t.id}", t.executed_at, t.ticker, t.action, t.shares, t.price, t.reason)
        by_day[_et_date(t.executed_at)].append({"action": t.action, "symbol": t.ticker, "value": t.total_value or 0,
                                                "score": t.canslim_score, "reason": t.reason,
                                                "pnl": (t.realized_gain / (t.cost_basis * t.shares)) if t.action == "SELL" and t.realized_gain is not None and t.cost_basis and t.shares else None})
    db.commit()
    for m in marks:
        cut = session_close_utc(m.date)
        book = replay_book(trades, float(st.starting_value or 25000.0), cut)
        avg = _cost_book([t for t in trades if (t.executed_at.replace(tzinfo=None) if t.executed_at.tzinfo else t.executed_at) < cut
                          and not is_sweep_reason(t.reason)])
        pos = []
        for tk, q in book["shares"].items():
            px = (closes.get(tk) or {}).get(m.date) or book["last_px"].get(tk) or 0.0
            ac = avg.get(tk)
            pos.append({"symbol": tk, "qty": round(q, 4), "market_value": round(q * px, 2),
                        "avg_entry_price": round(ac, 4) if ac else None,
                        "unrealized_plpc": round(px / ac - 1, 4) if ac else 0.0})
        if (m.sweep_value or 0) > 0:
            pos.append({"symbol": "SPY", "qty": round(book["sweep_shares"], 4), "market_value": round(m.sweep_value, 2),
                        "avg_entry_price": None, "unrealized_plpc": 0.0, "note": "idle cash parked in SPY"})
        pos.sort(key=lambda p: -p["market_value"])
        _upsert_mark(db, s, m.date, m.equity, m.cash, pos, spy_raw, spy_adj)
        _day_decision(db, s, m.date, by_day.get(m.date, []), len(book["shares"]), m.cash)
    return len(marks)


def mirror_ai_portfolio(db, s: LabStrategy, cfg: dict, spy_raw: dict, spy_adj: list, today: Optional[date] = None) -> int:
    from backend.ai_trader import is_trading_day
    from backend.database import AIPortfolioPosition, AIPortfolioSnapshot, AIPortfolioTrade
    uid = int(cfg.get("source_user", OWNER_ID))
    start = _start(cfg)
    today = today or datetime.now(ET).date()
    if start is None or today < start:
        return 0
    snaps = db.query(AIPortfolioSnapshot).filter(
        AIPortfolioSnapshot.user_id == uid,
        AIPortfolioSnapshot.timestamp >= datetime.combine(start, time(0)) ).order_by(AIPortfolioSnapshot.timestamp).all()
    last = {}
    for sn in snaps:
        d = _et_date(sn.timestamp)
        if d >= start:
            last[d] = sn                      # the session's last snapshot (closing mark runs after 16:05 ET)
    trades = db.query(AIPortfolioTrade).filter(AIPortfolioTrade.user_id == uid,
                                               AIPortfolioTrade.executed_at >= datetime.combine(start, time(0))
                                               ).order_by(AIPortfolioTrade.executed_at).all()
    by_day = defaultdict(list)
    for t in trades:
        if _et_date(t.executed_at) < start:
            continue
        _upsert_order(db, s, f"a{t.id}", t.executed_at, t.ticker, t.action, t.shares, t.price, t.reason)
        by_day[_et_date(t.executed_at)].append({"action": t.action, "symbol": t.ticker, "value": t.total_value or 0,
                                                "score": t.canslim_score, "reason": t.reason,
                                                "pnl": (t.realized_gain / (t.cost_basis * t.shares)) if t.action == "SELL" and t.realized_gain is not None and t.cost_basis and t.shares else None})
    db.commit()
    n = 0
    live_pos = db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == uid).all()
    for d, sn in sorted(last.items()):
        if not is_trading_day(datetime.combine(d, time(12), tzinfo=ET)):
            continue
        prev = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == d).first()
        pos = prev.positions if prev is not None and prev.positions else []
        if d == today:                        # holdings history isn't stored: snapshot them on their own day
            pos = sorted(({"symbol": p.ticker, "qty": round(p.shares, 4), "market_value": round(p.current_value or 0, 2),
                           "avg_entry_price": round(p.cost_basis, 4),
                           "unrealized_plpc": round((p.gain_loss_pct or 0) / 100, 4)} for p in live_pos),
                         key=lambda p: -p["market_value"])
        _upsert_mark(db, s, d, sn.total_value, sn.cash, pos, spy_raw, spy_adj)
        _day_decision(db, s, d, by_day.get(d, []), sn.positions_count or len(pos), sn.cash)
        n += 1
    return n


def run_mirror_job():
    from backend.ai_trader import EASTERN_TZ, is_trading_day
    from backend.database import SessionLocal
    from backend.lab import fmp_daily, safe_error, sync_strategies
    now = datetime.now(EASTERN_TZ)
    if not is_trading_day(now):
        return
    db = SessionLocal()
    try:
        mirrors = [x for x in sync_strategies(db) if x.is_active and x.kind in MIRROR_KINDS]
        if not mirrors:
            return
        spy_raw, spy_adj = dict(fmp_daily("SPY", days=60)), fmp_daily("SPY", days=420, adjusted=True)
        for s in mirrors:
            cfg = _cfg(s.name)
            try:
                if s.kind == "mirror_shadow":
                    n = mirror_shadow(db, s, cfg, spy_raw, spy_adj)
                else:
                    n = mirror_ai_portfolio(db, s, cfg, spy_raw, spy_adj, today=now.date())
                if s.activated_at is None and n:
                    s.activated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                    db.commit()
                logger.info(f"lab mirror {s.name}: {n} marks")
            except Exception as e:
                db.rollback()
                logger.error(f"lab mirror {s.name} failed: {safe_error(e)}")
    finally:
        db.close()


# ----------------------------------------------------------------- weekly review

def _marks(db, s):
    return db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id).order_by(LabEquityMark.date).all()


def _ret(marks, since: Optional[date] = None):
    ms = [m for m in marks if since is None or m.date >= since]
    if len(ms) < 2:
        return None, None
    r = ms[-1].equity / ms[0].equity - 1
    sa = [m.spy_adj_close or m.spy_close for m in (ms[0], ms[-1])]
    spy = sa[1] / sa[0] - 1 if all(sa) else None
    return r * 100, (spy * 100 if spy is not None else None)


def _week_start(marks, today: date) -> Optional[date]:
    """The last mark before this week's Monday = the base for 'this week'."""
    monday = today - timedelta(days=today.weekday())
    prior = [m.date for m in marks if m.date < monday]
    return prior[-1] if prior else (marks[0].date if marks else None)


def weekly_review(db, today: Optional[date] = None) -> dict:
    """The model vs the AI Portfolio: returns (week, since start), how invested, holdings overlap,
    trades, and flags for gaps worth a look."""
    today = today or datetime.now(ET).date()
    # Read-only: never upsert strategies here (the Lab page fires this next to /strategies, whose
    # sync inserts new config entries -- two concurrent inserts hit the unique name, Oct-12).
    rows = {x.kind: x for x in db.query(LabStrategy).filter(LabStrategy.kind.in_(MIRROR_KINDS)).all() if x.is_active}
    model, ai = rows.get("mirror_shadow"), rows.get("mirror_ai_portfolio")
    if model is None or ai is None:
        return {"status": "not_configured"}
    out = {"status": "ok", "as_of": None, "books": {}, "flags": []}
    books = {}
    for key, s in (("model", model), ("ai", ai)):
        ms = _marks(db, s)
        if not ms:
            out["status"] = "waiting"
            out["books"][key] = {"label": s.label, "marks": 0}
            continue
        last = ms[-1]
        ws = _week_start(ms, today)
        wk, wk_spy = _ret(ms, ws)
        tot, tot_spy = _ret(ms)
        invested = (last.positions_value or 0) / last.equity * 100 if last.equity else 0
        week_orders = db.query(LabOrder).filter(LabOrder.strategy_id == s.id,
                                                LabOrder.date >= today - timedelta(days=today.weekday())).all()
        stops = sum(1 for o in week_orders if o.side == "sell" and "STOP" in (o.reason or "").upper())
        holdings = sorted(p["symbol"] for p in (last.positions or []) if p.get("symbol") != "SPY" or not p.get("note"))
        books[key] = {"label": s.label, "marks": len(ms), "as_of": last.date.isoformat(), "equity": round(last.equity, 2),
                      "week_pct": None if wk is None else round(wk, 2), "since_start_pct": None if tot is None else round(tot, 2),
                      "spy_week_pct": None if wk_spy is None else round(wk_spy, 2),
                      "spy_since_start_pct": None if tot_spy is None else round(tot_spy, 2),
                      "invested_pct": round(invested, 1), "holdings": holdings,
                      "week_buys": sum(1 for o in week_orders if o.side == "buy"),
                      "week_sells": sum(1 for o in week_orders if o.side == "sell"), "week_stops": stops,
                      "best": sorted(((p["symbol"], p.get("unrealized_plpc") or 0) for p in (last.positions or [])),
                                     key=lambda x: -x[1])[:3]}
        out["as_of"] = max(out["as_of"] or "", books[key]["as_of"])
    out["books"].update(books)
    if len(books) < 2:
        return out
    m, a = books["model"], books["ai"]
    both = sorted(set(m["holdings"]) & set(a["holdings"]))
    out["overlap"] = {"both": both, "model_only": sorted(set(m["holdings"]) - set(a["holdings"])),
                      "ai_only": sorted(set(a["holdings"]) - set(m["holdings"]))}
    try:
        from backend.blend_trader import a5_exposure, small_cap_brake
        exp, src = a5_exposure(db, today)
        out["market"] = {"a5_exposure": exp, "source": src, "small_cap_brake": small_cap_brake()}
    except Exception as e:
        out["market"] = {"error": str(e)[:200]}
    flags = out["flags"]
    if m["week_pct"] is not None and a["week_pct"] is not None and abs(m["week_pct"] - a["week_pct"]) >= FLAG_RETURN_GAP_PP:
        lead = "model" if m["week_pct"] > a["week_pct"] else "AI Portfolio"
        flags.append(f"Return gap this week: model {m['week_pct']:+.1f}% vs AI Portfolio {a['week_pct']:+.1f}% "
                     f"({lead} ahead by {abs(m['week_pct'] - a['week_pct']):.1f} pts)")
    if abs(m["invested_pct"] - a["invested_pct"]) >= FLAG_INVESTED_GAP_PP:
        flags.append(f"Invested gap: model {m['invested_pct']:.0f}% in stocks vs AI Portfolio {a['invested_pct']:.0f}%"
                     + (f" (market switch at {out['market'].get('a5_exposure')}/2)" if "a5_exposure" in out.get("market", {}) else ""))
    if m["holdings"] and a["holdings"] and not both:
        flags.append("No stocks in common: the two books are making completely different bets")
    for key, nm in (("model", "Model"), ("ai", "AI Portfolio")):
        if books[key]["week_stops"] >= 2:
            flags.append(f"{nm} hit {books[key]['week_stops']} stop-losses this week")
    if (out.get("market") or {}).get("small_cap_brake"):
        flags.append(f"Small-cap brake is ON for the model ({out['market']['small_cap_brake']}): no new stocks under $2B")
    return out


def review_text(rv: dict) -> tuple:
    """(title, body) for the push."""
    if rv.get("status") != "ok":
        return "Weekly review: waiting for data", "The model and the AI Portfolio need at least two daily marks each."
    m, a = rv["books"]["model"], rv["books"]["ai"]
    fmt = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    title = f"Weekly review: model {fmt(m['week_pct'])} vs AI Portfolio {fmt(a['week_pct'])} (SPY {fmt(m['spy_week_pct'])})"
    body = (f"Since start: model {fmt(m['since_start_pct'])}, AI Portfolio {fmt(a['since_start_pct'])}, "
            f"SPY {fmt(m['spy_since_start_pct'])}. In common: {', '.join(rv['overlap']['both']) or 'none'}. ")
    body += ("Gaps: " + " | ".join(rv["flags"])) if rv["flags"] else "No notable gaps this week."
    return title, body[:900]


def run_weekly_review_job(notify=None):
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        rv = weekly_review(db)
        if rv.get("status") == "not_configured":
            return
        title, body = review_text(rv)
        if notify is None:
            from backend.email_utils import create_notification as notify
        notify(user_id=OWNER_ID, kind="lab_weekly_review", title=title, body=body, priority="default",
               data={"url": "/lab"})
        logger.info(f"lab weekly review pushed: {title}")
    except Exception as e:
        logger.error(f"lab weekly review failed: {e}")
    finally:
        db.close()
