"""Daily closing equity for shadow stacks (2026-09-23).

Shadow equity was only ever marked "now" (shadow_trader.shadow_stack_equity),
so no arm had a daily return series. The Oct-21 readout rules
(docs/oct21-readout-rules.md) need one: a chop arm only counts as a win if it
beats its comparator ON CHOP DAYS (the mechanism check). This module:

  * replay_book       -- the trade-log cash/share ledger as of a cutoff, the
                         same ledger ShadowSession._derive_synthetic_positions
                         keeps (pinned by test at the end of the log);
  * fill_shadow_equity_marks -- writes one ShadowEquityMark per stack per
                         finished NYSE session: history backfills on first
                         run, each new session after the close;
  * regime_excess_vs  -- arm-minus-comparator daily excess split into trend /
                         chop days (SPY > 1.5% above its 50MA = trend, the
                         go-live gate's definition).

Prices are RAW consolidated-tape closes (alpaca_data.daily_closes_raw): share
counts come from the trade log, which only changes at a SPLIT row, so
split-adjusted history would misprice every pre-split day. A trade belongs to
the first session close at or after it: after-hours shadow fills count toward
the next session.
"""

import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
# Alpaca's free plan serves SIP bars 16 minutes delayed: today's daily bar is
# complete only after ~16:16 ET. Before that, today is not marked.
MARK_READY_ET = (16, 17)
TREND_THRESHOLD_PCT = 1.5


def _naive_utc(dt):
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def session_close_utc(d: date) -> datetime:
    """16:00 ET on `d` as naive UTC (the DB's executed_at convention)."""
    return datetime.combine(d, time(16, 0), ET).astimezone(timezone.utc).replace(tzinfo=None)


def replay_book(trades, starting_cash: float, cutoff_utc=None) -> dict:
    """Cash, SPY-sweep shares and open shares per ticker after every trade
    executed strictly before `cutoff_utc` (naive UTC; None = whole log).
    `trades` must be sorted by executed_at."""
    from backend.shadow_trader import is_sweep_reason
    cash = float(starting_cash)
    sweep = 0.0
    shares = defaultdict(float)
    last_px = {}
    for t in trades:
        if cutoff_utc is not None and _naive_utc(t.executed_at) >= cutoff_utc:
            break
        q, px = float(t.shares or 0), float(t.price or 0)
        if is_sweep_reason(t.reason):
            if t.action == "BUY":
                cash -= q * px
                sweep += q
            elif t.action == "SELL":
                cash += q * px
                sweep = max(0.0, sweep - q)
            continue
        if t.action == "SPLIT":
            factor = t.signal_factors.get("split_factor") if isinstance(t.signal_factors, dict) else None
            factor = factor or t.shares
            if factor and factor > 0:
                shares[t.ticker] *= factor
                if t.ticker in last_px:
                    last_px[t.ticker] /= factor
            continue
        if t.action in ("BUY", "PYRAMID"):
            shares[t.ticker] += q
            cash -= q * px
        elif t.action == "SELL":
            cash += q * px
            shares[t.ticker] = max(0.0, shares[t.ticker] - q)
        last_px[t.ticker] = px
    return {"cash": cash, "sweep_shares": sweep,
            "shares": {k: v for k, v in shares.items() if v > 1e-9},
            "last_px": last_px}


def _close_on_or_before(series: dict, d: date):
    if not series:
        return None, False
    if d in series:
        return series[d], True
    earlier = [x for x in series if x <= d]
    return (series[max(earlier)], False) if earlier else (None, False)


def mark_for_day(trades, starting_cash: float, d: date, closes: dict) -> dict:
    """Closing equity on session `d`. A holding without a close ON `d`
    (halted, unknown to the feed) carries its last earlier close, else its
    last trade price, and counts as unpriced."""
    st = replay_book(trades, starting_cash, session_close_utc(d))
    positions_value, unpriced = 0.0, 0
    for tk, sh in st["shares"].items():
        px, exact = _close_on_or_before(closes.get(tk) or {}, d)
        if px is None:
            px = st["last_px"].get(tk, 0.0)
        if not exact:
            unpriced += 1
        positions_value += sh * px
    sweep_value = 0.0
    if st["sweep_shares"] > 1e-9:
        spy, _ = _close_on_or_before(closes.get("SPY") or {}, d)
        sweep_value = st["sweep_shares"] * (spy or 0.0)
    return {"equity": st["cash"] + positions_value + sweep_value, "cash": st["cash"],
            "positions_value": positions_value, "sweep_value": sweep_value,
            "n_positions": len(st["shares"]), "unpriced_positions": unpriced}


def session_days(start: date, end: date) -> list:
    from backend.ai_trader import is_trading_day
    out, d = [], start
    while d <= end:
        if is_trading_day(datetime(d.year, d.month, d.day, 12, tzinfo=ET)):
            out.append(d)
        d += timedelta(days=1)
    return out


def last_markable_day(now_utc: datetime) -> date:
    """Today once its bar is complete (>= MARK_READY_ET), else yesterday."""
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    return today if (now_et.hour, now_et.minute) >= MARK_READY_ET else today - timedelta(days=1)


REPRICE_SESSIONS = 5


def fill_shadow_equity_marks(db, now_utc: datetime | None = None, price_fn=None) -> int:
    """Insert every missing ShadowEquityMark for active stacks, from each
    stack's first session through the last markable day. Idempotent: existing
    rows are never rewritten. `price_fn(tickers, start) -> {tk: {date: close}}`
    is injectable; by default Alpaca raw closes. Returns rows written."""
    from backend.database import ShadowStrategy, ShadowTrade, ShadowEquityMark

    now_utc = now_utc or datetime.now(timezone.utc)
    through = last_markable_day(now_utc)
    # Recent sessions whose mark lacked a holding's close are re-tried until
    # the bar lands (Sep-23: marked at 16:20 ET before one holding's bar was
    # in, carried at the prior close for good). Past this window a missing
    # close is real (halt, delisting) and the mark stands, flagged unpriced.
    recent = set(session_days(through - timedelta(days=14), through)[-REPRICE_SESSIONS:])
    todo = []
    for s in db.query(ShadowStrategy).filter(ShadowStrategy.archived_at.is_(None)).all():
        act_utc = _naive_utc(s.activated_at)
        if act_utc is None:
            continue
        act = act_utc.replace(tzinfo=timezone.utc).astimezone(ET)
        first = act.date() if act.hour < 16 else act.date() + timedelta(days=1)
        rows = {r.date: r for r in db.query(ShadowEquityMark).filter(
            ShadowEquityMark.shadow_strategy_id == s.id)}
        have = {d for d, r in rows.items() if not (r.unpriced_positions and d in recent)}
        missing = [d for d in session_days(first, through) if d not in have]
        if missing:
            trades = db.query(ShadowTrade).filter(
                ShadowTrade.shadow_strategy_id == s.id).order_by(ShadowTrade.executed_at).all()
            todo.append((s, missing, trades, rows))
    if not todo:
        return 0

    tickers = {t.ticker for _, _, trades, _ in todo for t in trades if t.ticker}
    tickers.add("SPY")
    start = min(m[0] for _, m, _, _ in todo) - timedelta(days=10)
    if price_fn is None:
        from backend.alpaca_data import daily_closes_raw
        price_fn = daily_closes_raw
    closes = price_fn(sorted(tickers), start) or {}
    if not closes:
        logger.warning("Shadow equity marks: no closes available (Alpaca keys/feed?) -- nothing written")
        return 0

    # A session the feed doesn't have yet (no SPY close ON it) is left for a
    # later pass rather than written permanently at the prior day's prices.
    have_feed = set((closes.get("SPY") or {}).keys())
    written = 0
    for s, missing, trades, rows in todo:
        start_cash = float(s.starting_value or 25000.0)
        for d in missing:
            if d not in have_feed:
                continue
            m = mark_for_day(trades, start_cash, d, closes)
            if m["unpriced_positions"] and d in recent:
                continue                      # a holding's bar isn't in yet
            row = rows.get(d)
            if row is None:
                db.add(ShadowEquityMark(shadow_strategy_id=s.id, date=d, **m))
            else:
                for k, v in m.items():
                    setattr(row, k, v)
            written += 1
        db.commit()
    logger.info(f"Shadow equity marks: wrote {written} rows for {len(todo)} stacks through {through}")
    return written


def regime_excess_vs(db, arm_id: int, comparator_id: int,
                     trend_threshold_pct: float = TREND_THRESHOLD_PCT) -> dict:
    """Arm-minus-comparator daily return, split by regime, over the sessions
    both stacks have marks for. The mechanism check: a chop lever must win on
    chop days. Days classify on SPY's distance to its 50MA on the return day
    (market_snapshots; weekend/holiday rows are never session days here)."""
    from backend.database import ShadowEquityMark, MarketSnapshot

    def _series(sid):
        return {r.date: r.equity for r in db.query(ShadowEquityMark).filter(
            ShadowEquityMark.shadow_strategy_id == sid)}

    a, c = _series(arm_id), _series(comparator_id)
    days = sorted(set(a) & set(c))
    dist = {}
    if days:
        for ms in db.query(MarketSnapshot).filter(MarketSnapshot.date >= days[0],
                                                  MarketSnapshot.date <= days[-1]):
            if ms.spy_price and ms.spy_50_ma:
                dist[ms.date] = (ms.spy_price - ms.spy_50_ma) / ms.spy_50_ma * 100.0
    buckets = {"trend": [], "chop": []}
    for prev, d in zip(days, days[1:]):
        if not (a[prev] and c[prev]) or d not in dist:
            continue
        ex = (a[d] / a[prev] - 1.0) - (c[d] / c[prev] - 1.0)
        buckets["trend" if dist[d] > trend_threshold_pct else "chop"].append(ex)

    def _stats(xs):
        if not xs:
            return {"n_days": 0, "mean_excess_bps": None}
        return {"n_days": len(xs), "mean_excess_bps": round(sum(xs) / len(xs) * 1e4, 1)}

    return {"sessions": len(days), "trend": _stats(buckets["trend"]),
            "chop": _stats(buckets["chop"]), "threshold_pct": trend_threshold_pct}
