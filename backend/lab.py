"""Lab: research strategies paper-traded live on their own Alpaca paper accounts.

Each strategy in config ``lab_strategies`` has its own Alpaca paper credentials
(env vars named in the config), so Lab orders never touch the AI Portfolio
mirror's account. Every session the engine:

1. ``decide_and_submit`` -- ~15 minutes before the close: computes the rule from
   data known at the PRIOR close, records a LabDecision, and places
   market-on-close orders (whole shares) to reach the target.
2. ``record_close`` -- after the close: refreshes order fills and writes a
   LabEquityMark from the account (equity, cash, positions) plus SPY's close.

Strategy kinds:
- ``a1_trend_2x`` (docs/exposure-plan.md, PASSED 1928-93 + 1994-2026): hold the
  2x S&P fund (SSO) when the S&P 500 closed above its 200-day simple average
  on the prior session, else a T-bill ETF (SGOV).
"""
import logging
import math
import re
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import requests

from backend.broker_mirror import AlpacaError, AlpacaPaperClient
from backend.database import LabDecision, LabEquityMark, LabOrder, LabStrategy

logger = logging.getLogger(__name__)

FMP = "https://financialmodelingprep.com/stable"
CASH_BUFFER = 0.02            # leave 2% unspent so a move into the close can't overdraw
MOC_LEAD_MINUTES = 15         # submit when the close is this near (Alpaca accepts cls until 10 min before)
_OPEN = ("submitted", "new", "accepted", "pending_new", "partially_filled", "held")


# ----------------------------------------------------------------- config / clients

def lab_config() -> dict:
    try:
        from config_loader import config
        return config.get("lab_strategies", {}) or {}
    except Exception:
        return {}


def sync_strategies(db) -> list:
    """Upsert LabStrategy rows from config (idempotent; never deletes history)."""
    rows = []
    for name, cfg in lab_config().items():
        row = db.query(LabStrategy).filter(LabStrategy.name == name).first()
        if row is None:
            row = LabStrategy(name=name)
            db.add(row)
        row.kind = cfg.get("kind", name)
        row.label = cfg.get("label", name)
        row.description = cfg.get("description")
        row.starting_value = float(cfg.get("starting_value", 25000))
        row.is_active = bool(cfg.get("enabled", True))
        rows.append(row)
    db.commit()
    return rows


def get_client(name: str) -> Optional[AlpacaPaperClient]:
    cfg = lab_config().get(name) or {}
    key = os.environ.get(cfg.get("key_env", ""), "").strip()
    secret = os.environ.get(cfg.get("secret_env", ""), "").strip()
    return AlpacaPaperClient(key, secret) if key and secret else None


_SECRET_RE = re.compile(r"(apikey|api_key|token|secret)=[^&\s'\"]+", re.I)


def safe_error(e: Exception) -> str:
    """Exception text safe to store/return: FMP errors embed the request URL, apikey included."""
    if isinstance(e, requests.HTTPError) and e.response is not None:
        return f"market data error: HTTP {e.response.status_code}"
    return _SECRET_RE.sub(r"\1=***", f"{type(e).__name__}: {e}")[:500]


# ----------------------------------------------------------------- market data (patchable)

def fmp_daily(symbol: str, days: int = 420, adjusted: bool = False) -> list:
    """[(date, close)] oldest -> newest from FMP; adjusted=True -> dividend-adjusted close."""
    key = os.environ.get("FMP_API_KEY", "")
    start = (date.today() - timedelta(days=days)).isoformat()
    path = "historical-price-eod/dividend-adjusted" if adjusted else "historical-price-eod/light"
    r = requests.get(f"{FMP}/{path}", params={"symbol": symbol, "from": start, "apikey": key}, timeout=20)
    r.raise_for_status()
    rows = r.json() or []
    col = "adjClose" if adjusted else "price"
    out = [(date.fromisoformat(x["date"]), float(x[col])) for x in rows if x.get(col) is not None]
    return sorted(out)


def fmp_price(symbol: str) -> Optional[float]:
    key = os.environ.get("FMP_API_KEY", "")
    r = requests.get(f"{FMP}/quote", params={"symbol": symbol, "apikey": key}, timeout=15)
    r.raise_for_status()
    q = (r.json() or [{}])[0]
    return float(q["price"]) if q.get("price") else None


# ----------------------------------------------------------------- rules

def a1_target(closes: list, today: date, cfg: dict) -> tuple:
    """A1: risk-on fund if the PRIOR session's S&P close > its N-day SMA, else the cash ETF.
    Uses only closes strictly before ``today`` (today's bar is not final)."""
    n = int(cfg.get("sma_days", 200))
    prior = [c for d, c in closes if d < today]
    if len(prior) < n:
        raise ValueError(f"need {n} prior closes, have {len(prior)}")
    sma = sum(prior[-n:]) / n
    above = prior[-1] > sma
    sym = cfg.get("risk_on", "SSO") if above else cfg.get("risk_off", "SGOV")
    return {sym: 1.0}, {"index": cfg.get("index", "^GSPC"), "index_close": round(prior[-1], 2),
                        "sma": round(sma, 2), "sma_days": n, "above": above}


def _ret_over(series: list, today: date, n: int) -> Optional[float]:
    prior = [c for d, c in series if d < today]
    return prior[-1] / prior[-1 - n] - 1 if len(prior) > n else None


def a5_target(closes: list, today: date, cfg: dict, spy_adj: list = None, cash_adj: list = None) -> tuple:
    """A5 graded dual momentum (docs/exposure-plan.md, PASSED 1928-93 + 1994-2026):
    signal 1 = prior S&P close > its 200-day SMA; signal 2 = S&P 12-month total return
    (SPY dividend-adjusted) > cash's 12-month return (T-bill ETF, dividend-adjusted).
    Both -> 2x fund, exactly one -> 1x S&P fund, neither -> T-bill ETF."""
    _, a1 = a1_target(closes, today, cfg)
    n = int(cfg.get("mom_days", 252))
    r_mkt, r_cash = _ret_over(spy_adj or [], today, n), _ret_over(cash_adj or [], today, n)
    if r_mkt is None or r_cash is None:
        raise ValueError("need 12 months of SPY and cash-ETF history")
    s1, s2 = bool(a1["above"]), r_mkt > r_cash
    exposure = int(s1) + int(s2)
    sym = {2: cfg.get("risk_on_2x", "SSO"), 1: cfg.get("risk_on_1x", "SPY"), 0: cfg.get("risk_off", "SGOV")}[exposure]
    return {sym: 1.0}, {**a1, "mkt_12m": round(r_mkt * 100, 2), "cash_12m": round(r_cash * 100, 2),
                        "momentum_positive": s2, "exposure": exposure}


RULES = {"a1_trend_2x": a1_target, "a5_dual_momentum": a5_target}


# ----------------------------------------------------------------- alerts

def _describe(target: dict, inputs: dict) -> str:
    sym = next(iter(target), "?")
    lev = {2: "2×", 1: "1×", 0: "cash"}.get(inputs.get("exposure"), "")
    return f"{sym}{f' ({lev})' if lev else ''}"


def notify_flip(db, strategy: LabStrategy, dec: LabDecision) -> bool:
    """Push + in-app notification when the target differs from the previous session's."""
    prev = (db.query(LabDecision).filter(LabDecision.strategy_id == strategy.id, LabDecision.date < dec.date,
                                         LabDecision.status != "error")
            .order_by(LabDecision.date.desc()).first())
    if prev is None or prev.target == dec.target:
        return False
    i = dec.inputs or {}
    body = (f"{i.get('index', 'S&P')} {i.get('index_close')} vs {i.get('sma_days')}-day avg {i.get('sma')} "
            f"({'above' if i.get('above') else 'below'})")
    if "momentum_positive" in i:
        body += f"; 12-month {i.get('mkt_12m')}% vs T-bills {i.get('cash_12m')}%"
    body += f". Was {_describe(prev.target, prev.inputs or {})}."
    if dec.status == "no_broker":
        body += " (Not traded: broker not connected.)"
    try:
        from backend.email_utils import create_notification
        cfg = lab_config().get(strategy.name, {})
        return bool(create_notification(
            user_id=int(cfg.get("notify_user_id", 1)), kind="lab_signal",
            title=f"Lab {strategy.label}: switching to {_describe(dec.target, i)}", body=body,
            data={"strategy": strategy.name, "date": dec.date.isoformat(), "target": dec.target, "url": "/lab"}))
    except Exception as e:
        logger.warning(f"lab[{strategy.name}] flip notification failed: {safe_error(e)}")
        return False


# ----------------------------------------------------------------- engine

def _positions(client) -> dict:
    return {p["symbol"]: float(p["qty"]) for p in client.positions()}


def plan_orders(target: dict, holdings: dict, equity: float, prices: dict) -> list:
    """Whole-share orders to move ``holdings`` to ``target`` weights. Sells first."""
    orders = []
    for sym, qty in holdings.items():
        want = target.get(sym, 0.0)
        if want == 0 and qty > 0:
            orders.append({"symbol": sym, "side": "sell", "qty": qty})
    for sym, w in target.items():
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        want_qty = math.floor(equity * w * (1 - CASH_BUFFER) / price)
        have = holdings.get(sym, 0.0)
        diff = want_qty - have
        # only trade when the gap is material (> 5% of the target) -- no daily dust trades
        if want_qty > 0 and abs(diff) / want_qty > 0.05:
            orders.append({"symbol": sym, "side": "buy" if diff > 0 else "sell", "qty": abs(diff)})
        elif want_qty == 0 and have > 0:
            orders.append({"symbol": sym, "side": "sell", "qty": have})
    return orders


def short_filled_order(db, strategy_id: int, today: date, symbol: str, side: str) -> Optional[LabOrder]:
    """The previous session's order for this fund and side, if it ended short (expired or
    canceled before every share filled). Alpaca paper short-fills closing orders at random
    (exposure-plan breach log, Oct-8/9), so the order that finishes it is a regular market
    order, which paper fills in full. Normal signal trades stay market-on-close."""
    prev_day = (db.query(LabOrder.date).filter(LabOrder.strategy_id == strategy_id, LabOrder.date < today)
                .order_by(LabOrder.date.desc()).limit(1).scalar())
    if prev_day is None:
        return None
    for o in db.query(LabOrder).filter(LabOrder.strategy_id == strategy_id, LabOrder.date == prev_day,
                                       LabOrder.symbol == symbol, LabOrder.side == side).all():
        if (o.status or "").lower() in ("expired", "canceled", "cancelled") and (o.filled_qty or 0) < (o.qty or 0):
            return o
    return None


def decide_and_submit(db, strategy: LabStrategy, client=None, today: Optional[date] = None,
                      closes=None, price_fn=None, extra: Optional[dict] = None) -> LabDecision:
    """Idempotent per session: a second call the same day returns the existing decision."""
    if today is None:
        from backend.ai_trader import EASTERN_TZ
        today = datetime.now(EASTERN_TZ).date()
    existing = db.query(LabDecision).filter(LabDecision.strategy_id == strategy.id,
                                            LabDecision.date == today).first()
    if existing is not None:
        return existing
    cfg = lab_config().get(strategy.name, {})
    rule = RULES[strategy.kind]
    closes = closes if closes is not None else fmp_daily(cfg.get("index", "^GSPC"))
    if strategy.kind == "a5_dual_momentum":
        extra = extra or {"spy_adj": fmp_daily("SPY", days=420, adjusted=True),
                          "cash_adj": fmp_daily(cfg.get("cash_proxy", "BIL"), days=420, adjusted=True)}
        target, inputs = rule(closes, today, cfg, **extra)
    else:
        target, inputs = rule(closes, today, cfg)
    dec = LabDecision(strategy_id=strategy.id, date=today, inputs=inputs, target=target)
    db.add(dec)
    if client is None:
        dec.status, dec.note = "no_broker", "Alpaca paper credentials not configured"
        db.commit()
        notify_flip(db, strategy, dec)
        return dec
    try:
        acct = client.account()
        equity = float(acct["equity"])
        holdings = _positions(client)
        price_fn = price_fn or fmp_price
        prices = {s: price_fn(s) for s in target}
        orders = plan_orders(target, holdings, equity, prices)
        if not orders:
            dec.status = "unchanged"
            db.commit()
            return dec
        db.flush()
        for o in orders:
            cid = f"lab-{strategy.name}-{today:%Y%m%d}-{o['symbol']}-{o['side']}-{uuid.uuid4().hex[:6]}"
            short = short_filled_order(db, strategy.id, today, o["symbol"], o["side"])
            row = LabOrder(strategy_id=strategy.id, decision_id=dec.id, date=today, symbol=o["symbol"],
                           side=o["side"], qty=o["qty"], client_order_id=cid,
                           reason=f"target {target} ({'above' if inputs.get('above') else 'below'} "
                                  f"{inputs.get('sma_days')}d avg"
                                  + (f", 12m momentum {'positive' if inputs.get('momentum_positive') else 'negative'}"
                                     if 'momentum_positive' in inputs else "") + ")"
                                  + (f"; market order finishing the {short.date.isoformat()} order the paper "
                                     f"broker short-filled" if short else ""))
            db.add(row)
            try:
                submit = client.submit_order if short else client.submit_moc_order
                res = submit(o["symbol"], o["qty"], o["side"], cid)
                row.broker_order_id, row.status = res.get("id"), res.get("status", "submitted")
            except AlpacaError as e:
                row.status, row.error = "error", str(e)
        dec.status = "submitted"
        if strategy.activated_at is None:
            strategy.activated_at = datetime.now(timezone.utc)
        db.commit()
        notify_flip(db, strategy, dec)
    except Exception as e:  # never let one strategy break the job
        db.rollback()
        dec = LabDecision(strategy_id=strategy.id, date=today, inputs=inputs, target=target,
                          status="error", note=safe_error(e))
        db.add(dec)
        db.commit()
        logger.error(f"lab[{strategy.name}] decide failed: {safe_error(e)}")
    return dec


def chain_spy_adj(db, strategy_id: int, today: date, spy_adj: list) -> Optional[float]:
    """SPY total-return level for today's mark, chain-linked to the previous mark.
    A dividend-adjusted series is back-adjusted: its LATEST point always equals the raw
    close, so storing each day's latest value silently drops every dividend. Instead:
    previous stored level x (adj today / adj on the previous mark's date), both read from
    the same fetch (same adjustment basis). First mark (or a gap the fetch can't span):
    today's adjusted close starts the chain."""
    adj = dict(spy_adj or [])
    if today not in adj:
        return None
    prev = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == strategy_id, LabEquityMark.date < today,
                                           LabEquityMark.spy_adj_close.isnot(None))
            .order_by(LabEquityMark.date.desc()).first())
    if prev is not None and adj.get(prev.date):
        return prev.spy_adj_close * adj[today] / adj[prev.date]
    if prev is not None:
        logger.warning(f"lab: SPY adjusted series lacks {prev.date}; restarting the TR chain at {today}")
    return adj[today]


def record_close(db, strategy: LabStrategy, client=None, today: Optional[date] = None,
                 spy_closes=None, spy_adj=None) -> Optional[LabEquityMark]:
    """Refresh order fills, then upsert today's LabEquityMark from the account."""
    if today is None:
        from backend.ai_trader import EASTERN_TZ
        today = datetime.now(EASTERN_TZ).date()
    if client is None:
        return None
    for row in db.query(LabOrder).filter(LabOrder.strategy_id == strategy.id, LabOrder.status.in_(_OPEN)).all():
        try:
            o = client.order_by_client_id(row.client_order_id)
            row.status = o.get("status", row.status)
            row.filled_qty = float(o["filled_qty"]) if o.get("filled_qty") else row.filled_qty
            row.filled_avg_price = float(o["filled_avg_price"]) if o.get("filled_avg_price") else row.filled_avg_price
            if o.get("filled_at"):
                row.filled_at = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
        except AlpacaError as e:
            row.error = str(e)
    acct = client.account()
    pos = client.positions()
    spy_closes = spy_closes if spy_closes is not None else fmp_daily("SPY", days=10)
    spy_adj = spy_adj if spy_adj is not None else fmp_daily("SPY", days=40, adjusted=True)
    spy_c = dict(spy_closes).get(today)
    spy_a = chain_spy_adj(db, strategy.id, today, spy_adj)
    mark = db.query(LabEquityMark).filter(LabEquityMark.strategy_id == strategy.id,
                                          LabEquityMark.date == today).first() or LabEquityMark(
        strategy_id=strategy.id, date=today, equity=0.0)
    mark.equity = float(acct["equity"])
    mark.cash = float(acct.get("cash") or 0)
    mark.positions_value = float(acct.get("long_market_value") or 0)
    mark.positions = [{"symbol": p["symbol"], "qty": float(p["qty"]), "market_value": float(p["market_value"]),
                       "avg_entry_price": float(p["avg_entry_price"]),
                       "unrealized_plpc": float(p.get("unrealized_plpc") or 0)} for p in pos]
    mark.spy_close, mark.spy_adj_close = spy_c, spy_a
    db.add(mark)
    db.commit()
    return mark


# ----------------------------------------------------------------- scheduler entry points

def _session_minutes_to_close(client) -> Optional[float]:
    c = client.clock()
    if not c.get("is_open"):
        return None
    nxt = datetime.fromisoformat(c["next_close"].replace("Z", "+00:00"))
    return (nxt - datetime.now(timezone.utc)).total_seconds() / 60


def run_decision_job():
    """Every 5 min in the afternoon (ET): when the close is <= MOC_LEAD_MINUTES away,
    decide + submit for each active strategy (idempotent per session)."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        for s in sync_strategies(db):
            if not s.is_active or s.kind not in RULES:   # simulated kinds (canslim2_tilt) run their own job
                continue
            client = get_client(s.name)
            if client is not None:
                mins = _session_minutes_to_close(client)
                if mins is None or mins > MOC_LEAD_MINUTES or mins < 10.5:
                    continue
            else:
                from backend.ai_trader import is_trading_day, EASTERN_TZ
                now = datetime.now(EASTERN_TZ)
                if not is_trading_day(now) or not (now.hour == 15 and 40 <= now.minute < 50):
                    continue
            from backend.ai_trader import EASTERN_TZ
            decide_and_submit(db, s, client, today=datetime.now(EASTERN_TZ).date())
    except Exception as e:
        logger.error(f"lab decision job failed: {safe_error(e)}")
    finally:
        db.close()


def cached_daily() -> Callable:
    """fmp_daily memoized for one job run: (symbol, adjusted) -> [(date, close)].
    Raw closes reach back ~11 years so fill-cost checks (C1) see every fill's close."""
    cache = {}

    def daily(symbol: str, adjusted: bool):
        if (symbol, adjusted) not in cache:
            cache[(symbol, adjusted)] = fmp_daily(symbol, days=420 if adjusted else 4000, adjusted=adjusted)
        return cache[(symbol, adjusted)]
    return daily


def run_close_job():
    from backend.ai_trader import EASTERN_TZ, is_trading_day
    from backend.database import SessionLocal
    from backend.lab_checks import _safe_gap, run_checks
    if not is_trading_day(datetime.now(EASTERN_TZ)):
        return
    db = SessionLocal()
    today = datetime.now(EASTERN_TZ).date()
    daily, gaps = cached_daily(), None
    try:
        for s in sync_strategies(db):
            client = get_client(s.name)
            if s.is_active and client is not None:
                try:
                    record_close(db, s, client, today=today)
                except Exception as e:
                    db.rollback()
                    logger.error(f"lab[{s.name}] record_close failed: {safe_error(e)}")
                    continue
                try:   # stop rules (docs/exposure-plan.md): never let a check failure break the marks
                    if gaps is None:
                        gaps = {n: _safe_gap(daily, n) for n in (126, 252)}
                    run_checks(db, s, today, daily, gaps)
                except Exception as e:
                    db.rollback()
                    logger.error(f"lab[{s.name}] stop-rule checks failed: {safe_error(e)}")
    finally:
        db.close()
