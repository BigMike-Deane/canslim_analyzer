"""CANSLIM 2.0: the CANSLIM letters rebuilt from what held up in point-in-time testing.

Research record: docs/score-v3-plan.md. The v3 scoreboard (2016-2026, ~1,800 companies)
rebuilt every letter several ways; the walk-forward rule kept six signals (v5b, 2026
selection), and the v8 letter-variant batch (Oct-10, trial 10) FAILED, so nothing was
added. Out of sample the tilt made about +0.7%/yr over SPY total return (6 of 8 years),
below a best-of-N luck bar: this is the best-evidence ranking available, not a proven edge.

    C  beat_streak   consecutive quarters beating the EPS estimate        (+)
    C  surprise_pct  latest EPS surprise %, clamped +-200                 (+)
    A  roe           return on equity                                     (+)
    S  s3            buybacks: -log(diluted shares now / ~1 year earlier) (+)
    S  dtc           days to cover (FINRA short interest)                 (-)
    I  n_brokers     distinct brokers with a rating action, prior 365 days (+)
    N, L             no signal in any form tested (52-week high, breakouts, pivots,
                     3/6/12-1 month momentum, industry strength) -> not scored
    M                market direction = the Lab's A1/A5 exposure rules

Score = mean of the six signed, centered percentile ranks (missing -> median), as in
research/pit/v4_model.score. Universe = the research's: price > $5, market cap >= $1B,
20-day dollar volume >= $5M. The model portfolio (Lab strategy `canslim2_tilt`) holds the
500 largest at weight ~ cap x (2 x score percentile among them), rebalanced every 20
sessions with 19 bps per unit of one-way turnover, marked daily on dividend-adjusted
closes -- the research's exact portfolio rule, simulated (no broker).
"""
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import requests

from backend.database import (Canslim2Input, Canslim2Score, LabDecision, LabEquityMark, LabStrategy, Stock,
                              StockDataCache)

logger = logging.getLogger(__name__)

FMP = "https://financialmodelingprep.com/stable"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FEATURES = (("beat_streak", 1, "C"), ("surprise_pct", 1, "C"), ("roe", 1, "A"),
            ("s3", 1, "S"), ("dtc", -1, "S"), ("n_brokers", 1, "I"))
LETTERS = ("C", "A", "S", "I")
MIN_PRICE, MIN_MCAP, MIN_DVOL = 5.0, 1e9, 5e6
TILT_TOP, REBALANCE_SESSIONS, COST = 500, 20, 0.0019
STRATEGY = "canslim2_tilt"
FMP_PAUSE = 0.25            # ~240 calls/min, well under the plan limit, leaves room for the scanner

EXPLAIN = {
    "beat_streak": "Quarters in a row the company beat analysts' EPS estimate",
    "surprise_pct": "How far the latest EPS beat (or missed) the estimate",
    "roe": "Return on equity: profit per dollar of shareholder capital",
    "s3": "Buybacks: share count shrinking over the past year",
    "dtc": "Days to cover: how crowded the short sellers are (lower is better)",
    "n_brokers": "How many brokers actively cover the stock",
}
EVIDENCE = {   # walk-forward selection t-stats for 2026 (docs/score-v3-plan.md, v8 run)
    "beat_streak": 4.1, "surprise_pct": 3.3, "roe": 2.9, "s3": 2.8, "dtc": -4.3, "n_brokers": 2.9,
}


# ----------------------------------------------------------------- input fetchers (pure where possible)

def s3_from_income(rows: list, today: date) -> Optional[dict]:
    """Buyback signal from FMP quarterly income statements (newest first, split-adjusted
    share counts): -log(diluted shares latest / same quarter ~1y earlier)."""
    q = []
    for r in rows or []:
        try:
            d = date.fromisoformat(str(r.get("date"))[:10])
        except ValueError:
            continue
        sh = r.get("weightedAverageShsOutDil") or r.get("weightedAverageShsOut")
        if sh and sh > 0:
            q.append((d, float(sh)))
    q.sort(reverse=True)
    if not q or q[0][0] < today - timedelta(days=200):
        return None
    end0, now = q[0]
    then = next((sh for d, sh in q[1:] if end0 - timedelta(days=380) <= d <= end0 - timedelta(days=350)), None)
    if not then:
        return None
    return {"s3": -math.log(now / then), "shares_now": now, "shares_then": then, "shares_asof": end0}


def n_brokers_from_grades(rows: list, today: date) -> int:
    """Distinct brokers with a rating action dated strictly before today, within 365 days."""
    lo = today - timedelta(days=365)
    seen = set()
    for r in rows or []:
        try:
            d = date.fromisoformat(str(r.get("date"))[:10])
        except ValueError:
            continue
        b = (r.get("gradingCompany") or "").strip().lower()
        if b and lo <= d < today:
            seen.add(b)
    return len(seen)


def latest_dtc(rows: list, today: date) -> dict:
    """{symbol: (dtc, settle)} from FINRA rows, using only settlements published by today
    (settlement + 12 days, as in research/pit/v3_short_interest.py); 999.99 = 'no volume'."""
    out = {}
    for r in rows or []:
        try:
            settle = date.fromisoformat(str(r.get("settlementDate"))[:10])
        except ValueError:
            continue
        sym, dtc = r.get("symbolCode"), r.get("daysToCoverQuantity")
        if not sym or dtc is None or settle + timedelta(days=12) > today:
            continue
        if sym not in out or settle > out[sym][1]:
            out[sym] = (float(dtc) if float(dtc) < 999 else None, settle)
    return out


def fetch_finra(today: date, session=None) -> list:
    """FINRA consolidated short interest for settlements in the last ~5 weeks (free API)."""
    http = session or requests
    rows, offset = [], 0
    body = {"limit": 5000, "fields": ["symbolCode", "settlementDate", "daysToCoverQuantity"],
            "dateRangeFilters": [{"fieldName": "settlementDate",
                                  "startDate": (today - timedelta(days=40)).isoformat(),
                                  "endDate": (today - timedelta(days=12)).isoformat()}]}
    while True:
        r = http.post(FINRA_URL, json={**body, "offset": offset}, headers={"Accept": "application/json"}, timeout=120)
        r.raise_for_status()
        page = r.json() if r.text.strip() else []
        rows += page
        if len(page) < 5000:
            return rows
        offset += 5000
        time.sleep(0.3)


def _fmp_get(path: str, **params):
    """GET an FMP stable endpoint -> parsed JSON, None on failure (never logs the key)."""
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        return None
    for attempt in range(3):
        try:
            r = requests.get(f"{FMP}/{path}", params={**params, "apikey": key}, timeout=20)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2)
    return None


# ----------------------------------------------------------------- universe + weekly refresh

def candidates(db) -> list:
    """Tickers that pass the price / market-cap screen (dollar volume is checked from bars)."""
    rows = db.query(Stock.ticker).filter(Stock.current_price > MIN_PRICE, Stock.market_cap >= MIN_MCAP).all()
    return sorted(t for (t,) in rows if t)


def refresh_inputs(db, today: Optional[date] = None, tickers: Optional[list] = None,
                   fmp: Callable = _fmp_get, finra_rows: Optional[list] = None,
                   bars_fn: Optional[Callable] = None, pause: float = FMP_PAUSE) -> dict:
    """Weekly: dollar volume (Alpaca), days to cover (FINRA, one bulk pull), and per ticker
    buybacks + analyst coverage (two FMP calls). Upserts Canslim2Input; returns counts."""
    from backend.alpaca_data import daily_bars_multi
    today = today or date.today()
    tickers = tickers if tickers is not None else candidates(db)
    bars = (bars_fn or daily_bars_multi)(tickers, today - timedelta(days=40), adjustment="raw")
    try:
        dtc = latest_dtc(finra_rows if finra_rows is not None else fetch_finra(today), today)
    except Exception as e:
        logger.warning(f"canslim2: FINRA fetch failed ({type(e).__name__}); days-to-cover left as before")
        dtc = None
    n = {"tickers": len(tickers), "s3": 0, "n_brokers": 0, "dtc": 0, "dvol": 0}
    existing = {r.ticker: r for r in db.query(Canslim2Input).filter(Canslim2Input.ticker.in_(tickers)).all()}
    for i, t in enumerate(tickers):
        row = existing.get(t) or Canslim2Input(ticker=t)
        b = sorted((bars.get(t) or {}).items())[-20:]
        if len(b) >= 15:
            row.dvol20 = sum(v["c"] * v["v"] for _, v in b) / len(b)
            n["dvol"] += 1
        if dtc is not None:
            d = dtc.get(t.replace("-", ".")) or dtc.get(t)
            row.dtc, row.dtc_settle = (d if d else (None, None))
            n["dtc"] += bool(d and d[0] is not None)
        inc = fmp("income-statement", symbol=t, period="quarter", limit=6)
        if isinstance(inc, list):
            s = s3_from_income(inc, today)
            row.s3, row.shares_now, row.shares_then, row.shares_asof = (
                (s["s3"], s["shares_now"], s["shares_then"], s["shares_asof"]) if s else (None, None, None, None))
            n["s3"] += bool(s)
        g = fmp("grades", symbol=t, limit=1000)
        if isinstance(g, list):
            row.n_brokers = n_brokers_from_grades(g, today)
            n["n_brokers"] += 1
        row.fetched_at = datetime.now(timezone.utc)
        db.add(row)
        if i % 100 == 99:
            db.commit()
        if pause:
            time.sleep(pause)
    db.commit()
    logger.info(f"canslim2: inputs refreshed {n}")
    return n


# ----------------------------------------------------------------- scoring

def _pct_rank(values: dict) -> dict:
    """{key: percentile in (0, 1]} with average ranks for ties; None values stay out."""
    items = sorted((v, k) for k, v in values.items() if v is not None and not (isinstance(v, float) and math.isnan(v)))
    out, i, n = {}, 0, len(items)
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][0] == items[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[items[k][1]] = r / n
        i = j + 1
    return out


def score_universe(rows: list) -> list:
    """rows: [{ticker, market_cap, beat_streak, surprise_pct, roe, s3, dtc, n_brokers}] already in
    the universe. Returns the rows with score, score_pct, rank, letter pcts, in_tilt, tilt_mult."""
    if not rows:
        return []
    cent = {}
    for f, _, _ in FEATURES:
        pr = _pct_rank({r["ticker"]: r.get(f) for r in rows})
        cent[f] = {t: p - 0.5 for t, p in pr.items()}
    letter_raw = {L: {} for L in LETTERS}
    for r in rows:
        t = r["ticker"]
        signed = {f: sg * cent[f].get(t, 0.0) for f, sg, _ in FEATURES}      # missing -> median (0)
        r["score"] = sum(signed.values()) / len(FEATURES)
        for L in LETTERS:
            fs = [signed[f] for f, _, l in FEATURES if l == L]
            letter_raw[L][t] = sum(fs) / len(fs)
    sp = _pct_rank({r["ticker"]: r["score"] for r in rows})
    lp = {L: _pct_rank(letter_raw[L]) for L in LETTERS}
    order = sorted(rows, key=lambda r: -r["score"])
    for i, r in enumerate(order, 1):
        r["rank"] = i
    for r in rows:
        t = r["ticker"]
        r["score_pct"] = round(sp[t] * 100, 1)
        for L in LETTERS:
            r[f"{L.lower()}_pct"] = round(lp[L][t] * 100, 1)
    big = sorted(rows, key=lambda r: -(r.get("market_cap") or 0))[:TILT_TOP]
    bp = _pct_rank({r["ticker"]: r["score"] for r in big})
    for r in rows:
        r["in_tilt"] = r["ticker"] in bp
        r["tilt_mult"] = round(2 * bp[r["ticker"]], 4) if r["in_tilt"] else None
    return rows


def universe_rows(db) -> list:
    """Universe members with their raw signals (StockDataCache + Canslim2Input)."""
    q = (db.query(Stock.ticker, Stock.market_cap, Stock.current_price, StockDataCache.earnings_beat_streak,
                  StockDataCache.latest_surprise_pct, StockDataCache.roe, Canslim2Input.s3, Canslim2Input.dtc,
                  Canslim2Input.n_brokers, Canslim2Input.dvol20)
         .join(Canslim2Input, Canslim2Input.ticker == Stock.ticker)
         .outerjoin(StockDataCache, StockDataCache.ticker == Stock.ticker)
         .filter(Stock.current_price > MIN_PRICE, Stock.market_cap >= MIN_MCAP, Canslim2Input.dvol20 >= MIN_DVOL))
    out = []
    for t, cap, px, bs, sp, roe, s3, dtc, nb, dv in q.all():
        out.append({"ticker": t, "market_cap": cap, "price": px,
                    "beat_streak": bs or 0, "surprise_pct": sp or 0.0,     # research default: no report -> 0
                    # the FMP fetch stores a MISSING returnOnEquity as 0 ("or 0"); research treats
                    # missing as NaN -> median, so an exact 0 counts as missing (143 of 2,376 on Oct-10)
                    "roe": roe if roe else None,
                    "s3": s3, "dtc": dtc, "n_brokers": nb if nb is not None else 0, "dvol20": dv})
    return out


def compute_scores(db, today: Optional[date] = None) -> int:
    """Score today's universe and replace today's Canslim2Score rows. Returns the row count."""
    today = today or date.today()
    rows = score_universe(universe_rows(db))
    if not rows:
        logger.warning("canslim2: empty universe (inputs not refreshed yet?)")
        return 0
    db.query(Canslim2Score).filter(Canslim2Score.date == today).delete(synchronize_session=False)
    for r in rows:
        db.add(Canslim2Score(
            date=today, ticker=r["ticker"], score=r["score"], score_pct=r["score_pct"], rank=r["rank"],
            c_pct=r["c_pct"], a_pct=r["a_pct"], s_pct=r["s_pct"], i_pct=r["i_pct"], market_cap=r["market_cap"],
            in_tilt=r["in_tilt"], tilt_mult=r["tilt_mult"],
            inputs={f: r.get(f) for f, _, _ in FEATURES} | {"dvol20": r.get("dvol20")}))
    db.commit()
    logger.info(f"canslim2: scored {len(rows)} stocks for {today}")
    return len(rows)


def tilt_weights(db, day: date) -> dict:
    """{ticker: weight} for the model portfolio from `day`'s scores (sums to 1)."""
    rows = db.query(Canslim2Score.ticker, Canslim2Score.market_cap, Canslim2Score.tilt_mult).filter(
        Canslim2Score.date == day, Canslim2Score.in_tilt.is_(True)).all()
    raw = {t: (cap or 0) * (m or 0) for t, cap, m in rows}
    tot = sum(raw.values())
    return {t: w / tot for t, w in raw.items() if w > 0} if tot > 0 else {}


# ----------------------------------------------------------------- model portfolio (Lab, simulated)

def mark_model(db, s: LabStrategy, today: date, closes_fn: Callable, spy_closes: list, spy_adj: list,
               weights_fn: Callable = tilt_weights) -> Optional[LabEquityMark]:
    """Idempotent per day. First call starts at starting_value; later calls grow each holding by
    its dividend-adjusted close-to-close return since the previous mark; every REBALANCE_SESSIONS
    marks the book moves to the latest tilt weights, paying COST per unit of one-way turnover."""
    from backend.lab import chain_spy_adj
    if db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == today).first():
        return None
    prev = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date < today)
            .order_by(LabEquityMark.date.desc()).first())
    last_dec = (db.query(LabDecision).filter(LabDecision.strategy_id == s.id, LabDecision.date < today)
                .order_by(LabDecision.date.desc()).first())
    values, note = {}, None
    if prev is not None:
        held = {p["symbol"]: p["market_value"] for p in (prev.positions or [])}
        px = closes_fn(sorted(held), prev.date - timedelta(days=7))
        missing = [t for t in held if not ((px.get(t) or {}).get(today) and (px.get(t) or {}).get(prev.date))]
        if len(missing) > 0.2 * max(len(held), 1):
            logger.error(f"canslim2: {len(missing)}/{len(held)} holdings lack closes for {today}; mark skipped")
            return None
        for t, v in held.items():
            c = px.get(t) or {}
            values[t] = v * (c[today] / c[prev.date]) if t not in missing else v   # no bar -> flat (research: r20 fillna(0))
        if missing:
            note = f"{len(missing)} holdings without a close carried flat: {', '.join(sorted(missing)[:8])}"
    equity = sum(values.values()) if prev is not None else float(s.starting_value or 25000)
    sessions = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id,
                                               LabEquityMark.date > last_dec.date).count() + 1) if last_dec else None
    if prev is None or last_dec is None or sessions >= REBALANCE_SESSIONS:
        target = weights_fn(db, today)
        if not target:
            logger.error(f"canslim2: no tilt weights for {today}; mark skipped")
            return None
        old = {t: v / equity for t, v in values.items()} if values else {}
        turnover = 0.5 * sum(abs(target.get(t, 0) - old.get(t, 0)) for t in set(target) | set(old)) if old else 1.0
        cost = turnover * COST * equity
        equity -= cost
        values = {t: w * equity for t, w in target.items()}
        top = sorted(target.items(), key=lambda kv: -kv[1])[:10]
        db.add(LabDecision(strategy_id=s.id, date=today, status="simulated", target=target, note=note, inputs={
            "kind": "rebalance", "names": len(target), "turnover": round(turnover, 4), "cost": round(cost, 2),
            "sessions_since_last": sessions, "top_weights": [[t, round(w, 4)] for t, w in top]}))
        if s.activated_at is None:
            s.activated_at = datetime.now(timezone.utc)
    spy_c = dict(spy_closes or []).get(today)
    mark = LabEquityMark(strategy_id=s.id, date=today, equity=equity, cash=0.0, positions_value=equity,
                         positions=[{"symbol": t, "weight": round(v / equity, 6), "market_value": round(v, 2)}
                                    for t, v in sorted(values.items(), key=lambda kv: -kv[1])],
                         spy_close=spy_c, spy_adj_close=chain_spy_adj(db, s.id, today, spy_adj))
    db.add(mark)
    db.commit()
    return mark


def _adjusted_closes(tickers, start) -> dict:
    from backend.alpaca_data import daily_bars_multi
    return {t: {d: v["c"] for d, v in b.items()} for t, b in daily_bars_multi(tickers, start, adjustment="all").items()}


# ----------------------------------------------------------------- scheduler entry points

def run_refresh_job():
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        refresh_inputs(db)
        compute_scores(db)
    except Exception as e:
        db.rollback()
        logger.error(f"canslim2 refresh failed: {type(e).__name__}: {str(e)[:200]}")
    finally:
        db.close()


def run_daily_job():
    """After the close: score, then mark the model portfolio (trading days only)."""
    from backend.ai_trader import EASTERN_TZ, is_trading_day
    from backend.database import SessionLocal
    from backend.lab import fmp_daily, safe_error, sync_strategies
    now = datetime.now(EASTERN_TZ)
    if not is_trading_day(now):
        return
    today = now.date()
    db = SessionLocal()
    try:
        if not db.query(Canslim2Score).filter(Canslim2Score.date == today).first():
            compute_scores(db, today)
        s = next((x for x in sync_strategies(db) if x.kind == STRATEGY and x.is_active), None)
        if s is not None:
            mark_model(db, s, today, _adjusted_closes, fmp_daily("SPY", days=10), fmp_daily("SPY", days=40, adjusted=True))
    except Exception as e:
        db.rollback()
        logger.error(f"canslim2 daily job failed: {safe_error(e)}")
    finally:
        db.close()
