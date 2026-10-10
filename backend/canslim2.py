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
from sqlalchemy import func

from backend.database import (Canslim2Input, Canslim2Score, LabDecision, LabEquityMark, LabOrder, LabStrategy,
                              Stock, StockDataCache)

logger = logging.getLogger(__name__)

FMP = "https://financialmodelingprep.com/stable"
FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FEATURES = (("beat_streak", 1, "C"), ("surprise_pct", 1, "C"), ("roe", 1, "A"),
            ("s3", 1, "S"), ("dtc", -1, "S"), ("n_brokers", 1, "I"))
LETTERS = ("C", "A", "S", "I")
MIN_PRICE, MIN_MCAP, MIN_DVOL = 5.0, 1e9, 5e6
# Provisional "small" segment (Oct-10): price > $5, cap >= $100M, outside the core universe; ranked
# among themselves. Research the same day: the formula carries over to $250M-$1B (IC t 2.06, tilt
# +0.72%/yr at the 100th pct of random); untested below $250M. Never traded.
SMALL_MIN_MCAP, SMALL_CONFIRMED_MCAP, THIN_DVOL = 1e8, 2.5e8, 1e6


def segment_label(segment: str, market_cap, dvol20) -> Optional[str]:
    if segment != "small":
        return None
    base = ("Provisional: formula confirmed for $250M-$1B companies in 2016-26 testing"
            if (market_cap or 0) >= SMALL_CONFIRMED_MCAP else "Provisional: untested below $250M market cap")
    if (market_cap or 0) >= MIN_MCAP:
        base = "Provisional: under $5M/day traded, outside the tested universe"
    return base + ("; thinly traded (under $1M/day)" if dvol20 is not None and dvol20 < THIN_DVOL else "") + \
        ". Ranked among small caps, not traded by any portfolio."
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
    """Tickers whose inputs the weekly refresh keeps: core candidates (cap >= $1B; dollar volume is
    checked from bars) and the provisional small segment (cap >= $100M)."""
    rows = db.query(Stock.ticker).filter(Stock.current_price > MIN_PRICE, Stock.market_cap >= SMALL_MIN_MCAP).all()
    return sorted(t for (t,) in rows if t)


def refresh_inputs(db, today: Optional[date] = None, tickers: Optional[list] = None,
                   fmp: Callable = _fmp_get, finra_rows: Optional[list] = None,
                   bars_fn: Optional[Callable] = None, pause: float = FMP_PAUSE) -> dict:
    """Weekly: dollar volume (Alpaca), days to cover (FINRA, one bulk pull), and per ticker
    buybacks + analyst coverage (two FMP calls). Upserts Canslim2Input; returns counts."""
    from backend.alpaca_data import daily_bars_multi
    today = today or _et_today()
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


def score_universe(rows: list, tilt: bool = True) -> list:
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
    big = sorted(rows, key=lambda r: -(r.get("market_cap") or 0))[:TILT_TOP] if tilt else []
    bp = _pct_rank({r["ticker"]: r["score"] for r in big})
    for r in rows:
        r["in_tilt"] = r["ticker"] in bp
        r["tilt_mult"] = round(2 * bp[r["ticker"]], 4) if r["in_tilt"] else None
    return rows


def universe_rows(db, segment: str = "core") -> list:
    """Members of a segment with their raw signals (StockDataCache + Canslim2Input).
    core: price > $5, cap >= $1B, 20d $ volume >= $5M (the tested universe).
    small: price > $5, cap >= $100M, not core (provisional)."""
    from sqlalchemy import or_
    q = (db.query(Stock.ticker, Stock.market_cap, Stock.current_price, StockDataCache.earnings_beat_streak,
                  StockDataCache.latest_surprise_pct, StockDataCache.roe, Canslim2Input.s3, Canslim2Input.dtc,
                  Canslim2Input.n_brokers, Canslim2Input.dvol20)
         .join(Canslim2Input, Canslim2Input.ticker == Stock.ticker)
         .outerjoin(StockDataCache, StockDataCache.ticker == Stock.ticker)
         .filter(Stock.current_price > MIN_PRICE))
    if segment == "core":
        q = q.filter(Stock.market_cap >= MIN_MCAP, Canslim2Input.dvol20 >= MIN_DVOL)
    else:
        q = q.filter(Stock.market_cap >= SMALL_MIN_MCAP,
                     or_(Stock.market_cap < MIN_MCAP, Canslim2Input.dvol20 < MIN_DVOL, Canslim2Input.dvol20.is_(None)))
    out = []
    for t, cap, px, bs, sp, roe, s3, dtc, nb, dv in q.all():
        out.append({"ticker": t, "market_cap": cap, "price": px,
                    "beat_streak": bs or 0, "surprise_pct": sp or 0.0,     # research default: no report -> 0
                    # the FMP fetch stores a MISSING returnOnEquity as 0 ("or 0"); research treats
                    # missing as NaN -> median, so an exact 0 counts as missing (143 of 2,376 on Oct-10)
                    "roe": roe if roe else None,
                    "s3": s3, "dtc": dtc, "n_brokers": nb if nb is not None else 0, "dvol20": dv})
    return out


def _et_today() -> date:
    """The ET session date (the container clock is UTC: an evening scan must not stamp tomorrow)."""
    from backend.ai_trader import EASTERN_TZ
    return datetime.now(EASTERN_TZ).date()


def compute_scores(db, today: Optional[date] = None) -> int:
    """Score today's universe and replace today's Canslim2Score rows (re-run after every scan).
    Each row carries the previous scored date's percentile. Returns the row count."""
    today = today or _et_today()
    rows = [dict(r, segment="core") for r in score_universe(universe_rows(db, "core"))]
    small = [dict(r, segment="small") for r in score_universe(universe_rows(db, "small"), tilt=False)] if rows else []
    if not rows:
        logger.warning("canslim2: empty universe (inputs not refreshed yet?)")
        return 0
    prev_day = db.query(func.max(Canslim2Score.date)).filter(Canslim2Score.date < today).scalar()
    prev = dict(db.query(Canslim2Score.ticker, Canslim2Score.score_pct).filter(Canslim2Score.date == prev_day).all()) if prev_day else {}
    now = datetime.now(timezone.utc)
    db.query(Canslim2Score).filter(Canslim2Score.date == today).delete(synchronize_session=False)
    for r in rows + small:
        db.add(Canslim2Score(
            date=today, ticker=r["ticker"], segment=r["segment"], score=r["score"], score_pct=r["score_pct"], rank=r["rank"],
            c_pct=r["c_pct"], a_pct=r["a_pct"], s_pct=r["s_pct"], i_pct=r["i_pct"], market_cap=r["market_cap"],
            in_tilt=r["in_tilt"], tilt_mult=r["tilt_mult"], prev_score_pct=prev.get(r["ticker"]), scored_at=now,
            inputs={f: r.get(f) for f, _, _ in FEATURES} | {"dvol20": r.get("dvol20")}))
    db.commit()
    logger.info(f"canslim2: scored {len(rows)} stocks for {today} (+{len(small)} provisional small caps)")
    return len(rows) + len(small)


def core_only():
    """SQL filter: the tested universe only. EVERY trading path uses it (Lab tilt + picks, AI engine)."""
    from sqlalchemy import or_
    return or_(Canslim2Score.segment == "core", Canslim2Score.segment.is_(None))


def latest_map(db, tickers) -> dict:
    """{ticker: compact CANSLIM 2.0 summary} from the latest scored date (Screener / Watchlist rows)."""
    d = db.query(func.max(Canslim2Score.date)).scalar()
    if d is None or not tickers:
        return {}
    out = {}
    for r in db.query(Canslim2Score).filter(Canslim2Score.date == d, Canslim2Score.ticker.in_(list(tickers))).all():
        chg = (r.score_pct - r.prev_score_pct) if r.score_pct is not None and r.prev_score_pct is not None else None
        seg = r.segment or "core"
        out[r.ticker] = {"score_pct": r.score_pct, "rank": r.rank, "change": round(chg, 1) if chg is not None else None,
                         "segment": seg, "label": segment_label(seg, r.market_cap, (r.inputs or {}).get("dvol20")),
                         "letters": {"C": r.c_pct, "A": r.a_pct, "S": r.s_pct, "I": r.i_pct}}
    return out


def tilt_weights(db, day: date) -> dict:
    """{ticker: weight} for the model portfolio from `day`'s scores (sums to 1)."""
    rows = db.query(Canslim2Score.ticker, Canslim2Score.market_cap, Canslim2Score.tilt_mult).filter(
        Canslim2Score.date == day, Canslim2Score.in_tilt.is_(True), core_only()).all()
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


# ----------------------------------------------------------------- 20-stock picks (Lab, simulated)
# Pre-registered in docs/canslim2-forward-plan.md (2026-10-10, before any trade). Changing any of
# these makes a NEW strategy; never edit them in place.
PICKS = "canslim2_picks"
PICKS_N, PICKS_BUY_PCT, PICKS_SELL_PCT, PICKS_STOP, PICKS_SECTOR_MAX, PICKS_COST = 20, 90.0, 70.0, -0.15, 5, 0.001


def picks_universe(db, day: date) -> list:
    """Today's scored universe, best first: [{ticker, score_pct, sector}]."""
    rows = (db.query(Canslim2Score.ticker, Canslim2Score.score_pct, Stock.sector)
            .outerjoin(Stock, Stock.ticker == Canslim2Score.ticker)
            .filter(Canslim2Score.date == day, core_only()).order_by(Canslim2Score.score.desc()).all())
    return [{"ticker": t, "score_pct": p, "sector": sec or "Unknown"} for t, p, sec in rows]


def mark_picks(db, s: LabStrategy, today: date, closes_fn: Callable, raw_fn: Callable, spy_closes: list,
               spy_adj: list, scores_fn: Callable = picks_universe) -> Optional[LabEquityMark]:
    """Idempotent per day. Grow holdings on dividend-adjusted closes; sell below the 70th
    percentile, on leaving the universe, or at -15% vs cost; then fill up to 20 holdings from the
    top 10% (max 5 per sector, equal dollars = equity / 20). Trades are recorded as filled
    LabOrders at the raw close (so the Lab's order list and win rate work)."""
    from collections import Counter
    from backend.lab import chain_spy_adj
    if db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date == today).first():
        return None
    scores = scores_fn(db, today)
    if not scores:
        logger.error(f"canslim2 picks: no scores for {today}; mark skipped")
        return None
    prev = (db.query(LabEquityMark).filter(LabEquityMark.strategy_id == s.id, LabEquityMark.date < today)
            .order_by(LabEquityMark.date.desc()).first())
    pos = {p["symbol"]: dict(p) for p in (prev.positions or [])} if prev else {}
    cash = float(prev.cash or 0.0) if prev else float(s.starting_value or 25000)
    note = None
    if pos:
        px = closes_fn(sorted(pos), prev.date - timedelta(days=7))
        missing = [t for t in pos if not ((px.get(t) or {}).get(today) and (px.get(t) or {}).get(prev.date))]
        if len(missing) > 0.2 * len(pos):
            logger.error(f"canslim2 picks: {len(missing)}/{len(pos)} holdings lack closes for {today}; mark skipped")
            return None
        for t, p in pos.items():
            if t not in missing:
                c = px[t]
                p["market_value"] = p["market_value"] * c[today] / c[prev.date]
        if missing:
            note = f"{len(missing)} holdings without a close carried flat: {', '.join(sorted(missing))}"
    by = {r["ticker"]: r for r in scores}
    sells, buys = [], []
    for t in sorted(pos):
        p, r = pos[t], by.get(t)
        pnl = p["market_value"] / p["cost"] - 1
        reason = ("left the universe" if r is None else "stop: -15% from cost" if pnl <= PICKS_STOP
                  else f"score fell to {r['score_pct']:.0f}" if r["score_pct"] < PICKS_SELL_PCT else None)
        if reason:
            proceeds = p["market_value"] * (1 - PICKS_COST)
            cash += proceeds
            sells.append({"ticker": t, "reason": reason, "pnl": proceeds / p["cost"] - 1, "qty": p.get("qty")})
            del pos[t]
    equity = cash + sum(p["market_value"] for p in pos.values())
    per_sector = Counter(p.get("sector") for p in pos.values())
    sold = {x["ticker"] for x in sells}
    slot = equity / PICKS_N
    for r in scores:                                   # best first
        if len(pos) >= PICKS_N or r["score_pct"] is None or r["score_pct"] < PICKS_BUY_PCT:
            break
        t = r["ticker"]
        if t in pos or t in sold or per_sector[r["sector"]] >= PICKS_SECTOR_MAX:
            continue
        amt = min(cash, slot)
        if amt < 0.25 * slot:                          # no dust positions when cash runs out
            break
        cash -= amt
        pos[t] = {"symbol": t, "market_value": amt * (1 - PICKS_COST), "cost": amt, "entry_date": today.isoformat(),
                  "sector": r["sector"], "entry_score": r["score_pct"]}
        per_sector[r["sector"]] += 1
        buys.append({"ticker": t, "amount": amt, "score_pct": r["score_pct"]})
    if buys or sells:
        raw = raw_fn(sorted({x["ticker"] for x in buys + sells}), today - timedelta(days=7))
        now = datetime.now(timezone.utc)
        for x in buys:
            price = (raw.get(x["ticker"]) or {}).get(today)
            qty = x["amount"] / price if price else None
            pos[x["ticker"]]["qty"] = qty
            db.add(LabOrder(strategy_id=s.id, date=today, symbol=x["ticker"], side="buy", qty=qty or 0.0,
                            client_order_id=f"sim-{s.name}-{today:%Y%m%d}-{x['ticker']}-buy", status="filled",
                            filled_qty=qty, filled_avg_price=price, submitted_at=now, filled_at=now,
                            reason=f"score {x['score_pct']:.0f} (top 10%)"))
        for x in sells:
            price = (raw.get(x["ticker"]) or {}).get(today)
            db.add(LabOrder(strategy_id=s.id, date=today, symbol=x["ticker"], side="sell", qty=x["qty"] or 0.0,
                            client_order_id=f"sim-{s.name}-{today:%Y%m%d}-{x['ticker']}-sell", status="filled",
                            filled_qty=x["qty"], filled_avg_price=price, submitted_at=now, filled_at=now,
                            reason=f"{x['reason']} ({x['pnl'] * 100:+.1f}%)"))
    equity = cash + sum(p["market_value"] for p in pos.values())
    if buys or sells or prev is None:
        db.add(LabDecision(strategy_id=s.id, date=today, status="simulated", note=note,
                           target={t: round(p["market_value"] / equity, 4) for t, p in pos.items()},
                           inputs={"kind": "trades", "holdings": len(pos), "cash": round(cash, 2),
                                   "buys": [[x["ticker"], round(x["amount"], 2), x["score_pct"]] for x in buys],
                                   "sells": [[x["ticker"], x["reason"], round(x["pnl"], 4)] for x in sells]}))
        if s.activated_at is None:
            s.activated_at = datetime.now(timezone.utc)
    positions = sorted(({**p, "market_value": round(p["market_value"], 2), "weight": round(p["market_value"] / equity, 6),
                         "unrealized_plpc": round(p["market_value"] / p["cost"] - 1, 4)} for p in pos.values()),
                       key=lambda p: -p["market_value"])
    mark = LabEquityMark(strategy_id=s.id, date=today, equity=equity, cash=cash,
                         positions_value=equity - cash, positions=positions,
                         spy_close=dict(spy_closes or []).get(today), spy_adj_close=chain_spy_adj(db, s.id, today, spy_adj))
    db.add(mark)
    db.commit()
    return mark


def _raw_closes(tickers, start) -> dict:
    from backend.alpaca_data import daily_bars_multi
    return {t: {d: v["c"] for d, v in b.items()} for t, b in daily_bars_multi(tickers, start, adjustment="raw").items()}


def _adjusted_closes(tickers, start) -> dict:
    from backend.alpaca_data import daily_bars_multi
    return {t: {d: v["c"] for d, v in b.items()} for t, b in daily_bars_multi(tickers, start, adjustment="all").items()}


# ----------------------------------------------------------------- score-move alerts

MOVE_ALERT_POINTS = 20      # percentile points between two scoring runs


def latest_pcts(db) -> dict:
    d = db.query(func.max(Canslim2Score.date)).scalar()
    return dict(db.query(Canslim2Score.ticker, Canslim2Score.score_pct).filter(Canslim2Score.date == d).all()) if d else {}


def score_move_alerts(db, before: dict, notify: Optional[Callable] = None) -> int:
    """Push each user once per ticker per day when a stock they hold (AI Portfolio) or watch moved
    >= MOVE_ALERT_POINTS percentile points since the previous scoring run. Returns pushes sent."""
    from backend.database import AIPortfolioPosition, Notification, Watchlist
    if not before:
        return 0
    d = db.query(func.max(Canslim2Score.date)).scalar()
    after = {r.ticker: r for r in db.query(Canslim2Score).filter(Canslim2Score.date == d).all()}
    moved = {t: r for t, r in after.items() if t in before and before[t] is not None and r.score_pct is not None
             and abs(r.score_pct - before[t]) >= MOVE_ALERT_POINTS}
    if not moved:
        return 0
    if notify is None:
        from backend.email_utils import create_notification as notify
    interest = {}
    for uid, t in (db.query(AIPortfolioPosition.user_id, AIPortfolioPosition.ticker).all()
                   + db.query(Watchlist.user_id, Watchlist.ticker).all()):
        if uid and t in moved:
            interest.setdefault(uid, set()).add(t)
    since = datetime.now(timezone.utc) - timedelta(hours=20)
    sent = 0
    for uid, tickers in interest.items():
        done = {(n.data or {}).get("ticker") for n in db.query(Notification).filter(
            Notification.user_id == uid, Notification.kind == "canslim2_move", Notification.created_at >= since).all()}
        for t in sorted(tickers - done):
            r, was = moved[t], before[t]
            up = r.score_pct > was
            notify(user_id=uid, kind="canslim2_move",
                   title=f"{t} CANSLIM 2.0 {'up' if up else 'down'}: {was:.0f} → {r.score_pct:.0f}",
                   body=f"C {r.c_pct:.0f} · A {r.a_pct:.0f} · S {r.s_pct:.0f} · I {r.i_pct:.0f}"
                        f" (beat streak {(r.inputs or {}).get('beat_streak')}, surprise {(r.inputs or {}).get('surprise_pct') or 0:+.1f}%)",
                   data={"ticker": t, "url": f"/stock/{t}", "from": was, "to": r.score_pct})
            sent += 1
    return sent


def run_after_scan():
    """Rescore on the latest scan's earnings data, then alert on big moves (held / watched)."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        before = latest_pcts(db)
        if compute_scores(db):
            n = score_move_alerts(db, before)
            if n:
                logger.info(f"canslim2: {n} score-move alerts")
    except Exception as e:
        db.rollback()
        logger.error(f"canslim2 rescore after scan failed: {type(e).__name__}: {str(e)[:200]}")
    finally:
        db.close()


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
    """After the close: score, then mark the tilt and the 20-stock picks (trading days only)."""
    from backend.ai_trader import EASTERN_TZ, is_trading_day
    from backend.database import SessionLocal
    from backend.lab import fmp_daily, safe_error, sync_strategies
    now = datetime.now(EASTERN_TZ)
    if not is_trading_day(now):
        return
    today = now.date()
    db = SessionLocal()
    try:
        compute_scores(db, today)          # fresh at the close: the trades use this run's scores
        strategies = {x.kind: x for x in sync_strategies(db) if x.is_active}
        spy_c, spy_a = fmp_daily("SPY", days=10), fmp_daily("SPY", days=40, adjusted=True)
    except Exception as e:
        db.rollback()
        logger.error(f"canslim2 daily job failed: {safe_error(e)}")
        db.close()
        return
    try:
        for kind, run in ((STRATEGY, lambda s: mark_model(db, s, today, _adjusted_closes, spy_c, spy_a)),
                          (PICKS, lambda s: mark_picks(db, s, today, _adjusted_closes, _raw_closes, spy_c, spy_a))):
            if kind in strategies:
                try:   # one strategy failing never blocks the other
                    run(strategies[kind])
                except Exception as e:
                    db.rollback()
                    logger.error(f"canslim2 {kind} mark failed: {safe_error(e)}")
    finally:
        db.close()
