"""AI trader engine `blend`: the evidence-weighted model (docs/blend-model.md, owner Oct-12).

Not a backtested strategy: each part is weighted by how much evidence stands behind it -- the
2016-26 point-in-time research AND the live Jul-Oct signal check / AI Portfolio record agree on it.
A strategy profile with `engine: blend` replaces ONLY the buy decision (evaluate_buys) and the
score positions carry (update_position_prices / shadow synthetic positions). Exits, stops,
trailing stops, partial profits, pyramids, execution, cash and the circuit breaker stay on the
shared code paths, so it trades exactly like the AI Portfolio except for WHAT it buys and WHEN.

  market switch  A5 exposure (S&P > 200-day avg; S&P 12-month return > T-bills):
                 2 -> up to 10 stocks, 1 -> up to 5, 0 -> no new stocks (idle cash: SPY sweep
                 while exposure >= 1, profile spy_sweep.gate = a5)
  score          weighted CANSLIM 2.0 signals, percentile within segment (core / small):
                 beat streak 17.5 + surprise 17.5 (C 35%), analyst coverage 20 (I), low days-to-
                 cover 20, ROE 15 (A), buybacks 10 (S); breakouts / new highs / relative strength /
                 volume / the classic total score get 0 (no or negative evidence in both sources)
  buy            score >= 90th pct of its segment, best first; no stock > 15% above its 50-day
                 average; <= 3 per sector; <= 3 small caps ($250M-$1B, >= $1M/day), none of them
                 (or anything under $2B) while IWM < its 50-day average; equal dollars =
                 portfolio value / max_positions; no re-buy within 5 days of a sell
"""
import logging
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional

from sqlalchemy import func

logger = logging.getLogger(__name__)

ENGINE = "blend"
WEIGHTS = {"beat_streak": 0.175, "surprise_pct": 0.175, "n_brokers": 0.20, "dtc": 0.20, "roe": 0.15, "s3": 0.10}
SIGNS = {"dtc": -1}
DEFAULTS = {
    "buy_pct": 90.0,
    "regime_positions": {2: 10, 1: 5, 0: 0},
    "sector_max": 3,
    "small_max": 3,
    "small_min_mcap": 2.5e8,
    "small_min_dvol": 1e6,
    "small_brake_mcap": 2e9,
    "max_extension_pct": 15.0,
    "rebuy_cooldown_days": 5,
    "unscored_score": 1.0,       # a holding that left both universes reads as the weakest score -> score exit
}
A5_STRATEGY = "a5_dual_momentum"


def is_engine(profile: Optional[dict]) -> bool:
    return (profile or {}).get("engine") == ENGINE


def rules(profile: dict) -> dict:
    r = {**DEFAULTS, **((profile or {}).get("blend") or {})}
    r["regime_positions"] = {int(k): int(v) for k, v in r["regime_positions"].items()}
    return r


# ----------------------------------------------------------------- score

def _pct_rank(values: dict) -> dict:
    from backend.canslim2 import _pct_rank as pr
    return pr(values)


def score_rows(rows: list) -> dict:
    """rows: [{ticker, beat_streak, surprise_pct, roe, s3, dtc, n_brokers}] of ONE segment ->
    {ticker: (score, pct 0-100)}. Missing signal = median (0 after centering), as CANSLIM 2.0."""
    if not rows:
        return {}
    cent = {f: {t: p - 0.5 for t, p in _pct_rank({r["ticker"]: r.get(f) for r in rows}).items()} for f in WEIGHTS}
    raw = {r["ticker"]: sum(w * SIGNS.get(f, 1) * cent[f].get(r["ticker"], 0.0) for f, w in WEIGHTS.items())
           for r in rows}
    pct = _pct_rank(raw)
    return {t: (raw[t], round(pct[t] * 100, 1)) for t in raw}


_cache = {"key": None, "val": {}}


def blend_scores(db) -> dict:
    """{ticker: {score, pct, segment, market_cap, dvol20, letters}} from the latest CANSLIM 2.0 run.
    Recomputed only when a new run lands (keyed on its date + scored_at)."""
    from backend.database import Canslim2Score
    d = db.query(func.max(Canslim2Score.date)).scalar()
    if d is None:
        return {}
    stamp = db.query(func.max(Canslim2Score.scored_at)).filter(Canslim2Score.date == d).scalar()
    if _cache["key"] == (d, stamp):
        return _cache["val"]
    rows = db.query(Canslim2Score.ticker, Canslim2Score.segment, Canslim2Score.market_cap, Canslim2Score.inputs,
                    Canslim2Score.c_pct, Canslim2Score.a_pct, Canslim2Score.s_pct, Canslim2Score.i_pct
                    ).filter(Canslim2Score.date == d).all()
    by_seg, meta = {}, {}
    for t, seg, cap, inp, c, a, s, i in rows:
        seg = seg or "core"
        inp = inp or {}
        by_seg.setdefault(seg, []).append({"ticker": t, **{f: inp.get(f) for f in WEIGHTS}})
        meta[t] = {"segment": seg, "market_cap": cap, "dvol20": inp.get("dvol20"),
                   "letters": {"C": c, "A": a, "S": s, "I": i}}
    out = {}
    for seg, rs in by_seg.items():
        for t, (sc, pct) in score_rows(rs).items():
            out[t] = {"score": sc, "pct": pct, **meta[t]}
    _cache["key"], _cache["val"] = (d, stamp), out
    return out


def position_scores(db) -> dict:
    """{ticker: pct} for the score positions carry (sell/pyramid checks read it)."""
    return {t: v["pct"] for t, v in blend_scores(db).items()}


# ----------------------------------------------------------------- market switch

_a5_fallback = {"day": None, "exposure": None}


def a5_exposure(db, today: Optional[date] = None) -> tuple:
    """(exposure 0/1/2, source text). Reads the Lab's latest A5 decision (computed each session
    from the prior close); falls back to computing the rule from FMP once per day; if both fail,
    1 (mixed) -- never more risk than the data supports."""
    from backend.database import LabDecision, LabStrategy
    today = today or date.today()
    try:
        dec = (db.query(LabDecision).join(LabStrategy, LabStrategy.id == LabDecision.strategy_id)
               .filter(LabStrategy.name == A5_STRATEGY, LabDecision.status != "error",
                       LabDecision.date >= today - timedelta(days=6))
               .order_by(LabDecision.date.desc()).first())
        exp = (dec.inputs or {}).get("exposure") if dec is not None else None
        if exp is not None:
            return int(exp), f"A5 decision {dec.date}"
    except Exception as e:
        logger.warning(f"blend: A5 decision lookup failed: {e}")
    if _a5_fallback["day"] == today and _a5_fallback["exposure"] is not None:
        return _a5_fallback["exposure"], "A5 computed (cached)"
    try:
        from backend import lab
        cfg = lab.lab_config().get(A5_STRATEGY) or {"index": "^GSPC", "sma_days": 200, "mom_days": 252}
        _, inputs = lab.a5_target(lab.fmp_daily(cfg.get("index", "^GSPC")), today, cfg,
                                  lab.fmp_daily("SPY", adjusted=True), lab.fmp_daily(cfg.get("cash_proxy", "BIL"), adjusted=True))
        _a5_fallback.update(day=today, exposure=int(inputs["exposure"]))
        return _a5_fallback["exposure"], "A5 computed"
    except Exception as e:
        logger.warning(f"blend: A5 fallback failed ({e}); using mixed (1)")
        return 1, "A5 unavailable -> mixed"


def small_cap_brake(market_data: Optional[dict] = None) -> Optional[str]:
    """Why small caps are blocked this cycle (IWM below its 50-day average), else None. Fails open."""
    from backend.trading_utils import small_cap_gate_block
    if market_data is None:
        try:
            from data_fetcher import get_cached_market_direction
            market_data = get_cached_market_direction() or {}
        except Exception:
            market_data = {}
    return small_cap_gate_block({"small_cap_gate": {"enabled": True, "index": "IWM"}}, market_data)


# ----------------------------------------------------------------- buys

def blend_buys(db, user_id: int, profile: dict, portfolio: dict, positions: list, funnel=None,
               exposure: Optional[int] = None, brake: Optional[str] = "auto", today: Optional[date] = None) -> list:
    """Buy decisions in evaluate_buys' shape ({stock, value, reason, effective_score, signal_factors}).
    `stock` is a proxy whose canslim_score is the blend percentile, so trades and positions record
    the score the engine traded on."""
    from backend.database import AIPortfolioTrade, Stock
    r = rules(profile)
    held = {p.ticker for p in positions}
    if exposure is None:
        exposure, src = a5_exposure(db, today)
    else:
        src = "given"
    cap_n = r["regime_positions"].get(int(exposure), 0)
    room = cap_n - len(held)
    if room <= 0:
        if funnel is not None:
            funnel.note("blend_regime", f"A5 exposure {exposure} ({src}): {len(held)}/{cap_n} stocks")
        return []
    if brake == "auto":
        brake = small_cap_brake()
    scores = blend_scores(db)
    if not scores:
        logger.warning("blend engine: no CANSLIM 2.0 scores yet -- no buys this cycle")
        return []
    # Entity query + Python predicates: a shadow session serves AIPortfolioTrade from the stack's
    # own trade log and ignores SQL filters (shadow_trader._ShadowQuery); live, the SQL filter works.
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=r["rebuy_cooldown_days"])
    recent_sells = set()
    for tr in db.query(AIPortfolioTrade).filter(
            AIPortfolioTrade.user_id == user_id, AIPortfolioTrade.action == "SELL",
            AIPortfolioTrade.executed_at >= since).all():
        at = getattr(tr, "executed_at", None)
        if at is not None and at.tzinfo is not None:
            at = at.astimezone(timezone.utc).replace(tzinfo=None)
        if getattr(tr, "action", None) == "SELL" and at is not None and at >= since:
            recent_sells.add(tr.ticker)
    cands = sorted(((t, v) for t, v in scores.items() if v["pct"] >= r["buy_pct"]),
                   key=lambda kv: (-kv[1]["pct"], -kv[1]["score"]))
    stocks = {s.ticker: s for s in db.query(Stock).filter(Stock.ticker.in_([t for t, _ in cands] + list(held))).all()}
    per_sector = Counter((getattr(stocks.get(t), "sector", None) or "Unknown") for t in held)
    n_small = sum(1 for t in held if (scores.get(t) or {}).get("segment") == "small")
    slot = (portfolio.get("total_value") or 0) / max(1, int(profile.get("max_positions", 10)))
    skip = (lambda t, why: funnel.exec_skip(t, why)) if funnel is not None else (lambda t, why: None)
    out = []
    for t, v in cands:
        if len(out) >= room:
            break
        st = stocks.get(t)
        if t in held or st is None or not st.current_price or st.current_price <= 0:
            continue
        if t in recent_sells:
            skip(t, f"sold within {r['rebuy_cooldown_days']} days")
            continue
        small = v["segment"] == "small"
        cap = v.get("market_cap") or st.market_cap or 0
        if small and (cap < r["small_min_mcap"] or (v.get("dvol20") or 0) < r["small_min_dvol"]):
            continue                                   # outside the band the score was confirmed on
        if small and n_small >= r["small_max"]:
            skip(t, f"small-cap limit {r['small_max']}")
            continue
        if brake and cap < r["small_brake_mcap"]:
            skip(t, f"small-cap brake: {brake}")
            continue
        ma50 = getattr(st, "ma_50", None)
        if ma50 and ma50 > 0 and st.current_price > ma50 * (1 + r["max_extension_pct"] / 100):
            skip(t, f"stretched {st.current_price / ma50 - 1:+.0%} vs 50-day avg (max +{r['max_extension_pct']:.0f}%)")
            continue
        sector = st.sector or "Unknown"
        if per_sector[sector] >= r["sector_max"]:
            skip(t, f"sector limit {r['sector_max']} ({sector})")
            continue
        per_sector[sector] += 1
        n_small += int(small)
        L = v["letters"]
        proxy = SimpleNamespace(ticker=t, current_price=st.current_price, canslim_score=v["pct"],
                                growth_mode_score=None, is_growth_stock=False, sector=st.sector)
        out.append({"stock": proxy, "value": slot, "effective_score": v["pct"], "is_growth_stock": False,
                    "reason": f"MODEL top {max(1, math.ceil(100 - v['pct']))}%{' small cap' if small else ''}: score {v['pct']:.1f} "
                              f"(C {L['C'] or 0:.0f} · I {L['I'] or 0:.0f} · S {L['S'] or 0:.0f} · A {L['A'] or 0:.0f}) "
                              f"| market {exposure}/2",
                    "signal_factors": {"engine": ENGINE, "blend_pct": v["pct"], "segment": v["segment"],
                                       "a5_exposure": exposure, "letters": L}})
    if funnel is not None:
        funnel.note("blend_regime", f"A5 exposure {exposure} ({src}): room {room}, {len(out)} buys"
                    + (f"; small-cap brake ON ({brake})" if brake else ""))
    return out


def sweep_allowed(db) -> bool:
    """SPY sweep gate for spy_sweep.gate = a5: park idle cash while exposure >= 1."""
    return a5_exposure(db)[0] >= 1
