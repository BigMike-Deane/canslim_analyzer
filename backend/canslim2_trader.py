"""AI Portfolio engine `canslim2_picks`: the CANSLIM 2.0 picks rules, traded live.

Pre-registered in docs/canslim2-forward-plan.md ("Strategy 3", 2026-10-10). A strategy profile
with `engine: canslim2_picks` replaces the classic decision logic at five dispatch points in
backend/ai_trader.py (evaluate_sells, evaluate_buys, evaluate_pyramids, the stop-loss job,
update_position_prices) plus the broker mirror's resting stop and the Exit Plan card. Execution,
cash, the Alpaca paper mirror, snapshots and notifications stay on the shared code paths.

Rules (same as the Lab's close-only `canslim2_picks`):
  sell   score percentile < sell_pct (70) | left the universe | price <= cost x (1 - stop_pct)
  buy    up to max_positions (20) from percentile >= buy_pct (90), best first, <= sector_max (5)
         per sector, equal dollars = portfolio value / max_positions, no same-day re-buy
  none   trailing stops, take-profit, pyramids, SPY sweep, cash reserve, circuit breaker,
         buy throttle, seeds, ML veto, correction-zone rule

The switch from the classic champion happens by itself (apply_pivot_if_due, config
`canslim2_pivot`): real user portfolios only (user_id > 0; shadow arms run on negative sandbox
ids and are never touched).
"""
import logging
from collections import Counter
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Optional

from sqlalchemy import func

logger = logging.getLogger(__name__)

ENGINE = "canslim2_picks"
DEFAULTS = {"buy_pct": 90.0, "sell_pct": 70.0, "stop_pct": 15.0, "sector_max": 5}


def is_engine(profile: Optional[dict]) -> bool:
    return (profile or {}).get("engine") == ENGINE


def rules(profile: dict) -> dict:
    return {**DEFAULTS, **((profile or {}).get("canslim2") or {})}


def c2_pcts(db) -> dict:
    """{ticker: score percentile} from the latest CANSLIM 2.0 scoring run."""
    from backend.canslim2 import core_only
    from backend.database import Canslim2Score
    d = db.query(func.max(Canslim2Score.date)).scalar()
    if d is None:
        return {}
    return dict(db.query(Canslim2Score.ticker, Canslim2Score.score_pct).filter(Canslim2Score.date == d, core_only()).all())


def stop_price(cost_basis: float, profile: dict) -> float:
    return cost_basis * (1 - rules(profile)["stop_pct"] / 100.0)


# ----------------------------------------------------------------- decisions

def engine_sells(db, positions: list, profile: dict) -> list:
    """Sell decisions in evaluate_sells' shape ({position, reason, is_partial, sell_pct})."""
    r = rules(profile)
    pcts = c2_pcts(db)
    if not pcts:
        logger.warning("canslim2 engine: no CANSLIM 2.0 scores yet -- holding everything this cycle")
        return []
    out = []
    for p in positions:
        pct = pcts.get(p.ticker)
        price, cost = p.current_price, p.cost_basis
        if pct is None:
            reason = "CANSLIM 2.0: not scored (needs price > $5, market cap >= $1B, $5M/day volume)"
        elif price and cost and price <= stop_price(cost, profile):
            reason = f"CANSLIM 2.0 STOP: -{r['stop_pct']:.0f}% from cost (${cost:.2f} -> ${price:.2f})"
        elif pct < r["sell_pct"]:
            reason = f"CANSLIM 2.0: score fell to {pct:.0f} (sells below {r['sell_pct']:.0f})"
        else:
            continue
        out.append({"position": p, "reason": reason, "is_partial": False, "sell_pct": 100})
    return out


def engine_buys(db, user_id: int, profile: dict, portfolio: dict, positions: list, funnel=None) -> list:
    """Buy decisions in evaluate_buys' shape ({stock, value, reason, effective_score, signal_factors}).
    `stock` is a proxy whose canslim_score is the CANSLIM 2.0 percentile, so trades and new
    positions record the score the engine actually used."""
    from backend.canslim2 import core_only
    from backend.database import AIPortfolioTrade, Canslim2Score, Stock
    r = rules(profile)
    n_max = int(profile.get("max_positions", 20))
    d = db.query(func.max(Canslim2Score.date)).scalar()
    if d is None:
        return []
    held = {p.ticker for p in positions}
    since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    sold_today = {t for (t,) in db.query(AIPortfolioTrade.ticker).filter(
        AIPortfolioTrade.user_id == user_id, AIPortfolioTrade.action == "SELL",
        AIPortfolioTrade.executed_at >= since).all()}
    rows = (db.query(Canslim2Score, Stock).join(Stock, Stock.ticker == Canslim2Score.ticker)
            .filter(Canslim2Score.date == d, Canslim2Score.score_pct >= r["buy_pct"], core_only())
            .order_by(Canslim2Score.score.desc()).all())
    sectors = dict(db.query(Stock.ticker, Stock.sector).filter(Stock.ticker.in_(list(held))).all()) if held else {}
    per_sector = Counter((sectors.get(t) or "Unknown") for t in held)
    slot = (portfolio.get("total_value") or 0) / n_max
    out, room = [], n_max - len(held)
    for sc, st in rows:
        if len(out) >= room:
            break
        sector = st.sector or "Unknown"
        if st.ticker in held or st.ticker in sold_today:
            continue
        if per_sector[sector] >= r["sector_max"]:
            if funnel is not None:
                funnel.exec_skip(st.ticker, f"sector cap {r['sector_max']} ({sector})")
            continue
        if not st.current_price or st.current_price <= 0:
            continue
        per_sector[sector] += 1
        proxy = SimpleNamespace(ticker=st.ticker, current_price=st.current_price, canslim_score=sc.score_pct,
                                growth_mode_score=None, is_growth_stock=False, sector=st.sector)
        out.append({"stock": proxy, "value": slot, "effective_score": sc.score_pct, "is_growth_stock": False,
                    "reason": f"CANSLIM 2.0 top {100 - sc.score_pct:.0f}%: score {sc.score_pct:.0f} "
                              f"(C {sc.c_pct:.0f} · A {sc.a_pct:.0f} · S {sc.s_pct:.0f} · I {sc.i_pct:.0f})",
                    "signal_factors": {"engine": ENGINE, "c2_pct": sc.score_pct, "c2_rank": sc.rank,
                                       "c": sc.c_pct, "a": sc.a_pct, "s": sc.s_pct, "i": sc.i_pct}})
    return out


def engine_stop_check(db, user_id: int, profile: dict, config, positions: list) -> dict:
    """The intraday stop job for the engine: only the -stop_pct hard stop, market hours only.
    Executes like the classic job's full sells (execute_trade, cash, delete, snapshot)."""
    from backend.ai_trader import execute_trade, is_market_open, take_portfolio_snapshot
    if not is_market_open():
        return {"message": "market closed -- engine stops wait for the session", "sells_executed": []}
    r = rules(profile)
    sold = []
    for p in positions:
        price, cost = p.current_price, p.cost_basis
        if not price or not cost or price != price or price > stop_price(cost, profile):
            continue
        reason = f"CANSLIM 2.0 STOP: -{r['stop_pct']:.0f}% from cost (${cost:.2f} -> ${price:.2f})"
        held_days = (datetime.now(timezone.utc).date() - p.purchase_date.date()).days if p.purchase_date else None
        execute_trade(db=db, ticker=p.ticker, action="SELL", shares=p.shares, price=price, reason=reason,
                      score=p.current_score, cost_basis=cost, realized_gain=p.gain_loss,
                      signal_factors={"sell_reason": "STOP LOSS", "engine": ENGINE,
                                      "gain_pct": round((price / cost - 1) * 100, 1)},
                      user_id=user_id, holding_days=held_days)
        config.current_cash += p.current_value
        sold.append({"ticker": p.ticker, "shares": p.shares, "price": price, "gain_loss": p.gain_loss, "reason": reason})
        db.delete(p)
    db.commit()
    if sold:
        take_portfolio_snapshot(db, user_id=user_id)
    return {"message": f"Checked {len(positions)} positions, {len(sold)} stop losses triggered", "sells_executed": sold}


def exit_plan(cost_basis, current_price, current_score, profile: dict) -> dict:
    """Exit Plan card triggers for an engine position (compute_exit_plan's output shape)."""
    r = rules(profile)
    triggers = []
    if cost_basis and cost_basis > 0 and current_price and current_price > 0:
        sp = stop_price(cost_basis, profile)
        triggers.append({"kind": "stop_loss", "label": "Stop loss", "price": round(sp, 2), "direction": "down",
                         "distance_pct": round(abs((sp / current_price - 1) * 100), 1),
                         "note": f"{r['stop_pct']:.0f}% below entry · fixed (no trailing, no ATR widening)"})
    triggers.append({"kind": "score_exit", "label": "Score exit", "price": None, "direction": "score",
                     "distance_pct": None, "threshold": round(r["sell_pct"], 0),
                     "current_score": round(current_score, 0) if current_score is not None else None, "active": True,
                     "note": f"sells if the CANSLIM 2.0 score falls below {r['sell_pct']:.0f}"
                             + (f" (now {current_score:.0f})" if current_score is not None else "")})
    priced = [t for t in triggers if t.get("price") is not None]
    return {"triggers": triggers, "nearest_kind": priced[0]["kind"] if priced else None}


# ----------------------------------------------------------------- the switch (Oct-22)

def _pivot_cfg() -> dict:
    import os
    if os.environ.get("CANSLIM2_PIVOT_DISABLED") == "1":   # the test suite: no calendar time bombs
        return {}
    try:
        from config_loader import config
        return config.get("canslim2_pivot", {}) or {}
    except Exception:
        return {}


def pivot_active(today: Optional[date] = None) -> bool:
    cfg = _pivot_cfg()
    on = cfg.get("activate_on")
    if not on or not cfg.get("strategy"):
        return False
    if today is None:
        from backend.ai_trader import EASTERN_TZ
        today = datetime.now(EASTERN_TZ).date()
    return today >= date.fromisoformat(str(on))


def pivot_default_strategy(default: str) -> str:
    """What NEW portfolios start on: the pivot strategy once it is active."""
    return _pivot_cfg().get("strategy") if pivot_active() else default


def apply_pivot_if_due(db, today: Optional[date] = None, notify=None) -> int:
    """On/after activate_on: flip every real portfolio (user_id > 0) still on one of
    `from_strategies` to the pivot strategy. Idempotent; returns how many flipped."""
    from backend.database import AIPortfolioConfig
    if not pivot_active(today):
        return 0
    cfg = _pivot_cfg()
    target, frm = cfg["strategy"], list(cfg.get("from_strategies") or [])
    rows = db.query(AIPortfolioConfig).filter(AIPortfolioConfig.user_id > 0, AIPortfolioConfig.strategy.in_(frm)).all()
    if not rows:
        return 0
    from backend.trading_utils import get_strategy_profile
    prof = get_strategy_profile(target)
    for c in rows:
        logger.warning(f"CANSLIM 2.0 PIVOT: user {c.user_id} {c.strategy} -> {target}")
        c.strategy = target
        c.max_positions = prof.get("max_positions", 20)
    db.commit()
    if notify is None:
        from backend.email_utils import create_notification as notify
    notify(user_id=int(cfg.get("notify_user_id", 1)), kind="canslim2_pivot", priority="high",
           title=f"AI Portfolio switched to CANSLIM 2.0 ({len(rows)} portfolio{'s' if len(rows) != 1 else ''})",
           body="Pre-registered Oct-10 (docs/canslim2-forward-plan.md, Strategy 3). Holdings below the 70th "
                "percentile sell at this cycle; buys refill from the top 10%, up to 20 positions.",
           data={"url": "/ai-portfolio"})
    return len(rows)
