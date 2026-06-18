"""Live-vs-backtest exit reconciliation (Edge Validation project, Phase 2).

Pure and dependency-light (only ``categorize_sell_reason``). Compares the EXIT
BEHAVIOR of live trading against the backtester's modeled behavior, per exit
reason, so trader<->backtester divergences — like the D1 stop-guard gap or the
trailing-stop cadence churn discovered this cycle — are caught SYSTEMATICALLY
instead of by accident.

Why behavioral fingerprints, not trade-by-trade: live picks differ from a fresh
backtest, and ``BacktestStaticSnapshot`` freezes today's scalars, so an exact
match is impossible. But the *shape* of each exit reason — its share of sells,
win rate, average hold days, average realized % — is regime-robust and is exactly
where the live/backtest drift showed up (live trailing exits held 9d/50% vs the
model's 81d/76%). Divergence in those fingerprints is the alarm.

Zero trading-path risk: no DB/IO, ai_trader.py never imported.
"""

from __future__ import annotations

from typing import Optional, Sequence

from backend.trading_engine import categorize_sell_reason


def summarize_exits(trades: Sequence[dict]) -> dict:
    """Aggregate SELL trades into per-canonical-reason behavioral fingerprints.

    Each trade is a dict with ``reason`` (str), ``holding_days`` (num|None), and
    ``realized_pct`` (num|None). Returns ``{reason: {count, pct_of_sells,
    win_rate, avg_hold_days, avg_realized_pct}}`` plus an ``"ALL"`` roll-up.
    """
    buckets: dict = {}
    total = 0
    for t in trades:
        cat = categorize_sell_reason(t.get("reason") or "")
        total += 1
        for key in (cat, "ALL"):
            b = buckets.setdefault(key, {"count": 0, "hold": [], "ret": []})
            b["count"] += 1
            rp = t.get("realized_pct")
            if rp is not None:
                b["ret"].append(rp)
            hd = t.get("holding_days")
            if hd is not None:
                b["hold"].append(hd)

    out: dict = {}
    for k, b in buckets.items():
        ret = b["ret"]
        hold = b["hold"]
        out[k] = {
            "count": b["count"],
            "pct_of_sells": round(b["count"] / total * 100, 1) if total else None,
            "win_rate": round(sum(1 for r in ret if r > 0) / len(ret) * 100, 1) if ret else None,
            "avg_hold_days": round(sum(hold) / len(hold), 1) if hold else None,
            "avg_realized_pct": round(sum(ret) / len(ret), 2) if ret else None,
        }
    return out


def compare_exit_behavior(
    live: dict,
    backtest: dict,
    hold_ratio_tol: float = 0.5,
    win_gap_tol: float = 25.0,
    ret_gap_tol: float = 5.0,
    min_count: int = 3,
) -> dict:
    """Per-reason live-vs-backtest comparison with divergence flags.

    A reason is flagged when both sides have >= ``min_count`` exits AND any
    fingerprint diverges beyond tolerance: hold days by > ``hold_ratio_tol``
    (relative), win rate by > ``win_gap_tol`` (pp), or avg realized % by
    > ``ret_gap_tol`` (pp). Reasons present on only one side are reported as
    ``presence`` divergences (the engine takes an exit the other never does).
    """
    reasons = sorted(set(live) | set(backtest))
    rows = []
    diverged = []
    for r in reasons:
        if r == "ALL":
            continue
        l = live.get(r)
        b = backtest.get(r)
        flags: list = []
        if l and b and l["count"] >= min_count and b["count"] >= min_count:
            lh, bh = l["avg_hold_days"], b["avg_hold_days"]
            if lh is not None and bh:
                if abs(lh - bh) / bh > hold_ratio_tol:
                    flags.append("hold_days")
            if l["win_rate"] is not None and b["win_rate"] is not None:
                if abs(l["win_rate"] - b["win_rate"]) > win_gap_tol:
                    flags.append("win_rate")
            if l["avg_realized_pct"] is not None and b["avg_realized_pct"] is not None:
                if abs(l["avg_realized_pct"] - b["avg_realized_pct"]) > ret_gap_tol:
                    flags.append("realized_pct")
        elif (l and l["count"] >= min_count and not b) or (b and b["count"] >= min_count and not l):
            flags.append("presence")
        rows.append({"reason": r, "live": l, "backtest": b, "divergence_flags": flags})
        if flags:
            diverged.append(r)
    return {
        "reasons": rows,
        "diverged_reasons": diverged,
        "verdict": "diverged" if diverged else "aligned",
        "tolerances": {
            "hold_ratio": hold_ratio_tol,
            "win_gap_pp": win_gap_tol,
            "ret_gap_pp": ret_gap_tol,
            "min_count": min_count,
        },
    }


def live_trade_to_record(t) -> dict:
    """Normalize an AIPortfolioTrade SELL row to a reconciliation record."""
    realized_pct = None
    cost_basis = getattr(t, "cost_basis", None)
    shares = getattr(t, "shares", None)
    realized_gain = getattr(t, "realized_gain", None)
    if realized_gain is not None and cost_basis and shares and cost_basis * shares != 0:
        realized_pct = realized_gain / (cost_basis * shares) * 100
    return {
        "reason": getattr(t, "reason", "") or "",
        "holding_days": getattr(t, "holding_days", None),
        "realized_pct": realized_pct,
    }


def backtest_trade_to_record(t: dict) -> dict:
    """Normalize a backtest SELL trade dict to a reconciliation record."""
    return {
        "reason": t.get("reason", "") or "",
        "holding_days": t.get("holding_days"),
        "realized_pct": t.get("gain_pct"),
    }
