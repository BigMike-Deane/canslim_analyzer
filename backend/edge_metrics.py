"""Edge / alpha scorecard metrics.

Pure, dependency-light computation of risk-adjusted performance metrics for the
AI paper portfolio. Everything here is a function of two equity curves
(portfolio + a SPY benchmark rebased to the same starting scale) plus the list
of realized trade gains. No DB access, no I/O — so it is trivially unit-testable
and carries zero risk to the eval-sensitive trading path (``ai_trader.py``).

Conventions
-----------
- The portfolio is a *closed* paper account (no deposits/withdrawals), so
  ``total_value`` is a clean equity curve and simple daily returns are valid.
- Sharpe and volatility are **annualized** (sqrt(252) convention) so they are
  comparable to published figures.
- Return and alpha are reported at the **actual window scale** (NOT annualized).
  Annualizing a short sample produces noisy, misleading figures.
- Risk-free rate is assumed 0 (short horizons, paper account) — documented so a
  future caller can subtract one if desired.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

TRADING_DAYS_PER_YEAR = 252
# Below this many daily observations, annualized stats (Sharpe, vol, beta) are
# statistically thin — flagged via ``low_sample`` so the UI can caveat them.
LOW_SAMPLE_THRESHOLD = 20


def leading_flat_start_index(values: Sequence[float]) -> int:
    """Index to start at after dropping the pre-inception flat segment.

    The paper portfolio sits at exactly ``starting_cash`` for every snapshot
    before its first trade (cash undeployed). Those days have zero portfolio
    return while the benchmark moves, which would distort beta/Sharpe/alpha and
    anchor the SPY benchmark weeks too early. Mirrors the frontend chart's
    pre-inception trim.

    Returns the index of the *last flat day* (kept as the t0 baseline) so the
    active series begins with the correct anchor. If the series never deviates
    (or is too short), returns 0 — nothing is trimmed.
    """
    n = len(values)
    if n < 2:
        return 0
    first = values[0]
    i = 0
    while i < n and values[i] == first:
        i += 1
    if i >= n:
        return 0  # never deviated — keep everything
    return i - 1  # keep the last flat day as the baseline anchor


def _daily_returns(values: Sequence[float]) -> list[float]:
    """Simple period-over-period returns. Skips non-positive prior values."""
    out: list[float] = []
    for prev, cur in zip(values, values[1:]):
        if prev and prev > 0:
            out.append(cur / prev - 1.0)
    return out


def _max_drawdown_pct(values: Sequence[float]) -> Optional[float]:
    """Largest peak-to-trough decline as a negative percent (0.0 if monotonic)."""
    if len(values) < 2:
        return None
    peak = values[0]
    mdd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return round(mdd * 100, 2)


def _covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def _annualized_sharpe(returns: Sequence[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    sd = statistics.stdev(returns)
    if sd == 0:
        return None
    mean = statistics.fmean(returns)
    return round((mean / sd) * math.sqrt(TRADING_DAYS_PER_YEAR), 2)


def _annualized_vol_pct(returns: Sequence[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    return round(statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100, 2)


def compute_edge_metrics(
    port_values: Sequence[float],
    spy_values: Sequence[float],
    realized_gains: Sequence[float] = (),
) -> dict:
    """Compute the edge scorecard from aligned equity curves + realized P&L.

    Parameters
    ----------
    port_values
        Portfolio total_value, one point per trading day, ascending by date.
    spy_values
        SPY benchmark rebased to the same starting scale as ``port_values``
        (i.e. "what the starting cash would be worth in SPY"), aligned 1:1 by
        date. May contain ``None`` for days with no benchmark price — those days
        are dropped from benchmark-dependent stats (beta/alpha/spy_*) but the
        portfolio-only stats still use the full series.
    realized_gains
        ``realized_gain`` for each closed (SELL) trade. Used for win rate only.

    Returns
    -------
    dict with ``status`` of ``"ok"`` or ``"insufficient_data"`` and, when ok,
    return/alpha/beta/sharpe/drawdown/volatility/win-rate fields. Any metric
    that is undefined for the given data is ``None`` rather than raising.
    """
    port = [v for v in port_values if v is not None]
    if len(port) < 2:
        return {
            "status": "insufficient_data",
            "trading_days": len(port),
            "win_rate_pct": _win_rate(realized_gains),
            "closed_trades": len(realized_gains),
        }

    start_value = port[0]
    end_value = port[-1]
    total_return_pct = round((end_value / start_value - 1.0) * 100, 2) if start_value else None

    port_returns = _daily_returns(port)

    result = {
        "status": "ok",
        "trading_days": len(port),
        "low_sample": len(port) < LOW_SAMPLE_THRESHOLD,
        "start_value": round(start_value, 2),
        "end_value": round(end_value, 2),
        "total_return_pct": total_return_pct,
        "sharpe": _annualized_sharpe(port_returns),
        "volatility_annualized_pct": _annualized_vol_pct(port_returns),
        "max_drawdown_pct": _max_drawdown_pct(port),
        "annualized": False,  # return/alpha are window-scale, not annualized
        # Benchmark-dependent fields, filled below when SPY data is usable.
        "spy_return_pct": None,
        "excess_return_pct": None,
        "beta": None,
        "alpha_pct": None,
        "spy_sharpe": None,
        "spy_max_drawdown_pct": None,
        "win_rate_pct": _win_rate(realized_gains),
        "closed_trades": len(realized_gains),
    }

    # Benchmark-dependent stats require a date-aligned, gap-free SPY pair series.
    # Zip the *original* port_values (not the None-filtered `port`) so the SPY
    # series stays index-aligned with the portfolio day it belongs to.
    pairs = [
        (p, s)
        for p, s in zip(port_values, spy_values)
        if p is not None and s is not None and p > 0 and s > 0
    ]
    if len(pairs) >= 2:
        p_series = [p for p, _ in pairs]
        s_series = [s for _, s in pairs]
        spy_return_pct = round((s_series[-1] / s_series[0] - 1.0) * 100, 2)
        result["spy_return_pct"] = spy_return_pct
        result["spy_max_drawdown_pct"] = _max_drawdown_pct(s_series)
        if total_return_pct is not None:
            result["excess_return_pct"] = round(total_return_pct - spy_return_pct, 2)

        p_ret = _daily_returns(p_series)
        s_ret = _daily_returns(s_series)
        result["spy_sharpe"] = _annualized_sharpe(s_ret)
        if len(p_ret) >= 2 and len(s_ret) >= 2:
            var_s = statistics.variance(s_ret)
            if var_s > 0:
                beta = _covariance(p_ret, s_ret) / var_s
                result["beta"] = round(beta, 3)
                # Jensen's alpha over the window (rf=0), NOT annualized:
                # actual portfolio return minus what beta-exposure to SPY "earned".
                if total_return_pct is not None and spy_return_pct is not None:
                    result["alpha_pct"] = round(total_return_pct - beta * spy_return_pct, 2)

    return result


def _win_rate(realized_gains: Sequence[float]) -> Optional[float]:
    gains = [g for g in realized_gains if g is not None]
    if not gains:
        return None
    wins = sum(1 for g in gains if g > 0)
    return round(wins / len(gains) * 100, 1)
