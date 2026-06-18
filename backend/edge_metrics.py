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


# ─── Statistical significance (Edge Validation project, Phase 1) ──────────────
# Pure-stdlib so the same zero-risk, unit-testable guarantees hold. The point of
# these is to replace false-precision point estimates with honest uncertainty:
# with ~30 trades the alpha estimate is noisy, and the verdict should say so.

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 1e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: int) -> Optional[float]:
    """Two-sided p-value P(|T| > |t|) for Student's t with df degrees of freedom."""
    if df <= 0:
        return None
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def _t_critical_95(df: int) -> float:
    """Two-sided 95% critical t value via bisection on the t survival function."""
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if (_student_t_two_sided_p(mid, df) or 0.0) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> Optional[tuple]:
    """Wilson score 95% interval for a binomial proportion, as percents."""
    if n <= 0:
        return None
    phat = wins / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, center - half) * 100, 1), round(min(1.0, center + half) * 100, 1))


def _alpha_significance(p_ret: Sequence[float], s_ret: Sequence[float]) -> Optional[dict]:
    """OLS of daily portfolio returns on SPY returns; tests whether the intercept
    (Jensen's alpha) is distinguishable from zero. Returns t-stat, p-value, df,
    and a 95% CI on the annualized alpha. None if too few observations."""
    n = len(p_ret)
    if n < 3 or len(s_ret) != n:
        return None
    mx = statistics.fmean(s_ret)
    my = statistics.fmean(p_ret)
    sxx = sum((s - mx) ** 2 for s in s_ret)
    if sxx <= 0:
        return None
    beta = sum((s - mx) * (p - my) for s, p in zip(s_ret, p_ret)) / sxx
    alpha_daily = my - beta * mx
    sse = sum((p - (alpha_daily + beta * s)) ** 2 for s, p in zip(s_ret, p_ret))
    df = n - 2
    if df < 1:
        return None
    s2 = sse / df
    se_alpha = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    if se_alpha == 0:
        return None
    t = alpha_daily / se_alpha
    p_two = _student_t_two_sided_p(t, df)
    tcrit = _t_critical_95(df)
    ann = TRADING_DAYS_PER_YEAR
    return {
        "alpha_daily_bps": round(alpha_daily * 10000, 2),
        "alpha_annualized_pct": round(alpha_daily * ann * 100, 2),
        "alpha_annualized_ci_low_pct": round((alpha_daily - tcrit * se_alpha) * ann * 100, 2),
        "alpha_annualized_ci_high_pct": round((alpha_daily + tcrit * se_alpha) * ann * 100, 2),
        "t_stat": round(t, 2),
        "df": df,
        "p_value": round(p_two, 4) if p_two is not None else None,
        "significant_95": bool(p_two is not None and p_two < 0.05),
    }


def _edge_verdict(sig: Optional[dict], trading_days: int, closed_trades: int) -> str:
    """Synthesize a plain-language verdict from significance + sample size."""
    if sig is None or trading_days < LOW_SAMPLE_THRESHOLD or closed_trades < 10:
        return "inconclusive_small_sample"
    positive = sig["alpha_annualized_pct"] > 0
    if sig["significant_95"] and positive:
        return "significant_edge"
    if sig["significant_95"] and not positive:
        return "significant_negative"
    if positive:
        return "promising_insufficient_sample"
    return "no_measurable_edge"


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
                # Edge Validation Phase 1: is that alpha distinguishable from luck?
                result["alpha_significance"] = _alpha_significance(p_ret, s_ret)

    # Win-rate confidence interval (Wilson) + plain-language edge verdict.
    _gains = [g for g in realized_gains if g is not None]
    if _gains:
        result["win_rate_ci_95"] = _wilson_ci(sum(1 for g in _gains if g > 0), len(_gains))
    result["edge_verdict"] = _edge_verdict(
        result.get("alpha_significance"), result["trading_days"], len(_gains)
    )

    return result


def _win_rate(realized_gains: Sequence[float]) -> Optional[float]:
    gains = [g for g in realized_gains if g is not None]
    if not gains:
        return None
    wins = sum(1 for g in gains if g > 0)
    return round(wins / len(gains) * 100, 1)
