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
import random
import statistics
from datetime import datetime, timedelta, timezone
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


def _normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation
    (≈1e-9 abs error). Pure stdlib so the module stays scipy-free.
    Returns ±inf at the open boundaries."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _power_analysis(
    p_ret: Sequence[float],
    s_ret: Sequence[float],
    beta: float,
    alpha_level: float = 0.05,
    power: float = 0.80,
) -> Optional[dict]:
    """Edge Validation Phase 4 — how much MORE data to detect the edge if it's real.

    Treats the daily market-neutral residual r_t = p_ret_t − β·s_ret_t (whose
    mean is the daily Jensen alpha) as an iid one-sample series and solves the
    standard z-based sample-size formula for a two-sided test::

        N = ((z_{1−α/2} + z_power) / d)²,   d = |mean(r)| / sd(r)

    Returns required/additional *trading days* (the unit the alpha regression
    is measured in) plus a calendar-month estimate. ``None`` when the effect is
    undefined (degenerate variance) or the sample is too small to estimate d.

    Assumptions (intentionally explicit, and surfaced in the output): the
    observed effect size persists; daily residuals are iid (ignores
    autocorrelation, so the true requirement is likely somewhat larger); trade
    cadence stays constant for the calendar estimate; the z-approximation is
    mildly conservative vs the exact noncentral-t at small N.
    """
    n = len(p_ret)
    if n < 3 or len(s_ret) != n:
        return None
    resid = [p - beta * s for p, s in zip(p_ret, s_ret)]
    mean_r = statistics.fmean(resid)
    sd_r = statistics.stdev(resid) if len(resid) > 1 else 0.0
    if sd_r <= 0 or mean_r == 0:
        return None
    effect = abs(mean_r) / sd_r
    z_alpha = _normal_ppf(1 - alpha_level / 2)
    z_power = _normal_ppf(power)
    required = ((z_alpha + z_power) / effect) ** 2
    required_days = max(1, math.ceil(required))
    additional = max(0, required_days - n)
    # Calendar finish line (2026-09-01): trading days → calendar days at the
    # 365/252 ratio, from today. A date reads as a goal; "N trading days"
    # reads as a chore. None when already sufficient or absurdly far out
    # (>3y ≈ effect too weak to responsibly project).
    projected_date = None
    if 0 < additional <= TRADING_DAYS_PER_YEAR * 3:
        _cal_days = math.ceil(additional * 365 / TRADING_DAYS_PER_YEAR)
        projected_date = (
            datetime.now(timezone.utc).date() + timedelta(days=_cal_days)
        ).isoformat()
    return {
        "effect_size": round(effect, 4),
        "current_days": n,
        "required_days": required_days,
        "additional_days_needed": additional,
        "already_sufficient": additional == 0,
        "est_additional_months": round(additional / TRADING_DAYS_PER_YEAR * 12, 1),
        "projected_date": projected_date,
        "alpha_level": alpha_level,
        "target_power": power,
        "assumptions": (
            "z-approx two-sided one-sample test on daily alpha residuals; "
            "assumes effect size persists, residuals iid (ignores "
            "autocorrelation — true N likely larger), constant trade cadence."
        ),
    }


def attribute_returns(
    positions: Sequence[dict],
    starting_equity: Optional[float] = None,
    top_n: int = 5,
) -> dict:
    """Edge Validation Phase 3 — decompose the realized edge by closed position.

    ``positions`` is a list of per-SELL dicts the caller has already extracted
    (mirrors the Phase 2 "caller passes records" pattern), each with:
      ``ticker``, ``realized_gain`` (dollar P&L on the lot), ``deployed``
      (cost_basis × shares, the capital at risk), ``position_return_pct``,
      ``spy_return_pct`` (SPY's return over the SAME hold window), and
      ``holding_days``.

    For each position computes ``excess_gain`` — the dollars earned beyond what
    that same capital would have made sitting in SPY over the identical window
    (realized_gain − deployed × spy_return_pct/100) — and, when
    ``starting_equity`` is known, ``contribution_pct`` (the position's pp of
    starting equity). Returns positions ranked by dollar contribution, aggregate
    roll-ups, and a concentration block answering "do a few names drive the
    edge?". ``status='insufficient_data'`` when there are no closed positions.
    """
    rows = []
    for p in positions:
        rg = p.get("realized_gain")
        if rg is None:
            continue
        deployed = p.get("deployed")
        spy_ret = p.get("spy_return_pct")
        excess = None
        if deployed is not None and spy_ret is not None:
            excess = round(rg - deployed * (spy_ret / 100.0), 2)
        rows.append({
            "ticker": p.get("ticker"),
            "realized_gain": round(rg, 2),
            "deployed": round(deployed, 2) if deployed is not None else None,
            "position_return_pct": p.get("position_return_pct"),
            "spy_return_pct": spy_ret,
            "excess_gain": excess,
            "holding_days": p.get("holding_days"),
            "contribution_pct": (
                round(rg / starting_equity * 100, 2)
                if starting_equity else None
            ),
        })

    if not rows:
        return {"status": "insufficient_data", "closed_positions": 0}

    rows.sort(key=lambda r: r["realized_gain"], reverse=True)
    winners = [r for r in rows if r["realized_gain"] > 0]
    losers = [r for r in rows if r["realized_gain"] < 0]
    gross_winners = round(sum(r["realized_gain"] for r in winners), 2)
    gross_losers = round(sum(r["realized_gain"] for r in losers), 2)

    # Concentration of the *gains*: what share do the very top names carry?
    top1_share = round(winners[0]["realized_gain"] / gross_winners * 100, 1) if gross_winners > 0 else None
    top3_share = (
        round(sum(r["realized_gain"] for r in winners[:3]) / gross_winners * 100, 1)
        if gross_winners > 0 else None
    )
    names_for_half = None
    if gross_winners > 0:
        cum = 0.0
        for i, r in enumerate(winners, 1):
            cum += r["realized_gain"]
            if cum >= gross_winners / 2:
                names_for_half = i
                break

    excesses = [r["excess_gain"] for r in rows if r["excess_gain"] is not None]
    return {
        "status": "ok",
        "closed_positions": len(rows),
        "positions": rows,
        "total_realized_gain": round(sum(r["realized_gain"] for r in rows), 2),
        "total_excess_gain": round(sum(excesses), 2) if excesses else None,
        "gross_winners_gain": gross_winners,
        "gross_losers_loss": gross_losers,
        "winners": len(winners),
        "losers": len(losers),
        "top_contributors": rows[:top_n],
        "top_detractors": [r for r in reversed(rows) if r["realized_gain"] < 0][:top_n],
        "concentration": {
            "top1_share_of_gains_pct": top1_share,
            "top3_share_of_gains_pct": top3_share,
            "names_for_half_of_gains": names_for_half,
        },
    }


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
        # Measure the PORTFOLIO leg over the same paired window as SPY so
        # excess/alpha are apples-to-apples. The headline `total_return_pct`
        # spans the full portfolio series; subtracting the paired-window
        # `spy_return_pct` from it would bias excess/alpha whenever SPY has a
        # boundary gap (historical SPY coverage runs ~78-97%) — the two legs
        # would cover different calendar windows. When SPY coverage is
        # complete, `port_return_pairs == total_return_pct` and this reduces to
        # the original formula, so the common case is unchanged.
        port_return_pairs = (
            round((p_series[-1] / p_series[0] - 1.0) * 100, 2) if p_series[0] else None
        )
        if port_return_pairs is not None:
            result["excess_return_pct"] = round(port_return_pairs - spy_return_pct, 2)

        p_ret = _daily_returns(p_series)
        s_ret = _daily_returns(s_series)
        result["spy_sharpe"] = _annualized_sharpe(s_ret)
        if len(p_ret) >= 2 and len(s_ret) >= 2:
            var_s = statistics.variance(s_ret)
            if var_s > 0:
                beta = _covariance(p_ret, s_ret) / var_s
                result["beta"] = round(beta, 3)
                # Jensen's alpha over the paired window (rf=0), NOT annualized:
                # paired-window portfolio return minus what beta-exposure to SPY
                # "earned" over that same window.
                if port_return_pairs is not None and spy_return_pct is not None:
                    result["alpha_pct"] = round(port_return_pairs - beta * spy_return_pct, 2)
                # Edge Validation Phase 1: is that alpha distinguishable from luck?
                result["alpha_significance"] = _alpha_significance(p_ret, s_ret)
                # Edge Validation Phase 4: if it IS real, how much more data to prove it?
                result["power"] = _power_analysis(p_ret, s_ret, beta)

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


def _paired_excess_series(
    port_values: Sequence[float],
    spy_values: Sequence[Optional[float]],
    spy_dist_pct: Sequence[Optional[float]],
) -> list[tuple[float, float]]:
    """Aligned (daily excess return, SPY 50MA distance) pairs.

    Day i is included only when both series have valid values on i-1 and i
    and the regime distance is known on the RETURN day i — the exact
    filtering regime_conditional_edge has always used, factored out so the
    bootstrap resamples the identical series the t-test sees.
    """
    n = min(len(port_values), len(spy_values), len(spy_dist_pct))
    out: list[tuple[float, float]] = []
    for i in range(1, n):
        pv0, pv1 = port_values[i - 1], port_values[i]
        sv0, sv1 = spy_values[i - 1], spy_values[i]
        dist = spy_dist_pct[i]
        if not pv0 or not pv1 or not sv0 or not sv1 or dist is None:
            continue
        out.append(((pv1 / pv0 - 1.0) - (sv1 / sv0 - 1.0), dist))
    return out


def regime_conditional_edge(
    port_values: Sequence[float],
    spy_values: Sequence[Optional[float]],
    spy_dist_pct: Sequence[Optional[float]],
    trend_threshold_pct: float = 1.5,
) -> Optional[dict]:
    """Regime-conditional daily excess-return test (owner ask, 2026-07-22).

    The unconditional alpha clock is honest but answers the slowest possible
    question: this strategy's edge is REGIME-DEPENDENT (live attribution:
    trend days +33 bps/day over SPY, chop days -30), so the blended daily
    signal is tiny relative to noise and takes ~9 years to prove. Splitting
    by regime un-dilutes both effects: the trend-day edge alone reached
    p=0.14 in one summer and is provable ~10x sooner.

    Day i's regime comes from spy_dist_pct[i] — SPY's % distance above its
    50MA on the RETURN day: trend when > trend_threshold_pct, chop otherwise
    (below-MA days land in chop; the binary gate makes them rare and flat).
    Excess = raw daily portfolio return minus SPY return (not beta-adjusted —
    the deployment question is "would this money have done better in SPY",
    which is the raw comparison).

    Returns {'trend': {...}, 'chop': {...}, 'threshold_pct'} or None when
    there aren't enough aligned days. Each bucket: n, mean_daily_excess_bps,
    t_stat, p_value, significant_95, and for a positive mean the z-based
    required/additional day counts at 95%/80% (same formula and caveats as
    _power_analysis — assumes the effect persists, iid days).
    """
    series = _paired_excess_series(port_values, spy_values, spy_dist_pct)
    if not series:
        return None
    buckets: dict[str, list[float]] = {"trend": [], "chop": []}
    for excess, dist in series:
        buckets["trend" if dist > trend_threshold_pct else "chop"].append(excess)

    def _bucket_stats(xs: list[float]) -> Optional[dict]:
        if len(xs) < 3:
            return None
        m = statistics.fmean(xs)
        sd = statistics.stdev(xs)
        if sd <= 0:
            return None
        t = m / (sd / math.sqrt(len(xs)))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
        out = {
            "n_days": len(xs),
            "mean_daily_excess_bps": round(m * 1e4, 1),
            "t_stat": round(t, 2),
            "p_value": round(p, 4),
            "significant_95": bool(p < 0.05),
        }
        if m > 0:
            required = math.ceil(((1.96 + 0.8416) / (m / sd)) ** 2)
            out["required_days"] = required
            out["additional_days_needed"] = max(0, required - len(xs))
        return out

    trend = _bucket_stats(buckets["trend"])
    chop = _bucket_stats(buckets["chop"])
    if trend is None and chop is None:
        return None
    return {"trend": trend, "chop": chop,
            "threshold_pct": trend_threshold_pct}


def regime_mix_summary(
    spy_dist_pct: Sequence[Optional[float]],
    regime_edge: Optional[dict],
    window_days: int = 60,
    trend_threshold_pct: float = 1.5,
) -> Optional[dict]:
    """Trailing regime mix vs the breakeven mix (owner ask 2026-08-25).

    The strategy's blended edge is a weighted average of a positive
    trend-day mean and a negative chop-day mean, so whether it beats SPY
    is a function of the market's regime mix: breakeven trend share =
    |chop| / (trend + |chop|). This reports the trailing window's actual
    trend share against that bar, turning "should I worry about chop?"
    into a glanceable meter. Descriptive only — same persistence caveats
    as regime_conditional_edge (the means are point estimates).

    Returns None with <10 classified days. breakeven/blended fields are
    present only when the regime means have the canonical signs
    (trend > 0 > chop); with any other sign pattern a breakeven share
    doesn't exist, and the mix alone is still reported.
    """
    dists = [d for d in spy_dist_pct if d is not None]
    tail = dists[-window_days:]
    if len(tail) < 10:
        return None
    share = sum(1 for d in tail if d > trend_threshold_pct) / len(tail)
    out = {
        "window_days": len(tail),
        "trend_share_pct": round(share * 100.0, 1),
        "threshold_pct": trend_threshold_pct,
    }
    trend = (regime_edge or {}).get("trend")
    chop = (regime_edge or {}).get("chop")
    if trend and chop:
        tm = trend.get("mean_daily_excess_bps") or 0.0
        cm = chop.get("mean_daily_excess_bps") or 0.0
        if tm > 0 > cm:
            breakeven = -cm / (tm - cm)
            out["breakeven_trend_share_pct"] = round(breakeven * 100.0, 1)
            out["blended_daily_excess_bps"] = round(share * tm + (1.0 - share) * cm, 1)
            out["above_breakeven"] = bool(share > breakeven)
    return out


def bootstrap_regime_excess(
    port_values: Sequence[float],
    spy_values: Sequence[Optional[float]],
    spy_dist_pct: Sequence[Optional[float]],
    trend_threshold_pct: float = 1.5,
    n_boot: int = 2000,
    block_len: int = 5,
    seed: int = 20260825,
) -> Optional[dict]:
    """Moving-block bootstrap CIs on the paired daily excess-return series.

    Complements regime_conditional_edge on the SAME series and buckets: the
    t/normal tests there assume iid near-normal days, but daily excess
    returns are fat-tailed and mildly autocorrelated, so those intervals
    can be optimistic. Resampling contiguous blocks (block_len trading
    days, ~1 week) preserves short-range dependence and skew — these are
    the intervals to gate real-money decisions on.

    Deterministic for a given seed (fixed random.Random stream), so the
    numbers are stable across refreshes of the same window. Buckets with
    fewer than 8 days return None. prob_positive_pct is the fraction of
    bootstrap means above zero — the plain-language "chance the edge is
    real given the data so far".
    """
    series = _paired_excess_series(port_values, spy_values, spy_dist_pct)
    if len(series) < 8:
        return None
    samples = {
        "overall": [e for e, _ in series],
        "trend": [e for e, d in series if d > trend_threshold_pct],
        "chop": [e for e, d in series if d <= trend_threshold_pct],
    }
    rng = random.Random(seed)

    def _boot(xs: list[float]) -> Optional[dict]:
        n = len(xs)
        if n < 8:
            return None
        length = max(1, min(block_len, n // 2))
        starts = n - length + 1
        n_blocks = math.ceil(n / length)
        means = []
        for _ in range(n_boot):
            sample: list[float] = []
            for _ in range(n_blocks):
                s = rng.randrange(starts)
                sample.extend(xs[s:s + length])
            del sample[n:]
            means.append(sum(sample) / n)
        means.sort()

        def _pctile(p: float) -> float:
            idx = p * (len(means) - 1)
            lo_i = math.floor(idx)
            hi_i = math.ceil(idx)
            frac = idx - lo_i
            return means[lo_i] * (1.0 - frac) + means[hi_i] * frac

        lo, hi = _pctile(0.025), _pctile(0.975)
        mean = statistics.fmean(xs)
        prob_pos = sum(1 for m in means if m > 0) / len(means)

        def _annualize(v: float) -> float:
            return round(v * TRADING_DAYS_PER_YEAR * 100.0, 1)

        return {
            "n_days": n,
            "mean_daily_excess_bps": round(mean * 1e4, 1),
            "alpha_annualized_pct": _annualize(mean),
            "ci_low_annualized_pct": _annualize(lo),
            "ci_high_annualized_pct": _annualize(hi),
            "prob_positive_pct": round(prob_pos * 100.0, 1),
            "excludes_zero_95": bool(lo > 0 or hi < 0),
        }

    out = {key: _boot(xs) for key, xs in samples.items()}
    if all(v is None for v in out.values()):
        return None
    out["params"] = {
        "n_boot": n_boot,
        "block_len": block_len,
        "seed": seed,
        "threshold_pct": trend_threshold_pct,
    }
    return out
