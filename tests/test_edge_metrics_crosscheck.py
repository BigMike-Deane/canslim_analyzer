"""
Independent cross-check of backend/edge_metrics.py against `empyrical`.

Why this exists: the owner gate for this whole program is "no real-money
features until edge vs SPY is statistically proven", and the numbers that
gate rests on -- Sharpe, Jensen alpha, max drawdown, volatility -- are
computed by hand in edge_metrics.py using pure stdlib. Hand-rolled financial
math that nothing checks is exactly the kind of thing that is quietly wrong
for months.

This does NOT replace edge_metrics.py. It recomputes the same quantities
from the same series with an independent, widely-used implementation and
asserts they agree. A disagreement here means one of the two is wrong and
the number on the dashboard should not be trusted until it is resolved.

Conventions in edge_metrics.py, matched deliberately below:
  * simple (not log) returns, `_daily_returns`
  * sample standard deviation, ddof=1 (`statistics.stdev`)
  * risk-free rate of 0
  * annualization factor 252 (TRADING_DAYS_PER_YEAR)
  * max drawdown walked over LEVELS, not rebuilt from returns
"""

import math
import random

import pytest

from backend.edge_metrics import (
    TRADING_DAYS_PER_YEAR,
    _alpha_significance,
    _annualized_sharpe,
    _annualized_vol_pct,
    _covariance,
    _daily_returns,
    _max_drawdown_pct,
)

ep = pytest.importorskip(
    "empyrical", reason="empyrical not installed; cross-check skipped"
)
np = pytest.importorskip("numpy")


def _curves(n=400, seed=20260909):
    """A deterministic portfolio curve and a correlated SPY curve.

    Correlated on purpose -- an uncorrelated pair makes beta ~0 and hides
    disagreement in the alpha/beta regression, which is the part most likely
    to differ by convention.
    """
    rng = random.Random(seed)
    p, s = [25000.0], [100.0]
    for _ in range(n):
        market = rng.gauss(0.0004, 0.010)
        idio = rng.gauss(0.0002, 0.008)
        s.append(s[-1] * (1 + market))
        p.append(p[-1] * (1 + 1.15 * market + idio))
    return p, s


@pytest.fixture(scope="module")
def series():
    p_levels, s_levels = _curves()
    return {
        "p_levels": p_levels,
        "s_levels": s_levels,
        "p_ret": _daily_returns(p_levels),
        "s_ret": _daily_returns(s_levels),
    }


class TestAgainstEmpyrical:

    def test_sharpe(self, series):
        ours = _annualized_sharpe(series["p_ret"])
        theirs = ep.sharpe_ratio(
            np.array(series["p_ret"]), risk_free=0.0,
            annualization=TRADING_DAYS_PER_YEAR,
        )
        assert ours == pytest.approx(theirs, rel=0.02), (
            f"Sharpe: edge_metrics={ours} empyrical={theirs}"
        )

    def test_annualized_volatility(self, series):
        ours = _annualized_vol_pct(series["p_ret"])
        theirs = ep.annual_volatility(
            np.array(series["p_ret"]),
            annualization=TRADING_DAYS_PER_YEAR,
        ) * 100
        assert ours == pytest.approx(theirs, rel=0.02), (
            f"annualized vol %: edge_metrics={ours} empyrical={theirs}"
        )

    def test_max_drawdown(self, series):
        # edge_metrics walks LEVELS; empyrical rebuilds a curve from returns.
        # Those agree because the levels ARE the compounded returns.
        ours = _max_drawdown_pct(series["p_levels"])
        theirs = ep.max_drawdown(np.array(series["p_ret"])) * 100
        assert ours == pytest.approx(theirs, abs=0.1), (
            f"max drawdown %: edge_metrics={ours} empyrical={theirs}"
        )

    def test_beta(self, series):
        # edge_metrics derives beta from its own _covariance helper.
        var_s = _covariance(series["s_ret"], series["s_ret"])
        ours = _covariance(series["p_ret"], series["s_ret"]) / var_s
        _, theirs = ep.alpha_beta(
            np.array(series["p_ret"]), np.array(series["s_ret"]),
            risk_free=0.0, annualization=TRADING_DAYS_PER_YEAR,
        )
        assert ours == pytest.approx(theirs, rel=0.01), (
            f"beta: edge_metrics={ours} empyrical={theirs}"
        )

    def test_daily_alpha(self, series):
        """Compare DAILY alpha, which is convention-free.

        empyrical annualizes alpha by COMPOUNDING ((1+a)**252 - 1) while
        edge_metrics scales it LINEARLY (a * 252) -- see
        `alpha_annualized_pct`. Both are defensible and neither is a bug, but
        they are not the same number, so the comparison is done on the daily
        intercept and the convention gap is asserted separately below.
        """
        sig = _alpha_significance(series["p_ret"], series["s_ret"])
        assert sig is not None
        ours_daily = sig["alpha_daily_bps"] / 10000.0

        alpha_annual, _ = ep.alpha_beta(
            np.array(series["p_ret"]), np.array(series["s_ret"]),
            risk_free=0.0, annualization=TRADING_DAYS_PER_YEAR,
        )
        theirs_daily = (1.0 + alpha_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        assert ours_daily == pytest.approx(theirs_daily, abs=2e-5), (
            f"daily alpha: edge_metrics={ours_daily:.6%} "
            f"empyrical={theirs_daily:.6%}"
        )


class TestAnnualizationConventionIsDeliberate:
    """Pins the one place the two implementations legitimately differ.

    If someone later 'fixes' edge_metrics to compound, this test fails and
    forces the change to be a conscious decision -- the annualized alpha is
    rendered on the Edge Scorecard the owner gates real money on.
    """

    def test_edge_metrics_scales_alpha_linearly(self, series):
        sig = _alpha_significance(series["p_ret"], series["s_ret"])
        alpha_daily = sig["alpha_daily_bps"] / 10000.0

        linear = alpha_daily * TRADING_DAYS_PER_YEAR * 100
        assert sig["alpha_annualized_pct"] == pytest.approx(linear, abs=0.01)

        compounded = ((1 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1) * 100
        # Sanity: on a real-sized alpha the two conventions genuinely differ,
        # so this test is not vacuous.
        assert abs(compounded - linear) > 0.01


class TestDegenerateInputs:
    """Both implementations must agree that there is nothing to report."""

    def test_flat_curve_has_no_sharpe(self):
        flat = [1000.0] * 50
        assert _annualized_sharpe(_daily_returns(flat)) is None

    def test_flat_curve_has_zero_drawdown(self):
        assert _max_drawdown_pct([1000.0] * 50) == 0.0

    def test_too_few_points(self):
        assert _annualized_sharpe([0.01]) is None
        assert _max_drawdown_pct([1000.0]) is None
