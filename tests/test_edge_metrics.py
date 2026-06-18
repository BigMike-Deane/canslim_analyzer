"""Tests for backend/edge_metrics.py — the AI-portfolio edge scorecard math.

Pure-function tests with hand-computable series so the metric definitions
(return vs SPY, Jensen alpha over the window, beta, annualized Sharpe, max
drawdown, win rate) are pinned to known values.
"""
import math
import statistics

import pytest

from backend.edge_metrics import (
    compute_edge_metrics,
    leading_flat_start_index,
    TRADING_DAYS_PER_YEAR,
)


class TestLeadingFlatTrim:
    def test_keeps_last_flat_day_as_baseline(self):
        # 3 flat days then deviation -> start at index 2 (last flat day).
        vals = [25000, 25000, 25000, 25090, 25500]
        assert leading_flat_start_index(vals) == 2

    def test_no_flat_prefix(self):
        assert leading_flat_start_index([100, 110, 120]) == 0

    def test_all_flat_keeps_everything(self):
        assert leading_flat_start_index([100, 100, 100]) == 0

    def test_short_series_untouched(self):
        assert leading_flat_start_index([100]) == 0
        assert leading_flat_start_index([]) == 0

    def test_single_flat_day_then_move(self):
        assert leading_flat_start_index([100, 110]) == 0


class TestInsufficientData:
    def test_empty_series(self):
        m = compute_edge_metrics([], [], [])
        assert m["status"] == "insufficient_data"
        assert m["trading_days"] == 0

    def test_single_point(self):
        m = compute_edge_metrics([25000.0], [25000.0], [])
        assert m["status"] == "insufficient_data"
        assert m["trading_days"] == 1

    def test_insufficient_still_reports_win_rate(self):
        # Win rate comes from trades, not the equity curve, so it survives even
        # when the curve is too short for risk stats.
        m = compute_edge_metrics([25000.0], [], [100.0, -50.0])
        assert m["status"] == "insufficient_data"
        assert m["win_rate_pct"] == 50.0
        assert m["closed_trades"] == 2


class TestReturns:
    def test_total_and_spy_return(self):
        # Portfolio +21% (compounded 10%/day x2), SPY flat.
        m = compute_edge_metrics([100, 110, 121], [100, 100, 100], [])
        assert m["status"] == "ok"
        assert m["trading_days"] == 3
        assert m["total_return_pct"] == 21.0
        assert m["spy_return_pct"] == 0.0
        assert m["excess_return_pct"] == 21.0

    def test_low_sample_flag(self):
        m = compute_edge_metrics([100, 110, 121], [100, 100, 100], [])
        assert m["low_sample"] is True  # 3 < 20

    def test_not_annualized_flag(self):
        m = compute_edge_metrics([100, 110], [100, 105], [])
        assert m["annualized"] is False


class TestBetaAndAlpha:
    def test_beta_exactly_two(self):
        # Portfolio moves are exactly 2x SPY's each day -> beta == 2.0.
        spy = [100, 110, 121, 108.9]   # returns +0.1, +0.1, -0.1
        port = [100, 120, 144, 115.2]  # returns +0.2, +0.2, -0.2
        m = compute_edge_metrics(port, spy, [])
        assert m["beta"] == pytest.approx(2.0, abs=1e-9)

    def test_alpha_is_beta_adjusted_excess(self):
        spy = [100, 110, 121, 108.9]
        port = [100, 120, 144, 115.2]
        m = compute_edge_metrics(port, spy, [])
        # total_return = 15.2%, spy_return = 8.9%, beta = 2.0
        # Jensen alpha over window = 15.2 - 2.0 * 8.9 = -2.6
        assert m["total_return_pct"] == pytest.approx(15.2, abs=0.01)
        assert m["spy_return_pct"] == pytest.approx(8.9, abs=0.01)
        assert m["alpha_pct"] == pytest.approx(-2.6, abs=0.05)

    def test_positive_alpha_beats_beta_expectation(self):
        # Same beta-2 SPY path, but portfolio outruns the beta-implied return.
        spy = [100, 110, 121, 108.9]
        port = [100, 125, 150, 130]
        m = compute_edge_metrics(port, spy, [])
        assert m["alpha_pct"] > 0

    def test_zero_variance_spy_gives_no_beta(self):
        # Flat SPY -> var(spy returns) == 0 -> beta/alpha undefined.
        m = compute_edge_metrics([100, 110, 121], [100, 100, 100], [])
        assert m["beta"] is None
        assert m["alpha_pct"] is None


class TestSharpeAndVolatility:
    def test_sharpe_known_value(self):
        # returns = [0.2, 0.0]; mean 0.1, sample stdev sqrt(0.02).
        port = [100, 120, 120]
        expected = (0.1 / statistics.stdev([0.2, 0.0])) * math.sqrt(TRADING_DAYS_PER_YEAR)
        m = compute_edge_metrics(port, [None, None, None], [])
        assert m["sharpe"] == pytest.approx(round(expected, 2), abs=0.01)

    def test_flat_curve_has_no_sharpe(self):
        m = compute_edge_metrics([100, 100, 100], [100, 100, 100], [])
        assert m["sharpe"] is None          # zero stdev
        assert m["volatility_annualized_pct"] == 0.0


class TestDrawdown:
    def test_max_drawdown(self):
        # peak 120, trough 90 -> -25%.
        m = compute_edge_metrics([100, 120, 90, 110], [None, None, None, None], [])
        assert m["max_drawdown_pct"] == -25.0

    def test_monotonic_up_has_zero_drawdown(self):
        m = compute_edge_metrics([100, 110, 121], [None, None, None], [])
        assert m["max_drawdown_pct"] == 0.0


class TestWinRate:
    def test_win_rate_excludes_breakeven(self):
        m = compute_edge_metrics([100, 110], [100, 105], [100, -50, 30, 0, -10])
        # wins (>0): 100, 30 -> 2 of 5 = 40.0
        assert m["win_rate_pct"] == 40.0
        assert m["closed_trades"] == 5

    def test_no_trades_gives_none(self):
        m = compute_edge_metrics([100, 110], [100, 105], [])
        assert m["win_rate_pct"] is None


class TestSpyAlignment:
    def test_all_none_spy_keeps_portfolio_stats(self):
        m = compute_edge_metrics([100, 110, 121], [None, None, None], [])
        assert m["status"] == "ok"
        assert m["total_return_pct"] == 21.0
        assert m["spy_return_pct"] is None
        assert m["excess_return_pct"] is None
        assert m["beta"] is None

    def test_gapped_spy_uses_available_pairs(self):
        # Only indices 0 and 2 have both port+spy -> spy_return from those.
        m = compute_edge_metrics([100, 110, 121, 133], [100, None, 121, None], [])
        assert m["spy_return_pct"] == 21.0
        assert m["total_return_pct"] == 33.0
        assert m["excess_return_pct"] == 12.0
        assert m["beta"] is None  # only one usable benchmark return


# ─── Edge Validation Phase 1: statistical significance ────────────────────────
import math
from backend.edge_metrics import (
    _student_t_two_sided_p, _t_critical_95, _wilson_ci, _alpha_significance,
    _edge_verdict, compute_edge_metrics,
)


class TestStudentT:
    def test_two_sided_p_known_values(self):
        # t=2.228, df=10 → two-sided p ≈ 0.05 (textbook critical value)
        assert abs(_student_t_two_sided_p(2.228, 10) - 0.05) < 0.002
        # t=2.0, df=60 → p ≈ 0.0501
        assert abs(_student_t_two_sided_p(2.0, 60) - 0.0501) < 0.003
        # t=0 → p=1.0 (no evidence against null)
        assert abs(_student_t_two_sided_p(0.0, 30) - 1.0) < 1e-9

    def test_t_critical_95_known(self):
        assert abs(_t_critical_95(10) - 2.228) < 0.01   # df=10
        assert abs(_t_critical_95(30) - 2.042) < 0.01   # df=30
        assert _t_critical_95(1000) < 2.0               # → ~1.96 for large df


class TestWilsonCI:
    def test_basic(self):
        lo, hi = _wilson_ci(8, 10)        # 80% wins, n=10
        assert 0 <= lo < 80 < hi <= 100   # interval brackets the estimate, wide
        assert _wilson_ci(0, 0) is None

    def test_tighter_with_more_data(self):
        narrow = _wilson_ci(800, 1000)
        wide = _wilson_ci(8, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestAlphaSignificance:
    def test_pure_noise_is_insignificant(self):
        # Portfolio == SPY + tiny zero-mean wiggle → alpha ~0, NOT significant.
        spy = [0.001 * ((i % 7) - 3) for i in range(60)]
        port = [s + 0.0005 * ((i % 5) - 2) for i, s in enumerate(spy)]
        sig = _alpha_significance(port, spy)
        assert sig is not None
        assert sig["significant_95"] is False

    def test_strong_persistent_alpha_is_significant(self):
        # Portfolio = SPY beta-1 plus a steady +20bps/day with low noise → sig.
        spy = [0.002 * ((i % 9) - 4) for i in range(80)]
        port = [s + 0.002 + 0.0001 * ((i % 3) - 1) for i, s in enumerate(spy)]
        sig = _alpha_significance(port, spy)
        assert sig is not None
        assert sig["t_stat"] > 2.0
        assert sig["significant_95"] is True
        assert sig["alpha_annualized_ci_low_pct"] > 0  # CI excludes zero

    def test_too_few_obs(self):
        assert _alpha_significance([0.01, 0.02], [0.01, 0.02]) is None


class TestEdgeVerdict:
    def test_small_sample_inconclusive(self):
        assert _edge_verdict({"alpha_annualized_pct": 50, "significant_95": True}, 5, 3) \
            == "inconclusive_small_sample"

    def test_promising_but_not_significant(self):
        sig = {"alpha_annualized_pct": 12.0, "significant_95": False}
        assert _edge_verdict(sig, 60, 30) == "promising_insufficient_sample"

    def test_significant_edge(self):
        sig = {"alpha_annualized_pct": 12.0, "significant_95": True}
        assert _edge_verdict(sig, 60, 30) == "significant_edge"

    def test_no_edge(self):
        sig = {"alpha_annualized_pct": -3.0, "significant_95": False}
        assert _edge_verdict(sig, 60, 30) == "no_measurable_edge"


class TestComputeEdgeIntegration:
    def test_includes_significance_and_verdict(self):
        port = [25000 * (1.004 ** i) for i in range(40)]   # steady climber
        spy = [25000 * (1.001 ** i) for i in range(40)]
        m = compute_edge_metrics(port, spy, realized_gains=[100, -50, 200, 80, -30])
        assert "edge_verdict" in m
        assert "alpha_significance" in m and m["alpha_significance"] is not None
        assert "win_rate_ci_95" in m
