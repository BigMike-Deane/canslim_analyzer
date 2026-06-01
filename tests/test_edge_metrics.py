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
