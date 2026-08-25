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


class TestSpyBoundaryGapAlignment:
    """Excess/alpha must measure BOTH legs over the same SPY-paired window.

    Regression for the window-misalignment bug: with partial SPY coverage, the
    portfolio return used for excess/alpha must span the same paired days as
    the SPY return — not the full portfolio series.
    """

    def test_excess_uses_paired_window_when_spy_missing_at_start(self):
        # Portfolio compounds +10%/day over 4 days (full return = 33.1%).
        # SPY is missing on day 0, so the comparable window is days 1-3, where
        # the portfolio goes 110 -> 133.1 (+21%) and SPY 100 -> 121 (+21%).
        port = [100, 110, 121, 133.1]
        spy = [None, 100, 110, 121]
        m = compute_edge_metrics(port, spy, [])
        # Headline portfolio return still spans the full series.
        assert m["total_return_pct"] == pytest.approx(33.1, abs=0.05)
        # SPY return over the paired window.
        assert m["spy_return_pct"] == pytest.approx(21.0, abs=0.05)
        # Excess measured over the IDENTICAL paired window: 21% - 21% = 0%.
        # Pre-fix this was total(33.1) - spy(21.0) = 12.1, spuriously crediting
        # the portfolio's day0->day1 jump that SPY was never measured against.
        assert m["excess_return_pct"] == pytest.approx(0.0, abs=0.05)

    def test_full_coverage_excess_equals_total_minus_spy(self):
        # Complete SPY coverage -> paired window == full window, so excess still
        # equals total - spy exactly (no behavior change in the common case).
        port = [100, 110, 121]
        spy = [100, 105, 110.25]  # +5%/day -> +10.25%
        m = compute_edge_metrics(port, spy, [])
        assert m["total_return_pct"] == pytest.approx(21.0, abs=0.05)
        assert m["spy_return_pct"] == pytest.approx(10.25, abs=0.05)
        assert m["excess_return_pct"] == pytest.approx(
            m["total_return_pct"] - m["spy_return_pct"], abs=0.01
        )


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
        # Headline portfolio return still spans the full series (100 -> 133).
        assert m["total_return_pct"] == 33.0
        # Excess is measured over the SAME paired window as SPY (day0 -> day2),
        # where the portfolio also went 100 -> 121 (+21%): excess = 0.0. The
        # day2 -> day3 portfolio gain (121 -> 133) has no SPY counterpart and
        # must NOT count as excess. (Pre-window-alignment-fix this asserted
        # 12.0 = total(33) - spy(21), which credited that uncompared day.)
        assert m["excess_return_pct"] == 0.0
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

    def test_includes_power_block(self):
        # Phase 4: a beta-bearing series should also emit a `power` block.
        port = [25000 * (1.004 ** i) for i in range(40)]
        spy = [25000 * (1.001 ** i) for i in range(40)]
        m = compute_edge_metrics(port, spy, realized_gains=[100, -50, 200])
        assert "power" in m and m["power"] is not None
        assert m["power"]["required_days"] >= 1


from backend.edge_metrics import _normal_ppf, _power_analysis, attribute_returns


class TestNormalPPF:
    def test_known_quantiles(self):
        assert abs(_normal_ppf(0.975) - 1.95996) < 1e-3   # two-sided 95%
        assert abs(_normal_ppf(0.80) - 0.84162) < 1e-3    # 80% power
        assert abs(_normal_ppf(0.5)) < 1e-6               # median = 0
        assert abs(_normal_ppf(0.025) + 1.95996) < 1e-3   # symmetric

    def test_boundaries(self):
        assert _normal_ppf(0.0) == float("-inf")
        assert _normal_ppf(1.0) == float("inf")


class TestPowerAnalysis:
    def test_required_n_matches_formula(self):
        # Construct residuals with a known effect size d = mean/sd.
        # port = beta*spy + resid; use beta=0 so resid == port returns.
        # resid alternating ±a around mean m → mean=m, sd≈a. Pick m, a so d is clean.
        # Use m=0.001, sd via a spread → compute expected N and compare.
        n = 60
        # residual series: mean 0.001, with deterministic spread
        resid = [0.001 + (0.01 if i % 2 == 0 else -0.01) for i in range(n)]
        # Feed as p_ret with beta 0 and s_ret all zeros so residual == p_ret.
        p_ret = resid
        s_ret = [0.0] * n
        out = _power_analysis(p_ret, s_ret, beta=0.0)
        assert out is not None
        import statistics as st
        d = abs(st.fmean(resid)) / st.stdev(resid)
        z = _normal_ppf(0.975) + _normal_ppf(0.80)
        expected = math.ceil((z / d) ** 2)
        assert out["required_days"] == expected
        assert abs(out["effect_size"] - round(d, 4)) < 1e-4

    def test_already_sufficient_when_strong(self):
        # Huge, low-noise alpha → required N tiny → already sufficient.
        n = 80
        p_ret = [0.02 + (0.0001 if i % 2 else -0.0001) for i in range(n)]
        s_ret = [0.0] * n
        out = _power_analysis(p_ret, s_ret, beta=0.0)
        assert out["already_sufficient"] is True
        assert out["additional_days_needed"] == 0
        assert out["est_additional_months"] == 0.0

    def test_zero_variance_returns_none(self):
        # Constant residuals → sd 0 → undefined effect → None.
        assert _power_analysis([0.01] * 10, [0.0] * 10, beta=0.0) is None

    def test_too_few_obs_returns_none(self):
        assert _power_analysis([0.01, 0.02], [0.0, 0.0], beta=0.0) is None


class TestAttributeReturns:
    def _positions(self):
        # 3 closed positions: two winners, one loser. SPY return supplied per
        # hold window so excess is hand-computable.
        return [
            {"ticker": "AAA", "realized_gain": 600.0, "deployed": 1000.0,
             "position_return_pct": 60.0, "spy_return_pct": 10.0, "holding_days": 40},
            {"ticker": "BBB", "realized_gain": 300.0, "deployed": 1000.0,
             "position_return_pct": 30.0, "spy_return_pct": 5.0, "holding_days": 20},
            {"ticker": "CCC", "realized_gain": -100.0, "deployed": 1000.0,
             "position_return_pct": -10.0, "spy_return_pct": 8.0, "holding_days": 15},
        ]

    def test_empty_is_insufficient(self):
        out = attribute_returns([])
        assert out["status"] == "insufficient_data"
        assert out["closed_positions"] == 0

    def test_excess_and_aggregates(self):
        out = attribute_returns(self._positions(), starting_equity=25000.0)
        assert out["status"] == "ok"
        assert out["closed_positions"] == 3
        # excess = realized_gain − deployed*(spy/100)
        # AAA: 600 − 1000*0.10 = 500 ; BBB: 300 − 50 = 250 ; CCC: −100 − 80 = −180
        by = {p["ticker"]: p for p in out["positions"]}
        assert by["AAA"]["excess_gain"] == 500.0
        assert by["BBB"]["excess_gain"] == 250.0
        assert by["CCC"]["excess_gain"] == -180.0
        assert out["total_realized_gain"] == 800.0
        assert out["total_excess_gain"] == 570.0
        assert out["gross_winners_gain"] == 900.0
        assert out["gross_losers_loss"] == -100.0
        # contribution_pct = realized_gain / starting_equity * 100
        assert by["AAA"]["contribution_pct"] == round(600 / 25000 * 100, 2)

    def test_ranked_and_concentration(self):
        out = attribute_returns(self._positions(), starting_equity=25000.0)
        # Ranked by realized_gain desc: AAA, BBB, CCC
        assert [p["ticker"] for p in out["positions"]] == ["AAA", "BBB", "CCC"]
        # top contributor AAA, top detractor CCC
        assert out["top_contributors"][0]["ticker"] == "AAA"
        assert out["top_detractors"][0]["ticker"] == "CCC"
        c = out["concentration"]
        # AAA is 600/900 = 66.7% of gains; top3 = 100%; half of gains (450)
        # reached by AAA alone (600 >= 450) → names_for_half = 1
        assert c["top1_share_of_gains_pct"] == 66.7
        assert c["top3_share_of_gains_pct"] == 100.0
        assert c["names_for_half_of_gains"] == 1

    def test_missing_spy_leaves_excess_none(self):
        positions = [{"ticker": "ZZZ", "realized_gain": 100.0, "deployed": 1000.0,
                      "position_return_pct": 10.0, "spy_return_pct": None, "holding_days": 5}]
        out = attribute_returns(positions, starting_equity=10000.0)
        assert out["positions"][0]["excess_gain"] is None
        assert out["total_excess_gain"] is None  # no excess values at all


class TestRegimeConditionalEdge:
    """regime_conditional_edge — trend/chop split of daily excess returns
    (owner ask 2026-07-22: the decision-relevant, faster-provable clock)."""

    def _series(self, excesses, dists, spy_ret=0.001):
        """Build port/spy value series producing the given daily excess
        returns, with spy_dist_pct classifying each return day."""
        spy = [100.0]
        port = [100.0]
        for e in excesses:
            spy.append(spy[-1] * (1 + spy_ret))
            port.append(port[-1] * (1 + spy_ret + e))
        return port, spy, [0.0] + list(dists)

    def test_splits_by_threshold(self):
        from backend.edge_metrics import regime_conditional_edge
        # 5 trend days (mean +50 bps excess), 5 chop days (mean -30 bps) —
        # values varied: zero within-bucket variance is (correctly) rejected
        excesses = [0.004, 0.006, 0.005, 0.003, 0.007,
                    -0.002, -0.004, -0.003, -0.001, -0.005]
        dists = [2.0] * 5 + [0.5] * 5
        port, spy, dist = self._series(excesses, dists)
        out = regime_conditional_edge(port, spy, dist)
        assert out["trend"]["n_days"] == 5
        assert out["chop"]["n_days"] == 5
        assert abs(out["trend"]["mean_daily_excess_bps"] - 50.0) < 1.0
        assert abs(out["chop"]["mean_daily_excess_bps"] + 30.0) < 1.0

    def test_positive_mean_gets_required_days(self):
        from backend.edge_metrics import regime_conditional_edge
        import random
        rng = random.Random(7)
        excesses = [0.003 + rng.uniform(-0.01, 0.01) for _ in range(40)]
        dists = [2.5] * 40
        port, spy, dist = self._series(excesses, dists)
        out = regime_conditional_edge(port, spy, dist)
        assert out["trend"]["required_days"] > 0
        assert out["trend"]["additional_days_needed"] >= 0
        assert out["chop"] is None  # no chop days seeded

    def test_none_dist_days_skipped_and_small_sample_none(self):
        from backend.edge_metrics import regime_conditional_edge
        port, spy, dist = self._series([0.005, 0.004, -0.002], [2.0, None, 2.0])
        out = regime_conditional_edge(port, spy, dist)
        # Only 2 usable trend days (< 3) and 0 chop → whole result None
        assert out is None

    def test_below_ma_lands_in_chop(self):
        from backend.edge_metrics import regime_conditional_edge
        excesses = [-0.002] * 4
        dists = [-1.0] * 4  # below the 50MA
        port, spy, dist = self._series(excesses, dists)
        out = regime_conditional_edge(port, spy, dist)
        assert out["chop"]["n_days"] == 4
        assert out["trend"] is None


class TestBootstrapRegimeExcess:
    """bootstrap_regime_excess — moving-block bootstrap CIs on the same
    paired daily excess series regime_conditional_edge tests (PM program
    2026-08-25: fat-tail/autocorrelation-robust intervals for the owner
    gate; deterministic for a fixed seed)."""

    def _series(self, excesses, dists, spy_ret=0.001):
        spy = [100.0]
        port = [100.0]
        for e in excesses:
            spy.append(spy[-1] * (1 + spy_ret))
            port.append(port[-1] * (1 + spy_ret + e))
        return port, spy, [0.0] + list(dists)

    def _mixed(self, n_trend=30, n_chop=20, seed=11):
        import random
        rng = random.Random(seed)
        excesses = ([0.003 + rng.uniform(-0.006, 0.006) for _ in range(n_trend)]
                    + [-0.002 + rng.uniform(-0.006, 0.006) for _ in range(n_chop)])
        dists = [2.0] * n_trend + [0.5] * n_chop
        return self._series(excesses, dists)

    def test_buckets_and_point_estimates_match_series(self):
        from backend.edge_metrics import bootstrap_regime_excess
        port, spy, dist = self._mixed()
        out = bootstrap_regime_excess(port, spy, dist)
        assert out["overall"]["n_days"] == 50
        assert out["trend"]["n_days"] == 30
        assert out["chop"]["n_days"] == 20
        # Point estimate is the sample mean, not a bootstrap artifact
        assert out["trend"]["mean_daily_excess_bps"] > 0
        assert out["chop"]["mean_daily_excess_bps"] < out["trend"]["mean_daily_excess_bps"]
        # CI brackets the point estimate
        for key in ("overall", "trend", "chop"):
            b = out[key]
            assert b["ci_low_annualized_pct"] <= b["alpha_annualized_pct"] <= b["ci_high_annualized_pct"]
            assert 0.0 <= b["prob_positive_pct"] <= 100.0

    def test_deterministic_for_fixed_seed(self):
        from backend.edge_metrics import bootstrap_regime_excess
        port, spy, dist = self._mixed()
        a = bootstrap_regime_excess(port, spy, dist)
        b = bootstrap_regime_excess(port, spy, dist)
        assert a == b
        c = bootstrap_regime_excess(port, spy, dist, seed=1)
        assert c["overall"]["ci_low_annualized_pct"] != a["overall"]["ci_low_annualized_pct"] or \
               c["overall"]["ci_high_annualized_pct"] != a["overall"]["ci_high_annualized_pct"]

    def test_strong_edge_excludes_zero(self):
        from backend.edge_metrics import bootstrap_regime_excess
        import random
        rng = random.Random(3)
        # Large stable positive excess: CI must exclude zero, P(>0) ~ 100
        excesses = [0.005 + rng.uniform(-0.001, 0.001) for _ in range(60)]
        port, spy, dist = self._series(excesses, [2.0] * 60)
        out = bootstrap_regime_excess(port, spy, dist)
        assert out["trend"]["excludes_zero_95"] is True
        assert out["trend"]["ci_low_annualized_pct"] > 0
        assert out["trend"]["prob_positive_pct"] > 99.0
        assert out["chop"] is None

    def test_small_bucket_is_none_and_tiny_series_is_none(self):
        from backend.edge_metrics import bootstrap_regime_excess
        port, spy, dist = self._mixed(n_trend=30, n_chop=4)
        out = bootstrap_regime_excess(port, spy, dist)
        assert out["chop"] is None          # 4 < 8 days
        assert out["trend"] is not None
        port, spy, dist = self._series([0.001] * 5, [2.0] * 5)
        assert bootstrap_regime_excess(port, spy, dist) is None  # 5 < 8 pairs

    def test_matches_regime_test_series_filtering(self):
        from backend.edge_metrics import bootstrap_regime_excess, regime_conditional_edge
        # None-dist days must be dropped identically in both functions
        # (excesses varied — zero within-bucket variance is rejected by the
        # t-test side and would make the comparison vacuous)
        excesses = [0.003 + 0.0005 * (i % 5) for i in range(12)] + [0.004] * 12
        dists = [2.0] * 12 + [None] * 12
        port, spy, dist = self._series(excesses, dists)
        boot = bootstrap_regime_excess(port, spy, dist)
        reg = regime_conditional_edge(port, spy, dist)
        assert boot["overall"]["n_days"] == 12
        assert reg["trend"]["n_days"] == 12
