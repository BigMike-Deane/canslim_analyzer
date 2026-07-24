"""
Tests for backend/exit_lab.py — counterfactual exit-rule replay.

The parity tests here are the sync contract: exit_lab deliberately does NOT
import trading_engine (it must stay pure/standalone), so these tests pin its
band/ladder semantics to the live functions. If live exit tiers change,
these fail — update CHAMPION_BANDS / CHAMPION_LADDER to match.
"""

from datetime import date

import pytest

from backend.exit_lab import (
    CHAMPION_BANDS,
    ExitConfig,
    bootstrap_mean_ci,
    ladder_action,
    paired_diff,
    reconstruct_episodes,
    replay_episode,
    run_grid,
    stop_slippage,
    trailing_band_for,
)
from backend.trading_engine import (
    apply_pyramid_widening,
    get_partial_profit_action,
    get_trailing_stop_pct,
)

# The champion profile's trailing overrides (nostate_optimized / nostate_cs_bear
# in config/default.yaml) — gain_10_to_20 is 6, not the code default 8.
CHAMPION_PROFILE = {"trailing_stops": {
    "gain_50_plus": 25, "gain_30_to_50": 18, "gain_20_to_30": 12,
    "gain_10_to_20": 6, "gain_5_to_10": 4,
}}


def bar(d, o, h, l, c):  # noqa: E741 - l is fine here
    return {"date": d, "open": o, "high": h, "low": l, "close": c}


# ---------------------------------------------------------------------------
# Parity with trading_engine
# ---------------------------------------------------------------------------

class TestTradingEngineParity:
    def test_trailing_bands_match_live_champion_profile(self):
        cfg = ExitConfig(name="t")
        for peak_gain in [x * 0.5 for x in range(0, 260)]:  # 0..130%
            live = get_trailing_stop_pct(peak_gain, CHAMPION_PROFILE)
            lab = trailing_band_for(peak_gain, cfg, pyramid_count=0)
            assert lab == live, f"band mismatch at peak_gain={peak_gain}: {lab} != {live}"

    def test_pyramid_widening_matches_live(self):
        cfg = ExitConfig(name="t")
        for pyramids in range(0, 6):
            for peak_gain in (7, 15, 25, 40, 80):
                live = apply_pyramid_widening(
                    get_trailing_stop_pct(peak_gain, CHAMPION_PROFILE), pyramids)
                lab = trailing_band_for(peak_gain, cfg, pyramid_count=pyramids)
                assert lab == pytest.approx(live)

    def test_ladder_matches_live_partial_profit_action(self):
        cfg = ExitConfig(name="t")
        for gain in range(0, 80, 5):
            for taken in (0, 10, 25, 50, 60, 75):
                live = get_partial_profit_action(gain, current_score=100.0,
                                                 partial_taken=taken)
                lab = ladder_action(gain, taken, cfg)
                live_take = live["take_pct"] if live else None
                assert lab == live_take, (
                    f"ladder mismatch at gain={gain}, taken={taken}: "
                    f"{lab} != {live_take}")


# ---------------------------------------------------------------------------
# Episode reconstruction
# ---------------------------------------------------------------------------

def trade(user_id, ticker, action, shares, price, when, sell_reason=None):
    return {
        "user_id": user_id, "ticker": ticker, "action": action,
        "shares": shares, "price": price, "executed_at": when,
        "signal_factors": {"sell_reason": sell_reason} if sell_reason else None,
    }


class TestReconstructEpisodes:
    def test_full_lifecycle_with_pyramid_and_partial(self):
        eps = reconstruct_episodes([
            trade(1, "AAA", "BUY", 100, 10.0, "2026-01-05T15:00:00"),
            trade(1, "AAA", "PYRAMID", 20, 12.0, "2026-01-10T15:00:00"),
            trade(1, "AAA", "SELL", 30, 13.0, "2026-02-01T15:00:00", "PARTIAL PROFIT"),
            trade(1, "AAA", "SELL", 90, 14.0, "2026-03-01T15:00:00", "TRAILING STOP"),
        ])
        assert len(eps) == 1
        ep = eps[0]
        assert ep.closed and ep.exit_date == date(2026, 3, 1)
        assert ep.entry_price == 10.0 and ep.entry_date == date(2026, 1, 5)
        assert ep.pyramid_count == 1
        # cost 100*10 + 20*12 = 1240; proceeds 30*13 + 90*14 = 1650
        assert ep.actual_return_pct == pytest.approx((1650 / 1240 - 1) * 100)

    def test_rebuy_after_close_starts_new_episode(self):
        eps = reconstruct_episodes([
            trade(1, "BBB", "BUY", 10, 5.0, "2026-01-05T15:00:00"),
            trade(1, "BBB", "SELL", 10, 6.0, "2026-01-20T15:00:00", "STOP LOSS"),
            trade(1, "BBB", "BUY", 10, 5.5, "2026-02-05T15:00:00"),
        ])
        assert len(eps) == 2
        assert eps[0].closed and not eps[1].closed

    def test_orphan_sell_and_cross_user_isolation(self):
        eps = reconstruct_episodes([
            trade(1, "CCC", "SELL", 10, 6.0, "2026-01-02T15:00:00", "MANUAL"),
            trade(1, "CCC", "BUY", 10, 5.0, "2026-01-05T15:00:00"),
            trade(2, "CCC", "BUY", 10, 5.1, "2026-01-05T15:00:00"),
        ])
        assert len(eps) == 2
        assert all(not e.closed for e in eps)
        assert {e.user_id for e in eps} == {1, 2}


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

class TestReplayEpisode:
    def test_hard_stop_with_gap_fills_at_open(self):
        cfg = ExitConfig(name="t", stop_loss_pct=7.0)
        bars = [
            bar("2026-01-05", 100, 101, 99, 100),
            bar("2026-01-06", 90, 91, 89, 90),  # gaps far below the 93 stop
        ]
        r = replay_episode(100.0, bars, cfg)
        assert len(r.legs) == 1
        _, frac, ret, kind = r.legs[0]
        assert kind == "STOP" and frac == 1.0
        assert ret == pytest.approx(-10.0)  # filled at the open, not at -7%
        assert not r.censored

    def test_trailing_stop_fires_at_close_from_intraday_peak(self):
        cfg = ExitConfig(name="t", ladder_enabled=False)
        # Peak 112 intraday (tier 10-20% -> 6% band); close 104 = -7.1% off peak
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 105, 112, 104, 104),
        ]
        r = replay_episode(100.0, bars, cfg)
        assert [leg[3] for leg in r.legs] == ["TRAIL"]
        assert r.weighted_return_pct == pytest.approx(4.0)
        assert r.exit_efficiency == pytest.approx(1.04 / 1.12)

    def test_below_min_tier_no_trailing(self):
        cfg = ExitConfig(name="t", ladder_enabled=False)
        # Peak +4% (below the 5% min tier) then collapse to -3%: no trigger
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 101, 104, 97, 97),
        ]
        r = replay_episode(100.0, bars, cfg)
        assert r.censored and r.legs[-1][3] == "OPEN"
        assert r.weighted_return_pct == pytest.approx(-3.0)

    def test_ladder_partial_then_trail_remainder(self):
        cfg = ExitConfig(name="t")
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 120, 128, 118, 127),  # +27% close: ladder sells 25%
            bar("2026-01-07", 127, 127.5, 112, 112),  # peak 128 (+28%, 12% band), -12.5% drop
        ]
        r = replay_episode(100.0, bars, cfg)
        kinds = [leg[3] for leg in r.legs]
        assert kinds == ["LADDER_25", "TRAIL"]
        frac_ladder, ret_ladder = r.legs[0][1], r.legs[0][2]
        frac_trail, ret_trail = r.legs[1][1], r.legs[1][2]
        assert frac_ladder == pytest.approx(0.25) and ret_ladder == pytest.approx(27.0)
        assert frac_trail == pytest.approx(0.75) and ret_trail == pytest.approx(12.0)
        assert r.weighted_return_pct == pytest.approx(0.25 * 27 + 0.75 * 12)

    def test_partial_on_trailing_resets_peak_for_pyramided_position(self):
        cfg = ExitConfig(name="t", ladder_enabled=False)
        # 2 pyramids -> band widened +4 (10-20% tier: 6 -> 10)
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 110, 118, 105, 106),  # -10.2% off 118 peak: partial 50%
            bar("2026-01-07", 106, 107, 106, 106.5),
        ]
        r = replay_episode(100.0, bars, cfg, pyramid_count=2)
        assert r.legs[0][3] == "PARTIAL_TRAIL"
        assert r.legs[0][1] == pytest.approx(0.5)
        # Remainder survives day 3 (peak was reset to 106) and is censored open
        assert r.censored and r.legs[-1][3] == "OPEN"

    def test_stop_checked_before_trailing(self):
        cfg = ExitConfig(name="t")
        # Day gaps down through the stop AND would satisfy trailing: STOP wins
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 108, 110, 107, 109),
            bar("2026-01-07", 91, 92, 90, 91),
        ]
        r = replay_episode(100.0, bars, cfg)
        assert r.legs[0][3] == "STOP"


# ---------------------------------------------------------------------------
# Stats & grid plumbing
# ---------------------------------------------------------------------------

class TestStats:
    def test_bootstrap_ci_deterministic_and_brackets_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        lo1, hi1 = bootstrap_mean_ci(vals, n_boot=500, seed=7)
        lo2, hi2 = bootstrap_mean_ci(vals, n_boot=500, seed=7)
        assert (lo1, hi1) == (lo2, hi2)
        assert lo1 < 3.5 < hi1

    def test_paired_diff_constant_shift_significant(self):
        cfg_a, cfg_b = ExitConfig(name="a"), ExitConfig(name="b")
        bars = [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 105, 112, 104, 104),
        ]
        base, variant = [], []
        for i, ticker in enumerate(["X", "Y", "Z", "W", "V"]):
            ra = replay_episode(100.0, bars, cfg_a, ticker=ticker, user_id=1)
            rb = replay_episode(100.0, bars, cfg_b, ticker=ticker, user_id=1)
            rb.weighted_return_pct += 2.0  # constant +2pp shift
            base.append(ra)
            variant.append(rb)
        d = paired_diff(variant, base, n_boot=300, seed=3)
        assert d["n"] == 5
        assert d["mean_diff_pp"] == pytest.approx(2.0)
        assert d["significant"] and d["pct_episodes_better"] == 1.0

    def test_run_grid_skips_missing_prices_and_pairs_baseline(self):
        trades = [
            trade(1, "AAA", "BUY", 10, 100.0, "2026-01-05T15:00:00"),
            trade(1, "AAA", "SELL", 10, 104.0, "2026-01-06T20:00:00", "TRAILING STOP"),
            trade(1, "NOPX", "BUY", 10, 50.0, "2026-01-05T15:00:00"),
            trade(1, "NOPX", "SELL", 10, 55.0, "2026-01-06T20:00:00", "TRAILING STOP"),
        ]
        episodes = reconstruct_episodes(trades)
        prices = {"AAA": [
            bar("2026-01-05", 100, 100, 100, 100),
            bar("2026-01-06", 105, 112, 104, 104),
            bar("2026-01-07", 104, 104, 104, 104),
        ]}
        out = run_grid(episodes, prices)
        assert out["skipped_no_prices"] == 1
        summary = out["summary"]
        assert summary["baseline_champion"]["n"] == 1
        assert "vs_baseline" not in summary["baseline_champion"]
        assert all("vs_baseline" in s for name, s in summary.items()
                   if name != "baseline_champion" and s.get("n"))

    def test_stop_slippage(self):
        trades = [
            trade(1, "AAA", "SELL", 1, 1, "2026-01-05", "STOP LOSS"),
            trade(1, "BBB", "SELL", 1, 1, "2026-01-06", "STOP LOSS"),
            trade(1, "CCC", "SELL", 1, 1, "2026-01-07", "TRAILING STOP"),
        ]
        trades[0]["signal_factors"].update({"gain_pct": -9.5, "stop_pct": 8.0})
        trades[1]["signal_factors"].update({"gain_pct": -8.0, "stop_pct": 8.0})
        s = stop_slippage(trades)
        assert s["n"] == 2
        assert s["mean_slippage_pp"] == pytest.approx(0.75)
        assert s["max_slippage_pp"] == pytest.approx(1.5)
