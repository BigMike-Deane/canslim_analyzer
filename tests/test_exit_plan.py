"""Tests for backend/exit_plan.py — the read-only AI Portfolio exit-plan view.

These pin the trigger math to the shared threshold helpers so a future change to
stop/trailing/take-profit logic that forgets to update the display surface gets
caught here.
"""

import pytest

from backend.exit_plan import compute_exit_plan


def _kinds(plan):
    return {t["kind"] for t in plan["triggers"]}


def _by_kind(plan, kind):
    for t in plan["triggers"]:
        if t["kind"] == kind:
            return t
    return None


class TestStopLoss:
    def test_stop_price_is_profile_pct_below_entry(self):
        # nostate_cs_bear overrides the normal stop to 7% (not the 8% default).
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=110.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        stop = _by_kind(plan, "stop_loss")
        assert stop is not None
        assert stop["price"] == pytest.approx(93.0, abs=0.01)  # 7% below 100
        assert stop["direction"] == "down"

    def test_bearish_market_uses_tighter_stop(self):
        normal = compute_exit_plan(
            cost_basis=100.0, current_price=100.0, strategy="balanced",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
            is_bearish_market=False,
        )
        bear = compute_exit_plan(
            cost_basis=100.0, current_price=100.0, strategy="balanced",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
            is_bearish_market=True,
        )
        # Bearish stop sits at a higher price (tighter) than the normal stop.
        assert _by_kind(bear, "stop_loss")["price"] > _by_kind(normal, "stop_loss")["price"]

    def test_distance_is_downside_cushion(self):
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=110.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        # stop 93 vs current 110 → (93/110 - 1) ≈ -15.45% → magnitude 15.5
        assert _by_kind(plan, "stop_loss")["distance_pct"] == pytest.approx(15.5, abs=0.1)


class TestTrailingStop:
    def test_no_trailing_below_activation_tier(self):
        # Peak only +1.7% over cost → below the +5% tier → no trailing trigger.
        plan = compute_exit_plan(
            cost_basis=372.73, current_price=374.65, peak_price=379.10,
            pyramid_count=0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        assert "trailing_stop" not in _kinds(plan)

    def test_trailing_uses_tier_and_pyramid_widening(self):
        # MU: peak 1086.77 over cost 682.44 → +59% peak gain → 25% tier;
        # pyramid_count 2 → +4% → 29% off peak.
        plan = compute_exit_plan(
            cost_basis=682.44, current_price=1005.45, peak_price=1086.77,
            pyramid_count=2, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        trail = _by_kind(plan, "trailing_stop")
        assert trail is not None
        assert trail["price"] == pytest.approx(1086.77 * 0.71, abs=0.05)  # 29% off peak

    def test_trailing_anchored_at_peak_not_entry(self):
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=150.0, peak_price=160.0,
            pyramid_count=0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        trail = _by_kind(plan, "trailing_stop")
        # 60% peak gain → 25% tier → 160 * 0.75 = 120
        assert trail["price"] == pytest.approx(120.0, abs=0.01)


class TestTakeProfit:
    def test_target_is_pct_above_entry(self):
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=120.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        tp = _by_kind(plan, "take_profit")
        assert tp["price"] == pytest.approx(175.0, abs=0.01)
        assert tp["direction"] == "up"
        assert tp["reached"] is False

    def test_reached_flag_when_past_target(self):
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=200.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        assert _by_kind(plan, "take_profit")["reached"] is True


class TestScoreExit:
    def test_score_exit_condition_present(self):
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=110.0, current_score=78,
            strategy="nostate_cs_bear", sell_score_threshold=60,
            take_profit_pct=75, stop_loss_pct=8,
        )
        score = _by_kind(plan, "score_exit")
        assert score["threshold"] == 60
        assert score["current_score"] == 78
        assert score["price"] is None


class TestNearest:
    def test_nearest_is_closest_price_trigger(self):
        # Flat position: stop is much closer than the far-off take-profit target.
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=101.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        assert plan["nearest_kind"] == "stop_loss"

    def test_score_exit_never_nearest(self):
        # score_exit has no price/distance, so it can't be picked as nearest.
        plan = compute_exit_plan(
            cost_basis=100.0, current_price=120.0, peak_price=130.0,
            pyramid_count=0, current_score=70, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        assert plan["nearest_kind"] in {"stop_loss", "trailing_stop", "take_profit"}


class TestGuards:
    def test_missing_price_yields_only_score_exit(self):
        plan = compute_exit_plan(
            cost_basis=None, current_price=None, current_score=70,
            strategy="nostate_cs_bear", sell_score_threshold=60,
            take_profit_pct=75, stop_loss_pct=8,
        )
        assert _kinds(plan) == {"score_exit"}
        assert plan["nearest_kind"] is None

    def test_zero_cost_basis_is_safe(self):
        plan = compute_exit_plan(
            cost_basis=0.0, current_price=10.0, strategy="nostate_cs_bear",
            sell_score_threshold=60, take_profit_pct=75, stop_loss_pct=8,
        )
        # No divide-by-zero; no price triggers emitted.
        assert "stop_loss" not in _kinds(plan)
