"""
Tests for trading_engine.py — shared trading logic between ai_trader and backtester.

These functions are CRITICAL: they drive both live trading and historical
backtesting decisions. Every function extracted into trading_engine.py must
have direct unit test coverage here.
"""
import math
import pytest
from unittest.mock import MagicMock

from backend.trading_engine import (
    apply_pyramid_widening,
    check_score_stability_from_history,
    calculate_composite_score,
    calculate_position_size_pct,
    apply_position_size_multipliers,
    categorize_sell_reason,
    get_tightened_trailing_stop,
    sanitize_signal_factors,
    get_trailing_stop_pct,
    get_partial_profit_action,
    calculate_base_quality_bonus,
    calculate_entry_signals,
    evaluate_score_crash,
)


# ── Pyramid Widening ──────────────────────────────────────────────────────────

class TestApplyPyramidWidening:
    def test_no_pyramid_returns_unchanged(self):
        assert apply_pyramid_widening(18.0, 0) == 18.0

    def test_one_pyramid_adds_2pct(self):
        assert apply_pyramid_widening(18.0, 1) == 20.0

    def test_two_pyramids_adds_4pct(self):
        assert apply_pyramid_widening(18.0, 2) == 22.0

    def test_caps_at_6pct_widening(self):
        # 4 pyramids = 8% but capped at 6%
        assert apply_pyramid_widening(18.0, 4) == 24.0

    def test_caps_at_6pct_extreme(self):
        assert apply_pyramid_widening(18.0, 10) == 24.0


# ── Score Stability ───────────────────────────────────────────────────────────

class TestCheckScoreStabilityFromHistory:
    def test_empty_history_returns_stable(self):
        result = check_score_stability_from_history([], 40)
        assert result["is_stable"] is True
        assert result["avg_score"] == 40

    def test_single_score_returns_stable(self):
        result = check_score_stability_from_history([40], 40)
        assert result["is_stable"] is True
        assert result["consecutive_low"] == 1  # below default threshold 50

    def test_consistently_low_scores_are_stable(self):
        """Multiple consecutive low scores = real drop, not blip."""
        result = check_score_stability_from_history([30, 35, 32], 33)
        assert result["is_stable"] is True
        assert result["consecutive_low"] == 3

    def test_sudden_drop_from_high_is_blip(self):
        """Single low score after high average = potential blip."""
        result = check_score_stability_from_history([75, 80, 72, 35], 35)
        # avg = 65.5, current=35, variance=30.5>15, consecutive_low=1<2
        assert result["is_stable"] is False

    def test_two_consecutive_low_not_blip(self):
        """Two consecutive low scores = real drop, not blip."""
        result = check_score_stability_from_history([80, 75, 35, 30], 30)
        assert result["is_stable"] is True
        assert result["consecutive_low"] >= 2

    def test_high_score_always_stable(self):
        """Score above threshold is never flagged as blip."""
        result = check_score_stability_from_history([80, 75, 70], 65)
        assert result["is_stable"] is True

    def test_custom_threshold(self):
        result = check_score_stability_from_history([85, 80, 55], 55, threshold=60)
        # avg=73.3, current=55, avg > threshold+10 (73.3>70), var=18.3>15, consecutive_low=1 — blip
        assert result["is_stable"] is False

    def test_recent_low_count_with_bounce(self):
        """2 of last 3 low even though not consecutive (bounce in the middle)."""
        result = check_score_stability_from_history([30, 65, 35], 35)
        # consecutive_low = 1 (only last), but recent_low_count = 2 (30 and 35 in last 3)
        assert result["consecutive_low"] == 1
        assert result["recent_low_count"] == 2

    def test_recent_low_count_all_low(self):
        """All recent scores low → both consecutive and recent_low_count high."""
        result = check_score_stability_from_history([30, 35, 32], 33)
        assert result["consecutive_low"] == 3
        assert result["recent_low_count"] == 3

    def test_recent_low_count_early_return(self):
        """Early return path includes recent_low_count."""
        result = check_score_stability_from_history([40], 40)
        assert "recent_low_count" in result
        assert result["recent_low_count"] == 1  # below default threshold 50

    def test_recent_low_count_empty_history(self):
        """Empty history early return includes recent_low_count."""
        result = check_score_stability_from_history([], 60)
        assert "recent_low_count" in result
        assert result["recent_low_count"] == 0  # above threshold


# ── Composite Score ───────────────────────────────────────────────────────────

class TestCalculateCompositeScore:
    def test_basic_calculation_default_weights(self):
        score = calculate_composite_score(
            growth_projection=80, effective_score=75,
            momentum_score=30, breakout_bonus=10,
            pre_breakout_bonus=0, base_quality_bonus=8,
            extended_penalty=0, profile={},
        )
        # 80*0.25 + 75*0.25 + 30*0.20 + 10*0.20 + 8*0.10
        expected = 20 + 18.75 + 6 + 2 + 0.8
        assert abs(score - expected) < 0.01

    def test_extended_penalty_reduces_score(self):
        base = calculate_composite_score(
            growth_projection=80, effective_score=75,
            momentum_score=30, breakout_bonus=10,
            pre_breakout_bonus=0, base_quality_bonus=8,
            extended_penalty=0, profile={},
        )
        penalized = calculate_composite_score(
            growth_projection=80, effective_score=75,
            momentum_score=30, breakout_bonus=10,
            pre_breakout_bonus=0, base_quality_bonus=8,
            extended_penalty=-20, profile={},
        )
        assert penalized == base - 20

    def test_coiled_spring_bonus_adds(self):
        without_cs = calculate_composite_score(
            growth_projection=50, effective_score=50,
            momentum_score=20, breakout_bonus=0,
            pre_breakout_bonus=0, base_quality_bonus=0,
            extended_penalty=0, profile={},
        )
        with_cs = calculate_composite_score(
            growth_projection=50, effective_score=50,
            momentum_score=20, breakout_bonus=0,
            pre_breakout_bonus=0, base_quality_bonus=0,
            extended_penalty=0, coiled_spring_bonus=15, profile={},
        )
        assert with_cs == without_cs + 15

    def test_profile_weight_overrides(self):
        profile = {'scoring_weights': {
            'growth_projection': 0.50,
            'canslim_score': 0.50,
            'momentum': 0.0,
            'breakout': 0.0,
            'base_quality': 0.0,
        }}
        score = calculate_composite_score(
            growth_projection=100, effective_score=80,
            momentum_score=50, breakout_bonus=50,
            pre_breakout_bonus=0, base_quality_bonus=50,
            extended_penalty=0, profile=profile,
        )
        assert abs(score - 90.0) < 0.01  # 100*0.5 + 80*0.5


# ── Position Sizing ───────────────────────────────────────────────────────────

class TestCalculatePositionSizePct:
    def _mock_config(self, conv_enabled=False):
        cfg = MagicMock()
        if conv_enabled:
            cfg.get.return_value = {
                'enabled': True, 'min_pct': 8.0, 'max_pct': 20.0,
                'score_floor': 30, 'score_ceiling': 75,
            }
        else:
            cfg.get.return_value = {'enabled': False}
        return cfg

    def test_default_sizing_respects_max_positions(self):
        cfg = self._mock_config(conv_enabled=False)
        pct = calculate_position_size_pct(50, 8, {}, regime_max_pct=12.0, yaml_config=cfg)
        assert pct >= 90.0 / 8  # At least min position
        assert pct <= 12.0  # At most regime cap

    def test_conviction_sizing_scales_with_score(self):
        cfg = self._mock_config(conv_enabled=True)
        low = calculate_position_size_pct(30, 8, {}, yaml_config=cfg)  # Floor
        high = calculate_position_size_pct(75, 8, {}, yaml_config=cfg)  # Ceiling
        assert low < high
        assert abs(low - 8.0) < 0.01  # Min at floor
        assert abs(high - 20.0) < 0.01  # Max at ceiling

    def test_conviction_clamps_below_floor(self):
        cfg = self._mock_config(conv_enabled=True)
        pct = calculate_position_size_pct(10, 8, {}, yaml_config=cfg)  # Below floor
        assert abs(pct - 8.0) < 0.01


# ── Position Size Multipliers ─────────────────────────────────────────────────

class TestApplyPositionSizeMultipliers:
    def _base_args(self, **overrides):
        defaults = dict(
            position_pct=12.0, pre_breakout_bonus=0, has_base=False,
            is_breaking_out=False, breakout_volume_ratio=0,
            is_coiled_spring=False, in_soft_zone=False, soft_zone_mult=1.0,
            is_correction_zone=False, cz_position_mult=1.0,
            heat_penalty_active=False, market_state_multiplier=1.0,
            profile={}, yaml_config=MagicMock(get=lambda *a: {}),
        )
        defaults.update(overrides)
        return defaults

    def test_no_multipliers_returns_unchanged(self):
        result = apply_position_size_multipliers(**self._base_args())
        assert abs(result - 12.0) < 0.01

    def test_heat_penalty_halves_position(self):
        result = apply_position_size_multipliers(**self._base_args(heat_penalty_active=True))
        assert abs(result - 6.0) < 0.01

    def test_pre_breakout_strong_multiplier(self):
        result = apply_position_size_multipliers(**self._base_args(
            pre_breakout_bonus=40, has_base=True,
        ))
        assert result > 12.0  # 12 * 1.40 = 16.8

    def test_coiled_spring_boost(self):
        cfg = MagicMock()
        cfg.get.return_value = {'position_multiplier': 1.25}
        result = apply_position_size_multipliers(**self._base_args(
            is_coiled_spring=True, yaml_config=cfg,
        ))
        assert abs(result - 15.0) < 0.01  # 12 * 1.25

    def test_soft_zone_reduces_position(self):
        result = apply_position_size_multipliers(**self._base_args(
            in_soft_zone=True, soft_zone_mult=0.5,
        ))
        assert abs(result - 6.0) < 0.01

    def test_correction_zone_reduces_position(self):
        result = apply_position_size_multipliers(**self._base_args(
            is_correction_zone=True, cz_position_mult=0.60,
        ))
        assert abs(result - 7.2) < 0.01

    def test_profile_max_cap(self):
        result = apply_position_size_multipliers(**self._base_args(
            position_pct=30.0, profile={'max_single_position_pct': 20},
        ))
        assert abs(result - 20.0) < 0.01

    def test_multiple_multipliers_stack(self):
        """Heat + soft zone stack multiplicatively."""
        result = apply_position_size_multipliers(**self._base_args(
            heat_penalty_active=True, in_soft_zone=True, soft_zone_mult=0.5,
        ))
        assert abs(result - 3.0) < 0.01  # 12 * 0.5 * 0.5


# ── Sell Reason Categorization ────────────────────────────────────────────────

class TestCategorizeSellReason:
    @pytest.mark.parametrize("reason,expected", [
        ("STOP LOSS: Down 8.0%", "STOP LOSS"),
        ("TRAILING STOP: Peak $50 -> $40 (-20%)", "TRAILING STOP"),
        ("PARTIAL TRAILING STOP: 50% at peak", "PARTIAL TRAILING"),
        ("SCORE CRASH: 80 → 40", "SCORE CRASH"),
        ("TAKE PROFIT 75%: Up 80%", "TAKE PROFIT"),
        ("PARTIAL PROFIT 25%: Up 30%", "PARTIAL PROFIT"),
        ("PROTECT GAINS: Earnings approach", "PROTECT GAINS"),
        ("WEAK POSITION: Low score 35", "WEAK POSITION"),
        ("CIRCUIT BREAKER: Portfolio drawdown", "CIRCUIT BREAKER"),
        ("PRE-EARNINGS: Tightened stop", "PRE-EARNINGS"),
        ("Some unknown sell reason", "OTHER"),
    ])
    def test_categorization(self, reason, expected):
        assert categorize_sell_reason(reason) == expected


# ── Pre-Earnings Trailing Stop Tightening ─────────────────────────────────────

class TestGetTightenedTrailingStop:
    def test_tightens_by_factor(self):
        profile = {'trailing_stops': {'gain_50_plus': 25}}
        result = get_tightened_trailing_stop(55.0, profile, tighten_factor=0.50)
        assert abs(result - 12.5) < 0.01  # 25 * 0.50

    def test_returns_none_below_threshold(self):
        result = get_tightened_trailing_stop(3.0, {})
        assert result is None

    def test_tightens_5_to_10_tier(self):
        result = get_tightened_trailing_stop(7.0, {'trailing_stops': {'gain_5_to_10': 4}})
        assert abs(result - 2.0) < 0.01  # 4 * 0.50

    def test_custom_tighten_factor(self):
        profile = {'trailing_stops': {'gain_20_to_30': 12}}
        result = get_tightened_trailing_stop(25.0, profile, tighten_factor=0.75)
        assert abs(result - 9.0) < 0.01


# ── Sanitize Signal Factors ───────────────────────────────────────────────────

class TestSanitizeSignalFactors:
    def test_replaces_nan_with_zero(self):
        result = sanitize_signal_factors({"score": float('nan'), "gain": 5.0})
        assert result["score"] == 0
        assert result["gain"] == 5.0

    def test_replaces_inf_with_zero(self):
        result = sanitize_signal_factors({"val": float('inf')})
        assert result["val"] == 0

    def test_replaces_neg_inf_with_zero(self):
        result = sanitize_signal_factors({"val": float('-inf')})
        assert result["val"] == 0

    def test_preserves_strings_and_ints(self):
        result = sanitize_signal_factors({"reason": "STOP LOSS", "count": 3})
        assert result == {"reason": "STOP LOSS", "count": 3}

    def test_empty_dict_returns_empty(self):
        assert sanitize_signal_factors({}) == {}

    def test_none_returns_none(self):
        assert sanitize_signal_factors(None) is None

    def test_mixed_nan_and_valid(self):
        sf = {"a": float('nan'), "b": 1.5, "c": "text", "d": float('inf')}
        result = sanitize_signal_factors(sf)
        assert result == {"a": 0, "b": 1.5, "c": "text", "d": 0}
        # Verify all values are JSON-safe
        for v in result.values():
            if isinstance(v, float):
                assert not math.isnan(v) and not math.isinf(v)


# ── Score Crash Evaluation (additional edge cases) ────────────────────────────

class TestEvaluateScoreCrashEdgeCases:
    def _mock_config(self):
        cfg = MagicMock()
        cfg.get.return_value = {
            'consecutive_required': 3,
            'threshold': 50,
            'drop_required': 20,
            'ignore_if_profitable_pct': 10,
        }
        return cfg

    def test_no_crash_when_drop_insufficient(self):
        """Score drop of 15 when 20 is required → no crash."""
        cfg = self._mock_config()
        stability = {"is_stable": True, "consecutive_low": 5, "avg_score": 60}
        result = evaluate_score_crash(60, 75, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result is None

    def test_no_crash_when_above_threshold(self):
        """Current score 55 (above threshold 50) → no crash."""
        cfg = self._mock_config()
        stability = {"is_stable": True, "consecutive_low": 5, "avg_score": 55}
        result = evaluate_score_crash(55, 80, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result is None

    def test_profitable_exception_skips_sell(self):
        """Profitable positions skip score crash sell."""
        cfg = self._mock_config()
        stability = {"is_stable": True, "consecutive_low": 5, "avg_score": 30}
        result = evaluate_score_crash(25, 80, gain_pct=15.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result is not None
        assert result["should_sell"] is False
        assert "profitable" in result["reason"]

    def test_blip_detection_skips_sell(self):
        cfg = self._mock_config()
        stability = {"is_stable": False, "consecutive_low": 1, "avg_score": 65}
        result = evaluate_score_crash(30, 80, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result["should_sell"] is False
        assert "blip" in result["reason"].lower()

    def test_insufficient_consecutive_and_recent_skips_sell(self):
        """Neither 3 consecutive nor 2-of-3 recent low — no sell"""
        cfg = self._mock_config()
        stability = {"is_stable": True, "consecutive_low": 1, "recent_low_count": 1, "avg_score": 35}
        result = evaluate_score_crash(25, 80, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result["should_sell"] is False

    def test_real_crash_triggers_sell_consecutive(self):
        """3+ consecutive low scans triggers sell"""
        cfg = self._mock_config()
        stability = {"is_stable": True, "consecutive_low": 4, "recent_low_count": 3, "avg_score": 30}
        result = evaluate_score_crash(25, 80, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result["should_sell"] is True
        assert "SCORE CRASH" in result["reason"]
        assert "consecutive" in result["reason"]

    def test_2_of_3_recent_low_triggers_sell(self):
        """2 of last 3 scans below threshold triggers sell even without 3 consecutive"""
        cfg = self._mock_config()
        # Only 1 consecutive (last scan was a bounce), but 2 of last 3 were low
        stability = {"is_stable": True, "consecutive_low": 1, "recent_low_count": 2, "avg_score": 35}
        result = evaluate_score_crash(25, 80, gain_pct=-5.0, stability=stability,
                                       profile={}, yaml_config=cfg)
        assert result["should_sell"] is True
        assert "SCORE CRASH" in result["reason"]
        assert "2-of-3" in result["reason"]


# ── Coverage Gaps ─────────────────────────────────────────────────────────────

class TestCalculateEntrySignalsCoverage:
    """Branches inside calculate_entry_signals that the scenario tests miss:
    week_52_high guard, weeks_in_base bonus, moderately-extended pivot path,
    and the four no-base pct_from_high tiers."""

    def test_returns_zeros_when_week_52_high_is_zero(self):
        # Line 290-297: guard against missing 52-week-high data.
        result = calculate_entry_signals(
            current_price=50, week_52_high=0, pivot_price=0,
            base_type="cup", weeks_in_base=8,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert result["momentum_score"] == 0
        assert result["breakout_bonus"] == 0
        assert result["pre_breakout_bonus"] == 0
        assert result["extended_penalty"] == 0
        assert result["entry_type"] == "standard"
        assert result["pct_from_pivot"] == 0

    def test_pre_breakout_with_long_base_adds_extra_bonus(self):
        # Line 311: weeks_in_base >= 10 adds +5 to pre_breakout_bonus.
        # Setup: 10% below pivot (in 5-15 zone), volume below 1.3 so the
        # other bonus doesn't fire.
        short_base = calculate_entry_signals(
            current_price=90, week_52_high=120, pivot_price=100,
            base_type="cup", weeks_in_base=6,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        long_base = calculate_entry_signals(
            current_price=90, week_52_high=120, pivot_price=100,
            base_type="cup", weeks_in_base=12,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert long_base["pre_breakout_bonus"] == short_base["pre_breakout_bonus"] + 5

    def test_moderately_extended_penalty(self):
        # Lines 333-334: pct_from_pivot in (-10, -5) -> -10 penalty, momentum 10.
        # current=107, pivot=100 -> pct_from_pivot = -7
        result = calculate_entry_signals(
            current_price=107, week_52_high=120, pivot_price=100,
            base_type="cup", weeks_in_base=8,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert result["extended_penalty"] == -10
        assert result["momentum_score"] == 10

    def test_no_base_near_high_low_score_penalized(self):
        # Lines 339-341: no base + within 2% of 52w high + score < 85
        # -> extended_penalty=-15, momentum=5.
        result = calculate_entry_signals(
            current_price=99, week_52_high=100, pivot_price=0,
            base_type="none", weeks_in_base=0,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert result["extended_penalty"] == -15
        assert result["momentum_score"] == 5

    def test_no_base_near_high_high_score_no_penalty(self):
        # Lines 342-343: no base + within 2% of 52w high + score >= 85
        # -> no extended penalty, momentum=12.
        result = calculate_entry_signals(
            current_price=99, week_52_high=100, pivot_price=0,
            base_type="none", weeks_in_base=0,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=90,
        )
        assert result["extended_penalty"] == 0
        assert result["momentum_score"] == 12

    def test_no_base_mid_distance_from_high(self):
        # Lines 348-349: no base + 10-25% off 52w high -> momentum=8.
        result = calculate_entry_signals(
            current_price=80, week_52_high=100, pivot_price=0,
            base_type="none", weeks_in_base=0,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert result["momentum_score"] == 8

    def test_no_base_far_from_high_penalized(self):
        # Lines 350-351: no base + > 25% off 52w high -> momentum=-5.
        result = calculate_entry_signals(
            current_price=70, week_52_high=100, pivot_price=0,
            base_type="none", weeks_in_base=0,
            is_breaking_out=False, breakout_volume_ratio=0,
            volume_ratio=1.0, effective_score=70,
        )
        assert result["momentum_score"] == -5


class TestDefaultArgumentFallbacks:
    """Lines 395-396, 438-439, 482-483: each function defaults its config / profile
    arg to a sensible value when None is passed. These guards exist so the
    functions can be called from contexts that don't have a profile yet
    (e.g. unit tests, ad-hoc scoring)."""

    def test_composite_score_with_profile_none(self):
        # Line 396: profile=None -> profile={} default
        score = calculate_composite_score(
            growth_projection=80, effective_score=75,
            momentum_score=30, breakout_bonus=10,
            pre_breakout_bonus=0, base_quality_bonus=8,
            extended_penalty=0, profile=None,
        )
        expected = 80 * 0.25 + 75 * 0.25 + 30 * 0.20 + 10 * 0.20 + 8 * 0.10
        assert abs(score - expected) < 0.01

    def test_position_size_pct_with_yaml_config_none(self):
        # Line 439: yaml_config=None -> falls back to module-level config.
        # We only need the line to execute; the returned size depends on
        # real config but must be a positive float.
        result = calculate_position_size_pct(
            composite_score=60, max_positions=8, profile={}, regime_max_pct=12.0,
            yaml_config=None,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_apply_multipliers_with_yaml_config_none(self):
        # Line 483: yaml_config=None -> falls back to module-level config.
        # No coiled-spring / pre-breakout multipliers active, so size unchanged.
        result = apply_position_size_multipliers(
            position_pct=12.0, pre_breakout_bonus=0, has_base=False,
            is_breaking_out=False, breakout_volume_ratio=0,
            is_coiled_spring=False, in_soft_zone=False, soft_zone_mult=1.0,
            is_correction_zone=False, cz_position_mult=1.0,
            heat_penalty_active=False, market_state_multiplier=1.0,
            profile={}, yaml_config=None,
        )
        assert abs(result - 12.0) < 0.01


class TestApplyPositionSizeMultipliersCoverage:
    def _base(self, **overrides):
        defaults = dict(
            position_pct=12.0, pre_breakout_bonus=0, has_base=False,
            is_breaking_out=False, breakout_volume_ratio=0,
            is_coiled_spring=False, in_soft_zone=False, soft_zone_mult=1.0,
            is_correction_zone=False, cz_position_mult=1.0,
            heat_penalty_active=False, market_state_multiplier=1.0,
            profile={}, yaml_config=MagicMock(get=lambda *a, **k: {}),
        )
        defaults.update(overrides)
        return defaults

    def test_market_state_multiplier_applied(self):
        # Line 491: any market_state_multiplier != 1.0 scales position size.
        result = apply_position_size_multipliers(**self._base(
            market_state_multiplier=0.50,
        ))
        assert abs(result - 6.0) < 0.01

    def test_pre_breakout_moderate_uses_reduced_multiplier(self):
        # Line 498: pre_breakout_bonus in [25, 35) with has_base -> mult * 0.93
        # Default pre_breakout_multiplier=1.40, so 12 * (1.40 * 0.93) = 15.624
        result = apply_position_size_multipliers(**self._base(
            pre_breakout_bonus=30, has_base=True,
        ))
        assert abs(result - 12.0 * 1.40 * 0.93) < 0.01
