"""
Tests for Market State Machine

Validates:
1. Bull market stays TRENDING (no regression on #148 performance)
2. Correction → FTD → Recovery → Confirmed → Trending path
3. Fast-track recovery for brief dips
4. Distribution day tracking and expiry
5. Exposure limits per state
6. Recovery seeding thresholds
"""

import pytest
from datetime import date, timedelta
from backend.market_state import MarketStateManager, MarketState, DEFAULT_CONFIG


class TestMarketStateInitialization:
    """Test initial state setup."""

    def test_default_state_is_trending(self):
        """State machine starts in TRENDING (100% exposure)."""
        mgr = MarketStateManager()
        assert mgr.state == MarketState.TRENDING
        assert mgr.max_exposure_pct == 1.0
        assert mgr.can_buy is True

    def test_initialize_bullish_market(self):
        """SPY above 50MA → TRENDING."""
        mgr = MarketStateManager()
        mgr.initialize_state(spy_close=500, spy_ma50=480, spy_ema21=495)
        assert mgr.state == MarketState.TRENDING

    def test_initialize_mild_correction(self):
        """SPY slightly below 50MA → PRESSURE."""
        mgr = MarketStateManager()
        mgr.initialize_state(spy_close=475, spy_ma50=480, spy_ema21=478)
        assert mgr.state == MarketState.PRESSURE

    def test_initialize_deep_correction(self):
        """SPY well below 50MA (>3%) → CORRECTION."""
        mgr = MarketStateManager()
        mgr.initialize_state(spy_close=460, spy_ma50=480, spy_ema21=470)
        assert mgr.state == MarketState.CORRECTION


class TestBullMarketPreservation:
    """CRITICAL: Ensure bull market behavior is identical to the old system."""

    def test_stays_trending_when_spy_above_50ma(self):
        """In a sustained bull market, state never leaves TRENDING."""
        mgr = MarketStateManager()
        base_date = date(2025, 3, 1)

        # Simulate 60 days of bull market (SPY above all MAs)
        for day_offset in range(60):
            d = base_date + timedelta(days=day_offset)
            mgr.update(
                current_date=d,
                spy_close=500 + day_offset * 0.5,  # Slowly rising
                spy_prev_close=500 + (day_offset - 1) * 0.5,
                spy_volume=1_000_000,
                spy_prev_volume=950_000,
                spy_ma50=490,
                spy_ema21=498,
            )

        assert mgr.state == MarketState.TRENDING
        assert mgr.max_exposure_pct == 1.0
        assert mgr.can_buy is True
        assert len(mgr.state_history) == 0  # No state changes at all

    def test_trending_100_percent_exposure(self):
        """TRENDING state allows 100% portfolio investment."""
        mgr = MarketStateManager()
        assert mgr.max_exposure_pct == 1.0

    def test_trending_full_position_size(self):
        """TRENDING state has 1.0x position size multiplier (no reduction)."""
        mgr = MarketStateManager()
        assert mgr.position_size_multiplier == 1.0

    def test_no_distribution_accumulation_in_uptrend(self):
        """Distribution days don't accumulate when market is rising."""
        mgr = MarketStateManager()
        base_date = date(2025, 3, 1)

        # 20 days of rising market
        for i in range(20):
            d = base_date + timedelta(days=i)
            mgr.update(
                current_date=d,
                spy_close=500 + i,  # Rising
                spy_prev_close=499 + i,
                spy_volume=1_000_000,
                spy_prev_volume=1_100_000,  # Lower volume (not distribution)
                spy_ma50=490,
                spy_ema21=498,
            )

        assert mgr.state == MarketState.TRENDING
        assert len(mgr.distribution_days) == 0


class TestCorrectionAndRecovery:
    """Test the full correction → FTD → recovery → trending cycle."""

    def _push_to_correction(self, mgr, start_date=None):
        """Helper: push state machine into CORRECTION."""
        d = start_date or date(2025, 3, 1)
        # Simulate sharp drop below 50MA (>3%)
        mgr.update(
            current_date=d,
            spy_close=460,  # 4.2% below 50MA of 480
            spy_prev_close=475,
            spy_volume=2_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=470,
        )
        assert mgr.state == MarketState.CORRECTION

    def test_trending_to_correction_on_deep_drop(self):
        """TRENDING → CORRECTION when SPY drops 3%+ below 50MA."""
        mgr = MarketStateManager()
        self._push_to_correction(mgr)
        assert mgr.can_buy is False
        assert mgr.max_exposure_pct == 0.0

    def test_correction_blocks_buying(self):
        """CORRECTION state blocks all new buys."""
        mgr = MarketStateManager()
        self._push_to_correction(mgr)
        assert mgr.can_buy is False
        assert mgr.can_seed is False

    def test_ftd_triggers_recovery(self):
        """Follow-Through Day moves from CORRECTION to RECOVERY."""
        mgr = MarketStateManager()
        self._push_to_correction(mgr)

        base_date = date(2025, 3, 2)

        # Simulate rally attempt: 3 small up days
        for i in range(3):
            d = base_date + timedelta(days=i)
            mgr.update(
                current_date=d,
                spy_close=461 + i,
                spy_prev_close=460 + i,
                spy_volume=1_200_000,
                spy_prev_volume=1_100_000,
                spy_ma50=480,
                spy_ema21=470,
            )
        assert mgr.state == MarketState.CORRECTION  # Too early for FTD

        # Day 4+: Big gain on higher volume = FTD
        ftd_date = base_date + timedelta(days=3)
        mgr.update(
            current_date=ftd_date,
            spy_close=472,       # +2.1% gain from 462
            spy_prev_close=462,
            spy_volume=1_500_000,  # Higher volume
            spy_prev_volume=1_200_000,
            spy_ma50=480,
            spy_ema21=470,
        )
        assert mgr.state == MarketState.RECOVERY
        assert mgr.last_ftd_date == ftd_date
        assert mgr.can_buy is True
        assert mgr.max_exposure_pct == 0.30

    def test_recovery_to_confirmed(self):
        """RECOVERY → CONFIRMED after 3 days above 21 EMA."""
        mgr = MarketStateManager()
        mgr.state = MarketState.RECOVERY
        mgr.rally_attempt_low = 450
        mgr.recovery_confirmations = 0

        base_date = date(2025, 3, 10)
        for i in range(3):
            d = base_date + timedelta(days=i)
            mgr.update(
                current_date=d,
                spy_close=475,  # Above 21 EMA
                spy_prev_close=474,
                spy_volume=1_000_000,
                spy_prev_volume=950_000,
                spy_ma50=480,
                spy_ema21=472,
            )

        assert mgr.state == MarketState.CONFIRMED
        assert mgr.max_exposure_pct == 0.55

    def test_confirmed_to_trending(self):
        """CONFIRMED → TRENDING when above 50MA with sustained 21 EMA confirmation."""
        mgr = MarketStateManager()
        mgr.state = MarketState.CONFIRMED

        base_date = date(2025, 3, 20)
        for i in range(6):
            d = base_date + timedelta(days=i)
            mgr.update(
                current_date=d,
                spy_close=485,  # Above 50MA and 21 EMA
                spy_prev_close=484,
                spy_volume=1_000_000,
                spy_prev_volume=950_000,
                spy_ma50=480,
                spy_ema21=482,
            )

        assert mgr.state == MarketState.TRENDING
        assert mgr.max_exposure_pct == 1.0

    def test_recovery_fails_on_low_undercut(self):
        """RECOVERY → CORRECTION if rally low is undercut (FTD failure)."""
        mgr = MarketStateManager()
        mgr.state = MarketState.RECOVERY
        mgr.rally_attempt_low = 460
        mgr.recovery_confirmations = 1

        mgr.update(
            current_date=date(2025, 3, 15),
            spy_close=455,  # Below rally low of 460
            spy_prev_close=462,
            spy_volume=1_500_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=470,
        )

        assert mgr.state == MarketState.CORRECTION


class TestFastTrackRecovery:
    """Test fast-track from CORRECTION back to TRENDING for brief dips."""

    def test_fast_track_after_min_correction_days(self):
        """Fast-track to TRENDING only after min_correction_days (default 3)."""
        mgr = MarketStateManager()

        # Push to correction
        mgr.update(
            current_date=date(2025, 3, 1),
            spy_close=460,
            spy_prev_close=485,
            spy_volume=2_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.CORRECTION

        # Day 2: SPY back above MAs but too soon (< 3 days)
        mgr.update(
            current_date=date(2025, 3, 2),
            spy_close=485,
            spy_prev_close=460,
            spy_volume=1_500_000,
            spy_prev_volume=2_000_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.CORRECTION  # Still too early

        # Day 3: Still in CORRECTION (need 3 full days)
        mgr.update(
            current_date=date(2025, 3, 3),
            spy_close=486,
            spy_prev_close=485,
            spy_volume=1_200_000,
            spy_prev_volume=1_500_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.CORRECTION  # Still need 1 more day

        # Day 4: Now fast-track fires (3 days completed)
        mgr.update(
            current_date=date(2025, 3, 4),
            spy_close=487,
            spy_prev_close=486,
            spy_volume=1_100_000,
            spy_prev_volume=1_200_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.TRENDING
        assert mgr.last_transition_was_fast_track is True

    def test_fast_track_with_zero_min_days(self):
        """With min_correction_days=0, fast-track is immediate (backward compat)."""
        config = {"transitions": {"min_correction_days": 0}}
        mgr = MarketStateManager(config)

        # Push to correction
        mgr.update(
            current_date=date(2025, 3, 1),
            spy_close=460,
            spy_prev_close=485,
            spy_volume=2_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.CORRECTION

        # Next day: immediate fast-track
        mgr.update(
            current_date=date(2025, 3, 2),
            spy_close=485,
            spy_prev_close=460,
            spy_volume=1_500_000,
            spy_prev_volume=2_000_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.TRENDING


class TestConfirmedStickiness:
    """Test that CONFIRMED state doesn't immediately revert to CORRECTION."""

    def test_confirmed_stays_sticky_for_min_days(self):
        """CONFIRMED doesn't go back to CORRECTION within min_confirmed_days."""
        mgr = MarketStateManager()
        mgr.state = MarketState.CONFIRMED
        mgr.state_days_count = 0  # Just entered CONFIRMED

        # Day 1 in CONFIRMED: SPY drops well below 50MA (would normally trigger CORRECTION)
        mgr.update(
            current_date=date(2025, 4, 9),
            spy_close=460,  # 4.2% below 50MA
            spy_prev_close=470,
            spy_volume=1_500_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=470,
        )
        assert mgr.state == MarketState.CONFIRMED  # Still sticky

    def test_confirmed_to_correction_after_min_days(self):
        """CONFIRMED can go to CORRECTION after min_confirmed_days."""
        mgr = MarketStateManager()
        mgr.state = MarketState.CONFIRMED
        mgr.state_days_count = 3  # Been in CONFIRMED for 3 days

        # Now the deep drop triggers CORRECTION
        mgr.update(
            current_date=date(2025, 4, 12),
            spy_close=460,
            spy_prev_close=470,
            spy_volume=1_500_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=470,
        )
        assert mgr.state == MarketState.CORRECTION


class TestStateDaysCounting:
    """Test the state_days_count tracking."""

    def test_days_count_increments(self):
        """state_days_count increments each day in same state."""
        mgr = MarketStateManager()
        assert mgr.state_days_count == 0

        for i in range(5):
            mgr.update(
                current_date=date(2025, 3, 1) + timedelta(days=i),
                spy_close=500 + i,
                spy_prev_close=499 + i,
                spy_volume=1_000_000,
                spy_prev_volume=950_000,
                spy_ma50=490,
                spy_ema21=498,
            )

        assert mgr.state_days_count == 5

    def test_days_count_resets_on_state_change(self):
        """state_days_count resets to 0 when state changes."""
        mgr = MarketStateManager()
        # Accumulate some days in TRENDING
        for i in range(3):
            mgr.update(
                current_date=date(2025, 3, 1) + timedelta(days=i),
                spy_close=500,
                spy_prev_close=499,
                spy_volume=1_000_000,
                spy_prev_volume=950_000,
                spy_ma50=490,
                spy_ema21=498,
            )
        assert mgr.state_days_count == 3

        # Now push to CORRECTION
        mgr.update(
            current_date=date(2025, 3, 4),
            spy_close=460,
            spy_prev_close=485,
            spy_volume=2_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=478,
        )
        assert mgr.state == MarketState.CORRECTION
        assert mgr.state_days_count == 0


class TestDistributionDays:
    """Test distribution day tracking and expiry."""

    def test_distribution_day_counted(self):
        """Down day on higher volume counts as distribution."""
        mgr = MarketStateManager()
        mgr.update(
            current_date=date(2025, 3, 1),
            spy_close=498,  # -0.4% decline
            spy_prev_close=500,
            spy_volume=1_200_000,  # Higher than previous
            spy_prev_volume=1_000_000,
            spy_ma50=490,
            spy_ema21=498,
        )
        assert len(mgr.distribution_days) == 1

    def test_down_day_low_volume_not_distribution(self):
        """Down day on lower volume is NOT distribution."""
        mgr = MarketStateManager()
        mgr.update(
            current_date=date(2025, 3, 1),
            spy_close=498,
            spy_prev_close=500,
            spy_volume=800_000,  # Lower volume
            spy_prev_volume=1_000_000,
            spy_ma50=490,
            spy_ema21=498,
        )
        assert len(mgr.distribution_days) == 0

    def test_distribution_days_expire(self):
        """Distribution days expire after ~25 trading days (35 calendar days)."""
        mgr = MarketStateManager()

        # Add distribution day
        mgr.update(
            current_date=date(2025, 3, 1),
            spy_close=498,
            spy_prev_close=500,
            spy_volume=1_200_000,
            spy_prev_volume=1_000_000,
            spy_ma50=490,
            spy_ema21=498,
        )
        assert len(mgr.distribution_days) == 1

        # 36 days later: should be expired
        mgr.update(
            current_date=date(2025, 4, 6),
            spy_close=510,
            spy_prev_close=509,
            spy_volume=1_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=500,
            spy_ema21=508,
        )
        assert len(mgr.distribution_days) == 0

    def test_rally_off_removes_distribution(self):
        """6%+ rally from distribution day close removes it."""
        mgr = MarketStateManager()

        # Add distribution day at SPY 470
        mgr.distribution_days = [{"date": date(2025, 3, 1), "close": 470, "decline_pct": -0.5}]

        # SPY rallies to 500 (6.4% above 470) — distribution day should be removed
        mgr.update(
            current_date=date(2025, 3, 10),
            spy_close=500,
            spy_prev_close=499,
            spy_volume=1_000_000,
            spy_prev_volume=1_000_000,
            spy_ma50=490,
            spy_ema21=498,
        )
        assert len(mgr.distribution_days) == 0


class TestExposureLimits:
    """Test exposure limits for each state."""

    def test_exposure_by_state(self):
        """Each state has correct exposure limit."""
        mgr = MarketStateManager()

        expected = {
            MarketState.TRENDING: 1.0,
            MarketState.PRESSURE: 0.6,
            MarketState.CORRECTION: 0.0,
            MarketState.RECOVERY: 0.3,
            MarketState.CONFIRMED: 0.55,
        }

        for state, expected_exposure in expected.items():
            mgr.state = state
            assert mgr.max_exposure_pct == expected_exposure, \
                f"State {state}: expected {expected_exposure}, got {mgr.max_exposure_pct}"

    def test_can_buy_by_state(self):
        """Only CORRECTION blocks buying."""
        mgr = MarketStateManager()

        for state in MarketState:
            mgr.state = state
            if state == MarketState.CORRECTION:
                assert mgr.can_buy is False
            else:
                assert mgr.can_buy is True

    def test_can_seed_by_state(self):
        """Can seed in RECOVERY, CONFIRMED, and TRENDING only."""
        mgr = MarketStateManager()

        seedable = {MarketState.RECOVERY, MarketState.CONFIRMED, MarketState.TRENDING}
        for state in MarketState:
            mgr.state = state
            if state in seedable:
                assert mgr.can_seed is True, f"{state} should allow seeding"
            else:
                assert mgr.can_seed is False, f"{state} should NOT allow seeding"

    def test_position_size_multiplier_by_state(self):
        """Position size multiplier matches state risk level."""
        mgr = MarketStateManager()

        mgr.state = MarketState.TRENDING
        assert mgr.position_size_multiplier == 1.0

        mgr.state = MarketState.PRESSURE
        assert mgr.position_size_multiplier == 0.75

        mgr.state = MarketState.RECOVERY
        assert mgr.position_size_multiplier == 0.60

        mgr.state = MarketState.CONFIRMED
        assert mgr.position_size_multiplier == 0.85


class TestPressureState:
    """Test PRESSURE state transitions."""

    def test_trending_to_pressure_on_distribution(self):
        """5+ distribution days moves TRENDING → PRESSURE."""
        mgr = MarketStateManager()
        base_date = date(2025, 3, 1)

        # Create 5 distribution days
        for i in range(5):
            d = base_date + timedelta(days=i * 2)  # Every other day
            mgr.update(
                current_date=d,
                spy_close=498 - i,  # Small declines
                spy_prev_close=500 - i,
                spy_volume=1_200_000 + i * 100_000,  # Higher volume each time
                spy_prev_volume=1_000_000 + i * 100_000,
                spy_ma50=490,
                spy_ema21=498,
            )

        assert mgr.state == MarketState.PRESSURE
        assert mgr.max_exposure_pct == 0.60

    def test_pressure_back_to_trending(self):
        """PRESSURE → TRENDING when distribution clears and above 21 EMA."""
        mgr = MarketStateManager()
        mgr.state = MarketState.PRESSURE
        mgr.distribution_days = []  # Cleared
        mgr.consecutive_above_21ema = 0

        base_date = date(2025, 3, 15)
        for i in range(4):
            d = base_date + timedelta(days=i)
            mgr.update(
                current_date=d,
                spy_close=495,
                spy_prev_close=494,
                spy_volume=1_000_000,
                spy_prev_volume=1_100_000,
                spy_ma50=490,
                spy_ema21=492,
            )

        assert mgr.state == MarketState.TRENDING


class TestConfigCustomization:
    """Test custom configuration overrides."""

    def test_custom_exposure_limits(self):
        """Custom exposure limits are respected."""
        custom_config = {
            "exposure_limits": {
                "trending": 0.90,
                "recovery": 0.40,
                "correction": 0.0,
                "pressure": 0.50,
                "confirmed": 0.60,
            }
        }
        mgr = MarketStateManager(custom_config)
        assert mgr.max_exposure_pct == 0.90  # Custom trending

        mgr.state = MarketState.RECOVERY
        assert mgr.max_exposure_pct == 0.40  # Custom recovery

    def test_custom_ftd_threshold(self):
        """Stricter FTD threshold requires larger gain."""
        custom_config = {
            "ftd": {"min_gain_pct": 2.5, "earliest_day": 4, "latest_day": 25}
        }
        mgr = MarketStateManager(custom_config)
        mgr.state = MarketState.CORRECTION
        mgr.rally_attempt_low = 450
        mgr.rally_attempt_day = 4

        # 2.0% gain — below custom 2.5% threshold
        mgr.update(
            current_date=date(2025, 3, 10),
            spy_close=469.2,  # +2.0%
            spy_prev_close=460,
            spy_volume=1_500_000,
            spy_prev_volume=1_000_000,
            spy_ma50=480,
            spy_ema21=465,
        )
        assert mgr.state == MarketState.CORRECTION  # Not enough for FTD


class TestStateSummary:
    """Test state summary reporting."""

    def test_get_state_summary(self):
        """Summary includes all key fields."""
        mgr = MarketStateManager()
        summary = mgr.get_state_summary()

        assert "state" in summary
        assert "max_exposure_pct" in summary
        assert "can_buy" in summary
        assert "can_seed" in summary
        assert "distribution_days" in summary
        assert summary["state"] == "trending"
        assert summary["max_exposure_pct"] == 1.0


class TestCoverageGaps:
    """Targeted coverage for branches the scenario tests don't reach:
    v-bottom early-returns, distribution-day guard against zero prev close,
    explicit state-transition checks (TRENDING→PRESSURE on 21EMA streak,
    PRESSURE→CORRECTION, rally-low undercut restart, RECOVERY fast-track,
    CONFIRMED→PRESSURE), and the empty-history short-circuit on
    last_transition_was_fast_track.
    """

    def test_v_bottom_rejected_when_rally_too_small(self):
        # Drop is large enough (20%) but rally from low is only 5% (< 8% min).
        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.spy_52w_high = 500
        mgr.correction_low = 400
        mgr.correction_low_date = date(2024, 1, 1)
        assert mgr._check_v_bottom(date(2024, 1, 10), spy_close=420) is False

    def test_v_bottom_rejected_when_rally_too_slow(self):
        # Rally is 10% (passes 8% threshold) but happens 20 days after the low.
        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.spy_52w_high = 500
        mgr.correction_low = 400
        mgr.correction_low_date = date(2024, 1, 1)
        assert mgr._check_v_bottom(date(2024, 1, 21), spy_close=440) is False

    def test_distribution_day_skipped_when_prev_close_zero(self):
        # Guards against bad SPY data (first-bar fixture with prev=0).
        mgr = MarketStateManager()
        mgr._update_distribution_days(
            current_date=date(2024, 1, 5),
            spy_close=480,
            spy_prev_close=0,
            spy_volume=1_000_000,
            spy_prev_volume=900_000,
        )
        assert mgr.distribution_days == []


class TestVBottomConfig:
    """V-bottom detection thresholds are config-driven (market_state.v_bottom).

    Historically the thresholds were hardcoded with a comment claiming profile
    config would override them — it never did. Now they read from
    self.config['v_bottom'] with DEFAULT_CONFIG fallbacks; these tests pin
    both the default behavior and the override path.
    """

    def _correction_mgr(self, config=None):
        """Manager in CORRECTION with a 20% drop from high (500 → 400)."""
        mgr = MarketStateManager(config)
        mgr.state = MarketState.CORRECTION
        mgr.spy_52w_high = 500
        mgr.correction_low = 400
        mgr.correction_low_date = date(2024, 1, 1)
        return mgr

    def test_defaults_unchanged_fires_on_classic_v_bottom(self):
        # 20% drop (>= 15 default), 10% rally (>= 8 default), 9 days (<= 15).
        mgr = self._correction_mgr()
        assert mgr._check_v_bottom(date(2024, 1, 10), spy_close=440) is True

    def test_min_drop_pct_override_blocks_shallow_crash(self):
        # Same 20% drop no longer qualifies when min_drop_pct raised to 25.
        mgr = self._correction_mgr({"v_bottom": {"min_drop_pct": 25.0}})
        assert mgr._check_v_bottom(date(2024, 1, 10), spy_close=440) is False

    def test_min_rally_pct_override_accepts_smaller_rally(self):
        # 6% rally fails the 8% default but passes with min_rally_pct=5.
        mgr = self._correction_mgr({"v_bottom": {"min_rally_pct": 5.0}})
        assert mgr._check_v_bottom(date(2024, 1, 10), spy_close=424) is True

    def test_max_days_override_accepts_slower_rally(self):
        # 20 days since low fails the 15-day default but passes with 25.
        mgr = self._correction_mgr({"v_bottom": {"max_days_since_low": 25}})
        assert mgr._check_v_bottom(date(2024, 1, 21), spy_close=440) is True

    def test_partial_override_keeps_other_defaults(self):
        # Overriding one key must not disable the other thresholds: with only
        # min_drop_pct set, the 8% default rally floor still rejects a 5% rally.
        mgr = self._correction_mgr({"v_bottom": {"min_drop_pct": 10.0}})
        assert mgr._check_v_bottom(date(2024, 1, 10), spy_close=420) is False


class TestInitVsMidRunPressureBoundary:
    """Pin the INTENTIONAL asymmetry documented on initialize_state:
    just-below-50MA → PRESSURE at init (no history to confirm with), but an
    established TRENDING state tolerates the same price until distribution
    days or a persistent 21EMA breakdown confirm weakness. If either of these
    tests starts failing, someone 'aligned' the two paths — read the
    initialize_state docstring before accepting that change.
    """

    def test_init_just_below_50ma_is_pressure(self):
        mgr = MarketStateManager()
        # 1% below 50MA — inside the 3% deep-drop band.
        mgr.initialize_state(spy_close=475.2, spy_ma50=480, spy_ema21=478)
        assert mgr.state == MarketState.PRESSURE

    def test_midrun_trending_tolerates_same_price_without_confirmation(self):
        mgr = MarketStateManager()
        mgr.initialize_state(spy_close=500, spy_ma50=480, spy_ema21=495)
        assert mgr.state == MarketState.TRENDING

        # One day at the identical just-below-50MA price: above 21EMA, flat
        # volume (no distribution day), no prior weakness streak.
        mgr.update(
            current_date=date(2025, 3, 3),
            spy_close=475.2,
            spy_prev_close=500,
            spy_volume=900_000,
            spy_prev_volume=1_000_000,  # volume down → not a distribution day
            spy_ma50=480,
            spy_ema21=474,  # still above 21EMA → no weakness streak
        )
        assert mgr.state == MarketState.TRENDING

    def test_trending_to_pressure_on_21ema_streak_below_50ma(self):
        # 3 days below 21EMA + close below 50MA (but < 3% — too shallow for
        # direct CORRECTION) trips the secondary PRESSURE transition.
        mgr = MarketStateManager()
        mgr.consecutive_below_21ema = 3
        mgr.consecutive_above_21ema = 0
        mgr._check_trending_exit(
            current_date=date(2024, 6, 1),
            spy_close=485,
            spy_ma50=490,
            spy_ema21=487,
            dist_count=0,
        )
        assert mgr.state == MarketState.PRESSURE

    def test_pressure_to_correction_on_close_below_50ma(self):
        # PRESSURE breaks the 50MA -> CORRECTION + rally tracking initialized.
        mgr = MarketStateManager()
        mgr.state = MarketState.PRESSURE
        mgr._check_pressure_transitions(
            current_date=date(2024, 6, 1),
            spy_close=480,
            spy_ma50=490,
            spy_ema21=485,
            dist_count=0,
        )
        assert mgr.state == MarketState.CORRECTION
        assert mgr.rally_attempt_day == 0
        assert mgr.rally_attempt_low == 480
        assert mgr.rally_attempt_start_date == date(2024, 6, 1)

    def test_correction_undercut_restarts_rally_tracking(self):
        # In CORRECTION with a rally-low set, a down day below that low resets
        # rally tracking to the new low (elif branch at line 447-449).
        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.rally_attempt_day = 2
        mgr.rally_attempt_low = 470
        mgr.rally_attempt_start_date = date(2024, 6, 1)
        mgr._check_correction_exit(
            current_date=date(2024, 6, 5),
            spy_close=460,
            spy_prev_close=475,
            spy_volume=1_000_000,
            spy_prev_volume=900_000,
            spy_ma50=490,
            spy_ema21=485,
        )
        assert mgr.rally_attempt_day == 0
        assert mgr.rally_attempt_low == 460
        assert mgr.rally_attempt_start_date == date(2024, 6, 5)

    def test_recovery_fast_tracks_to_trending(self):
        # Above both 50MA and 21EMA with 3-day 21EMA streak -> direct TRENDING
        # without going through CONFIRMED.
        mgr = MarketStateManager()
        mgr.state = MarketState.RECOVERY
        mgr.consecutive_above_21ema = 3
        mgr.rally_attempt_low = None
        mgr._check_recovery_transitions(
            current_date=date(2024, 6, 1),
            spy_close=500,
            spy_ma50=490,
            spy_ema21=495,
            dist_count=0,
        )
        assert mgr.state == MarketState.TRENDING

    def test_confirmed_to_pressure_on_distribution_buildup(self):
        # CONFIRMED with 5 distribution days falls back to PRESSURE (close is
        # above 50MA and 21EMA streak too short for TRENDING).
        mgr = MarketStateManager()
        mgr.state = MarketState.CONFIRMED
        mgr.state_days_count = 10
        mgr.consecutive_above_21ema = 0
        mgr._check_confirmed_transitions(
            current_date=date(2024, 6, 1),
            spy_close=495,
            spy_ma50=490,
            spy_ema21=498,
            dist_count=5,
        )
        assert mgr.state == MarketState.PRESSURE

    def test_last_transition_was_fast_track_empty_history(self):
        # Fresh manager has no state history; property must short-circuit
        # rather than IndexError on state_history[-1].
        mgr = MarketStateManager()
        assert mgr.state_history == []
        assert mgr.last_transition_was_fast_track is False
