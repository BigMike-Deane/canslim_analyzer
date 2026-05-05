"""
Tests for sync helpers in backend/trading_utils.py.

These pure functions replace previously inline-duplicated logic blocks
in ai_trader.py and backtester.py. The tests intentionally exercise both
"call site perspectives" so a future refactor of either side cannot
silently desync without a regression here.

Past sync drift between ai_trader and backtester cost 25-100pp on the
4-yr backtest each time — these helpers + tests close that hole
structurally.
"""
import pytest

from backend.trading_utils import (
    apply_sector_allocation_cap,
    should_take_partial_on_trailing_stop,
    SECTOR_MIN_ROOM_FRACTION,
    MAX_SECTOR_ALLOCATION,
)


# ── Partial Trailing Stop Gate ────────────────────────────────────────────────
# Used by:
#   - ai_trader.evaluate_sells (passes score=score if score_available else 0)
#   - backtester._evaluate_sells (passes score=current_score, NaN-safe)


class TestShouldTakePartialOnTrailingStop:
    """All defaults match YAML (partial_on_trailing=True, min_pyramids=2,
    min_score=65)."""

    def test_default_path_high_conviction_takes_partial(self):
        # Backtester perspective: pyramided 2x with strong score
        assert should_take_partial_on_trailing_stop(
            pyramid_count=2, score=70
        ) is True

    def test_zero_pyramids_takes_full(self):
        # Live perspective: no pyramids ⇒ no partial gating
        assert should_take_partial_on_trailing_stop(
            pyramid_count=0, score=80
        ) is False

    def test_one_pyramid_below_threshold_takes_full(self):
        # Boundary: pyramid_count = min - 1 ⇒ full sell
        assert should_take_partial_on_trailing_stop(
            pyramid_count=1, score=80
        ) is False

    def test_score_below_min_takes_full(self):
        # Strong pyramiding but score collapsed ⇒ full sell
        assert should_take_partial_on_trailing_stop(
            pyramid_count=3, score=60
        ) is False

    def test_score_at_threshold_takes_partial(self):
        # Boundary: score == min_score is allowed (>=, not >)
        assert should_take_partial_on_trailing_stop(
            pyramid_count=2, score=65
        ) is True

    def test_master_kill_switch_disables_partial(self):
        # YAML override: partial_on_trailing=false ⇒ always full
        assert should_take_partial_on_trailing_stop(
            pyramid_count=5, score=95, partial_on_trailing=False
        ) is False

    def test_score_zero_means_unavailable_takes_full(self):
        # Reproduces ai_trader's `score_available and score >= min` semantic.
        # Live passes score=0 when score_available is False.
        assert should_take_partial_on_trailing_stop(
            pyramid_count=4, score=0
        ) is False

    def test_custom_thresholds_strict_config(self):
        # Strategy profile could tighten thresholds — helper must honor them.
        assert should_take_partial_on_trailing_stop(
            pyramid_count=2, score=70,
            partial_min_pyramid_count=3, partial_min_score=75
        ) is False
        assert should_take_partial_on_trailing_stop(
            pyramid_count=3, score=80,
            partial_min_pyramid_count=3, partial_min_score=75
        ) is True


# ── Sector Allocation Cap ─────────────────────────────────────────────────────
# Used by:
#   - ai_trader.check_sector_limit (live, computes current_sector_value
#     from cached current_value)
#   - backtester._evaluate_buys (recomputes sector_value from live
#     prices on the simulation date)
# Both pass identical scalar args — the helper is the single source of
# truth for the math.


class TestApplySectorAllocationCap:
    """Tests use explicit max_sector_allocation=0.30 so they are
    immune to environment-specific config drift."""

    CAP = 0.30

    def test_well_under_cap_returns_unchanged(self):
        # 10% sector + 5% buy = 15% < 30% cap ⇒ pass through
        out = apply_sector_allocation_cap(
            requested_position_value=5_000,
            current_sector_value=10_000,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == 5_000

    def test_exactly_at_cap_returns_unchanged(self):
        # 25% sector + 5% buy = 30% = cap ⇒ no scale-down (boundary is <=)
        out = apply_sector_allocation_cap(
            requested_position_value=5_000,
            current_sector_value=25_000,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == 5_000

    def test_over_cap_with_room_squeezes_to_remaining(self):
        # 25% sector + 8% buy = 33% > 30% cap; room = 5% > 2%
        # ⇒ scale to 5% * 100k = 5_000
        out = apply_sector_allocation_cap(
            requested_position_value=8_000,
            current_sector_value=25_000,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == pytest.approx(5_000)

    def test_at_cap_minus_room_floor_rejects(self):
        # Sector at 28.5% (only 1.5% room) ⇒ rejection (return 0.0)
        out = apply_sector_allocation_cap(
            requested_position_value=10_000,
            current_sector_value=28_500,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == 0.0

    def test_room_exactly_2pct_rejects(self):
        # Boundary: remaining_room == min_room_fraction ⇒ reject (<= check)
        # Sector at 28% means remaining_room = 30% - 28% = 2.0% exactly
        out = apply_sector_allocation_cap(
            requested_position_value=10_000,
            current_sector_value=28_000,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == 0.0

    def test_room_just_over_2pct_squeezes(self):
        # Room = 2.001% (just over the 2% floor) ⇒ scale, do NOT reject
        out = apply_sector_allocation_cap(
            requested_position_value=10_000,
            current_sector_value=27_999,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        # Adjusted = (30000 - 27999) = 2001
        assert out == pytest.approx(2_001)

    def test_zero_portfolio_value_passes_through(self):
        # Bootstrap edge case: empty portfolio, no math possible
        out = apply_sector_allocation_cap(
            requested_position_value=5_000,
            current_sector_value=0,
            portfolio_value=0,
            max_sector_allocation=self.CAP,
        )
        assert out == 5_000

    def test_negative_portfolio_value_passes_through(self):
        # Pathological state — never reached in production but safe
        out = apply_sector_allocation_cap(
            requested_position_value=5_000,
            current_sector_value=0,
            portfolio_value=-100,
            max_sector_allocation=self.CAP,
        )
        assert out == 5_000

    def test_already_full_sector_rejects(self):
        # Sector at exactly the cap ⇒ remaining_room = 0 ⇒ reject
        out = apply_sector_allocation_cap(
            requested_position_value=1_000,
            current_sector_value=30_000,
            portfolio_value=100_000,
            max_sector_allocation=self.CAP,
        )
        assert out == 0.0

    def test_custom_cap_honored(self):
        # Strategy profile could lower cap to 20% — helper must honor.
        # 15% sector + 8% buy = 23% > 20% cap; room = 5% > 2% ⇒ scale to 5%
        out = apply_sector_allocation_cap(
            requested_position_value=8_000,
            current_sector_value=15_000,
            portfolio_value=100_000,
            max_sector_allocation=0.20,
        )
        assert out == pytest.approx(5_000)


# ── Constants exported for both call sites ────────────────────────────────────


class TestSyncHelperConstants:

    def test_min_room_fraction_matches_call_site_literal(self):
        # The 0.02 literal historically appeared inline in BOTH files. Now
        # both call sites use this constant. If you change it, both flip
        # together — that's the whole point.
        assert SECTOR_MIN_ROOM_FRACTION == 0.02

    def test_max_sector_allocation_loaded_from_config(self):
        # Sanity — should be a fraction in (0, 1).
        assert 0 < MAX_SECTOR_ALLOCATION < 1
