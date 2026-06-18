"""
AI Trader <> Backtester sync tests.

These tests pin behavior on the shared trading helpers AND on the thin
ai_trader.py wrappers around them. The hard project rule is "every change
to ai_trader.py trading logic MUST be mirrored in backtester.py" — these
tests fail when the shared helpers change semantics or when ai_trader's
wrappers diverge from documented contracts.

Coverage focus (highest sync-risk paths):
- get_partial_profit_action 25/40/50 tier delta math (documented past bug:
  ai_trader once used `sell_pct` instead of `sell_pct - partial_taken`)
- get_trailing_stop_pct band thresholds (champion config: 25/18/12/6/4)
- evaluate_score_crash decision matrix (drop_required, profitable skip,
  consecutive vs 2-of-3 gates)
- get_tightened_trailing_stop pre-earnings half-stop math
- _nan_safe NaN/None handling (Round 2 bug: NaN truthy through `or 0`)
- get_effective_score CANSLIM-vs-Growth path with NaN safety
- check_sector_limit DB-backed wrapper (count cap + 50% allocation cap +
  2% min-room rule, all mirrored in backtester._check_sector_limit)
- check_score_stability DB→engine adapter (DB returns desc, engine wants
  chronological — divergence here historically masks real crashes)
- categorize_sell_reason categorization (used by ai_trader signal_factors,
  matched against backtester reason strings)
- calculate_base_quality_bonus (used by both evaluate_buys flows)
"""
import math
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fresh_session():
    """In-memory SQLite with backend.database schema applied."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _make_position(**overrides):
    """Build an AIPortfolioPosition row with sane defaults."""
    from backend.database import AIPortfolioPosition

    defaults = dict(
        user_id=1,
        ticker="TEST",
        shares=100.0,
        cost_basis=100.0,
        purchase_date=datetime.now(timezone.utc) - timedelta(days=30),
        purchase_score=80.0,
        current_price=110.0,
        current_value=11000.0,
        gain_loss=1000.0,
        gain_loss_pct=10.0,
        current_score=75.0,
        is_growth_stock=False,
        peak_price=115.0,
        peak_date=datetime.now(timezone.utc) - timedelta(days=5),
        pyramid_count=0,
        partial_profit_taken=0.0,
    )
    defaults.update(overrides)
    return AIPortfolioPosition(**defaults)


# ── 1. Partial profit tier delta (documented past bug) ────────────────────────


class TestPartialProfitTierDelta:
    """The 25%/40% delta bug shipped to prod once — pin it down hard.

    Both ai_trader.evaluate_sells and backtester._evaluate_sells now route
    through trading_engine.get_partial_profit_action. If anyone reverts the
    `take_pct = sell_pct - partial_taken` math (or only patches one side),
    these tests fail.
    """

    def test_25_pct_tier_first_take_returns_25(self):
        from backend.trading_engine import get_partial_profit_action

        action = get_partial_profit_action(gain_pct=25.0, current_score=70, partial_taken=0)
        assert action is not None
        assert action["tier"] == 25
        assert action["take_pct"] == 25  # first take, full slug

    def test_40_pct_tier_with_25_already_taken_returns_25_delta(self):
        """The exact bug: when 25% already taken, 40% tier should sell 25 more (50-25)."""
        from backend.trading_engine import get_partial_profit_action

        action = get_partial_profit_action(gain_pct=40.0, current_score=70, partial_taken=25)
        assert action is not None
        assert action["tier"] == 40
        assert action["take_pct"] == 25  # 50 - 25, NOT 50

    def test_50_pct_tier_with_50_already_taken_returns_25_delta(self):
        """50%+ tier reaches sell_pct=75 cumulative; if 50 taken, sell 25 more."""
        from backend.trading_engine import get_partial_profit_action

        action = get_partial_profit_action(gain_pct=55.0, current_score=70, partial_taken=50)
        assert action is not None
        assert action["tier"] == 50
        assert action["take_pct"] == 25  # 75 - 50

    def test_no_action_when_already_at_tier_target(self):
        """If 25% already taken and only at 25% gain, no further action (delta would be 0)."""
        from backend.trading_engine import get_partial_profit_action

        action = get_partial_profit_action(gain_pct=25.0, current_score=70, partial_taken=25)
        assert action is None

    def test_min_score_gates_each_tier(self):
        """Below tier min_score (60 for 25/40, 55 for 50), no partial action."""
        from backend.trading_engine import get_partial_profit_action

        # Score 50 < 60 (25 tier min) and < 55 (50 tier min)
        action = get_partial_profit_action(gain_pct=30.0, current_score=50, partial_taken=0)
        assert action is None


# ── 2. Trailing stop bands (champion config: 25/18/12/6/4) ───────────────────


class TestTrailingStopBands:
    """Both ai_trader.evaluate_sells and backtester._evaluate_sells call
    get_trailing_stop_pct with peak_gain_pct → expect identical results.

    The champion `nostate_optimized` profile bands these as
    50+:25%, 30-50:18%, 20-30:12%, 10-20:6%, 5-10:4%. Defaults match
    except 10-20 (8 vs 6) — verify both default and override paths.
    """

    def test_default_bands_match_yaml_defaults(self):
        from backend.trading_engine import get_trailing_stop_pct

        # Empty profile → engine defaults (25/18/12/8/4)
        assert get_trailing_stop_pct(60, {}) == 25
        assert get_trailing_stop_pct(35, {}) == 18
        assert get_trailing_stop_pct(25, {}) == 12
        assert get_trailing_stop_pct(15, {}) == 8
        assert get_trailing_stop_pct(7, {}) == 4
        assert get_trailing_stop_pct(3, {}) is None  # Below floor

    def test_champion_profile_overrides_apply(self):
        """nostate_optimized overrides 10-20 band to 6 (was default 8)."""
        from backend.trading_engine import get_trailing_stop_pct
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        # Champion overrides 10-20 band to 6, not the default 8
        assert get_trailing_stop_pct(15, profile) == 6
        # Other bands match the champion config exactly
        assert get_trailing_stop_pct(60, profile) == 25
        assert get_trailing_stop_pct(35, profile) == 18
        assert get_trailing_stop_pct(25, profile) == 12

    def test_band_boundaries_inclusive_lower(self):
        """Boundary at exactly 50%, 30%, 20%, 10%, 5% — inclusive of lower bound."""
        from backend.trading_engine import get_trailing_stop_pct

        # Champion config to disambiguate from defaults
        from backend.trading_utils import get_strategy_profile
        profile = get_strategy_profile("nostate_optimized")

        assert get_trailing_stop_pct(50.0, profile) == 25  # Just enters 50+ band
        assert get_trailing_stop_pct(49.99, profile) == 18  # Stays in 30-50 band
        assert get_trailing_stop_pct(20.0, profile) == 12  # Just enters 20-30 band
        assert get_trailing_stop_pct(10.0, profile) == 6  # Just enters 10-20 band
        assert get_trailing_stop_pct(5.0, profile) == 4  # Just enters 5-10 band
        assert get_trailing_stop_pct(4.99, profile) is None


# ── 3. Pre-earnings tightening (mirrored sell rule in both files) ─────────────


class TestPreEarningsTightening:
    """ai_trader.evaluate_sells and backtester._evaluate_sells both call
    get_tightened_trailing_stop with tighten_factor=0.50 to halve the
    base trailing stop. Verify the math is symmetric and propagates None
    correctly when the base is None (peak_gain_pct < 5)."""

    def test_halves_50plus_band(self):
        from backend.trading_engine import get_tightened_trailing_stop
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        # 50+ band base = 25 → tightened = 12.5
        assert get_tightened_trailing_stop(60.0, profile, tighten_factor=0.5) == 12.5

    def test_halves_10_to_20_champion_band(self):
        """Champion 10-20 band is 6 (overridden); tightened = 3.0.
        This catches both engine math AND profile-override propagation."""
        from backend.trading_engine import get_tightened_trailing_stop
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        assert get_tightened_trailing_stop(15.0, profile, tighten_factor=0.5) == 3.0

    def test_returns_none_below_5pct_floor(self):
        """No trailing stop below 5% peak gain → tightening returns None too."""
        from backend.trading_engine import get_tightened_trailing_stop

        assert get_tightened_trailing_stop(3.0, {}, tighten_factor=0.5) is None


# ── 4. Score crash decision matrix ────────────────────────────────────────────


class TestScoreCrashEvaluation:
    """Both files call evaluate_score_crash with identical args. The two
    documented config knobs — score_crash_drop_required and
    score_crash_ignore_if_profitable — are profile-overridable and the
    champion sets them to 25/20. Verify the override path AND the
    profitability skip path."""

    def _stable_history(self):
        """Shape returned by check_score_stability_from_history when
        scores have been consistently low for several scans."""
        return {
            "is_stable": True,
            "recent_scores": [40, 38, 35, 33],
            "avg_score": 36.5,
            "consecutive_low": 4,
            "recent_low_count": 3,
        }

    def test_champion_drop_required_25_holds_at_drop_22(self):
        """Champion requires 25pt drop (vs default 20). Drop of 22 should NOT trigger."""
        from backend.trading_engine import evaluate_score_crash
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        # purchase=70, current=48 → drop=22, current<50 threshold but drop<25 required
        result = evaluate_score_crash(
            current_score=48,
            purchase_score=70,
            gain_pct=0,
            stability=self._stable_history(),
            profile=profile,
        )
        assert result is None  # No crash registered

    def test_champion_drop_required_25_triggers_at_drop_30(self):
        """Same champion config, drop of 30 → real crash."""
        from backend.trading_engine import evaluate_score_crash
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        result = evaluate_score_crash(
            current_score=40,
            purchase_score=70,
            gain_pct=0,
            stability=self._stable_history(),
            profile=profile,
        )
        assert result is not None
        assert result["should_sell"] is True

    def test_champion_profitable_skip_at_20pct(self):
        """Champion sets ignore_if_profitable=20; up 22% should skip even on real crash."""
        from backend.trading_engine import evaluate_score_crash
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        result = evaluate_score_crash(
            current_score=40,
            purchase_score=70,
            gain_pct=22.0,  # > 20 profitable threshold
            stability=self._stable_history(),
            profile=profile,
        )
        assert result is not None
        assert result["should_sell"] is False
        assert "profitable" in result["reason"].lower()


# ── 5. NaN-safe handling (Round 1 + 2 of the documented poisoning bug) ────────


class TestNanSafeHandling:
    """Round 2 of the NaN poisoning bug (commits ~Mar 12) found that
    `float('nan')` is truthy and passes `or 0`. _nan_safe is the dedicated
    fix — verify it catches NaN, None, and that valid floats pass through."""

    def test_none_returns_default(self):
        from backend.trading_utils import _nan_safe

        assert _nan_safe(None) == 0
        assert _nan_safe(None, default=-1) == -1

    def test_nan_returns_default(self):
        from backend.trading_utils import _nan_safe

        nan = float("nan")
        assert _nan_safe(nan) == 0  # NaN != NaN per IEEE 754
        assert _nan_safe(nan, default=99) == 99

    def test_valid_float_passes_through(self):
        from backend.trading_utils import _nan_safe

        assert _nan_safe(0.0) == 0.0
        assert _nan_safe(75.5) == 75.5
        assert _nan_safe(-3.0) == -3.0

    def test_get_effective_score_handles_nan_score(self):
        """get_effective_score uses _nan_safe to defang NaN current_score.
        Without this, a NaN here propagates into score_crash gates and
        partial profit decisions in evaluate_sells."""
        from backend.ai_trader import get_effective_score

        position = _make_position(current_score=float("nan"), is_growth_stock=False)
        # NaN should be treated as 0, not propagated
        result = get_effective_score(position, use_current=True)
        assert result == 0
        assert not (isinstance(result, float) and result != result)  # not NaN

    def test_get_effective_score_growth_stock_uses_growth_score(self):
        """Growth path is a separate code branch that also uses _nan_safe."""
        from backend.ai_trader import get_effective_score

        position = _make_position(
            is_growth_stock=True,
            current_growth_score=82.0,
            current_score=None,  # ignored for growth stocks
        )
        assert get_effective_score(position, use_current=True) == 82.0


# ── 6. Sector limit (DB-backed; mirrored in backtester._check_sector_limit) ───


class TestSectorLimit:
    """ai_trader.check_sector_limit gates buys/pyramids on:
       - count cap (MAX_STOCKS_PER_SECTOR=4)
       - allocation cap (MAX_SECTOR_ALLOCATION=0.50)
       - 2% min-room rule (return 0 if remaining_room <= 0.02)
    backtester._check_sector_limit mirrors only the count piece;
    backtester._evaluate_buys mirrors the allocation+2% logic inline.
    Sync tests below pin the count cap and the empty-portfolio path."""

    def test_empty_portfolio_returns_full_amount(self):
        from backend.ai_trader import check_sector_limit
        from backend.database import Stock

        db = _fresh_session()
        db.add(Stock(ticker="AAPL", sector="Technology"))
        db.commit()

        amount, reason = check_sector_limit(db, "AAPL", buy_amount=5000.0, user_id=1)
        # Empty portfolio (no positions) — should return original amount
        assert amount == 5000.0
        assert reason == ""

    def test_max_stocks_per_sector_blocks_5th_stock(self):
        """4 positions in same sector → 5th candidate blocked (returns 0)."""
        from backend.ai_trader import check_sector_limit
        from backend.database import Stock, AIPortfolioPosition, AIPortfolioConfig

        db = _fresh_session()
        # Seed config so get_portfolio_value works
        db.add(AIPortfolioConfig(user_id=1, current_cash=10000.0))
        # Seed 4 tech positions + 1 candidate in same sector
        for t in ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]:
            db.add(Stock(ticker=t, sector="Technology"))
        for t in ["AAPL", "MSFT", "GOOGL", "META"]:
            db.add(_make_position(ticker=t, current_value=5000.0))
        db.commit()

        amount, reason = check_sector_limit(db, "NVDA", buy_amount=5000.0, user_id=1)
        assert amount == 0
        assert "Max" in reason and "Technology" in reason

    def test_unknown_sector_treated_as_unknown_bucket(self):
        """Stock with no sector falls into 'Unknown' bucket — should still
        permit the buy when no other 'Unknown' positions exist."""
        from backend.ai_trader import check_sector_limit
        from backend.database import Stock, AIPortfolioConfig

        db = _fresh_session()
        db.add(AIPortfolioConfig(user_id=1, current_cash=10000.0))
        db.add(Stock(ticker="MYSTERY", sector=None))
        db.commit()

        amount, reason = check_sector_limit(db, "MYSTERY", buy_amount=2000.0, user_id=1)
        # With no positions in same sector and no portfolio value, fall through
        assert amount == 2000.0


# ── 7. Score stability DB adapter ─────────────────────────────────────────────


class TestCheckScoreStabilityAdapter:
    """ai_trader.check_score_stability is a DB→engine adapter:
       - Fetches StockScore rows ordered desc.timestamp (most recent first).
       - Reverses to chronological order before calling engine.
    backtester._check_score_stability stores scores chronologically in memory.
    The reversal is the easy place to introduce a divergence — pin it."""

    def test_empty_history_returns_stable_with_warning(self):
        from backend.ai_trader import check_score_stability
        from backend.database import Stock

        db = _fresh_session()
        # No Stock row → ai_trader's special "Stock not found" path
        result = check_score_stability(db, "GHOST", current_score=40)
        assert result["is_stable"] is True
        assert result["warning"] == "Stock not found"

    def test_stock_with_no_scores_gets_limited_history_warning(self):
        from backend.ai_trader import check_score_stability
        from backend.database import Stock

        db = _fresh_session()
        db.add(Stock(ticker="AAPL"))
        db.commit()

        result = check_score_stability(db, "AAPL", current_score=40)
        assert result["is_stable"] is True
        assert result["warning"] == "Limited history"

    def test_db_desc_order_reversed_to_chronological_for_engine(self):
        """Insert scores in chronological order (high → low). DB returns
        them as desc (most recent low first). The adapter must reverse so
        the engine sees [80, 75, 35, 30] (chronological) and detects a
        real consecutive drop, not a blip."""
        from backend.ai_trader import check_score_stability
        from backend.database import Stock, StockScore

        db = _fresh_session()
        stock = Stock(ticker="AAPL")
        db.add(stock)
        db.commit()

        # Chronological: 80 (oldest) → 75 → 35 → 30 (most recent)
        # We insert in chronological order so timestamps are increasing.
        # DB will return them desc (30, 35, 75, 80); adapter reverses.
        # One score per DAY: the D2 daily-clock dedups multiple same-day scans
        # to one row/day, so the lookback must span distinct days to see 4 points.
        base_ts = datetime.now(timezone.utc) - timedelta(days=8)
        for i, score in enumerate([80, 75, 35, 30]):
            ts = base_ts + timedelta(days=i)
            db.add(StockScore(
                stock_id=stock.id,
                total_score=float(score),
                timestamp=ts,
                date=ts.date(),
            ))
        db.commit()

        # current_score=30 with prior scans [80, 75, 35, 30] (chronological):
        # 2 consecutive low + 2 of last 3 low → real crash, not blip → is_stable=True
        result = check_score_stability(db, "AAPL", current_score=30, lookback=4)
        assert result["is_stable"] is True
        assert result["consecutive_low"] >= 2


# ── 8. Sell reason categorization (used in ai_trader signal_factors) ──────────


class TestCategorizeSellReason:
    """ai_trader writes categorize_sell_reason output into trade.signal_factors;
    backtester writes the same reason strings into BacktestTrade.reason. The
    string contracts must match — backtest analytics group by these prefixes."""

    @pytest.mark.parametrize("reason,expected", [
        ("STOP LOSS: Down 7.5%", "STOP LOSS"),
        ("TRAILING STOP: Peak $100 → $85", "TRAILING STOP"),
        ("PARTIAL TRAILING STOP: Peak $100 → $85, keeping 50%", "PARTIAL TRAILING"),
        ("PARTIAL PROFIT 25%: Up 25%", "PARTIAL PROFIT"),
        ("SCORE CRASH: 80 → 40", "SCORE CRASH"),
        ("TAKE PROFIT: Up 75%", "TAKE PROFIT"),
        ("PROTECT GAINS: Up 25% but score weak", "PROTECT GAINS"),
        ("WEAK POSITION: -3.2%", "WEAK POSITION"),
        ("PRE-EARNINGS STOP: 3d to earnings", "PRE-EARNINGS"),
        ("Some other reason", "OTHER"),
    ])
    def test_categorization_contract(self, reason, expected):
        from backend.trading_engine import categorize_sell_reason

        assert categorize_sell_reason(reason) == expected


# ── 9. Base quality bonus (used in evaluate_buys composite scoring) ───────────


class TestBaseQualityBonus:
    """Both ai_trader.evaluate_buys and backtester._evaluate_buys feed
    calculate_base_quality_bonus into composite_score. The pattern→points
    map is a hard contract — base ranking changes here shift the ranked
    candidate list in BOTH simulators identically."""

    def test_cup_with_handle_max_with_long_consolidation(self):
        from backend.trading_engine import calculate_base_quality_bonus

        # cup_with_handle base = 10pts, +5 for 8+ weeks → 15
        assert calculate_base_quality_bonus("cup_with_handle", 10) == 15

    def test_pattern_priority_ordering(self):
        from backend.trading_engine import calculate_base_quality_bonus

        # Same weeks (5w → +1 bonus). Verify pattern hierarchy:
        # cup_with_handle(10) > cup(8) > double_bottom(7) > flat(6).
        cwh = calculate_base_quality_bonus("cup_with_handle", 5)
        cup = calculate_base_quality_bonus("cup", 5)
        db = calculate_base_quality_bonus("double_bottom", 5)
        flat = calculate_base_quality_bonus("flat", 5)

        assert cwh > cup > db > flat
        assert cwh == 11  # 10 + 1
        assert cup == 9   # 8 + 1
        assert db == 8    # 7 + 1
        assert flat == 7  # 6 + 1

    def test_no_base_returns_zero(self):
        from backend.trading_engine import calculate_base_quality_bonus

        assert calculate_base_quality_bonus("none", 10) == 0
        assert calculate_base_quality_bonus("", 10) == 0
        assert calculate_base_quality_bonus(None, 10) == 0


# ── 10. Partial trailing stop gate — both ai_trader sites use the helper ──────


class TestPartialTrailingGateMigrated:
    """Both ai_trader entry points (`_check_and_execute_stop_losses_impl` and
    `evaluate_sells`) must route the partial-trailing-stop decision through
    `should_take_partial_on_trailing_stop`. Pre-migration, the first site
    had an inline `partial_on_trailing AND pyramid >= min AND score >= min`
    gate while the second called the helper — same gate, two implementations
    in the same file, ready to silently desync.

    Backtester._evaluate_sells already uses the helper. These tests pin the
    invariant that all three call sites stay routed through one source of
    truth.
    """

    def test_check_stop_losses_impl_uses_helper(self):
        import inspect
        from backend import ai_trader

        src = inspect.getsource(ai_trader._check_and_execute_stop_losses_impl)
        # Helper must be invoked (refactor preserved).
        assert "should_take_partial_on_trailing_stop(" in src, (
            "_check_and_execute_stop_losses_impl no longer routes the partial-"
            "trailing-stop decision through should_take_partial_on_trailing_stop. "
            "If the gate moved, this regression test must move with it."
        )
        # The inline 3-line gate must NOT have come back. Strip comments
        # so a documentation reference doesn't trip this guard.
        code_only = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        assert "score >= partial_min_score" not in code_only, (
            "Inline `score >= partial_min_score` reappeared in "
            "_check_and_execute_stop_losses_impl. Use should_take_partial_on_"
            "trailing_stop instead — that is the single source of truth shared "
            "with backtester._evaluate_sells."
        )

    def test_evaluate_sells_uses_helper(self):
        import inspect
        from backend import ai_trader

        src = inspect.getsource(ai_trader.evaluate_sells)
        assert "should_take_partial_on_trailing_stop(" in src
        # Strip comments before scanning so the documentation reference to
        # the legacy `score_available and score >= partial_min_score` gate
        # in the helper-call block doesn't trip the inline-pattern guard.
        code_only = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        assert "score >= partial_min_score" not in code_only

    def test_backtester_evaluate_sells_uses_helper(self):
        import inspect
        from backend.backtester import BacktestEngine

        src = inspect.getsource(BacktestEngine._evaluate_sells)
        assert "should_take_partial_on_trailing_stop(" in src
        code_only = "\n".join(
            line.split("#", 1)[0] for line in src.splitlines()
        )
        assert "score >= partial_min_score" not in code_only

    def test_helper_semantics_unchanged_after_migration(self):
        """End-to-end invariant: the migrated site computes the same boolean
        as the helper for the canonical (pyramid_count, score) corners.

        If anyone reverts the refactor with a subtly different inline gate
        (e.g., `>` instead of `>=`, or drops a config knob), the helper's
        behavior contract diverges from the call site and this test catches
        the resulting comparison.
        """
        from backend.trading_utils import should_take_partial_on_trailing_stop

        # Same corners as the boundary tests in test_trading_utils_sync.py,
        # re-asserted here so the migrated call site's contract stays pinned
        # next to the source-inspection guards above.
        cases = [
            (2, 65, True),    # both at minimum
            (2, 64.99, False),  # score just below
            (1, 95, False),   # one short on pyramids
            (0, 95, False),   # no pyramids
            (5, 0, False),    # score-unavailable sentinel
        ]
        for pyramid_count, score, expected in cases:
            assert should_take_partial_on_trailing_stop(
                pyramid_count=pyramid_count, score=score
            ) is expected, f"helper boundary moved for ({pyramid_count}, {score})"
