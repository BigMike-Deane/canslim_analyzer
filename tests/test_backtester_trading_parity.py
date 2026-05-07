"""
Behavioral parity tests: backtester._evaluate_* must produce the same trade
decisions as ai_trader.evaluate_* under matching scenarios.

Why this file exists
--------------------
The Approach 2 live A/B (commit `ec73f83`, deployed 2026-05-07) reads its
keep/revert signal from frozen-snapshot backtester replays of pre vs post
trades. If a future ai_trader change drifts the live decision logic but the
backtester isn't updated in lockstep, the backtest replay produces a
different decision tree from live trading — and the eval signal becomes
untrustworthy regardless of how clean its math is.

`tests/test_ai_trader_sync.py` pins STRUCTURAL sync (helper imports,
source-grep guards). `tests/test_backtester_scorer_parity.py` pins C-SCORE
parity from the May 6 canslim_scorer refactor. What was missing — and what
this file pins — is BEHAVIORAL parity on the trading-decision functions:
same scenario → same trade. The matrix-based ML graduation that worked in
April was trustable BECAUSE this kind of parity held silently. A single
silent drift would invalidate the next round of decisions.

The companion file `tests/test_ai_trader_coverage.py` (commit 45948a4)
covers the live ai_trader side of these scenarios. This file covers the
backtester sibling under the same scenario shape — different harness, same
expected decision branch.

Triage
------
  Tier 1 (live A/B reads from these decisions every replay):
    - _evaluate_sells: stop loss, trailing stop (with pyramid widening),
      partial trailing for high-conviction, score crash with profitable
      bypass, partial profit 25%/40% tier delta math, pre-earnings
      tightening, PROTECT GAINS, WEAK POSITION, NaN-safe skip.
    - _evaluate_pyramids: gain<2.5%, score<70, pyramid_count>=2,
      MAX_POSITION_ALLOCATION cap, insufficient cash.

  Tier 2 (helpers consumed by Tier 1, already covered by other parity
  files but reaffirmed here under the BacktestEngine harness):
    - _check_score_stability adapter (delegates to engine helper).

  Tier 3 (intentionally NOT covered):
    - _evaluate_buys: 1100+ lines of ranking + composite-score logic.
      Early-exit gates differ from ai_trader (backtester drives gating
      from _simulate_day → market_state path, not inline). Behavioral
      parity at the ranking level needs full Stock graph + earnings
      audit + ML model + correlation check — better suited to the
      live A/B framework itself, which already runs this comparison
      every time a backtest queues. Pinning the early-exit gates here
      would create a false sense of full-pipeline parity.
    - SCALE-IN PULLBACK pyramid path: depends on volume_ratio +
      peak-pullback geometry that's noisy enough to deserve its own
      dedicated test class. Shape is verified structurally in
      test_pyramid_action_sync.py.

Pinned regression guards (named):
  - Partial Profit 25%/40% tier delta (Feb 24 fix +25.9pp): backtester
    must take the same delta = sell_pct - partial_taken via shared helper.
    TestBacktesterSellsPartialProfitDelta.
  - Score-crash 20% profitable bypass (champion ignore_if_profitable=20):
    backtester must NOT sell when up >20% even on real crash.
    TestBacktesterSellsScoreCrash.
  - Trailing stop pyramid widening (+2%/pyramid, max +6%): backtester
    must apply same widening as ai_trader.
    TestBacktesterSellsTrailingStop.
  - Pyramid eligibility gates (gain>=2.5%, score>=70, count<2): backtester
    must reject under same conditions.
    TestBacktesterPyramids.

Harness
-------
- BacktestEngine constructed with a MagicMock(Session) + a real BacktestRun
  shape. SimulatedPosition objects added directly to engine.positions.
- data_provider mocked at the boundary (get_price_on_date,
  get_market_direction, get_atr, get_vix_proxy, get_volume_ratio).
- Strategy profile loaded via get_strategy_profile (real YAML), so band
  thresholds match production exactly.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import BacktestRun


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """MagicMock Session pre-loaded with a BacktestRun shaped to the
    nostate_optimized champion config. Mirrors the pattern in test_backtester.py
    for consistency with the codebase's existing engine-construction style."""
    session = MagicMock()
    backtest = BacktestRun(
        id=1,
        name="parity test",
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
        starting_cash=25000.0,
        stock_universe="all",
        max_positions=8,
        min_score_to_buy=72,
        sell_score_threshold=45,
        stop_loss_pct=7.0,
        strategy="nostate_optimized",
        status="pending",
    )
    session.get.return_value = backtest
    session.query.return_value.all.return_value = []
    return session, backtest


@pytest.fixture
def engine(mock_db):
    """BacktestEngine instance with bullish-market mocked data_provider."""
    from backend.backtester import BacktestEngine

    session, _ = mock_db
    eng = BacktestEngine(session, 1)
    eng.data_provider = MagicMock()
    # Bullish market by default — mirrors stub_market_bullish in
    # tests/test_ai_trader_coverage.py
    eng.data_provider.get_market_direction.return_value = {
        "weighted_signal": 2.0,
        "spy": {"price": 500.0, "ma_50": 480.0, "ma_200": 450.0, "ema_21": 495.0},
    }
    eng.data_provider.get_atr.return_value = 0.0  # No ATR widening — keep tests focused
    eng.data_provider.get_vix_proxy.return_value = 18.0  # Mid-range — no VIX adjustment
    eng.data_provider.get_volume_ratio.return_value = 1.0
    eng.static_data = {}
    return eng


def _add_position(engine, ticker, **overrides):
    """Insert a SimulatedPosition into engine.positions with sensible defaults
    mirroring _seed_position in tests/test_ai_trader_coverage.py."""
    from backend.backtester import SimulatedPosition

    defaults = dict(
        ticker=ticker,
        shares=100.0,
        cost_basis=100.0,
        purchase_date=date.today() - timedelta(days=30),
        purchase_score=80.0,
        peak_price=115.0,
        peak_date=date.today() - timedelta(days=5),
        is_growth_stock=False,
        sector="Technology",
        partial_profit_taken=0.0,
        pyramid_count=0,
    )
    defaults.update(overrides)
    pos = SimulatedPosition(**defaults)
    engine.positions[ticker] = pos
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells stop loss
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsStopLoss:
    """Mirrors TestSellsStopLoss in test_ai_trader_coverage.py.
    Same scenario → same trade decision."""

    def test_stop_loss_triggers_below_threshold(self, engine):
        """Down 7.5% with champion stop=7 → STOP LOSS fires."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=100.0)
        engine.data_provider.get_price_on_date.return_value = 92.5

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert any("STOP LOSS" in s.reason for s in sells)
        assert sells[0].priority == 1

    def test_stop_loss_holds_above_threshold(self, engine):
        """Down 6% < 7% threshold → no stop."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=100.0)
        engine.data_provider.get_price_on_date.return_value = 94.0

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert all("STOP LOSS" not in s.reason for s in sells)

    def test_skip_position_with_nan_price(self, engine):
        """NaN price must be skipped silently — same defensive guard as ai_trader."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=100.0)
        engine.data_provider.get_price_on_date.return_value = float("nan")

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert sells == []

    def test_bearish_market_uses_tighter_stop(self, engine):
        """Champion bearish_stop_loss_pct=6 kicks in when SPY < 50MA. -6.5%
        breaches it."""
        engine.data_provider.get_market_direction.return_value = {
            "weighted_signal": -2.0,
            "spy": {"price": 400.0, "ma_50": 480.0},
        }
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=100.0)
        engine.data_provider.get_price_on_date.return_value = 93.5

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert any("STOP LOSS" in s.reason and "bearish" in s.reason for s in sells)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells trailing stop (+ pyramid widening + partial)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsTrailingStop:
    """Mirrors TestSellsTrailingStop. Same band thresholds, same pyramid widening
    (+2%/pyramid, max +6% via apply_pyramid_widening shared helper)."""

    def test_trailing_triggers_at_50plus_band(self, engine):
        """peak_gain=60, champion 50+ band=25%. 26% drop from peak triggers."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=160.0)
        engine.data_provider.get_price_on_date.return_value = 118.0  # 26.25% drop

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert any("TRAILING STOP" in s.reason for s in sells)

    def test_trailing_holds_within_band(self, engine):
        """20% drop < 25% band → no trigger."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=160.0)
        engine.data_provider.get_price_on_date.return_value = 128.0  # exactly 20%

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 70}})
        assert all("TRAILING STOP" not in s.reason for s in sells)

    def test_partial_trailing_when_pyramided_2x_and_high_score(self, engine):
        """High-conviction (pyramid=2 AND score>=65) → partial sell, peak resets.
        With 2 pyramids the 25% band widens to 29%; current=$110 → 31.25% drop > 29%."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=160.0, pyramid_count=2)
        engine.data_provider.get_price_on_date.return_value = 110.0

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 75}})
        partials = [s for s in sells if s.is_partial and "TRAILING STOP" in s.reason]
        assert len(partials) == 1
        assert partials[0].sell_pct == 50  # default partial_sell_pct
        # Side-effect: peak resets to current_price for the remainder
        assert engine.positions["TEST"].peak_price == 110.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells score crash with profitable bypass
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsScoreCrash:
    """Champion: score_crash_drop_required=25, ignore_if_profitable=20.
    Same engine helper (evaluate_score_crash) drives both — but we still
    pin the integration boundary at the backtester to catch any wiring drift."""

    def _seed_stable_low_history(self, engine, ticker, scores=(40, 38, 35, 33)):
        """Pre-populate score_history so check_score_stability_from_history
        classifies the current low as a stable consecutive run, not a blip."""
        engine.score_history[ticker] = list(scores)

    def test_score_crash_triggers_at_drop_30(self, engine):
        """Purchase=80, current=40 → drop=40, current<50, NOT profitable → SCORE CRASH."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=100.0,
                      purchase_score=80.0)
        self._seed_stable_low_history(engine, "TEST")
        engine.data_provider.get_price_on_date.return_value = 98.0  # gain=-2%

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 40}})
        assert any("SCORE CRASH" in s.reason for s in sells)

    def test_score_crash_skipped_when_profitable(self, engine):
        """Same crash, but up 22% → ignore_if_profitable=20 bypass kicks in.
        Pinned regression: load-bearing for the champion config."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=125.0,
                      purchase_score=80.0)
        self._seed_stable_low_history(engine, "TEST")
        engine.data_provider.get_price_on_date.return_value = 122.0  # gain=22%

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 40}})
        assert all("SCORE CRASH" not in s.reason for s in sells)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells partial profit delta (Feb 24 +25.9pp regression guard)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsPartialProfitDelta:
    """Pinned regression: backtester must use the SAME delta math as ai_trader
    (take_pct = sell_pct - partial_taken, NOT sell_pct). Both files route through
    trading_engine.get_partial_profit_action — but pin the integration boundary
    so a future divergent inline reimplementation can't slip in."""

    def test_25pct_tier_first_take_full_25(self, engine):
        """First partial at 25%+ gain: sell exactly 25% (delta = 25 - 0)."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=125.0,
                      partial_profit_taken=0)
        engine.data_provider.get_price_on_date.return_value = 125.0

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 75}})
        partials = [s for s in sells if s.is_partial and "PARTIAL PROFIT" in s.reason]
        assert len(partials) == 1
        assert partials[0].sell_pct == 25  # delta = 25 - 0

    def test_40pct_tier_with_25_already_taken_returns_25_delta(self, engine):
        """The exact bug shape from Feb 24: when 25% already taken, 40% tier
        delta should be (50 - 25) = 25, NOT 50. +25.9pp 4yr depends on this."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=140.0,
                      partial_profit_taken=25)
        engine.data_provider.get_price_on_date.return_value = 140.0

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 75}})
        partials = [s for s in sells if s.is_partial and "PARTIAL PROFIT" in s.reason]
        assert len(partials) == 1
        assert partials[0].sell_pct == 25  # 50 - 25, not 50


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells pre-earnings tightening
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsPreEarningsTighten:
    """Non-CS positions within days_to_earnings <= 5 get half-stop tightening
    + a 25% partial-protection slug. Mirrors ai_trader's pre-earnings path."""

    def test_non_cs_within_earnings_gets_partial_protection(self, engine):
        """Up 12%, 3 days to earnings, partial_taken=0 → 25% partial fires."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=113.0,
                      partial_profit_taken=0)
        engine.data_provider.get_price_on_date.return_value = 112.0
        # static_data drives the days_to_earnings lookup in _evaluate_sells
        engine.static_data["TEST"] = {"days_to_earnings": 3}

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 75}})
        pre_earn = [s for s in sells if "PRE-EARNINGS" in s.reason]
        assert len(pre_earn) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_sells take-profit / weak-position cutoff
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterSellsTakeProfitAndWeak:
    """PROTECT GAINS (gain>=20 AND score<sell_threshold) and WEAK POSITION
    (gain<10 AND score<sell_threshold). Mirrors TestSellsTakeProfitAndWeak."""

    def test_protect_gains_when_up_30_but_score_weak(self, engine):
        """purchase=55 keeps score-crash from triggering (drop<drop_required=25),
        so the engine flows down to the PROTECT GAINS branch."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=132.0,
                      purchase_score=55.0)
        engine.data_provider.get_price_on_date.return_value = 130.0  # gain=30%

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 40}})
        assert any("PROTECT GAINS" in s.reason for s in sells)

    def test_weak_position_flat_with_low_score(self, engine):
        """Up 5%, score 40 < sell_threshold=45 → WEAK POSITION cutoff."""
        _add_position(engine, "TEST", cost_basis=100.0, peak_price=106.0,
                      purchase_score=55.0)
        engine.data_provider.get_price_on_date.return_value = 105.0

        sells = engine._evaluate_sells(date.today(), {"TEST": {"total_score": 40}})
        assert any("WEAK POSITION" in s.reason for s in sells)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _evaluate_pyramids
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterPyramids:
    """Mirrors TestEvaluatePyramids. Same eligibility gates: gain>=2.5%,
    score>=70, pyramid_count<2, alloc<MAX_POSITION_ALLOCATION, cash>=200."""

    def test_eligible_winner_added_to_pyramid_list(self, engine):
        """Position 18% of portfolio (under 25% cap), up 10%, score 80, no
        pyramids yet, purchased 35d ago (past cooldown) → pyramid candidate."""
        engine.cash = 20000.0
        # peak_price needed for SCALE-IN code; set above current to avoid scale path
        _add_position(engine, "TEST", shares=40.0, cost_basis=100.0, peak_price=112.0,
                      purchase_score=80.0, pyramid_count=0,
                      purchase_date=date.today() - timedelta(days=35))
        engine.data_provider.get_price_on_date.return_value = 110.0
        # _get_portfolio_value uses cost_basis when no current_date provided;
        # but here _evaluate_pyramids passes current_date. Mock get_price_on_date
        # for portfolio value too — same MagicMock returns 110.0 for any ticker.

        pyramids = engine._evaluate_pyramids(date.today(),
                                             {"TEST": {"total_score": 80, "is_breaking_out": False, "volume_ratio": 1.0}})
        assert len(pyramids) == 1
        assert pyramids[0].ticker == "TEST"
        assert pyramids[0].action == "PYRAMID"

    def test_max_2_pyramids_blocks_third(self, engine):
        """pyramid_count=2 → no further adds (O'Neil 60/40 rule)."""
        engine.cash = 20000.0
        _add_position(engine, "TEST", shares=40.0, cost_basis=100.0, peak_price=112.0,
                      purchase_score=80.0, pyramid_count=2,
                      purchase_date=date.today() - timedelta(days=35))
        engine.data_provider.get_price_on_date.return_value = 110.0

        pyramids = engine._evaluate_pyramids(
            date.today(),
            {"TEST": {"total_score": 80, "is_breaking_out": False, "volume_ratio": 1.0}},
        )
        # The SCALE-IN branch can fire on pyramid_count>=2 if pullback geometry
        # matches — but our peak=112, current=110 is only 1.8% pullback (< 3% min)
        # so SCALE-IN won't trigger either. Standard pyramid is blocked by count.
        # Result: pyramids list should be empty.
        scale_ins = [p for p in pyramids if "SCALE-IN" in p.reason]
        regular_pyrs = [p for p in pyramids if "Winner" in p.reason]
        assert regular_pyrs == [], "regular pyramid must be blocked at count=2"
        # SCALE-IN may or may not fire depending on geometry; if it does it's
        # not a regression of the count-cap rule.

    def test_low_score_blocks_pyramid(self, engine):
        """Score 65 < 70 threshold → no pyramid."""
        engine.cash = 20000.0
        _add_position(engine, "TEST", shares=40.0, cost_basis=100.0, peak_price=112.0,
                      pyramid_count=0,
                      purchase_date=date.today() - timedelta(days=35))
        engine.data_provider.get_price_on_date.return_value = 110.0

        pyramids = engine._evaluate_pyramids(
            date.today(),
            {"TEST": {"total_score": 65, "is_breaking_out": False, "volume_ratio": 1.0}},
        )
        assert pyramids == []

    def test_low_gain_blocks_pyramid(self, engine):
        """Gain 1.5% < 2.5% threshold → no pyramid."""
        engine.cash = 20000.0
        _add_position(engine, "TEST", shares=40.0, cost_basis=100.0, peak_price=102.0,
                      pyramid_count=0,
                      purchase_date=date.today() - timedelta(days=35))
        engine.data_provider.get_price_on_date.return_value = 101.5

        pyramids = engine._evaluate_pyramids(
            date.today(),
            {"TEST": {"total_score": 80, "is_breaking_out": False, "volume_ratio": 1.0}},
        )
        assert pyramids == []

    def test_insufficient_cash_blocks_pyramid(self, engine):
        """cash < 200 → no pyramid."""
        engine.cash = 50.0  # below floor
        _add_position(engine, "TEST", shares=40.0, cost_basis=100.0, peak_price=112.0,
                      pyramid_count=0,
                      purchase_date=date.today() - timedelta(days=35))
        engine.data_provider.get_price_on_date.return_value = 110.0

        pyramids = engine._evaluate_pyramids(
            date.today(),
            {"TEST": {"total_score": 80, "is_breaking_out": False, "volume_ratio": 1.0}},
        )
        assert pyramids == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — _check_score_stability adapter (delegation parity)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktesterScoreStability:
    """Pinned: backtester._check_score_stability and ai_trader.check_score_stability
    must produce equivalent results given identical (chronological) score histories.
    The delegation target is the same engine helper, but the input adapter shape
    differs — backtester passes from in-memory list (chronological), ai_trader
    passes from DB query (desc, then reversed). Both must end up identical."""

    def test_chronological_history_matches_engine_helper(self, engine):
        """Same input as engine helper → same result."""
        from backend.trading_engine import check_score_stability_from_history

        engine.score_history["TEST"] = [80, 75, 35, 30]
        result_via_engine = engine._check_score_stability("TEST", current_score=30)
        result_via_helper = check_score_stability_from_history(
            [80, 75, 35, 30], 30, threshold=50,
        )

        # The adapter is a thin pass-through — keys must match exactly
        assert result_via_engine == result_via_helper

    def test_no_history_handled_gracefully(self, engine):
        """Empty history → engine helper returns the limited-history shape."""
        from backend.trading_engine import check_score_stability_from_history

        result = engine._check_score_stability("UNKNOWN", current_score=30)
        expected = check_score_stability_from_history([], 30, threshold=50)
        assert result == expected
