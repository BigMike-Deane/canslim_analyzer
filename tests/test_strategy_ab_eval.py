"""Tests for /api/admin/strategy-ab-eval endpoint logic.

Targets the live A/B comparison framework: the pre/post window summarizer,
the delta computation, the decision rule, and the endpoint's input
validation + filtering invariants.

Built for the Approach 2 evaluation milestones: 2026-05-21 (sanity check)
and 2026-06-04 to 2026-06-18 (real evaluation, cutoff 2026-05-07).
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi import HTTPException

from backend.database import (
    init_db, SessionLocal, AIPortfolioTrade, AIPortfolioConfig,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.commit()
        db.close()


def _make_trade(action="BUY", reason=None, signal_factors=None,
                realized_gain=None, cost_basis=None, shares=100.0,
                price=10.0, executed_at=None, user_id=1, ticker="TEST"):
    return AIPortfolioTrade(
        ticker=ticker, action=action, shares=shares, price=price,
        total_value=shares * price,
        executed_at=executed_at or datetime.now(timezone.utc),
        cost_basis=cost_basis, realized_gain=realized_gain,
        signal_factors=json.dumps(signal_factors) if signal_factors else None,
        reason=reason, user_id=user_id,
    )


def _seed_user_strategy(db, user_id, strategy, starting_cash=25000.0):
    cfg = AIPortfolioConfig(
        user_id=user_id, strategy=strategy, is_active=True,
        starting_cash=starting_cash, current_cash=starting_cash,
        max_positions=8, max_position_pct=12.0,
        min_score_to_buy=72, sell_score_threshold=45,
        stop_loss_pct=7.0, take_profit_pct=75.0,
    )
    db.add(cfg)
    db.commit()


def _call(db, **kwargs):
    """Call the endpoint function directly with all Query-defaulted params
    explicitly resolved. When invoked outside FastAPI's dependency-resolution,
    Query() defaults arrive as Query objects, not their underlying values —
    which breaks numeric comparisons inside the endpoint. Match strategy-health's
    convention of passing all params explicitly."""
    from backend.routes.admin import run_strategy_ab_comparison
    defaults = dict(
        strategy="nostate_optimized",
        cutoff_date="2026-04-15",  # 22 days before today=2026-05-07
        pre_window_days=14,
        post_window_days=14,
        exclude_pyramids=True,
        min_return_delta_pp=-5.0,
        min_sharpe_delta=0.0,
        min_post_sells=5,
        current_user=None,
        db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(run_strategy_ab_comparison(**defaults))


# ============================================================================
# _summarize_window — combines _summarize_trades with portfolio-level stats
# ============================================================================
class TestSummarizeWindow:
    def test_empty_window(self):
        from backend.routes.admin import _summarize_window
        s = _summarize_window([], days=14, starting_value=25000.0)
        assert s['total_trades'] == 0
        # None, not 0.0 (changed Jul-22): an empty window has NO baseline —
        # returning 0.0 let the UI compute deltas against a phantom flat
        # baseline when strategy reassignment emptied the pre window.
        assert s['total_return_pct'] is None
        assert s['capital_efficiency_pct'] is None  # no SELLs → undefined
        assert s['sharpe_per_trade'] is None
        assert s['realized_max_drawdown_pct'] is None
        assert s['starting_value'] == 25000.0

    def test_total_return_uses_starting_value(self):
        """Total return is realized_gain / starting_value. With $1000 gain
        on $25k base = 4%."""
        from backend.routes.admin import _summarize_window
        trades = [_make_trade(
            action="SELL", shares=10.0, cost_basis=100.0, realized_gain=1000.0,
        )]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        assert s['total_return_pct'] == 4.0
        assert s['total_realized_gain'] == 1000.0

    def test_capital_efficiency_separates_from_total_return(self):
        """capital_efficiency_pct = realized / cost basis (independent of
        portfolio size). 1 trade with $1000 gain on $1000 cost = 100%."""
        from backend.routes.admin import _summarize_window
        trades = [_make_trade(
            action="SELL", shares=10.0, cost_basis=100.0, realized_gain=1000.0,
        )]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        # cost_total = 10 * 100 = 1000; gain/cost = 100%
        assert s['capital_efficiency_pct'] == 100.0

    def test_sharpe_per_trade_requires_two_sells(self):
        from backend.routes.admin import _summarize_window
        trades = [_make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=5.0)]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        assert s['sharpe_per_trade'] is None  # n=1, no std

    def test_sharpe_per_trade_zero_variance_returns_none(self):
        """Two trades with identical returns → std=0 → sharpe undefined."""
        from backend.routes.admin import _summarize_window
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=10.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=10.0),
        ]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        assert s['sharpe_per_trade'] is None

    def test_sharpe_per_trade_positive_for_good_trades(self):
        """Win-heavy distribution should produce positive sharpe."""
        from backend.routes.admin import _summarize_window
        # 4 winners + 1 small loser
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=15.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=25.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=10.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=-5.0),
        ]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        assert s['sharpe_per_trade'] is not None
        assert s['sharpe_per_trade'] > 0

    def test_realized_max_drawdown_walks_chronologically(self):
        """+100, -50, -200 → peak 100, trough -150 → dd = 250 / (25000+100) ≈ 1.00%"""
        from backend.routes.admin import _summarize_window
        base_dt = datetime(2026, 4, 1, tzinfo=timezone.utc)
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=100.0,
                        executed_at=base_dt),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=-50.0,
                        executed_at=base_dt + timedelta(days=1)),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=-200.0,
                        executed_at=base_dt + timedelta(days=2)),
        ]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        # peak cum=100 at t1; cum at t3 = -150; dd = (100 - (-150)) / (25000+100) * 100
        expected = (250.0 / 25100.0) * 100
        assert abs(s['realized_max_drawdown_pct'] - round(expected, 2)) < 0.01

    def test_pyramid_rows_excluded_from_sells_summary(self):
        """If a PYRAMID row is in the trades list (caller chose not to
        filter), it shouldn't appear in SELL stats — only SELLs do."""
        from backend.routes.admin import _summarize_window
        trades = [
            _make_trade(action="PYRAMID", reason="PYRAMID: add"),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0),
        ]
        s = _summarize_window(trades, days=14, starting_value=25000.0)
        assert s['realized_sell_pct']['n'] == 1


# ============================================================================
# _compute_delta — straight subtraction with None handling
# ============================================================================
class TestComputeDelta:
    def test_basic_subtraction(self):
        from backend.routes.admin import _compute_delta
        pre = {'total_return_pct': 5.0, 'sharpe_per_trade': 0.3,
               'capital_efficiency_pct': 10.0, 'realized_max_drawdown_pct': 8.0,
               'entry_rate_per_day': 1.5, 'exit_rate_per_day': 0.8}
        post = {'total_return_pct': 7.5, 'sharpe_per_trade': 0.5,
                'capital_efficiency_pct': 12.0, 'realized_max_drawdown_pct': 6.0,
                'entry_rate_per_day': 1.2, 'exit_rate_per_day': 0.9}
        d = _compute_delta(pre, post)
        assert d['total_return_pct_delta'] == 2.5
        assert d['sharpe_per_trade_delta'] == 0.2
        assert d['realized_max_drawdown_pct_delta'] == -2.0
        assert d['entry_rate_per_day_delta'] == -0.3
        assert abs(d['exit_rate_per_day_delta'] - 0.1) < 1e-9

    def test_none_propagates_to_delta(self):
        """If either side is None on a metric, delta is None — comparator
        will refuse to decide on a missing metric."""
        from backend.routes.admin import _compute_delta
        pre = {'total_return_pct': None, 'sharpe_per_trade': 0.3}
        post = {'total_return_pct': 5.0, 'sharpe_per_trade': None}
        d = _compute_delta(pre, post)
        assert d['total_return_pct_delta'] is None
        assert d['sharpe_per_trade_delta'] is None

    def test_negative_pre_baseline(self):
        """Pre window had a losing strategy (-3%), post recovered to +2%.
        Delta should be +5pp."""
        from backend.routes.admin import _compute_delta
        pre = {'total_return_pct': -3.0}
        post = {'total_return_pct': 2.0}
        d = _compute_delta(pre, post)
        assert d['total_return_pct_delta'] == 5.0


# ============================================================================
# _decide — three branches + insufficient_data
# ============================================================================
class TestDecide:
    def _summary_with_sells(self, n=10):
        return {'realized_sell_pct': {'n': n}}

    def test_keep_when_both_pass(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': 2.0, 'sharpe_per_trade_delta': 0.1}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'keep'
        assert 'meets both bars' in d['decision_reason']

    def test_revert_when_both_fail(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': -10.0, 'sharpe_per_trade_delta': -0.2}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'revert'
        assert 'regressed' in d['decision_reason']

    def test_marginal_when_return_fails_but_sharpe_passes(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': -10.0, 'sharpe_per_trade_delta': 0.3}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'marginal'

    def test_marginal_when_sharpe_fails_but_return_passes(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': 2.0, 'sharpe_per_trade_delta': -0.2}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'marginal'

    def test_keep_at_exact_thresholds(self):
        """Boundary case: deltas exactly at threshold → still keep
        (>= comparison)."""
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': -5.0, 'sharpe_per_trade_delta': 0.0}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'keep'

    def test_insufficient_data_when_post_sells_too_few(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(2)  # too few
        delta = {'total_return_pct_delta': 5.0, 'sharpe_per_trade_delta': 0.5}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'insufficient_data'
        assert '2 closed SELLs' in d['decision_reason']

    def test_insufficient_data_when_delta_is_none(self):
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': None, 'sharpe_per_trade_delta': 0.5}
        criteria = {'min_return_delta_pp': -5.0, 'min_sharpe_delta': 0.0, 'min_post_sells': 5}
        d = _decide(pre, post, delta, criteria)
        assert d['decision'] == 'insufficient_data'

    def test_decision_criteria_echoed_in_response(self):
        """Operators reading the response should see exactly which thresholds
        were applied — important when defaults are overridden via query param."""
        from backend.routes.admin import _decide
        pre = self._summary_with_sells(15)
        post = self._summary_with_sells(15)
        delta = {'total_return_pct_delta': 2.0, 'sharpe_per_trade_delta': 0.1}
        criteria = {'min_return_delta_pp': -3.0, 'min_sharpe_delta': 0.05, 'min_post_sells': 10}
        d = _decide(pre, post, delta, criteria)
        assert d['decision_criteria'] == criteria


# ============================================================================
# Endpoint integration — input validation + filtering
# ============================================================================
class TestEndpointValidation:
    def test_rejects_bad_iso_date(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date="not-a-date")
        assert exc.value.status_code == 400
        assert "valid ISO date" in exc.value.detail

    def test_rejects_future_cutoff(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date=future)
        assert exc.value.status_code == 400
        assert "future" in exc.value.detail

    def test_rejects_post_window_exceeding_days_since_cutoff(self, db_session):
        """Asking for 60-day post window when cutoff was 5 days ago is
        meaningless — there's nothing to evaluate."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date=recent_cutoff, post_window_days=60)
        assert exc.value.status_code == 400
        assert "exceeds days-since-cutoff" in exc.value.detail

    def test_404_when_no_users_on_strategy(self, db_session):
        """No users on the requested strategy → 404, not silently zero
        trades (which would mislead the operator)."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        with pytest.raises(HTTPException) as exc:
            _call(db_session, strategy="nostate_cs_bear")
        assert exc.value.status_code == 404


# ============================================================================
# Endpoint integration — happy-path comparisons against fixtures
# ============================================================================
class TestEndpointHappyPath:
    def test_pre_post_split_at_cutoff(self, db_session):
        """Trades before cutoff land in pre, trades on/after land in post.
        Uses a synthetic past cutoff (2026-04-15) since today=2026-05-07
        and the production Approach 2 cutoff (2026-05-07) leaves only
        days_since_cutoff=1 of post-window headroom."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")

        # 4 pre-cutoff SELLs (Apr 5-14, all within pre_window=14d before Apr 15)
        for i, gain in enumerate([20, 15, -5, 30]):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=float(gain),
                executed_at=datetime(2026, 4, 5 + i*2, tzinfo=timezone.utc),
            ))
        # 3 post-cutoff SELLs (Apr 15, 17, 20 — all within post_window=14d)
        for i, gain in enumerate([15, 20, -3]):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=float(gain),
                executed_at=datetime(2026, 4, 15 + i*2, tzinfo=timezone.utc),
            ))
        db_session.commit()

        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14)
        assert result['pre']['realized_sell_pct']['n'] == 4
        assert result['post']['realized_sell_pct']['n'] == 3
        # Window dates
        assert result['experiment']['pre_window']['start'] == '2026-04-01'
        assert result['experiment']['pre_window']['end'] == '2026-04-15'
        assert result['experiment']['post_window']['start'] == '2026-04-15'
        assert result['experiment']['post_window']['end'] == '2026-04-29'

    def test_approach2_cutoff_with_minimal_post_window(self, db_session):
        """Concretely test against the Approach 2 deploy cutoff (2026-05-07).
        With today=2026-05-07, post_window can only be 1 day, but the date
        math should still work: a trade on May 6 lands pre, May 7 lands post."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")

        db_session.add(_make_trade(
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0,
            executed_at=datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=10.0,
            executed_at=datetime(2026, 5, 7, 8, 0, tzinfo=timezone.utc),
        ))
        db_session.commit()

        result = _call(db_session, cutoff_date="2026-05-07",
                       pre_window_days=30, post_window_days=1)
        assert result['pre']['realized_sell_pct']['n'] == 1
        assert result['post']['realized_sell_pct']['n'] == 1
        assert result['experiment']['pre_window']['start'] == '2026-04-07'
        assert result['experiment']['pre_window']['end'] == '2026-05-07'
        assert result['experiment']['post_window']['start'] == '2026-05-07'

    def test_pyramid_rows_excluded_by_default(self, db_session):
        """Same shape as strategy-health: filter both action='PYRAMID' and
        action='BUY' + reason LIKE 'PYRAMID:%'."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")

        # 1 standard BUY, 1 PYRAMID action, 1 BUY+PYRAMID reason — all pre
        db_session.add(_make_trade(
            action="BUY", reason="Standard breakout",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="PYRAMID", reason="PYRAMID: add",
            executed_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", reason="PYRAMID: live trader add",
            executed_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        ))
        db_session.commit()

        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14,
                       exclude_pyramids=True)
        # Only 1 standard BUY survives in pre window
        assert result['pre']['total_trades'] == 1

    def test_pyramid_inclusion_toggle(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")

        db_session.add(_make_trade(
            action="PYRAMID", reason="PYRAMID: add",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", reason="Standard",
            executed_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        ))
        db_session.commit()

        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14,
                       exclude_pyramids=False)
        assert result['pre']['total_trades'] == 2

    def test_strategy_filters_to_users_on_that_strategy(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        _seed_user_strategy(db_session, user_id=2, strategy="nostate_cs_bear")

        db_session.add(_make_trade(
            action="BUY", reason="opt trade", user_id=1,
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        # User 2 (cs_bear) — should NOT be counted under 'nostate_optimized'
        db_session.add(_make_trade(
            action="BUY", reason="bear trade", user_id=2,
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.commit()

        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14)
        assert result['pre']['total_trades'] == 1

    def test_starting_value_sums_across_users(self, db_session):
        """Two users on same strategy with $25k each → starting_value $50k.
        Affects total_return_pct denominator."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized", starting_cash=25000.0)
        _seed_user_strategy(db_session, user_id=2, strategy="nostate_optimized", starting_cash=25000.0)

        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14)
        assert result['experiment']['starting_value_reference'] == 50000.0
        assert result['pre']['starting_value'] == 50000.0
        assert result['post']['starting_value'] == 50000.0

    def test_response_shape_matches_brief(self, db_session):
        """Lock the top-level shape: experiment / summary / pre / post / delta / warnings."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14)
        assert set(result.keys()) == {'experiment', 'summary', 'pre', 'post', 'delta', 'warnings'}
        assert 'decision' in result['summary']
        assert 'decision_reason' in result['summary']
        assert 'decision_criteria' in result['summary']
        assert isinstance(result['warnings'], list)

    def test_warnings_fire_on_short_post_window(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=30, post_window_days=14)  # < 21 → should warn
        assert any('Post window has only 14 days' in w for w in result['warnings'])

    def test_decision_criteria_overrides_propagate(self, db_session):
        """Operator passes custom thresholds; they should appear in the
        decision response."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        result = _call(db_session, cutoff_date="2026-04-15",
                       pre_window_days=14, post_window_days=14,
                       min_return_delta_pp=-3.0, min_sharpe_delta=0.05,
                       min_post_sells=10)
        criteria = result['summary']['decision_criteria']
        assert criteria['min_return_delta_pp'] == -3.0
        assert criteria['min_sharpe_delta'] == 0.05
        assert criteria['min_post_sells'] == 10
