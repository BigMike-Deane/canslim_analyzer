"""Tests for /api/admin/strategy-ab-eval-trades endpoint.

Sibling of /api/admin/strategy-ab-eval that returns per-trade rows +
cumulative-return curves instead of aggregates. Powers the ABEval
dashboard chart + collapsible trade table.

Same window math, same user scope, same pyramid filter as the eval
endpoint — those invariants are tested here as parity checks against
the aggregate endpoint to catch any future drift between the two.
"""

import asyncio
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


def _make_trade(action="BUY", reason=None, realized_gain=None, cost_basis=None,
                shares=100.0, price=10.0, executed_at=None, user_id=1,
                ticker="TEST", holding_days=None):
    return AIPortfolioTrade(
        ticker=ticker, action=action, shares=shares, price=price,
        total_value=shares * price,
        executed_at=executed_at or datetime.now(timezone.utc),
        cost_basis=cost_basis, realized_gain=realized_gain,
        reason=reason, user_id=user_id, holding_days=holding_days,
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


def _call_trades(db, **kwargs):
    """Invoke the trades endpoint with all Query-defaulted params explicit.
    Same convention as test_strategy_ab_eval._call — when called outside
    FastAPI's dependency-resolution, Query() defaults arrive as Query
    objects and break numeric comparisons inside the endpoint."""
    from backend.routes.admin import run_strategy_ab_trades
    defaults = dict(
        strategy="nostate_optimized",
        cutoff_date="2026-04-15",
        pre_window_days=14,
        post_window_days=14,
        exclude_pyramids=True,
        current_user=None,
        db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(run_strategy_ab_trades(**defaults))


def _call_aggregate(db, **kwargs):
    """Invoke the aggregate /strategy-ab-eval endpoint — used for parity."""
    from backend.routes.admin import run_strategy_ab_comparison
    defaults = dict(
        strategy="nostate_optimized",
        cutoff_date="2026-04-15",
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
# _serialize_trade_row — row shape + realized_pct + hold_days
# ============================================================================
class TestSerializeTradeRow:
    def test_buy_row_has_no_realized_pct_or_hold_days(self):
        from backend.routes.admin import _serialize_trade_row
        t = _make_trade(action="BUY", price=12.5, shares=10.0)
        t.id = 1
        row = _serialize_trade_row(t, prior_buys_by_ticker_user={})
        assert row['action'] == 'BUY'
        assert row['realized_pct'] is None
        assert row['hold_days'] is None
        assert row['price'] == 12.5
        assert row['shares'] == 10.0

    def test_sell_realized_pct_uses_cost_total(self):
        """realized_pct = realized_gain / (cost_basis * shares) * 100."""
        from backend.routes.admin import _serialize_trade_row
        t = _make_trade(
            action="SELL", shares=10.0, cost_basis=20.0, realized_gain=50.0,
        )
        t.id = 1
        row = _serialize_trade_row(t, prior_buys_by_ticker_user={})
        # cost_total = 200; gain/cost = 25%
        assert row['realized_pct'] == 25.0

    def test_sell_realized_pct_null_when_cost_basis_missing(self):
        """Legacy SELL rows can lack cost_basis — realized_pct must be null
        rather than crashing or division-by-zeroing."""
        from backend.routes.admin import _serialize_trade_row
        t = _make_trade(action="SELL", shares=10.0, cost_basis=None, realized_gain=50.0)
        t.id = 1
        row = _serialize_trade_row(t, prior_buys_by_ticker_user={})
        assert row['realized_pct'] is None

    def test_sell_realized_pct_null_when_realized_gain_missing(self):
        from backend.routes.admin import _serialize_trade_row
        t = _make_trade(action="SELL", shares=10.0, cost_basis=20.0, realized_gain=None)
        t.id = 1
        row = _serialize_trade_row(t, prior_buys_by_ticker_user={})
        assert row['realized_pct'] is None

    def test_hold_days_uses_persisted_column_when_present(self):
        from backend.routes.admin import _serialize_trade_row
        t = _make_trade(action="SELL", holding_days=42, shares=10.0,
                        cost_basis=20.0, realized_gain=50.0)
        t.id = 1
        row = _serialize_trade_row(t, prior_buys_by_ticker_user={})
        assert row['hold_days'] == 42

    def test_hold_days_falls_back_to_prior_buy_diff(self):
        """If holding_days is null, derive from a prior BUY in the window."""
        from backend.routes.admin import _serialize_trade_row
        buy = _make_trade(
            action="BUY", ticker="ABC", user_id=1,
            executed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        buy.id = 1
        sell = _make_trade(
            action="SELL", ticker="ABC", user_id=1, holding_days=None,
            shares=10.0, cost_basis=20.0, realized_gain=50.0,
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        )
        sell.id = 2
        row = _serialize_trade_row(sell, prior_buys_by_ticker_user={('ABC', 1): buy})
        assert row['hold_days'] == 7

    def test_hold_days_null_when_no_prior_buy(self):
        from backend.routes.admin import _serialize_trade_row
        sell = _make_trade(
            action="SELL", ticker="XYZ", user_id=1, holding_days=None,
            shares=10.0, cost_basis=20.0, realized_gain=50.0,
        )
        sell.id = 1
        row = _serialize_trade_row(sell, prior_buys_by_ticker_user={})
        assert row['hold_days'] is None


# ============================================================================
# _build_cumulative_curve — anchored at 0%, walks SELLs, terminal carry
# ============================================================================
class TestBuildCumulativeCurve:
    def test_empty_trades_anchors_and_terminates_at_zero(self):
        from backend.routes.admin import _build_cumulative_curve
        anchor = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 15, tzinfo=timezone.utc)
        curve = _build_cumulative_curve([], anchor, end, starting_value=25000.0)
        assert curve[0] == {'date': '2026-04-01', 'cum_pct': 0.0}
        assert curve[-1] == {'date': '2026-04-15', 'cum_pct': 0.0}

    def test_curve_accumulates_sells_chronologically(self):
        """Three SELLs at +250, +500, -100 on $25k base → cum % steps to
        1.0, 3.0, 2.6."""
        from backend.routes.admin import _build_cumulative_curve
        anchor = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 15, tzinfo=timezone.utc)
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=250.0,
                        executed_at=datetime(2026, 4, 3, tzinfo=timezone.utc)),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=500.0,
                        executed_at=datetime(2026, 4, 5, tzinfo=timezone.utc)),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=-100.0,
                        executed_at=datetime(2026, 4, 10, tzinfo=timezone.utc)),
        ]
        curve = _build_cumulative_curve(trades, anchor, end, starting_value=25000.0)
        assert len(curve) == 5
        assert curve[0] == {'date': '2026-04-01', 'cum_pct': 0.0}
        assert curve[1] == {'date': '2026-04-03', 'cum_pct': 1.0}
        assert curve[2] == {'date': '2026-04-05', 'cum_pct': 3.0}
        assert curve[3] == {'date': '2026-04-10', 'cum_pct': 2.6}
        # Terminal row carries the last cum_pct forward
        assert curve[4] == {'date': '2026-04-15', 'cum_pct': 2.6}

    def test_curve_ignores_buys_and_pyramids(self):
        """Only SELLs move the curve. BUY/PYRAMID rows should not produce
        new points."""
        from backend.routes.admin import _build_cumulative_curve
        anchor = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 15, tzinfo=timezone.utc)
        trades = [
            _make_trade(action="BUY",
                        executed_at=datetime(2026, 4, 2, tzinfo=timezone.utc)),
            _make_trade(action="PYRAMID",
                        executed_at=datetime(2026, 4, 3, tzinfo=timezone.utc)),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=250.0,
                        executed_at=datetime(2026, 4, 4, tzinfo=timezone.utc)),
        ]
        curve = _build_cumulative_curve(trades, anchor, end, starting_value=25000.0)
        assert len(curve) == 3  # anchor + 1 SELL + terminal
        assert curve[1] == {'date': '2026-04-04', 'cum_pct': 1.0}

    def test_zero_starting_value_returns_flat_zero_line(self):
        """Defensive: division by zero would crash the curve. Must return
        a flat anchor → terminal line at 0%."""
        from backend.routes.admin import _build_cumulative_curve
        anchor = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 15, tzinfo=timezone.utc)
        trades = [_make_trade(action="SELL", realized_gain=100.0,
                              executed_at=datetime(2026, 4, 3, tzinfo=timezone.utc))]
        curve = _build_cumulative_curve(trades, anchor, end, starting_value=0.0)
        assert all(p['cum_pct'] == 0.0 for p in curve)

    def test_same_day_sells_collapse_to_single_row(self):
        """Two SELLs on the same day → one row with the latter cum_pct
        rather than duplicates."""
        from backend.routes.admin import _build_cumulative_curve
        anchor = datetime(2026, 4, 1, tzinfo=timezone.utc)
        end = datetime(2026, 4, 15, tzinfo=timezone.utc)
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=250.0,
                        executed_at=datetime(2026, 4, 5, 9, 0, tzinfo=timezone.utc)),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0,
                        realized_gain=500.0,
                        executed_at=datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)),
        ]
        curve = _build_cumulative_curve(trades, anchor, end, starting_value=25000.0)
        same_day = [p for p in curve if p['date'] == '2026-04-05']
        assert len(same_day) == 1
        assert same_day[0]['cum_pct'] == 3.0


# ============================================================================
# Endpoint validation — same 400/404 errors as the aggregate endpoint
# ============================================================================
class TestEndpointValidation:
    def test_rejects_bad_iso_date(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, cutoff_date="not-a-date")
        assert exc.value.status_code == 400
        assert "valid ISO date" in exc.value.detail

    def test_rejects_future_cutoff(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, cutoff_date=future)
        assert exc.value.status_code == 400
        assert "future" in exc.value.detail

    def test_rejects_post_window_exceeding_days_since_cutoff(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        recent_cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, cutoff_date=recent_cutoff, post_window_days=60)
        assert exc.value.status_code == 400
        assert "exceeds days-since-cutoff" in exc.value.detail

    def test_404_when_no_users_on_strategy(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, strategy="nostate_cs_bear")
        assert exc.value.status_code == 404


# ============================================================================
# Endpoint shape — top-level response contract
# ============================================================================
class TestEndpointShape:
    def test_response_keys(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        result = _call_trades(db_session)
        assert set(result.keys()) == {
            'experiment', 'pre_trades', 'post_trades', 'cumulative_returns',
        }
        assert set(result['cumulative_returns'].keys()) == {'pre', 'post'}
        # Empty windows still emit anchor + terminal rows
        assert len(result['cumulative_returns']['pre']) >= 2
        assert len(result['cumulative_returns']['post']) >= 2

    def test_experiment_block_matches_aggregate_endpoint(self, db_session):
        """The experiment block — strategy, dates, user_ids, etc. — must
        be identical between the two endpoints. Lock it explicitly so a
        future helper change can't drift one without the other."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        trades = _call_trades(db_session)
        agg = _call_aggregate(db_session)
        assert trades['experiment'] == agg['experiment']

    def test_trade_row_shape(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="SELL", ticker="ABC", shares=10.0, cost_basis=10.0,
            realized_gain=20.0, reason="TAKE PROFIT",
            executed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        row = result['pre_trades'][0]
        assert set(row.keys()) >= {
            'id', 'ticker', 'action', 'executed_at', 'shares', 'price',
            'cost_basis', 'realized_gain', 'realized_pct', 'reason',
            'hold_days', 'user_id',
        }
        assert row['ticker'] == 'ABC'
        assert row['action'] == 'SELL'
        assert row['realized_pct'] == 20.0  # 20 / (10*10) * 100
        assert row['reason'] == 'TAKE PROFIT'


# ============================================================================
# Endpoint integration — trade splitting + curves
# ============================================================================
class TestEndpointIntegration:
    def test_pre_post_split_at_cutoff(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        for i, gain in enumerate([20, 15, -5, 30]):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 5 + i*2, tzinfo=timezone.utc),
            ))
        for i, gain in enumerate([15, 20, -3]):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 15 + i*2, tzinfo=timezone.utc),
            ))
        db_session.commit()
        result = _call_trades(db_session)
        assert len(result['pre_trades']) == 4
        assert len(result['post_trades']) == 3

    def test_cumulative_curve_anchors_to_window_starts(self, db_session):
        """Pre curve starts at pre_window.start, post curve starts at
        cutoff. Both anchor at 0%."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0,
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        pre_curve = result['cumulative_returns']['pre']
        post_curve = result['cumulative_returns']['post']
        assert pre_curve[0]['cum_pct'] == 0.0
        assert post_curve[0]['cum_pct'] == 0.0
        assert pre_curve[0]['date'] == result['experiment']['pre_window']['start']
        assert post_curve[0]['date'] == result['experiment']['post_window']['start']

    def test_buy_rows_in_trades_but_not_in_curves(self, db_session):
        """BUY rows appear in the trade list (so the table can show them)
        but do not create points on the curve."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="BUY", ticker="ABC", reason="Standard",
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        assert len(result['post_trades']) == 1
        assert result['post_trades'][0]['action'] == 'BUY'
        # Curve has only anchor + terminal — no SELL rows to move it
        assert len(result['cumulative_returns']['post']) == 2
        assert result['cumulative_returns']['post'][0]['cum_pct'] == 0.0
        assert result['cumulative_returns']['post'][-1]['cum_pct'] == 0.0

    def test_pyramid_filter_parity_with_aggregate(self, db_session):
        """exclude_pyramids must produce the same trade count in both
        endpoints. Drift would mean the dashboard chart shows trades the
        aggregate stats don't, which is misleading."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="BUY", reason="Standard",
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
        trades_excluded = _call_trades(db_session, exclude_pyramids=True)
        agg_excluded = _call_aggregate(db_session, exclude_pyramids=True)
        assert len(trades_excluded['pre_trades']) == agg_excluded['pre']['total_trades']
        trades_included = _call_trades(db_session, exclude_pyramids=False)
        agg_included = _call_aggregate(db_session, exclude_pyramids=False)
        assert len(trades_included['pre_trades']) == agg_included['pre']['total_trades']

    def test_starting_value_matches_aggregate(self, db_session):
        """starting_value_reference echoes the aggregate endpoint and
        total cum_pct matches its total_return_pct (both walk the same
        SELLs / starting_value math)."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        _seed_user_strategy(db_session, user_id=2, strategy="nostate_optimized")
        for i, gain in enumerate([100, 200, -50]):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 16 + i, tzinfo=timezone.utc),
            ))
        db_session.commit()
        trades = _call_trades(db_session)
        agg = _call_aggregate(db_session)
        assert trades['experiment']['starting_value_reference'] == 50000.0
        post_sum = sum(t['realized_gain'] for t in trades['post_trades']
                       if t['action'] == 'SELL' and t['realized_gain'] is not None)
        assert post_sum == agg['post']['total_realized_gain']
        # Final cum_pct in post curve equals total_return_pct (within
        # rounding — curve uses 4 decimals, summary uses 2).
        final_cum = trades['cumulative_returns']['post'][-1]['cum_pct']
        assert abs(final_cum - agg['post']['total_return_pct']) < 0.01

    def test_strategy_filters_users(self, db_session):
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        _seed_user_strategy(db_session, user_id=2, strategy="nostate_cs_bear")
        db_session.add(_make_trade(
            action="BUY", user_id=1, ticker="OPT",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", user_id=2, ticker="BEAR",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        tickers = {t['ticker'] for t in result['pre_trades']}
        assert tickers == {'OPT'}

    def test_hold_days_computed_with_prior_buy_in_pre_window(self, db_session):
        """SELL in post window matches a prior BUY in pre window via the
        cross-window prior_buy_map. Confirms the integration wires the
        per-row helper through correctly."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="BUY", ticker="ABC", user_id=1,
            executed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="SELL", ticker="ABC", user_id=1,
            shares=10.0, cost_basis=10.0, realized_gain=20.0,
            holding_days=None,
            executed_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        sell_row = next(r for r in result['post_trades'] if r['action'] == 'SELL')
        assert sell_row['hold_days'] == 12

    def test_realized_pct_null_for_legacy_sell_without_cost_basis(self, db_session):
        """End-to-end check that a SELL with missing cost_basis surfaces
        realized_pct=null in the response (no crash)."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        db_session.add(_make_trade(
            action="SELL", ticker="LEG", user_id=1,
            shares=10.0, cost_basis=None, realized_gain=50.0,
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        sell_row = result['post_trades'][0]
        assert sell_row['realized_pct'] is None

    def test_empty_window_returns_flat_curves_and_empty_arrays(self, db_session):
        """No trades at all: pre_trades and post_trades are empty arrays;
        both curves are anchor → terminal at 0%."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        result = _call_trades(db_session)
        assert result['pre_trades'] == []
        assert result['post_trades'] == []
        assert result['cumulative_returns']['pre'][0]['cum_pct'] == 0.0
        assert result['cumulative_returns']['pre'][-1]['cum_pct'] == 0.0
        assert result['cumulative_returns']['post'][0]['cum_pct'] == 0.0
        assert result['cumulative_returns']['post'][-1]['cum_pct'] == 0.0
