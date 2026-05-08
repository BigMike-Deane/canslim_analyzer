"""Tests for /api/admin/strategy-ab-eval-trades with source='shadow'.

Sibling of test_strategy_ab_eval_shadow.py for the per-trade endpoint.
Covers:
  - Window math + trade splitting parity with the live path
  - Cumulative-return curve over ShadowTrade rows
  - hold_days fallback uses shadow_strategy_id (no user_id column on
    ShadowTrade) so SELL→BUY pairing stays correct across windows
  - Same 400/404 wording as the live path
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
    ShadowStrategy, ShadowTrade,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.commit()
        db.close()


def _make_shadow_strategy(name="shadow_baseline", parent="nostate_cs_bear",
                          starting_value=25000.0, archived=False):
    archived_at = datetime.now(timezone.utc) if archived else None
    return ShadowStrategy(
        name=name, parent_strategy=parent, config_snapshot={},
        scorer_overrides={}, starting_value=starting_value,
        archived_at=archived_at,
    )


def _make_shadow_trade(strategy_id, action="BUY", reason=None,
                       realized_gain=None, cost_basis=None, shares=10.0,
                       price=10.0, executed_at=None, ticker="TEST",
                       holding_days=None):
    return ShadowTrade(
        shadow_strategy_id=strategy_id,
        ticker=ticker, action=action, shares=shares, price=price,
        total_value=shares * price,
        executed_at=executed_at or datetime.now(timezone.utc),
        cost_basis=cost_basis, realized_gain=realized_gain,
        holding_days=holding_days, reason=reason,
    )


def _call_trades(db, **kwargs):
    from backend.routes.admin import run_strategy_ab_trades
    defaults = dict(
        strategy="shadow_baseline",
        cutoff_date="2026-04-15",
        pre_window_days=14,
        post_window_days=14,
        exclude_pyramids=True,
        source="shadow",
        current_user=None,
        db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(run_strategy_ab_trades(**defaults))


def _call_aggregate(db, **kwargs):
    from backend.routes.admin import run_strategy_ab_comparison
    defaults = dict(
        strategy="shadow_baseline",
        cutoff_date="2026-04-15",
        pre_window_days=14,
        post_window_days=14,
        exclude_pyramids=True,
        min_return_delta_pp=-5.0,
        min_sharpe_delta=0.0,
        min_post_sells=5,
        source="shadow",
        current_user=None,
        db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(run_strategy_ab_comparison(**defaults))


# ============================================================================
# Endpoint validation — parity with live error wording
# ============================================================================
class TestShadowTradesValidation:
    def test_unknown_shadow_404(self, db_session):
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, strategy="never_registered")
        assert exc.value.status_code == 404
        assert "Shadow strategy 'never_registered' not found" in exc.value.detail

    def test_archived_shadow_404(self, db_session):
        ss = _make_shadow_strategy(name="legacy", archived=True)
        db_session.add(ss)
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, strategy="legacy")
        assert exc.value.status_code == 404
        assert "is archived" in exc.value.detail

    def test_bad_iso_date_400(self, db_session):
        db_session.add(_make_shadow_strategy())
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, cutoff_date="not-a-date")
        assert exc.value.status_code == 400
        assert "valid ISO date" in exc.value.detail

    def test_future_cutoff_400(self, db_session):
        db_session.add(_make_shadow_strategy())
        db_session.commit()
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call_trades(db_session, cutoff_date=future)
        assert exc.value.status_code == 400


# ============================================================================
# Endpoint shape — top-level response keys + source surface
# ============================================================================
class TestShadowTradesShape:
    def test_response_keys(self, db_session):
        db_session.add(_make_shadow_strategy())
        db_session.commit()
        result = _call_trades(db_session)
        assert set(result.keys()) == {
            'experiment', 'pre_trades', 'post_trades', 'cumulative_returns',
        }
        assert set(result['cumulative_returns'].keys()) == {'pre', 'post'}

    def test_experiment_block_matches_aggregate_endpoint(self, db_session):
        """Same parity check as the live lockfile — both endpoints agree
        on the experiment metadata for the same shadow strategy."""
        db_session.add(_make_shadow_strategy())
        db_session.commit()
        trades = _call_trades(db_session)
        agg = _call_aggregate(db_session)
        assert trades['experiment'] == agg['experiment']
        assert trades['experiment'].get('source') == 'shadow'

    def test_trade_row_includes_user_id_as_scope(self, db_session):
        """ShadowTrade has no user_id column — _serialize_trade_row falls
        back to shadow_strategy_id. The serialized row must surface that
        scope id under the same 'user_id' key."""
        ss = _make_shadow_strategy()
        db_session.add(ss)
        db_session.commit()
        db_session.refresh(ss)
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", ticker="ABC",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        row = result['pre_trades'][0]
        assert row['ticker'] == 'ABC'
        assert row['user_id'] == ss.id


# ============================================================================
# Endpoint integration — splitting + curves over ShadowTrade rows
# ============================================================================
class TestShadowTradesIntegration:
    def _seed(self, db, **kwargs):
        ss = _make_shadow_strategy(**kwargs)
        db.add(ss)
        db.commit()
        db.refresh(ss)
        return ss

    def test_pre_post_split_at_cutoff(self, db_session):
        ss = self._seed(db_session)
        for i, gain in enumerate([20, 15, -5, 30]):
            db_session.add(_make_shadow_trade(
                ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 5 + i*2, tzinfo=timezone.utc),
            ))
        for i, gain in enumerate([15, 20, -3]):
            db_session.add(_make_shadow_trade(
                ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 15 + i*2, tzinfo=timezone.utc),
            ))
        db_session.commit()
        result = _call_trades(db_session)
        assert len(result['pre_trades']) == 4
        assert len(result['post_trades']) == 3

    def test_cumulative_curve_anchors_to_window_starts(self, db_session):
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0,
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

    def test_buy_in_trades_but_not_in_curves(self, db_session):
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", ticker="ABC", reason="Standard",
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        assert len(result['post_trades']) == 1
        assert result['post_trades'][0]['action'] == 'BUY'
        assert len(result['cumulative_returns']['post']) == 2
        assert result['cumulative_returns']['post'][-1]['cum_pct'] == 0.0

    def test_pyramid_filter_parity_with_aggregate(self, db_session):
        """exclude_pyramids must produce the same trade count in both
        shadow endpoints — same drift-detection check the live lockfile
        runs against AIPortfolioTrade."""
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", reason="Standard",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss.id, action="PYRAMID", reason="PYRAMID: add",
            executed_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", reason="PYRAMID: live trader add",
            executed_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
        ))
        db_session.commit()
        trades_excluded = _call_trades(db_session, exclude_pyramids=True)
        agg_excluded = _call_aggregate(db_session, exclude_pyramids=True)
        assert len(trades_excluded['pre_trades']) == agg_excluded['pre']['total_trades']
        trades_included = _call_trades(db_session, exclude_pyramids=False)
        agg_included = _call_aggregate(db_session, exclude_pyramids=False)
        assert len(trades_included['pre_trades']) == agg_included['pre']['total_trades']

    def test_starting_value_from_shadow_record(self, db_session):
        """Curve denominator + starting_value_reference both come from
        ShadowStrategy.starting_value, not from any user config."""
        ss = self._seed(db_session, starting_value=42000.0)
        for i, gain in enumerate([100, 200, -50]):
            db_session.add(_make_shadow_trade(
                ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 16 + i, tzinfo=timezone.utc),
            ))
        db_session.commit()
        trades = _call_trades(db_session)
        agg = _call_aggregate(db_session)
        assert trades['experiment']['starting_value_reference'] == 42000.0
        post_sum = sum(
            t['realized_gain'] for t in trades['post_trades']
            if t['action'] == 'SELL' and t['realized_gain'] is not None
        )
        assert post_sum == agg['post']['total_realized_gain']
        # Final cum_pct in post curve equals total_return_pct (within rounding)
        final_cum = trades['cumulative_returns']['post'][-1]['cum_pct']
        assert abs(final_cum - agg['post']['total_return_pct']) < 0.01

    def test_hold_days_falls_back_via_shadow_strategy_id_scope(self, db_session):
        """SELL with holding_days=None should pair with a prior BUY using
        (ticker, shadow_strategy_id) as the scope key — proving the
        _scope_id helper handles ShadowTrade correctly."""
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", ticker="ABC",
            executed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss.id, action="SELL", ticker="ABC",
            shares=10.0, cost_basis=10.0, realized_gain=20.0,
            holding_days=None,
            executed_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call_trades(db_session)
        sell_row = next(r for r in result['post_trades'] if r['action'] == 'SELL')
        assert sell_row['hold_days'] == 12

    def test_two_shadow_strategies_dont_pair_buys(self, db_session):
        """A BUY in shadow_a must NOT pair as the prior buy for a SELL in
        shadow_b — scoping must isolate per shadow_strategy_id."""
        # Each shadow_strategy is its own endpoint call; we cannot mix
        # them in a single call. But we can confirm shadow_a's BUY does not
        # influence shadow_b's hold_days math by querying shadow_b alone.
        ss_a = self._seed(db_session, name="shadow_a")
        ss_b = self._seed(db_session, name="shadow_b")
        db_session.add(_make_shadow_trade(
            ss_a.id, action="BUY", ticker="ABC",
            executed_at=datetime(2026, 4, 5, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss_b.id, action="SELL", ticker="ABC",
            shares=10.0, cost_basis=10.0, realized_gain=20.0,
            holding_days=None,
            executed_at=datetime(2026, 4, 17, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result_b = _call_trades(db_session, strategy="shadow_b")
        sell_row = next(r for r in result_b['post_trades'] if r['action'] == 'SELL')
        # No prior BUY in shadow_b's window → hold_days unresolved
        assert sell_row['hold_days'] is None

    def test_empty_window_returns_flat_curves(self, db_session):
        """Newly-registered shadow with no trades — endpoint returns clean
        empty arrays + flat anchor→terminal curves."""
        self._seed(db_session)
        result = _call_trades(db_session)
        assert result['pre_trades'] == []
        assert result['post_trades'] == []
        assert result['cumulative_returns']['pre'][0]['cum_pct'] == 0.0
        assert result['cumulative_returns']['pre'][-1]['cum_pct'] == 0.0
        assert result['cumulative_returns']['post'][0]['cum_pct'] == 0.0
        assert result['cumulative_returns']['post'][-1]['cum_pct'] == 0.0
