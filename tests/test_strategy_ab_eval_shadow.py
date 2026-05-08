"""Tests for /api/admin/strategy-ab-eval with source='shadow'.

Step 3 of the shadow paper-trading harness (design in
docs/shadow-paper-trading-design.md). The aggregate A/B endpoint accepts
a source query param that routes the trade lookup through ShadowTrade
when source='shadow'. Same window math, same summarizer, same decision
rule — only the trade-source changes.

Also covers the GET /api/admin/shadow-strategies listing route.
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
                          starting_value=25000.0, archived=False, description=None):
    archived_at = datetime.now(timezone.utc) if archived else None
    return ShadowStrategy(
        name=name,
        parent_strategy=parent,
        config_snapshot={},
        scorer_overrides={},
        starting_value=starting_value,
        description=description,
        archived_at=archived_at,
    )


def _make_shadow_trade(strategy_id, action="BUY", reason=None,
                       realized_gain=None, cost_basis=None, shares=10.0,
                       price=10.0, executed_at=None, ticker="TEST"):
    return ShadowTrade(
        shadow_strategy_id=strategy_id,
        ticker=ticker,
        action=action,
        shares=shares,
        price=price,
        total_value=shares * price,
        executed_at=executed_at or datetime.now(timezone.utc),
        cost_basis=cost_basis,
        realized_gain=realized_gain,
        reason=reason,
    )


def _call(db, **kwargs):
    """Direct call into run_strategy_ab_comparison with source='shadow'."""
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


def _call_listing(db, archived=False):
    from backend.routes.admin import list_shadow_strategies
    return asyncio.run(list_shadow_strategies(
        archived=archived, current_user=None, db=db,
    ))


# ============================================================================
# _build_trade_query — branch dispatch
# ============================================================================
class TestBuildTradeQueryShadow:
    def test_unknown_shadow_name_404(self, db_session):
        from backend.routes.admin import _build_trade_query
        with pytest.raises(HTTPException) as exc:
            _build_trade_query(db_session, "shadow", "does_not_exist")
        assert exc.value.status_code == 404
        assert "Shadow strategy 'does_not_exist' not found" in exc.value.detail

    def test_archived_shadow_404(self, db_session):
        from backend.routes.admin import _build_trade_query
        ss = _make_shadow_strategy(name="legacy_stack", archived=True)
        db_session.add(ss)
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _build_trade_query(db_session, "shadow", "legacy_stack")
        assert exc.value.status_code == 404
        assert "is archived" in exc.value.detail

    def test_active_shadow_returns_query(self, db_session):
        from backend.routes.admin import _build_trade_query
        ss = _make_shadow_strategy(starting_value=42000.0)
        db_session.add(ss)
        db_session.commit()
        db_session.refresh(ss)
        base_q, sv, ids = _build_trade_query(db_session, "shadow", ss.name)
        assert sv == 42000.0
        assert ids == [ss.id]
        # Base query should be empty until trades are seeded
        assert base_q.count() == 0

    def test_live_branch_unchanged_by_shadow_addition(self, db_session):
        """Adding source='shadow' must not regress source='live'."""
        from backend.routes.admin import _build_trade_query
        cfg = AIPortfolioConfig(
            user_id=1, strategy="nostate_optimized", is_active=True,
            starting_cash=25000.0, current_cash=25000.0, max_positions=8,
            max_position_pct=12.0, min_score_to_buy=72,
            sell_score_threshold=45, stop_loss_pct=7.0, take_profit_pct=75.0,
        )
        db_session.add(cfg)
        db_session.commit()
        base_q, sv, ids = _build_trade_query(db_session, "live", "nostate_optimized")
        assert sv == 25000.0
        assert ids == [1]


# ============================================================================
# Endpoint integration — source='shadow' happy paths
# ============================================================================
class TestShadowEndpointHappyPath:
    def _seed(self, db, **kwargs):
        ss = _make_shadow_strategy(**kwargs)
        db.add(ss)
        db.commit()
        db.refresh(ss)
        return ss

    def test_pre_post_split_at_cutoff(self, db_session):
        ss = self._seed(db_session)
        # 4 pre-cutoff SELLs (Apr 5, 7, 9, 11)
        for i, gain in enumerate([20, 15, -5, 30]):
            db_session.add(_make_shadow_trade(
                ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 5 + i*2, tzinfo=timezone.utc),
            ))
        # 3 post-cutoff SELLs (Apr 15, 17, 19)
        for i, gain in enumerate([15, 20, -3]):
            db_session.add(_make_shadow_trade(
                ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=float(gain),
                executed_at=datetime(2026, 4, 15 + i*2, tzinfo=timezone.utc),
            ))
        db_session.commit()

        result = _call(db_session)
        assert result['pre']['realized_sell_pct']['n'] == 4
        assert result['post']['realized_sell_pct']['n'] == 3

    def test_response_shape_matches_live(self, db_session):
        """Same top-level keys as the live endpoint — frontend can render
        identical."""
        self._seed(db_session)
        result = _call(db_session)
        assert set(result.keys()) == {
            'experiment', 'summary', 'pre', 'post', 'delta', 'warnings',
        }
        assert 'decision' in result['summary']
        assert isinstance(result['warnings'], list)

    def test_experiment_block_includes_source(self, db_session):
        """The experiment block surfaces source='shadow' so the frontend
        can label the comparison clearly."""
        self._seed(db_session)
        result = _call(db_session)
        assert result['experiment'].get('source') == 'shadow'
        assert result['experiment']['strategy'] == 'shadow_baseline'

    def test_starting_value_from_shadow_strategy(self, db_session):
        """starting_value comes from ShadowStrategy.starting_value, not from
        AIPortfolioConfig.starting_cash like the live path."""
        self._seed(db_session, starting_value=42000.0)
        result = _call(db_session)
        assert result['experiment']['starting_value_reference'] == 42000.0
        assert result['pre']['starting_value'] == 42000.0
        assert result['post']['starting_value'] == 42000.0

    def test_pyramid_filter_excludes_shadow_pyramids(self, db_session):
        """exclude_pyramids must filter ShadowTrade rows the same way it
        filters AIPortfolioTrade rows — by action and reason LIKE."""
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", reason="Standard breakout",
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
        result = _call(db_session, exclude_pyramids=True)
        assert result['pre']['total_trades'] == 1

    def test_pyramid_inclusion_toggle_for_shadow(self, db_session):
        ss = self._seed(db_session)
        db_session.add(_make_shadow_trade(
            ss.id, action="PYRAMID", reason="PYRAMID: add",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss.id, action="BUY", reason="Standard",
            executed_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result = _call(db_session, exclude_pyramids=False)
        assert result['pre']['total_trades'] == 2

    def test_isolates_per_shadow_strategy(self, db_session):
        """Two shadow strategies should isolate their trades — querying one
        must not return rows from the other."""
        ss1 = self._seed(db_session, name="shadow_a")
        ss2 = self._seed(db_session, name="shadow_b")
        db_session.add(_make_shadow_trade(
            ss1.id, action="BUY", ticker="ALPHA",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.add(_make_shadow_trade(
            ss2.id, action="BUY", ticker="BETA",
            executed_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
        ))
        db_session.commit()
        result_a = _call(db_session, strategy="shadow_a")
        result_b = _call(db_session, strategy="shadow_b")
        assert result_a['pre']['total_trades'] == 1
        assert result_b['pre']['total_trades'] == 1

    def test_pre_window_zero_when_registered_after_cutoff(self, db_session):
        """A shadow strategy registered today comparing against a
        weeks-old cutoff has zero pre-window trades — the endpoint must
        return cleanly with insufficient_data, not error."""
        self._seed(db_session)
        # No trades seeded — straddles registration vs cutoff
        result = _call(db_session)
        assert result['pre']['total_trades'] == 0
        assert result['post']['total_trades'] == 0
        assert result['summary']['decision'] == 'insufficient_data'


# ============================================================================
# Endpoint integration — source='shadow' error paths
# ============================================================================
class TestShadowEndpointValidation:
    def test_unknown_shadow_strategy_404(self, db_session):
        with pytest.raises(HTTPException) as exc:
            _call(db_session, strategy="never_registered")
        assert exc.value.status_code == 404
        assert "Shadow strategy 'never_registered' not found" in exc.value.detail

    def test_archived_shadow_strategy_404(self, db_session):
        ss = _make_shadow_strategy(name="legacy_stack", archived=True)
        db_session.add(ss)
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _call(db_session, strategy="legacy_stack")
        assert exc.value.status_code == 404
        assert "is archived" in exc.value.detail

    def test_bad_iso_date_still_400(self, db_session):
        """Date validation runs before source dispatch — same 400 wording
        as the live path."""
        ss = _make_shadow_strategy()
        db_session.add(ss)
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date="not-a-date")
        assert exc.value.status_code == 400
        assert "valid ISO date" in exc.value.detail

    def test_future_cutoff_still_400(self, db_session):
        ss = _make_shadow_strategy()
        db_session.add(ss)
        db_session.commit()
        future = (datetime.now(timezone.utc) + timedelta(days=10)).date().isoformat()
        with pytest.raises(HTTPException) as exc:
            _call(db_session, cutoff_date=future)
        assert exc.value.status_code == 400


# ============================================================================
# /api/admin/shadow-strategies listing route
# ============================================================================
class TestShadowStrategiesListing:
    def test_empty_listing(self, db_session):
        rows = _call_listing(db_session)
        assert rows == []

    def test_active_only_by_default(self, db_session):
        db_session.add(_make_shadow_strategy(name="alive_a"))
        db_session.add(_make_shadow_strategy(name="alive_b"))
        db_session.add(_make_shadow_strategy(name="legacy", archived=True))
        db_session.commit()
        rows = _call_listing(db_session, archived=False)
        names = {r['name'] for r in rows}
        assert names == {"alive_a", "alive_b"}

    def test_archived_true_includes_all(self, db_session):
        db_session.add(_make_shadow_strategy(name="alive"))
        db_session.add(_make_shadow_strategy(name="legacy", archived=True))
        db_session.commit()
        rows = _call_listing(db_session, archived=True)
        names = {r['name'] for r in rows}
        assert names == {"alive", "legacy"}

    def test_row_shape(self, db_session):
        ss = _make_shadow_strategy(
            name="shadow_baseline",
            parent="nostate_cs_bear",
            starting_value=25000.0,
            description="Forward-only baseline parity check",
        )
        db_session.add(ss)
        db_session.commit()
        rows = _call_listing(db_session)
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) >= {
            'id', 'name', 'parent_strategy', 'scorer_overrides',
            'starting_value', 'description', 'created_at',
            'activated_at', 'archived_at',
        }
        assert row['name'] == 'shadow_baseline'
        assert row['parent_strategy'] == 'nostate_cs_bear'
        assert row['starting_value'] == 25000.0
        assert row['description'] == 'Forward-only baseline parity check'
        assert row['archived_at'] is None

    def test_listing_ordered_by_id(self, db_session):
        """Stable order so the frontend dropdown doesn't flicker on
        re-render."""
        for name in ["zeta", "alpha", "mu"]:
            db_session.add(_make_shadow_strategy(name=name))
        db_session.commit()
        rows = _call_listing(db_session)
        ids = [r['id'] for r in rows]
        assert ids == sorted(ids)

    def test_archived_at_iso_format_when_present(self, db_session):
        ss = _make_shadow_strategy(name="legacy", archived=True)
        db_session.add(ss)
        db_session.commit()
        rows = _call_listing(db_session, archived=True)
        legacy = next(r for r in rows if r['name'] == 'legacy')
        assert legacy['archived_at'] is not None
        # Parseable ISO
        datetime.fromisoformat(legacy['archived_at'])
