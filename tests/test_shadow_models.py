"""Migration + schema tests for the shadow paper-trading tables.

The shadow stack lets alternative scoring rules run in parallel with the
live scanner. These tests pin the table schema so a deploy can't silently
drop a column the scoring/eval code depends on. See
docs/shadow-paper-trading-design.md for context.

Pattern matches tests/test_breakout_monitor_dedup.py's
TestBreakoutAlertTableMigration — Base.metadata.create_all in memory,
inspect the resulting schema.
"""
import os

os.environ.setdefault("CANSLIM_ENV", "test")

import pytest
from sqlalchemy import create_engine, inspect

from backend.database import Base, ShadowStrategy, ShadowTrade


# ── Schema / migration ────────────────────────────────────────────────────────


class TestShadowStrategyTable:
    """shadow_strategies must exist with the columns the eval/scan code reads.

    init_db() iterates Base.metadata.sorted_tables with checkfirst=True, so
    the test mirrors that path.
    """

    def test_table_present_after_create_all(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        tables = set(inspect(engine).get_table_names())
        assert "shadow_strategies" in tables

    def test_columns_match_design(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        cols = {c["name"] for c in inspect(engine).get_columns("shadow_strategies")}
        # Locked subset — these are read by run_shadow_strategies + the
        # _build_trade_query helper in /admin/strategy-ab-eval.
        assert {"id", "name", "parent_strategy", "config_snapshot",
                "scorer_overrides", "description", "starting_value",
                "created_at", "activated_at", "archived_at"} <= cols

    def test_name_is_unique(self):
        """Two shadow stacks with the same name would corrupt the eval read.
        The unique constraint is the load-bearing guard."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            db.add(ShadowStrategy(
                name="approach_3",
                parent_strategy="nostate_cs_bear",
                config_snapshot={"min_score": 72},
                scorer_overrides={},
                starting_value=25000.0,
            ))
            db.commit()
            db.add(ShadowStrategy(
                name="approach_3",
                parent_strategy="nostate_cs_bear",
                config_snapshot={"min_score": 70},
                scorer_overrides={},
                starting_value=25000.0,
            ))
            with pytest.raises(Exception):
                db.commit()
        finally:
            db.close()


class TestShadowTradeTable:
    """shadow_trades mirrors AIPortfolioTrade so _summarize_window /
    _decide work over either source via the same field names."""

    def test_table_present_after_create_all(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        tables = set(inspect(engine).get_table_names())
        assert "shadow_trades" in tables

    def test_columns_match_ai_trade_shape(self):
        """Locked subset — the eval helpers read these by name."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        cols = {c["name"] for c in inspect(engine).get_columns("shadow_trades")}
        assert {"id", "shadow_strategy_id", "ticker", "action", "shares",
                "price", "total_value", "reason", "canslim_score",
                "cost_basis", "realized_gain", "holding_days",
                "signal_factors", "executed_at"} <= cols

    def test_no_user_id_column(self):
        """Shadow trades belong to a strategy, not a user. If a future
        change adds user_id, the eval read needs to update its filter."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        cols = {c["name"] for c in inspect(engine).get_columns("shadow_trades")}
        assert "user_id" not in cols

    def test_strategy_executed_index_present(self):
        """Hot-path index for the _resolve_ab_window time-range query."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        index_names = {i["name"] for i in inspect(engine).get_indexes("shadow_trades")}
        assert "ix_shadow_trades_strategy_executed" in index_names

    def test_foreign_key_to_shadow_strategy(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        fks = inspect(engine).get_foreign_keys("shadow_trades")
        assert any(
            fk["referred_table"] == "shadow_strategies"
            and "shadow_strategy_id" in fk["constrained_columns"]
            for fk in fks
        )


class TestCreateAllIdempotency:
    """Calling create_all twice mirrors init_db's checkfirst=True pattern.
    Required because init_db is called on every container start."""

    def test_create_all_is_idempotent(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Base.metadata.create_all(bind=engine)  # must not raise
        tables = set(inspect(engine).get_table_names())
        assert "shadow_strategies" in tables
        assert "shadow_trades" in tables


# ── Round-trip persistence ────────────────────────────────────────────────────


class TestShadowStrategyPersistence:
    def test_round_trip_with_json_columns(self):
        """config_snapshot + scorer_overrides + signal_factors are JSON.
        Pin that they survive insert/select unchanged — past JSON-column
        bugs (BacktestRun.profile_overrides stored as TEXT in PG) burned us."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            strat = ShadowStrategy(
                name="approach_3_test",
                parent_strategy="nostate_cs_bear",
                config_snapshot={"min_score": 72, "max_positions": 8,
                                 "trailing_stops": {"gain_50_plus": 25}},
                scorer_overrides={"sector_alias_map_enabled": True,
                                  "turnaround_roe_signal": True},
                description="Sector aliases experiment",
                starting_value=25000.0,
            )
            db.add(strat)
            db.commit()

            row = db.query(ShadowStrategy).filter_by(name="approach_3_test").first()
            assert row is not None
            assert row.config_snapshot["min_score"] == 72
            assert row.config_snapshot["trailing_stops"]["gain_50_plus"] == 25
            assert row.scorer_overrides["sector_alias_map_enabled"] is True
            assert row.starting_value == 25000.0
            assert row.archived_at is None
        finally:
            db.close()


class TestShadowTradePersistence:
    def test_round_trip_with_signal_factors(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            strat = ShadowStrategy(
                name="trade_test",
                parent_strategy="nostate_cs_bear",
                config_snapshot={},
                scorer_overrides={},
                starting_value=25000.0,
            )
            db.add(strat)
            db.commit()

            t = ShadowTrade(
                shadow_strategy_id=strat.id,
                ticker="AAPL",
                action="BUY",
                shares=10.0,
                price=150.0,
                total_value=1500.0,
                reason="entry",
                canslim_score=82.0,
                signal_factors={"entry_type": "pre-breakout",
                                "market_regime": "bullish",
                                "soft_zone": False},
            )
            db.add(t)
            db.commit()

            row = db.query(ShadowTrade).first()
            assert row.action == "BUY"
            assert row.signal_factors["entry_type"] == "pre-breakout"
            assert row.signal_factors["soft_zone"] is False
            assert row.shadow_strategy_id == strat.id
        finally:
            db.close()
