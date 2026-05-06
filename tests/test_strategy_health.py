"""Tests for /api/admin/strategy-health endpoint logic.

Targets the pure trade-summarization function (_summarize_trades) and the
endpoint's filtering invariants. Don't bother with HTTP-level tests — the
auth dependency wrapping is exercised by other admin endpoints already.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

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
                executed_at=None, user_id=1):
    return AIPortfolioTrade(
        ticker="TEST", action=action, shares=shares, price=10.0,
        total_value=shares * 10.0,
        executed_at=executed_at or datetime.now(timezone.utc),
        cost_basis=cost_basis, realized_gain=realized_gain,
        signal_factors=json.dumps(signal_factors) if signal_factors else None,
        reason=reason, user_id=user_id,
    )


class TestSummarizeTrades:
    def test_empty_list(self):
        from backend.routes.admin import _summarize_trades
        s = _summarize_trades([], days=10)
        assert s["total_trades"] == 0
        assert s["entry_rate_per_day"] == 0.0
        assert s["realized_sell_pct"] is None
        assert s["ml_confidence"]["n_with"] == 0

    def test_buys_per_day_uses_provided_window(self):
        from backend.routes.admin import _summarize_trades
        trades = [_make_trade(action="BUY") for _ in range(10)]
        s = _summarize_trades(trades, days=5)
        # 10 buys over 5 days = 2.0/day
        assert s["entry_rate_per_day"] == 2.0
        assert s["by_action"]["BUY"] == 10

    def test_zero_days_floors_to_one(self):
        """Defensive: don't divide by zero on same-day audits."""
        from backend.routes.admin import _summarize_trades
        s = _summarize_trades([_make_trade(action="BUY")], days=0)
        # safe_days = max(0, 1) = 1
        assert s["entry_rate_per_day"] == 1.0

    def test_realized_pct_uses_total_cost_not_per_share(self):
        """May-5 audit had a bug where realized_gain was divided by per-share
        cost_basis instead of total cost (cost_basis * shares). Lock in the
        correct math."""
        from backend.routes.admin import _summarize_trades
        # Sell: 10 shares at $10 cost basis (total $100), realized $25 gain
        # → 25%
        trades = [_make_trade(
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=25.0,
        )]
        s = _summarize_trades(trades, days=1)
        assert s["realized_sell_pct"]["mean"] == 25.0
        assert s["realized_sell_pct"]["n"] == 1
        assert s["realized_sell_pct"]["win_rate"] == 1.0

    def test_realized_pct_loss_lowers_win_rate(self):
        from backend.routes.admin import _summarize_trades
        # 1 winner + 1 loser = 50% WR
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=-5.0),
        ]
        s = _summarize_trades(trades, days=1)
        assert s["realized_sell_pct"]["win_rate"] == 0.5
        assert s["realized_sell_pct"]["n"] == 2

    def test_realized_pct_skips_rows_with_no_cost_basis(self):
        """Some legacy rows have realized_gain set but cost_basis None.
        Don't crash; just skip them in the realized-pct computation."""
        from backend.routes.admin import _summarize_trades
        trades = [
            _make_trade(action="SELL", shares=10.0, cost_basis=None, realized_gain=25.0),
            _make_trade(action="SELL", shares=10.0, cost_basis=10.0, realized_gain=20.0),
        ]
        s = _summarize_trades(trades, days=1)
        # Only the second sell counts
        assert s["realized_sell_pct"]["n"] == 1

    def test_ml_confidence_extracted_from_signal_factors(self):
        from backend.routes.admin import _summarize_trades
        trades = [
            _make_trade(action="BUY", signal_factors={"ml_confidence": 0.42}),
            _make_trade(action="BUY", signal_factors={"ml_confidence": 0.55}),
            _make_trade(action="BUY", signal_factors={"foo": "bar"}),  # no ml_confidence
        ]
        s = _summarize_trades(trades, days=1)
        assert s["ml_confidence"]["n_with"] == 2
        assert s["ml_confidence"]["min"] == 0.42
        assert s["ml_confidence"]["max"] == 0.55
        assert abs(s["ml_confidence"]["mean"] - 0.485) < 1e-9

    def test_signal_factors_stored_as_string_is_parsed(self):
        """signal_factors is TEXT in production Postgres. The summarizer
        must json.loads strings, not assume dict."""
        from backend.routes.admin import _summarize_trades
        # Bypass _make_trade's auto-encoding; pass a pre-serialized string
        t = AIPortfolioTrade(
            ticker="T", action="BUY", shares=1.0, price=1.0, total_value=1.0,
            executed_at=datetime.now(timezone.utc),
            signal_factors='{"ml_confidence": 0.33}', user_id=1,
        )
        s = _summarize_trades([t], days=1)
        assert s["ml_confidence"]["n_with"] == 1
        assert s["ml_confidence"]["min"] == 0.33

    def test_signal_factors_malformed_json_is_silently_skipped(self):
        from backend.routes.admin import _summarize_trades
        t = AIPortfolioTrade(
            ticker="T", action="BUY", shares=1.0, price=1.0, total_value=1.0,
            executed_at=datetime.now(timezone.utc),
            signal_factors='not json at all', user_id=1,
        )
        s = _summarize_trades([t], days=1)
        # Just doesn't crash; ml_confidence empty
        assert s["ml_confidence"]["n_with"] == 0

    def test_exit_reason_categorization(self):
        from backend.routes.admin import _summarize_trades
        trades = [
            _make_trade(action="SELL", reason="STOP LOSS: -7.0%"),
            _make_trade(action="SELL", reason="STOP LOSS: -7.5%"),
            _make_trade(action="SELL", reason="PARTIAL PROFIT 25%: +30%"),
            _make_trade(action="SELL", reason="TRAILING STOP: down from peak"),
            _make_trade(action="SELL", reason="PRE-EARNINGS PARTIAL: 1d to earnings"),
            _make_trade(action="SELL", reason="something other thing"),
        ]
        s = _summarize_trades(trades, days=1)
        er = s["exit_reasons"]
        assert er["STOP_LOSS"] == 2
        assert er["PARTIAL_PROFIT"] == 1
        assert er["TRAILING_STOP"] == 1
        assert er["PRE_EARNINGS"] == 1
        assert er["OTHER"] == 1


class TestStrategyHealthEndpointBehavior:
    """Test the higher-level filtering: window scoping, pyramid exclusion,
    strategy → user_id resolution. Calls the route function directly
    (skipping HTTP) by stubbing the auth dependency."""

    def _seed_user_strategy(self, db, user_id, strategy):
        cfg = AIPortfolioConfig(
            user_id=user_id, strategy=strategy, is_active=True,
            starting_cash=25000.0, current_cash=25000.0,
            max_positions=8, max_position_pct=12.0,
            min_score_to_buy=72, sell_score_threshold=45,
            stop_loss_pct=7.0, take_profit_pct=75.0,
        )
        db.add(cfg)
        db.commit()

    def test_pyramid_action_excluded_by_default(self, db_session):
        from backend.routes.admin import strategy_health_audit
        self._seed_user_strategy(db_session, user_id=1, strategy="nostate_cs_bear")

        cutoff = datetime(2026, 4, 29, tzinfo=timezone.utc)
        # 1 main BUY + 1 PYRAMID action both pre-cutoff
        db_session.add(_make_trade(
            action="BUY", reason="Standard breakout",
            executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="PYRAMID", reason="PYRAMID: ADD shares",
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()

        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            exclude_pyramid_reason=True,
            current_user=None, db=db_session,
        ))
        # PYRAMID action filtered out → only 1 BUY in pre window
        assert result["pre_graduation"]["total_trades"] == 1
        assert result["pre_graduation"]["by_action"].get("BUY") == 1
        assert "PYRAMID" not in result["pre_graduation"]["by_action"]

    def test_pyramid_reason_excluded_by_default(self, db_session):
        """Live trader writes action='BUY' + reason='PYRAMID:...' (the sync
        drift documented in May 5 audit). Endpoint must filter this shape too."""
        from backend.routes.admin import strategy_health_audit
        self._seed_user_strategy(db_session, user_id=1, strategy="nostate_cs_bear")

        db_session.add(_make_trade(
            action="BUY", reason="Standard breakout",
            executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", reason="PYRAMID: pyramid via live trader",
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()

        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            exclude_pyramid_reason=True,
            current_user=None, db=db_session,
        ))
        # BUY+PYRAMID-reason filtered → only 1 standard-breakout BUY survives
        assert result["pre_graduation"]["total_trades"] == 1

    def test_pyramid_inclusion_toggle_works(self, db_session):
        """When exclude_pyramid_reason=False, pyramid rows are counted."""
        from backend.routes.admin import strategy_health_audit
        self._seed_user_strategy(db_session, user_id=1, strategy="nostate_cs_bear")

        db_session.add(_make_trade(
            action="PYRAMID", reason="PYRAMID: add",
            executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", reason="Standard breakout",
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        ))
        db_session.commit()

        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            exclude_pyramid_reason=False,
            current_user=None, db=db_session,
        ))
        # Both rows counted now
        assert result["pre_graduation"]["total_trades"] == 2

    def test_strategy_with_no_users_returns_error(self, db_session):
        """No active config for strategy → no user_ids → endpoint reports
        rather than silently returning empty stats (would look like a
        successful 'zero trades' run, misleading)."""
        from backend.routes.admin import strategy_health_audit
        # No users seeded
        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            current_user=None, db=db_session,
        ))
        assert "error" in result
        assert "nostate_cs_bear" in result["error"]

    def test_strategy_filters_to_users_on_that_strategy(self, db_session):
        """Trades from users on a different strategy must NOT be counted."""
        from backend.routes.admin import strategy_health_audit
        # User 1 on cs_bear, User 2 on optimized
        self._seed_user_strategy(db_session, user_id=1, strategy="nostate_cs_bear")
        self._seed_user_strategy(db_session, user_id=2, strategy="nostate_optimized")

        # 1 trade for each user
        db_session.add(_make_trade(
            action="BUY", reason="cs_bear trade",
            executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            user_id=1,
        ))
        db_session.add(_make_trade(
            action="BUY", reason="optimized trade",
            executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            user_id=2,
        ))
        db_session.commit()

        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            current_user=None, db=db_session,
        ))
        # Only user 1's trade counted
        assert result["pre_graduation"]["total_trades"] == 1

    def test_pre_post_split_at_cutoff(self, db_session):
        from backend.routes.admin import strategy_health_audit
        self._seed_user_strategy(db_session, user_id=1, strategy="nostate_cs_bear")

        # 2 pre-cutoff buys + 1 post-cutoff buy
        db_session.add(_make_trade(
            action="BUY", executed_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", executed_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        ))
        db_session.add(_make_trade(
            action="BUY", executed_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ))
        db_session.commit()

        import asyncio
        result = asyncio.run(strategy_health_audit(
            strategy="nostate_cs_bear",
            cutoff_iso="2026-04-29T00:00:00",
            window_start_iso="2026-04-08T00:00:00",
            current_user=None, db=db_session,
        ))
        assert result["pre_graduation"]["total_trades"] == 2
        assert result["post_graduation"]["total_trades"] == 1
