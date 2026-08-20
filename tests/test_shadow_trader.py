"""Tests for backend/shadow_trader.py — Step 2 of shadow paper-trading.

Covers:
  - No-op when no active ShadowStrategy rows.
  - Archived stacks are skipped (archived_at IS NOT NULL).
  - FIFO position re-derivation from ShadowTrade log (BUY/SELL pairing,
    weighted-cost-basis consolidation).
  - ShadowSession.query intercepts AI-portfolio models, passes through others.
  - ShadowSession.add intercepts AIPortfolioTrade as ShadowTrade, captures
    AIPortfolioPosition into synthetic state, no-ops for SystemState/MLPrediction
    side effects (sandboxed and rolled back).
  - emit_shadow_buy / emit_shadow_sell update synthetic state and queue
    ShadowTrade rows.
  - run_shadow_strategies orchestrator: try/except boundary swallows
    per-strategy failures, persists pending shadow trades on a separate
    session, sets activated_at on first trade.
  - Scoring-swap correctness: two stacks with different live-evaluator
    behavior produce different shadow trades.
  - Parity test: when sandbox state mirrors live state exactly, the same
    decision yields a ShadowTrade row that mirrors the AIPortfolioTrade row.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db,
    SessionLocal,
    AIPortfolioConfig,
    AIPortfolioPosition,
    AIPortfolioTrade,
    AIPortfolioSnapshot,
    ShadowStrategy,
    ShadowTrade,
    Stock,
    StockScore,
)
from backend.shadow_trader import (
    SHADOW_USER_ID,
    ShadowSession,
    _ShadowQuery,
    _SyntheticConfig,
    _SyntheticPosition,
    _run_one_strategy,
    run_shadow_strategies,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


from backend.database import ShadowPositionPeak


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    # Clean shadow tables — they accumulate across tests if not cleared.
    db.query(ShadowPositionPeak).delete()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.query(AIPortfolioPosition).delete()
    db.query(AIPortfolioSnapshot).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowPositionPeak).delete()
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.query(AIPortfolioPosition).delete()
        db.query(AIPortfolioSnapshot).delete()
        db.commit()
        db.close()


def _make_strategy(db, name="shadow_baseline", parent="nostate_cs_bear",
                   archived=False, scorer_overrides=None, starting_value=25000.0):
    s = ShadowStrategy(
        name=name,
        parent_strategy=parent,
        config_snapshot={"strategy": parent},
        scorer_overrides=scorer_overrides or {},
        starting_value=starting_value,
        archived_at=datetime.now(timezone.utc) if archived else None,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add_shadow_trade(db, strategy_id, ticker, action, shares, price,
                     executed_at=None, **kwargs):
    st = ShadowTrade(
        shadow_strategy_id=strategy_id,
        ticker=ticker,
        action=action,
        shares=shares,
        price=price,
        total_value=shares * price,
        reason=kwargs.get("reason"),
        canslim_score=kwargs.get("canslim_score"),
        cost_basis=kwargs.get("cost_basis"),
        realized_gain=kwargs.get("realized_gain"),
        signal_factors=kwargs.get("signal_factors"),
        holding_days=kwargs.get("holding_days"),
        executed_at=executed_at or datetime.now(timezone.utc),
    )
    db.add(st)
    db.commit()
    db.refresh(st)
    return st


# ── _ShadowQuery: query proxy contract ────────────────────────────────────────


class TestShadowQuery:
    def test_filter_chain_is_pass_through_for_synthetic_data(self):
        items = [object(), object(), object()]
        q = _ShadowQuery(items)
        assert q.filter(True).filter_by(x=1).order_by(None).all() == items

    def test_first_returns_none_for_empty(self):
        q = _ShadowQuery([])
        assert q.first() is None
        assert q.scalar() is None

    def test_count_matches_list_size(self):
        q = _ShadowQuery([1, 2, 3])
        assert q.count() == 3

    def test_limit_returns_truncated_query(self):
        q = _ShadowQuery([1, 2, 3, 4, 5])
        assert q.limit(2).all() == [1, 2]

    def test_delete_clears_and_invokes_callback(self):
        deleted = []
        items = ["a", "b", "c"]
        q = _ShadowQuery(items, on_delete=deleted.append)
        n = q.delete()
        assert n == 3
        assert deleted == ["a", "b", "c"]
        assert q.all() == []

    def test_iter_protocol(self):
        q = _ShadowQuery([1, 2, 3])
        assert list(q) == [1, 2, 3]


# ── ShadowSession: position re-derivation ─────────────────────────────────────


class TestSyntheticPositionDerivation:
    def test_no_trades_yields_zero_positions(self, db_session):
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        assert ss._synthetic_positions == []
        assert ss._synthetic_config.current_cash == strategy.starting_value

    def test_single_buy_creates_one_open_position(self, db_session):
        strategy = _make_strategy(db_session)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0,
                          canslim_score=80)
        ss = ShadowSession(db_session, strategy, [{"ticker": "AAPL", "current_price": 110.0,
                                                   "total_score": 82}])
        assert len(ss._synthetic_positions) == 1
        p = ss._synthetic_positions[0]
        assert p.ticker == "AAPL"
        assert p.shares == 10
        assert p.cost_basis == 100.0
        assert p.current_price == 110.0
        # Cash decremented by buy value
        assert ss._synthetic_config.current_cash == strategy.starting_value - 1000

    def test_cs_provenance_rebuilt_from_buy_signal_factors(self, db_session):
        """is_coiled_spring on the rebuilt position comes from the opening
        BUY's signal_factors (read by the profile-gated pre-earnings
        exemption, cs_exempt arm). Non-CS buys rebuild False."""
        strategy = _make_strategy(db_session)
        _add_shadow_trade(db_session, strategy.id, "CSPX", "BUY", 10, 100.0,
                          signal_factors={"coiled_spring": True,
                                          "cs_confidence": 80})
        _add_shadow_trade(db_session, strategy.id, "PLAIN", "BUY", 5, 50.0,
                          signal_factors={"breakout": True})
        _add_shadow_trade(db_session, strategy.id, "NOSF", "BUY", 5, 50.0)
        ss = ShadowSession(db_session, strategy, [])
        by_ticker = {p.ticker: p for p in ss._synthetic_positions}
        assert by_ticker["CSPX"].is_coiled_spring is True
        assert by_ticker["PLAIN"].is_coiled_spring is False
        assert by_ticker["NOSF"].is_coiled_spring is False

    def test_buy_then_full_sell_yields_no_open_positions(self, db_session):
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0, executed_at=t0)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "SELL", 10, 110.0,
                          executed_at=t0 + timedelta(days=5))
        ss = ShadowSession(db_session, strategy, [])
        assert ss._synthetic_positions == []
        # Cash: -1000 buy + 1100 sell = +100 net vs starting
        assert ss._synthetic_config.current_cash == strategy.starting_value + 100

    def test_partial_sell_reduces_open_shares_fifo(self, db_session):
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0, executed_at=t0)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 110.0,
                          executed_at=t0 + timedelta(days=2))
        # Sell 5 — comes off first BUY (FIFO)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "SELL", 5, 120.0,
                          executed_at=t0 + timedelta(days=5))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        p = ss._synthetic_positions[0]
        assert p.shares == 15  # 10 + 10 - 5
        # Weighted cost: (5*100 + 10*110) / 15 = (500+1100)/15 = 106.666...
        assert abs(p.cost_basis - (1600 / 15)) < 1e-6

    def test_pyramid_action_treated_as_buy_for_fifo(self, db_session):
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0, executed_at=t0)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "PYRAMID", 5, 110.0,
                          executed_at=t0 + timedelta(days=3))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        p = ss._synthetic_positions[0]
        assert p.shares == 15
        assert p.pyramid_count == 1

    def test_partial_profit_taken_rebuilt_from_sell_log(self, db_session):
        """Regression for the 2026-07-13 geometric partial-sell loop: the
        rebuilt position hard-coded partial_profit_taken=0, so the 25% tier
        re-fired every cycle on 25% of the REMAINDER (live: PANW emitted
        1,381 partial SELLs; 4,282 SELLs vs 90 BUYs per stack). The target
        pct in the reason string equals the live accumulator after the sell
        (take_pct = target - already_taken), so the replay restores it."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "PANW", "BUY", 10, 200.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 2.5, 260.0,
            reason="PARTIAL PROFIT 25%: Up 25.2%, score 80 still strong",
            executed_at=t0 + timedelta(days=5))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        assert ss._synthetic_positions[0].partial_profit_taken == 25.0

    def test_partial_profit_taken_keeps_highest_tier(self, db_session):
        """A later 40-tier sell (target 50%) supersedes the 25% record."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "PANW", "BUY", 10, 200.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 2.5, 260.0,
            reason="PARTIAL PROFIT 25%: Up 25.2%, score 80 still strong",
            executed_at=t0 + timedelta(days=5))
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 2.5, 290.0,
            reason="PARTIAL PROFIT 50%: Up 42.0%, score 78 still strong",
            executed_at=t0 + timedelta(days=7))
        ss = ShadowSession(db_session, strategy, [])
        assert ss._synthetic_positions[0].partial_profit_taken == 50.0

    def test_partial_trailing_stop_accumulates_additively(self, db_session):
        """PARTIAL TRAILING STOP adds its pct each firing (live resets the
        peak, so the tier can re-fire on a fresh run-up) — additive, unlike
        the set-to-target profit tiers. 25%-profit then 50%-trailing = 75."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "PANW", "BUY", 10, 200.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 2.5, 260.0,
            reason="PARTIAL PROFIT 25%: Up 25.2%, score 80 still strong",
            executed_at=t0 + timedelta(days=5))
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 3.75, 250.0,
            reason="PARTIAL TRAILING STOP (50%): Peak $290.00 → $250.00 (-13.8%)",
            executed_at=t0 + timedelta(days=7))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        assert ss._synthetic_positions[0].partial_profit_taken == 75.0

    def test_pre_earnings_partial_sets_accumulator_to_25(self, db_session):
        """Regression for the 2026-07-29 pre-earnings Zeno loop: the reason
        format ("PRE-EARNINGS PARTIAL: 5d to earnings, up 12.1% ...") matched
        neither rebuild regex, so the accumulator rebuilt to 0 and the rule
        re-fired every scan, selling 25% of the remainder (live: GEF/SLDE,
        one duplicate fire each before the fix). ai_trader tops the
        accumulator up to exactly 25 (take_pct = 25 − taken, gated < 25),
        so max-to-25 reconstructs it — and duplicate logged fires self-heal
        to the same 25, not 50."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "GEF", "BUY", 10, 76.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "GEF", "SELL", 2.5, 85.29,
            reason="PRE-EARNINGS PARTIAL: 5d to earnings, up 12.1% - protecting gains",
            executed_at=t0 + timedelta(days=5))
        _add_shadow_trade(
            db_session, strategy.id, "GEF", "SELL", 1.875, 85.68,
            reason="PRE-EARNINGS PARTIAL: 5d to earnings, up 12.6% - protecting gains",
            executed_at=t0 + timedelta(days=5, hours=2))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        assert ss._synthetic_positions[0].partial_profit_taken == 25.0

    def test_partial_trailing_keep_format_accumulates(self, db_session):
        """evaluate_sells emits the complement format ("..., keeping 50%")
        with no parenthesized pct; the accumulator must add 100 − keep, and
        the regex must anchor on "keeping" — the drop-from-peak pct earlier
        in the string (-5.3%) must not be the captured number."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "FTI", "BUY", 10, 70.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "FTI", "SELL", 5, 72.66,
            reason="PARTIAL TRAILING STOP: Peak $76.75 → $72.66 (-5.3%), keeping 50%",
            executed_at=t0 + timedelta(days=5))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        assert ss._synthetic_positions[0].partial_profit_taken == 50.0

    def test_partial_profit_taken_resets_after_full_close(self, db_session):
        """Full close ends the lot: a re-entry BUY must start at 0% taken,
        and non-partial SELL reasons must not set the accumulator."""
        strategy = _make_strategy(db_session)
        t0 = datetime.now(timezone.utc) - timedelta(days=10)
        _add_shadow_trade(db_session, strategy.id, "PANW", "BUY", 10, 200.0,
                          executed_at=t0)
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 2.5, 260.0,
            reason="PARTIAL PROFIT 25%: Up 25.2%, score 80 still strong",
            executed_at=t0 + timedelta(days=5))
        _add_shadow_trade(
            db_session, strategy.id, "PANW", "SELL", 7.5, 250.0,
            reason="TRAILING STOP (18%): peak $290.00",
            executed_at=t0 + timedelta(days=6))
        _add_shadow_trade(db_session, strategy.id, "PANW", "BUY", 8, 240.0,
                          executed_at=t0 + timedelta(days=8))
        ss = ShadowSession(db_session, strategy, [])
        assert len(ss._synthetic_positions) == 1
        p = ss._synthetic_positions[0]
        assert p.shares == 8
        assert p.partial_profit_taken == 0.0


# ── ShadowSession: query / add / delete intercepts ────────────────────────────


class TestShadowSessionInterception:
    def test_query_aiportfolio_position_returns_synthetic(self, db_session):
        strategy = _make_strategy(db_session)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0, canslim_score=80)
        ss = ShadowSession(db_session, strategy, [{"ticker": "AAPL", "current_price": 110.0}])
        positions = ss.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == 1).all()
        assert len(positions) == 1
        assert positions[0].ticker == "AAPL"
        assert isinstance(positions[0], _SyntheticPosition)

    def test_query_aiportfolio_config_returns_synthetic(self, db_session):
        strategy = _make_strategy(db_session, starting_value=50000.0)
        ss = ShadowSession(db_session, strategy, [])
        cfg = ss.query(AIPortfolioConfig).filter(AIPortfolioConfig.user_id == 1).first()
        assert isinstance(cfg, _SyntheticConfig)
        assert cfg.starting_cash == 50000.0
        assert cfg.strategy == strategy.parent_strategy

    def test_query_aiportfolio_snapshot_returns_empty(self, db_session):
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        result = ss.query(AIPortfolioSnapshot).filter().order_by().limit(60).all()
        assert result == []

    def test_query_aiportfolio_trade_surfaces_shadow_log(self, db_session):
        strategy = _make_strategy(db_session)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "SELL", 10, 110.0,
                          reason="STOP LOSS")
        ss = ShadowSession(db_session, strategy, [])
        trades = ss.query(AIPortfolioTrade).filter(AIPortfolioTrade.action == "SELL").all()
        assert len(trades) == 1
        assert trades[0].reason == "STOP LOSS"
        assert trades[0].user_id == SHADOW_USER_ID

    def test_query_pass_through_for_other_models(self, db_session):
        strategy = _make_strategy(db_session)
        # Use a unique ticker + clean any leftover row first
        db_session.query(Stock).filter(Stock.ticker == "SHDWZZ").delete()
        db_session.commit()
        stock = Stock(ticker="SHDWZZ", current_price=42.0, canslim_score=90)
        db_session.add(stock)
        db_session.commit()

        ss = ShadowSession(db_session, strategy, [])
        result = ss.query(Stock).filter(Stock.ticker == "SHDWZZ").all()
        assert len(result) == 1
        assert result[0].ticker == "SHDWZZ"

        # Cleanup
        db_session.query(Stock).filter(Stock.ticker == "SHDWZZ").delete()
        db_session.commit()

    def test_add_aiportfolio_trade_translates_to_pending_shadow_trade(self, db_session):
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        live_trade = AIPortfolioTrade(
            ticker="AAPL", action="BUY", shares=5, price=100.0, total_value=500.0,
            reason="test", canslim_score=80, user_id=1,
        )
        ss.add(live_trade)
        pending = ss.drain_pending_shadow_trades()
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"
        assert pending[0].action == "BUY"
        assert pending[0].shadow_strategy_id == strategy.id
        # No real-DB AIPortfolioTrade row should have been written
        assert db_session.query(AIPortfolioTrade).count() == 0

    def test_add_aiportfolio_snapshot_is_silently_dropped(self, db_session):
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        snap = AIPortfolioSnapshot(
            user_id=1, total_value=25000.0, cash=25000.0,
            positions_value=0.0, positions_count=0,
            timestamp=datetime.now(timezone.utc),
            date=datetime.now(timezone.utc).date(),
        )
        ss.add(snap)
        # Sandbox shouldn't have the snapshot either
        assert db_session.query(AIPortfolioSnapshot).count() == 0

    def test_delete_synthetic_position_removes_from_state(self, db_session):
        strategy = _make_strategy(db_session)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0)
        ss = ShadowSession(db_session, strategy, [])
        position = ss._synthetic_positions[0]
        ss.delete(position)
        assert ss._synthetic_positions == []


# ── emit_shadow_buy / emit_shadow_sell ─────────────────────────────────────────


class TestEmitShadowTrades:
    def test_emit_buy_queues_shadow_trade_and_creates_position(self, db_session):
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        starting_cash = ss._synthetic_config.current_cash
        ss.emit_shadow_buy(ticker="AAPL", shares=10, price=100.0,
                           reason="test buy", canslim_score=85)
        pending = ss.drain_pending_shadow_trades()
        assert len(pending) == 1
        assert pending[0].ticker == "AAPL"
        assert pending[0].action == "BUY"
        assert pending[0].canslim_score == 85
        assert pending[0].total_value == 1000.0
        assert ss._synthetic_config.current_cash == starting_cash - 1000.0
        assert len(ss._synthetic_positions) == 1

    def test_emit_buy_stamps_cs_provenance_on_synthetic_position(self, db_session):
        """Intra-cycle CS buys carry is_coiled_spring immediately (same
        provenance the FIFO rebuild derives on the next tick)."""
        strategy = _make_strategy(db_session)
        ss = ShadowSession(db_session, strategy, [])
        ss.emit_shadow_buy(ticker="CSPX", shares=5, price=20.0,
                           reason="🌀 COILED SPRING (12d to earnings)",
                           canslim_score=80,
                           signal_factors={"coiled_spring": True})
        ss.emit_shadow_buy(ticker="PLAIN", shares=5, price=20.0,
                           reason="test buy", canslim_score=80)
        by_ticker = {p.ticker: p for p in ss._synthetic_positions}
        assert by_ticker["CSPX"].is_coiled_spring is True
        assert by_ticker["PLAIN"].is_coiled_spring is False

    def test_emit_sell_full_removes_position_and_records_realized_gain(self, db_session):
        strategy = _make_strategy(db_session)
        purchase_date = datetime.now(timezone.utc) - timedelta(days=20)
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10, 100.0,
                          executed_at=purchase_date)
        ss = ShadowSession(db_session, strategy, [{"ticker": "AAPL", "current_price": 130.0}])
        position = ss._synthetic_positions[0]
        ss.emit_shadow_sell(position=position, shares=10, price=130.0,
                            reason="TAKE PROFIT")
        pending = ss.drain_pending_shadow_trades()
        assert len(pending) == 1
        assert pending[0].action == "SELL"
        assert pending[0].cost_basis == 100.0
        assert pending[0].realized_gain == 300.0  # (130-100)*10
        assert pending[0].holding_days == 20
        assert ss._synthetic_positions == []


# ── run_shadow_strategies orchestrator ────────────────────────────────────────


class TestRunShadowStrategies:
    def test_no_active_strategies_is_noop(self, db_session, monkeypatch):
        # If anything inside _run_one_strategy is called, this would raise.
        from backend import shadow_trader
        monkeypatch.setattr(shadow_trader, "_run_one_strategy",
                            lambda *a, **kw: pytest.fail("should not run"))
        summary = run_shadow_strategies(db_session, [])
        assert summary == {"strategies_run": 0, "total_shadow_trades": 0, "errors": 0}

    def test_archived_strategies_are_skipped(self, db_session, monkeypatch):
        _make_strategy(db_session, name="dead", archived=True)
        called = []
        from backend import shadow_trader
        monkeypatch.setattr(shadow_trader, "_run_one_strategy",
                            lambda sid, ar: called.append(sid) or 0)
        run_shadow_strategies(db_session, [])
        assert called == []  # archived stack was filtered out

    def test_per_strategy_exception_is_caught(self, db_session, monkeypatch):
        _make_strategy(db_session, name="alpha")
        _make_strategy(db_session, name="beta")
        from backend import shadow_trader

        def fake_run(strategy_id, analysis):
            if strategy_id == 1:
                raise RuntimeError("alpha exploded")
            return 2

        monkeypatch.setattr(shadow_trader, "_run_one_strategy", fake_run)
        summary = run_shadow_strategies(db_session, [])
        assert summary["errors"] == 1
        assert summary["strategies_run"] == 1
        assert summary["total_shadow_trades"] == 2

    def test_run_one_strategy_persists_pending_trades_and_marks_activated(
            self, db_session, monkeypatch):
        strategy = _make_strategy(db_session, name="active_stack")
        assert strategy.activated_at is None

        # Stub the live evaluators so we control the decision shape.
        def fake_evaluate_buys(db, **kw):
            stock = type("S", (), {})()
            stock.ticker = "AAPL"
            stock.current_price = 100.0
            stock.canslim_score = 85
            return [{"stock": stock, "value": 1000.0, "reason": "test buy",
                     "signal_factors": {"entry_type": "pre-breakout"},
                     "is_growth_stock": False}]

        def fake_evaluate_sells(db, **kw):
            return []

        monkeypatch.setattr("backend.ai_trader.evaluate_buys", fake_evaluate_buys)
        monkeypatch.setattr("backend.ai_trader.evaluate_sells", fake_evaluate_sells)

        n = _run_one_strategy(strategy.id, [])
        assert n == 1

        # Verify ShadowTrade row landed in the real DB
        rows = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id
        ).all()
        assert len(rows) == 1
        assert rows[0].action == "BUY"
        assert rows[0].ticker == "AAPL"

        # activated_at should be set now
        db_session.refresh(strategy)
        assert strategy.activated_at is not None

    def test_run_one_strategy_with_no_decisions_persists_nothing(
            self, db_session, monkeypatch):
        strategy = _make_strategy(db_session)
        monkeypatch.setattr("backend.ai_trader.evaluate_buys", lambda db, **kw: [])
        monkeypatch.setattr("backend.ai_trader.evaluate_sells", lambda db, **kw: [])

        n = _run_one_strategy(strategy.id, [])
        assert n == 0
        assert db_session.query(ShadowTrade).count() == 0
        # activated_at remains None on a no-trade tick
        db_session.refresh(strategy)
        assert strategy.activated_at is None

    def test_evaluate_buys_exception_does_not_break_strategy(
            self, db_session, monkeypatch):
        strategy = _make_strategy(db_session)
        monkeypatch.setattr(
            "backend.ai_trader.evaluate_buys",
            lambda db, **kw: (_ for _ in ()).throw(RuntimeError("simulated")),
        )
        monkeypatch.setattr("backend.ai_trader.evaluate_sells", lambda db, **kw: [])
        # Should NOT raise — internal try/except swallows
        n = _run_one_strategy(strategy.id, [])
        assert n == 0
        assert db_session.query(ShadowTrade).count() == 0

    def test_scoring_swap_correctness(self, db_session, monkeypatch):
        """Two stacks with different (mocked) evaluator behavior produce
        different shadow trades. Stand-in for real scorer-overrides.
        """
        strat_a = _make_strategy(db_session, name="stack_a", scorer_overrides={"k": 1})
        strat_b = _make_strategy(db_session, name="stack_b", scorer_overrides={"k": 2})

        def fake_buys_factory(score):
            def fake(db, **kw):
                stock = type("S", (), {})()
                stock.ticker = "ZZZZ"
                stock.current_price = 100.0
                stock.canslim_score = score
                return [{"stock": stock, "value": 500.0, "reason": "test"}]
            return fake

        # First strategy emits a BUY at score=80, second at score=90 (mocked).
        # Use a dispatch on shadow_strategy_id via the wrapper.
        def dispatch_evaluate_buys(db, **kw):
            score = 80 if db.shadow_strategy.name == "stack_a" else 90
            return fake_buys_factory(score)(db, **kw)

        monkeypatch.setattr("backend.ai_trader.evaluate_buys", dispatch_evaluate_buys)
        monkeypatch.setattr("backend.ai_trader.evaluate_sells", lambda db, **kw: [])

        run_shadow_strategies(db_session, [])
        rows_a = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strat_a.id
        ).all()
        rows_b = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strat_b.id
        ).all()
        assert len(rows_a) == 1 and rows_a[0].canslim_score == 80
        assert len(rows_b) == 1 and rows_b[0].canslim_score == 90

    def test_parity_baseline_mirrors_decision_to_shadow_trade(
            self, db_session, monkeypatch):
        """Load-bearing parity test (unit form): given a known live decision,
        the shadow_baseline emits a ShadowTrade row whose action / ticker /
        shares / price / canslim_score match the decision exactly. This is
        the property that makes the no-op shadow_baseline a meaningful
        comparison anchor for non-baseline stacks.
        """
        strategy = _make_strategy(db_session, name="shadow_baseline",
                                  scorer_overrides={})

        def known_buy(db, **kw):
            stock = type("S", (), {})()
            stock.ticker = "MSFT"
            stock.current_price = 250.0
            stock.canslim_score = 75
            return [{"stock": stock, "value": 2500.0, "reason": "shadow_baseline buy",
                     "signal_factors": {"entry_type": "active"}, "is_growth_stock": False}]

        monkeypatch.setattr("backend.ai_trader.evaluate_buys", known_buy)
        monkeypatch.setattr("backend.ai_trader.evaluate_sells", lambda db, **kw: [])

        _run_one_strategy(strategy.id, [])

        rows = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id
        ).all()
        assert len(rows) == 1
        st = rows[0]
        assert st.ticker == "MSFT"
        assert st.action == "BUY"
        assert st.canslim_score == 75
        # value=2500 / price=250 = 10 shares
        assert abs(st.shares - 10.0) < 1e-9
        assert st.price == 250.0


# ── Scheduler hook smoke ──────────────────────────────────────────────────────


class TestSchedulerHook:
    def test_scheduler_hook_swallows_shadow_exception(self, monkeypatch):
        """If run_shadow_strategies raises, scheduler.run_continuous_scan must
        catch it. Tested by inspecting the scheduler source for the try/except
        boundary directly — we don't run the full scan here.
        """
        import backend.scheduler as sched
        src = open(sched.__file__).read()
        # Ensure the shadow block is wrapped in try/except with an error log
        idx_try = src.find("from backend.shadow_trader import run_shadow_strategies")
        assert idx_try > -1, "shadow_trader import not found in scheduler"
        # The lines after the import should be inside a try block; find the
        # nearest "try:" before the import + nearest "except" after.
        prefix = src[:idx_try]
        last_try = prefix.rfind("try:")
        suffix = src[idx_try:]
        next_except = suffix.find("except Exception")
        assert last_try > -1
        assert next_except > -1
        # Confirm the except logs a shadow-specific failure
        nearby = src[idx_try: idx_try + next_except + 200]
        assert "Shadow paper-trading failed" in nearby


class TestBuyExecutionGuard:
    """2026-07-22 incident: evaluate_buys returns a ranked DECISION list and
    the translation loop persisted ALL of it — shadow_chop_damper's first
    cycle (fresh stack, wide candidate pool) bought 76 names / $127.7k on a
    $25k stack. The loop must mirror live execution: consume slots and cash
    in rank order, stop when either runs out."""

    @staticmethod
    def _buy_decision(ticker, value=2500.0, price=50.0):
        stock = SimpleNamespace(ticker=ticker, current_price=price,
                                canslim_score=80.0)
        return {"stock": stock, "value": value, "reason": "test buy",
                "signal_factors": None, "is_growth_stock": False}

    def test_fresh_stack_capped_at_slots_and_cash(self, db_session, monkeypatch):
        from backend import shadow_trader
        from backend import ai_trader
        strategy = _make_strategy(db_session, name="guard_test")

        # 40 qualifying decisions of $2.5k each = $100k demanded on a $25k
        # stack with 8 slots. Guard must persist at most 8 buys and at most
        # $25k of total value.
        decisions = [self._buy_decision(f"T{i:02d}") for i in range(40)]
        monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: decisions)

        shadow_trader._run_one_strategy(strategy.id, [])

        rows = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id).all()
        assert 0 < len(rows) <= 8, f"expected <=8 buys, got {len(rows)}"
        assert sum(r.total_value for r in rows) <= 25000.0 + 1e-6

    def test_cash_exhaustion_clamps_last_fill_to_reserve(self, db_session, monkeypatch):
        # Cash-reserve parity (2026-08-06 audit #1): live enforces a dynamic
        # reserve at buy execution and never spends to $0; the shadow loop used
        # to exhaust cash, so shadow arms ran 5-20pp more invested than live and
        # inflated returns in trend. Pin the reserve at 5% for determinism and
        # assert the last fill is clamped so the cushion survives.
        from backend import shadow_trader
        from backend import ai_trader
        strategy = _make_strategy(db_session, name="guard_cash_test")

        # 3 decisions of $12k each on a $25k stack with a 5% ($1,250) reserve:
        # full $12k, clamped $11,750, then STOP (only the reserve remains).
        decisions = [self._buy_decision(f"C{i}", value=12000.0) for i in range(3)]
        monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: decisions)
        monkeypatch.setattr(ai_trader, "compute_dynamic_reserve_pct", lambda *a, **kw: 0.05)

        shadow_trader._run_one_strategy(strategy.id, [])

        rows = sorted(db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id).all(),
            key=lambda r: r.id)
        assert len(rows) == 2
        assert rows[0].total_value == pytest.approx(12000.0)
        assert rows[1].total_value == pytest.approx(11750.0)
        # Reserve preserved: spent <= 25000 - 1250, never drives cash to $0.
        assert sum(r.total_value for r in rows) == pytest.approx(23750.0)

    def test_growth_buy_persists_growth_score_for_rebuild(self, db_session, monkeypatch):
        # Audit #10 (Zeno-class): buy_signal_factors carries no growth_mode_score
        # key, so the FIFO rebuild default was always None → growth-mode shadow
        # positions could never SCORE CRASH or TAKE PROFIT (effective purchase
        # score read as 0). The emit path now injects the stock's growth score.
        from backend import shadow_trader
        from backend import ai_trader
        strategy = _make_strategy(db_session, name="growth_score_test")

        stock = SimpleNamespace(ticker="GRW", current_price=50.0,
                                canslim_score=70.0, growth_mode_score=88.0)
        decision = {"stock": stock, "value": 5000.0, "reason": "growth buy",
                    "signal_factors": {"composite_score": 70.0},
                    "is_growth_stock": True}
        monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: [decision])
        monkeypatch.setattr(ai_trader, "compute_dynamic_reserve_pct", lambda *a, **kw: 0.05)

        shadow_trader._run_one_strategy(strategy.id, [])

        row = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id,
            ShadowTrade.ticker == "GRW").first()
        assert row is not None
        assert row.signal_factors is not None
        assert row.signal_factors.get("growth_mode_score") == pytest.approx(88.0)


# ── Pyramid mirroring (Step 3+, 2026-08-18) ───────────────────────────────────


class TestShadowPyramids:
    """Shadow pyramid phase: emit_shadow_pyramid state math, the
    pyramided_today cooldown (the live cooldown query is multi-entity and
    passes through the intercept → must be enforced shadow-side), FIFO
    rebuild round-trip of PYRAMID rows, and orchestrator execution."""

    def test_emit_shadow_pyramid_updates_position_and_cash(self, db_session):
        strategy = _make_strategy(db_session, name="pyr_emit")
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10.0, 100.0,
                          executed_at=datetime.now(timezone.utc) - timedelta(days=5))
        ss = ShadowSession(db_session, strategy,
                           [{"ticker": "AAPL", "current_price": 110.0, "total_score": 80}])
        pos = ss._synthetic_positions[0]
        cash_before = ss._synthetic_config.current_cash  # 25000 - 1000

        st = ss.emit_shadow_pyramid(position=pos, shares=6.0, price=110.0,
                                    reason="PYRAMID: Winner +10% | Score 80")
        assert st.action == "PYRAMID"
        assert st.total_value == pytest.approx(660.0)
        assert ss._synthetic_config.current_cash == pytest.approx(cash_before - 660.0)
        assert pos.shares == pytest.approx(16.0)
        # (10*100 + 660) / 16 = 103.75 — live cost-basis re-average
        assert pos.cost_basis == pytest.approx(103.75)
        assert pos.pyramid_count == 1

    def test_pyramided_today_reads_persisted_and_pending(self, db_session):
        strategy = _make_strategy(db_session, name="pyr_cooldown")
        _add_shadow_trade(db_session, strategy.id, "OLD", "PYRAMID", 5.0, 100.0,
                          reason="PYRAMID: Winner +12%",
                          executed_at=datetime.now(timezone.utc) - timedelta(days=2))
        _add_shadow_trade(db_session, strategy.id, "HOT", "PYRAMID", 5.0, 100.0,
                          reason="PYRAMID: Winner +9%",
                          executed_at=datetime.now(timezone.utc))
        ss = ShadowSession(db_session, strategy, [])

        assert ss.pyramided_today("HOT") is True     # persisted, today
        assert ss.pyramided_today("OLD") is False    # persisted, 2 days ago
        assert ss.pyramided_today("NEW") is False

        # A pending (this-cycle) pyramid must also arm the cooldown.
        _add_shadow_trade(db_session, strategy.id, "NEW", "BUY", 10.0, 100.0,
                          executed_at=datetime.now(timezone.utc) - timedelta(days=3))
        ss2 = ShadowSession(db_session, strategy,
                            [{"ticker": "NEW", "current_price": 110.0, "total_score": 80}])
        pos = [p for p in ss2._synthetic_positions if p.ticker == "NEW"][0]
        ss2.emit_shadow_pyramid(position=pos, shares=1.0, price=110.0,
                                reason="PYRAMID: Winner +10%")
        assert ss2.pyramided_today("NEW") is True

    def test_fifo_rebuild_roundtrips_pyramid_rows(self, db_session):
        """A PYRAMID row must reconstruct exactly like live: shares merge,
        cost basis re-averages, pyramid_count = open lots - 1, cash ledger
        includes the add."""
        strategy = _make_strategy(db_session, name="pyr_rebuild")
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10.0, 100.0,
                          executed_at=datetime.now(timezone.utc) - timedelta(days=5))
        _add_shadow_trade(db_session, strategy.id, "AAPL", "PYRAMID", 6.0, 110.0,
                          reason="PYRAMID: Winner +10%",
                          executed_at=datetime.now(timezone.utc) - timedelta(days=2))
        ss = ShadowSession(db_session, strategy,
                           [{"ticker": "AAPL", "current_price": 115.0, "total_score": 80}])

        assert len(ss._synthetic_positions) == 1
        pos = ss._synthetic_positions[0]
        assert pos.shares == pytest.approx(16.0)
        assert pos.cost_basis == pytest.approx((1000.0 + 660.0) / 16.0)
        assert pos.pyramid_count == 1
        assert ss._synthetic_config.current_cash == pytest.approx(25000.0 - 1000.0 - 660.0)

    def test_orchestrator_executes_pyramid_between_sells_and_buys(
            self, db_session, monkeypatch):
        """_run_one_strategy persists a PYRAMID ShadowTrade for an eligible
        decision, and the same-day cooldown suppresses a second add on the
        next tick."""
        from backend import shadow_trader
        from backend import ai_trader

        strategy = _make_strategy(db_session, name="pyr_orch")
        _add_shadow_trade(db_session, strategy.id, "AAPL", "BUY", 10.0, 100.0,
                          executed_at=datetime.now(timezone.utc) - timedelta(days=5))

        def fake_evaluate_pyramids(db, **kw):
            positions = db.query(AIPortfolioPosition).all()
            if not positions:
                return []
            return [{"position": positions[0], "amount": 600.0,
                     "shares": 600.0 / (positions[0].current_price or 1),
                     "reason": "Winner +10% | Score 80", "priority": 0}]

        monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_pyramids", fake_evaluate_pyramids)

        analysis = [{"ticker": "AAPL", "current_price": 110.0, "total_score": 80}]
        n = shadow_trader._run_one_strategy(strategy.id, analysis)
        assert n == 1
        rows = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id,
            ShadowTrade.action == "PYRAMID").all()
        assert len(rows) == 1
        assert rows[0].ticker == "AAPL"
        assert rows[0].reason.startswith("PYRAMID:")
        assert rows[0].total_value == pytest.approx(600.0)

        # Second tick same day: cooldown must suppress a repeat add.
        n2 = shadow_trader._run_one_strategy(strategy.id, analysis)
        assert n2 == 0
        rows = db_session.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == strategy.id,
            ShadowTrade.action == "PYRAMID").all()
        assert len(rows) == 1

    def test_orchestrator_caps_three_pyramids_per_cycle(
            self, db_session, monkeypatch):
        from backend import shadow_trader
        from backend import ai_trader

        strategy = _make_strategy(db_session, name="pyr_cap3")
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        for t in tickers:
            _add_shadow_trade(db_session, strategy.id, t, "BUY", 10.0, 100.0,
                              executed_at=datetime.now(timezone.utc) - timedelta(days=5))

        def fake_evaluate_pyramids(db, **kw):
            return [{"position": p, "amount": 300.0, "shares": 3.0,
                     "reason": "Winner +10% | Score 80", "priority": 0}
                    for p in db.query(AIPortfolioPosition).all()]

        monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: [])
        monkeypatch.setattr(ai_trader, "evaluate_pyramids", fake_evaluate_pyramids)

        analysis = [{"ticker": t, "current_price": 110.0, "total_score": 80}
                    for t in tickers]
        n = shadow_trader._run_one_strategy(strategy.id, analysis)
        assert n == 3  # live caps 3 pyramids per cycle
