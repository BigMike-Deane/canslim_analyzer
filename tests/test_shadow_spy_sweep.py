"""Tests for SPY-sweep support in the shadow paper-trading harness
(chop→SPY rotation stacks, 2026-08-03).

The serialization contract under test: sweep moves are ShadowTrade rows
with ticker='SPY' and reason prefix 'SPY SWEEP'. The FIFO rebuild routes
them into the cash ledger + config.spy_sweep_shares and NEVER into the
position FIFO (a swept SPY holding is not a position — no stops, no slot,
no score). Breaking this contract is the Zeno-bug class this repo has been
bitten by three times: state derived from the trade log must round-trip.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db,
    SessionLocal,
    ShadowPositionPeak,
    ShadowStrategy,
    ShadowTrade,
)
from backend.shadow_trader import (
    SHADOW_USER_ID,
    ShadowSession,
    is_sweep_reason,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowPositionPeak).delete()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowPositionPeak).delete()
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.commit()
        db.close()


@pytest.fixture(autouse=True)
def no_history_backfill(monkeypatch):
    monkeypatch.setattr(ShadowSession, "_init_peak_from_history", lambda self, pos: None)


def _make_strategy(db, name="shadow_sweep_test", parent="nostate_chop_spy"):
    s = ShadowStrategy(
        name=name,
        parent_strategy=parent,
        config_snapshot={"strategy": parent},
        scorer_overrides={},
        starting_value=25000.0,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _seed_trade(db, strategy, *, ticker, action, shares, price, reason,
                days_ago=1, cost_basis=None, realized_gain=None):
    t = ShadowTrade(
        shadow_strategy_id=strategy.id,
        ticker=ticker,
        action=action,
        shares=shares,
        price=price,
        total_value=shares * price,
        reason=reason,
        cost_basis=cost_basis,
        realized_gain=realized_gain,
        executed_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    db.add(t)
    db.commit()
    return t


class TestSweepReasonContract:
    def test_prefix_matches(self):
        assert is_sweep_reason("SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        assert is_sweep_reason("SPY SWEEP SOLD: SPY < 50MA")
        assert not is_sweep_reason("shadow buy")
        assert not is_sweep_reason(None)
        assert not is_sweep_reason("")


class TestFifoRebuildWithSweepRows:
    def test_sweep_buy_reconstitutes_shares_not_position(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=40.0, price=500.0,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        assert abs(cfg.spy_sweep_shares - 40.0) < 1e-9
        # Cash ledger: 25,000 - 20,000 swept
        assert abs(cfg.current_cash - 5000.0) < 1e-6
        # SPY must NOT be a synthetic position (no slot, no stops)
        tickers = [p.ticker for p in session._synthetic_positions]
        assert "SPY" not in tickers
        # Average cost carries for the next liquidation's realized gain
        assert abs(session._sweep_avg_cost - 500.0) < 1e-9

    def test_sweep_round_trip_liquidation(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=40.0, price=500.0, days_ago=3,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        _seed_trade(db_session, strategy, ticker="SPY", action="SELL",
                    shares=40.0, price=510.0, days_ago=1,
                    reason="SPY SWEEP SOLD: SPY < 50MA",
                    cost_basis=500.0, realized_gain=400.0)
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        assert cfg.spy_sweep_shares == 0.0
        # 25,000 - 20,000 + 20,400 = 25,400 (sweep P&L landed in cash)
        assert abs(cfg.current_cash - 25400.0) < 1e-6
        assert session._sweep_avg_cost == 0.0
        assert session._synthetic_positions == []

    def test_rebuild_is_idempotent(self, db_session):
        """Zeno-class guard: rebuilding twice from the same log must produce
        identical sweep state — a drifting rebuild re-fires forever."""
        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=10.0, price=480.0,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        s1 = ShadowSession(db_session, strategy, [])
        s2 = ShadowSession(db_session, strategy, [])
        assert s1._synthetic_config.spy_sweep_shares == s2._synthetic_config.spy_sweep_shares
        assert s1._synthetic_config.current_cash == s2._synthetic_config.current_cash
        assert s1._sweep_avg_cost == s2._sweep_avg_cost

    def test_regular_trades_unaffected_by_sweep_rows(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="NVDA", action="BUY",
                    shares=10.0, price=100.0, days_ago=5, reason="shadow buy")
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=20.0, price=500.0, days_ago=2,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        session = ShadowSession(db_session, strategy,
                                [{"ticker": "NVDA", "current_price": 110.0}])
        cfg = session._synthetic_config
        tickers = [p.ticker for p in session._synthetic_positions]
        assert tickers == ["NVDA"]
        assert abs(cfg.spy_sweep_shares - 20.0) < 1e-9
        # 25,000 - 1,000 (NVDA) - 10,000 (sweep)
        assert abs(cfg.current_cash - 14000.0) < 1e-6


class TestSweepDeltaEmitter:
    def test_buy_delta_emits_sweep_buy_row(self, db_session):
        strategy = _make_strategy(db_session)
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        pre_cash, pre_shares = cfg.current_cash, cfg.spy_sweep_shares or 0.0
        # Simulate handle_spy_sweep parking $20,000 at $500
        cfg.current_cash -= 20000.0
        cfg.spy_sweep_shares = 40.0
        st = session.emit_shadow_sweep_delta(pre_cash, pre_shares)
        assert st is not None
        assert st.action == "BUY"
        assert st.ticker == "SPY"
        assert is_sweep_reason(st.reason)
        assert abs(st.shares - 40.0) < 1e-9
        assert abs(st.price - 500.0) < 1e-9
        assert st in session._pending_shadow_trades

    def test_sell_delta_stamps_realized_gain_from_avg_cost(self, db_session):
        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=40.0, price=500.0,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        pre_cash, pre_shares = cfg.current_cash, cfg.spy_sweep_shares
        # Simulate liquidate_spy_sweep at $510
        cfg.current_cash += 40.0 * 510.0
        cfg.spy_sweep_shares = 0.0
        st = session.emit_shadow_sweep_delta(pre_cash, pre_shares, "SPY < 50MA")
        assert st is not None
        assert st.action == "SELL"
        assert "SPY < 50MA" in st.reason
        assert abs(st.cost_basis - 500.0) < 1e-9
        assert abs(st.realized_gain - 400.0) < 1e-6  # (510-500)*40

    def test_no_delta_is_noop(self, db_session):
        strategy = _make_strategy(db_session)
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        st = session.emit_shadow_sweep_delta(cfg.current_cash, cfg.spy_sweep_shares or 0.0)
        assert st is None
        assert session._pending_shadow_trades == []


class TestLiveSweepHelperThroughProxy:
    """The live handle_spy_sweep must work against ShadowSession unmodified —
    its get_portfolio_value/get_or_create_config reads hit the intercepted
    synthetic state. Full parity, no mirrored sweep logic in shadow_trader."""

    def _patch_market(self, monkeypatch, spy_price, ma_50):
        import data_fetcher
        monkeypatch.setattr(
            data_fetcher, "get_cached_market_direction",
            lambda: {"success": True,
                     "indexes": {"SPY": {"price": spy_price, "ma_50": ma_50}}})
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price",
                            lambda ticker: spy_price)

    def test_sweep_buys_idle_cash_above_50ma(self, db_session, monkeypatch):
        self._patch_market(monkeypatch, spy_price=500.0, ma_50=480.0)
        import backend.ai_trader as ai_trader
        from backend.trading_utils import get_strategy_profile

        strategy = _make_strategy(db_session)
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        profile = get_strategy_profile("nostate_chop_spy")
        assert (profile.get("spy_sweep") or {}).get("enabled") is True

        pre_cash, pre_shares = cfg.current_cash, cfg.spy_sweep_shares or 0.0
        ai_trader.handle_spy_sweep(session, cfg, profile, user_id=SHADOW_USER_ID)
        st = session.emit_shadow_sweep_delta(pre_cash, pre_shares)

        # idle = 25,000 - 5% reserve (1,250) = 23,750 > 10% min_idle → swept
        assert st is not None and st.action == "BUY"
        assert abs(cfg.current_cash - 1250.0) < 1.0
        assert abs(cfg.spy_sweep_shares * 500.0 - 23750.0) < 1.0

        # Round-trip: persist the row, rebuild, state must match exactly.
        db_session.add(st)
        db_session.commit()
        rebuilt = ShadowSession(db_session, strategy, [])
        assert abs(rebuilt._synthetic_config.spy_sweep_shares - cfg.spy_sweep_shares) < 1e-6
        assert abs(rebuilt._synthetic_config.current_cash - cfg.current_cash) < 1e-4
        assert "SPY" not in [p.ticker for p in rebuilt._synthetic_positions]

    def test_sweep_liquidates_below_50ma(self, db_session, monkeypatch):
        self._patch_market(monkeypatch, spy_price=470.0, ma_50=480.0)
        import backend.ai_trader as ai_trader
        from backend.trading_utils import get_strategy_profile

        strategy = _make_strategy(db_session)
        _seed_trade(db_session, strategy, ticker="SPY", action="BUY",
                    shares=40.0, price=500.0,
                    reason="SPY SWEEP BUY: parking idle cash (SPY > 50MA)")
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        profile = get_strategy_profile("nostate_chop_spy")

        pre_cash, pre_shares = cfg.current_cash, cfg.spy_sweep_shares
        ai_trader.handle_spy_sweep(session, cfg, profile, user_id=SHADOW_USER_ID)
        st = session.emit_shadow_sweep_delta(pre_cash, pre_shares, "SPY < 50MA")

        assert st is not None and st.action == "SELL"
        assert cfg.spy_sweep_shares == 0.0
        # realized loss: (470-500)*40 = -1,200
        assert abs(st.realized_gain - (-1200.0)) < 1e-6


class TestPartialSweepLiquidation:
    """target_value sells only the shortfall a tick's buys need — the fix
    for the every-tick full-sweep round trip that bloated the ledger and
    minted phantom P&L from mismatched price sources (2026-08-04)."""

    def _cfg(self, cash=1000.0, shares=40.0):
        class Cfg:
            pass
        c = Cfg()
        c.current_cash = cash
        c.spy_sweep_shares = shares
        return c

    def test_partial_sale_raises_target_cash(self, db_session, monkeypatch):
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        cfg = self._cfg()
        ai_trader.liquidate_spy_sweep(db_session, cfg, "test", user_id=SHADOW_USER_ID,
                                      target_value=2000.0)
        assert abs(cfg.current_cash - 3000.0) < 1e-6
        assert abs(cfg.spy_sweep_shares - 36.0) < 1e-9  # sold 4 of 40 shares

    def test_target_above_holdings_sells_everything(self, db_session, monkeypatch):
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        cfg = self._cfg()
        ai_trader.liquidate_spy_sweep(db_session, cfg, "test", user_id=SHADOW_USER_ID,
                                      target_value=999_999.0)
        assert abs(cfg.current_cash - 21000.0) < 1e-6  # 1000 + 40 * 500
        assert cfg.spy_sweep_shares == 0.0

    def test_none_target_keeps_full_liquidation(self, db_session, monkeypatch):
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        cfg = self._cfg()
        ai_trader.liquidate_spy_sweep(db_session, cfg, "test", user_id=SHADOW_USER_ID)
        assert abs(cfg.current_cash - 21000.0) < 1e-6
        assert cfg.spy_sweep_shares == 0.0

    def test_zero_target_is_noop(self, db_session, monkeypatch):
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        cfg = self._cfg()
        ai_trader.liquidate_spy_sweep(db_session, cfg, "test", user_id=SHADOW_USER_ID,
                                      target_value=0.0)
        assert cfg.current_cash == 1000.0
        assert cfg.spy_sweep_shares == 40.0


class TestSweepBuyExecutesAtLivePrice:
    """The 50MA gate reads the cached market direction, but execution must
    price at fetch_live_price — the sell leg does, and a stale buy leg mints
    phantom sweep P&L on every sell/re-park round trip."""

    def test_buy_leg_uses_live_price_not_cached(self, db_session, monkeypatch):
        import data_fetcher
        monkeypatch.setattr(
            data_fetcher, "get_cached_market_direction",
            lambda: {"success": True,
                     "indexes": {"SPY": {"price": 500.0, "ma_50": 480.0}}})
        import backend.ai_trader as ai_trader
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 505.0)

        strategy = _make_strategy(db_session)
        session = ShadowSession(db_session, strategy, [])
        cfg = session._synthetic_config
        profile = {"spy_sweep": {"enabled": True, "min_idle_pct": 10}}

        ai_trader.handle_spy_sweep(session, cfg, profile, user_id=SHADOW_USER_ID)

        # idle = 25,000 - 5% reserve (1,250) = 23,750, bought at LIVE 505
        assert abs(cfg.spy_sweep_shares - 23750.0 / 505.0) < 1e-9
        assert abs(cfg.current_cash - 1250.0) < 1e-6


class TestChopSpyConfigRegistration:
    """Pin the experiment's config so a YAML refactor can't silently drop
    the lever or the stack registration."""

    def test_profile_has_both_levers(self):
        from backend.trading_utils import get_strategy_profile
        p = get_strategy_profile("nostate_chop_spy")
        assert p.get("chop_damper", {}).get("enabled") is True
        assert p.get("chop_damper", {}).get("band_pct") == 1.5
        assert p.get("spy_sweep", {}).get("enabled") is True
        assert p.get("spy_sweep", {}).get("min_idle_pct") == 10

    def test_shadow_stack_registered(self):
        from config_loader import config as app_config
        config_obj = app_config.get("shadow_strategy_profiles", default={}) or {}
        stack = config_obj.get("shadow_chop_spy")
        assert stack is not None, "shadow_chop_spy missing from shadow_strategy_profiles"
        assert stack["parent_strategy"] == "nostate_chop_spy"
        assert stack["starting_value"] == 25000
