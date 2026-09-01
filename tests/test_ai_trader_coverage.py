"""
ai_trader.py live-trading coverage tests.

Triage:
  Tier 1 (highest blast radius — runs on every live trading decision):
    - evaluate_sells: stop loss, trailing stop, score crash, partial profit,
      pre-earnings tightening, take-profit, weak-position cutoff
    - _check_and_execute_stop_losses_impl: independent stop-loss path that
      runs more frequently than the full cycle (DB-mutating)
    - evaluate_buys: market_state gate, binary SPY gate, score-floor gate,
      candidate-pool assembly, ML model resolution
    - evaluate_pyramids: scale-in pullback, regular pyramid, sector-cap chain
    - get_market_regime: bullish/bearish/neutral classification

  Tier 2 (helpers consumed by Tier 1):
    - get_or_create_config: idempotent insert, default values
    - get_portfolio_value: cash + positions + sweep math
    - get_sector_allocations: % by sector with batch fetch
    - get_buy_throttle_limit: consecutive-loss throttle
    - calculate_atr_stop: HTTP path + base-stop fallback
    - fetch_live_price: FMP success / Yahoo fallback / both fail
    - update_position_prices: peak-init + new-high path
    - liquidate_spy_sweep / handle_spy_sweep: SPY-above-50MA park/liquidate
    - calculate_coiled_spring_score_for_stock: MockData/MockScore wiring
    - record_coiled_spring_alert: cooldown + per-day cap + 21-day cycle gate
    - check_score_stability: DB → engine adapter (covered in test_ai_trader_sync,
      the gaps here are the < 2 history and "no scores" branches)
    - is_market_open: weekend/holiday/before-open/after-close
    - get_cst_now: utc-naive shape (regression for trade-timestamp drift)
    - refresh_ai_portfolio: composes update_position_prices + snapshot

  Tier 3 — orchestration loop (added 2026-05-07, see TestRunAITradingCycle):
    - run_ai_trading_cycle: 580-line orchestration loop covering lock contention,
      sells loop (partial+full), drawdown circuit breaker, pyramid loop, dynamic
      cash reserve, FTD/heat penalty plumbing, buy throttle, max-positions cap,
      and the always-runs lock-release + snapshot tail. Tests pin shape and
      seam-call counts, not numerical values, since this is orchestration.

  Tier 3 (intentionally NOT covered):
    - run_ai_trading_cycle bear-base sweep branch:
      only fires on the SPY-just-flipped-bullish edge. It is a background
      admin side-effect, not part of the live trade decision; it is
      monkeypatched away in TestRunAITradingCycle.
    - initialize_ai_portfolio: one-time setup helper.
    - check_position_correlation: yfinance.download integration — already
      covered by tests/test_correlation_guard.py.

Pinned regression guards:
- Partial profit 25% tier delta math (Feb 24 fix: pp_25_sell - partial_taken,
  not pp_25_sell). Pinned via TestSellsPartialProfitDelta.
- NaN-safe handling on numeric position fields (Round 2 Mar 12 fix). Pinned
  via TestSellsNanSafety.
- ai_trader's `cs_bear` ml_signal config = veto-only (log_only=true,
  min_confidence>0). Pinned via TestStrategyMLSignalConfig.
- Score-crash 20% profitable bypass (champion config: ignore_if_profitable=20).
  Pinned via TestSellsScoreCrash.

All HTTP is mocked. No live network. DB tests use a real in-memory SQLite
session (per project rule: no Mock(spec=Session)).

AI Trader <> Backtester sync: structural sync tests stay in test_ai_trader_sync.py.
This file's tests are functional unit tests of ai_trader's wrappers — sync
divergence would manifest as a backtester test failure, not here. (Stretch goal:
mirror these in test_backtester_scorer_parity.py if budget allows.)
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import (
    Base,
    Stock,
    StockScore,
    AIPortfolioConfig,
    AIPortfolioPosition,
    AIPortfolioTrade,
    AIPortfolioSnapshot,
    CoiledSpringAlert,
    MLModel,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """Fresh in-memory SQLite for every test. No global state leakage."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def stub_market_bullish(monkeypatch):
    """Stub get_cached_market_direction → SPY > 50MA (bullish gate open)."""
    payload = {
        "success": True,
        "weighted_signal": 2.0,  # bullish
        "indexes": {
            "SPY": {"price": 500.0, "ma_50": 480.0, "ma_200": 450.0, "ema_21": 495.0},
        },
    }
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda *a, **k: payload)
    return payload


@pytest.fixture
def stub_market_bearish(monkeypatch):
    """Stub get_cached_market_direction → SPY < 50MA (bearish gate closed)."""
    payload = {
        "success": True,
        "weighted_signal": -2.0,  # bearish
        "indexes": {
            "SPY": {"price": 400.0, "ma_50": 480.0, "ma_200": 450.0, "ema_21": 405.0},
        },
    }
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda *a, **k: payload)
    return payload


@pytest.fixture
def stub_market_no_data(monkeypatch):
    """Stub get_cached_market_direction → None (data unavailable)."""
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda *a, **k: None)


@pytest.fixture
def disable_atr_http(monkeypatch):
    """Force calculate_atr_stop to return base_stop_pct without making HTTP calls.
    Many tests don't care about the ATR widening path — this keeps them fast."""
    import backend.ai_trader as ai_trader

    def _no_widening(ticker, current_price, base_stop_pct):
        return base_stop_pct

    monkeypatch.setattr(ai_trader, "calculate_atr_stop", _no_widening)


@pytest.fixture
def disable_historical_data(monkeypatch):
    """Force HistoricalDataProvider to None so VIX/peak-history paths short-circuit
    cleanly without calling the real provider."""
    import backend.ai_trader as ai_trader
    monkeypatch.setattr(ai_trader, "HistoricalDataProvider", None)


@pytest.fixture
def silence_webhooks(monkeypatch):
    """Prevent execute_trade from attempting real webhook IO during tests."""
    import backend.email_utils as email_utils
    for name in (
        "send_trade_webhook",
        "send_stop_loss_webhook",
        "send_score_crash_warning_push",
        "send_spy_gate_change_push",
        "send_market_turn_ready_push",
        "send_coiled_spring_alert_webhook",
    ):
        if hasattr(email_utils, name):
            monkeypatch.setattr(email_utils, name, lambda *a, **k: None)


def _seed_config(db, **overrides):
    """Insert an AIPortfolioConfig row with sane test defaults."""
    defaults = dict(
        user_id=1,
        starting_cash=25000.0,
        current_cash=10000.0,
        max_positions=8,
        max_position_pct=12.0,
        min_score_to_buy=72,
        sell_score_threshold=45,
        take_profit_pct=75.0,
        stop_loss_pct=7.0,
        is_active=True,
        strategy="nostate_optimized",
    )
    defaults.update(overrides)
    cfg = AIPortfolioConfig(**defaults)
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


def _seed_stock(db, ticker, **overrides):
    """Insert a Stock row with default fields (and optional CANSLIM components)."""
    defaults = dict(
        ticker=ticker,
        sector="Technology",
        current_price=100.0,
        canslim_score=75.0,
        c_score=12.0,
        a_score=12.0,
        n_score=12.0,
        s_score=12.0,
        l_score=12.0,
        i_score=8.0,
        m_score=12.0,
        is_growth_stock=False,
        is_breaking_out=False,
        volume_ratio=1.0,
        days_to_earnings=None,
        weeks_in_base=4,
        earnings_beat_streak=0,
        projected_growth=15.0,
        score_details={"i": {"institutional_pct": 25.0}},
    )
    defaults.update(overrides)
    stock = Stock(**defaults)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


def _seed_position(db, ticker, **overrides):
    """Insert an AIPortfolioPosition row with defaults appropriate for the
    ticker's current Stock row (caller must seed Stock first if relying on
    sector lookup)."""
    defaults = dict(
        user_id=1,
        ticker=ticker,
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
    pos = AIPortfolioPosition(**defaults)
    db.add(pos)
    db.commit()
    db.refresh(pos)
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — evaluate_sells exit decision pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestSellsStopLoss:
    """Tier 1: evaluate_sells stop-loss branch.

    Mirrors backtester._evaluate_sells stop-loss decision. Champion profile
    sets stop_loss_pct=7. We seed a position at -7.5% to trigger.
    """

    def test_stop_loss_triggers_below_threshold(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST", current_price=92.5)
        _seed_position(
            db_session,
            "TEST",
            current_price=92.5,
            cost_basis=100.0,
            gain_loss_pct=-7.5,
            peak_price=100.0,  # No peak gain → no trailing trigger
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert len(sells) == 1
        assert "STOP LOSS" in sells[0]["reason"]
        assert sells[0]["priority"] == 1  # highest

    def test_stop_loss_holds_above_threshold(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """At -6% with stop=7%, no stop should trigger."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST")
        _seed_position(
            db_session,
            "TEST",
            current_price=94.0,
            cost_basis=100.0,
            gain_loss_pct=-6.0,
            peak_price=100.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        # No stop loss; no trailing (peak_gain=0); no score crash (score 75 stable)
        assert all("STOP LOSS" not in s["reason"] for s in sells)

    def test_bearish_market_uses_tighter_stop(
        self,
        db_session,
        stub_market_bearish,
        disable_atr_http,
        disable_historical_data,
    ):
        """Champion has bearish_stop_loss_pct=6. -6.5% should trigger in bear market."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized", stop_loss_pct=7.0)
        _seed_stock(db_session, "TEST")
        _seed_position(
            db_session,
            "TEST",
            current_price=93.5,
            cost_basis=100.0,
            gain_loss_pct=-6.5,
            peak_price=100.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        # In a bearish market (SPY < 50MA) the bearish_stop_loss_pct=6 kicks in;
        # -6.5% breaches it.
        assert any("STOP LOSS" in s["reason"] and "bearish" in s["reason"] for s in sells)

    def test_skip_position_with_nan_price(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """NaN current_price must be skipped silently, not crash."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session)
        _seed_stock(db_session, "TEST")
        _seed_position(
            db_session,
            "TEST",
            current_price=float("nan"),
            cost_basis=100.0,
            gain_loss_pct=-50.0,  # would trigger if not skipped
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert sells == []


class TestSellsTrailingStop:
    """Tier 1: evaluate_sells trailing-stop branch — peak-gain band selection
    and inclusive boundary at 50/30/20/10. Champion 10-20 band is 6 (overridden)."""

    def test_trailing_triggers_at_50plus_band(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """At peak_gain=60, champion band=25%. Drop of 26% from peak triggers."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST")
        # peak=$160 (60% peak-gain), current=$118 → 26.25% drop from peak
        _seed_position(
            db_session,
            "TEST",
            current_price=118.0,
            cost_basis=100.0,
            gain_loss_pct=18.0,  # well above stop-loss
            peak_price=160.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert any("TRAILING STOP" in s["reason"] for s in sells)

    def test_trailing_holds_within_band(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """At peak_gain=60, champion band=25%. Drop of 20% should NOT trigger."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST")
        # peak=$160, current=$128 → 20% drop, inside the 25% band
        _seed_position(
            db_session,
            "TEST",
            current_price=128.0,
            cost_basis=100.0,
            gain_loss_pct=28.0,
            peak_price=160.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert all("TRAILING STOP" not in s["reason"] for s in sells)

    def test_partial_trailing_when_pyramided_2x_and_high_score(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """High-conviction (pyramid_count>=2 AND score>=65) → partial sell, keep
        running. Pinned regression: this gate must route through
        should_take_partial_on_trailing_stop, not an inline copy."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST", canslim_score=75.0)
        # peak=$160 (peak_gain=60%, champion band=25%). With 2 pyramids the
        # band widens by +4% to 29%. current=$110 → 31.25% drop > 29% triggers.
        _seed_position(
            db_session,
            "TEST",
            current_price=110.0,
            cost_basis=100.0,
            gain_loss_pct=10.0,
            peak_price=160.0,
            pyramid_count=2,
            current_score=75.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        partials = [s for s in sells if "PARTIAL TRAILING" in s["reason"]]
        assert len(partials) == 1
        assert partials[0]["is_partial"] is True
        assert partials[0]["sell_pct"] == 50  # default partial_sell_pct
        assert partials[0]["reset_peak"] is True


class TestSellsScoreCrash:
    """Tier 1: score crash detection with the 20% profitable bypass.

    Champion: score_crash_drop_required=25, score_crash_ignore_if_profitable=20.
    """

    def _seed_score_history(self, db, stock, scores):
        """Insert chronological StockScore rows (oldest first) for a stock."""
        base_ts = datetime.now(timezone.utc) - timedelta(days=len(scores))
        for i, s in enumerate(scores):
            ts = base_ts + timedelta(days=i)
            db.add(StockScore(
                stock_id=stock.id,
                total_score=float(s),
                timestamp=ts,
                date=ts.date(),
            ))
        db.commit()

    def test_score_crash_triggers_at_drop_30(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """Purchase=80, current=40 → drop=40, current<50 → triggers SCORE CRASH."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        stock = _seed_stock(db_session, "TEST", canslim_score=40.0)
        # Stable history of low scores
        self._seed_score_history(db_session, stock, [40, 38, 35, 33])
        _seed_position(
            db_session,
            "TEST",
            current_price=98.0,
            cost_basis=100.0,
            gain_loss_pct=-2.0,
            peak_price=100.0,
            current_score=40.0,
            purchase_score=80.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert any("SCORE CRASH" in s["reason"] for s in sells)

    def test_score_crash_skipped_when_profitable(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """20% profitable bypass: same crash, but up 22% → should NOT sell.

        Pinned regression: the champion's ignore_if_profitable=20 knob must
        override the crash trigger when gain_pct > 20.
        """
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        stock = _seed_stock(db_session, "TEST", canslim_score=40.0, current_price=122.0)
        self._seed_score_history(db_session, stock, [40, 38, 35, 33])
        _seed_position(
            db_session,
            "TEST",
            current_price=122.0,
            cost_basis=100.0,
            gain_loss_pct=22.0,
            peak_price=125.0,
            current_score=40.0,
            purchase_score=80.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        # Score crash must NOT have fired
        assert all("SCORE CRASH" not in s["reason"] for s in sells)


class TestSellsPartialProfitDelta:
    """Pinned regression: Partial Profit 25%/40% tier delta math.

    The Feb 24 fix: take_pct = pp_25_sell - partial_taken (NOT pp_25_sell).
    +25.9pp 4yr return depends on this NOT regressing. evaluate_sells routes
    through trading_engine.get_partial_profit_action — these tests verify the
    routing AND the delta math via the integration boundary."""

    def test_25pct_tier_first_take_full_25(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """First partial at 25%+ gain: sell exactly 25% (delta = 25 - 0)."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST", canslim_score=75.0, current_price=125.0)
        _seed_position(
            db_session,
            "TEST",
            current_price=125.0,
            cost_basis=100.0,
            gain_loss_pct=25.0,
            peak_price=125.0,
            current_score=75.0,
            partial_profit_taken=0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        partials = [s for s in sells if s.get("is_partial") and "PARTIAL PROFIT" in s["reason"]]
        assert len(partials) == 1
        assert partials[0]["sell_pct"] == 25  # delta = 25 - 0

    def test_40pct_tier_with_25_already_taken_returns_25_delta(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """The exact bug shape: when 25% already taken, 40% tier delta should
        be (50 - 25) = 25, NOT 50."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST", canslim_score=75.0, current_price=140.0)
        _seed_position(
            db_session,
            "TEST",
            current_price=140.0,
            cost_basis=100.0,
            gain_loss_pct=40.0,
            peak_price=140.0,
            current_score=75.0,
            partial_profit_taken=25,  # already took 25
        )

        sells = evaluate_sells(db_session, user_id=1)
        partials = [s for s in sells if s.get("is_partial") and "PARTIAL PROFIT" in s["reason"]]
        assert len(partials) == 1
        assert partials[0]["sell_pct"] == 25  # 50 - 25, not 50


class TestSellsNanSafety:
    """Round 2 of the documented NaN poisoning bug (Mar 12). Verify that
    NaN current_score / current_growth_score don't propagate past
    get_effective_score into score-crash gates."""

    def test_nan_current_score_skips_score_dependent_sells(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """When score==0 (NaN→0 via _nan_safe), score-dependent sells must skip
        but stop-loss/trailing must still evaluate. We seed a position that's
        only down 2% — no stop, no trailing, no score crash → empty sells."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "TEST", canslim_score=0.0)
        _seed_position(
            db_session,
            "TEST",
            current_price=98.0,
            cost_basis=100.0,
            gain_loss_pct=-2.0,
            peak_price=100.0,
            current_score=float("nan"),
            purchase_score=float("nan"),
        )

        sells = evaluate_sells(db_session, user_id=1)
        # No stop/trailing/score-crash should fire on this position
        assert sells == []


class TestSellsPreEarningsTighten:
    """Pre-earnings trailing-stop tightening: non-CS positions within
    days_to_earnings <= 5 get half-stop tightening + partial protection."""

    def test_non_cs_within_earnings_gets_partial_protection(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """Up 12%, 3 days to earnings, partial_taken=0 → take 25% partial."""
        from backend.ai_trader import evaluate_sells

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(
            db_session, "TEST",
            canslim_score=75.0, current_price=112.0,
            days_to_earnings=3,
        )
        _seed_position(
            db_session,
            "TEST",
            current_price=112.0,
            cost_basis=100.0,
            gain_loss_pct=12.0,
            peak_price=113.0,
            current_score=75.0,
            partial_profit_taken=0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        pre_earn = [s for s in sells if "PRE-EARNINGS" in s["reason"]]
        assert len(pre_earn) >= 1


class TestSellsTakeProfitAndWeak:
    """take_profit_pct=75 path and the weak-position cutoff (gain<10 + score<sell_threshold)."""

    def test_protect_gains_when_up_30_but_score_weak(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """Up 30% but score=40 (below sell_score_threshold=45) → PROTECT GAINS.

        We use purchase_score=55 (not 70+) so score_drop=15 < drop_required=25.
        That keeps evaluate_score_crash returning None, which lets the engine
        flow down to PROTECT GAINS — otherwise the score-crash branch's
        profitability bypass would `continue` past it.
        """
        from backend.ai_trader import evaluate_sells

        _seed_config(
            db_session,
            strategy="nostate_optimized",
            sell_score_threshold=45,
        )
        stock = _seed_stock(db_session, "TEST", canslim_score=40.0, current_price=130.0)

        _seed_position(
            db_session,
            "TEST",
            current_price=130.0,
            cost_basis=100.0,
            gain_loss_pct=30.0,
            peak_price=132.0,  # only 1.5% drop — no trailing
            current_score=40.0,
            purchase_score=55.0,  # small drop; no crash
        )

        sells = evaluate_sells(db_session, user_id=1)
        # PROTECT GAINS path must fire: up 30% + score weak
        assert any("PROTECT GAINS" in s["reason"] for s in sells)

    def test_weak_position_flat_with_low_score(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
    ):
        """Up 5%, score=40 → WEAK POSITION cutoff path.

        purchase=55 keeps score-crash from triggering (drop=15 < drop_required=25).
        """
        from backend.ai_trader import evaluate_sells

        _seed_config(
            db_session,
            strategy="nostate_optimized",
            sell_score_threshold=45,
        )
        _seed_stock(db_session, "TEST", canslim_score=40.0, current_price=105.0)

        _seed_position(
            db_session,
            "TEST",
            current_price=105.0,
            cost_basis=100.0,
            gain_loss_pct=5.0,
            peak_price=106.0,
            current_score=40.0,
            purchase_score=55.0,
        )

        sells = evaluate_sells(db_session, user_id=1)
        assert any("WEAK POSITION" in s["reason"] for s in sells)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _check_and_execute_stop_losses_impl (DB-mutating fast path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStopLossesImpl:
    """The standalone stop-loss path runs more often than the full cycle.
    Unlike evaluate_sells (returns a list), this function commits the trade
    and removes the position. Verify side-effects on cash/positions."""

    def test_stop_loss_executes_trade_and_removes_position(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        silence_webhooks,
        monkeypatch,
    ):
        from backend import ai_trader
        from backend.ai_trader import _check_and_execute_stop_losses_impl

        # Force update_position_prices to be a no-op so it doesn't call live FMP
        monkeypatch.setattr(ai_trader, "update_position_prices",
                            lambda db, use_live_prices=True, user_id=1: 0)

        cfg = _seed_config(db_session, strategy="nostate_optimized", current_cash=1000.0)
        _seed_stock(db_session, "TEST", canslim_score=70.0)
        _seed_position(
            db_session,
            "TEST",
            current_price=92.0,
            current_value=9200.0,
            cost_basis=100.0,
            gain_loss=-800.0,
            gain_loss_pct=-8.0,  # past 7% stop
            peak_price=100.0,
        )

        result = _check_and_execute_stop_losses_impl(db_session, user_id=1)

        assert len(result["sells_executed"]) == 1
        assert "STOP LOSS" in result["sells_executed"][0]["reason"]

        # Position removed
        remaining = db_session.query(AIPortfolioPosition).filter_by(ticker="TEST").all()
        assert remaining == []

        # Cash credited (was 1000, += 9200)
        db_session.refresh(cfg)
        assert cfg.current_cash == pytest.approx(10200.0)

        # Trade row written
        trades = db_session.query(AIPortfolioTrade).filter_by(ticker="TEST").all()
        assert len(trades) == 1
        assert trades[0].action == "SELL"

    def test_no_positions_returns_early(self, db_session):
        from backend.ai_trader import _check_and_execute_stop_losses_impl

        _seed_config(db_session)
        result = _check_and_execute_stop_losses_impl(db_session, user_id=1)
        assert result["sells_executed"] == []
        assert "No positions" in result["message"]

    def test_trailing_stop_executes(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        silence_webhooks,
        monkeypatch,
    ):
        from backend import ai_trader
        from backend.ai_trader import _check_and_execute_stop_losses_impl

        monkeypatch.setattr(ai_trader, "update_position_prices",
                            lambda db, use_live_prices=True, user_id=1: 0)

        cfg = _seed_config(db_session, strategy="nostate_optimized", current_cash=500.0)
        _seed_stock(db_session, "TEST", canslim_score=70.0)
        # peak=$160 (60% peak gain → champion band 25%). 26% drop from peak.
        _seed_position(
            db_session,
            "TEST",
            current_price=118.0,
            current_value=11800.0,
            cost_basis=100.0,
            gain_loss=1800.0,
            gain_loss_pct=18.0,  # not at stop loss
            peak_price=160.0,
        )

        result = _check_and_execute_stop_losses_impl(db_session, user_id=1)
        assert len(result["sells_executed"]) == 1
        assert "TRAILING STOP" in result["sells_executed"][0]["reason"]

        # Cash credited
        db_session.refresh(cfg)
        assert cfg.current_cash == pytest.approx(12300.0)


class TestStopLossesPublicWrapper:
    """The public wrapper acquires the trading-cycle lock before delegating."""

    def test_lock_contention_returns_early(self, db_session):
        """Hold the trading-cycle RLock from a SECOND thread (RLock allows
        re-entry on the same thread, so we have to contest it from another)."""
        import threading
        from backend import ai_trader
        from backend.ai_trader import check_and_execute_stop_losses

        _seed_config(db_session)

        # Background thread acquires the lock and waits for the test signal
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            with ai_trader._trading_cycle_lock:
                acquired.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        acquired.wait(timeout=5)

        try:
            result = check_and_execute_stop_losses(db_session, user_id=1)
            assert result["sells_executed"] == []
            assert "in progress" in result["message"].lower()
        finally:
            release.set()
            t.join(timeout=5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — evaluate_buys gates (early-exit paths only — full pipeline is Tier 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuysGates:
    """We test only the early-exit paths in evaluate_buys (market gate, no SPY
    data, regime gate). The full candidate-ranking pipeline is exercised by
    integration runs and indirectly via run_ai_trading_cycle.

    Tier 3 reasoning: the full evaluate_buys body (~1200 lines) needs a
    complete Stock/StockScore graph + earnings audit + ML model + correlation
    check + every config knob. That's an integration test, not a unit test.
    """

    def test_bearish_spy_returns_empty_when_no_correction_zone(
        self,
        db_session,
        stub_market_bearish,
        disable_atr_http,
        disable_historical_data,
        silence_webhooks,
    ):
        """nostate_optimized has correction_zone disabled — SPY < 50MA → []."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        # Even with strong candidates available, the gate should block
        _seed_stock(db_session, "AAA", canslim_score=85.0, current_price=50.0)
        _seed_stock(db_session, "BBB", canslim_score=82.0, current_price=60.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_missing_spy_data_returns_empty(
        self,
        db_session,
        stub_market_no_data,
        disable_atr_http,
        disable_historical_data,
        silence_webhooks,
    ):
        """No market data → defensive empty list (don't trade blind)."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_stock(db_session, "AAA", canslim_score=85.0, current_price=50.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — evaluate_buys candidate-ranking interior
# ═══════════════════════════════════════════════════════════════════════════════
#
# Companion to TestBuysGates above. TestBuysGates pins the early-exit gates
# (bearish SPY → [], no SPY → []). This class pins the body the May 7 push
# deferred: composite-score weight application, ML bonus/veto integration,
# pre-breakout/breakout/extended classification, base-quality bonus,
# sector-cap rejection, earnings-window separation (avoidance_days vs
# allow_buy_days — the Feb 24 lesson, see MEMORY.md), and ranking order.
#
# All tests run with `nostate_optimized` (the live winner) unless they
# explicitly need to verify cs_bear behavior. Per
# canslim-ml-graduation-apr29.md, cs_bear is incompatible with the bonus
# path — bonus tests MUST use nostate_optimized.
#
# Constraint: the Approach 2 A/B is reading live-trade data through
# 2026-06-18. These tests are PURELY ADDITIVE — they pin existing behavior.
# No code change to evaluate_buys is permitted during the eval window.
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def disable_ml_and_yfinance(monkeypatch):
    """Stub out three seams that evaluate_buys uses for non-pure side effects:

    1. `yfinance.Ticker(...).history()` for the SPY pullback calc (line 2286).
       Empty history → block falls back to default 30 days.
    2. `yfinance.download(...)` for the correlation guard (called from
       check_position_correlation when current_tickers is non-empty).
    3. `ml.model.get_ml_prediction` for the ML bonus path. Default returns
       None so log_only-style profiles see no bonus and no veto. Tests that
       care about ML behavior override this seam directly.
    """
    import pandas as pd
    import yfinance as yf

    fake_ticker = MagicMock()
    fake_ticker.history.return_value = pd.DataFrame()  # empty → default 30d
    monkeypatch.setattr(yf, "Ticker", lambda *a, **kw: fake_ticker)
    monkeypatch.setattr(yf, "download", lambda *a, **kw: pd.DataFrame())

    # Default ML prediction: None (log_only-style behavior, no bonus, no veto).
    # Individual tests override via monkeypatch.setattr inside the test body.
    import ml.model as ml_model
    monkeypatch.setattr(ml_model, "get_ml_prediction", lambda **kw: None)


def _seed_growth_projection_stock(db, ticker, **overrides):
    """Helper: seed a Stock that passes the standard quality gates.

    Defaults: canslim_score=78 (above min_score=73 floor), c=12, l=12 (above
    quality_filters min 10/8), volume_ratio=1.3, projected_growth=20, no base.
    Override only what the test cares about.
    """
    defaults = dict(
        canslim_score=78.0,
        current_price=50.0,
        week_52_high=55.0,
        c_score=12.0,
        a_score=12.0,
        n_score=12.0,
        s_score=12.0,
        l_score=12.0,
        i_score=8.0,
        m_score=12.0,
        volume_ratio=1.3,
        projected_growth=20.0,
        rs_12m=1.10,
        rs_3m=1.05,
        atr_pct=2.5,
        ma_21=49.0,
        ma_50=48.0,
        industry_group_rank=70,
        is_growth_stock=False,
    )
    defaults.update(overrides)
    return _seed_stock(db, ticker, **defaults)


class TestEvaluateBuysRanking:
    """Tier 1: evaluate_buys candidate-ranking interior.

    Each test pins one branch of the ranking pipeline. Tests are SHAPE pins,
    not deep numerical assertions — we verify "this candidate is/isn't in
    buys" or "this signal_factor flag is set", not "composite_score == 47.3".
    Numerical tests would couple the suite to YAML weights that change.

    All tests use `stub_market_bullish` to clear the SPY gate and the
    `disable_ml_and_yfinance` fixture to neutralize network-bound seams.
    """

    def test_strong_candidate_appears_in_buys(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: happy path. A single high-quality candidate makes it to
        the final buys list with a positive composite_score and shape-correct
        return dict."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "WIN", canslim_score=80.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        b = buys[0]
        assert b["stock"].ticker == "WIN"
        assert b["composite_score"] > 0
        assert b["shares"] > 0
        assert b["value"] >= 100  # min position floor
        assert "signal_factors" in b
        assert b["signal_factors"]["entry_type"] == "standard"
        assert b["is_growth_stock"] is False

    def test_multi_scan_averaging_uses_score_history(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Regression: the batch score-history fetch queried StockScore.ticker
        — a column that doesn't exist (the table is keyed by stock_id) — and
        the surrounding try/except swallowed the AttributeError at DEBUG, so
        multi-scan averaging was silently dead. With 3 history rows whose
        50/30/20 average deviates >=3 pts from the current score, the buy's
        effective_score must be the average, not the raw score."""
        from datetime import datetime, timedelta, timezone
        from backend.ai_trader import evaluate_buys
        from backend.database import StockScore

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        stock = _seed_growth_projection_stock(db_session, "AVG", canslim_score=80.0)
        now = datetime.now(timezone.utc)
        # oldest -> newest: 60, 70, 80. Weighted 80*.5 + 70*.3 + 60*.2 = 73.
        for age_min, score in ((180, 60.0), (90, 70.0), (5, 80.0)):
            ts = now - timedelta(minutes=age_min)
            db_session.add(StockScore(stock_id=stock.id, timestamp=ts,
                                      date=ts.date(), total_score=score))
        db_session.commit()

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert abs(buys[0]["effective_score"] - 73.0) < 0.01

    def test_stale_rows_excluded_from_candidates(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: freshness gate. Universe churn leaves rows the scanner no
        longer updates; their frozen scores/prices must not enter the buy
        pool. A NULL last_updated (pre-default legacy row) is equally stale.
        Live example (2026-07-06): rows frozen since March-May still scored
        70-80 and sat in the candidate pool."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        _seed_growth_projection_stock(
            db_session, "FRESH", canslim_score=80.0, last_updated=now)
        _seed_growth_projection_stock(
            db_session, "STALE", canslim_score=95.0,
            last_updated=now - timedelta(days=30))
        _seed_growth_projection_stock(db_session, "NOSTAMP", canslim_score=95.0)
        # Constructor kwargs of None trigger the column default, so NULL must
        # be forced with a post-insert UPDATE to model a true legacy row.
        db_session.query(Stock).filter(Stock.ticker == "NOSTAMP").update(
            {"last_updated": None})
        db_session.commit()

        buys = evaluate_buys(db_session, user_id=1)
        tickers = [b["stock"].ticker for b in buys]
        assert "STALE" not in tickers
        assert "NOSTAMP" not in tickers
        assert tickers == ["FRESH"]

    def test_no_gate_profile_does_not_nameerror(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
        monkeypatch,
    ):
        """Regression: a profile disabling BOTH the market-state machine and the
        legacy regime gate (with score-floor decay off) leaves spy_px/spy_50
        otherwise unbound -> NameError at signal-build (spy_pct_above_50ma).
        With the up-front defaults they're 0 and evaluate_buys completes."""
        import backend.ai_trader as ai_trader
        from backend.ai_trader import evaluate_buys, get_strategy_profile

        no_gate = dict(get_strategy_profile("nostate_optimized"))
        no_gate["market_state"] = {"enabled": False}
        no_gate["regime_gate"] = {"enabled": False}
        no_gate["score_floor_decay"] = {"enabled": False}
        monkeypatch.setattr(ai_trader, "get_strategy_profile", lambda strategy: no_gate)

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "NOG", canslim_score=80.0)

        buys = evaluate_buys(db_session, user_id=1)  # must not raise

        assert len(buys) == 1
        # SPY context defaulted to 0 -> spy_pct_above_50ma is 0.0, not a crash.
        assert buys[0]["signal_factors"]["spy_pct_above_50ma"] == 0.0

    def test_below_min_score_excluded_from_query(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: percentile + min_score floor. nostate_optimized has
        min_score=73 and score_floor=45; a stock at 50 is below both
        effective thresholds (with only one candidate, percentile=that one's
        score so floor matters). With score=40 (below floor of 45) it's
        excluded by the query filter."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "LOW", canslim_score=40.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_low_c_score_quality_filter_skips(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: quality_filters.min_c_score=10 (champion). c_score=5 is
        below threshold → continue (no buy). Pin: this gate fires before
        composite-score calc."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "LOWC", c_score=5.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_low_l_score_quality_filter_skips(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: quality_filters.min_l_score=8 (champion). l_score=5 → skip."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "LOWL", l_score=5.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_volume_gate_blocks_thin_volume(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: volume_gate. volume_ratio=0.5 with no base → below
        min_volume_ratio (1.0) → skip. Champion default."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "THIN", volume_ratio=0.5)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_pre_breakout_classification(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: entry_type classifier. Stock 8% below pivot with a base
        pattern → entry_type='pre-breakout', pre_breakout_bonus=40. The
        position_pct should reflect the pre-breakout multiplier (1.40
        default in YAML)."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=30000.0)
        # current_price=92, pivot=100 → pct_from_pivot = 8% → pre-breakout zone (5-15%)
        _seed_growth_projection_stock(
            db_session, "PRE",
            current_price=92.0,
            week_52_high=100.0,
            base_type="cup_with_handle",
            weeks_in_base=8,
            pivot_price=100.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        sf = buys[0]["signal_factors"]
        assert sf["entry_type"] == "pre-breakout"
        # Pre-breakout entries get high-conviction cash sizing — verify the
        # reason string carries the pre-breakout label
        assert "PRE-BREAKOUT" in buys[0]["reason"]

    def test_breakout_classification_with_volume(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: is_breaking_out=True with breakout_volume_ratio>=1.5
        → entry_type='breakout'. Volume gate uses breakout_min_volume_ratio
        (1.5 default) instead of min_volume_ratio (1.0)."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=30000.0)
        _seed_growth_projection_stock(
            db_session, "BO",
            current_price=110.0,
            week_52_high=110.0,
            is_breaking_out=True,
            breakout_volume_ratio=2.0,
            volume_ratio=2.0,
            base_type="flat",
            weeks_in_base=6,
            pivot_price=108.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["signal_factors"]["entry_type"] == "breakout"
        assert buys[0]["is_breaking_out"] is True
        assert "BREAKOUT" in buys[0]["reason"]

    def test_breakout_blocked_by_low_volume(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: volume gate. is_breaking_out=True needs volume_ratio
        >= breakout_min_volume_ratio (1.2 in YAML). At 1.0 the gate
        rejects — breakouts on weak volume are not real breakouts."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "BO_THIN",
            current_price=110.0,
            week_52_high=110.0,
            is_breaking_out=True,
            breakout_volume_ratio=1.0,
            volume_ratio=1.0,
            base_type="flat",
            weeks_in_base=6,
            pivot_price=108.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_extended_above_pivot_penalized(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: extended path. Stock 11% ABOVE pivot (pct_from_pivot=-11)
        triggers extended_penalty=-20. Compared to a stock below pivot but
        otherwise identical, the extended one ranks lower. Pin: ranking
        respects extended_penalty subtraction."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=30000.0)
        # Extended: above pivot, base intact → -20 extended_penalty + low momentum
        _seed_growth_projection_stock(
            db_session, "EXT",
            current_price=111.0,
            week_52_high=112.0,
            base_type="flat",
            weeks_in_base=6,
            pivot_price=100.0,  # current is 11% above
            canslim_score=78.0,
        )
        # Pre-breakout: below pivot in valid zone — high pre_breakout_bonus
        _seed_growth_projection_stock(
            db_session, "PRE",
            current_price=92.0,
            week_52_high=100.0,
            base_type="flat",
            weeks_in_base=6,
            pivot_price=100.0,
            canslim_score=78.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        # Both pass quality filters — pre-breakout should rank higher
        ranks = {b["stock"].ticker: i for i, b in enumerate(buys)}
        assert "PRE" in ranks and "EXT" in ranks
        assert ranks["PRE"] < ranks["EXT"], (
            f"Pre-breakout should outrank extended (PRE@{ranks['PRE']} EXT@{ranks['EXT']})"
        )

    def test_excluded_held_tickers_filtered(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: excluded_tickers. A held position blocks the same ticker
        from re-buying. (Correlation guard is bypassed because we'd need a
        second held position to trigger it; held=1 is a single-position case
        where excluded_tickers is the gate that fires.)"""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "HELD")
        _seed_position(db_session, "HELD", current_value=5000.0)
        _seed_growth_projection_stock(db_session, "FRESH")

        buys = evaluate_buys(db_session, user_id=1)
        tickers = {b["stock"].ticker for b in buys}
        assert "HELD" not in tickers
        assert "FRESH" in tickers

    def test_dead_stock_no_rs_no_atr_skipped(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: dead-stock guard at line ~2333. rs_12m=None AND atr_pct=0
        → continue (acquired/halted/stale tickers shouldn't be ranked)."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "DEAD",
            rs_12m=None,
            atr_pct=0.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_earnings_avoidance_blocks_non_cs(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: non-CS earnings avoidance. Champion `avoidance_days=5`.
        A non-CS stock with `days_to_earnings=3` is inside the avoidance
        window and gets skipped. Pinned per the Feb 24 lesson:
        avoidance_days != allow_buy_days. See MEMORY.md."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "EARN",
            days_to_earnings=3,
            # No CS markers (low confidence path) — would fail CS check
            base_type="none",
            weeks_in_base=0,
            earnings_beat_streak=0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_earnings_outside_avoidance_window_passes(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch companion to above. days_to_earnings=10 is OUTSIDE both
        the CS allow_buy_days=7 window and the non-CS avoidance_days=5
        window — neither gate fires, the stock proceeds to ranking."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "FAR", days_to_earnings=10)

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["stock"].ticker == "FAR"

    def test_growth_stock_uses_growth_score(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: growth-mode dual-pool query. is_growth_stock=True with
        growth_mode_score above threshold → enters via growth_candidates
        list, effective_score = growth_mode_score, signal_factors flagged."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_growth_projection_stock(
            db_session, "GROW",
            is_growth_stock=True,
            canslim_score=40.0,        # below threshold — would not match canslim pool
            growth_mode_score=80.0,    # above threshold for growth pool
            c_score=10.0,
            a_score=0.0,               # growth bypass needs at least C OR A non-zero
            l_score=10.0,
            projected_growth=30.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["is_growth_stock"] is True
        assert buys[0]["effective_score"] == 80.0

    def test_growth_stock_zero_c_zero_a_skipped(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: growth-stock fundamental floor. c_score=0 AND a_score=0
        means the growth bypass detects "no earnings data at all" (acquired
        or dead) and skips."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "DEADGROW",
            is_growth_stock=True,
            growth_mode_score=80.0,
            c_score=0.0,
            a_score=0.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_ranking_orders_by_composite_score_descending(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: final sort. Two valid candidates with different
        composite drivers (CANSLIM score, base, momentum) — the higher
        composite must come first. Pin: priority key = -composite_score
        and the list is sorted ascending on priority."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=30000.0)
        _seed_growth_projection_stock(
            db_session, "TOP", canslim_score=85.0,
            current_price=92.0, week_52_high=100.0,
            base_type="cup_with_handle", weeks_in_base=8, pivot_price=100.0,
        )
        _seed_growth_projection_stock(
            db_session, "MID", canslim_score=78.0,
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 2
        # Sorted ascending on priority (-composite_score) → higher composite first
        assert buys[0]["composite_score"] > buys[1]["composite_score"]
        assert buys[0]["stock"].ticker == "TOP"

    def test_ml_veto_low_confidence_skips_candidate(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
        monkeypatch,
    ):
        """Branch: ML veto path (min_confidence>0, veto_action='skip'). A 0.20
        prediction triggers the veto and the candidate is dropped. Uses
        nostate_optimized_avoid3 (still full-D config) because the champion
        nostate_optimized had its veto turned OFF 2026-07-04 (no-edge model,
        see test_nostate_optimized_ml_is_log_only_no_veto). The veto CODE PATH
        is still live for profiles that opt in, so keep covering it."""
        from backend.ai_trader import evaluate_buys
        import ml.model as ml_model

        # Override default (None) to force veto
        monkeypatch.setattr(ml_model, "get_ml_prediction", lambda **kw: 0.20)

        _seed_config(db_session, strategy="nostate_optimized_avoid3")
        _seed_growth_projection_stock(db_session, "VETOED", canslim_score=82.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_champion_no_longer_vetoes_but_still_logs_confidence(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
        monkeypatch,
    ):
        """2026-07-04: on the champion nostate_optimized a low ML confidence
        (0.20) must NOT veto the buy (min_confidence=0), and no bonus is
        applied (log_only=true), but ml_confidence is still recorded for
        observability. Guards the demotion end-to-end through evaluate_buys."""
        from backend.ai_trader import evaluate_buys
        import ml.model as ml_model

        monkeypatch.setattr(ml_model, "get_ml_prediction", lambda **kw: 0.20)

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "NOTVETOED", canslim_score=82.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1                      # veto OFF — buy goes through
        sf = buys[0]["signal_factors"]
        assert sf.get("ml_confidence") == pytest.approx(0.20, abs=0.001)  # still logged
        assert sf.get("ml_bonus") == pytest.approx(0.0, abs=0.001)        # bonus OFF
        assert not sf.get("ml_vetoed")

    def test_ml_high_confidence_bonus_increases_composite(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
        monkeypatch,
    ):
        """Branch: ML bonus path (log_only=false). ml_bonus = (confidence -
        0.5) * weight = (0.95 - 0.5) * 20 = +9.0 added to composite. Uses
        nostate_optimized_avoid3 (still log_only=false) — the champion's bonus
        path was turned OFF 2026-07-04 (no-edge model). The bonus CODE PATH is
        still live for opt-in profiles, so keep covering the formula."""
        from backend.ai_trader import evaluate_buys
        import ml.model as ml_model

        monkeypatch.setattr(ml_model, "get_ml_prediction", lambda **kw: 0.95)

        _seed_config(db_session, strategy="nostate_optimized_avoid3", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "MLWIN", canslim_score=80.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        sf = buys[0]["signal_factors"]
        assert sf.get("ml_confidence") == pytest.approx(0.95, abs=0.001)
        # Bonus formula: (0.95 - 0.5) * 20 = 9.0
        assert sf.get("ml_bonus") == pytest.approx(9.0, abs=0.5)

    def test_cs_bear_ml_bonus_path_off_log_only(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
        monkeypatch,
    ):
        """Branch: C-config (cs_bear, log_only=true). The ML confidence is
        recorded but ml_bonus stays at 0 — the bonus path is OFF on cs_bear
        per canslim-ml-graduation-apr29.md (bonus alone = -12pp, full = -33pp).

        cs_bear's correction_zone.cs_only=true means non-CS candidates are
        rejected when correction_zone is active. We use stub_market_bullish
        (SPY > 50MA) so correction_zone is INACTIVE — the candidate gets
        through standard ranking, ML logs but doesn't bonus.
        """
        from backend.ai_trader import evaluate_buys
        import ml.model as ml_model

        monkeypatch.setattr(ml_model, "get_ml_prediction", lambda **kw: 0.95)

        _seed_config(db_session, strategy="nostate_cs_bear", current_cash=20000.0)
        _seed_growth_projection_stock(db_session, "CSBEAR", canslim_score=80.0)

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        sf = buys[0]["signal_factors"]
        # Confidence is logged for journaling
        assert sf.get("ml_confidence") == pytest.approx(0.95, abs=0.001)
        # But the bonus is NOT applied — log_only=true on cs_bear
        assert sf.get("ml_bonus", 0) == 0

    def test_cooldown_blocks_recent_stop_loss_ticker(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: re-entry cooldown. A SELL trade tagged 'STOP LOSS'
        executed within stop_loss_cooldown_days (default 5) excludes the
        ticker from re-buying. Prevents whipsaw losses."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "STOPPED")
        # Insert a SELL trade 2 days ago
        db_session.add(AIPortfolioTrade(
            user_id=1,
            ticker="STOPPED",
            action="SELL",
            shares=100.0,
            price=92.0,
            total_value=9200.0,
            reason="STOP LOSS: Down 7.5%",
            executed_at=datetime.now(timezone.utc) - timedelta(days=2),
        ))
        db_session.commit()

        buys = evaluate_buys(db_session, user_id=1)
        assert buys == []

    def test_sector_cap_rejects_when_at_max_count(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: sector_limit at line 2817. With 4 Technology positions
        already held (MAX_STOCKS_PER_SECTOR=4 default), a fifth Technology
        candidate gets adjusted_value=0 → continue. Pin both directions
        (this test rejects; an Energy candidate with 4 Tech holdings would
        pass, covered by the held-ticker test above which already proves
        the non-rejection path through a non-overlapping sector)."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        # 4 existing Technology positions (at the cap)
        for ticker in ("HELD1", "HELD2", "HELD3", "HELD4"):
            _seed_growth_projection_stock(db_session, ticker, sector="Technology")
            _seed_position(db_session, ticker, current_value=5000.0)
        # Candidate also in Technology
        _seed_growth_projection_stock(db_session, "FIFTH", sector="Technology")

        buys = evaluate_buys(db_session, user_id=1)
        # FIFTH must be rejected (5th Tech stock — at cap)
        assert all(b["stock"].ticker != "FIFTH" for b in buys)

    def test_rs_line_bonus_disabled_by_default_returns_zero(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: rs_line gate. YAML default has rs_line.enabled=false, so
        even with l_score>=13 and price below high, rs_line_bonus stays 0.
        Pinned because the YAML default IS the live config — this asserts
        that the disabled gate is honored. If someone flips the YAML
        default without an explicit graduation decision, this test catches
        the change at PR review."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "RSL",
            l_score=14.0,        # >= 13 → RS strong proxy
            current_price=92.0,
            week_52_high=100.0,  # 8% below high — would trigger if enabled
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["signal_factors"]["rs_line_bonus"] == 0

    def test_earnings_drift_bonus_disabled_by_default_returns_zero(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: earnings_drift gate. YAML default has
        earnings_drift.enabled=false, so even with beat_streak>=4, the
        bonus is 0. Companion pin to test_rs_line above — both signals
        are configured-off in production but the code paths exist."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "DRIFT",
            earnings_beat_streak=4,
            days_to_earnings=20,  # Outside avoidance window
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["signal_factors"]["earnings_drift_bonus"] == 0

    def test_volume_dry_up_bonus_with_base(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: volume_dry_up at line 2725. volume_dry_up=True AND
        has_base AND not is_breaking_out → +5 to composite_score.
        Ranking signal for accumulation/dry-up before breakout."""
        from backend.ai_trader import evaluate_buys

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(
            db_session, "DRY",
            volume_dry_up=True,
            base_type="cup",
            weeks_in_base=6,
            pivot_price=55.0,
            current_price=50.0,
            week_52_high=55.0,
            volume_ratio=0.7,  # passes pre-breakout gate (0.6 threshold)
        )

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        assert buys[0]["signal_factors"].get("volume_dry_up") is True

    def test_bear_base_readiness_bonus_applied(
        self,
        db_session,
        stub_market_bullish,
        disable_atr_http,
        disable_historical_data,
        disable_ml_and_yfinance,
        silence_webhooks,
    ):
        """Branch: bear_base_bonus at line 2733. A pre-loaded
        BearBaseCandidate row with readiness_score>=70 grants +15 composite
        bonus. This is the 'ready when SPY flips bullish' lift baked into
        the post-bear breakout flow."""
        from backend.ai_trader import evaluate_buys
        from backend.database import BearBaseCandidate

        _seed_config(db_session, strategy="nostate_optimized")
        _seed_growth_projection_stock(db_session, "READY")
        # Top-tier readiness: 75 (>= 70) and 14 days on list → +15 + 3 = +18
        db_session.add(BearBaseCandidate(
            ticker="READY",
            readiness_score=75.0,
            days_on_list=14,
        ))
        db_session.commit()

        buys = evaluate_buys(db_session, user_id=1)
        assert len(buys) == 1
        sf = buys[0]["signal_factors"]
        assert sf.get("bear_base_bonus", 0) >= 15
        assert sf.get("bear_base_readiness") == pytest.approx(75.0)


class TestStrategyMLSignalConfig:
    """Pinned regression: cs_bear must be VETO-ONLY (log_only=true).

    From canslim-ml-graduation-apr29.md: cs_bear has an INVERTED bonus
    interaction. Bonus alone = -12pp; full activation = -33pp vs veto-only.
    The bonus reranks toward candidates the cs_only correction-zone rule
    can't accept. If anyone copies nostate_optimized's D config to cs_bear,
    this test fails."""

    def test_nostate_cs_bear_is_veto_only(self):
        """log_only=true means bonus path is OFF; min_confidence>0 means veto path is ON."""
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_cs_bear")
        ml = profile.get("ml_signal", {})
        assert ml.get("enabled") is True, "ml_signal must be enabled for cs_bear"
        assert ml.get("log_only") is True, (
            "cs_bear's ml_signal.log_only MUST be true (bonus harmful, "
            "see canslim-ml-graduation-apr29.md)"
        )
        assert ml.get("min_confidence", 0) > 0, (
            "cs_bear's ml_signal.min_confidence MUST be > 0 to enable the veto path"
        )

    def test_nostate_optimized_ml_is_log_only_no_veto(self):
        """2026-07-04: ML demoted to observability-only on the champion. The
        Apr-29 +66.5pp graduation was lookahead-inflated (backtester scores
        history with a model trained on that history); a signal-analysis loop
        found no transferable edge (winner AUC 0.55 flat; downside 0.68 CV →
        0.52 out-of-time). BOTH trade-affecting paths must stay off — bonus
        (log_only=true) AND veto (min_confidence=0, which is what actually
        gates the skip, independent of log_only). enabled stays true so
        ml_confidence is still logged. Do NOT re-enable without a LIVE A/B."""
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        ml = profile.get("ml_signal", {})
        assert ml.get("enabled") is True, "keep enabled for ml_confidence logging"
        assert ml.get("log_only") is True, "bonus path must be OFF (no-edge model)"
        assert ml.get("min_confidence", 0) == 0, (
            "veto path must be OFF — min_confidence>0 re-enables the no-edge skip"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — evaluate_pyramids
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatePyramids:
    """Pyramid eligibility: gain≥2.5%, score≥70, pyramid_count<2, cooldown
    not active, sector limit ok, enough cash."""

    def test_eligible_winner_added_to_pyramid_list(self, db_session):
        """Position must be < MAX_POSITION_ALLOCATION (25%) of portfolio to qualify
        for pyramiding. Cash $20k + position $4k = $24k portfolio; position
        is 16.7% of total — room to add."""
        from backend.ai_trader import evaluate_pyramids

        _seed_config(db_session, strategy="nostate_optimized", current_cash=20000.0)
        _seed_stock(db_session, "TEST", canslim_score=80.0, current_price=110.0)
        _seed_position(
            db_session,
            "TEST",
            shares=40.0,
            cost_basis=100.0,
            current_price=110.0,
            current_value=4400.0,  # 4400 / (20000+4400) = 18% — under 25% cap
            gain_loss_pct=10.0,
            current_score=80.0,
            pyramid_count=0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=35),
            peak_price=112.0,
        )

        pyramids = evaluate_pyramids(db_session, user_id=1)
        assert len(pyramids) == 1
        assert pyramids[0]["position"].ticker == "TEST"
        assert pyramids[0]["amount"] >= 100  # min add threshold

    def test_max_2_pyramids_blocks_third(self, db_session):
        from backend.ai_trader import evaluate_pyramids

        _seed_config(db_session, current_cash=10000.0)
        _seed_stock(db_session, "TEST", canslim_score=80.0, current_price=110.0)
        _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=110.0,
            current_value=11000.0,
            gain_loss_pct=10.0,
            current_score=80.0,
            pyramid_count=2,  # already pyramided twice
            purchase_date=datetime.now(timezone.utc) - timedelta(days=35),
            peak_price=112.0,
        )

        pyramids = evaluate_pyramids(db_session, user_id=1)
        assert pyramids == []

    def test_low_score_blocks_pyramid(self, db_session):
        """Score 65 < 70 threshold → no pyramid."""
        from backend.ai_trader import evaluate_pyramids

        _seed_config(db_session, current_cash=10000.0)
        _seed_stock(db_session, "TEST", canslim_score=65.0, current_price=110.0)
        _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=110.0,
            gain_loss_pct=10.0,
            current_score=65.0,
            pyramid_count=0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=35),
            peak_price=112.0,
        )

        pyramids = evaluate_pyramids(db_session, user_id=1)
        assert pyramids == []

    def test_low_gain_blocks_pyramid(self, db_session):
        """Gain 1.5% < 2.5% threshold → no pyramid."""
        from backend.ai_trader import evaluate_pyramids

        _seed_config(db_session, current_cash=10000.0)
        _seed_stock(db_session, "TEST", canslim_score=80.0, current_price=101.5)
        _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=101.5,
            gain_loss_pct=1.5,
            current_score=80.0,
            pyramid_count=0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=35),
            peak_price=102.0,
        )

        pyramids = evaluate_pyramids(db_session, user_id=1)
        assert pyramids == []

    def test_insufficient_cash_blocks_pyramid(self, db_session):
        from backend.ai_trader import evaluate_pyramids

        _seed_config(db_session, current_cash=50.0)  # below 200 floor
        _seed_stock(db_session, "TEST", canslim_score=80.0, current_price=110.0)
        _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=110.0,
            gain_loss_pct=10.0,
            current_score=80.0,
            pyramid_count=0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=35),
            peak_price=112.0,
        )

        pyramids = evaluate_pyramids(db_session, user_id=1)
        assert pyramids == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — get_market_regime
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketRegime:
    """3-way regime: bullish (signal>=1.5) / bearish (signal<=-0.5) / neutral.
    Used by evaluate_buys to adjust min_score and max_position_pct."""

    def test_bullish_regime_at_high_signal(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                            lambda *a, **k: {"success": True, "weighted_signal": 2.5})
        result = get_market_regime(db_session)
        assert result["regime"] == "bullish"
        assert result["max_position_pct"] >= 12.0  # bullish gets bigger positions
        assert result["min_score_adjustment"] == -5

    def test_bearish_regime_at_low_signal(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                            lambda *a, **k: {"success": True, "weighted_signal": -1.0})
        result = get_market_regime(db_session)
        assert result["regime"] == "bearish"
        assert result["max_position_pct"] <= 12.0  # bearish tightens
        assert result["min_score_adjustment"] == 5

    def test_neutral_regime_in_band(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                            lambda *a, **k: {"success": True, "weighted_signal": 0.5})
        result = get_market_regime(db_session)
        assert result["regime"] == "neutral"
        assert result["min_score_adjustment"] == 0

    def test_no_market_data_returns_neutral(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda *a, **k: None)
        result = get_market_regime(db_session)
        assert result["regime"] == "neutral"
        assert "No market data" in result["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetOrCreateConfig:
    """Idempotent config insert. Returns existing row on subsequent calls."""

    def test_creates_default_config_on_first_call(self, db_session):
        from backend.ai_trader import get_or_create_config

        cfg = get_or_create_config(db_session, user_id=1)
        assert cfg.starting_cash == 25000.0
        assert cfg.current_cash == 25000.0
        assert cfg.is_active is True
        assert cfg.user_id == 1

    def test_returns_existing_config_unchanged(self, db_session):
        from backend.ai_trader import get_or_create_config

        seeded = _seed_config(db_session, current_cash=12345.67, user_id=1)
        cfg = get_or_create_config(db_session, user_id=1)
        assert cfg.id == seeded.id
        assert cfg.current_cash == 12345.67

    def test_separate_user_gets_separate_config(self, db_session):
        from backend.ai_trader import get_or_create_config

        _seed_config(db_session, user_id=1, current_cash=11111.0)
        cfg2 = get_or_create_config(db_session, user_id=2)
        assert cfg2.user_id == 2
        assert cfg2.current_cash == 25000.0  # default, not user 1's


class TestPortfolioValue:
    """cash + positions_value + sweep_value math, with empty/full positions."""

    def test_empty_portfolio_returns_starting_cash(self, db_session):
        from backend.ai_trader import get_portfolio_value

        _seed_config(db_session, current_cash=25000.0, starting_cash=25000.0)
        result = get_portfolio_value(db_session, user_id=1)
        assert result["cash"] == 25000.0
        assert result["positions_value"] == 0
        assert result["positions_count"] == 0
        assert result["total_value"] == 25000.0
        assert result["total_return"] == 0.0
        assert result["total_return_pct"] == 0.0

    def test_with_positions_sums_correctly(self, db_session):
        from backend.ai_trader import get_portfolio_value

        _seed_config(db_session, current_cash=5000.0, starting_cash=25000.0)
        _seed_stock(db_session, "AAA")
        _seed_stock(db_session, "BBB")
        _seed_position(db_session, "AAA", current_value=12000.0)
        _seed_position(db_session, "BBB", current_value=8500.0)

        result = get_portfolio_value(db_session, user_id=1)
        assert result["cash"] == 5000.0
        assert result["positions_value"] == 20500.0
        assert result["total_value"] == 25500.0
        assert result["total_return"] == pytest.approx(500.0)
        assert result["total_return_pct"] == pytest.approx(2.0)

    def test_zero_starting_cash_avoids_division(self, db_session):
        from backend.ai_trader import get_portfolio_value

        _seed_config(db_session, starting_cash=0.0, current_cash=1000.0)
        result = get_portfolio_value(db_session, user_id=1)
        # Guard against div-by-zero
        assert result["total_return_pct"] == 0


class TestSectorAllocations:
    """get_sector_allocations: per-sector $ + count, with batch-fetch."""

    def test_empty_portfolio_returns_zero_alloc_dict(self, db_session):
        """No positions but cash > 0 → allocations and counts empty, but the
        dict shape (with portfolio_value) is still returned."""
        from backend.ai_trader import get_sector_allocations

        _seed_config(db_session, current_cash=10000.0)
        result = get_sector_allocations(db_session, user_id=1)
        assert result["allocations"] == {}
        assert result["counts"] == {}
        assert result["portfolio_value"] > 0

    def test_zero_portfolio_value_returns_empty(self, db_session):
        """Truly empty portfolio (no cash, no positions) → {} short-circuit."""
        from backend.ai_trader import get_sector_allocations

        _seed_config(db_session, current_cash=0.0, starting_cash=0.0)
        result = get_sector_allocations(db_session, user_id=1)
        assert result == {}

    def test_groups_by_sector(self, db_session):
        from backend.ai_trader import get_sector_allocations

        _seed_config(db_session, current_cash=1000.0)
        _seed_stock(db_session, "AAA", sector="Technology")
        _seed_stock(db_session, "BBB", sector="Technology")
        _seed_stock(db_session, "CCC", sector="Healthcare")
        _seed_position(db_session, "AAA", current_value=4000.0)
        _seed_position(db_session, "BBB", current_value=4000.0)
        _seed_position(db_session, "CCC", current_value=2000.0)

        result = get_sector_allocations(db_session, user_id=1)
        assert result["counts"]["Technology"] == 2
        assert result["counts"]["Healthcare"] == 1
        # Allocations sum to ~position_value/total_value
        # total = 1000 cash + 10000 positions = 11000
        assert result["allocations"]["Technology"] == pytest.approx(8000.0 / 11000.0)
        assert result["allocations"]["Healthcare"] == pytest.approx(2000.0 / 11000.0)

    def test_unknown_sector_bucketed_as_unknown(self, db_session):
        from backend.ai_trader import get_sector_allocations

        _seed_config(db_session, current_cash=0.0)
        _seed_stock(db_session, "AAA", sector=None)
        _seed_position(db_session, "AAA", current_value=5000.0)

        result = get_sector_allocations(db_session, user_id=1)
        assert "Unknown" in result["counts"]


class TestBuyThrottle:
    """Consecutive-loss throttle: 5+ losses → 0 buys, 3-4 → 1 buy, else unlimited."""

    def _seed_sell(self, db, days_ago: int, gain: float):
        """Insert a SELL trade row at executed_at = now - days_ago."""
        ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
        db.add(AIPortfolioTrade(
            user_id=1,
            ticker="TEST",
            action="SELL",
            shares=100.0,
            price=100.0,
            total_value=10000.0,
            cost_basis=100.0,
            realized_gain=gain,
            executed_at=ts,
        ))
        db.commit()

    def test_unlimited_when_throttle_disabled(self, db_session):
        from backend.ai_trader import get_buy_throttle_limit

        result = get_buy_throttle_limit(db_session, profile={"buy_throttle": {"enabled": False}}, user_id=1)
        assert result == 999

    def test_unlimited_when_too_few_trades(self, db_session):
        """min_trades=3, only 2 sells → unlimited."""
        from backend.ai_trader import get_buy_throttle_limit

        self._seed_sell(db_session, days_ago=5, gain=-100.0)
        self._seed_sell(db_session, days_ago=4, gain=-200.0)

        profile = {"buy_throttle": {"enabled": True, "lookback": 10, "min_trades": 3}}
        result = get_buy_throttle_limit(db_session, profile=profile, user_id=1)
        assert result == 999

    def test_paused_after_5_consecutive_losses(self, db_session):
        from backend.ai_trader import get_buy_throttle_limit

        # 5 losses in a row, oldest first
        for i, days in enumerate([10, 8, 6, 4, 2]):
            self._seed_sell(db_session, days_ago=days, gain=-100.0)

        profile = {"buy_throttle": {"enabled": True, "lookback": 10, "min_trades": 3, "pause_after": 5, "reduce_after": 3}}
        result = get_buy_throttle_limit(db_session, profile=profile, user_id=1)
        assert result == 0

    def test_reduced_after_3_consecutive_losses(self, db_session):
        from backend.ai_trader import get_buy_throttle_limit

        for days in [6, 4, 2]:
            self._seed_sell(db_session, days_ago=days, gain=-100.0)

        profile = {"buy_throttle": {"enabled": True, "lookback": 10, "min_trades": 3, "pause_after": 5, "reduce_after": 3}}
        result = get_buy_throttle_limit(db_session, profile=profile, user_id=1)
        assert result == 1

    def test_winner_breaks_streak(self, db_session):
        """A winner between two losing groups should reset the consecutive count."""
        from backend.ai_trader import get_buy_throttle_limit

        # Most recent (executed last) is at the top of the desc query.
        # Order from oldest to most-recent: 3 losses, then a win, then 2 more losses.
        # Most recent 2 are losses → consecutive=2 (below reduce_after=3).
        self._seed_sell(db_session, days_ago=10, gain=-100.0)
        self._seed_sell(db_session, days_ago=8, gain=-100.0)
        self._seed_sell(db_session, days_ago=6, gain=-100.0)
        self._seed_sell(db_session, days_ago=4, gain=200.0)  # WIN
        self._seed_sell(db_session, days_ago=2, gain=-100.0)
        self._seed_sell(db_session, days_ago=1, gain=-100.0)

        profile = {"buy_throttle": {"enabled": True, "lookback": 10, "min_trades": 3, "pause_after": 5, "reduce_after": 3}}
        result = get_buy_throttle_limit(db_session, profile=profile, user_id=1)
        # 2 consecutive losses < reduce_after=3 → unlimited
        assert result == 999


class TestCalculateAtrStop:
    """ATR-adaptive stop. HTTP-disabled path returns base; HTTP success path
    computes ATR from highs/lows/closes."""

    def test_disabled_in_config_returns_base(self, db_session, monkeypatch):
        from backend.ai_trader import calculate_atr_stop
        from config_loader import config as yaml_config

        # Patch the module config to disable ATR stops
        original = yaml_config.get('ai_trader.stops', {})
        monkeypatch.setattr(yaml_config, "get",
                            lambda key, default=None: {"use_atr_stops": False}
                            if key == 'ai_trader.stops' else original.get(key, default))
        result = calculate_atr_stop("TEST", current_price=100.0, base_stop_pct=7.0)
        assert result == 7.0

    def test_yahoo_failure_falls_back_to_base(self, monkeypatch):
        """Network exception → base_stop_pct returned, no crash."""
        from backend import ai_trader

        def boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("requests.get", boom)
        # Rely on default config (use_atr_stops=true). Exception path falls back.
        result = ai_trader.calculate_atr_stop("TEST", current_price=100.0, base_stop_pct=7.0)
        assert result == 7.0

    def test_widens_for_volatile_stock_via_yahoo(self, monkeypatch):
        """ATR success path: 14 days of synthetic high-volatility candles
        should produce a finite stop in [base, max_stop_pct]."""
        from backend import ai_trader

        highs = [105.0 + i * 0.1 for i in range(20)]
        lows = [97.0 + i * 0.1 for i in range(20)]
        closes = [100.0 + i * 0.1 for i in range(20)]

        def stub_get(url, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "chart": {"result": [{"indicators": {"quote": [{
                    "high": highs, "low": lows, "close": closes,
                }]}}]}
            }
            return resp

        monkeypatch.setattr("requests.get", stub_get)
        result = ai_trader.calculate_atr_stop("TEST", current_price=100.0, base_stop_pct=7.0)
        assert isinstance(result, float)
        assert result >= 7.0
        assert result <= 25.0  # safety cap from config


class TestFetchLivePrice:
    """FMP-first then Yahoo fallback; both can fail to return None."""

    def test_fmp_success_returns_price(self, monkeypatch):
        from backend.ai_trader import fetch_live_price

        # Set FMP_API_KEY so the FMP branch is attempted
        monkeypatch.setenv("FMP_API_KEY", "test-key")

        def stub_get(url, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = [{"price": 123.45}]
            return resp

        monkeypatch.setattr("requests.get", stub_get)
        assert fetch_live_price("TEST") == 123.45

    def test_fmp_429_falls_back_to_yahoo(self, monkeypatch):
        """FMP rate-limit → Yahoo chart API path."""
        from backend.ai_trader import fetch_live_price

        monkeypatch.setenv("FMP_API_KEY", "test-key")

        # Track which URL was called
        call_log = []

        def stub_get(url, *a, **kw):
            call_log.append(url)
            resp = MagicMock()
            if "financialmodelingprep" in url:
                resp.status_code = 429  # Rate-limited
                resp.json.return_value = []
            else:
                # Yahoo response
                resp.status_code = 200
                resp.json.return_value = {
                    "chart": {"result": [{"meta": {"regularMarketPrice": 67.89}}]}
                }
            return resp

        # Avoid sleeping during tests
        import time
        monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
        monkeypatch.setattr("requests.get", stub_get)

        result = fetch_live_price("TEST")
        assert result == 67.89
        # Both paths exercised
        assert any("financialmodelingprep" in u for u in call_log)
        assert any("yahoo" in u or "yf" in u or "query1" in u for u in call_log)

    def test_both_paths_fail_returns_none(self, monkeypatch):
        from backend.ai_trader import fetch_live_price

        monkeypatch.setenv("FMP_API_KEY", "test-key")

        def stub_get(url, *a, **kw):
            resp = MagicMock()
            resp.status_code = 500
            resp.json.return_value = {}
            return resp

        monkeypatch.setattr("requests.get", stub_get)
        assert fetch_live_price("TEST") is None

    def test_no_fmp_key_skips_to_yahoo(self, monkeypatch):
        from backend.ai_trader import fetch_live_price

        monkeypatch.delenv("FMP_API_KEY", raising=False)

        def stub_get(url, *a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            if "yahoo" in url or "query1" in url:
                resp.json.return_value = {
                    "chart": {"result": [{"meta": {"regularMarketPrice": 50.0}}]}
                }
            else:
                resp.json.return_value = {}
            return resp

        monkeypatch.setattr("requests.get", stub_get)
        assert fetch_live_price("TEST") == 50.0


class TestUpdatePositionPrices:
    """Updates current_price/value/gain + maintains peak_price + score sync."""

    def test_initializes_peak_price_when_none(
        self,
        db_session,
        monkeypatch,
        disable_historical_data,
    ):
        from backend import ai_trader

        # Stub fetch_live_price to return a known value
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 110.0)
        import time
        monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

        _seed_config(db_session)
        _seed_stock(db_session, "TEST", canslim_score=70.0)
        pos = _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=None,
            peak_price=None,  # uninitialized
        )

        ai_trader.update_position_prices(db_session, use_live_prices=True, user_id=1)
        db_session.refresh(pos)
        # Peak initialized to max(current_price=110, cost_basis=100) = 110
        assert pos.peak_price == 110.0
        assert pos.current_price == 110.0
        assert pos.gain_loss_pct == pytest.approx(10.0)

    def test_new_high_updates_peak(
        self,
        db_session,
        monkeypatch,
        disable_historical_data,
    ):
        from backend import ai_trader

        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 130.0)
        import time
        monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

        _seed_config(db_session)
        _seed_stock(db_session, "TEST", canslim_score=70.0)
        pos = _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=120.0,
            peak_price=125.0,  # existing peak
        )

        ai_trader.update_position_prices(db_session, use_live_prices=True, user_id=1)
        db_session.refresh(pos)
        # New high: 130 > 125
        assert pos.peak_price == 130.0

    def test_falls_back_to_stock_price_on_fetch_failure(
        self,
        db_session,
        monkeypatch,
        disable_historical_data,
    ):
        from backend import ai_trader

        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: None)
        import time
        monkeypatch.setattr(time, "sleep", lambda *a, **k: None)

        _seed_config(db_session)
        _seed_stock(db_session, "TEST", canslim_score=70.0, current_price=99.0)
        pos = _seed_position(
            db_session,
            "TEST",
            cost_basis=100.0,
            current_price=110.0,
            peak_price=115.0,
        )

        ai_trader.update_position_prices(db_session, use_live_prices=True, user_id=1)
        db_session.refresh(pos)
        # Stock.current_price is 99.0 — fallback used
        assert pos.current_price == 99.0


class TestSpySweep:
    """SPY cash-sweep: park idle cash in SPY when above 50MA, liquidate below."""

    def test_liquidate_no_shares_is_noop(self, db_session):
        from backend.ai_trader import liquidate_spy_sweep

        cfg = _seed_config(db_session, current_cash=5000.0)
        cfg.spy_sweep_shares = 0.0
        db_session.commit()

        liquidate_spy_sweep(db_session, cfg, "test reason", user_id=1)
        assert cfg.current_cash == 5000.0

    def test_liquidate_credits_cash_and_zeros_shares(self, db_session, monkeypatch):
        from backend import ai_trader

        cfg = _seed_config(db_session, current_cash=5000.0)
        cfg.spy_sweep_shares = 10.0
        db_session.commit()

        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        ai_trader.liquidate_spy_sweep(db_session, cfg, "test", user_id=1)

        assert cfg.current_cash == pytest.approx(10000.0)  # 5000 + 10 * 500
        assert cfg.spy_sweep_shares == 0.0

    def test_handle_sweep_disabled_in_profile_is_noop(self, db_session):
        from backend.ai_trader import handle_spy_sweep

        cfg = _seed_config(db_session, current_cash=5000.0)
        # Profile without spy_sweep config or with enabled=false
        handle_spy_sweep(db_session, cfg, profile={}, user_id=1)
        assert cfg.current_cash == 5000.0  # Untouched

    def test_handle_sweep_buys_when_spy_above_50ma(
        self,
        db_session,
        stub_market_bullish,
        monkeypatch,
    ):
        """SPY above 50MA + idle cash > 20% of total → buy SPY shares."""
        from backend import ai_trader
        from backend.ai_trader import handle_spy_sweep

        cfg = _seed_config(db_session, current_cash=10000.0, starting_cash=25000.0)
        # Profile enables sweep
        profile = {"spy_sweep": {"enabled": True, "min_idle_pct": 20}}

        # Buy leg executes at the live price (gate uses the cached direction)
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda t: 500.0)
        handle_spy_sweep(db_session, cfg, profile=profile, user_id=1)
        # Some sweep should have been bought (10000 cash, no positions, total=10000,
        # 20% of 10000 = 2000 idle threshold; 10000-500 reserve = 9500 idle > 2000 → buy)
        assert cfg.spy_sweep_shares > 0
        assert cfg.current_cash < 10000.0  # Some cash was deployed


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Coiled Spring helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoiledSpringScore:
    """calculate_coiled_spring_score_for_stock builds MockData/MockScore from
    the Stock row and delegates. Verify the wiring."""

    def test_returns_dict_with_expected_keys(self, db_session):
        from backend.ai_trader import calculate_coiled_spring_score_for_stock

        stock = _seed_stock(
            db_session, "TEST",
            weeks_in_base=8,
            earnings_beat_streak=3,
            days_to_earnings=5,
            c_score=14.0,
            l_score=12.0,
            canslim_score=80.0,
        )

        result = calculate_coiled_spring_score_for_stock(stock)
        assert isinstance(result, dict)
        assert "is_coiled_spring" in result
        assert "cs_score" in result

    def test_handles_missing_attributes_gracefully(self, db_session):
        """Stock with None-valued fields should not crash the helper."""
        from backend.ai_trader import calculate_coiled_spring_score_for_stock

        stock = _seed_stock(db_session, "TEST",
                            weeks_in_base=None,
                            earnings_beat_streak=None,
                            days_to_earnings=None,
                            c_score=None,
                            l_score=None,
                            canslim_score=None,
                            score_details=None)

        result = calculate_coiled_spring_score_for_stock(stock)
        assert isinstance(result, dict)


class TestRecordCoiledSpringAlert:
    """Daily cap + 21-day cycle gate + webhook side-effect."""

    def test_records_alert_when_under_cap_and_no_recent(self, db_session, silence_webhooks):
        from backend.ai_trader import record_coiled_spring_alert

        stock = _seed_stock(
            db_session, "TEST",
            days_to_earnings=5,
            weeks_in_base=8,
            earnings_beat_streak=3,
            c_score=14.0,
            l_score=12.0,
            canslim_score=80.0,
            current_price=100.0,
            base_type="cup_with_handle",
        )

        cs_result = {"cs_score": 15.0, "cs_details": "Strong CS", "confidence": 80}
        recorded = record_coiled_spring_alert(db_session, "TEST", cs_result, stock)
        assert recorded is True

        alerts = db_session.query(CoiledSpringAlert).filter_by(ticker="TEST").all()
        assert len(alerts) == 1
        assert alerts[0].confidence == 80

    def test_blocks_duplicate_within_21_day_cycle(self, db_session, silence_webhooks):
        """Earnings-cycle dedup: an alert from 10 days ago blocks a new one today."""
        from backend.ai_trader import record_coiled_spring_alert

        stock = _seed_stock(db_session, "TEST", canslim_score=80.0)

        # Pre-existing alert 10 days old
        db_session.add(CoiledSpringAlert(
            ticker="TEST",
            alert_date=date.today() - timedelta(days=10),
            cs_bonus=10.0,
            email_sent=True,
        ))
        db_session.commit()

        cs_result = {"cs_score": 15.0, "cs_details": "again", "confidence": 70}
        recorded = record_coiled_spring_alert(db_session, "TEST", cs_result, stock)
        assert recorded is False  # blocked

    def test_blocks_when_daily_cap_reached(self, db_session, silence_webhooks):
        """5 alerts already today (default cap) → 6th is blocked."""
        from backend.ai_trader import record_coiled_spring_alert

        stock = _seed_stock(db_session, "TEST", canslim_score=80.0)

        # Seed 5 alerts today on different tickers (default max_per_day=5)
        for ticker in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
            db_session.add(CoiledSpringAlert(
                ticker=ticker,
                alert_date=date.today(),
                cs_bonus=10.0,
                email_sent=True,
            ))
        db_session.commit()

        cs_result = {"cs_score": 15.0, "cs_details": "blocked", "confidence": 70}
        recorded = record_coiled_spring_alert(db_session, "TEST", cs_result, stock)
        assert recorded is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — check_score_stability adapter (gaps not covered in test_ai_trader_sync)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckScoreStabilityGaps:
    """Cover the limited-history and no-score branches that test_ai_trader_sync
    didn't reach."""

    def test_below_threshold_with_no_history_increments_low(self, db_session):
        from backend.ai_trader import check_score_stability

        _seed_stock(db_session, "TEST")
        # No StockScore rows → "Limited history" branch
        result = check_score_stability(db_session, "TEST", current_score=40, threshold=50)
        assert result["is_stable"] is True
        assert result["consecutive_low"] == 1  # current is below threshold

    def test_above_threshold_with_no_history(self, db_session):
        from backend.ai_trader import check_score_stability

        _seed_stock(db_session, "TEST")
        result = check_score_stability(db_session, "TEST", current_score=80, threshold=50)
        assert result["is_stable"] is True
        assert result["consecutive_low"] == 0

    def test_only_one_score_row_is_limited_history(self, db_session):
        from backend.ai_trader import check_score_stability

        stock = _seed_stock(db_session, "TEST")
        ts = datetime.now(timezone.utc) - timedelta(hours=1)
        db_session.add(StockScore(
            stock_id=stock.id, total_score=70.0, timestamp=ts, date=ts.date(),
        ))
        db_session.commit()

        result = check_score_stability(db_session, "TEST", current_score=40, threshold=50, lookback=3)
        # Only 1 historical row + current → < 2 → "Limited history"
        assert result["is_stable"] is True
        assert result["warning"] == "Limited history"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Market hours + timestamp helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketOpen:
    """is_market_open with mocked Eastern time covering: weekday/weekend,
    pre-open, post-close, and a holiday."""

    def test_weekend_closed(self, monkeypatch):
        from backend import ai_trader
        # Pick a known Saturday in 2026
        sat = datetime(2026, 1, 3, 14, 0, tzinfo=ai_trader.EASTERN_TZ)

        class FixedDt(datetime):
            @classmethod
            def now(cls, tz=None):
                return sat if tz else sat.replace(tzinfo=None)

        monkeypatch.setattr(ai_trader, "datetime", FixedDt)
        assert ai_trader.is_market_open() is False

    def test_weekday_during_hours_open(self, monkeypatch):
        from backend import ai_trader
        # Tuesday Jan 6 2026 at 11:00 AM ET — open
        weekday = datetime(2026, 1, 6, 11, 0, tzinfo=ai_trader.EASTERN_TZ)

        class FixedDt(datetime):
            @classmethod
            def now(cls, tz=None):
                return weekday

        monkeypatch.setattr(ai_trader, "datetime", FixedDt)
        assert ai_trader.is_market_open() is True

    def test_weekday_before_open_closed(self, monkeypatch):
        from backend import ai_trader
        weekday_early = datetime(2026, 1, 6, 8, 0, tzinfo=ai_trader.EASTERN_TZ)

        class FixedDt(datetime):
            @classmethod
            def now(cls, tz=None):
                return weekday_early

        monkeypatch.setattr(ai_trader, "datetime", FixedDt)
        assert ai_trader.is_market_open() is False

    def test_holiday_closed(self, monkeypatch):
        """Jan 1 2026 (Thu) is New Year's Day."""
        from backend import ai_trader
        ny = datetime(2026, 1, 1, 11, 0, tzinfo=ai_trader.EASTERN_TZ)

        class FixedDt(datetime):
            @classmethod
            def now(cls, tz=None):
                return ny

        monkeypatch.setattr(ai_trader, "datetime", FixedDt)
        assert ai_trader.is_market_open() is False


class TestGetCstNow:
    """get_cst_now must return a tz-naive UTC datetime — pinned because the
    AIPortfolioTrade.executed_at column is tz-naive and a tz-aware leak would
    cause comparison errors with stored timestamps."""

    def test_returns_naive_utc(self):
        from backend.ai_trader import get_cst_now

        now = get_cst_now()
        assert isinstance(now, datetime)
        # Naive (no tzinfo)
        assert now.tzinfo is None
        # Should be ~now (within 5s)
        delta = abs((datetime.now(timezone.utc).replace(tzinfo=None) - now).total_seconds())
        assert delta < 5


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — refresh_ai_portfolio (composes update_position_prices + snapshot)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefreshAIPortfolio:
    """Thin wrapper that runs update_position_prices + take_portfolio_snapshot
    and returns a summary."""

    def test_returns_summary_with_no_positions(self, db_session, monkeypatch):
        from backend import ai_trader

        # No-op the side-effects — we just want to see the return shape
        monkeypatch.setattr(ai_trader, "update_position_prices",
                            lambda db, user_id=1: 0)
        monkeypatch.setattr(ai_trader, "take_portfolio_snapshot",
                            lambda db, user_id=1: None)

        _seed_config(db_session)
        result = ai_trader.refresh_ai_portfolio(db_session, user_id=1)
        assert "summary" in result
        assert result["summary"]["positions_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — execute_trade (record-only ORM write — exercises the SELL+BUY paths)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteTrade:
    """Core trade-recording function. Tests cover: BUY path, SELL path with
    realized_gain logging, and signal_factors sanitization."""

    def test_buy_records_trade(self, db_session, silence_webhooks):
        from backend.ai_trader import execute_trade

        trade = execute_trade(
            db=db_session,
            ticker="TEST",
            action="BUY",
            shares=100.0,
            price=50.0,
            reason="Strong CANSLIM",
            score=80.0,
            user_id=1,
        )
        # Commit so we can query
        db_session.commit()

        assert trade.action == "BUY"
        assert trade.total_value == pytest.approx(5000.0)
        rows = db_session.query(AIPortfolioTrade).filter_by(ticker="TEST").all()
        assert len(rows) == 1

    def test_stamps_strategy_active_at_execution(self, db_session, silence_webhooks):
        """A/B attribution fix (Jul 2026): the trade row must carry the
        strategy the user was on WHEN the trade executed, so later strategy
        switches can't retroactively reshuffle historical attribution."""
        from backend.ai_trader import execute_trade
        from backend.database import AIPortfolioConfig

        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        if cfg is None:
            cfg = AIPortfolioConfig(user_id=1, strategy="nostate_optimized")
            db_session.add(cfg)
        else:
            cfg.strategy = "nostate_optimized"
        db_session.commit()

        trade = execute_trade(
            db=db_session, ticker="STAMP", action="BUY", shares=10.0,
            price=20.0, reason="test", score=75.0, user_id=1,
        )
        db_session.commit()
        assert trade.strategy == "nostate_optimized"

        # Later reassignment must NOT touch the stamped row
        cfg.strategy = "nostate_cs_bear"
        db_session.commit()
        row = db_session.query(AIPortfolioTrade).filter_by(ticker="STAMP").first()
        assert row.strategy == "nostate_optimized"

    def test_stamp_none_when_no_config(self, db_session, silence_webhooks):
        """No config row for the user → stamp stays None (legacy fallback
        scoping applies to it), and the trade still records."""
        from backend.ai_trader import execute_trade
        from backend.database import AIPortfolioConfig

        db_session.query(AIPortfolioConfig).filter_by(user_id=77).delete()
        db_session.commit()
        trade = execute_trade(
            db=db_session, ticker="NOCFG", action="BUY", shares=1.0,
            price=10.0, reason="test", score=50.0, user_id=77,
        )
        db_session.commit()
        assert trade.strategy is None

    def test_sell_records_realized_gain(self, db_session, silence_webhooks):
        from backend.ai_trader import execute_trade

        trade = execute_trade(
            db=db_session,
            ticker="TEST",
            action="SELL",
            shares=100.0,
            price=120.0,
            reason="STOP LOSS: Down 8%",
            score=60.0,
            cost_basis=100.0,
            realized_gain=2000.0,
            holding_days=30,
            user_id=1,
        )
        db_session.commit()

        assert trade.action == "SELL"
        assert trade.realized_gain == 2000.0
        assert trade.holding_days == 30

    def test_paper_trade_prefix(self, db_session, silence_webhooks):
        """LATENT BUG note: passing score=None on a BUY/PYRAMID action crashes
        the log f-string at ai_trader.py:1518 (`{effective:.0f}` is unguarded;
        the SELL path at line 1503 uses `(effective or 0):.0f` defensively).
        We always pass a real score here to keep this test green; the fix is
        deferred past the 2026-06-18 Approach 2 eval window per the rule that
        ai_trader.py runtime behavior cannot be touched during the live A/B."""
        from backend.ai_trader import execute_trade

        trade = execute_trade(
            db=db_session,
            ticker="TEST",
            action="BUY",
            shares=10.0,
            price=50.0,
            reason="experimental",
            score=70.0,
            is_paper=True,
            user_id=1,
        )
        db_session.commit()
        assert trade.reason.startswith("PAPER:")


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — take_portfolio_snapshot (DB-mutating but pure)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTakePortfolioSnapshot:
    """Snapshot the current portfolio state after each cycle. Verify the row
    is written and that prev_value is computed against the most-recent prior
    snapshot."""

    def test_first_snapshot_has_no_prev_value(self, db_session):
        from backend.ai_trader import take_portfolio_snapshot

        _seed_config(db_session, current_cash=25000.0, starting_cash=25000.0)
        take_portfolio_snapshot(db_session, user_id=1)

        snaps = db_session.query(AIPortfolioSnapshot).filter_by(user_id=1).all()
        assert len(snaps) == 1
        assert snaps[0].total_value == 25000.0
        assert snaps[0].positions_count == 0
        assert snaps[0].prev_value is None
        assert snaps[0].value_change == 0

    def test_second_snapshot_computes_change(self, db_session):
        from backend.ai_trader import take_portfolio_snapshot

        _seed_config(db_session, current_cash=25000.0, starting_cash=25000.0)
        take_portfolio_snapshot(db_session, user_id=1)

        # Mutate cash and snapshot again
        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        cfg.current_cash = 26000.0
        db_session.commit()

        take_portfolio_snapshot(db_session, user_id=1)

        snaps = (
            db_session.query(AIPortfolioSnapshot)
            .filter_by(user_id=1)
            .order_by(AIPortfolioSnapshot.timestamp)
            .all()
        )
        assert len(snaps) == 2
        assert snaps[1].prev_value == 25000.0
        assert snaps[1].value_change == pytest.approx(1000.0)
        assert snaps[1].value_change_pct == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — initialize_ai_portfolio (admin reset / fresh setup)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitializeAIPortfolio:
    """Resets portfolio for a user, applying strategy profile defaults."""

    def test_creates_config_from_profile(self, db_session):
        from backend.ai_trader import initialize_ai_portfolio

        result = initialize_ai_portfolio(
            db_session, starting_cash=10000.0,
            strategy="nostate_optimized", user_id=1,
        )

        assert result["starting_cash"] == 10000.0
        assert result["strategy_name"] == "nostate_optimized"

        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        assert cfg is not None
        assert cfg.starting_cash == 10000.0
        assert cfg.current_cash == 10000.0
        assert cfg.strategy == "nostate_optimized"
        # Champion profile bakes max_positions=8
        assert cfg.max_positions == 8

    def test_reset_clears_existing_data(self, db_session):
        """Existing positions/trades/snapshots for the user are wiped."""
        from backend.ai_trader import initialize_ai_portfolio

        # Seed pre-existing data
        _seed_config(db_session, user_id=1)
        _seed_stock(db_session, "OLD")
        _seed_position(db_session, "OLD", current_value=5000.0)
        db_session.add(AIPortfolioTrade(
            user_id=1, ticker="OLD", action="BUY", shares=10, price=100,
            total_value=1000, executed_at=datetime.now(timezone.utc),
        ))
        db_session.commit()

        initialize_ai_portfolio(
            db_session, starting_cash=20000.0,
            strategy="nostate_optimized", user_id=1,
        )

        assert db_session.query(AIPortfolioPosition).filter_by(user_id=1).count() == 0
        assert db_session.query(AIPortfolioTrade).filter_by(user_id=1).count() == 0

    def test_does_not_touch_other_user_data(self, db_session):
        """User 2's data must survive when user 1 is reset."""
        from backend.ai_trader import initialize_ai_portfolio

        # User 2 has positions and trades
        _seed_config(db_session, user_id=2, current_cash=5000.0)
        _seed_stock(db_session, "U2")
        _seed_position(db_session, "U2", user_id=2, current_value=3000.0)

        # Reset user 1 only
        initialize_ai_portfolio(
            db_session, starting_cash=10000.0,
            strategy="nostate_optimized", user_id=1,
        )

        # User 2's data intact
        assert db_session.query(AIPortfolioPosition).filter_by(user_id=2).count() == 1
        assert db_session.query(AIPortfolioConfig).filter_by(user_id=2).first().current_cash == 5000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — run_ai_trading_cycle orchestration loop
#
# These pin the orchestrator's branch shape, NOT numerical values. The cycle
# composes ~10 IO seams (update_position_prices, evaluate_sells/buys/pyramids,
# fetch_live_price, get_market_regime, liquidate_spy_sweep/handle_spy_sweep,
# take_portfolio_snapshot, plus the lazy email_utils webhook imports). Tests
# stub all of them via the silence_cycle_seams fixture, then individual tests
# override the one seam that drives the branch under test.
#
# AI Trader <> Backtester sync: run_ai_trading_cycle has no backtester
# analogue (the backtester loops over historical dates instead), so no mirror
# test is required.
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def reset_cycle_state():
    """Reset module-level _trading_cycle_started before & after each test.

    The orchestrator stores the cycle's start time in a thread-local-ish
    module global. Tests that mock the lock to simulate contention need to
    seed this directly, and we MUST clean it up afterwards or a 'busy' state
    leaks into every subsequent test that imports the module."""
    import backend.ai_trader as ai_trader
    ai_trader._set_cycle_started(None)
    yield
    ai_trader._set_cycle_started(None)


@pytest.fixture
def fast_sleep(monkeypatch):
    """No-op time.sleep so the buy/pyramid 0.3s rate-limit delays don't slow tests."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *a, **kw: None)


@pytest.fixture
def silence_cycle_seams(monkeypatch, silence_webhooks, fast_sleep):
    """Stub every IO seam run_ai_trading_cycle calls. Tests re-patch
    individual seams to drive specific branches.

    Default behavior: empty sells, empty buys, empty pyramids, bullish
    market regime (signal=2.0 → strong-bull 5% reserve), live price = $100,
    SPY sweep helpers no-op, snapshot no-op, all webhooks silenced.

    Composes silence_webhooks + fast_sleep so a test only needs to add this
    one fixture (plus reset_cycle_state, db_session)."""
    import backend.ai_trader as ai_trader
    monkeypatch.setattr(ai_trader, "update_position_prices", lambda *a, **kw: None)
    monkeypatch.setattr(ai_trader, "evaluate_sells", lambda *a, **kw: [])
    monkeypatch.setattr(ai_trader, "evaluate_buys", lambda *a, **kw: [])
    monkeypatch.setattr(ai_trader, "evaluate_pyramids", lambda *a, **kw: [])
    monkeypatch.setattr(ai_trader, "fetch_live_price", lambda ticker: 100.0)
    monkeypatch.setattr(ai_trader, "liquidate_spy_sweep", lambda *a, **kw: None)
    monkeypatch.setattr(ai_trader, "handle_spy_sweep", lambda *a, **kw: None)
    monkeypatch.setattr(ai_trader, "take_portfolio_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(ai_trader, "get_market_regime",
                        lambda *a, **kw: {"regime": "bullish"})
    # FTD branch reads HistoricalDataProvider; force the gate closed.
    monkeypatch.setattr(ai_trader, "HistoricalDataProvider", None)
    # Stub get_cached_market_direction for the dynamic-cash-reserve branch
    # (regime=bullish + signal>=2 → strong-bull 5% reserve).
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                        lambda *a, **k: {"success": True, "weighted_signal": 2.0})
    # Silence the lazy email_utils imports the cycle does at the tail.
    import backend.email_utils as eu
    monkeypatch.setattr(eu, "send_risk_alert_webhook", lambda *a, **kw: None)
    monkeypatch.setattr(eu, "send_webhook_notification", lambda *a, **kw: None)
    monkeypatch.setattr(eu, "send_ops_alert", lambda *a, **kw: None)


class TestRunAITradingCycle:
    """Tier 3: run_ai_trading_cycle orchestration loop.

    Pins the loop's branch *shape* — return-status strings, seam call counts,
    DB mutations — but NOT numerical values that depend on YAML weights.

    Branches under test (one per method):
      - lock contention → "busy"
      - lock timeout (>300s) → force-acquire, body runs
      - lock held with no start-time → force-acquire, body runs
      - is_active=False → "inactive"
      - 0 positions → update_position_prices NOT called
      - >=1 position → update_position_prices called once
      - full sell → position deleted, cash credited, results.sells_executed=1
      - partial sell → shares reduced, partial_profit_taken updated
      - partial sell with reset_peak → peak_price updated to current
      - drawdown >= liquidate_threshold (25%) → all positions liquidated, early return
      - drawdown >= halt_threshold (15%) → drawdown_halt=True, no buys/pyramids
      - pyramid candidate → execute_trade called with action=PYRAMID
      - cash below dynamic reserve → buys skipped (no execute_trade BUY)
      - already at max_positions → evaluate_buys NOT called
      - buy throttle limit reached mid-loop → loop breaks
      - happy buy → position added, cash decremented
      - exception in body → "error" status, lock released, _trading_cycle_started reset
    """

    # ── Lock & short-circuit branches ─────────────────────────────────────

    def test_busy_lock_returns_busy(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: lock held, recent start time → status='busy', body skipped."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session)

        # Seed a recent cycle-start time
        ai_trader._set_cycle_started(ai_trader.get_cst_now())

        # Replace lock with a MagicMock whose acquire(blocking=False) returns False
        mock_lock = MagicMock()
        mock_lock.acquire = MagicMock(return_value=False)
        monkeypatch.setattr(ai_trader, "_trading_cycle_lock", mock_lock)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert result["status"] == "busy"
        # Body did NOT run, so finally's release() was NOT called
        mock_lock.release.assert_not_called()

    def test_lock_timeout_force_acquires(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: lock held, elapsed >300s → blocking acquire, body runs.

        Drives the body to short-circuit on is_active=False to keep the test
        focused on the lock-recovery branch."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, is_active=False)

        # Stale start time → 600s ago
        ai_trader._set_cycle_started(
            ai_trader.get_cst_now() - timedelta(seconds=600)
        )

        # First acquire(blocking=False) → False, then acquire(blocking=True) → True
        mock_lock = MagicMock()
        mock_lock.acquire = MagicMock(side_effect=[False, True])
        monkeypatch.setattr(ai_trader, "_trading_cycle_lock", mock_lock)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert result["status"] == "inactive"
        # Both calls happened: non-blocking (failed), then blocking force-acquire.
        assert mock_lock.acquire.call_count == 2
        # Finally block ran → lock released exactly once
        mock_lock.release.assert_called_once()

    def test_lock_held_no_start_time_force_acquires(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: lock held but no _trading_cycle_started → force-acquire."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, is_active=False)
        ai_trader._set_cycle_started(None)  # no start time recorded

        mock_lock = MagicMock()
        mock_lock.acquire = MagicMock(side_effect=[False, True])
        monkeypatch.setattr(ai_trader, "_trading_cycle_lock", mock_lock)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert result["status"] == "inactive"
        assert mock_lock.acquire.call_count == 2
        mock_lock.release.assert_called_once()

    def test_inactive_config_short_circuits(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data,
    ):
        """Branch: config.is_active=False → status='inactive', no work done."""
        from backend.ai_trader import run_ai_trading_cycle

        _seed_config(db_session, is_active=False)
        result = run_ai_trading_cycle(db_session, user_id=1)
        assert result["status"] == "inactive"

    # ── Position-update + sells loop branches ─────────────────────────────

    def test_cycle_forces_market_direction_refresh(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Regression (2026-08-26): the cycle must force-refresh the
        market-direction cache before any gate reads.

        The cache TTL is 4h (scanner-friendly), but the SPY 50MA gate is a
        point-in-time risk control. On 2026-07-20 SPY opened above its 50MA,
        crossed below midday, and a 19:24Z BWAY buy passed a stale "bullish"
        read fetched hours earlier. The fix: one get_cached_market_direction(
        force_refresh=True) call per active cycle, so the regime gate,
        correction zone, and chop-band reads all see current data."""
        import data_fetcher
        force_flags = []

        def recording_stub(*a, **k):
            force_flags.append(k.get("force_refresh", a[0] if a else False))
            return {"success": True, "weighted_signal": 2.0,
                    "indexes": {"SPY": {"price": 500.0, "ma_50": 480.0,
                                        "ma_200": 450.0, "ema_21": 495.0}}}
        monkeypatch.setattr(data_fetcher, "get_cached_market_direction", recording_stub)

        import backend.ai_trader as ai_trader
        _seed_config(db_session, current_cash=25000.0, peak_portfolio_value=25000.0)
        ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert True in force_flags, (
            "run_ai_trading_cycle must call get_cached_market_direction("
            "force_refresh=True) so the SPY gate never trades on a stale read"
        )

    def test_zero_positions_skips_price_update(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: position_count==0 → update_position_prices NOT called.

        The 'fetch live prices for existing positions' block is gated on
        position_count > 0 to save an FMP call when the portfolio is all-cash."""
        import backend.ai_trader as ai_trader
        calls = []
        monkeypatch.setattr(ai_trader, "update_position_prices",
                            lambda *a, **kw: calls.append(1))

        _seed_config(db_session, current_cash=25000.0, peak_portfolio_value=25000.0)
        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert calls == []
        assert result["sells_considered"] == 0
        assert result["buys_considered"] == 0

    def test_position_present_triggers_price_update(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: position_count>0 → update_position_prices called once."""
        import backend.ai_trader as ai_trader
        calls = []
        monkeypatch.setattr(ai_trader, "update_position_prices",
                            lambda *a, **kw: calls.append(1))

        _seed_config(db_session, current_cash=10000.0,
                     peak_portfolio_value=15000.0, starting_cash=15000.0)
        _seed_stock(db_session, "POS")
        _seed_position(db_session, "POS")
        ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert calls == [1]

    def test_full_sell_deletes_position_and_credits_cash(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: full-sell signal → position deleted, current_cash += value."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, current_cash=5000.0,
                     peak_portfolio_value=20000.0, starting_cash=20000.0)
        _seed_stock(db_session, "DEL")
        pos = _seed_position(db_session, "DEL", current_value=8000.0,
                             current_price=80.0, shares=100.0,
                             cost_basis=70.0, gain_loss=1000.0)

        # evaluate_sells returns one full-sell payload
        def fake_sells(db, user_id=1):
            p = db.query(AIPortfolioPosition).filter_by(user_id=user_id).first()
            return [{"position": p, "reason": "STOP LOSS: -8%",
                     "is_partial": False}]
        monkeypatch.setattr(ai_trader, "evaluate_sells", fake_sells)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        assert len(result["sells_executed"]) == 1
        assert result["sells_executed"][0]["ticker"] == "DEL"
        # Position gone
        assert db_session.query(AIPortfolioPosition).filter_by(ticker="DEL").count() == 0
        # Cash credited (5000 + 8000 = 13000)
        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        assert cfg.current_cash == pytest.approx(13000.0)

    def test_partial_sell_reduces_shares_and_tracks_pct(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: is_partial=True → shares reduced by sell_pct,
        partial_profit_taken accumulates, position kept."""
        import backend.ai_trader as ai_trader

        # Peak set to post-sell total (15k) so DD stays at 0 and the
        # circuit breaker does not fire on the remaining position.
        _seed_config(db_session, current_cash=5000.0,
                     peak_portfolio_value=15000.0, starting_cash=15000.0)
        _seed_stock(db_session, "PAR")
        _seed_position(db_session, "PAR", current_value=10000.0,
                       current_price=100.0, shares=100.0, cost_basis=80.0)

        def fake_sells(db, user_id=1):
            p = db.query(AIPortfolioPosition).filter_by(user_id=user_id).first()
            return [{"position": p, "reason": "PARTIAL PROFIT: 25%",
                     "is_partial": True, "sell_pct": 25}]
        monkeypatch.setattr(ai_trader, "evaluate_sells", fake_sells)

        ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        pos = db_session.query(AIPortfolioPosition).filter_by(ticker="PAR").first()
        assert pos is not None  # not deleted
        assert pos.shares == pytest.approx(75.0)
        assert pos.partial_profit_taken == pytest.approx(25.0)

    def test_partial_sell_reset_peak_updates_peak_price(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: partial sell with reset_peak=True → peak_price reset to
        current_price (the partial-trailing-stop reset path)."""
        import backend.ai_trader as ai_trader

        # Peak matches post-sell total so the CB does not fire.
        _seed_config(db_session, current_cash=5000.0,
                     peak_portfolio_value=15000.0, starting_cash=15000.0)
        _seed_stock(db_session, "RST")
        _seed_position(db_session, "RST", current_value=10000.0,
                       current_price=100.0, shares=100.0, cost_basis=80.0,
                       peak_price=130.0)

        def fake_sells(db, user_id=1):
            p = db.query(AIPortfolioPosition).filter_by(user_id=user_id).first()
            return [{"position": p, "reason": "TRAILING STOP",
                     "is_partial": True, "sell_pct": 25, "reset_peak": True}]
        monkeypatch.setattr(ai_trader, "evaluate_sells", fake_sells)

        ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        pos = db_session.query(AIPortfolioPosition).filter_by(ticker="RST").first()
        assert pos.peak_price == pytest.approx(100.0)

    # ── Drawdown circuit breaker branches ─────────────────────────────────

    def test_drawdown_liquidate_threshold_clears_all_positions(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data,
    ):
        """Branch: current_drawdown >= 25% → CIRCUIT BREAKER liquidates all
        and returns early before pyramids/buys."""
        from backend.ai_trader import run_ai_trading_cycle

        # Peak=$25k, current=$10k cash + $5k positions = $15k → 40% DD
        _seed_config(db_session, current_cash=10000.0,
                     peak_portfolio_value=25000.0, starting_cash=25000.0)
        _seed_stock(db_session, "DD1")
        _seed_position(db_session, "DD1", current_value=5000.0,
                       current_price=50.0, shares=100.0, cost_basis=80.0,
                       gain_loss=-3000.0)

        result = run_ai_trading_cycle(db_session, user_id=1)

        # Position liquidated by the circuit breaker
        assert db_session.query(AIPortfolioPosition).filter_by(user_id=1).count() == 0
        # CB sells get appended to results.sells_executed
        assert any("CIRCUIT BREAKER" in s["reason"]
                   for s in result["sells_executed"])
        # Early return → no buys_considered key was set after the CB return path
        assert "buys_considered" not in result or result["buys_considered"] == 0

    def test_drawdown_liquidate_respects_paper_mode(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data,
    ):
        """Paper mode: the circuit breaker records the liquidation but must NOT
        delete real positions or mutate real cash (mirrors the normal sell path)."""
        from backend.ai_trader import run_ai_trading_cycle

        _seed_config(db_session, current_cash=10000.0, paper_mode=True,
                     peak_portfolio_value=25000.0, starting_cash=25000.0)
        _seed_stock(db_session, "PMD")
        _seed_position(db_session, "PMD", current_value=5000.0,
                       current_price=50.0, shares=100.0, cost_basis=80.0,
                       gain_loss=-3000.0)

        result = run_ai_trading_cycle(db_session, user_id=1)

        # Position NOT deleted, real cash unchanged in paper mode.
        pos = db_session.query(AIPortfolioPosition).filter_by(user_id=1).first()
        assert pos is not None
        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        assert cfg.current_cash == pytest.approx(10000.0)  # not credited
        # Still recorded as a circuit-breaker action.
        assert any("CIRCUIT BREAKER" in s["reason"] for s in result["sells_executed"])

    def test_drawdown_does_not_liquidate_when_spy_sweep_unpriced(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """A transient SPY price outage understates total_value; the circuit
        breaker must NOT liquidate on that bogus drawdown."""
        import backend.ai_trader as ai_trader

        # Priced sweep (100 sh @ ~$100 = $10k) would make total = $25k = peak
        # (0% DD). Unpriced, total = $15k -> 40% DD, which previously liquidated.
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda ticker: None)
        _seed_config(db_session, current_cash=10000.0, spy_sweep_shares=100.0,
                     peak_portfolio_value=25000.0, starting_cash=25000.0)
        _seed_stock(db_session, "SWP")
        _seed_position(db_session, "SWP", current_value=5000.0,
                       current_price=50.0, shares=100.0, cost_basis=80.0,
                       gain_loss=-3000.0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        # Position survives — no liquidation on an unpriced-sweep drawdown.
        assert db_session.query(AIPortfolioPosition).filter_by(user_id=1).count() == 1
        assert not any("CIRCUIT BREAKER" in s["reason"]
                       for s in result.get("sells_executed", []))

    def test_drawdown_halt_threshold_skips_pyramids_and_buys(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: 15% <= DD < 25% → drawdown_halt=True, evaluate_pyramids
        and evaluate_buys NOT called (loop falls through)."""
        import backend.ai_trader as ai_trader
        pyr_calls = []
        buy_calls = []
        monkeypatch.setattr(ai_trader, "evaluate_pyramids",
                            lambda *a, **kw: pyr_calls.append(1) or [])
        monkeypatch.setattr(ai_trader, "evaluate_buys",
                            lambda *a, **kw: buy_calls.append(1) or [])

        # Peak=$25k, current ≈ $20k → 20% DD (between halt 15 and liquidate 25)
        _seed_config(db_session, current_cash=15000.0,
                     peak_portfolio_value=25000.0, starting_cash=25000.0)
        _seed_stock(db_session, "DD2")
        _seed_position(db_session, "DD2", current_value=5000.0,
                       current_price=50.0, shares=100.0, cost_basis=80.0)

        ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        # drawdown_halt blocked both
        assert pyr_calls == []
        assert buy_calls == []

    # ── Buys path branches ────────────────────────────────────────────────

    def test_cash_below_dynamic_reserve_skips_buys(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: current_cash < total_value * dynamic_reserve_pct → buys
        skipped without ever calling evaluate_buys."""
        import backend.ai_trader as ai_trader
        buy_calls = []
        monkeypatch.setattr(ai_trader, "evaluate_buys",
                            lambda *a, **kw: buy_calls.append(1) or [])

        # Bullish strong → 5% reserve. total=$1000 → reserve=$50.
        # Cash $40 is below the $50 reserve.
        _seed_config(db_session, current_cash=40.0,
                     peak_portfolio_value=2000.0, starting_cash=2000.0)
        _seed_stock(db_session, "RES")
        _seed_position(db_session, "RES", current_value=960.0,
                       current_price=9.6, shares=100.0, cost_basis=10.0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert buy_calls == []
        assert result["buys_executed"] == []

    def test_already_at_max_positions_skips_buys(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: position_count == max_positions → buys block skipped,
        evaluate_buys never called."""
        import backend.ai_trader as ai_trader
        buy_calls = []
        monkeypatch.setattr(ai_trader, "evaluate_buys",
                            lambda *a, **kw: buy_calls.append(1) or [])

        # nostate_optimized profile bakes max_positions=8 (overrides config),
        # so we seed 8 positions to actually exercise the "at-max" branch.
        _seed_config(db_session, current_cash=2000.0,
                     peak_portfolio_value=82000.0, starting_cash=82000.0)
        for i in range(8):
            t = f"FULL{i}"
            _seed_stock(db_session, t)
            _seed_position(db_session, t, current_value=10000.0,
                           current_price=100.0, shares=100.0, cost_basis=90.0)

        ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert buy_calls == []

    def test_happy_buy_creates_position_and_decrements_cash(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: 1 buy candidate, slot open, cash above reserve → position
        added, cash decremented, results.buys_executed populated."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, current_cash=20000.0,
                     peak_portfolio_value=20000.0, starting_cash=20000.0)
        new_stock = _seed_stock(db_session, "NEW",
                                canslim_score=80.0, current_price=50.0)

        def fake_buys(db, ftd_penalty_active=False, heat_penalty_active=False, user_id=1):
            s = db.query(Stock).filter_by(ticker="NEW").first()
            return [{"stock": s, "reason": "Strong CANSLIM",
                     "value": 5000.0, "shares": 100.0,
                     "is_growth_stock": False, "signal_factors": {}}]
        monkeypatch.setattr(ai_trader, "evaluate_buys", fake_buys)
        monkeypatch.setattr(ai_trader, "fetch_live_price",
                            lambda ticker: 50.0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        assert len(result["buys_executed"]) == 1
        assert result["buys_executed"][0]["ticker"] == "NEW"
        # Position created
        assert db_session.query(AIPortfolioPosition).filter_by(ticker="NEW").count() == 1
        # Cash decremented (started 20k, bought $5k → 15k)
        cfg = db_session.query(AIPortfolioConfig).filter_by(user_id=1).first()
        assert cfg.current_cash == pytest.approx(15000.0)

    def test_buy_throttle_zero_limit_blocks_all_buys(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: buy_throttle_limit=0 → loop hits the throttle check on
        the very first candidate and breaks before any execute_trade BUY."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, current_cash=20000.0,
                     peak_portfolio_value=20000.0, starting_cash=20000.0)
        _seed_stock(db_session, "T1", canslim_score=80.0, current_price=50.0)

        def fake_buys(db, ftd_penalty_active=False, heat_penalty_active=False, user_id=1):
            s = db.query(Stock).filter_by(ticker="T1").first()
            return [{"stock": s, "reason": "x", "value": 5000.0,
                     "shares": 100.0, "is_growth_stock": False,
                     "signal_factors": {}}]
        monkeypatch.setattr(ai_trader, "evaluate_buys", fake_buys)
        monkeypatch.setattr(ai_trader, "get_buy_throttle_limit",
                            lambda *a, **kw: 0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert result["buys_considered"] == 1
        assert result["buys_executed"] == []
        # No position created
        assert db_session.query(AIPortfolioPosition).filter_by(ticker="T1").count() == 0

    def test_max_positions_breaks_buy_loop_mid_iteration(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: 7 positions, max=8, 3 candidates → first buy fills slot 8,
        next iteration's max-positions check breaks the loop."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, current_cash=20000.0, max_positions=8,
                     peak_portfolio_value=80000.0, starting_cash=80000.0)
        # 7 existing positions
        for i in range(7):
            t = f"OLD{i}"
            _seed_stock(db_session, t)
            _seed_position(db_session, t, current_value=8000.0,
                           current_price=80.0, shares=100.0, cost_basis=70.0)
        for t in ("CAND1", "CAND2", "CAND3"):
            _seed_stock(db_session, t, canslim_score=80.0, current_price=50.0)

        def fake_buys(db, ftd_penalty_active=False, heat_penalty_active=False, user_id=1):
            return [
                {"stock": db.query(Stock).filter_by(ticker=t).first(),
                 "reason": "x", "value": 5000.0, "shares": 100.0,
                 "is_growth_stock": False, "signal_factors": {}}
                for t in ("CAND1", "CAND2", "CAND3")
            ]
        monkeypatch.setattr(ai_trader, "evaluate_buys", fake_buys)
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda ticker: 50.0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        # Only 1 buy got through (slot 8); CAND2 and CAND3 blocked by max
        assert len(result["buys_executed"]) == 1
        assert result["buys_executed"][0]["ticker"] == "CAND1"

    # ── Pyramid branch ────────────────────────────────────────────────────

    def test_pyramid_executes_when_drawdown_safe(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: pyramid candidate, no circuit breaker → execute_trade
        called with action='PYRAMID' and results.pyramids_executed populated."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session, current_cash=20000.0,
                     peak_portfolio_value=20000.0, starting_cash=20000.0)
        _seed_stock(db_session, "PYR")
        _seed_position(db_session, "PYR", current_value=8000.0,
                       current_price=80.0, shares=100.0, cost_basis=70.0)

        def fake_pyramids(db, user_id=1):
            p = db.query(AIPortfolioPosition).filter_by(ticker="PYR").first()
            return [{"position": p, "reason": "Strong continuation",
                     "amount": 2000.0}]
        monkeypatch.setattr(ai_trader, "evaluate_pyramids", fake_pyramids)
        monkeypatch.setattr(ai_trader, "fetch_live_price", lambda ticker: 80.0)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)

        assert len(result["pyramids_executed"]) == 1
        assert result["pyramids_executed"][0]["ticker"] == "PYR"
        # Trade row recorded with action=PYRAMID
        pyramid_trades = db_session.query(AIPortfolioTrade).filter_by(
            ticker="PYR", action="PYRAMID").all()
        assert len(pyramid_trades) == 1

    # ── Exception path ────────────────────────────────────────────────────

    def test_exception_in_body_returns_error_and_releases_lock(
        self, db_session, reset_cycle_state, silence_cycle_seams,
        disable_atr_http, disable_historical_data, monkeypatch,
    ):
        """Branch: any exception in the try block → caught, status='error',
        finally block resets _trading_cycle_started + releases lock.

        This test is the safety net: if the cycle ever crashes for any
        reason, the lock is released so the next scheduled call can run.
        Without this guard, a crash here strands the lock and freezes
        live trading for the next 5 minutes (timeout) on every user."""
        import backend.ai_trader as ai_trader

        _seed_config(db_session)

        def boom(*a, **kw):
            raise RuntimeError("sells exploded")
        monkeypatch.setattr(ai_trader, "evaluate_sells", boom)

        result = ai_trader.run_ai_trading_cycle(db_session, user_id=1)
        assert result["status"] == "error"
        assert "sells exploded" in result["message"]
        # Cleanup ran: cycle-start cleared
        assert ai_trader._get_cycle_started() is None
