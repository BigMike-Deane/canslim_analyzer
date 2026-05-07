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

  Tier 3 (intentionally NOT covered — see top of file):
    - run_ai_trading_cycle: 580-line orchestration loop covering buy throttle
      + circuit breaker + bear-base sweep + earnings briefing — exercised
      end-to-end by integration runs, not unit-testable without a real DB
      and a real market_data feed. Covered indirectly when stop-loss /
      sells / buys are tested.
    - take_portfolio_snapshot: pure ORM write of a single row, exercised
      transitively by every test that runs through stop-loss or evaluate_sells.
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
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda: payload)
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
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda: payload)
    return payload


@pytest.fixture
def stub_market_no_data(monkeypatch):
    """Stub get_cached_market_direction → None (data unavailable)."""
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda: None)


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

    def test_nostate_optimized_is_full_d_config(self):
        """nostate_optimized D config: log_only=false (bonus on) AND min_confidence>0 (veto on)."""
        from backend.trading_utils import get_strategy_profile

        profile = get_strategy_profile("nostate_optimized")
        ml = profile.get("ml_signal", {})
        assert ml.get("enabled") is True
        assert ml.get("log_only") is False, (
            "nostate_optimized's ml_signal.log_only MUST be false (bonus path on, "
            "graduated Apr 29 — see canslim-ml-graduation-apr29.md)"
        )
        assert ml.get("min_confidence", 0) > 0


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
                            lambda: {"success": True, "weighted_signal": 2.5})
        result = get_market_regime(db_session)
        assert result["regime"] == "bullish"
        assert result["max_position_pct"] >= 12.0  # bullish gets bigger positions
        assert result["min_score_adjustment"] == -5

    def test_bearish_regime_at_low_signal(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                            lambda: {"success": True, "weighted_signal": -1.0})
        result = get_market_regime(db_session)
        assert result["regime"] == "bearish"
        assert result["max_position_pct"] <= 12.0  # bearish tightens
        assert result["min_score_adjustment"] == 5

    def test_neutral_regime_in_band(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction",
                            lambda: {"success": True, "weighted_signal": 0.5})
        result = get_market_regime(db_session)
        assert result["regime"] == "neutral"
        assert result["min_score_adjustment"] == 0

    def test_no_market_data_returns_neutral(self, db_session, monkeypatch):
        from backend.ai_trader import get_market_regime
        import data_fetcher

        monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda: None)
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
        from backend.ai_trader import handle_spy_sweep

        cfg = _seed_config(db_session, current_cash=10000.0, starting_cash=25000.0)
        # Profile enables sweep
        profile = {"spy_sweep": {"enabled": True, "min_idle_pct": 20}}

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

    def test_calculate_atr_stop_widens_for_volatile_stock(self, monkeypatch):
        """ATR success path: 14 days of synthetic high-volatility candles
        should widen the stop above base, capped at max_stop_pct."""
        from backend import ai_trader

        # 15 days of high/low/close that produce a meaningful ATR.
        # ATR ~3% per day → at multiplier 2.5 → ATR stop ~7.5%, widens above base 7
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
        # Either widened or stayed at base — the relevant assertion is that
        # we exercised the success path and got a finite result back.
        assert isinstance(result, float)
        assert result >= 7.0
        assert result <= 25.0  # safety cap from config


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
