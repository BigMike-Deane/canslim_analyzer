"""Shared-fixture parity harness: ai_trader.evaluate_sells ≡ backtester._evaluate_sells.

June-19 exit-parity item 3. `test_ai_trader_coverage.py` (live) and
`test_backtester_trading_parity.py` (backtester) test the SAME scenarios under
SEPARATE harnesses — but those two scenario sets can drift independently and
nobody would notice. This file closes that gap: each scenario builds its inputs
ONCE and drives BOTH engines, asserting the same sell decision. A future change
that drifts one engine from the other becomes a test failure here, not a silent
live loss (and not a corrupted backtest-replay eval signal).

Method: feed matched position state (gain, score, peak, days-held, ATR-stop) to
  - live:       evaluate_sells(db, user_id)        → [{position, reason, ...}]
  - backtester: engine._evaluate_sells(date, scores) → [SimulatedTrade]
normalize each side's output to a decision CATEGORY per ticker, and compare.

The `d1_new_position_fast_fall` scenario is a KNOWN divergence (the missing
new_position_guard in evaluate_sells, D1). It is marked xfail(strict): the
engines disagree today and will AGREE once the June-19 guard port lands — at
which point the strict-xfail flips to a hard failure, forcing its removal. So
this harness is also a tripwire for the D1 fix.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import Base, Stock, AIPortfolioConfig, AIPortfolioPosition

STRATEGY = "nostate_optimized"
COST_BASIS = 100.0


# ── decision normalization ───────────────────────────────────────────────────

def _categorize(reason: str) -> str:
    r = (reason or "").upper()
    if "PARTIAL TRAILING" in r: return "PARTIAL_TRAILING"
    if "TRAILING STOP" in r:    return "TRAILING"
    if "STOP LOSS" in r:        return "STOP_LOSS"
    if "SCORE CRASH" in r:      return "SCORE_CRASH"
    if "PARTIAL PROFIT" in r:   return "PARTIAL_PROFIT"
    if "TAKE PROFIT" in r:      return "TAKE_PROFIT"
    if "PROTECT GAINS" in r:    return "PROTECT_GAINS"
    if "WEAK POSITION" in r:    return "WEAK"
    if "PRE-EARNINGS" in r:     return "PRE_EARNINGS"
    return "OTHER"


# ── scenarios ─────────────────────────────────────────────────────────────────

# gain_pct → current price; peak_price for trailing; atr_stop_pct = the ATR-widened
# stop BOTH engines compute (before any guard); expected = agreeing category.
SCENARIOS = {
    # Clear stop loss: -8% under a 7% stop, healthy score, no peak gain.
    "stop_loss":  dict(gain_pct=-8.0,  score=75, peak=100.0, days_held=30, atr_stop=7.0,  expected="STOP_LOSS"),
    # Clear hold: small gain, healthy score, peak == current (no trailing).
    "hold":       dict(gain_pct=3.0,   score=75, peak=103.0, days_held=30, atr_stop=7.0,  expected="HOLD"),
    # Trailing stop: +10% now, peaked at 140 (champion profile: peak-gain 40% →
    # 18% trail; drop_from_peak 21.4% fires). NOTE: nostate_optimized overrides
    # default.yaml trailing bands — both engines read the same profile, so parity holds.
    "trailing":   dict(gain_pct=10.0,  score=75, peak=140.0, days_held=30, atr_stop=7.0,  expected="TRAILING"),
}


@pytest.fixture
def db_session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def bt_engine():
    """BacktestEngine with a bullish, ATR-off mocked data_provider (mirrors the
    live stubs). Per-scenario price/ATR are set in the runner."""
    from backend.backtester import BacktestEngine
    from backend.database import BacktestRun
    session = MagicMock()
    session.get.return_value = BacktestRun(
        id=1, name="parity", start_date=date.today() - timedelta(days=30),
        end_date=date.today(), starting_cash=25000.0, stock_universe="all",
        max_positions=8, min_score_to_buy=72, sell_score_threshold=45,
        stop_loss_pct=7.0, strategy=STRATEGY, status="pending",
    )
    session.query.return_value.all.return_value = []
    eng = BacktestEngine(session, 1)
    eng.data_provider = MagicMock()
    eng.data_provider.get_market_direction.return_value = {
        "weighted_signal": 2.0,
        "spy": {"price": 500.0, "ma_50": 480.0, "ma_200": 450.0, "ema_21": 495.0},
    }
    eng.data_provider.get_atr.return_value = 0.0
    eng.data_provider.get_vix_proxy.return_value = 18.0
    eng.data_provider.get_volume_ratio.return_value = 1.0
    eng.static_data = {}
    return eng


def _run_live(db, monkeypatch, sc, ticker="PAR"):
    """Drive evaluate_sells with the scenario's matched inputs → category."""
    import backend.ai_trader as ai_trader
    import data_fetcher
    monkeypatch.setattr(data_fetcher, "get_cached_market_direction", lambda: {
        "success": True, "weighted_signal": 2.0,
        "indexes": {"SPY": {"price": 500.0, "ma_50": 480.0, "ma_200": 450.0, "ema_21": 495.0}},
    })
    monkeypatch.setattr(ai_trader, "HistoricalDataProvider", None)
    # ATR-widened stop both engines see (the live ATR fetch is replaced by a constant).
    monkeypatch.setattr(ai_trader, "calculate_atr_stop", lambda t, p, base: sc["atr_stop"])

    price = COST_BASIS * (1 + sc["gain_pct"] / 100)
    db.add(AIPortfolioConfig(
        user_id=1, starting_cash=25000.0, current_cash=10000.0, max_positions=8,
        max_position_pct=12.0, min_score_to_buy=72, sell_score_threshold=45,
        take_profit_pct=75.0, stop_loss_pct=7.0, is_active=True, strategy=STRATEGY))
    db.add(Stock(ticker=ticker, sector="Technology", current_price=price,
                 canslim_score=sc["score"], is_growth_stock=False, is_breaking_out=False,
                 volume_ratio=1.0, days_to_earnings=None, weeks_in_base=4,
                 earnings_beat_streak=0, projected_growth=15.0))
    db.add(AIPortfolioPosition(
        user_id=1, ticker=ticker, shares=100.0, cost_basis=COST_BASIS,
        purchase_date=datetime.now(timezone.utc) - timedelta(days=sc["days_held"]),
        purchase_score=80.0, current_price=price, current_value=price * 100,
        gain_loss=(price - COST_BASIS) * 100, gain_loss_pct=sc["gain_pct"],
        current_score=sc["score"], is_growth_stock=False, peak_price=sc["peak"],
        peak_date=datetime.now(timezone.utc) - timedelta(days=5),
        pyramid_count=sc.get("pyramid_count", 0), partial_profit_taken=0.0))
    db.commit()

    sells = ai_trader.evaluate_sells(db, user_id=1)
    for s in sells:
        if s["position"].ticker == ticker:
            return _categorize(s["reason"])
    return "HOLD"


def _run_backtest(engine, sc, ticker="PAR"):
    """Drive engine._evaluate_sells with the same scenario → category."""
    from backend.backtester import SimulatedPosition
    price = COST_BASIS * (1 + sc["gain_pct"] / 100)
    engine.data_provider.get_price_on_date.return_value = price
    # Reproduce the ATR-widened stop: get_atr huge → atr_stop hits the max_stop_pct
    # cap; atr_stop==7 base means ATR off. Use a value that lands on the scenario stop.
    if sc["atr_stop"] <= 7.0:
        engine.data_provider.get_atr.return_value = 0.0          # ATR off → base 7%
    else:
        engine.data_provider.get_atr.return_value = 100.0        # → caps at max_stop_pct (20%)
    engine.positions = {ticker: SimulatedPosition(
        ticker=ticker, shares=100.0, cost_basis=COST_BASIS,
        purchase_date=date.today() - timedelta(days=sc["days_held"]),
        purchase_score=80.0, peak_price=sc["peak"],
        peak_date=date.today() - timedelta(days=5), is_growth_stock=False,
        sector="Technology", partial_profit_taken=0.0, pyramid_count=sc.get("pyramid_count", 0))}
    trades = engine._evaluate_sells(date.today(), {ticker: {"total_score": sc["score"]}})
    for t in trades:
        if t.ticker == ticker:
            return _categorize(t.reason)
    return "HOLD"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_engine_decision_parity(name, db_session, bt_engine, monkeypatch):
    """Both engines must reach the same sell decision for matched inputs."""
    sc = SCENARIOS[name]
    live = _run_live(db_session, monkeypatch, sc)
    back = _run_backtest(bt_engine, sc)
    assert live == sc["expected"], f"live {name}: got {live}, expected {sc['expected']}"
    assert back == sc["expected"], f"backtest {name}: got {back}, expected {sc['expected']}"
    assert live == back, f"PARITY BREAK on {name}: live={live} vs backtest={back}"


@pytest.mark.xfail(
    reason="D1: new_position_guard missing from evaluate_sells — live HOLDs a "
           "fast-falling new position the backtester STOP_LOSSes. Engines AGREE "
           "once the June-19 guard port lands; strict flips this to a hard "
           "failure then (remove it + see docs/d1-stoploss-clamp-playbook.md).",
    strict=True,
)
def test_engine_parity_d1_new_position(db_session, bt_engine, monkeypatch):
    """Known divergence: 5-day-old −12% position under a 20% ATR stop. Backtester
    clamps to the 8% guard and sells; live (no guard) holds. This asserts parity,
    so it FAILS today and XPASSes when D1 is fixed."""
    sc = dict(gain_pct=-12.0, score=75, peak=88.0, days_held=5, atr_stop=20.0)
    live = _run_live(db_session, monkeypatch, sc)
    back = _run_backtest(bt_engine, sc)
    assert live == back, f"D1 divergence: live={live} vs backtest={back}"
